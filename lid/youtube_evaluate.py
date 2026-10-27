"""Held-out YouTube inference and auditable model-specific English exclusions."""

import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from .data import waveform
from .extension_data import BASE, CODES, REVISION, RUN, file_hash, write_json
from .extension_model import head_scores, state_fingerprint
from .languages import CODES as OLD_CODES
from .languages import selected_indices
from .model import load_encoder, new_head
from .youtube_data import TARGETS, prepare
from .youtube_metrics import common_cohort, score_segments

MODELS = {
    "frozen15": "Frozen encoder (15)",
    "partial15": "Partial encoder (15)",
    "original107": "Original (107)",
    "filtered12": "Filtered (12)",
    "previous12": "Previous fine-tuned (12)",
}


def embeddings_for(encoder, rows, output, name):
    path = output / f"{name}_embeddings.pt"
    identity = {
        "encoder_sha256": state_fingerprint(encoder.mods.embedding_model),
        "revision": REVISION,
        "rows_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        "preprocessing": "mono 16kHz float32; full 3-10s segment; no added gain normalization",
    }
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Embedding cache does not match this model or data")
        return saved["embeddings"], identity
    parts = output / f"{name}_embeddings.parts"
    parts.mkdir(exist_ok=True)
    context = parts / "context.json"
    if context.exists() and json.loads(context.read_text()) != identity:
        raise ValueError("Interrupted cache belongs to different data/model")
    write_json(context, identity)
    shards = sorted(parts.glob("part-*.pt"))
    completed = sum(
        torch.load(p, map_location="cpu", weights_only=True).shape[0] for p in shards
    )
    encoder.eval()
    with torch.inference_mode():
        for start in range(completed, len(rows), 64):
            values = []
            for row in rows[start : start + 64]:
                signal = torch.from_numpy(waveform(row, 10.0)).unsqueeze(0).to("cuda")
                values.append(
                    encoder.encode_batch(signal, normalize=False).reshape(1, 256).cpu()
                )
            shard = parts / f"part-{len(shards):05d}.pt"
            temporary = shard.with_suffix(".tmp")
            torch.save(torch.cat(values), temporary)
            temporary.replace(shard)
            shards.append(shard)
            write_json(
                output / "status.json",
                {
                    "phase": f"embed_{name}",
                    "completed": min(start + 64, len(rows)),
                    "total": len(rows),
                },
            )
    embeddings = torch.cat(
        [torch.load(path, map_location="cpu", weights_only=True) for path in shards]
    )
    torch.save({"embeddings": embeddings, "identity": identity}, path)
    return embeddings, identity


def infer(rows, output):
    encoder, original_labels, _ = load_encoder("cuda", REVISION)
    original_codes = [label.split(":", 1)[0].strip() for label in original_labels]
    indices = selected_indices(original_labels)
    base_embeddings, base_identity = embeddings_for(encoder, rows, output, "base")
    old_checkpoint = torch.load(
        BASE / "head/head.pt", map_location="cpu", weights_only=True
    )
    old_calibration = json.loads((BASE / "comparison/calibration.json").read_text())
    if old_calibration["head_sha256"] != file_hash(BASE / "head/head.pt"):
        raise ValueError("Old calibration/model mismatch")
    raw_original = head_scores(encoder.mods.classifier, base_embeddings).numpy()
    old_head = new_head(old_checkpoint["initial_state"], old_checkpoint["indices"])
    old_head.load_state_dict(old_checkpoint["state"])
    old_head.to("cuda").eval()
    scores = {
        "original107": (
            raw_original,
            original_codes,
            old_calibration["original_temperature"],
        ),
        "filtered12": (
            raw_original[:, indices],
            OLD_CODES,
            old_calibration["filtered_temperature"],
        ),
        "previous12": (
            head_scores(old_head, base_embeddings).numpy(),
            OLD_CODES,
            old_calibration["head_temperature"],
        ),
    }
    model_fingerprints = {
        model: {
            "revision": REVISION,
            "base_encoder": base_identity["encoder_sha256"],
            "head_sha256": file_hash(BASE / "head/head.pt")
            if model == "previous12"
            else "original pretrained classifier",
            "calibration_sha256": file_hash(BASE / "comparison/calibration.json"),
        }
        for model in scores
    }
    del old_head
    for method, name in (("frozen", "frozen15"), ("partial", "partial15")):
        checkpoint_path = RUN / method / "model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        calibration = json.loads((RUN / method / "calibration.json").read_text())
        if checkpoint["codes"] != CODES or calibration["model_sha256"] != file_hash(
            checkpoint_path
        ):
            raise ValueError(f"Model/calibration mismatch: {method}")
        encoder.mods.embedding_model.load_state_dict(checkpoint["encoder"])
        encoder.eval()
        actual = state_fingerprint(encoder.mods.embedding_model)
        if actual != calibration["encoder_fingerprint"]:
            raise ValueError("Wrong encoder loaded")
        if method == "frozen":
            if actual != base_identity["encoder_sha256"]:
                raise ValueError("Frozen encoder differs from baseline cache")
            embeddings = base_embeddings
        else:
            embeddings, _ = embeddings_for(encoder, rows, output, "partial")
        head = new_head(old_checkpoint["initial_state"], old_checkpoint["indices"])
        head.out = torch.nn.Linear(512, len(CODES))
        head.load_state_dict(checkpoint["head"])
        head.to("cuda").eval()
        scores[name] = (
            head_scores(head, embeddings).numpy(),
            CODES,
            calibration["temperature"],
        )
        model_fingerprints[name] = {
            "checkpoint_sha256": file_hash(checkpoint_path),
            "encoder_sha256": actual,
            "calibration_sha256": file_hash(RUN / method / "calibration.json"),
        }
        del head, checkpoint
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    np.savez_compressed(
        output / "logits.npz",
        ids=np.array([row["id"] for row in rows]),
        **{name: value[0] for name, value in scores.items()},
    )
    for name, (_, codes, temperature) in scores.items():
        model_fingerprints[name].update(
            codes=list(codes), temperature=float(temperature)
        )
    write_json(output / "model_provenance.json", model_fingerprints)
    return scores


def evaluate(root, destination, min_seconds=3, languages=None):
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, prepared = prepare(root, output, min_seconds=min_seconds, languages=languages)
    write_json(output / "status.json", {"phase": "inference", "segments": len(rows)})
    scores = infer(rows, output)
    predicted, probabilities = {}, {}
    for name, (logits, codes, temperature) in scores.items():
        probabilities[name] = softmax(logits.astype(np.float64) / temperature, axis=-1)
        predicted[name] = [codes[index] for index in logits.argmax(-1)]
    common = common_cohort(predicted)
    reports, prediction_rows = {}, []
    for name, display_name in MODELS.items():
        logits, codes, temperature = scores[name]
        confidence = probabilities[name].max(-1)
        raw_confidence = softmax(logits.astype(np.float64), axis=-1).max(-1)
        own = score_segments(rows, predicted[name], confidence)
        own["ece_raw"] = score_segments(rows, predicted[name], raw_confidence)["ece"]
        reports[name] = {
            "display_name": display_name,
            "codes": list(codes),
            "supports_new_languages": set(TARGETS).issubset(codes),
            "temperature": temperature,
            "conditional": own,
        }
        if name in ("frozen15", "partial15"):
            reports[name]["common_cohort"] = score_segments(
                rows, predicted[name], confidence, common
            )
        for index, row in enumerate(rows):
            prediction_rows.append(
                {
                    "model": name,
                    "segment_id": row["id"],
                    "source_id": row["source_id"],
                    "source": row["source"],
                    "folder_language": row["language"],
                    "start_seconds": row["start_seconds"],
                    "end_seconds": row["end_seconds"],
                    "duration": row["duration"],
                    "predicted": predicted[name][index],
                    "confidence": float(confidence[index]),
                    "excluded_english": predicted[name][index] == "en",
                    "in_common_cohort": bool(common[index]),
                    "folder_label_match": predicted[name][index] == row["language"],
                    "segment_path": row["path"],
                    "probabilities": json.dumps(
                        dict(zip(codes, probabilities[name][index].tolist()))
                    ),
                }
            )
    write_json(
        output / "report.json",
        {
            "models": reports,
            "shared_cohort_segments": int(common.sum()),
            "source_summaries": prepared["source_summaries"],
            "labels": "provisional directory language; code-switches unannotated",
            "exclusion": "top-1 English prediction excluded separately for each model; raw and calibrated top-1 agree",
            "shared_cohort": "exclude a segment if either 15-language model predicts English",
            "no_fitting_on_this_dataset": True,
        },
    )
    write_csv(output / "predictions.csv", prediction_rows)
    summary_rows, language_rows, source_rows = [], [], []
    for name, report in reports.items():
        for cohort in ("conditional", "common_cohort"):
            if cohort not in report:
                continue
            result = report[cohort]
            columns = (
                "segments_total",
                "english_predictions",
                "segments_scored",
                "coverage",
                "english_seconds",
                "seconds_scored",
                "duration_coverage",
                "correct_scored",
                "other_errors",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "ece",
                "duration_weighted_accuracy",
                "mean_source_accuracy",
                "sources_scored",
            )
            summary_rows.append(
                {
                    "model": name,
                    "cohort": cohort,
                    "supports_new_languages": report["supports_new_languages"],
                    **{key: result[key] for key in columns},
                }
            )
            for code, counts in result["by_language"].items():
                language_rows.append(
                    {
                        "model": name,
                        "cohort": cohort,
                        "language": code,
                        **counts,
                        **result["per_language"].get(
                            code,
                            {
                                "precision": None,
                                "recall": None,
                                "f1": None,
                                "support": 0,
                            },
                        ),
                    }
                )
            for source_id, counts in result["by_source"].items():
                source_rows.append(
                    {"model": name, "cohort": cohort, "source_id": source_id, **counts}
                )
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "per_language.csv", language_rows)
    write_csv(output / "per_source.csv", source_rows)
    annotation_rows, review_rows = [], []
    for index, row in enumerate(rows):
        predictions = {
            name: predicted[name][index] for name in ("frozen15", "partial15")
        }
        reasons = []
        if "en" in predictions.values():
            reasons.append("English predicted; verify code switching")
        if len(set(predictions.values())) > 1:
            reasons.append("models disagree")
        if any(code not in (row["language"], "en") for code in predictions.values()):
            reasons.append("non-English mismatch with folder label")
        annotation = {
            "segment_id": row["id"],
            "source": row["source"],
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "folder_language": row["language"],
            "human_label": "",
            "notes": "",
            "segment_path": row["path"],
        }
        annotation_rows.append(annotation)
        if reasons:
            review_rows.append(
                {**annotation, **predictions, "review_reasons": "; ".join(reasons)}
            )
    write_csv(output / "annotations_template.csv", annotation_rows)
    if review_rows:
        write_csv(output / "review_queue.csv", review_rows)
    else:
        with (output / "review_queue.csv").open("w", newline="") as stream:
            csv.writer(stream).writerow(
                [*annotation_rows[0], "frozen15", "partial15", "review_reasons"]
            )
    report_path = write_report(output, rows, prepared, reports)
    from .speaker_study import resources

    write_json(output / "resources.json", resources())
    write_json(
        output / "status.json",
        {
            "phase": "complete",
            "segments": len(rows),
            "sources": len(prepared["source_summaries"]),
            "report": str(report_path),
        },
    )
    print(
        json.dumps(
            {
                name: {
                    key: report["conditional"][key]
                    for key in (
                        "segments_total",
                        "english_predictions",
                        "segments_scored",
                        "accuracy",
                        "macro_f1",
                    )
                }
                for name, report in reports.items()
            },
            indent=2,
        ),
        flush=True,
    )


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, rows, prepared, reports):
    from .youtube_report import write_report as render

    return render(output, rows, prepared, reports)


def read_prediction_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))
