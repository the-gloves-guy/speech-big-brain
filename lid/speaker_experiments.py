"""Four paired retraining runs and fixed-cohort external comparisons."""

import gc
import hashlib
import json
import shutil

import numpy as np
import torch
from scipy.special import softmax

from .extension_data import CODES, file_hash, read_rows, write_json
from .extension_data import RUN as ORIGINAL
from .extension_evaluate import evaluate_method
from .extension_model import head_scores, load_initial, state_fingerprint
from .extension_train import train as train_model
from .speaker_study import EXTERNAL, ROOT, SPLITS, status, write_csv
from .youtube_evaluate import embeddings_for
from .youtube_metrics import score_segments


def prepare_run(condition):
    run = ROOT / "runs" / condition
    run.mkdir(parents=True, exist_ok=True)
    full = torch.load(
        ROOT / "frozen_lid_embeddings.pt", map_location="cpu", weights_only=True
    )
    rows = read_rows(ROOT / "inventory.jsonl")
    positions = {row["id"]: index for index, row in enumerate(rows)}
    for split in SPLITS:
        source = ROOT / f"new_{split}.jsonl"
        target = run / source.name
        if target.exists() and file_hash(target) != file_hash(source):
            raise ValueError("Experiment split changed after run preparation")
        shutil.copyfile(source, target)
        cache = run / f"new_{split}.pt"
        if not cache.exists():
            selected = read_rows(target)
            template = torch.load(
                ORIGINAL / f"new_{split}.pt", map_location="cpu", weights_only=True
            )
            identity = dict(template["identity"])
            identity["rows_sha256"] = hashlib.sha256(
                json.dumps(selected, sort_keys=True).encode()
            ).hexdigest()
            torch.save(
                {
                    "identity": identity,
                    "embeddings": full["embeddings"][
                        [positions[r["id"]] for r in selected]
                    ],
                },
                cache,
            )
    ready = ROOT / "data_ready.json"
    if (run / ready.name).exists() and file_hash(run / ready.name) != file_hash(ready):
        raise ValueError("Experiment data identity changed")
    shutil.copyfile(ready, run / ready.name)
    return run


def evaluate_external(run, method):
    output = run / method / "external"
    output.mkdir(exist_ok=True)
    checkpoint_path = run / method / "model.pt"
    identity = {
        "model_sha256": file_hash(checkpoint_path),
        "calibration_sha256": file_hash(run / method / "calibration.json"),
        "segments_sha256": file_hash(EXTERNAL / "segments.jsonl"),
        "encoder_fingerprint": json.loads(
            (run / method / "calibration.json").read_text()
        )["encoder_fingerprint"],
    }
    marker = output / "identity.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("External scoring identity mismatch")
    if marker.exists() and (output / "logits.npz").exists():
        return
    rows = read_rows(EXTERNAL / "segments.jsonl")
    encoder, head, _ = load_initial()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder.mods.embedding_model.load_state_dict(checkpoint["encoder"])
    if (
        state_fingerprint(encoder.mods.embedding_model)
        != identity["encoder_fingerprint"]
    ):
        raise ValueError("Classifier and calibration encoder differ")
    head.load_state_dict(checkpoint["head"])
    encoder.eval()
    if method == "frozen":
        embeddings = torch.load(
            EXTERNAL / "base_embeddings.pt", map_location="cpu", weights_only=True
        )["embeddings"]
    else:
        embeddings, _ = embeddings_for(encoder, rows, output, "partial")
    logits = head_scores(head, embeddings).numpy()
    np.savez_compressed(
        output / "logits.npz", logits=logits, ids=np.array([r["id"] for r in rows])
    )
    write_json(marker, identity)
    del encoder, head, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def run_experiments(train=True):
    for condition, sampling in (
        ("disjoint_uniform", "uniform"),
        ("disjoint_balanced", "speaker_balanced"),
    ):
        run = prepare_run(condition)
        for method in ("frozen", "partial"):

            def emit(phase, condition=condition, method=method, **details):
                status(phase, condition=condition, method=method, **details)

            if train:
                train_model(method, run=run, sampling=sampling, progress=emit)
            if not (run / method / "complete.json").exists():
                raise ValueError("Training is incomplete")
            evaluate_method(method, run=run, progress=emit)
            evaluate_external(run, method)
            gc.collect()
            torch.cuda.empty_cache()
    compare()


def compare():
    rows = read_rows(EXTERNAL / "segments.jsonl")
    scores = {}
    baseline = np.load(EXTERNAL / "logits.npz")
    if baseline["ids"].tolist() != [r["id"] for r in rows]:
        raise ValueError("Baseline external IDs do not align")
    for method in ("frozen", "partial"):
        calibration = json.loads((ORIGINAL / method / "calibration.json").read_text())
        scores[f"recording_uniform/{method}"] = (
            baseline[f"{method}15"],
            calibration["temperature"],
        )
    for condition in ("disjoint_uniform", "disjoint_balanced"):
        for method in ("frozen", "partial"):
            path = ROOT / "runs" / condition / method
            saved = np.load(path / "external/logits.npz")
            if saved["ids"].tolist() != [r["id"] for r in rows]:
                raise ValueError("External IDs do not align")
            calibration = json.loads((path / "calibration.json").read_text())
            scores[f"{condition}/{method}"] = (
                saved["logits"],
                calibration["temperature"],
            )
    predictions = {
        key: [CODES[i] for i in value[0].argmax(-1)] for key, value in scores.items()
    }
    common = np.array(
        [
            all(values[i] != "en" for values in predictions.values())
            for i in range(len(rows))
        ]
    )
    summaries, all_predictions = [], []
    reports = {}
    for name, (logits, temperature) in scores.items():
        confidence = softmax(logits.astype(np.float64) / temperature, axis=-1).max(-1)
        raw_confidence = softmax(logits.astype(np.float64), axis=-1).max(-1)
        conditional = score_segments(rows, predictions[name], confidence)
        shared = score_segments(rows, predictions[name], confidence, common)
        raw = score_segments(rows, predictions[name], raw_confidence)
        reports[name] = {
            "conditional": conditional,
            "common": shared,
            "ece_raw": raw["ece"],
            "temperature": temperature,
        }
        summaries.append(
            {
                "model": name,
                "accuracy": conditional["accuracy"],
                "macro_f1": conditional["macro_f1"],
                "macro_precision": conditional["macro_precision"],
                "macro_recall": conditional["macro_recall"],
                "ece_raw": raw["ece"],
                "ece_calibrated": conditional["ece"],
                "english": conditional["english_predictions"],
                "coverage": conditional["coverage"],
                "common_n": int(common.sum()),
                "common_accuracy": shared["accuracy"],
            }
        )
        for i, row in enumerate(rows):
            all_predictions.append(
                {
                    "model": name,
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "language": row["language"],
                    "prediction": predictions[name][i],
                    "confidence": float(confidence[i]),
                    "common": bool(common[i]),
                }
            )
    write_csv(ROOT / "external_summary.csv", summaries)
    write_csv(ROOT / "external_predictions.csv", all_predictions)
    write_json(ROOT / "external_comparison.json", reports)

    def percent(value):
        return "undefined" if value is None else f"{100 * value:.2f}"

    table = "\n".join(
        f"| {r['model']} | {percent(r['accuracy'])} | {percent(r['macro_f1'])} | {r['english']} | {percent(r['common_accuracy'])} |"
        for r in summaries
    )
    regression = []
    for condition in ("recording_uniform", "disjoint_uniform", "disjoint_balanced"):
        for method in ("frozen", "partial"):
            path = (
                ORIGINAL / method
                if condition == "recording_uniform"
                else ROOT / "runs" / condition / method
            )
            groups = json.loads((path / "report.json").read_text())["groups"]
            for group in ("old12", "new3"):
                value = groups[group]["raw"]
                regression.append(
                    {
                        "condition": condition,
                        "method": method,
                        "group": group,
                        **{
                            key: value[key]
                            for key in (
                                "accuracy",
                                "macro_f1",
                                "macro_precision",
                                "macro_recall",
                                "ece",
                            )
                        },
                    }
                )
    write_csv(ROOT / "radio_and_fleurs_summary.csv", regression)
    text = "## Classifier comparisons\n\nOne seed, identical initialization and update budgets. Speaker-balanced sampling changes Pacific examples only.\n\n"
    text += f"External results use provisional folder labels. English predictions are counted and excluded; the common cohort contains {int(common.sum())} segments excluded by none of the six models.\n\n"
    text += (
        "| Condition / model | Accuracy (%) | Macro F1 (%) | English exclusions | Common accuracy (%) |\n|---|---:|---:|---:|---:|\n"
        + table
        + "\n\n"
    )
    text += "Radio test populations differ between recording and speaker splits; their scores do not isolate a training improvement. The two speaker-disjoint sampling conditions share exactly the same radio test set.\n\n"
    text += "[Full external metrics](../../artifacts/speaker_study/external_comparison.json) · [Radio and FLEURS metrics](../../artifacts/speaker_study/radio_and_fleurs_summary.csv).\n"
    (ROOT / "experiment_comparison.md").write_text(text)


def duration_diagnostic():
    """Matched 3/5/10s center crops of the same external 10s speech windows."""
    import soundfile as sf

    marker = ROOT / "duration_complete.json"
    if marker.exists():
        return
    destination = ROOT / "duration"
    destination.mkdir(exist_ok=True)
    groups = {seconds: [] for seconds in (3, 5, 10)}
    prepared = json.loads((EXTERNAL / "segments_ready.json").read_text())
    for source in prepared["source_summaries"]:
        saved = json.loads((EXTERNAL / f"{source['source_id']}.json").read_text())
        path = EXTERNAL / "decoded" / f"{source['source_id']}.wav"
        with sf.SoundFile(path) as audio:
            if audio.samplerate != 16000 or audio.channels != 1:
                raise ValueError("Duration diagnostic audio protocol changed")
            for interval in saved["vad_intervals_samples"]:
                size = interval["end"] - interval["start"]
                count = size // 160000
                offset = (size - count * 160000) // 2
                for index in range(count):
                    start = interval["start"] + offset + index * 160000
                    center = start + 80000
                    audio.seek(start)
                    signal = audio.read(160000, dtype="float32")
                    for seconds in (3, 5, 10):
                        length = seconds * 16000
                        left = (160000 - length) // 2
                        identifier = f"{source['source_id']}-{center}-{seconds}s"
                        cut = destination / f"{identifier}.wav"
                        if not cut.exists():
                            sf.write(
                                cut,
                                signal[left : left + length],
                                16000,
                                subtype="FLOAT",
                            )
                        groups[seconds].append(
                            {
                                "id": identifier,
                                "parent_id": f"{source['source_id']}-{center}",
                                "source_id": source["source_id"],
                                "source": source["source"],
                                "group": source["source_id"],
                                "path": str(cut.resolve()),
                                "language": source["language"],
                                "duration": seconds,
                                "start_seconds": (start + left) / 16000,
                                "end_seconds": (start + left + length) / 16000,
                                "sha256": file_hash(cut),
                            }
                        )
    if not groups[10]:
        write_json(
            marker,
            {
                "eligible_windows": 0,
                "reason": "no contiguous speech region at least 10 seconds",
            },
        )
        return
    predicted, confidences, provenance = {}, {}, {}
    for method in ("frozen", "partial"):
        encoder, head, _ = load_initial()
        checkpoint_path = ORIGINAL / method / "model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        encoder.mods.embedding_model.load_state_dict(checkpoint["encoder"])
        head.load_state_dict(checkpoint["head"])
        encoder.eval()
        calibration = json.loads((ORIGINAL / method / "calibration.json").read_text())
        provenance[method] = {
            "model_sha256": file_hash(checkpoint_path),
            "calibration_sha256": file_hash(ORIGINAL / method / "calibration.json"),
        }
        for seconds, rows in groups.items():
            status(
                "duration_diagnostic", method=method, seconds=seconds, windows=len(rows)
            )
            embeddings, _ = embeddings_for(
                encoder, rows, destination, f"{method}_{seconds}s"
            )
            logits = head_scores(head, embeddings).numpy()
            key = f"{method}_{seconds}s"
            predicted[key] = [CODES[i] for i in logits.argmax(-1)]
            confidences[key] = softmax(
                logits.astype(np.float64) / calibration["temperature"], axis=-1
            ).max(-1)
        del encoder, head, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    common = np.array(
        [
            all(predictions[i] != "en" for predictions in predicted.values())
            for i in range(len(groups[10]))
        ]
    )
    metrics, records = {}, []
    for method in ("frozen", "partial"):
        for seconds, rows in groups.items():
            key = f"{method}_{seconds}s"
            conditional = score_segments(rows, predicted[key], confidences[key])
            shared = score_segments(rows, predicted[key], confidences[key], common)
            metrics[key] = {"conditional": conditional, "common": shared}
            for i, row in enumerate(rows):
                records.append(
                    {
                        "model": method,
                        **row,
                        "predicted": predicted[key][i],
                        "confidence": float(confidences[key][i]),
                        "in_common_cohort": bool(common[i]),
                    }
                )
    write_csv(ROOT / "duration_predictions.csv", records)
    write_json(ROOT / "duration_metrics.json", metrics)
    write_json(
        marker,
        {
            "eligible_windows": len(groups[10]),
            "common_windows": int(common.sum()),
            "models": provenance,
            "external_vad_sha256": file_hash(EXTERNAL / "segments_ready.json"),
            "note": "paired center crops; folder labels remain provisional; crops may contain different proportions of code switching",
        },
    )
