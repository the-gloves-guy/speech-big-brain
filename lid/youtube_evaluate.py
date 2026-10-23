"""Held-out YouTube inference and auditable model-specific English exclusions."""

import csv
import gc
import hashlib
import json
from collections import Counter
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


def evaluate(root, destination, min_seconds=3):
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, prepared = prepare(root, output, min_seconds=min_seconds)
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
    write_report(output, rows, prepared, reports)
    write_json(
        output / "status.json",
        {
            "phase": "complete",
            "segments": len(rows),
            "sources": len(prepared["source_summaries"]),
            "report": "reports/youtube_validation/held_out_evaluation.md",
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path("reports/youtube_validation")
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = ["#0072B2", "#D55E00"]
    primary = ["frozen15", "partial15"]

    def save(fig, name):
        for extension in ("png", "pdf", "svg"):
            fig.savefig(figures / f"{name}.{extension}", dpi=350, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for offset, model, color in zip((-0.17, 0.17), primary, colors):
        values = reports[model]["conditional"]["by_language"]
        for ax, metric in zip(axes, ("accuracy", "coverage")):
            bars = [
                values[code][metric] * 100
                if values[code][metric] is not None
                else np.nan
                for code in TARGETS
            ]
            ax.bar(
                np.arange(3) + offset,
                bars,
                width=0.32,
                color=color,
                label=MODELS[model],
            )
            for index, value in enumerate(bars):
                if np.isfinite(value):
                    ax.text(
                        index + offset,
                        value + 1.2,
                        f"{value:.1f}",
                        ha="center",
                        fontsize=8,
                    )
        axes[0].set(
            title="Folder-label accuracy, English excluded",
            ylabel="Conditional accuracy (%)",
        )
        axes[1].set(title="Segments retained for scoring", ylabel="Coverage (%)")
    for ax in axes:
        ax.set_xticks(range(3), ["Samoan", "Niuean", "Tongan"])
        ax.set_ylim(0, 112)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=2,
        fontsize=9,
    )
    save(fig, "accuracy_and_coverage")

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    sources = prepared["source_summaries"]
    labels = []
    counts = Counter()
    for source in sources:
        counts[source["language"]] += 1
        labels.append(f"{source['language']} clip {counts[source['language']]}")
    for offset, model, color in zip((-0.1, 0.1), primary, colors):
        values = reports[model]["conditional"]["by_source"]
        ax.scatter(
            [
                100 * values[source["source_id"]]["accuracy"]
                if values[source["source_id"]]["accuracy"] is not None
                else np.nan
                for source in sources
            ],
            np.arange(len(sources)) + offset,
            color=color,
            label=MODELS[model],
            s=45,
        )
    ax.set_yticks(range(len(sources)), labels)
    ax.invert_yaxis()
    ax.set(
        xlim=(-2, 105),
        xlabel="Conditional folder-label accuracy (%)",
        title="Results by original source clip",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    save(fig, "per_source")

    def pct(value):
        return "—" if value is None else f"{100 * value:.2f}"

    main_rows, common_rows, language_rows, control_rows = [], [], [], []
    for name in primary:
        value = reports[name]["conditional"]
        main_rows.append(
            f"| {MODELS[name]} | {value['segments_total']} | {value['english_predictions']} | {value['segments_scored']} | {pct(value['coverage'])} | {pct(value['accuracy'])} | {pct(value['macro_precision'])} | {pct(value['macro_recall'])} | {pct(value['macro_f1'])} |"
        )
        shared = reports[name]["common_cohort"]
        common_rows.append(
            f"| {MODELS[name]} | {shared['segments_scored']} | {pct(shared['accuracy'])} | {pct(shared['macro_f1'])} | {pct(shared['mean_source_accuracy'])} |"
        )
        for code, counts in value["by_language"].items():
            language_rows.append(
                f"| {MODELS[name]} | {code} | {counts['total']} | {counts['english']} | {counts['scored']} | {pct(counts['accuracy'])} |"
            )
    for name in ("original107", "filtered12", "previous12"):
        value = reports[name]["conditional"]
        top = ", ".join(
            f"{code}: {n}"
            for code, n in sorted(
                value["prediction_counts_all"].items(), key=lambda item: -item[1]
            )[:4]
        )
        control_rows.append(
            f"| {MODELS[name]} | {value['english_predictions']} | {value['segments_scored']} | {top} |"
        )
    source_rows = [
        f"| {Path(value['source']).name} | {value['language']} | {value['decoded_seconds']:.1f} | {value['retained_segments']} | {value['retained_seconds']:.1f} | {value['dropped_short_regions']} |"
        for value in sources
    ]
    confusions = []
    for name in primary:
        counts = reports[name]["conditional"]["confusion_scored"]
        pairs = sorted(
            [
                (count, truth, prediction)
                for truth, values in counts.items()
                for prediction, count in values.items()
                if prediction != truth
            ],
            reverse=True,
        )
        confusions.extend(
            f"| {MODELS[name]} | {truth} | {prediction} | {count} |"
            for count, truth, prediction in pairs[:5]
        )
    calibration = [
        f"| {MODELS[name]} | {pct(reports[name]['conditional']['ece_raw'])} | {pct(reports[name]['conditional']['ece'])} | {reports[name]['temperature']:.3f} |"
        for name in primary
    ]
    minimum = prepared["configuration"]["minimum_seconds"]
    total_seconds = sum(source["decoded_seconds"] for source in sources)
    kept_seconds = sum(row["duration"] for row in rows)
    dropped = sum(source["dropped_short_regions"] for source in sources)
    frozen = reports["frozen15"]["conditional"]
    partial = reports["partial15"]["conditional"]
    common_frozen = reports["frozen15"]["common_cohort"]
    common_partial = reports["partial15"]["common_cohort"]
    language_names = {"sam": "Samoan", "niu": "Niuean", "ton": "Tongan"}
    weakest = sorted(
        TARGETS,
        key=lambda code: np.mean(
            [
                report["conditional"]["by_language"][code]["accuracy"]
                for report in (reports["frozen15"], reports["partial15"])
                if report["conditional"]["by_language"][code]["accuracy"] is not None
            ]
        ),
    )
    priority = " and ".join(language_names[code] for code in weakest[:2])
    findings = (
        f"- **Conditional accuracy is {pct(frozen['accuracy'])}% (frozen) and {pct(partial['accuracy'])}% (partial)**, compared with about 99.7–99.8% on the radio-news new-language holdout. These datasets/protocols differ, so this does not isolate the cause of the difference.\n"
        f"- **English exclusions:** {frozen['english_predictions']} / {partial['english_predictions']} segments. On the identical {common_frozen['segments_scored']}-segment cohort, accuracy is {pct(common_frozen['accuracy'])}% / {pct(common_partial['accuracy'])}%.\n"
        f"- **Per-language accuracy:** Tongan {pct(frozen['by_language']['ton']['accuracy'])}% / {pct(partial['by_language']['ton']['accuracy'])}%; Niuean {pct(frozen['by_language']['niu']['accuracy'])}% / {pct(partial['by_language']['niu']['accuracy'])}%; Samoan {pct(frozen['by_language']['sam']['accuracy'])}% / {pct(partial['by_language']['sam']['accuracy'])}% (frozen / partial).\n"
        f"- **Equal weighting of source clips gives lower accuracy:** {pct(frozen['mean_source_accuracy'])}% / {pct(partial['mean_source_accuracy'])}%. Long, easier clips can raise pooled accuracy while performance on individual sources remains uneven.\n"
        f"- **Prioritize review and independent evaluation sources for {priority}**, the weaker languages in this collection. Test noise/channel/reverberation augmentation as an ablation only if annotated errors support an acoustic mismatch. These results do not establish that augmentation alone will fix the difference.\n"
    )
    update_section = ""
    previous_path = Path("artifacts/youtube_validation/report.json").resolve()
    if previous_path.exists() and previous_path.parent != output.resolve():
        previous = json.loads(previous_path.read_text())
        update_rows = []
        for name in primary:
            before = previous["models"][name]["conditional"]
            after = reports[name]["conditional"]
            update_rows.append(
                f"| {MODELS[name]} | {before['segments_total']} | {after['segments_total']} | {pct(before['accuracy'])} | {pct(after['accuracy'])} | {before['english_predictions']} | {after['english_predictions']} |"
            )
        update_section = (
            "## Change from the original seven-clip evaluation\n\nSame model checkpoints and VAD settings; the evaluation corpus changed. Differences below reflect audio composition, not model improvement.\n\n| Model | Previous cuts | Updated cuts | Previous accuracy (%) | Updated accuracy (%) | Previous English | Updated English |\n|---|---:|---:|---:|---:|---:|---:|\n"
            + "\n".join(update_rows)
            + "\n\n[Previous report](../youtube_validation_previous_7clips/held_out_evaluation.md).\n"
        )
    length_rows = []
    for name in primary:
        for low, high in ((minimum, 5), (5, 8), (8, 10.001)):
            if low >= high:
                continue
            selected = [
                row
                for row in read_prediction_rows(output / "predictions.csv")
                if row["model"] == name
                and low <= float(row["duration"]) < high
                and row["excluded_english"] == "False"
            ]
            value = (
                sum(row["folder_label_match"] == "True" for row in selected)
                / len(selected)
                if selected
                else None
            )
            length_rows.append(
                f"| {MODELS[name]} | {low:g}–{min(high, 10):g} | {len(selected)} | {pct(value)} |"
            )
    text = f"""# Held-out YouTube speech evaluation

**Purpose:** screen for generalization beyond the radio-news training domain. These are **provisional directory-label metrics**, not a manually annotated conversational benchmark.

## Key findings

{findings}

{update_section}

## Data and segmentation

- **{len(sources)} source clips**, {total_seconds / 60:.1f} minutes; folder labels `sam`, `niu`, `ton`. The directory spelling `Nuiean` maps to Niuean (`niu`). This is a mixed convenience sample; recording styles are not manually annotated.
- Silero VAD {prepared["configuration"]["vad_version"]} on CPU; threshold 0.5, minimum silence 300 ms, padding 150 ms. [Model implementation](https://github.com/snakers4/silero-vad).
- Decode one file at a time to mono 16 kHz float32. Keep contiguous **{minimum:g}–10 second** speech segments; split long regions evenly without overlap or concatenating distant speech. Discard regions below {minimum:g} seconds.
- **{len(rows)} retained segments**, {kept_seconds / 60:.1f} minutes; **{dropped} short regions discarded**. Detailed VAD boundaries and discarded durations are saved per source.
- No extra gain normalization, model fitting, augmentation, checkpoint selection, or temperature fitting used these clips. Apply previously fitted radio/FLEURS calibration once; raw/calibrated top-1 labels agree.

| Source audio | Label | Duration (s) | Retained cuts | Retained (s) | Short regions dropped |
|---|---|---:|---:|---:|---:|
{chr(10).join(source_rows)}

## English-excluded results

- Count every top-1 **English prediction** separately and exclude it from that model's accuracy, precision, recall, F1 and ECE. Other predicted languages remain errors against the folder label.
- **English predictions are not verified English speech.** A missed Pacific-language segment predicted English is excluded too; actual English predicted as a Pacific language can remain in the metrics. Mixed-language segments also have no verified language label.
- Denominators therefore differ by model. Coverage and the same-cohort comparison below are essential for interpreting scores. Precision/recall/F1 are macro-averaged across target languages with retained ground-label examples; empty cohorts have undefined scores.

| Model | Cuts | English excluded | Scored | Coverage (%) | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(main_rows)}

![Conditional accuracy and scoring coverage](figures/accuracy_and_coverage.png)

### Same-cohort comparison

Exclude a segment if **either** 15-language model predicts English, then compare both on the identical remaining segments. This controls differences in model-specific exclusion sets; it does not establish their true language. Source accuracy is the unweighted mean of clip-level conditional accuracies for clips with retained segments.

| Model | Common cuts | Accuracy (%) | Macro F1 (%) | Mean source accuracy (%) |
|---|---:|---:|---:|---:|
{chr(10).join(common_rows)}

### By language and source

| Model | Folder label | Cuts | English excluded | Scored | Accuracy (%) |
|---|---|---:|---:|---:|---:|
{chr(10).join(language_rows)}

![Conditional performance for each original clip](figures/per_source.png)

### Segment-duration diagnostic

Descriptive scores after English exclusions; duration is another possible contributor to the gap and was not tuned using these results. Bins start at the configured minimum, with boundaries at 5, 8, and 10 seconds; upper boundaries are excluded except the final 10-second boundary.

| Model | Duration (s) | Scored cuts | Accuracy (%) |
|---|---:|---:|---:|
{chr(10).join(length_rows)}

### Largest non-English mismatches

| Model | Folder label | Predicted | Segments |
|---|---|---|---:|
{chr(10).join(confusions)}

### Calibration diagnostics

ECE uses 15 equal-width bins on each model's non-English subset. These values are conditional and use provisional labels; they are not an unbiased calibration estimate for verified code-switched speech.

| Model | Raw ECE (%) | Previously calibrated ECE (%) | Temperature |
|---|---:|---:|---:|
{chr(10).join(calibration)}

### Older model controls

The original 107-language, filtered 12-language, and previous fine-tuned 12-language classifiers **lack all three target outputs**. Their target-language accuracy/F1 is structurally zero on retained folder labels, so they are reference controls rather than competing recognition models. Their English counts and error-label distributions are still useful.

| Control | English predictions | Non-English cuts | Most frequent predictions (all cuts) |
|---|---:|---:|---|
{chr(10).join(control_rows)}

## Interpretation and next experiment

- Compare conditional accuracy **with coverage and per-source results**, rather than treating all retained cuts as independent evidence. Source-file counts are {dict(Counter(source["language"] for source in sources))}; files do not establish independent speakers or videos, and short/long extracts may overlap. No narrow segment-level confidence intervals are reported.
- The radio-held-out test previously gave approximately 99.7–99.8% accuracy on the three new languages. These YouTube scores use different audio, segment durations and English exclusions; the difference is a domain-shift diagnostic, not a controlled estimate of augmentation benefit.
- Manually review English predictions, disagreements and non-English mismatches using the timestamped review queue before deciding which errors are acoustic domain shift versus label uncertainty or code switching.
- If verified errors cluster around noise, room acoustics or recording channels, test corresponding augmentation as a controlled training ablation with old-language replay. If they cluster around conversational pronunciation or speaking style, prioritize additional independent conversational speakers. Keep these evaluation sources held out from that training.
- Gain-only augmentation is unlikely to resolve a speaking-style mismatch by itself; the prior Samoan validation gain probe showed nearly invariant embeddings under uniform gain changes. This is a working hypothesis to test, not a measured augmentation result.

## Artifacts

- [Summary CSV](../../{output.relative_to(Path.cwd())}/summary.csv) · [Per-language CSV](../../{output.relative_to(Path.cwd())}/per_language.csv) · [Per-source CSV](../../{output.relative_to(Path.cwd())}/per_source.csv).
- [Every prediction and probability](../../{output.relative_to(Path.cwd())}/predictions.csv) · [Full report JSON](../../{output.relative_to(Path.cwd())}/report.json).
- [Review queue](../../{output.relative_to(Path.cwd())}/review_queue.csv) · [Annotation template](../../{output.relative_to(Path.cwd())}/annotations_template.csv) · [Speech WAVs](../../{output.relative_to(Path.cwd())}/segments/).
- Source/model/VAD hashes, exact boundaries and settings are recorded. Plots are 350 dpi PNG plus vector PDF and SVG.
"""
    (destination / "held_out_evaluation.md").write_text(text)


def read_prediction_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))
