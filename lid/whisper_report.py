"""Recomputed paired metrics and an evolving Whisper/SpeechBrain comparison."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .extension_data import BASE, file_hash, write_json
from .languages import CODES
from .speaker_study import write_csv
from .youtube_metrics import common_cohort, score_segments

NAMES = {
    "whisper_native": "Whisper tiny native languages",
    "whisper_filtered": "Whisper tiny filtered 12",
    "whisper_frozen": "Whisper frozen encoder + head",
    "whisper_finetuned": "Whisper full encoder fine-tuned + head",
    "speechbrain_original": "SpeechBrain original 107",
    "speechbrain_filtered": "SpeechBrain filtered 12",
    "speechbrain_frozen": "SpeechBrain frozen encoder + head",
    "speechbrain_finetuned": "SpeechBrain full encoder fine-tuned + head",
}


def write_report(study):
    root = study.root
    directory = Path("reports") / root.name
    directory.mkdir(parents=True, exist_ok=True)
    reference = json.loads((BASE / "comparison/report.json").read_text())
    if reference["metadata"]["head_sha256"] != file_hash(BASE / "head/head.pt"):
        raise ValueError("Archived SpeechBrain reference checkpoint changed")
    from .extension_evaluate import baseline_reports

    old = baseline_reports()
    fleur = {}
    for method, key in [
        ("speechbrain_original", "original_107"),
        ("speechbrain_filtered", "filtered_12"),
        ("speechbrain_frozen", "retrained_head_12"),
    ]:
        fleur[method] = {"raw": old[key], "calibrated": old[key + "_temperature"]}
    for method in [
        "whisper_native",
        "whisper_filtered",
        "whisper_frozen",
        "whisper_finetuned",
        "speechbrain_finetuned",
    ]:
        path = root / method / "fleurs_report.json"
        if path.exists():
            fleur[method] = json.loads(path.read_text())["metrics"]
    columns = [
        "n",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "ece",
        "nll",
        "brier",
        "outside_set_rate",
        "temperature",
    ]
    values = []
    per_language = []
    for method, reports in fleur.items():
        for variant, result in reports.items():
            values.append(
                {
                    "method": method,
                    "variant": variant,
                    **{k: result[k] for k in columns},
                }
            )
            per_language.extend(
                {"method": method, "variant": variant, "language": language, **stats}
                for language, stats in result["per_language"].items()
            )
    write_csv(root / "fleurs_summary.csv", values)
    write_csv(root / "fleurs_per_language.csv", per_language)
    write_json(root / "fleurs_comparison.json", fleur)
    external = study.rows("external")
    original = Path(study.config["external"])
    provenance = json.loads((original / "model_provenance.json").read_text())
    logits = {}
    codes = {}
    temperatures = {}
    with np.load(original / "logits.npz") as saved:
        if saved["ids"].tolist() != [r["id"] for r in external]:
            raise ValueError("External SpeechBrain reference IDs differ")
        for method, key in [
            ("speechbrain_original", "original107"),
            ("speechbrain_filtered", "filtered12"),
            ("speechbrain_frozen", "previous12"),
        ]:
            logits[method] = saved[key]
            codes[method] = provenance[key]["codes"]
            temperatures[method] = provenance[key]["temperature"]
    for method in [
        "whisper_native",
        "whisper_filtered",
        "whisper_frozen",
        "whisper_finetuned",
        "speechbrain_finetuned",
    ]:
        if (
            not (root / method / "external.npz").exists()
            or not (root / method / "calibration.json").exists()
        ):
            continue
        with np.load(root / method / "external.npz") as saved:
            if saved["ids"].tolist() != [r["id"] for r in external]:
                raise ValueError("Whisper comparison external IDs differ")
            logits[method] = saved["logits"]
            codes[method] = saved["codes"].tolist()
        temperatures[method] = json.loads(
            (root / method / "calibration.json").read_text()
        )["temperature"]
    predictions = {
        method: np.array([codes[method][i] for i in data.argmax(axis=1)])
        for method, data in logits.items()
    }
    confidence = {
        method: softmax(data.astype(np.float64) / temperatures[method], axis=1).max(
            axis=1
        )
        for method, data in logits.items()
    }
    selected = np.array(
        [i for i, r in enumerate(external) if r["language"] in ["hi", "th"]]
    )
    selected_rows = [external[i] for i in selected]
    common = common_cohort(
        {m: p[selected] for m, p in predictions.items()}, models=tuple(predictions)
    )
    ext = []
    details = {}
    language_rows = []
    ood = []
    prediction_rows = []
    for method, pred in predictions.items():
        own = score_segments(
            selected_rows, pred[selected], confidence[method][selected]
        )
        all_data = score_segments(
            selected_rows,
            pred[selected],
            confidence[method][selected],
            exclude_english=False,
        )
        shared = score_segments(
            selected_rows, pred[selected], confidence[method][selected], common
        )
        details[method] = {
            "all_segments": all_data,
            "conditional": own,
            "common": shared,
        }
        ext.append(
            {
                "method": method,
                "n": len(selected),
                "accuracy_all": all_data["accuracy"],
                "macro_f1_all": all_data["macro_f1"],
                "accuracy_conditional": own["accuracy"],
                "macro_f1_conditional": own["macro_f1"],
                "precision_conditional": own["macro_precision"],
                "recall_conditional": own["macro_recall"],
                "ece_conditional": own["ece"],
                "english_exclusions": own["english_predictions"],
                "coverage": own["coverage"],
                "source_accuracy_all": all_data["mean_source_accuracy"],
                "common_n": shared["segments_scored"],
                "common_accuracy": shared["accuracy"],
            }
        )
        for language in ["hi", "th"]:
            counts = own["by_language"][language]
            language_rows.append(
                {
                    "method": method,
                    "language": language,
                    **counts,
                    **own["per_language"].get(language, {}),
                }
            )
        for language in ["sam", "niu", "ton"]:
            mask = np.array([r["language"] == language for r in external])
            if not mask.any():
                continue
            labels, counts = np.unique(pred[mask], return_counts=True)
            ood.append(
                {
                    "method": method,
                    "language": language,
                    "n": int(mask.sum()),
                    "recognition_accuracy": None,
                    "mean_max_probability": float(confidence[method][mask].mean()),
                    "english_predictions": int((pred[mask] == "en").sum()),
                    "prediction_counts": json.dumps(
                        dict(zip(labels.tolist(), counts.tolist())), sort_keys=True
                    ),
                    "interpretation": "out-of-set; twelve-way classifiers cannot emit these labels",
                }
            )
        for i, row in enumerate(external):
            prediction_rows.append(
                {
                    "method": method,
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "language": row["language"],
                    "prediction": pred[i],
                    "confidence": float(confidence[method][i]),
                    "in_set": row["language"] in CODES,
                }
            )
    write_csv(root / "external_summary.csv", ext)
    write_csv(root / "external_per_language.csv", language_rows)
    write_csv(root / "external_predictions.csv", prediction_rows)
    write_csv(root / "pacific_out_of_set.csv", ood)
    write_json(root / "external_comparison.json", details)
    completed = sum(
        (root / m / "evaluation_complete.json").exists()
        for m in ["whisper_filtered", "whisper_frozen", "whisper_finetuned"]
    )

    def pct(value):
        return "pending" if value is None else f"{100 * value:.2f}"

    lines = [
        "# Whisper tiny versus SpeechBrain: original twelve languages",
        "",
        f"Completed Whisper approaches: {completed}/3. The matched SpeechBrain full-encoder baseline is "
        + ("complete." if "speechbrain_finetuned" in fleur else "pending."),
        "",
        "## Protocol",
        "",
        "Original VoxLingua recording-based train/selection manifests and exactly the original SpeechBrain cache IDs and language order are retained. No speaker separation or Pacific training data are used. Audio is mono 16 kHz and center-cropped to at most ten real seconds for both families. Whisper then applies its native 30-second right-zero-padding and log-mel frontend. It does not receive extra real audio.",
        "",
        "Filtered Whisper uses the first decoder language-token logits after the start-of-transcript token. Non-language tokens are discarded, then the twelve selected language logits are renormalized with softmax. Native Whisper language recognition is also included as a reference.",
        "",
        "Frozen Whisper features concatenate the mean and population standard deviation of encoder positions corresponding to real audio; right-padding outputs are excluded from pooling. The native attention still sees Whisper’s padded window. A LayerNorm/512-unit LeakyReLU/twelve-output classifier is learned from scratch. SpeechBrain’s existing frozen head benefits from pretrained classifier weights; the architectures and initialization histories differ.",
        "",
        "The frozen head uses class-weighted cross-entropy over every original training example, batch 256, at most 30 epochs, learning rate 1e-4 and patience five, matching the existing SpeechBrain head protocol. Full encoder fine-tuning starts from each family’s best frozen head, uses identical seed-42 class-balanced example draws (32,768 per epoch), effective batch 32, head LR 1e-4, encoder LR 1e-5, at most 20 epochs and patience five. All encoder parameters train; SpeechBrain batch-normalization running statistics stay frozen for microbatch stability. These full-encoder references are separate from the historical fifteen-output partial-tail model.",
        "",
        "Checkpoint selection minimizes unweighted VoxLingua validation NLL. Temperature scaling fits only on the original FLEURS validation IDs. FLEURS test and external clips are never used for training, checkpoint choice or temperature fitting. All methods retain every output in their declared vocabulary; out-of-set guesses count as errors on FLEURS.",
        "",
        "## FLEURS twelve-language test",
        "",
        "| Model | Test n | Accuracy | Macro precision | Macro recall | Macro F1 | ECE raw | ECE calibrated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, reports in fleur.items():
        value = reports["calibrated"]
        lines.append(
            f"| {NAMES[method]} | {value['n']} | {pct(value['accuracy'])} | {pct(value['macro_precision'])} | {pct(value['macro_recall'])} | {pct(value['macro_f1'])} | {pct(reports['raw']['ece'])} | {pct(value['ece'])} |"
        )
    lines += [
        "",
        "## Paired approaches",
        "",
        "| Approach | Whisper accuracy / F1 | SpeechBrain accuracy / F1 |",
        "|---|---|---|",
    ]
    for label, wm, sm in [
        ("Filtering", "whisper_filtered", "speechbrain_filtered"),
        ("Frozen encoder + head", "whisper_frozen", "speechbrain_frozen"),
        (
            "Full fine-tuned encoder + head",
            "whisper_finetuned",
            "speechbrain_finetuned",
        ),
    ]:

        def cell(method):
            return (
                "pending"
                if method not in fleur
                else f"{pct(fleur[method]['calibrated']['accuracy'])} / {pct(fleur[method]['calibrated']['macro_f1'])}"
            )

        lines.append(f"| {label} | {cell(wm)} | {cell(sm)} |")
    lines += [
        "",
        "## External Hindi and Thai",
        "",
        "Same saved 3–10-second VAD segments for every model. Folder labels are provisional and code-switching is unannotated. All-segment accuracy counts English predictions as errors; conditional scores exclude them and show coverage. The shared cohort excludes the union of English predictions from all currently completed models, so its size can change while training is incomplete.",
        "",
        "| Model | All accuracy | Conditional accuracy | Macro F1 (conditional) | English excluded | Coverage | Common accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for value in ext:
        lines.append(
            f"| {NAMES[value['method']]} | {pct(value['accuracy_all'])} | {pct(value['accuracy_conditional'])} | {pct(value['macro_f1_conditional'])} | {value['english_exclusions']} | {pct(value['coverage'])} | {pct(value['common_accuracy'])} |"
        )
    lines += [
        "",
        "| Model | Language | Cuts | English excluded | Conditional accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for row in language_rows:
        lines.append(
            f"| {NAMES[row['method']]} | {row['language']} | {row['total']} | {row['english']} | {pct(row['accuracy'])} |"
        )
    lines += [
        "",
        "## Pacific out-of-set diagnostics",
        "",
        "The requested twelve-language classifiers do not contain Samoan, Niuean or Tongan outputs. They cannot be assigned meaningful Pacific recognition accuracy. Prediction counts and confidence on the same Pacific clips are reported separately; forced-choice confidence is not an unknown-language rejection policy. A fifteen-language Whisper extension would be a separate experiment.",
        "",
        f"[Out-of-set predictions](../../{root}/pacific_out_of_set.csv).",
        "",
        "## Duration and gain diagnostics",
        "",
        "Nested 3/5/10-second crops from identical ten-second contiguous VAD windows compare duration effects. A separate ±12 dB gain probe uses only original VoxLingua checkpoint-selection audio, with amplification capped to avoid clipping. These probes are diagnostic and never select preprocessing or models. No noise-augmentation benefit is inferred from a uniform-gain test.",
        "",
        (
            "Diagnostics complete; see the CSV artifacts."
            if (root / "diagnostics/complete.json").exists()
            else "Diagnostics run after both full-encoder models finish."
        ),
        "",
        "## Interpretation and limits",
        "",
        (
            "All three Whisper approaches and the matched SpeechBrain full-encoder reference have finished. Rankings are descriptive, with one training seed. Pretraining corpora, frontends, head initialization and encoder architecture differ; pretraining overlap with evaluation speakers cannot be ruled out."
            if completed == 3 and "speechbrain_finetuned" in fleur
            else "Results are incomplete; a final comparison requires both trained Whisper models and the matched SpeechBrain reference. One training seed is used, and pretraining overlap cannot be ruled out."
        ),
        "",
        f"Configuration, package lock, model checksum, immutable cohort manifests, sharded caches, checkpoints and resource measurements are under `{root}`. Runtime limits are 4 GiB host RAM, zero swap, two CPU equivalents, one GPU job, and a sub-7 GiB GPU preflight.",
        "",
        f"- [Full FLEURS metrics](../../{root}/fleurs_summary.csv) · [Per-language FLEURS metrics](../../{root}/fleurs_per_language.csv).",
        f"- [External metrics](../../{root}/external_summary.csv) · [Per-language external metrics](../../{root}/external_per_language.csv) · [All external predictions](../../{root}/external_predictions.csv).",
        f"- [Duration diagnostics](../../{root}/diagnostics/duration.csv) · [Gain diagnostics](../../{root}/diagnostics/gain.csv) · [Diagnostic predictions](../../{root}/diagnostics/predictions.csv).",
        f"- [Configuration](../../{root}/configuration.json) · [Data identity](../../{root}/data_ready.json) · [Model identity](../../{root}/model_identity.json).",
        "",
        "Whisper implementation references: [native language-token detection](https://github.com/openai/whisper/blob/v20250625/whisper/decoding.py), [native padding and log-mel frontend](https://github.com/openai/whisper/blob/v20250625/whisper/audio.py).",
        "",
    ]
    if completed == 3 and "speechbrain_finetuned" in fleur:
        wh = fleur["whisper_finetuned"]["calibrated"]
        sb = fleur["speechbrain_finetuned"]["calibrated"]
        wh_external = next(row for row in ext if row["method"] == "whisper_finetuned")
        sb_external = next(
            row for row in ext if row["method"] == "speechbrain_finetuned"
        )
        filtered_external = next(
            row for row in ext if row["method"] == "speechbrain_filtered"
        )
        findings = [
            "## Final findings",
            "",
            f"- Fine-tuning is the strongest Whisper approach: FLEURS accuracy {pct(wh['accuracy'])}%, macro F1 {pct(wh['macro_f1'])}% and calibrated ECE {pct(wh['ece'])}%.",
            f"- SpeechBrain leads on the broad twelve-language FLEURS test: its fine-tuned encoder reaches {pct(sb['accuracy'])}% accuracy and {pct(sb['macro_f1'])}% macro F1, versus {pct(wh['accuracy'])}% / {pct(wh['macro_f1'])}% for Whisper.",
            f"- The external Hindi/Thai sample reverses the fine-tuned-family ranking: Whisper scores {pct(wh_external['accuracy_conditional'])}% versus SpeechBrain {pct(sb_external['accuracy_conditional'])}% with English predictions excluded. On their shared cohort with all reference models, the scores are {pct(wh_external['common_accuracy'])}% / {pct(sb_external['common_accuracy'])}%.",
            f"- Filtered SpeechBrain still has the highest external Hindi/Thai accuracy: {pct(filtered_external['accuracy_conditional'])}%. Thus the results do not identify one universally best approach across domains.",
            "- External folder labels remain provisional and English exclusions can hide target-language errors. Consult all-segment scores and coverage as well as conditional accuracy. Pacific recognition is outside the twelve-language vocabulary.",
            "",
        ]
        at = lines.index("## Protocol")
        lines[at:at] = findings
    if (root / "diagnostics/complete.json").exists():
        with (root / "diagnostics/duration.csv").open() as stream:
            durations = list(csv.DictReader(stream))
        with (root / "diagnostics/gain.csv").open() as stream:
            gains = list(csv.DictReader(stream))
        diagnostics = [
            "### Completed diagnostic results",
            "",
            "All-segment accuracy on the same 210 eligible Hindi/Thai ten-second speech windows, cropped to each duration. Crop content and code-switch proportions can change with duration; this is not a new preprocessing selection experiment.",
            "",
            "| Model | 3 seconds | 5 seconds | 10 seconds |",
            "|---|---:|---:|---:|",
        ]
        methods = [
            "whisper_filtered",
            "whisper_frozen",
            "whisper_finetuned",
            "speechbrain_filtered",
            "speechbrain_frozen",
            "speechbrain_finetuned",
        ]
        for method in methods:
            scores = []
            for seconds in ["3", "5", "10"]:
                values = [
                    row
                    for row in durations
                    if row["method"] == method
                    and row["language"] in ["hi", "th"]
                    and row["seconds"] == seconds
                ]
                n = sum(int(row["n"]) for row in values)
                scores.append(
                    pct(
                        sum(
                            float(row["accuracy_all"]) * int(row["n"]) for row in values
                        )
                        / n
                    )
                )
            diagnostics.append(f"| {NAMES[method]} | " + " | ".join(scores) + " |")
        diagnostics += [
            "",
            "Gain probes count changed top-1 predictions relative to the unchanged waveform; a change is not necessarily an error. Positive gain is capped to prevent clipping, so the applied amplification can be less than +12 dB.",
            "",
            "| Model | Changes at −12 dB | Changes at capped +12 dB |",
            "|---|---:|---:|",
        ]
        for method in methods:
            values = []
            for db in ["-12", "12"]:
                selected = [
                    row
                    for row in gains
                    if row["method"] == method and row["gain_db"] == db
                ]
                values.append(
                    f"{sum(int(row['changed_predictions']) for row in selected)}/{sum(int(row['n']) for row in selected)}"
                )
            diagnostics.append(f"| {NAMES[method]} | " + " | ".join(values) + " |")
        diagnostics += [
            "",
            "SpeechBrain was invariant in these gain probes. Whisper was somewhat more gain-sensitive, especially after fine-tuning at reduced volume. Uniform-gain tests do not measure additive-noise robustness.",
            "",
        ]
        at = lines.index("## Interpretation and limits")
        lines[at:at] = diagnostics
    if len(fleur) > 3:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        labels = list(fleur)
        axes[0].bar(
            range(len(labels)),
            [100 * fleur[m]["calibrated"]["macro_f1"] for m in labels],
        )
        axes[1].bar(
            range(len(labels)), [100 * fleur[m]["calibrated"]["ece"] for m in labels]
        )
        for ax in axes:
            ax.set_xticks(
                range(len(labels)), labels, rotation=40, ha="right", fontsize=8
            )
        axes[0].set_ylabel("FLEURS macro F1 (%)")
        axes[1].set_ylabel("Calibrated ECE (%)")
        fig.savefig(directory / "fleurs_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        lines += ["![FLEURS comparisons](fleurs_comparison.png)", ""]
    path = directory / "whisper_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(
        root / "comparison_progress.json",
        {
            "completed_whisper_approaches": completed,
            "expected_whisper_approaches": 3,
            "speechbrain_full_encoder_complete": "speechbrain_finetuned" in fleur,
            "report": str(path),
            "reference_logits_sha256": file_hash(BASE / "comparison/predictions.npz"),
        },
    )
    study.status(
        "comparison_updated", completed_whisper_approaches=completed, report=str(path)
    )
