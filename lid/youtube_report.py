"""Comparable external-validation reports for legacy and Pacific language groups."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .extension_data import BASE, RUN, write_json
from .youtube_metrics import common_cohort, score_segments

LANGUAGES = {
    "hi": "Hindi",
    "th": "Thai",
    "sam": "Samoan",
    "niu": "Niuean",
    "ton": "Tongan",
}
GROUPS = {"original_sample": ["hi", "th"], "pacific": ["sam", "niu", "ton"]}


def summarize_groups(rows, predictions, confidence, codes, raw_confidence=None):
    """Compare only models that support every target language in the group."""
    result = {}
    for group, targets in GROUPS.items():
        indices = np.array(
            [i for i, row in enumerate(rows) if row["language"] in targets], dtype=int
        )
        if not len(indices):
            continue
        data = [rows[i] for i in indices]
        present = {r["language"] for r in data}
        supported = tuple(name for name in predictions if present.issubset(codes[name]))
        selected_predictions = {
            name: np.asarray(value)[indices] for name, value in predictions.items()
        }
        common = (
            common_cohort(selected_predictions, models=supported)
            if supported
            else np.zeros(len(data), dtype=bool)
        )
        result[group] = {}
        for name, values in selected_predictions.items():
            if name not in supported:
                result[group][name] = {
                    "supported": False,
                    "missing_outputs": sorted(present - set(codes[name])),
                    "all_segments": None,
                    "conditional": None,
                    "common_cohort": None,
                }
                continue
            scores = np.asarray(confidence[name])[indices]
            own = score_segments(data, values, scores)
            all_segments = score_segments(data, values, scores, exclude_english=False)
            if raw_confidence is not None:
                raw = np.asarray(raw_confidence[name])[indices]
                own["ece_raw"] = score_segments(data, values, raw)["ece"]
                all_segments["ece_raw"] = score_segments(
                    data, values, raw, exclude_english=False
                )["ece"]
            result[group][name] = {
                "supported": True,
                "all_segments": all_segments,
                "conditional": own,
                "common_cohort": score_segments(data, values, scores, common),
                "common_models": list(supported),
            }
    return result


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def domain_comparison(reports):
    """Per-language recall is accuracy conditional on the true language."""
    legacy = json.loads((BASE / "comparison/report.json").read_text())["methods"]
    old_keys = {
        "original107": "original_107",
        "filtered12": "filtered_12",
        "previous12": "retrained_head_12",
    }
    rows = []
    for model, report in reports.items():
        for code, language in LANGUAGES.items():
            external = report["conditional"]["by_language"].get(code)
            if not external or not external["total"] or code not in report["codes"]:
                continue
            if model in old_keys:
                reference = legacy[old_keys[model]]["per_language"].get(language)
            else:
                method = "frozen" if model == "frozen15" else "partial"
                group = "old12" if code in ["hi", "th"] else "new3"
                reference = json.loads((RUN / method / "report.json").read_text())[
                    "groups"
                ][group]["raw"]["per_language"].get(language)
            if reference is None:
                continue
            accuracy_all = external["correct"] / external["total"]
            rows.append(
                {
                    "model": model,
                    "language": code,
                    "reference_dataset": "FLEURS test"
                    if code in ["hi", "th"]
                    else "radio recording-held-out test",
                    "reference_n": reference["support"],
                    "reference_accuracy": reference["recall"],
                    "external_n": external["total"],
                    "external_accuracy_all": accuracy_all,
                    "external_accuracy_conditional": external["accuracy"],
                    "all_accuracy_delta_pp": 100 * (accuracy_all - reference["recall"]),
                }
            )
    return rows


def previous_cohort_check(output, ids, predictions, reports):
    previous = Path("artifacts/youtube_validation_v2")
    if previous.resolve() == output.resolve() or not (previous / "logits.npz").exists():
        return {}
    provenance = json.loads((previous / "model_provenance.json").read_text())
    current_provenance = json.loads((output / "model_provenance.json").read_text())
    with np.load(previous / "logits.npz") as saved:
        positions = {value: index for index, value in enumerate(ids)}
        old_indices = [
            i for i, value in enumerate(saved["ids"].tolist()) if value in positions
        ]
        new_indices = [positions[saved["ids"][i]] for i in old_indices]
        result = {
            "previous_segments": len(saved["ids"]),
            "unchanged_segments": len(old_indices),
            "models": {},
        }
        for name in predictions:
            if name not in saved or not old_indices:
                continue
            old = [
                provenance[name]["codes"][i]
                for i in saved[name][old_indices].argmax(axis=1)
            ]
            new = np.asarray(predictions[name])[new_indices]
            matches = int((new == old).sum())
            result["models"][name] = {
                "matching_top1": matches,
                "total": len(old_indices),
                "agreement": matches / len(old_indices),
                "identical_model_provenance": provenance[name]
                == current_provenance[name],
            }
    return result


def write_report(output, rows, prepared, reports):
    from .youtube_evaluate import MODELS

    output = Path(output).resolve()
    destination = Path("reports") / output.name
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with np.load(output / "logits.npz") as saved:
        ids = saved["ids"].tolist()
        if ids != [row["id"] for row in rows]:
            raise ValueError("Prediction IDs do not match validation segments")
        predictions, confidence, raw_confidence = {}, {}, {}
        for name, report in reports.items():
            values = saved[name].astype(np.float64)
            predictions[name] = np.array(
                [report["codes"][i] for i in values.argmax(axis=1)]
            )
            confidence[name] = softmax(values / report["temperature"], axis=1).max(
                axis=1
            )
            raw_confidence[name] = softmax(values, axis=1).max(axis=1)
    groups = summarize_groups(
        rows,
        predictions,
        confidence,
        {name: report["codes"] for name, report in reports.items()},
        raw_confidence,
    )
    write_json(output / "group_report.json", groups)
    group_rows = []
    columns = [
        "segments_total",
        "segments_scored",
        "english_predictions",
        "coverage",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "ece",
        "mean_source_accuracy",
        "duration_weighted_accuracy",
    ]
    for group, models in groups.items():
        for name, result in models.items():
            for cohort in ["all_segments", "conditional", "common_cohort"]:
                value = result[cohort]
                group_rows.append(
                    {
                        "group": group,
                        "model": name,
                        "cohort": cohort,
                        "supported": result["supported"],
                        **{key: value.get(key) if value else None for key in columns},
                        "ece_raw": value.get("ece_raw") if value else None,
                    }
                )
    write_csv(output / "group_summary.csv", group_rows)
    domains = domain_comparison(reports)
    write_csv(output / "domain_comparison.csv", domains)
    parity = previous_cohort_check(output, ids, predictions, reports)
    write_json(output / "previous_cohort_check.json", parity)
    existing = json.loads((output / "report.json").read_text())
    existing.update(
        group_reports=groups,
        training_policy="original recording-based training sets; no speaker-separated models",
        previous_cohort_check=parity,
    )
    write_json(output / "report.json", existing)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = ["frozen15", "partial15"]
    targets = [
        code for code in LANGUAGES if any(row["language"] == code for row in rows)
    ]

    def save(fig, name):
        for suffix in ["png", "pdf", "svg"]:
            fig.savefig(figures / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for model, offset, color in [
        ("frozen15", -0.18, "#0072B2"),
        ("partial15", 0.18, "#D55E00"),
    ]:
        by_language = reports[model]["conditional"]["by_language"]
        axes[0].bar(
            np.arange(len(targets)) + offset,
            [
                100 * by_language[c]["correct"] / by_language[c]["total"]
                for c in targets
            ],
            width=0.34,
            label=MODELS[model],
            color=color,
        )
        axes[1].bar(
            np.arange(len(targets)) + offset,
            [
                100 * by_language[c]["accuracy"]
                if by_language[c]["accuracy"] is not None
                else np.nan
                for c in targets
            ],
            width=0.34,
            label=MODELS[model],
            color=color,
        )
    for ax, title in zip(axes, ["All speech segments", "English predictions excluded"]):
        ax.set_xticks(range(len(targets)), [LANGUAGES[c] for c in targets])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Folder-label accuracy (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    save(fig, "language_accuracy")
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for j, name in enumerate(reports):
        values = []
        for code in ["hi", "th"]:
            counts = reports[name]["conditional"]["by_language"].get(code, {})
            values.append(
                100 * counts["correct"] / counts["total"]
                if counts.get("total")
                else np.nan
            )
        ax.bar(np.arange(2) + (j - 2) * 0.15, values, width=0.14, label=MODELS[name])
    ax.set_xticks(range(2), ["Hindi", "Thai"])
    ax.set_ylabel("All-segment folder-label accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    save(fig, "original_language_controls")
    sources = prepared["source_summaries"]
    fig, ax = plt.subplots(
        figsize=(11, max(5, len(sources) * 0.3)), layout="constrained"
    )
    for model, offset, color in [
        ("frozen15", -0.12, "#0072B2"),
        ("partial15", 0.12, "#D55E00"),
    ]:
        source_metrics = reports[model]["conditional"]["by_source"]
        values = [
            100 * source_metrics[s["source_id"]]["accuracy"]
            if s["source_id"] in source_metrics
            and source_metrics[s["source_id"]]["accuracy"] is not None
            else np.nan
            for s in sources
        ]
        ax.scatter(
            values,
            np.arange(len(sources)) + offset,
            label=MODELS[model],
            color=color,
            s=20,
        )
    ax.set_yticks(
        range(len(sources)),
        [
            f"{LANGUAGES[source['language']]} clip {sum(s['language'] == source['language'] for s in sources[: i + 1])}"
            for i, source in enumerate(sources)
        ],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Conditional folder-label accuracy (%)")
    ax.legend(fontsize=8)
    save(fig, "per_source")

    def pct(value):
        return "—" if value is None else f"{100 * value:.2f}"

    names = {
        "original_sample": "Original-language sample: Hindi and Thai",
        "pacific": "New Pacific languages: Samoan, Niuean and Tongan",
    }
    lines = [
        "# Expanded external validation: original and Pacific languages",
        "",
        "All models use the original recording-based training sets. No speaker-separated checkpoint is used, and the stopped speaker-split worker remains stopped. This run performs inference only: no retraining, checkpoint selection or calibration fitting on these clips.",
        "",
        "Folder labels are provisional. Code-switching, translations, music and errors in folder labels have not been manually annotated. All-segment scores count English predictions as errors against the folder label; conditional scores exclude them, matching the previous policy. Coverage is reported because exclusions may hide both real English and target-language errors.",
        "",
        "## Validation corpus",
        "",
        f"{len(sources)} source files; {sum(s['decoded_seconds'] for s in sources) / 60:.1f} minutes before VAD; {len(rows)} retained 3–10-second speech segments ({sum(r['duration'] for r in rows) / 60:.1f} minutes). VAD, decoding and crop settings are identical across models. Ordinary source clips may overlap or share speakers; segment counts are not independent sample counts.",
        "",
        "| Language | Source files | Retained segments | Retained minutes |",
        "|---|---:|---:|---:|",
    ]
    for code in targets:
        selected = [r for r in rows if r["language"] == code]
        lines.append(
            f"| {LANGUAGES[code]} | {sum(s['language'] == code for s in sources)} | {len(selected)} | {sum(r['duration'] for r in selected) / 60:.1f} |"
        )
    lines += [
        "",
        "## Group comparisons",
        "",
        "All values below are percentages. Macro metrics use only the group’s folder labels while retaining all model outputs as possible predictions. Source mean weights each source file equally, whereas pooled accuracy weights each segment equally. Unsupported model/language combinations are not treated as valid zero-accuracy baselines.",
        "",
    ]
    for group, models in groups.items():
        lines += [
            f"### {names[group]}",
            "",
            "| Model | All accuracy | Conditional accuracy | Macro F1 (conditional) | English excluded | Coverage | Source mean (all) | ECE (conditional) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, result in models.items():
            if not result["supported"]:
                lines.append(
                    f"| {MODELS[name]} | Unsupported | — | — | — | — | — | — |"
                )
                continue
            all_data = result["all_segments"]
            own = result["conditional"]
            lines.append(
                f"| {MODELS[name]} | {pct(all_data['accuracy'])} | {pct(own['accuracy'])} | {pct(own['macro_f1'])} | {own['english_predictions']} | {pct(own['coverage'])} | {pct(all_data['mean_source_accuracy'])} | {pct(own['ece'])} |"
            )
        lines += [
            "",
            "Shared-cohort comparison excludes a cut if any supported model in this group predicts English. This gives those models identical denominators, but does not verify the true language of excluded cuts.",
            "",
            "| Model | Common cuts | Accuracy | Macro F1 |",
            "|---|---:|---:|---:|",
        ]
        for name, result in models.items():
            if result["supported"]:
                value = result["common_cohort"]
                lines.append(
                    f"| {MODELS[name]} | {value['segments_scored']} | {pct(value['accuracy'])} | {pct(value['macro_f1'])} |"
                )
        lines.append("")
    lines += [
        "## Per-language results",
        "",
        "| Model | Language | Cuts | English excluded | All accuracy | Conditional accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        for code in targets:
            value = report["conditional"]["by_language"][code]
            if code not in report["codes"]:
                continue
            lines.append(
                f"| {MODELS[name]} | {LANGUAGES[code]} | {value['total']} | {value['english']} | {pct(value['correct'] / value['total'])} | {pct(value['accuracy'])} |"
            )
    lines += [
        "",
        "![Accuracy by language](figures/language_accuracy.png)",
        "",
        "![Original-language controls](figures/original_language_controls.png)",
        "",
        "## Comparison with the original in-domain tests",
        "",
        "These are the same checkpoint weights, but different audio populations and segmentation protocols. The reference is FLEURS read speech for Hindi/Thai and recording-held-out radio speech for Pacific languages. Per-language accuracy is recall conditional on that true folder language. Differences are descriptive domain comparisons, not controlled estimates of the cause of generalisation loss.",
        "",
        "| Model | Language | Reference | Reference cuts | Reference accuracy | External all accuracy | External conditional accuracy | All-accuracy change (pp) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for value in domains:
        lines.append(
            f"| {MODELS[value['model']]} | {LANGUAGES[value['language']]} | {value['reference_dataset']} | {value['reference_n']} | {pct(value['reference_accuracy'])} | {pct(value['external_accuracy_all'])} | {pct(value['external_accuracy_conditional'])} | {value['all_accuracy_delta_pp']:+.2f} |"
        )
    lines += ["", "## Interpretation", ""]
    for model in primary:
        values = [
            r for r in domains if r["model"] == model and r["language"] in ["hi", "th"]
        ]
        lines.append(
            f"- {MODELS[model]}: "
            + "; ".join(
                f"{LANGUAGES[v['language']]} external all-segment accuracy differs from FLEURS by {v['all_accuracy_delta_pp']:+.2f} pp"
                for v in values
            )
            + "."
        )
    if "hi" in targets:
        hindi = [
            value
            for value in domains
            if value["model"] in primary and value["language"] == "hi"
        ]
        if hindi and all(value["all_accuracy_delta_pp"] >= 0 for value in hindi):
            lines.append(
                "- Hindi does not show the large decline seen in Pacific validation: both 15-language models have higher all-segment accuracy here than on their Hindi FLEURS test subset."
            )
    if "th" in targets:
        detail = []
        for name in primary:
            value = reports[name]["conditional"]["by_language"]["th"]
            detail.append(
                f"{MODELS[name]} excludes {value['english']}/{value['total']} Thai-folder segments as predicted English; its all-segment accuracy is {pct(value['correct'] / value['total'])}% and retained-segment accuracy is {pct(value['accuracy'])}%"
            )
        lines.append(
            "- "
            + "; ".join(detail)
            + ". The English predictions may be genuine code-switching or classification errors; the folder labels cannot distinguish these. The all-segment decline must not be read as a confirmed Thai-recognition failure."
        )
    if "original_sample" in groups:
        comparisons = []
        for code in ["hi", "th"]:
            if code not in targets:
                continue
            baseline = reports["previous12"]["conditional"]["by_language"][code][
                "accuracy"
            ]
            if baseline is not None:
                differences = [
                    reports[name]["conditional"]["by_language"][code]["accuracy"]
                    for name in primary
                ]
                if all(value is not None for value in differences):
                    comparisons.append(
                        f"{LANGUAGES[code]}: {100 * (differences[0] - baseline):+.2f} / {100 * (differences[1] - baseline):+.2f} pp"
                    )
        lines.append(
            "- Relative to the previous fine-tuned 12-language model on exactly these clips, conditional accuracy changes by "
            + "; ".join(comparisons)
            + " (frozen / partial)."
        )
    if "original_sample" in groups:
        lines.append(
            "- Compare the original/filtered/fine-tuned 12-language controls on the same Hindi/Thai clips to distinguish an existing external-domain difficulty from a change associated with adding the Pacific outputs. This is a sample of two original languages and cannot establish the trend for all twelve."
        )
    lines += [
        "- The expanded Pacific corpus differs from the previous validation corpus. A change in its pooled score is not evidence that the unchanged model improved or degraded; long clips can dominate pooled metrics. Consult the per-source results and equal-source mean.",
        "",
        "## Sources and review",
        "",
        "| Source | Language | Audio minutes | Cuts | Scored minutes |",
        "|---|---|---:|---:|---:|",
    ]
    for source in sources:
        lines.append(
            f"| {Path(source['source']).name.replace('|', '/')} | {source['language']} | {source['decoded_seconds'] / 60:.1f} | {source['retained_segments']} | {source['retained_seconds'] / 60:.1f} |"
        )
    lines += [
        "",
        "![Results by source](figures/per_source.png)",
        "",
        "The timestamped review queue lists English predictions, disagreements and other folder-label mismatches. No annotation has been inferred from model consensus.",
        "",
        "## Reproducibility and artifacts",
        "",
        f"- Source/model/VAD hashes and checkpoint provenance: `{output.relative_to(Path.cwd())}`. All weights come from `artifacts/full_run` or `artifacts/extension15`.",
        f"- [Group metrics](../../{output.relative_to(Path.cwd())}/group_summary.csv), [per-language metrics](../../{output.relative_to(Path.cwd())}/per_language.csv), [per-source metrics](../../{output.relative_to(Path.cwd())}/per_source.csv), [domain comparisons](../../{output.relative_to(Path.cwd())}/domain_comparison.csv).",
        f"- [Every prediction](../../{output.relative_to(Path.cwd())}/predictions.csv), [full JSON report](../../{output.relative_to(Path.cwd())}/report.json), [review queue](../../{output.relative_to(Path.cwd())}/review_queue.csv).",
        f"- Unchanged segments matched to the previous Pacific evaluation: {parity.get('unchanged_segments', 0)}. [Prediction-agreement check](../../{output.relative_to(Path.cwd())}/previous_cohort_check.json).",
        "- Previously fitted temperatures are applied once; no validation audio is used for fitting. All-segment, conditional and shared-cohort metrics are saved separately. Figures are available as PNG, PDF and SVG.",
        "",
    ]
    path = destination / "held_out_evaluation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
