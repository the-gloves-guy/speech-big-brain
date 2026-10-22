"""Build the concise report and publication figures from saved experiment results."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "artifacts/full_run"
OUTPUT = ROOT / "reports"
FIGURES = OUTPUT / "figures"
METHODS = ["original_107", "filtered_12", "retrained_head_12"]
LABELS = ["Original (107)", "Filtered (12)", "Fine-tuned (12)"]
COLORS = ["#666666", "#0072B2", "#D55E00"]


def save(figure, name):
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            FIGURES / f"{name}.{extension}",
            dpi=350,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    report = json.loads((RUN / "comparison/report.json").read_text())
    methods = report["methods"]
    raw = [methods[key] for key in METHODS]
    calibrated = [methods[key + "_temperature"] for key in METHODS]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.axisbelow": True,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.pad_inches": 0.12,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), layout="constrained")
    x = np.arange(3)
    errors = [100 * (1 - row["accuracy"]) for row in raw]
    axes[0].bar(x, errors, width=0.58, color=COLORS)
    for i, value in enumerate(errors):
        axes[0].text(i, value + 0.18, f"{value:.2f}%", ha="center", fontsize=10)
    axes[0].set(
        ylim=(0, 10.5), ylabel="Test error (%)", title="a  Classification error"
    )
    axes[0].set_xticks(x, LABELS)
    axes[0].grid(axis="y")
    for i, (before, after) in enumerate(zip(raw, calibrated)):
        values = [before["ece"] * 100, after["ece"] * 100]
        axes[1].plot([i - 0.14, i + 0.14], values, color=COLORS[i], linewidth=1.6)
        axes[1].scatter(i - 0.14, values[0], color=COLORS[i], marker="o", s=45)
        axes[1].scatter(i + 0.14, values[1], color=COLORS[i], marker="s", s=45)
        for offset, value in zip((-0.14, 0.14), values):
            axes[1].annotate(
                f"{value:.2f}",
                (i + offset, value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    axes[1].scatter([], [], color="#333333", marker="o", label="Uncalibrated")
    axes[1].scatter([], [], color="#333333", marker="s", label="Calibrated")
    axes[1].set(ylim=(0, 3.15), ylabel="ECE (%)", title="b  Calibration error")
    axes[1].set_xticks(x, LABELS)
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    save(fig, "overview")

    languages = list(raw[0]["per_language"])
    # Lowest adapted F1 first so remaining weaknesses are immediately visible.
    order = sorted(languages, key=lambda name: raw[2]["per_language"][name]["f1"])
    fig, ax = plt.subplots(figsize=(8.2, 5.4), layout="constrained")
    y = np.arange(len(order))
    for i, language in enumerate(order):
        values = [row["per_language"][language]["f1"] for row in raw]
        ax.plot(
            [min(values), max(values)], [i, i], color="#D3D3D3", linewidth=2, zorder=1
        )
    for row, label, color, marker, offset in zip(
        raw, LABELS, COLORS, ("o", "s", "^"), (-0.15, 0, 0.15)
    ):
        ax.scatter(
            [row["per_language"][name]["f1"] for name in order],
            y + offset,
            color=color,
            marker=marker,
            s=40,
            label=label,
            zorder=3,
        )
    ax.set_yticks(
        y,
        [f"{name}  (n={raw[0]['per_language'][name]['support']:,})" for name in order],
    )
    ax.invert_yaxis()
    ax.set(
        xlim=(0.50, 1.015), xlabel="Per-language F1", title="Language-level performance"
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "language_f1")

    fig, axes = plt.subplots(
        1, 3, figsize=(10, 3.6), sharex=True, sharey=True, layout="constrained"
    )
    for ax, before, after, title in zip(axes, raw, calibrated, LABELS):
        ax.plot(
            [0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="Ideal"
        )
        for row, color, marker, label in (
            (before, "#0072B2", "o", "Uncalibrated"),
            (after, "#D55E00", "s", "Calibrated"),
        ):
            bins = [entry for entry in row["reliability"] if entry["count"]]
            ax.scatter(
                [entry["confidence"] for entry in bins],
                [entry["accuracy"] for entry in bins],
                s=[10 + 100 * entry["count"] / row["n"] for entry in bins],
                color=color,
                marker=marker,
                alpha=0.8,
                label=label,
                edgecolors="white",
                linewidths=0.35,
            )
        ax.set(xlim=(0, 1.025), ylim=(0, 1.025), title=title, xlabel="Mean confidence")
        ax.set_xticks([0, 0.5, 1])
        ax.grid()
        ax.text(
            0.04,
            0.95,
            f"ECE {before['ece'] * 100:.2f}% → {after['ece'] * 100:.2f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    axes[0].set_ylabel("Observed accuracy")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "reliability")

    def pct(value):
        return f"{100 * value:.2f}"

    performance_rows = []
    calibration_rows = []
    for label, before, after in zip(LABELS, raw, calibrated):
        performance_rows.append(
            "| "
            + " | ".join(
                [label]
                + [
                    pct(before[key])
                    for key in (
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                    )
                ]
            )
            + " |"
        )
        calibration_rows.append(
            f"| {label} | {after['temperature']:.3f} | {pct(before['ece'])} | {pct(after['ece'])} | {before['nll']:.4f} | {after['nll']:.4f} |"
        )
    focus_rows = []
    for name in ("Hindi", "Urdu"):
        for label, row in zip(LABELS[1:], raw[1:]):
            values = row["per_language"][name]
            focus_rows.append(
                f"| {name} | {label} | {pct(values['precision'])} | {pct(values['recall'])} | {pct(values['f1'])} |"
            )
    errors_n = [round((1 - row["accuracy"]) * row["n"]) for row in raw]
    matrix = np.array(raw[2]["confusion_matrix"])
    hi, ur = languages.index("Hindi"), languages.index("Urdu")
    pair_errors = int(matrix[hi, ur] + matrix[ur, hi])
    counts = {}
    for split in ("train", "validation"):
        with (RUN / f"manifests/{split}.jsonl").open() as stream:
            counts[split] = sum(1 for _ in stream)
    text = f"""# Adapting SpeechBrain language identification to 12 languages

Experiment completed 22 October 2026. Percentages are rounded to two decimal places; **pp** means percentage points.

## Key findings

- **Filtering delivers most of the gain:** accuracy rises from **{pct(raw[0]["accuracy"])}% to {pct(raw[1]["accuracy"])}%** (+{100 * (raw[1]["accuracy"] - raw[0]["accuracy"]):.2f} pp), without retraining.
- **Fine-tuning improves accuracy to {pct(raw[2]["accuracy"])}%** (+{100 * (raw[2]["accuracy"] - raw[1]["accuracy"]):.2f} pp over filtering) and macro F1 to **{pct(raw[2]["macro_f1"])}%**. Errors fall from {errors_n[1]} to {errors_n[2]} ({100 * (errors_n[1] - errors_n[2]) / errors_n[1]:.1f}% relative reduction).
- **The calibrated fine-tuned head has the lowest ECE: {pct(calibrated[2]["ece"])}%**, versus {pct(calibrated[1]["ece"])}% for calibrated filtering.
- **Hindi–Urdu confusion dominates the residual errors:** {pair_errors}/{errors_n[2]} ({100 * pair_errors / errors_n[2]:.1f}%) are predictions between this pair.

## Experiments and protocol

| Experiment | Modification | Training |
|---|---|---|
| Original (107) | Unmodified 107-language classifier | None |
| Filtered (12) | Select target logits, then softmax over the 12 languages | None |
| Fine-tuned (12) | Initialize from the pretrained classifier; retrain its MLP and 12-way output | Frozen ECAPA encoder; selected VoxLingua data |

- **Languages:** Arabic, English, Spanish, Persian, Hindi, Indonesian, Māori, Pushto, Russian, Thai, Urdu, Mandarin Chinese.
- **Base model:** `speechbrain/lang-id-voxlingua107-ecapa`; immutable model and dataset revisions are recorded in the linked provenance report.
- **Audio:** mono, 16 kHz; center crop to at most 10 seconds. Embeddings and original logits are cached once.
- **Head fitting:** {counts["train"]:,} VoxLingua training clips; {counts["validation"]:,} validation clips. Source-video groups are disjoint between these splits.
- **Optimization:** RTX 3070 (8 GB); AdamW, learning rate 10⁻⁴, weight decay 10⁻⁴, batch size 256, inverse-frequency class weights, seed 42. Early stopping selects epoch 6 of 11 by VoxLingua validation NLL.
- **Calibration:** a separate scalar temperature for each model, fitted by NLL on **4,172 FLEURS validation recordings**; no FLEURS data used for head fitting or checkpoint selection.
- **Evaluation:** **8,519 FLEURS test recordings** across the selected languages. Accuracy is sample-weighted; precision, recall and F1 below are macro-averaged over the 12 target languages. Original-model predictions outside the target set count as errors.
- **ECE:** sample-weighted, top-label expected calibration error with 15 equal-width confidence bins; lower is better. Positive scalar temperature scaling leaves predicted labels unchanged.

## Overall performance

Classification metrics and ECE are percentages; temperature and NLL are unitless. Classification metrics are identical before and after temperature scaling.

| Model | Accuracy ↑ | Macro precision ↑ | Macro recall ↑ | Macro F1 ↑ |
|---|---:|---:|---:|---:|
{chr(10).join(performance_rows)}

| Model | Temperature | ECE before ↓ | ECE after ↓ | NLL before ↓ | NLL after ↓ |
|---|---:|---:|---:|---:|---:|
{chr(10).join(calibration_rows)}

![Classification and calibration error](figures/overview.png)

*Figure 1. Filtering sharply reduces classification error; fine-tuning adds a smaller gain. Temperature calibration reduces ECE for all three models. Both axes start at zero. ECE values are shown as percentages.*

- The original model predicts outside the allowed set on **{pct(raw[0]["outside_set_rate"])}%** of test recordings. Restricting the output vocabulary removes these predictions by construction.
- Calibration improves test ECE for all models; original-model test NLL increases slightly despite fitting temperature on validation NLL. Calibration gains depend on the metric and evaluation distribution.

## Language-level effects

![F1 for each target language](figures/language_f1.png)

*Figure 2. Per-language F1, with test support in parentheses. The point-plot axis spans 50–100%; markers distinguish all three models. Temperature calibration does not change these scores.*

- **Filtering:** Russian F1 increases from {pct(raw[0]["per_language"]["Russian"]["f1"])}% to {pct(raw[1]["per_language"]["Russian"]["f1"])}%; Indonesian F1 from {pct(raw[0]["per_language"]["Indonesian"]["f1"])}% to {pct(raw[1]["per_language"]["Indonesian"]["f1"])}%.
- **Fine-tuning:** Urdu F1 increases by {100 * (raw[2]["per_language"]["Urdu"]["f1"] - raw[1]["per_language"]["Urdu"]["f1"]):.2f} pp and Māori F1 by {100 * (raw[2]["per_language"]["Maori"]["f1"] - raw[1]["per_language"]["Maori"]["f1"]):.2f} pp over filtering.
- **Trade-offs:** Hindi precision improves while recall falls; Urdu recall improves while precision falls. Pushto and Indonesian have small F1 declines after fine-tuning.

| Language | Model | Precision (%) | Recall (%) | F1 (%) |
|---|---|---:|---:|---:|
{chr(10).join(focus_rows)}

![Reliability diagrams before and after calibration](figures/reliability.png)

*Figure 3. Reliability diagrams using the same 15 bins as ECE. Each point is a nonempty bin; marker area is proportional to its sample count plus a visibility floor. The diagonal indicates ideal calibration. Sparse bins can be noisy; most recordings have high confidence.*

## Interpretation and scope

- **Filtering is a strong low-cost adaptation.** Fine-tuning offers a further accuracy/F1 gain, and its calibrated variant gives the best observed calibration.
- **Next experiment:** target Hindi–Urdu discrimination and test on deployment audio, including languages outside the selected set and silence.
- Results are from **one training seed and one benchmark**, with no uncertainty intervals or significance tests. The 0.38 pp fine-tuning gain is an observed difference, not a demonstrated generalization guarantee.
- This is a closed-set evaluation: no rejection policy was evaluated. FLEURS Arabic uses `ar_eg`, Spanish `es_419`, and Chinese `cmn_hans_cn`; conclusions are limited to these tested configurations.

## Artifacts and reproduction

- [Full metrics](../artifacts/full_run/comparison/summary.csv) · [Per-language metrics](../artifacts/full_run/comparison/per_language.csv) · [Detailed report and provenance](../artifacts/full_run/comparison/report.json).
- [Deployable SpeechBrain bundle](../artifacts/full_run/speechbrain_bundle/): calibration folded into classifier weights; offline loading and all 8,519 cached test predictions verified.
- Figures are supplied as **350 dpi PNG**, editable **SVG**, and vector **PDF** in [figures/](figures/). [Combined results CSV](overall_results.csv) preserves unrounded metrics as fractions.
- Regenerate this document, figures and results CSV with `uv run python scripts/write_experiment_report.py`. This reads saved results; it does not retrain or rerun evaluation.
"""
    (OUTPUT / "language_id_experiments.md").write_text(text)
    with (OUTPUT / "overall_results.csv").open("w", newline="") as stream:
        columns = [
            "model",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "ece_before",
            "ece_after",
            "temperature",
            "nll_before",
            "nll_after",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for label, before, after in zip(LABELS, raw, calibrated):
            writer.writerow(
                {
                    "model": label,
                    **{key: before[key] for key in columns[1:5]},
                    "ece_before": before["ece"],
                    "ece_after": after["ece"],
                    "temperature": after["temperature"],
                    "nll_before": before["nll"],
                    "nll_after": after["nll"],
                }
            )
    print(
        f"Wrote {OUTPUT / 'language_id_experiments.md'} and three PNG/PDF/SVG figures"
    )


if __name__ == "__main__":
    main()
