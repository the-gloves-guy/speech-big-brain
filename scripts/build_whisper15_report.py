"""Build a portable illustrated analysis from the completed, immutable study results.

No inference, model selection, preprocessing changes or fitting occurs here.
"""

import base64
import csv
import hashlib
import html
import json
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lid.extension_data import CODES, NAMES, file_hash, write_json

ROOT = PROJECT / "artifacts/whisper_tiny15"
EXTERNAL = PROJECT / "artifacts/youtube_validation_v4"
OUTPUT = PROJECT / "reports/whisper_tiny15"
METHODS = ["frozen", "finetuned"]
COLORS = ["#087e8b", "#c05717", "#6356a3", "#455a64"]
LABELS = {
    "frozen": "Whisper15 frozen",
    "finetuned": "Whisper15 fine-tuned",
    "speechbrain_frozen15": "SpeechBrain15 frozen",
    "speechbrain_partial15": "SpeechBrain15 partial",
}
LABELS.update({f"whisper15_{m}": LABELS[m] for m in METHODS})


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def pct(value):
    return "—" if value is None or value == "" else f"{100 * float(value):.2f}%"


def number(value):
    return "—" if value is None or value == "" else f"{float(value):.4f}"


def esc(value):
    return html.escape(str(value))


def table(headers, values):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{esc(v)}</th>" for v in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>"
            for row in values
        )
        + "</tbody></table></div>"
    )


def image_data(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode(
        "ascii"
    )


def figure(path, caption, appendix=False):
    return f'<figure class="{"appendix-figure" if appendix else ""}"><img src="{image_data(path)}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'


def save_plot(fig, name, caption):
    path = OUTPUT / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(path.with_suffix("." + suffix), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return figure(path.with_suffix(".png"), caption)


def audit_exports():
    result = {}
    for method in METHODS:
        directory = PROJECT / f"artifacts/deployment/whisper_tiny15_{method}"
        expected = {}
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            checksum, name = line.split(maxsplit=1)
            expected[name] = checksum
            if file_hash(directory / name) != checksum:
                raise ValueError(f"Export file checksum failed: {name}")
        archive = directory.with_suffix(".tar.gz")
        checksum = file_hash(archive)
        if checksum != Path(str(archive) + ".sha256").read_text().split()[0]:
            raise ValueError("Archive checksum failed")
        with tarfile.open(archive, "r:gz") as stream:
            for name, expected_hash in expected.items():
                with stream.extractfile(f"{directory.name}/{name}") as member:
                    if (
                        hashlib.file_digest(member, "sha256").hexdigest()
                        != expected_hash
                    ):
                        raise ValueError(f"Archived export differs: {name}")
        verified = load(directory / "verification.json")
        if (
            verified["n"] != 14400
            or verified["prediction_disagreements"] != 0
            or not verified["offline"]
        ):
            raise ValueError("Incomplete offline export verification")
        result[method] = {
            "archive": str(archive.relative_to(PROJECT)),
            "bytes": archive.stat().st_size,
            "sha256": checksum,
            "verification": verified,
            "metadata": load(directory / "export.json"),
        }
    return result


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if load(ROOT / "status.json")["phase"] != "complete":
        raise ValueError("Study has not completed")
    exports = audit_exports()
    bench = {m: load(ROOT / m / "benchmarks.json") for m in METHODS}
    ext = load(EXTERNAL / "matched_report.json")
    ext_models = ext["models"]
    inventory = load(ROOT / "external_inventory.json")
    data = load(ROOT / "data_ready.json")["cohorts"]
    training = {m: load(ROOT / m / "history.json") for m in METHODS}
    calibration = {m: load(ROOT / m / "calibration.json") for m in METHODS}
    durations = rows(ROOT / "diagnostics/duration.csv")
    gains = rows(ROOT / "diagnostics/gain.csv")
    regression = rows(ROOT / "original_language_regression.csv")
    class_regression = rows(ROOT / "original_language_per_class_regression.csv")
    parts = []
    sections = []

    def section(identifier, title, body):
        sections.append((identifier, title))
        parts.append(
            f'<section id="{identifier}"><h2>{esc(title)}</h2>{body}</section>'
        )

    frozen = bench["frozen"]["combined_test"]["joint"]["all15"]
    tuned = bench["finetuned"]["combined_test"]["joint"]["all15"]
    extf = ext_models["whisper15_frozen"]["complete"]["all"]
    extt = ext_models["whisper15_finetuned"]["complete"]["all"]
    section(
        "findings",
        "1. Main findings",
        f"""<div class="callout"><strong>Test-set gains do not transfer uniformly to the external recordings.</strong>
    Fine-tuning raises combined held-out accuracy from {pct(frozen["accuracy"])} to {pct(tuned["accuracy"])},
    while complete external accuracy moves from {pct(extf["accuracy"])} to {pct(extt["accuracy"])}.</div>
    <p>Both models complete the 15-language task and are exported. Fine-tuning makes {round(13064 * (tuned["accuracy"] - frozen["accuracy"]))} fewer combined-test errors,
    mostly through better Pacific recognition. On external recordings, its biggest losses are Tongan and the three added Niuean sources;
    Samoan improves. These observations describe this fixed experiment, not a deployment winner chosen from external labels.</p>
    <p>Both runs selected epoch 8 and stopped after epoch 13. All 15 outputs stay enabled. The original-language accuracy cost relative to each corresponding Whisper12 control is about 0.44 percentage points.
    Hindi/Urdu remain the main weakness on FLEURS. High aggregate accuracy therefore hides meaningful differences between languages and domains.</p>""",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, title, labels, values in [
        (
            axes[0],
            "Held-out test accuracy",
            ["Original 12", "Pacific 3", "Combined"],
            [
                [
                    bench[m][c]["joint"]["all15"]["accuracy"] * 100
                    for c in ["fleurs_test", "pacific_test", "combined_test"]
                ]
                for m in METHODS
            ],
        ),
        (
            axes[1],
            "External accuracy: all segments",
            ["Complete", "Historical", "Added Niuean"],
            [
                [
                    ext_models[f"whisper15_{m}"][c]["all"]["accuracy"] * 100
                    for c in ["complete", "historical", "added"]
                ]
                for m in METHODS
            ],
        ),
    ]:
        x = np.arange(3)
        for i, m in enumerate(METHODS):
            ax.bar(
                x + (i - 0.5) * 0.34, values[i], 0.34, label=LABELS[m], color=COLORS[i]
            )
        ax.set(
            xticks=x,
            xticklabels=labels,
            ylim=(0, 105),
            ylabel="Accuracy (%)",
            title=title,
        )
        ax.legend(fontsize=8)
    parts[-1] += save_plot(
        fig,
        "domain_comparison",
        "Figure 1. Matched held-out and external cohorts. External references are provisional folder labels; the added cohort contains 93 Niuean segments from three recordings.",
    )

    section(
        "protocol",
        "2. Data, training and comparison protocol",
        """<p>The original twelve-label order is preserved and Samoan (sam), Niuean (niu) and Tongan (ton) are appended.
    Both variants start from the selected frozen-encoder Whisper12 classifier. Hidden layers and original output rows are copied exactly;
    the new rows share seed-42 initialization. Frozen training changes the classifier only. Fine-tuning updates the full audio encoder and classifier.
    SpeechBrain partial15 updates its encoder tail, so it is a different adaptation policy.</p>
    <p>Training replays the original VoxLingua recording splits and uses the unchanged Pacific manifests. Both variants use identical class-balanced sampled orders:
    32,768 draws per epoch, effective batch 32, AdamW, classifier LR 1e-4, encoder LR 1e-5, weight decay 1e-4,
    at most 20 epochs and patience 5. Checkpoints maximize combined selection macro F1, breaking ties with class-balanced NLL.</p>"""
        + table(
            ["Cohort", "Examples", "Role"],
            [
                [name, f"{data[name]['n']:,}", role]
                for name, role in [
                    ("vox_train", "Original-language replay"),
                    ("new_train", "Pacific training"),
                    ("vox_validation", "Checkpoint selection"),
                    ("new_selection", "Checkpoint selection"),
                    ("fleurs_validation", "Calibration only"),
                    ("new_calibration", "Calibration only"),
                    ("fleurs_test", "Original-language test"),
                    ("new_test", "Recording-held-out Pacific test"),
                    ("external", "External evaluation only"),
                ]
            ],
        )
        + """<p>Preprocessing remains mono 16 kHz, maximum ten-second center crop, native thirty-second Whisper padding, and mean/std pooling over real audio tokens.
        The external corpus uses one frozen Silero VAD snapshot with common segment IDs and boundaries for every model.
        Its 24 sources yield 1,336 segments: 1,243 historical plus 93 added Niuean segments.
        Historical PCM samples and timestamps match v3 exactly; float-WAV creation timestamps explain differing container hashes.</p>
        <p>One paired seed is available. Recording separation does not establish speaker independence. Correlated cuts from a recording are not independent trials;
        no segment-level significance claims or confidence intervals are inferred from these counts.</p>""",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, m in enumerate(METHODS):
        for ax, key, title in [
            (axes[0], "macro_f1", "Selection macro F1"),
            (axes[1], "balanced_nll", "Selection class-balanced NLL"),
        ]:
            ax.plot(
                [r["epoch"] for r in training[m]],
                [r[key] for r in training[m]],
                marker="o",
                color=COLORS[i],
                label=LABELS[m],
            )
            ax.axvline(8, color="grey", alpha=0.2)
            ax.set(xlabel="Epoch", title=title)
    axes[0].legend(fontsize=8)
    parts[-1] += save_plot(
        fig,
        "training_selection",
        "Figure 2. Checkpoint selection uses these curves only; the selected epoch is 8 for both models. Neither test nor external performance enters selection.",
    )

    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "nll",
        "brier",
        "ece",
    ]
    test_rows = []
    for m in METHODS:
        for cohort in ["fleurs_test", "pacific_test", "combined_test"]:
            r = bench[m][cohort]["joint"]["all15"]
            test_rows.append(
                [LABELS[m], cohort, f"{r['n']:,}"]
                + [
                    pct(r[k]) if k in metrics[:4] + ["ece"] else number(r[k])
                    for k in metrics
                ]
            )
    section(
        "test",
        "3. Held-out test performance",
        table(
            [
                "Model",
                "Test",
                "N",
                "Accuracy",
                "Precision",
                "Recall",
                "Macro F1",
                "NLL",
                "Brier",
                "ECE",
            ],
            test_rows,
        )
        + f"""<p>Joint calibration is used in this table. Combined accuracy improves by {(tuned["accuracy"] - frozen["accuracy"]) * 100:.2f} percentage points and macro F1 by {(tuned["macro_f1"] - frozen["macro_f1"]) * 100:.2f} points with fine-tuning.
        Pacific error counts fall from {4545 - round(bench["frozen"]["pacific_test"]["joint"]["all15"]["accuracy"] * 4545)} to {4545 - round(bench["finetuned"]["pacific_test"]["joint"]["all15"]["accuracy"] * 4545)}.
        The combined test mixes FLEURS speech with Pacific radio cuts and should be read alongside both constituent results.</p>
        <p>Macro metrics average represented reference languages: twelve for FLEURS, three for Pacific and fifteen for the combined test.
        Confusions into any of the other enabled outputs remain errors. Class-specific precision in a Pacific-only cohort can differ from precision on the combined test because false positives from old-language examples are absent.</p>""",
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(15)
    for i, m in enumerate(METHODS):
        r = bench[m]["combined_test"]["joint"]["all15"]["per_language"]
        ax.bar(
            x + (i - 0.5) * 0.36,
            [r[c]["f1"] * 100 for c in CODES],
            0.36,
            color=COLORS[i],
            label=LABELS[m],
        )
    ax.set(
        xticks=x,
        xticklabels=CODES,
        ylim=(0, 105),
        ylabel="F1 (%)",
        title="Combined test: per-language F1",
    )
    ax.legend(fontsize=8)
    parts[-1] += save_plot(
        fig,
        "per_language_test",
        "Figure 3. Hindi and Urdu constrain the macro score despite near-perfect performance on several other languages.",
    )
    for m in METHODS:
        parts[-1] += figure(
            ROOT / m / "figures/combined_test_joint_confusion.png",
            f"Combined-test confusion counts: {LABELS[m]}. Rows are references and columns predictions; all 15 outputs remain enabled.",
        )
    parts[-1] += table(
        [
            "Model",
            "Language",
            "N",
            "Precision",
            "Recall / accuracy",
            "F1",
            "NLL",
            "Brier",
            "ECE",
        ],
        [
            [
                LABELS[m],
                f"{c} — {name}",
                r["support"],
                pct(r["precision"]),
                pct(r["recall"]),
                pct(r["f1"]),
                number(r["nll"]),
                number(r["brier"]),
                pct(r["ece"]),
            ]
            for m in METHODS
            for c, name in zip(CODES, NAMES)
            for r in [bench[m]["combined_test"]["joint"]["all15"]["per_language"][c]]
        ],
    )

    section(
        "regression",
        "4. Original-language regression and SpeechBrain controls",
        """<p>Each Whisper15 variant is compared with its corresponding Whisper12 variant using FLEURS-only temperature fits for both; all 15 extension outputs remain active.
    Accuracy falls by 0.43 points for the frozen variant and 0.45 points for the fine-tuned variant.
    Fine-tuning's largest per-language F1 regression is Urdu; the frozen variant's largest is Hindi. Aggregate accuracy alone masks these language-specific costs.</p>"""
        + table(
            [
                "Model",
                "Δ accuracy (pp)",
                "Δ macro F1 (pp)",
                "Δ NLL",
                "Δ Brier",
                "Δ ECE (pp)",
            ],
            [
                [
                    r["method"],
                    f"{float(r['delta_accuracy']) * 100:+.2f}",
                    f"{float(r['delta_macro_f1']) * 100:+.2f}",
                    f"{float(r['delta_nll']):+.4f}",
                    f"{float(r['delta_brier']):+.4f}",
                    f"{float(r['delta_ece']) * 100:+.2f}",
                ]
                for r in regression
            ],
        ),
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(12)
    for i, m in enumerate(METHODS):
        by_code = {
            r["language"]: float(r["delta_f1"]) * 100
            for r in class_regression
            if r["method"] == m
        }
        ax.bar(
            x + (i - 0.5) * 0.36,
            [by_code[c] for c in CODES[:12]],
            0.36,
            color=COLORS[i],
            label=LABELS[m],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xticks=x,
        xticklabels=CODES[:12],
        ylabel="F1 change (percentage points)",
        title="Extension regression against matched Whisper12 controls",
    )
    ax.legend(fontsize=8)
    parts[-1] += save_plot(
        fig,
        "old_language_regression",
        "Figure 4. Differences are extension minus corresponding twelve-language control on the same FLEURS test recordings.",
    )
    comparisons = rows(ROOT / "speechbrain_comparison.csv")
    matched = [
        r for r in comparisons if r["calibration"] in ["joint", "joint_temperature"]
    ]
    parts[-1] += table(
        ["Model", "Group", "N", "Accuracy", "Macro F1", "NLL", "Brier", "ECE"],
        [
            [
                r["model"],
                r["group"],
                r["n"],
                pct(r["accuracy"]),
                pct(r["macro_f1"]),
                number(r["nll"]),
                number(r["brier"]),
                pct(r["ece"]),
            ]
            for r in matched
        ],
    )
    parts[
        -1
    ] += """<p>SpeechBrain partial15 has higher combined-test accuracy (98.45%) than Whisper15 fine-tuning (97.59%) in this study.
    SpeechBrain's frozen encoder already handles Pacific test cuts extremely well (99.71%), whereas Whisper's frozen encoder reaches 96.61%.
    The backbone pretraining and encoder update policies differ; these are system comparisons, not an isolated causal comparison of update depth.</p>"""

    calrows = []
    for m in METHODS:
        for cohort in ["fleurs_test", "pacific_test", "combined_test"]:
            for variant in ["raw", "joint", "old_only"]:
                r = bench[m][cohort][variant]["all15"]
                calrows.append(
                    [
                        LABELS[m],
                        cohort,
                        variant,
                        number(r["temperature"]),
                        number(r["nll"]),
                        number(r["brier"]),
                        pct(r["ece"]),
                    ]
                )
    section(
        "calibration",
        "5. Calibration and confidence",
        """<p>Joint temperature fitting uses 4,172 FLEURS validation recordings and 2,284 Pacific calibration cuts. The old-only comparison fit uses FLEURS validation alone.
    No test or external audio fits either temperature. Calibration rescales confidence and does not change top-1 predictions.</p>"""
        + table(
            ["Model", "Joint temperature", "FLEURS-only temperature"],
            [
                [
                    LABELS[m],
                    number(calibration[m]["temperature"]),
                    number(calibration[m]["old_only_temperature"]),
                ]
                for m in METHODS
            ],
        )
        + """<p>The frozen temperature below 1 sharpens scores; the fine-tuned temperature above 1 softens them.
        Joint calibration reduces combined-test ECE for both models. It does not improve every subgroup: fine-tuned Pacific raw ECE is 0.07%, versus 0.40% after joint calibration,
        and Pacific NLL increases. A single fitted temperature trades off the pooled calibration domains.</p>"""
        + table(
            ["Model", "Test", "Variant", "Temperature", "NLL", "Brier", "ECE"], calrows
        ),
    )
    for m in METHODS:
        for variant in ["raw", "joint"]:
            parts[-1] += figure(
                ROOT / m / f"figures/combined_test_{variant}_reliability.png",
                f"Combined-test reliability: {LABELS[m]}, {variant}. Empty bins are omitted; ECE uses fifteen confidence bins.",
            )

    allmodels = [
        "whisper15_frozen",
        "whisper15_finetuned",
        "speechbrain_frozen15",
        "speechbrain_partial15",
    ]
    exrows = []
    for model in allmodels:
        for policy in ["all", "english_excluded", "shared"]:
            r = ext_models[model]["complete"][policy]
            exrows.append(
                [
                    LABELS[model],
                    policy,
                    r["segments_scored"],
                    pct(r["coverage"]),
                    pct(r["accuracy"]),
                    pct(r["macro_f1"]),
                    pct(r["ece"]),
                    pct(r["mean_source_accuracy"]),
                ]
            )
    section(
        "external",
        "6. External recordings: transfer, exclusions and source balance",
        """<p>External labels are provisional directory labels, not segment-by-segment human annotations.
    Predictions of English can reflect real code-switching. All-segment accuracy counts every mismatch; English-excluded accuracy omits each model's English predictions.
    Shared cohorts exclude any segment predicted English by any of the four extension models, plus the compared model's own English exclusions. For these four extension models the shared cohort is identical: 1,266 segments (94.76% coverage).</p>"""
        + table(
            [
                "Model",
                "Policy",
                "Scored",
                "Coverage",
                "Accuracy",
                "Macro F1",
                "ECE",
                "Equal-source accuracy",
            ],
            exrows,
        )
        + f"""<p>The fine-tuning deficit persists on the shared cohort ({pct(ext_models["whisper15_finetuned"]["complete"]["shared"]["accuracy"])} versus {pct(ext_models["whisper15_frozen"]["complete"]["shared"]["accuracy"])}), so unequal English exclusions do not explain it.
        Source-balanced accuracy is lower than segment-weighted accuracy for both Whisper variants: performance varies substantially by recording and longer/easier sources affect the pooled score.</p>""",
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    langs = ["sam", "niu", "ton", "hi", "th"]
    x = np.arange(5)
    for i, model in enumerate(allmodels):
        ax.bar(
            x + (i - 1.5) * 0.2,
            [
                ext_models[model]["complete"]["all"]["by_language"][c]["accuracy"] * 100
                for c in langs
            ],
            0.2,
            label=LABELS[model],
            color=COLORS[i],
        )
    ax.set(
        xticks=x,
        xticklabels=langs,
        ylabel="All-segment accuracy (%)",
        ylim=(0, 105),
        title="External recognition by folder language",
    )
    ax.legend(fontsize=8, ncol=2)
    parts[-1] += save_plot(
        fig,
        "external_languages",
        "Figure 5. All 1,336 segments are included. Twelve-language models are omitted from Pacific recognition comparisons because their label sets cannot represent these languages.",
    )
    parts[
        -1
    ] += """<p>Fine-tuning improves external Samoan from 63.16% to 72.37%, but Niuean falls from 57.54% to 44.69% and Tongan from 81.35% to 50.19%.
    The Tongan loss is dominated by Māori predictions: 155/520 fine-tuned predictions versus 21/520 frozen predictions.
    Niuean-to-Tongan confusion decreases (53/179 to 9/179), yet total Niuean accuracy falls because Māori errors grow (12/179 to 58/179).
    Reducing one directional confusion therefore does not establish better recognition.</p>"""
    confusion_rows = []
    for m in METHODS:
        r = ext_models[f"whisper15_{m}"]["complete"]["all"]
        for ref, pred in [("niu", "ton"), ("ton", "niu"), ("niu", "mi"), ("ton", "mi")]:
            count = r["confusion_scored"][ref].get(pred, 0)
            n = r["by_language"][ref]["total"]
            confusion_rows.append(
                [LABELS[m], f"{ref} → {pred}", count, n, pct(count / n)]
            )
    parts[-1] += table(
        ["Model", "Direction", "Count", "Reference support", "Rate"], confusion_rows
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, m in zip(axes, METHODS):
        r = ext_models[f"whisper15_{m}"]["complete"]["all"]
        matrix = np.array(
            [[r["confusion_scored"][c].get(pred, 0) for pred in CODES] for c in langs],
            dtype=float,
        )
        matrix /= matrix.sum(1, keepdims=True)
        ax.imshow(matrix * 100, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.set(
            xticks=range(15),
            xticklabels=CODES,
            yticks=range(5),
            yticklabels=langs,
            title=LABELS[m],
            xlabel="Prediction",
        )
        for j in range(5):
            for k in range(15):
                if matrix[j, k] >= 0.02:
                    ax.text(
                        k,
                        j,
                        f"{matrix[j, k] * 100:.0f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if matrix[j, k] > 0.55 else "black",
                    )
    parts[-1] += save_plot(
        fig,
        "external_confusions_normalized",
        "Figure 6. External confusion percentages normalized within each reference language; all 15 predicted labels are visible.",
    )
    extcal = []
    for m in METHODS:
        for variant, key in [
            ("raw", "raw_probability_metrics"),
            ("joint", "probability_metrics"),
        ]:
            r = ext_models[f"whisper15_{m}"]["complete"]["all"][key]["all15"]
            extcal.append(
                [
                    LABELS[m],
                    variant,
                    number(r["nll"]),
                    number(r["brier"]),
                    pct(r["ece"]),
                ]
            )
    parts[-1] += table(["Model", "External confidence", "NLL", "Brier", "ECE"], extcal)
    parts[
        -1
    ] += """<p>Fine-tuned external ECE is 19.12% after joint calibration, compared with 0.30% on the combined test.
    This gap is consistent with a domain/calibration-transfer problem; it does not prove a specific cause such as channel, speaker, genre or volume.
    External confidence was not recalibrated to remove it.</p>"""
    parts[-1] += table(
        ["Model", "Cohort", "N", "Accuracy", "Equal-source accuracy"],
        [
            [
                LABELS[model],
                cohort,
                ext_models[model][cohort]["all"]["segments_total"],
                pct(ext_models[model][cohort]["all"]["accuracy"]),
                pct(ext_models[model][cohort]["all"]["mean_source_accuracy"]),
            ]
            for model in allmodels
            for cohort in ["historical", "added"]
        ],
    )
    parts[
        -1
    ] += """<p>Added-source accuracy is 58.06% frozen and 21.51% fine-tuned; the older Niuean cohort is 56.98% and 69.77%, respectively.
    The apparent Niuean ordering changes with the recording cohort. The report keeps both cohorts separate to make that sensitivity explicit.</p>"""
    source_ids = [r["source_id"] for r in inventory["sources"]]
    source_names = {
        r["source_id"]: Path(r["source"]).name for r in inventory["sources"]
    }
    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(source_ids))
    for i, m in enumerate(METHODS):
        stats = ext_models[f"whisper15_{m}"]["complete"]["all"]["by_source"]
        ax.scatter(
            [stats[s]["accuracy"] * 100 for s in source_ids],
            y,
            label=LABELS[m],
            color=COLORS[i],
            s=30,
        )
    ax.set(
        yticks=y,
        yticklabels=[
            s + " · " + source_names[s].encode("ascii", "ignore").decode()[:35]
            for s in source_ids
        ],
        xlim=(-3, 103),
        xlabel="All-segment source accuracy (%)",
        title="Per-recording external performance",
    )
    ax.tick_params(axis="y", labelsize=7)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    parts[-1] += save_plot(
        fig,
        "external_per_source",
        "Figure 7. One point per original source; differences across recordings limit interpretations based on pooled cuts.",
    )

    section(
        "diagnostics",
        "7. Duration and volume diagnostics",
        """<p>Duration diagnostics use nested 3/5/10-second center crops of identical ten-second VAD speech windows, shared across models.
    They are a subset of the external corpus, not the complete segmentation. There are 134 Hindi, 76 Thai, 10 Samoan and 124 Tongan windows per duration.
    <strong>No Niuean VAD region qualified for a ten-second window, so no matched Niuean duration estimate exists.</strong>
    Directional Niuean/Tongan rates on the complete corpus are reported above instead.</p>""",
    )
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharey=True)
    for ax, code in zip(axes, ["hi", "th", "sam", "ton"]):
        for i, m in enumerate(METHODS):
            selected = sorted(
                [
                    r
                    for r in durations
                    if r["model"] == f"whisper15_{m}" and r["language"] == code
                ],
                key=lambda r: int(r["seconds"]),
            )
            ax.plot(
                [int(r["seconds"]) for r in selected],
                [float(r["accuracy"]) * 100 for r in selected],
                marker="o",
                color=COLORS[i],
                label=LABELS[m],
            )
        ax.set(
            title=code, xlabel="Real audio (seconds)", xticks=[3, 5, 10], ylim=(0, 105)
        )
    axes[0].set_ylabel("All-window accuracy (%)")
    axes[0].legend(fontsize=7)
    parts[-1] += save_plot(
        fig,
        "duration_diagnostics",
        "Figure 8. Longer excerpts do not improve all cohorts monotonically. Samoan has only ten windows, so one changed prediction is ten percentage points.",
    )
    samgain = [
        r
        for r in gains
        if r["language"] == "sam" and r["model"].startswith("whisper15")
    ]
    parts[
        -1
    ] += """<p>Volume probes use twelve checkpoint-selection cuts per language, never external recordings.
    Positive gain is capped before clipping. For Samoan the mean applied positive gain is 10.38 dB rather than 12 dB.
    Frozen Samoan changes one prediction under −12 dB; fine-tuning changes one under positive gain. These small probes do not establish general volume invariance or explain the external Tongan/Niuean loss.</p>""" + table(
        [
            "Model",
            "Requested dB",
            "Mean applied dB",
            "N",
            "Changed",
            "Accuracy before",
            "Accuracy after",
            "Mean |Δ confidence|",
        ],
        [
            [
                r["model"],
                r["gain_db"],
                f"{float(r['mean_applied_gain_db']):.2f}",
                r["n"],
                r["changed_predictions"],
                pct(r["accuracy_before"]),
                pct(r["accuracy_after"]),
                number(r["mean_absolute_confidence_change"]),
            ]
            for r in samgain
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, m in zip(axes, METHODS):
        for i, db in enumerate(["-12", "12"]):
            selected = {
                r["language"]: r
                for r in gains
                if r["model"] == f"whisper15_{m}" and r["gain_db"] == db
            }
            ax.bar(
                np.arange(15) + (i - 0.5) * 0.36,
                [int(selected[c]["changed_predictions"]) for c in CODES],
                0.36,
                color=COLORS[i],
                label=f"{db} dB requested",
            )
        ax.set(
            xticks=range(15),
            xticklabels=CODES,
            title=LABELS[m],
            ylabel="Changed predictions / 12 cuts",
        )
        ax.legend(fontsize=8)
    parts[-1] += save_plot(
        fig,
        "gain_diagnostics",
        "Figure 9. Selection-only gain sensitivity. Positive applied gain varies by recording because clipping is prevented.",
    )

    export_rows = []
    for m, v in exports.items():
        export_rows.append(
            [
                LABELS[m],
                v["archive"],
                f"{v['bytes'] / 1024**2:.2f} MiB",
                v["verification"]["n"],
                v["verification"]["prediction_disagreements"],
                f"{v['verification']['max_absolute_log_probability_error']:.2e}",
            ]
        )
    section(
        "deployment",
        "8. Deployment artifacts and validation",
        table(
            [
                "Model",
                "Archive",
                "Size",
                "Verified examples",
                "Top-1 differences",
                "Max |Δ log p|",
            ],
            export_rows,
        )
        + """<p>Both decoder-free bundles support SpeechBrain <code>EncoderClassifier.from_hparams(...)</code>, contain fifteen labels and a custom Whisper backend,
        and have calibration folded exactly once into the classifier. Keep the complete extracted folder together.
        Each bundle passed relocated offline checks on 8,519 FLEURS test recordings, 4,545 Pacific test cuts and 1,336 external segments,
        plus CPU/file and padded-batch checks. Archive hashes and the hashes of archived files were rechecked while building this report.</p>
<pre><code>from speechbrain.inference.classifiers import EncoderClassifier

model = EncoderClassifier.from_hparams(
    source="/path/to/whisper_tiny15_finetuned",
    savedir="/tmp/whisper15_loaded",
    run_opts={"device": "cpu"},  # "cuda" is also supported
)
log_probs, score, index, labels = model.classify_file("audio.wav")
probabilities = log_probs.exp()
</code></pre><p>Use the frozen folder name to load the other variant. VAD and unknown-language rejection are not included.
These are language-ID exports; native Whisper transcription loaders expect a different checkpoint.</p>"""
        + table(
            ["Archive", "SHA-256"],
            [[v["archive"], v["sha256"]] for v in exports.values()],
        ),
    )

    section(
        "limits",
        "9. Interpretation and limits",
        """<p>The held-out tests support improved Pacific recognition from full Whisper encoder fine-tuning under these recording splits.
    The external results show that this gain is not a reliable predictor across the present collection of channels and genres.
    Frozen Whisper and partial SpeechBrain have similar pooled external accuracy, but their language-specific strengths differ.
    No universal deployment winner is established by this one paired seed and provisional external reference set.</p>
    <p>Original-language regression should be reviewed per language, especially Urdu and Hindi. English exclusion is a reporting policy, not ground-truth annotation or an unknown-language detector.
    Equal-source averages reduce domination by long recordings, but the corpus still contains only 24 sources with uneven language coverage.
    Manual timestamped review queues are saved for follow-up annotation; no labels were revised to improve these metrics.</p>
    <p>The study enforced a 4 GiB host-memory ceiling, zero swap, a 200% CPU quota and a single GPU job below 7 GiB.
    Training completion records report peak reserved VRAM of 0.24 GiB frozen and 0.50 GiB fine-tuned; these are allocator measurements, not total GPU memory or latency benchmarks.
    Some stage resource files reflect the final resumed verification run, so they should not be interpreted as a continuous memory trace of the entire experiment.</p>""",
    )

    section(
        "sources",
        "10. Corpus inventory and numerical appendices",
        table(
            ["Source ID", "Language", "File", "Cohort"],
            [
                [
                    r["source_id"],
                    r["language"],
                    Path(r["source"]).name,
                    "added"
                    if r["source_id"]
                    in {s["source_id"] for s in inventory["added_sources"]}
                    else "historical",
                ]
                for r in inventory["sources"]
            ],
        ),
    )
    # Include every summary/control/source table without requiring companion files.
    csv_paths = [
        ROOT / "speechbrain_comparison.csv",
        ROOT / "original_language_per_class_regression.csv",
        ROOT / "diagnostics/duration.csv",
        ROOT / "diagnostics/gain.csv",
        EXTERNAL / "matched_summary.csv",
        EXTERNAL / "matched_per_language.csv",
        EXTERNAL / "matched_per_source.csv",
    ]
    for path in csv_paths:
        values = rows(path)
        headers = list(values[0]) if values else []
        parts[-1] += (
            f"<details><summary>{esc(str(path.relative_to(PROJECT)))} — {len(values):,} rows</summary>"
            + table(headers, [[r[k] for k in headers] for r in values])
            + "</details>"
        )
    numerical = {
        "benchmarks": bench,
        "external_metrics": ext_models,
        "calibration": calibration,
        "training_histories": training,
        "exports": exports,
    }
    download = (
        "data:application/json;base64,"
        + base64.b64encode(json.dumps(numerical).encode()).decode()
    )
    parts[-1] += (
        f'<p><a download="whisper15_report_data.json" href="{download}">Download the complete numerical metrics embedded in this report</a>.</p>'
    )

    existing = sorted(
        [*ROOT.rglob("figures/*.png"), *(EXTERNAL / "figures").glob("*.png")]
    )
    section(
        "plots",
        f"11. Complete plot appendix ({len(existing)} original plots)",
        """<p>Every original study and refreshed-validation plot is embedded below, including selection/calibration cohorts, raw/joint/old-only calibration variants,
    and all supported external baseline/cohort/exclusion comparisons. Selection and calibration panels are development diagnostics, not independent test evidence.
    The main analysis above adds training, regression, domain-transfer, per-language, per-source, duration and gain figures.</p><div class="gallery">"""
        + "".join(
            figure(path, str(path.relative_to(PROJECT)), appendix=True)
            for path in existing
        )
        + "</div>",
    )
    nav = (
        "<nav><strong>Contents</strong>"
        + "".join(
            f'<a href="#{identifier}">{esc(title)}</a>'
            for identifier, title in sections
        )
        + "</nav>"
    )
    css = """body{font:16px/1.6 system-ui,sans-serif;color:#182731;background:#f2f5f6;margin:0}main{max-width:1120px;margin:auto;padding:40px;background:white}h1{font-size:38px;line-height:1.2}h2{margin-top:2.2em;color:#165968}p{max-width:95ch}.subtitle{color:#536570}.callout{padding:20px;background:#eaf5f5;border-left:5px solid #087e8b}nav{display:flex;flex-wrap:wrap;gap:8px 20px;padding:20px;background:#f2f5f6}nav strong{width:100%}nav a{font-size:14px;color:#165968}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #d6e0e2;padding:7px;text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}th{background:#eaf1f2;position:sticky;top:0}.table-wrap{overflow-x:auto;margin:20px 0}figure{margin:26px 0;break-inside:avoid}img{width:100%;height:auto}figcaption{font-size:13px;color:#536570;overflow-wrap:anywhere}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:22px}.gallery figure{margin:0}pre{padding:18px;background:#f1f4f5;overflow-x:auto;font-size:13px}details{margin:20px 0}summary{cursor:pointer;font-weight:600}code,td{overflow-wrap:anywhere}@page{size:A4;margin:16mm}@media print{body{background:white;font-size:10pt}main{padding:0;max-width:none}h1{font-size:24pt}h2{font-size:16pt;break-after:avoid}nav{display:none}.table-wrap{overflow:visible}table{font-size:7pt}th{position:static}tr{break-inside:avoid}figure{max-width:165mm;margin:5mm auto}figcaption{font-size:8pt}.gallery{display:block}.appendix-figure{width:145mm;max-height:126mm;margin:4mm auto!important}.appendix-figure img{max-height:109mm;object-fit:contain}details:not([open])>:not(summary){display:none}a{color:inherit;text-decoration:none}pre{white-space:pre-wrap}}"""
    title = "Whisper tiny 15-language performance analysis"
    rendered = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{css}</style></head><body><main><header><h1>{title}</h1><p class="subtitle">Completed seed-42 study · Original recording splits · Self-contained report with all plots<br>Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")} from saved results; no models were refitted.</p></header>{nav}'
        + "".join(parts)
        + "</main></body></html>"
    )
    output = OUTPUT / "whisper_tiny15_analysis.html"
    output.write_text(rendered, encoding="utf-8")
    inputs = [ROOT / m / "benchmarks.json" for m in METHODS] + [
        EXTERNAL / "matched_report.json",
        ROOT / "external_inventory.json",
        *csv_paths,
        *existing,
    ]
    manifest = {
        "report": str(output.relative_to(PROJECT)),
        "report_sha256": file_hash(output),
        "original_plots_embedded": len(existing),
        "analysis_plots": len(list((OUTPUT / "figures").glob("*.png"))),
        "export_audit": exports,
        "input_sha256": {str(p.relative_to(PROJECT)): file_hash(p) for p in inputs},
    }
    write_json(OUTPUT / "illustrated_report_manifest.json", manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ["report", "original_plots_embedded", "analysis_plots"]
            }
        )
    )


if __name__ == "__main__":
    build()
