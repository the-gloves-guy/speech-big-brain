"""One portable Small15 analysis with all plots, controls, diagnostics and exports."""

import csv
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .extension_data import CODES, NAMES, file_hash, write_json
from .whisper_small_results import portable_figure, render_pdf
from .youtube_evaluate import write_csv


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def pct(x):
    return "—" if x is None or x == "" else f"{float(x) * 100:.2f}%"


def num(x):
    return "—" if x is None or x == "" else f"{float(x):.4f}"


def table(headers, rows):
    return (
        '<div class="table"><table><thead><tr>'
        + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def build(study):
    root = study.root
    output = Path("reports") / root.name
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    methods = ["frozen", "finetuned"]
    colors = ["#087e8b", "#c05717", "#6551a0", "#54666c", "#698b32", "#a94363"]
    bench = {m: read(root / m / "benchmarks.json") for m in methods}
    tiny = {m: read(study.reference / m / "benchmarks.json") for m in methods}
    external = read(root / "external/matched_report.json")
    history = {m: read(root / m / "history.json") for m in methods}
    metadata = read(root / "data_ready.json")
    warm = read(study.base / "whisper_frozen/fleurs_report.json")["metrics"][
        "calibrated"
    ]
    titles = {f"whisper_small15_{m}": f"Small15 {m}" for m in methods}
    titles.update({f"whisper_tiny15_{m}": f"Tiny15 {m}" for m in methods})
    titles.update(
        {
            "speechbrain_frozen15": "SpeechBrain frozen15",
            "speechbrain_partial15": "SpeechBrain partial15",
        }
    )
    model_names = list(titles)
    pieces = []
    section_titles = []
    plot_paths = []

    def section(title, body):
        section_titles.append(title)
        pieces.append(f"<section><h2>{html.escape(title)}</h2>{body}</section>")

    def plot(fig, name, caption):
        path = figures / name
        fig.tight_layout()
        for suffix in ["png", "pdf", "svg"]:
            fig.savefig(path.with_suffix("." + suffix), dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(path.with_suffix(".png"))
        pieces.append(portable_figure(path.with_suffix(".png"), caption))

    f = bench["frozen"]["combined_test"]["joint"]["all15"]
    t = bench["finetuned"]["combined_test"]["joint"]["all15"]
    ef = external["models"]["whisper_small15_frozen"]["complete"]["all"]
    et = external["models"]["whisper_small15_finetuned"]["complete"]["all"]
    section(
        "1. Main findings",
        f"""<div class="callout">Whisper Small15 combined-test accuracy is <strong>{pct(f["accuracy"])} frozen</strong> and <strong>{pct(t["accuracy"])} fine-tuned</strong>.
        On the complete external corpus it is <strong>{pct(ef["accuracy"])}</strong> and <strong>{pct(et["accuracy"])}</strong>, respectively.</div>
        <p>The held-out difference is {(t["accuracy"] - f["accuracy"]) * 100:+.2f} percentage points; the external difference is {(et["accuracy"] - ef["accuracy"]) * 100:+.2f} points.
        These are descriptive results from one paired seed, not significance tests or an externally selected deployment winner.
        Read original-language, Pacific and per-recording results separately before interpreting the pooled score.</p>""",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, m in enumerate(methods):
        x = np.arange(3)
        axes[0].bar(
            x + (i - 0.5) * 0.35,
            [
                bench[m][c]["joint"]["all15"]["accuracy"] * 100
                for c in ["fleurs_test", "pacific_test", "combined_test"]
            ],
            0.35,
            color=colors[i],
            label=m,
        )
        axes[1].bar(
            x + (i - 0.5) * 0.35,
            [
                external["models"][f"whisper_small15_{m}"][c]["all"]["accuracy"] * 100
                for c in ["complete", "historical", "added"]
            ],
            0.35,
            color=colors[i],
            label=m,
        )
    for ax, title, labels in [
        (axes[0], "Held-out accuracy", ["Original 12", "Pacific 3", "Combined"]),
        (axes[1], "External accuracy", ["Complete", "Historical", "Added Niuean"]),
    ]:
        ax.set(
            title=title,
            xticks=range(3),
            xticklabels=labels,
            ylim=(0, 105),
            ylabel="Accuracy (%)",
        )
        ax.legend(fontsize=8)
    plot(
        fig,
        "domains",
        "Held-out and external cohort results. External references remain provisional directory labels.",
    )

    section(
        "2. Protocol, data and training",
        """<p>This focused rerun trains a frozen Small12 classifier as the warm-start, then two Small15 variants with identical initial weights and seed-42 class-balanced sample orders.
        It does not train native/filtered Small controls or a fine-tuned Small12 control. Classifier hidden layers and twelve output rows are copied from the selected Small12 head; sam, niu and ton are appended.
        Tiny parameters and embeddings are never reused as Small parameters or embeddings.</p>
        <p>The original recording-based VoxLingua/Pacific splits are unchanged. The extension uses 32,768 sampled examples per epoch, effective batch 32, AdamW,
        classifier LR 1e-4, encoder LR 1e-5, weight decay 1e-4, maximum 20 epochs and patience 5.
        Selection maximizes combined fifteen-class macro F1 with class-balanced NLL as tie-breaker. The warm-start instead uses original VoxLingua validation NLL and the original weighted classifier schedule.</p>
        <p>Audio is mono 16 kHz, at most ten real seconds by center crop, padded to Whisper's native thirty-second window. Pooling uses real-token mean and population standard deviation.
        Small's decoder is omitted from classifier training and deployment. Fine-tuning updates the whole Small audio encoder; SpeechBrain partial15 updates its encoder tail.</p>"""
        + table(
            ["Cohort", "N"], [[name, r["n"]] for name, r in metadata["cohorts"].items()]
        )
        + "<p>Joint calibration uses FLEURS validation plus Pacific calibration only. External clips never enter training, selection, calibration fitting or preprocessing tuning. Recording splits do not establish speaker independence.</p>",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, m in enumerate(methods):
        for ax, key in zip(axes, ["macro_f1", "balanced_nll"]):
            ax.plot(
                [r["epoch"] for r in history[m]],
                [r[key] for r in history[m]],
                marker="o",
                label=m,
                color=colors[i],
            )
            ax.set(xlabel="Epoch", title=key)
    axes[0].legend(fontsize=8)
    plot(
        fig,
        "selection",
        "Selection histories; checkpoint choice uses these development metrics only.",
    )

    testrows = []
    for m in methods:
        for cohort in [
            "fleurs_test",
            "pacific_test",
            "combined_test",
            "selection",
            "calibration",
        ]:
            for group, r in bench[m][cohort]["joint"].items():
                if group == "all15":
                    testrows.append(
                        [
                            m,
                            cohort,
                            r["n"],
                            pct(r["accuracy"]),
                            pct(r["macro_precision"]),
                            pct(r["macro_recall"]),
                            pct(r["macro_f1"]),
                            num(r["nll"]),
                            num(r["brier"]),
                            pct(r["ece"]),
                        ]
                    )
    section(
        "3. Held-out tests and development cohorts",
        table(
            [
                "Small variant",
                "Cohort",
                "N",
                "Accuracy",
                "Precision",
                "Recall",
                "Macro F1",
                "NLL",
                "Brier",
                "ECE",
            ],
            testrows,
        )
        + "<p>Selection and calibration are development cohorts, not independent test results. All fifteen outputs are enabled in every subgroup. Macro metrics average represented reference classes: twelve on FLEURS, three on Pacific and fifteen on the combined test.</p>",
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(15)
    for i, m in enumerate(methods):
        values = bench[m]["combined_test"]["joint"]["all15"]["per_language"]
        ax.bar(
            x + (i - 0.5) * 0.35,
            [values[c]["f1"] * 100 for c in CODES],
            0.35,
            label=m,
            color=colors[i],
        )
    ax.set(
        xticks=x,
        xticklabels=CODES,
        ylim=(0, 105),
        ylabel="F1 (%)",
        title="Small15 combined test per-language F1",
    )
    ax.legend(fontsize=8)
    plot(
        fig,
        "test_languages",
        "Combined-test per-language F1; the complete confusion matrices appear in the plot appendix.",
    )
    perclass = []
    for m in methods:
        for c, name in zip(CODES, NAMES):
            r = bench[m]["combined_test"]["joint"]["all15"]["per_language"][c]
            perclass.append(
                [
                    m,
                    c,
                    name,
                    r["support"],
                    pct(r["precision"]),
                    pct(r["recall"]),
                    pct(r["f1"]),
                    num(r["nll"]),
                    num(r["brier"]),
                    pct(r["ece"]),
                ]
            )
    pieces.append(
        table(
            [
                "Variant",
                "Code",
                "Language",
                "N",
                "Precision",
                "Recall / accuracy",
                "F1",
                "NLL",
                "Brier",
                "ECE",
            ],
            perclass,
        )
    )

    comparison = []
    for family, values in [("Small", bench), ("Tiny", tiny)]:
        for m in methods:
            for group, r in values[m]["combined_test"]["joint"].items():
                comparison.append(
                    {
                        "model": f"{family}15 {m}",
                        "group": group,
                        **{
                            k: r[k]
                            for k in [
                                "n",
                                "accuracy",
                                "macro_f1",
                                "nll",
                                "brier",
                                "ece",
                            ]
                        },
                    }
                )
    for m in ["frozen", "partial"]:
        values = read(study.pacific / m / "report.json")["groups"]
        for group, variants in values.items():
            r = variants["joint_temperature"]
            comparison.append(
                {
                    "model": f"SpeechBrain15 {m}",
                    "group": group,
                    **{
                        k: r[k]
                        for k in ["n", "accuracy", "macro_f1", "nll", "brier", "ece"]
                    },
                }
            )
    write_csv(root / "family_comparison.csv", comparison)
    section(
        "4. Tiny/SpeechBrain comparisons and original-language changes",
        table(
            ["Model", "Group", "N", "Accuracy", "Macro F1", "NLL", "Brier", "ECE"],
            [
                [
                    r["model"],
                    r["group"],
                    r["n"],
                    pct(r["accuracy"]),
                    pct(r["macro_f1"]),
                    num(r["nll"]),
                    num(r["brier"]),
                    pct(r["ece"]),
                ]
                for r in comparison
            ],
        )
        + "<p>All systems use the same held-out IDs. Backbone pretraining and update policies differ; these are system comparisons rather than a controlled causal test of encoder depth.</p>",
    )
    regression = []
    for m in methods:
        current = bench[m]["fleurs_test"]["old_only"]["old12"]
        regression.append(
            {
                "variant": m,
                "baseline": "frozen Small12 warm-start",
                **{
                    f"delta_{k}": current[k] - warm[k]
                    for k in ["accuracy", "macro_f1", "nll", "brier", "ece"]
                },
            }
        )
    write_csv(root / "warm_start_regression.csv", regression)
    pieces.append(
        table(
            [
                "Small15 variant",
                "Baseline",
                "Δ accuracy (pp)",
                "Δ F1 (pp)",
                "Δ NLL",
                "Δ Brier",
                "Δ ECE (pp)",
            ],
            [
                [
                    r["variant"],
                    r["baseline"],
                    f"{r['delta_accuracy'] * 100:+.2f}",
                    f"{r['delta_macro_f1'] * 100:+.2f}",
                    f"{r['delta_nll']:+.4f}",
                    f"{r['delta_brier']:+.4f}",
                    f"{r['delta_ece'] * 100:+.2f}",
                ]
                for r in regression
            ],
        )
    )
    pieces.append(
        "<p>These comparisons fit temperature on FLEURS validation alone and retain all fifteen extension outputs. Frozen15 versus frozen12 measures extension regression with a fixed encoder. Fine-tuned15 versus frozen12 combines encoder adaptation and vocabulary expansion; no matched fine-tuned Small12 regression is available.</p>"
    )

    calrows = []
    for m in methods:
        for cohort in ["fleurs_test", "pacific_test", "combined_test"]:
            for variant in ["raw", "joint", "old_only"]:
                r = bench[m][cohort][variant]["all15"]
                calrows.append(
                    [
                        m,
                        cohort,
                        variant,
                        num(r["temperature"]),
                        num(r["nll"]),
                        num(r["brier"]),
                        pct(r["ece"]),
                    ]
                )
    section(
        "5. Calibration and confidence",
        table(
            ["Variant", "Test", "Calibration", "Temperature", "NLL", "Brier", "ECE"],
            calrows,
        )
        + "<p>A single temperature can improve pooled confidence while worsening one constituent domain. Calibration never changes top-1 predictions. Raw and joint reliability figures for every test and development cohort are included in the complete appendix.</p>",
    )

    extrows = []
    for model in model_names:
        for policy in ["all", "english_excluded", "shared"]:
            r = external["models"][model]["complete"][policy]
            extrows.append(
                [
                    titles[model],
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
        "6. External recordings, cohort transfer and source balance",
        table(
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
            extrows,
        )
        + """<p>The unchanged v4 snapshot has 24 sources and 1,336 segments, including 1,243 historical and 93 added Niuean segments. Labels are provisional directory labels. English predictions can reflect genuine code-switching.
        English-excluded scores remove each model's own English predictions. Shared-cohort scores use all six extension models listed below; this cohort may differ from the earlier Tiny-only report and is explicitly recomputed for Small comparisons.</p>"""
        + html.escape(", ".join(external["shared_models"])),
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    languages = ["sam", "niu", "ton", "hi", "th"]
    x = np.arange(5)
    for i, model in enumerate(model_names):
        values = external["models"][model]["complete"]["all"]["by_language"]
        ax.bar(
            x + (i - 2.5) * 0.14,
            [values[c]["accuracy"] * 100 for c in languages],
            0.14,
            label=titles[model],
            color=colors[i],
        )
    ax.set(
        xticks=x,
        xticklabels=languages,
        ylim=(0, 105),
        ylabel="All-segment accuracy (%)",
        title="External recognition by language",
    )
    ax.legend(fontsize=7, ncol=3)
    plot(
        fig,
        "external_languages",
        "Language-level results distinguish Pacific transfer from Hindi/Thai recognition. Twelve-language controls are out-of-set diagnostics for Pacific audio.",
    )
    cohortrows = []
    confusionrows = []
    for model in model_names:
        for cohort in ["historical", "added", "hindi_thai"]:
            r = external["models"][model][cohort]["all"]
            cohortrows.append(
                [
                    titles[model],
                    cohort,
                    r["segments_total"],
                    pct(r["accuracy"]),
                    pct(r["mean_source_accuracy"]),
                ]
            )
        r = external["models"][model]["complete"]["all"]
        for a, b in [("niu", "ton"), ("ton", "niu"), ("niu", "mi"), ("ton", "mi")]:
            count = r["confusion_scored"][a].get(b, 0)
            n = r["by_language"][a]["total"]
            confusionrows.append(
                [titles[model], f"{a} → {b}", count, n, pct(count / n)]
            )
    pieces += [
        table(
            ["Model", "Cohort", "N", "Accuracy", "Equal-source accuracy"], cohortrows
        ),
        "<p>Keep the historical and added recording cohorts separate: a changed mixture can change the apparent model ordering. Equal-source accuracy weights each recording equally, whereas pooled accuracy weights each segment equally.</p>",
        table(
            ["Model", "Confusion direction", "Count", "Reference N", "Rate"],
            confusionrows,
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, methods):
        r = external["models"][f"whisper_small15_{m}"]["complete"]["all"]
        matrix = np.array(
            [[r["confusion_scored"][c].get(p, 0) for p in CODES] for c in languages],
            dtype=float,
        )
        matrix /= matrix.sum(1, keepdims=True)
        ax.imshow(matrix * 100, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.set(
            xticks=range(15),
            xticklabels=CODES,
            yticks=range(5),
            yticklabels=languages,
            title=f"Small15 {m}",
            xlabel="Predicted language",
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
                        fontsize=7,
                        color="white" if matrix[j, k] > 0.55 else "black",
                    )
    plot(
        fig,
        "external_confusions",
        "Row-normalized confusion percentages retain all fifteen predicted labels. A reduction in Niuean-to-Tongan errors alone need not imply better Niuean accuracy.",
    )
    source_rows = csv_rows(root / "external/matched_per_source.csv")
    selected = [
        r
        for r in source_rows
        if r["cohort"] == "complete"
        and r["policy"] == "all"
        and r["model"].startswith("whisper_small15")
    ]
    pieces.append(
        table(
            ["Small variant", "Source ID", "Language", "N", "Accuracy"],
            [
                [
                    r["model"],
                    r["source_id"],
                    r["language"],
                    r["total"],
                    pct(r["accuracy"]),
                ]
                for r in selected
            ],
        )
    )

    duration = csv_rows(root / "diagnostics/duration.csv")
    gain = csv_rows(root / "diagnostics/gain.csv")
    diagnostic = read(root / "diagnostics/complete.json")
    missing = sorted(
        {"sam", "niu", "ton", "hi", "th"} - set(diagnostic["duration_languages"])
    )
    section(
        "7. Duration and volume diagnostics",
        """<p>Duration diagnostics reuse exactly the same nested 3/5/10-second windows as Tiny and SpeechBrain. They are a subset of the external corpus, not the full segmentation.
        Missing matched duration languages: <strong>"""
        + html.escape(", ".join(missing) or "none")
        + "</strong>. A language absent here has no qualifying ten-second VAD region; no duration estimate is fabricated.</p>",
    )
    available = [c for c in ["hi", "th", "sam", "niu", "ton"] if c not in missing]
    fig, axes = plt.subplots(
        1, len(available), figsize=(3.2 * len(available), 3.5), sharey=True
    )
    for ax, code in zip(np.atleast_1d(axes), available):
        for i, m in enumerate(methods):
            values = sorted(
                [
                    r
                    for r in duration
                    if r["model"] == f"whisper_small15_{m}" and r["language"] == code
                ],
                key=lambda r: int(r["seconds"]),
            )
            ax.plot(
                [int(r["seconds"]) for r in values],
                [float(r["accuracy"]) * 100 for r in values],
                marker="o",
                color=colors[i],
                label=m,
            )
        ax.set(title=code, xticks=[3, 5, 10], xlabel="Seconds", ylim=(0, 105))
    np.atleast_1d(axes)[0].legend(fontsize=7)
    plot(
        fig,
        "duration",
        "Matched-duration all-window accuracy. Small window counts and correlated sources limit interpretation; longer audio need not improve every recording.",
    )
    sam = [
        r
        for r in gain
        if r["model"].startswith("whisper_small15") and r["language"] == "sam"
    ]
    pieces.append(
        "<p>Volume probes use twelve checkpoint-selection cuts per language, never external audio. Positive gain is capped to avoid clipping. Results are descriptive probes, not a reason to tune preprocessing on the external corpus.</p>"
        + table(
            [
                "Variant",
                "Requested dB",
                "Applied mean dB",
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
                    num(r["mean_applied_gain_db"]),
                    r["n"],
                    r["changed_predictions"],
                    pct(r["accuracy_before"]),
                    pct(r["accuracy_after"]),
                    num(r["mean_absolute_confidence_change"]),
                ]
                for r in sam
            ],
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, m in zip(axes, methods):
        for i, db in enumerate(["-12", "12"]):
            values = {
                r["language"]: r
                for r in gain
                if r["model"] == f"whisper_small15_{m}" and r["gain_db"] == db
            }
            ax.bar(
                np.arange(15) + (i - 0.5) * 0.35,
                [int(values[c]["changed_predictions"]) for c in CODES],
                0.35,
                label=db + " dB requested",
                color=colors[i],
            )
        ax.set(
            xticks=range(15),
            xticklabels=CODES,
            title=m,
            ylabel="Changed predictions / 12 cuts",
        )
        ax.legend(fontsize=8)
    plot(
        fig,
        "gain",
        "Selection-only gain probes; actual positive gain varies across recordings because of the clipping cap.",
    )

    exportrows = []
    export_data = {}
    for m in methods:
        directory = Path(f"artifacts/deployment/whisper_small15_{m}")
        info = read(directory / "export.json")
        verified = read(directory / "verification.json")
        archive = directory.with_suffix(".tar.gz")
        if file_hash(archive) != Path(str(archive) + ".sha256").read_text().split()[0]:
            raise ValueError("Small export archive checksum failed")
        if verified["n"] != 14400 or verified["prediction_disagreements"] != 0:
            raise ValueError("Small export verification is incomplete")
        export_data[m] = {
            "metadata": info,
            "verification": verified,
            "archive_sha256": file_hash(archive),
        }
        exportrows.append(
            [
                m,
                str(archive),
                f"{archive.stat().st_size / 1024**2:.2f} MiB",
                verified["n"],
                verified["prediction_disagreements"],
                f"{verified['max_absolute_log_probability_error']:.2e}",
            ]
        )
    section(
        "8. Deployment, resources and limitations",
        table(
            [
                "Variant",
                "Archive",
                "Size",
                "Verified N",
                "Top-1 differences",
                "Max |Δ log p|",
            ],
            exportrows,
        )
        + """<p>Both architecture-correct decoder-free bundles load through SpeechBrain EncoderClassifier.from_hparams, include fifteen labels and have joint temperature folded once into the classifier.
        Each was verified offline after relocation against every test and external example, plus CPU/file and padded-batch inference. VAD and unknown-language rejection are not bundled.</p><pre>from speechbrain.inference.classifiers import EncoderClassifier\nmodel = EncoderClassifier.from_hparams(\n    source="/path/to/whisper_small15_finetuned",\n    savedir="/tmp/whisper_small15_loaded",\n    run_opts={"device": "cpu"},\n)\nlog_probs, score, index, labels = model.classify_file("audio.wav")</pre>"""
        + """<p>The guard enforces 4 GiB host RAM, zero swap, two CPU threads and a single GPU job below 7 GiB. The encoder-only preflight verifies a full optimizer step and checkpoint save before the long run.
        This is one seed and 24 provisionally labelled external sources. Segment counts do not establish independent speakers or statistical significance. No model is automatically chosen from external performance.</p>"""
        + "<pre>"
        + html.escape(json.dumps(read(root / "preflight.json"), indent=2))
        + "</pre>",
    )

    csv_paths = (
        [root / m / "summary.csv" for m in methods]
        + [root / m / "per_language.csv" for m in methods]
        + [
            root / "family_comparison.csv",
            root / "warm_start_regression.csv",
            root / "external/matched_summary.csv",
            root / "external/matched_per_language.csv",
            root / "external/matched_per_source.csv",
            root / "diagnostics/duration.csv",
            root / "diagnostics/gain.csv",
        ]
    )
    section(
        "9. Complete numerical appendices",
        "<p>Expandable tables retain all variants, controls, sources and diagnostics. Full numerical data are also embedded in this HTML report.</p>",
    )
    for path in csv_paths:
        values = csv_rows(path)
        headers = list(values[0]) if values else []
        pieces.append(
            f"<details><summary>{html.escape(str(path))} — {len(values):,} rows</summary>"
            + table(headers, [[r[k] for k in headers] for r in values])
            + "</details>"
        )
    figures_all = sorted(
        {*root.glob("*/figures/*.png"), *(root / "external/figures").glob("*.png")}
    )
    section(
        "10. Complete plot appendix",
        f"<p>All {len(figures_all)} saved benchmark and external plots are embedded below, including raw, joint and old-only calibration and all available control/cohort/exclusion variants.</p>",
    )
    for path in figures_all:
        pieces.append(portable_figure(path, str(path)))
    css = """body{font:16px/1.6 system-ui,sans-serif;background:#f1f5f6;color:#182c35;margin:0}main{max-width:1120px;margin:auto;padding:36px;background:white}h1{font-size:38px}h2{color:#165968;margin-top:2em}.callout{background:#e7f3f3;padding:20px;border-left:5px solid #087e8b}.table{overflow:auto;margin:18px 0}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:7px;border-bottom:1px solid #d5dfe2;text-align:right;overflow-wrap:anywhere}th{background:#edf3f4}th:first-child,td:first-child{text-align:left}figure{margin:24px 0;break-inside:avoid}img{width:100%;height:auto}figcaption{font-size:12px;overflow-wrap:anywhere;color:#526571}pre{white-space:pre-wrap;font-size:12px;padding:16px;background:#f1f4f5}details{margin:14px 0}summary{cursor:pointer;font-weight:600}@page{size:A4;margin:16mm}@media print{body{font-size:10pt;background:white}main{padding:0}h1{font-size:24pt}h2{font-size:16pt;break-after:avoid}.table{overflow:visible}table{font-size:7pt}tr{break-inside:avoid}figure{width:145mm;margin:4mm auto;max-height:126mm}img{max-height:109mm;object-fit:contain}figcaption{font-size:8pt}details:not([open])>:not(summary){display:none}}"""
    raw = {
        "benchmarks": bench,
        "external": external,
        "history": history,
        "export_verification": export_data,
        "cohorts": metadata,
    }
    import base64

    embedded = base64.b64encode(json.dumps(raw).encode()).decode()
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Whisper Small15 performance analysis</title><style>'
        + css
        + "</style></head><body><main><h1>Whisper Small15 performance analysis</h1><p>Focused seed-42 study · Original recording splits · "
        + datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        + "</p>"
        + "".join(pieces)
        + f'<p><a download="small15_report_data.json" href="data:application/json;base64,{embedded}">Download embedded numerical results</a></p></main></body></html>'
    )
    html_path = output / "whisper_small15_analysis.html"
    html_path.write_text(document)
    pdf_path = output / "whisper_small15_analysis.pdf"
    text = render_pdf(html_path, pdf_path)
    compact = re.sub(r"\s+", "", text)
    if any(re.sub(r"\s+", "", str(path)) not in compact for path in figures_all):
        raise ValueError("Small PDF omitted a plot caption")
    manifest = {
        "html_sha256": file_hash(html_path),
        "pdf_sha256": file_hash(pdf_path),
        "plots_embedded": len(figures_all),
        "analysis_figures": len(plot_paths),
        "input_sha256": {
            str(path): file_hash(path)
            for path in csv_paths
            + [root / m / "benchmarks.json" for m in methods]
            + [root / "external/matched_report.json", *figures_all]
        },
    }
    write_json(output / "report_manifest.json", manifest)
    for path in [html_path, pdf_path]:
        Path(str(path) + ".sha256").write_text(f"{file_hash(path)}  {path.name}\n")
    write_json(
        root / "report_complete.json",
        {
            "html": str(html_path),
            "pdf": str(pdf_path),
            "plots": len(figures_all) + len(plot_paths),
        },
    )
