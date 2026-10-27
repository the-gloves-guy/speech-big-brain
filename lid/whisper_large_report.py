"""Standalone analysis of Large v3 frozen versus the three requested comparators."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .extension_data import file_hash, write_json
from .whisper_large_results import csv_rows, embedded_data, figure, render_pdf, safe
from .whisper_small_report import num, pct, table
from .youtube_evaluate import write_csv


def read(path):
    return json.loads(path.read_text())


def build(study):
    root = study.root
    output = Path("reports") / root.name
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = list(study.comparison_models)
    titles = ["Large v3 frozen", "SpeechBrain partial", "Tiny frozen", "Small frozen"]
    bench = read(root / "frozen/benchmarks.json")
    tests = {names[0]: bench["combined_test"]["joint"]}
    tests[names[1]] = {
        g: v["joint_temperature"]
        for g, v in read(study.pacific / "partial/report.json")["groups"].items()
    }
    for name, folder in [
        (names[2], study.reference),
        (names[3], study.small_reference),
    ]:
        tests[name] = read(folder / "frozen/benchmarks.json")["combined_test"]["joint"]
    external = read(root / "external/matched_report.json")
    pieces, analysis_figures = [], []

    def section(title, body):
        pieces.append(f"<h2>{safe(title)}</h2>{body}")

    def plot(fig, name, caption):
        fig.tight_layout()
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(figures / f"{name}.{extension}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        path = figures / f"{name}.png"
        analysis_figures.append(path)
        pieces.append(figure(path) + f"<p>{safe(caption)}</p>")

    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "nll",
        "brier",
        "ece",
    ]
    section(
        "Main findings",
        f"<p>Large v3 frozen combined-test accuracy: <strong>{pct(tests[names[0]]['all15']['accuracy'])}</strong>. Complete external accuracy: <strong>{pct(external['models'][names[0]]['complete']['all']['accuracy'])}</strong>. These measure different domains. No external result was used to fit or select this model.</p>",
    )
    section(
        "Protocol and provenance",
        "<p>Official multilingual Large v3 audio encoder, frozen throughout. A twelve-language classifier warm-start is expanded to fifteen outputs, copying hidden layers and original rows and initialising sam, niu and ton with seed 42. The original recording-based splits and previous balanced extension sample orders are retained. No speaker separation is claimed.</p><p>Encoder computation is float32 with TF32 disabled. Audio uses mono 16 kHz, a maximum ten-second centre crop, 128 mel channels, native thirty-second padding and mean/population-standard-deviation pooling over real tokens. The classifier architecture and training schedule match the Small study. Selection maximises combined development-set macro F1, breaking ties with class-balanced NLL. Joint temperature uses only FLEURS validation and Pacific calibration.</p><p>SpeechBrain partial updates its encoder tail; all three Whisper comparators here freeze their complete encoder. Pretraining data and architectures differ, so this comparison does not isolate model size as a causal factor. Larger pretraining does not guarantee better language identification.</p>"
        + table(
            ["Cohort", "N"],
            [[c, m["n"]] for c, m in read(root / "data_ready.json")["cohorts"].items()],
        ),
    )
    rows = []
    for name in names:
        for group, result in tests[name].items():
            rows.append(
                {
                    "model": name,
                    "group": group,
                    **{k: result[k] for k in ["n", *metrics]},
                }
            )
    write_csv(root / "family_comparison.csv", rows)
    section(
        "Held-out comparisons",
        table(
            ["Model", "Group", "N", *metrics],
            [
                [r["model"], r["group"], r["n"], *[num(r[k]) for k in metrics]]
                for r in rows
            ],
        ),
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, name in enumerate(names):
        ax.bar(
            np.arange(3) + (i - 1.5) * 0.2,
            [tests[name][g]["accuracy"] * 100 for g in ["old12", "new3", "all15"]],
            0.2,
            label=titles[i],
        )
    ax.set(
        xticks=range(3),
        xticklabels=["Original 12", "Pacific 3", "All 15"],
        ylabel="Accuracy (%)",
        ylim=(0, 105),
    )
    ax.legend(fontsize=8)
    plot(
        fig,
        "held_out_comparison",
        "All fifteen outputs remain enabled for every reference-language subset.",
    )
    development = []
    for cohort, variants in bench.items():
        for variant, groups in variants.items():
            r = groups["all15"]
            development.append([cohort, variant, r["n"], *[num(r[k]) for k in metrics]])
    section(
        "Large v3 tests, selection and calibration",
        table(["Cohort", "Calibration", "N", *metrics], development)
        + "<p>Selection and calibration are development cohorts and are reported separately from test results. Raw, joint and old-only temperatures retain all fifteen output classes.</p>",
    )
    history = read(root / "frozen/history.json")
    warm_history_path = study.base / "whisper_frozen/history.json"
    warm_history = read(warm_history_path) if warm_history_path.exists() else []
    if warm_history:
        section(
            "Twelve-language warm-start history",
            table(
                list(warm_history[0]),
                [[row[k] for k in warm_history[0]] for row in warm_history],
            ),
        )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ["macro_f1", "balanced_nll"]):
        ax.plot([r["epoch"] for r in history], [r[key] for r in history], marker="o")
        ax.set(xlabel="Epoch", title=key)
    plot(
        fig,
        "training",
        "Checkpoint-selection history; external clips play no part in selection.",
    )
    warm = read(study.base / "whisper_frozen/fleurs_report.json")["metrics"][
        "calibrated"
    ]
    current = bench["fleurs_test"]["old_only"]["old12"]
    regression = {
        "baseline": "Large v3 twelve-language frozen warm-start",
        **{
            f"delta_{k}": current[k] - warm[k]
            for k in ["accuracy", "macro_f1", "nll", "brier", "ece"]
        },
    }
    write_csv(root / "warm_start_regression.csv", [regression])
    section(
        "Original-language regression",
        table(
            list(regression),
            [[num(v) if isinstance(v, float) else v for v in regression.values()]],
        )
        + "<p>Both temperatures in this comparison use FLEURS validation only. All fifteen extension outputs remain enabled. No regression cutoff was imposed.</p>",
    )
    section(
        "External validation and exclusion coverage",
        "<p>External folder labels remain provisional and may include English or code-switching. Shared-cohort results use one common English-exclusion mask across exactly the four compared models. Equal-source accuracy weights recordings equally. Historical and added Niuean recordings are reported separately.</p>",
    )
    erows = []
    for name in names:
        for cohort, policies in external["models"][name].items():
            for policy, r in policies.items():
                erows.append(
                    [
                        name,
                        cohort,
                        policy,
                        r["segments_scored"],
                        pct(r["coverage"]),
                        pct(r["accuracy"]),
                        pct(r["macro_f1"]),
                        pct(r["mean_source_accuracy"]),
                    ]
                )
    pieces.append(
        table(
            [
                "Model",
                "Cohort",
                "Policy",
                "N",
                "Coverage",
                "Accuracy",
                "F1",
                "Equal-source accuracy",
            ],
            erows,
        )
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    langs = ["sam", "niu", "ton", "hi", "th"]
    for i, name in enumerate(names):
        values = external["models"][name]["complete"]["all"]["per_language"]
        ax.bar(
            np.arange(5) + (i - 1.5) * 0.2,
            [values[c]["recall"] * 100 for c in langs],
            0.2,
            label=titles[i],
        )
    ax.set(xticks=range(5), xticklabels=langs, ylabel="Accuracy (%)", ylim=(0, 105))
    ax.legend(fontsize=8)
    plot(
        fig,
        "external_languages",
        "Complete external corpus; English predictions count as errors against the provisional language labels.",
    )
    predictions = csv_rows(root / "external/matched_predictions.csv")
    confusion = []
    for name in names:
        for origin, target in [("niu", "ton"), ("ton", "niu")]:
            subset = [
                r
                for r in predictions
                if r["model"] == name and r["reference"] == origin
            ]
            if subset:
                count = sum(r["prediction"] == target for r in subset)
                confusion.append(
                    [
                        name,
                        origin + " → " + target,
                        count,
                        len(subset),
                        pct(count / len(subset)),
                    ]
                )
    section(
        "Directional confusions and per-source results",
        table(["Model", "Direction", "Count", "Reference N", "Rate"], confusion),
    )
    sources = csv_rows(root / "external/matched_per_source.csv")
    pieces.append(
        table(list(sources[0]), [[r[k] for k in sources[0]] for r in sources])
    )
    diagnostic = read(root / "diagnostics/complete.json")
    section(
        "Duration and selection-only gain probes",
        f"<p>All models use identical diagnostic windows and gain inputs. Matched duration languages: {safe(diagnostic['duration_languages'])}. No qualifying matched Niuean duration windows are available. Gain probes are restricted to checkpoint-selection cuts; positive gain is capped to avoid clipping. No external preprocessing tuning was performed.</p>",
    )
    for name, xkey, title in [
        ("duration", "seconds", "Matched duration accuracy"),
        ("gain", "gain_db", "Selection-only gain accuracy"),
    ]:
        values = csv_rows(root / "diagnostics" / f"{name}.csv")
        fig, axes = plt.subplots(
            1, 4 if name == "duration" else 3, figsize=(13, 4), sharey=True
        )
        codes = (
            diagnostic["duration_languages"]
            if name == "duration"
            else ["sam", "niu", "ton"]
        )
        for ax, code in zip(np.atleast_1d(axes), codes):
            for i, model in enumerate(names):
                subset = sorted(
                    [
                        r
                        for r in values
                        if r["model"] == model and r["language"] == code
                    ],
                    key=lambda r: float(r[xkey]),
                )
                ax.plot(
                    [float(r[xkey]) for r in subset],
                    [
                        float(
                            r["accuracy"] if name == "duration" else r["accuracy_after"]
                        )
                        * 100
                        for r in subset
                    ],
                    marker="o",
                    label=titles[i],
                )
            ax.set(title=code, xlabel=xkey, ylim=(0, 105))
        np.atleast_1d(axes)[0].legend(fontsize=6)
        plot(fig, name, title)
        if name == "gain":
            values = [r for r in values if r["language"] == "sam"]
        pieces.append(
            table(list(values[0]), [[r[k] for k in values[0]] for r in values])
        )
    bundle = Path("artifacts/deployment/whisper_large_v3_15_frozen")
    verified = read(bundle / "verification.json")
    archive = bundle.with_suffix(".tar.gz")
    if (
        verified["n"] != 13064 + len(study.rows("external"))
        or verified["prediction_disagreements"]
        or file_hash(archive) != Path(str(archive) + ".sha256").read_text().split()[0]
    ):
        raise ValueError("Export verification or checksum is incomplete")
    resources = {p.name: read(p) for p in root.glob("resources_*.json")}
    section(
        "Deployment, resources and limitations",
        "<p>The decoder-free export loads offline using SpeechBrain EncoderClassifier.from_hparams. Joint calibration is folded once into the classifier. Relocated verification covers every test and external example, plus CPU/file and variable-length batch inference. Supported labels are bundled. VAD and unknown-language rejection are not included.</p><p>One seed and a small number of correlated external recordings limit conclusions. Recording-held-out partitions do not establish speaker independence. All runs enforce 4 GiB RAM, zero swap, two CPU threads and one GPU job below 7 GiB.</p><pre>"
        + safe(
            json.dumps(
                {
                    "verification": verified,
                    "preflight": read(root / "preflight.json"),
                    "resources": resources,
                },
                indent=2,
            )
        )
        + "</pre>",
    )
    paths = [
        root / "frozen/summary.csv",
        root / "frozen/per_language.csv",
        root / "family_comparison.csv",
        root / "warm_start_regression.csv",
        root / "external/matched_summary.csv",
        root / "external/matched_per_language.csv",
        root / "external/matched_per_source.csv",
        root / "diagnostics/duration.csv",
        root / "diagnostics/gain.csv",
    ]
    section(
        "Complete numerical appendices",
        "<p>Full per-class probability metrics, calibration variants, sources and diagnostics follow.</p>",
    )
    for path in paths:
        values = csv_rows(path)
        pieces.append(
            f"<details><summary>{safe(path)}</summary>"
            + table(list(values[0]), [[r[k] for k in values[0]] for r in values])
            + "</details>"
        )
    all_figures = sorted(
        {
            *root.glob("*/figures/*.png"),
            *[
                p
                for folder in [study.reference, study.small_reference]
                for p in (folder / "frozen/figures").glob("*.png")
            ],
        }
    )
    section(
        "Complete plot appendix",
        f"<p>All {len(all_figures)} saved benchmark and matched external figures are embedded, including confusion matrices and reliability diagrams. New figures are also exported as PNG, PDF and SVG.</p>",
    )
    pieces.extend(figure(p) for p in all_figures)
    raw = {
        "warm_start_history": warm_history,
        "tests": tests,
        "large_benchmarks": bench,
        "external": external,
        "history": history,
        "resources": resources,
    }
    css = "body{font:15px/1.5 sans-serif;color:#18323c;max-width:1200px;margin:32px auto}h2{color:#136776;margin-top:2em}table{border-collapse:collapse;font-size:11px}td,th{padding:5px;border:1px solid #ddd}th{background:#eef5f5}.table{overflow:auto}img{max-width:100%}figure{break-inside:avoid}pre{white-space:pre-wrap}summary{font-weight:bold}@page{size:A4;margin:14mm}@media print{body{font-size:10pt}.table{overflow:visible}table{font-size:7pt;overflow-wrap:anywhere}figure{max-width:150mm}img{max-height:112mm;object-fit:contain}details:not([open])>:not(summary){display:none}}"
    html_path = output / "whisper_large_v3_15_analysis.html"
    html_path.write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>Whisper Large v3 performance analysis</title><style>{css}</style></head><body><h1>Whisper Large v3 performance analysis</h1><p>{datetime.now(UTC).isoformat()} · Seed 42 · Original recording splits</p>'
        + "".join(pieces)
        + f'<p><a download="large_v3_data.json" href="data:application/json;base64,{embedded_data(raw)}">Download numerical results</a></p></body></html>'
    )
    pdf = html_path.with_suffix(".pdf")
    text = render_pdf(html_path, pdf)
    compact = re.sub(r"\s+", "", text)
    if any(re.sub(r"\s+", "", str(p)) not in compact for p in all_figures):
        raise ValueError("PDF omitted a figure caption")
    manifest = {
        "html_sha256": file_hash(html_path),
        "pdf_sha256": file_hash(pdf),
        "plots_embedded": len(all_figures),
        "analysis_figures": len(analysis_figures),
        "input_sha256": {str(p): file_hash(p) for p in paths + all_figures},
    }
    write_json(output / "report_manifest.json", manifest)
    for path in [html_path, pdf]:
        Path(str(path) + ".sha256").write_text(f"{file_hash(path)}  {path.name}\n")
    write_json(
        root / "report_complete.json",
        {
            "html": str(html_path),
            "pdf": str(pdf),
            "plots": len(all_figures) + len(analysis_figures),
        },
    )
