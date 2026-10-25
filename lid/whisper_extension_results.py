"""Matched benchmarks, calibration, diagnostics and reports for Whisper15."""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from .extension_data import CODES, file_hash, write_json
from .metrics import evaluate, fit_temperature
from .speaker_pilot import bind, digest
from .whisper_extension import ExtensionAdapter
from .whisper_model import Adapter
from .youtube_evaluate import write_csv
from .youtube_metrics import score_segments


def grouped_metrics(logits, rows, codes, temperature):
    """Keep all output columns enabled for every subset of reference labels."""
    targets = np.array([codes.index(r["language"]) for r in rows])
    result = {}
    for group, labels in [
        ("all15", CODES),
        ("old12", CODES[:12]),
        ("new3", CODES[12:]),
    ]:
        mask = np.array([r["language"] in labels for r in rows])
        present = [code for code in labels if any(r["language"] == code for r in rows)]
        if mask.any():
            result[group] = evaluate(
                logits[mask],
                targets[mask],
                present,
                temperature,
                report_indices=[codes.index(c) for c in present],
            )
            for code in present:
                selected = np.array([r["language"] == code for r in rows])
                single = evaluate(
                    logits[selected],
                    targets[selected],
                    [code],
                    temperature,
                    report_indices=[codes.index(code)],
                )
                result[group]["per_language"][code].update(
                    {k: single[k] for k in ["accuracy", "nll", "brier", "ece"]}
                )
    return result


def plots(result, path, codes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for kind in ["reliability", "confusion"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        if kind == "reliability":
            bins = result["reliability"]
            ax.plot([0, 1], [0, 1], "--", color="grey")
            ax.scatter(
                [r["confidence"] for r in bins if r["count"]],
                [r["accuracy"] for r in bins if r["count"]],
            )
            ax.set(xlabel="Confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1))
        else:
            matrix = np.array(result["confusion_matrix"])
            ax.imshow(matrix, aspect="auto")
            ax.set_xticks(range(len(codes)), codes, rotation=90)
            ax.set_yticks(
                range(len(result["per_language"])), list(result["per_language"])
            )
            ax.set(xlabel="Prediction", ylabel="Reference")
        fig.tight_layout()
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(f"{path}_{kind}.{extension}")
        plt.close(fig)


def evaluate_models(study):
    for method in ["frozen", "finetuned"]:
        directory = study.root / method
        if not (directory / "training_complete.json").exists():
            raise ValueError(f"Training incomplete: {method}")
        adapter = getattr(study, "adapter_class", ExtensionAdapter)(
            study.root, directory / "model.pt"
        )
        values = {}
        for cohort in [
            "vox_validation",
            "new_selection",
            "fleurs_validation",
            "new_calibration",
            "fleurs_test",
            "new_test",
            "external",
        ]:
            rows = study.rows(cohort)
            if method == "frozen":
                logits = study.feature_scores(adapter.head, [cohort])
            else:
                logits = study.score_adapter(
                    adapter,
                    rows,
                    directory / f"{cohort}_scores.npz",
                    f"{method}_{cohort}",
                )
            study.save_scores(directory, cohort, logits, CODES)
            values[cohort] = logits
        adapter.release()
        old = values["fleurs_validation"]
        new = values["new_calibration"]
        cal_rows = study.rows("fleurs_validation") + study.rows("new_calibration")
        temperature = fit_temperature(
            np.concatenate([old, new]),
            np.array([CODES.index(r["language"]) for r in cal_rows]),
        )
        old_temperature = fit_temperature(
            old,
            np.array(
                [CODES.index(r["language"]) for r in study.rows("fleurs_validation")]
            ),
        )
        calibration = {
            "temperature": temperature,
            "old_only_temperature": old_temperature,
            "codes": CODES,
            "fit": "joint FLEURS validation + Pacific calibration; old-only FLEURS validation; all 15 outputs enabled",
            "model_sha256": file_hash(directory / "model.pt"),
            "calibration_ids_sha256": digest([r["id"] for r in cal_rows]),
            "calibration_scores": {
                c: file_hash(directory / f"{c}.npz")
                for c in ["fleurs_validation", "new_calibration"]
            },
        }
        bind(directory / "calibration.json", calibration)
        cohorts = {
            "fleurs_test": ["fleurs_test"],
            "pacific_test": ["new_test"],
            "combined_test": ["fleurs_test", "new_test"],
            "selection": ["vox_validation", "new_selection"],
            "calibration": ["fleurs_validation", "new_calibration"],
        }
        reports, summary, per_class = {}, [], []
        for name, parts in cohorts.items():
            rows = [r for c in parts for r in study.rows(c)]
            logits = np.concatenate([values[c] for c in parts])
            reports[name] = {}
            for variant, t in [
                ("raw", 1.0),
                ("joint", temperature),
                ("old_only", old_temperature),
            ]:
                metrics = grouped_metrics(logits, rows, CODES, t)
                reports[name][variant] = metrics
                for group, result in metrics.items():
                    summary.append(
                        {
                            "method": method,
                            "cohort": name,
                            "variant": variant,
                            "group": group,
                            **{
                                k: result[k]
                                for k in [
                                    "n",
                                    "accuracy",
                                    "macro_precision",
                                    "macro_recall",
                                    "macro_f1",
                                    "nll",
                                    "brier",
                                    "ece",
                                ]
                            },
                        }
                    )
                    per_class.extend(
                        {
                            "method": method,
                            "cohort": name,
                            "variant": variant,
                            "group": group,
                            "language": c,
                            **v,
                        }
                        for c, v in result["per_language"].items()
                    )
                plots(
                    metrics["all15"], directory / "figures" / f"{name}_{variant}", CODES
                )
            # Explicit IDs and probabilities for all test/selection/calibration examples.
            probs = softmax(logits.astype(np.float64) / temperature, axis=1)
            write_csv(
                directory / f"{name}_predictions.csv",
                [
                    {
                        "id": r["id"],
                        "reference": r["language"],
                        "prediction": CODES[int(logits[i].argmax())],
                        "probabilities": json.dumps(probs[i].tolist()),
                    }
                    for i, r in enumerate(rows)
                ],
            )
        if (
            len(study.rows("fleurs_test")) != 8519
            or len(study.rows("new_test")) != 4545
        ):
            raise ValueError("Test counts changed")
        write_json(directory / "benchmarks.json", reports)
        write_csv(directory / "summary.csv", summary)
        write_csv(directory / "per_language.csv", per_class)
        write_json(
            directory / "evaluation_complete.json",
            {
                "fleurs_test": 8519,
                "pacific_test": 4545,
                "combined_test": 13064,
                "calibration": calibration,
                "selection_is_not_test": True,
            },
        )


def baselines(study):
    """Rerun unchanged checkpoints on the v4 segments, retaining fitted temperatures."""
    from .youtube_evaluate import infer

    output = Path(study.config["external"])
    rows = study.rows("external")
    context = {
        "rows": digest(rows),
        "snapshot": file_hash(output / "segments_ready.json"),
    }
    sb_files = [
        Path("artifacts/full_run/head/head.pt"),
        Path("artifacts/full_run/comparison/calibration.json"),
    ]
    sb_files += [
        study.pacific / method / filename
        for method in ["frozen", "partial"]
        for filename in ["model.pt", "calibration.json"]
    ]
    bind(
        output / "speechbrain_checkpoint_identity.json",
        {str(p): file_hash(p) for p in sb_files},
    )
    bind(output / "baseline_context.json", context)
    if not (output / "speechbrain_baselines_complete.json").exists():
        infer(rows, output)
        write_json(output / "speechbrain_baselines_complete.json", context)
    for method in [
        "whisper_native",
        "whisper_filtered",
        "whisper_frozen",
        "whisper_finetuned",
        "speechbrain_finetuned",
    ]:
        directory = study.root / "baselines" / method
        directory.mkdir(parents=True, exist_ok=True)
        source = study.base / method
        calibration = json.loads((source / "calibration.json").read_text())
        checkpoint = (
            source / "model.pt"
            if (source / "model.pt").exists()
            else study.base / "pretrained/tiny.pt"
        )
        identity = {
            **context,
            "checkpoint": file_hash(checkpoint),
            "calibration": file_hash(source / "calibration.json"),
        }
        bind(directory / "identity.json", identity)
        path = directory / "external.npz"
        if not path.exists():
            if method in ["whisper_native", "whisper_filtered"]:
                # Native and filtered share one decoder pass on precisely the same crops.
                from .whisper_model import (
                    load_whisper,
                    mel_batch,
                    native_logits,
                    selected_languages,
                )

                model = load_whisper(study.root)
                scores = []
                with torch.inference_mode():
                    for start in range(0, len(rows), 2):
                        mels, _ = mel_batch(rows[start : start + 2])
                        logits, codes = native_logits(model, model.encoder(mels))
                        if method == "whisper_filtered":
                            logits = logits[:, selected_languages(codes)]
                        scores.append(logits.cpu().numpy())
                        if start % 64 == 0:
                            study.status(
                                "baseline",
                                model=method,
                                completed=start,
                                total=len(rows),
                            )
                if method == "whisper_filtered":
                    codes = CODES[:12]
                logits = np.concatenate(scores)
                del model
            else:
                family = "whisper" if method.startswith("whisper") else "speechbrain"
                adapter = Adapter(family, study.base, source / "model.pt")
                logits = study.score_adapter(
                    adapter, rows, directory / "scores.npz", method
                )
                adapter.release()
                codes = CODES[:12]
            if not np.isfinite(logits).all():
                raise ValueError("Nonfinite baseline logits")
            np.savez_compressed(
                path,
                ids=[r["id"] for r in rows],
                logits=logits,
                codes=codes,
                identity=digest(identity),
            )
            gc.collect()
            torch.cuda.empty_cache()
        else:
            with np.load(path) as saved:
                if saved["identity"].item() != digest(identity) or saved[
                    "ids"
                ].tolist() != [r["id"] for r in rows]:
                    raise ValueError("Stale baseline cache")
        bind(directory / "calibration.json", calibration)
    write_json(study.root / "baselines_complete.json", context)


def external_results(study):
    output = Path(study.config["external"])
    rows = study.rows("external")
    inventory = json.loads((study.root / "external_inventory.json").read_text())
    historical = set(inventory["historical_segment_ids"])
    provenance = json.loads((output / "model_provenance.json").read_text())
    models = {}
    prefix = getattr(study, "extension_prefix", "whisper15")
    with np.load(output / "logits.npz") as saved:
        if saved["ids"].tolist() != [r["id"] for r in rows]:
            raise ValueError("SpeechBrain segment IDs differ")
        for name, meta in provenance.items():
            models[f"speechbrain_{name}"] = (
                saved[name],
                meta["codes"],
                meta["temperature"],
            )
    for method in [
        "whisper_native",
        "whisper_filtered",
        "whisper_frozen",
        "whisper_finetuned",
        "speechbrain_finetuned",
        "frozen",
        "finetuned",
    ]:
        directory = (
            study.root / method
            if method in ["frozen", "finetuned"]
            else study.root / "baselines" / method
        )
        calibration = json.loads((directory / "calibration.json").read_text())
        with np.load(directory / "external.npz") as saved:
            if saved["ids"].tolist() != [r["id"] for r in rows]:
                raise ValueError("Compared model segment IDs differ")
            models[
                f"{prefix}_{method}" if method in ["frozen", "finetuned"] else method
            ] = (saved["logits"], saved["codes"].tolist(), calibration["temperature"])
    if hasattr(study, "additional_external_models"):
        models.update(study.additional_external_models())
    predictions = {
        m: np.array([codes[i] for i in logits.argmax(1)])
        for m, (logits, codes, _) in models.items()
    }
    extension_names = [m for m in models if "15" in m]
    shared = np.logical_and.reduce([predictions[m] != "en" for m in extension_names])
    reports, summary, per_source, per_language, predictions_table, queue = (
        {},
        [],
        [],
        [],
        [],
        [],
    )
    for name, (logits, codes, temperature) in models.items():
        confidence = softmax(logits.astype(np.float64) / temperature, axis=1).max(1)
        reports[name] = {}
        for cohort, mask in [
            ("complete", np.ones(len(rows), dtype=bool)),
            ("historical", np.array([r["id"] in historical for r in rows])),
            ("added", np.array([r["id"] not in historical for r in rows])),
            ("hindi_thai", np.array([r["language"] in ["hi", "th"] for r in rows])),
        ]:
            ix = np.flatnonzero(mask)
            if not len(ix):
                continue
            selected = [rows[i] for i in ix]
            reports[name][cohort] = {}
            for policy in ["all", "english_excluded", "shared"]:
                result = score_segments(
                    selected,
                    predictions[name][ix],
                    confidence[ix],
                    shared[ix] if policy == "shared" else None,
                    exclude_english=policy != "all",
                )
                # Twelve-way Pacific outputs are diagnostics, not recognition metrics.
                result["out_of_set_languages"] = sorted(
                    {r["language"] for r in selected} - set(codes)
                )
                if (
                    all(r["language"] in codes for r in selected)
                    and result["segments_scored"]
                ):
                    valid = (
                        np.ones(len(ix), dtype=bool)
                        if policy == "all"
                        else predictions[name][ix] != "en"
                    )
                    if policy == "shared":
                        valid &= shared[ix]
                    details = grouped_metrics(
                        logits[ix][valid],
                        [r for r, keep in zip(selected, valid) if keep],
                        codes,
                        temperature,
                    )
                    result["probability_metrics"] = details
                    result["raw_probability_metrics"] = grouped_metrics(
                        logits[ix][valid],
                        [r for r, keep in zip(selected, valid) if keep],
                        codes,
                        1.0,
                    )
                    plots(
                        details["all15"],
                        output / "figures" / f"{name}_{cohort}_{policy}",
                        codes,
                    )
                reports[name][cohort][policy] = result
                summary.append(
                    {
                        "model": name,
                        "cohort": cohort,
                        "policy": policy,
                        **{
                            k: result[k]
                            for k in [
                                "segments_total",
                                "segments_scored",
                                "coverage",
                                "duration_coverage",
                                "english_predictions",
                                "accuracy",
                                "macro_precision",
                                "macro_recall",
                                "macro_f1",
                                "ece",
                                "mean_source_accuracy",
                            ]
                        },
                        "pacific_recognition_supported": all(
                            c in codes for c in CODES[12:]
                        ),
                    }
                )
                per_source.extend(
                    {
                        "model": name,
                        "cohort": cohort,
                        "policy": policy,
                        "source_id": s,
                        **v,
                    }
                    for s, v in result["by_source"].items()
                )
                per_language.extend(
                    {
                        "model": name,
                        "cohort": cohort,
                        "policy": policy,
                        "language": c,
                        **v,
                        **result["per_language"].get(c, {}),
                    }
                    for c, v in result["by_language"].items()
                )
        for i, row in enumerate(rows):
            item = {
                "model": name,
                "segment_id": row["id"],
                "source": row["source"],
                "source_id": row["source_id"],
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "reference": row["language"],
                "prediction": predictions[name][i],
                "confidence": float(confidence[i]),
                "historical": row["id"] in historical,
                "probabilities": json.dumps(
                    dict(
                        zip(
                            codes,
                            softmax(logits[i].astype(float) / temperature).tolist(),
                        )
                    )
                ),
            }
            predictions_table.append(item)
            if predictions[name][i] != row["language"]:
                queue.append(
                    {
                        **{k: v for k, v in item.items() if k != "probabilities"},
                        "review_reason": "English/code-switch"
                        if predictions[name][i] == "en"
                        else "folder-label mismatch",
                        "human_label": "",
                        "notes": "",
                    }
                )
    write_json(
        output / "matched_report.json",
        {
            "models": reports,
            "inventory": inventory,
            "shared_models": extension_names,
            "labels": "provisional directory labels; English/code switches unannotated",
            "external_evaluation_only": True,
        },
    )
    write_csv(output / "matched_summary.csv", summary)
    write_csv(output / "matched_per_source.csv", per_source)
    write_csv(output / "matched_per_language.csv", per_language)
    write_csv(output / "matched_predictions.csv", predictions_table)
    write_csv(output / "matched_review_queue.csv", queue)
    return summary


def diagnostics(study):
    # The established nested-duration protocol is reused, with all 15 selection languages.
    from . import whisper_diagnostics as original
    from .whisper_extension_diagnostics import run

    run(study, original)


def report(study):
    summary = external_results(study)
    directory = Path("reports/whisper_tiny15")
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Whisper tiny 15-language extension",
        "",
        "One paired seed-42 experiment. Original recording-based splits; speaker independence is not established.",
        "Both variants start from the selected frozen Whisper12 classifier and pretrained tiny audio encoder. Hidden layers and old output rows are copied exactly; three seeded rows are appended.",
        "Whisper fine-tuning updates the complete audio encoder. SpeechBrain partial15 updates its encoder tail.",
        "Selection uses combined VoxLingua/Pacific macro F1, breaking ties by class-balanced NLL. Calibration uses FLEURS validation and Pacific calibration only. External clips never affect fitting, selection or preprocessing.",
        "",
        "| Model | Test cohort | Calibration | Group | N | Accuracy | F1 | NLL | Brier | ECE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ["frozen", "finetuned"]:
        import csv

        with (study.root / method / "summary.csv").open() as stream:
            for row in csv.DictReader(stream):
                if row["cohort"].endswith("test") and row["variant"] in [
                    "joint",
                    "old_only",
                ]:
                    lines.append(
                        "| "
                        + " | ".join(
                            str(row[k])
                            for k in [
                                "method",
                                "cohort",
                                "variant",
                                "group",
                                "n",
                                "accuracy",
                                "macro_f1",
                                "nll",
                                "brier",
                                "ece",
                            ]
                        )
                        + " |"
                    )
    regression, class_regression = [], []
    for method, old in [
        ("frozen", "whisper_frozen"),
        ("finetuned", "whisper_finetuned"),
    ]:
        new = json.loads((study.root / method / "benchmarks.json").read_text())[
            "fleurs_test"
        ]["old_only"]["old12"]
        previous = json.loads((study.base / old / "fleurs_report.json").read_text())[
            "metrics"
        ]["calibrated"]
        regression.append(
            {
                "method": method,
                **{
                    f"delta_{k}": new[k] - previous[k]
                    for k in ["accuracy", "macro_f1", "nll", "brier", "ece"]
                },
            }
        )
        from .languages import NAMES as OLD_NAMES

        for code, old_name in zip(CODES[:12], OLD_NAMES):
            current_class = new["per_language"][code]
            previous_class = previous["per_language"][old_name]
            class_regression.append(
                {
                    "method": method,
                    "language": code,
                    **{
                        f"delta_{k}": current_class[k] - previous_class[k]
                        for k in ["precision", "recall", "f1"]
                    },
                }
            )
    write_csv(study.root / "original_language_regression.csv", regression)
    write_csv(
        study.root / "original_language_per_class_regression.csv", class_regression
    )
    comparisons = []
    for method in ["frozen", "partial"]:
        archived = json.loads((study.pacific / method / "report.json").read_text())
        for group, variants in archived["groups"].items():
            for variant, result in variants.items():
                comparisons.append(
                    {
                        "model": f"speechbrain15_{method}",
                        "group": group,
                        "calibration": variant,
                        **{
                            k: result[k]
                            for k in [
                                "n",
                                "accuracy",
                                "macro_precision",
                                "macro_recall",
                                "macro_f1",
                                "nll",
                                "brier",
                                "ece",
                            ]
                        },
                    }
                )
    for method in ["frozen", "finetuned"]:
        benchmarks = json.loads((study.root / method / "benchmarks.json").read_text())
        for variant in ["raw", "joint", "old_only"]:
            for group, result in benchmarks["combined_test"][variant].items():
                comparisons.append(
                    {
                        "model": f"whisper15_{method}",
                        "group": group,
                        "calibration": variant,
                        **{
                            k: result[k]
                            for k in [
                                "n",
                                "accuracy",
                                "macro_precision",
                                "macro_recall",
                                "macro_f1",
                                "nll",
                                "brier",
                                "ece",
                            ]
                        },
                    }
                )
    write_csv(study.root / "speechbrain_comparison.csv", comparisons)
    lines += [
        "",
        "## SpeechBrain15 test comparison",
        "",
        "The archived SpeechBrain test evaluations use the same original FLEURS and recording-held-out Pacific manifests. Both families retain all 15 outputs.",
        "",
        "| Model | Group | Calibration | N | Accuracy | F1 | NLL | Brier | ECE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| "
            + " | ".join(
                str(row[k])
                for k in [
                    "model",
                    "group",
                    "calibration",
                    "n",
                    "accuracy",
                    "macro_f1",
                    "nll",
                    "brier",
                    "ece",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Original-language regression",
        "",
        "All 15 outputs remain enabled. Matched old-only temperature calibration is used for comparisons with Whisper12.",
        "",
        "```json",
        json.dumps(regression, indent=2),
        "```",
        "",
        "## Updated external validation",
        "",
        "Directory labels remain provisional. Domain differences and English code-switching limit interpretation. Twelve-language Pacific predictions are out-of-set diagnostics; Hindi/Thai provide matched recognition comparisons. No winner is selected using external validation.",
        "",
        "| Model | Cohort | Policy | N scored | Coverage | Accuracy | F1 | Equal-source accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                str(row[k])
                for k in [
                    "model",
                    "cohort",
                    "policy",
                    "segments_scored",
                    "coverage",
                    "accuracy",
                    "macro_f1",
                    "mean_source_accuracy",
                ]
            )
            + " |"
        )
    with (study.root / "diagnostics/gain.csv").open() as stream:
        samoan = [r for r in csv.DictReader(stream) if r["language"] == "sam"]
    with (study.root / "diagnostics/duration.csv").open() as stream:
        pacific = [
            r for r in csv.DictReader(stream) if r["language"] in ["sam", "niu", "ton"]
        ]
    lines += [
        "",
        "## Samoan volume sensitivity",
        "",
        "These probes use checkpoint-selection audio only. Positive gain is capped to avoid clipping; applied gain is recorded in the CSV. They do not alter preprocessing or model selection.",
        "",
        "| Model | Requested gain dB | Applied mean dB | N | Changed predictions | Accuracy before | Accuracy after |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in samoan:
        lines.append(
            "| "
            + " | ".join(
                str(row[k])
                for k in [
                    "model",
                    "gain_db",
                    "mean_applied_gain_db",
                    "n",
                    "changed_predictions",
                    "accuracy_before",
                    "accuracy_after",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Matched Pacific duration diagnostics",
        "",
        "Nested 3/5/10-second excerpts share the same ten-second speech windows. Missing accuracy denotes an out-of-set twelve-language control. Directional confusion rates retain English and other errors in the denominator.",
        "",
        "| Model | Language | Seconds | N | Accuracy | English | Niuean to Tongan | Tongan to Niuean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pacific:
        lines.append(
            "| "
            + " | ".join(
                str(row[k])
                for k in [
                    "model",
                    "language",
                    "seconds",
                    "n",
                    "accuracy",
                    "english",
                    "niu_to_ton_rate",
                    "ton_to_niu_rate",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Detailed predictions, per-language/source tables, confusion matrices, reliability plots (PNG/PDF/SVG), calibration provenance, histories, gain/duration diagnostics and resource measurements are saved under `artifacts/whisper_tiny15` and `artifacts/youtube_validation_v4`.",
    ]
    (directory / "whisper_extension.md").write_text("\n".join(lines) + "\n")
    external_dir = Path("reports/youtube_validation_v4")
    external_dir.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((study.root / "external_inventory.json").read_text())
    external_text = (
        "# Refreshed matched validation\n\nAll models use identical segment IDs and crop boundaries. Historical segments are checked against v3 by content hash and timestamps.\n\nAdded sources:\n\n"
        + "\n".join(
            f"- {s['language']}: {Path(s['source']).name} ({s['sha256']})"
            for s in inventory["added_sources"]
        )
    )
    external_text += (
        "\n\n"
        + "\n".join(lines[lines.index("## Updated external validation") + 1 :])
        + "\n"
    )
    (external_dir / "held_out_evaluation.md").write_text(external_text)
    write_json(
        study.root / "report_complete.json",
        {
            "report": str(directory / "whisper_extension.md"),
            "external": str(external_dir / "held_out_evaluation.md"),
        },
    )
