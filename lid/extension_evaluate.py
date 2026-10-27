"""Full-vocabulary regression evaluation, calibration, and SpeechBrain export."""

import csv
import gc
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from .data import waveform
from .extension_data import (
    BASE,
    CODES,
    NAMES,
    REVISION,
    RUN,
    file_hash,
    read_rows,
    status,
    write_json,
)
from .extension_model import (
    cache_embeddings,
    head_scores,
    load_initial,
    old_cache,
    state_fingerprint,
)
from .fleurs import REVISION as FLEURS_REVISION
from .fleurs import records
from .metrics import evaluate, fit_temperature
from .model import cpu_state


def fleurs_rows(split, bundle):
    rows = {row["id"]: row for row in records(split, FLEURS_REVISION)}
    if set(rows) != set(bundle["ids"]):
        raise ValueError("FLEURS audio and original evaluation cache differ")
    return [rows[item] for item in bundle["ids"]]


def gain_diagnostic(encoder, head, rows, output):
    chosen = []
    for code in CODES:
        candidates = [row for row in rows if row["language"] == code]
        if all("rms_dbfs" in row for row in candidates):
            candidates.sort(key=lambda row: row["rms_dbfs"])
        chosen.extend(
            candidates[int(index)]
            for index in np.linspace(0, len(candidates) - 1, min(12, len(candidates)))
        )
    results = []
    encoder.eval()
    head.eval()
    with torch.inference_mode():
        for row in chosen:
            signal = torch.from_numpy(waveform(row, 10)).unsqueeze(0).to("cuda")
            original = encoder.encode_batch(signal, normalize=False)
            original_prediction = head(original).argmax(-1).item()
            for requested in (-12, 0, 12):
                gain = min(
                    10 ** (requested / 20),
                    0.999 / max(signal.abs().max().item(), 1e-12),
                )
                embedding = encoder.encode_batch(signal * gain, normalize=False)
                prediction = head(embedding).argmax(-1).item()
                results.append(
                    {
                        "id": row.get("id", row["path"]),
                        "language": row["language"],
                        "gain_db_requested": requested,
                        "gain_db_applied": 20 * np.log10(gain),
                        "embedding_cosine": torch.nn.functional.cosine_similarity(
                            original.reshape(1, -1), embedding.reshape(1, -1)
                        ).item(),
                        "prediction_changed": prediction != original_prediction,
                        "correct": prediction == CODES.index(row["language"]),
                        "rms_dbfs": 20
                        * torch.log10(
                            signal.square().mean().sqrt().clamp_min(1e-12)
                        ).item(),
                    }
                )
    write_json(
        output,
        {
            "protocol": "selection data only; gain capped to peak 0.999; no language-specific normalization",
            "recordings": len(chosen),
            "measurements": results,
        },
    )


def export_bundle(
    checkpoint,
    head,
    temperature,
    output,
    expected_logits,
    test_embeddings,
    calibration,
    sample_row,
):
    from speechbrain.dataio.encoder import CategoricalEncoder
    from speechbrain.inference.classifiers import EncoderClassifier

    output.mkdir(parents=True, exist_ok=True)
    yaml = (Path("artifacts/pretrained") / REVISION / "hyperparams.yaml").read_text()
    yaml = (
        yaml.replace("pretrained_path: speechbrain/lang-id-voxlingua107-ecapa", "")
        .replace("out_n_neurons: 107", "out_n_neurons: 15")
        .split("    paths:\n", 1)[0]
    )
    if "pretrained_path" in yaml or "out_n_neurons: 15" not in yaml:
        raise ValueError("Unexpected SpeechBrain template")
    weights = cpu_state(head)
    for parameter in ("weight", "bias"):
        weights[f"out.w.{parameter}"] = weights.pop(f"out.{parameter}") / temperature
    torch.save(checkpoint["encoder"], output / "embedding_model.ckpt")
    torch.save(weights, output / "classifier.ckpt")
    labels = CategoricalEncoder()
    labels.update_from_iterable([f"{code}: {name}" for code, name in zip(CODES, NAMES)])
    labels.expect_len(15)
    labels.save(output / "label_encoder.ckpt")
    shutil.copyfile(output / "label_encoder.ckpt", output / "label_encoder.txt")
    (output / "hyperparams.yaml").write_text(yaml)
    write_json(output / "calibration.json", calibration)
    metadata = {
        "codes": CODES,
        "names": NAMES,
        "method": checkpoint["identity"]["method"],
        "base_revision": REVISION,
        "epoch": checkpoint["epoch"],
        "temperature": temperature,
        "temperature_folded_into_classifier": True,
        "preprocessing": checkpoint["preprocessing"],
        "source_model_sha256": calibration["model_sha256"],
        "files_sha256": {
            path.name: file_hash(path)
            for path in sorted(output.iterdir())
            if path.is_file()
            and path.name not in ("export.json", "verification.json", "README.md")
        },
    }
    write_json(output / "export.json", metadata)
    (output / "README.md").write_text(
        "# SpeechBrain 15-language identification\n\n"
        + "```python\nfrom speechbrain.inference.classifiers import EncoderClassifier\nmodel = EncoderClassifier.from_hparams(source='/path/to/this/bundle', run_opts={'device': 'cuda'})\nlog_probs, score, index, labels = model.classify_file('recording.wav')\nprobabilities = log_probs.exp()\n```\n\n"
        + "Copy this entire directory. No project code is required. Temperature calibration is already folded into the classifier; do not apply it again.\n\n"
        + "Output order: "
        + ", ".join(CODES)
        + ". Cook Islands Māori is not included.\n\n"
        + "Evaluation uses mono 16 kHz audio, center-cropped to at most 10 seconds. Standard classify_file resamples but does not crop; reproduce this crop for benchmark equivalence.\n"
    )
    # Load an independent copy: no original paths or project-specific classes.
    with tempfile.TemporaryDirectory(prefix="lid-extension15-verify-") as temporary:
        relocated = Path(temporary) / "model"
        shutil.copytree(output, relocated)
        bundle = EncoderClassifier.from_hparams(
            source=str(relocated),
            savedir=str(Path(temporary) / "loaded"),
            run_opts={"device": "cpu"},
        )
        assert [
            bundle.hparams.label_encoder.ind2lab[i].split(":")[0] for i in range(15)
        ] == CODES
        with torch.inference_mode():
            actual = torch.cat(
                [
                    bundle.mods.classifier(batch.unsqueeze(1)).squeeze(1).exp()
                    for batch in test_embeddings.split(128)
                ]
            )
            expected = torch.softmax(expected_logits / temperature, dim=-1)
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
            mismatches = int((actual.argmax(-1) != expected.argmax(-1)).sum())
            if mismatches:
                raise ValueError(f"Export changed {mismatches} predictions")
            signal = torch.from_numpy(waveform(sample_row, 10)).unsqueeze(0)
            probabilities, _, _, _ = bundle.classify_batch(signal)
            torch.testing.assert_close(probabilities.exp().sum(-1), torch.ones(1))
            file_probabilities, _, _, _ = bundle.classify_file(sample_row["path"])
            assert file_probabilities.shape == (1, 15)
        write_json(
            output / "verification.json",
            {
                "offline_relocated_load": True,
                "matching_predictions": len(actual),
                "maximum_probability_error": float((actual - expected).abs().max()),
                "classify_batch": "passed",
                "classify_file": "passed",
            },
        )
        del bundle
    gc.collect()


def baseline_reports():
    """Recompute original reference rows directly from the saved test logits."""
    saved = json.loads((BASE / "comparison/report.json").read_text())
    predictions = np.load(BASE / "comparison/predictions.npz")
    bundle = old_cache("fleurs_test")
    from .languages import selected_indices

    indices = selected_indices(bundle["labels"])
    output = {}
    for name, key, targets, report_indices in (
        ("original_107", "original_logits", predictions["original_targets"], indices),
        ("filtered_12", "filtered_logits", predictions["targets"], None),
        ("retrained_head_12", "head_logits", predictions["targets"], None),
    ):
        for suffix in ("", "_temperature"):
            method = name + suffix
            result = evaluate(
                predictions[key],
                targets,
                NAMES[:12],
                temperature=saved["methods"][method]["temperature"],
                report_indices=report_indices,
            )
            for metric in (
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "ece",
                "nll",
            ):
                if abs(result[metric] - saved["methods"][method][metric]) > 1e-8:
                    raise ValueError(
                        f"Old baseline no longer reproduced: {method}/{metric}"
                    )
            output[method] = result
    return output


def evaluate_method(method, run=RUN, progress=None):
    run = Path(run)
    emit = progress or status
    output = run / method
    if (output / "evaluation_complete.json").exists():
        return
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["codes"] != CODES:
        raise ValueError("Wrong extension vocabulary")
    encoder, head, _ = load_initial()
    encoder.mods.embedding_model.load_state_dict(checkpoint["encoder"])
    head.load_state_dict(checkpoint["head"])
    encoder.eval()
    head.eval()
    fingerprint = state_fingerprint(encoder.mods.embedding_model)
    datasets = {}
    for split in ("calibration", "test"):
        old_split = "validation" if split == "calibration" else "test"
        old = old_cache(f"fleurs_{old_split}")
        old_rows = fleurs_rows(old_split, old)
        new_rows = read_rows(run / f"new_{split}.jsonl")
        rows = old_rows + new_rows
        if method == "frozen":
            new = cache_embeddings(
                encoder,
                new_rows,
                run / f"new_{split}.pt",
                fingerprint,
                f"cache_new_{split}",
                progress=emit,
            )
            embeddings = torch.cat([old["embeddings"], new])
        else:
            embeddings = cache_embeddings(
                encoder,
                rows,
                output / f"{split}_embeddings.pt",
                fingerprint,
                f"evaluate_{method}_{split}",
                progress=emit,
            )
        targets = np.array([CODES.index(row["language"]) for row in rows])
        logits = head_scores(head, embeddings)
        datasets[split] = {
            "rows": rows,
            "targets": targets,
            "logits": logits,
            "embeddings": embeddings,
            "old_n": len(old_rows),
        }
        del old
    calibration_data, test = datasets["calibration"], datasets["test"]
    if {row["id"] for row in calibration_data["rows"]} & {
        row["id"] for row in test["rows"]
    }:
        raise ValueError("Calibration and test overlap")
    temperature = fit_temperature(
        calibration_data["logits"].numpy(), calibration_data["targets"]
    )
    count = calibration_data["old_n"]
    old_temperature = fit_temperature(
        calibration_data["logits"][:count].numpy(), calibration_data["targets"][:count]
    )
    reports = {}
    for group, mask, names, indices in (
        ("old12", test["targets"] < 12, NAMES[:12], list(range(12))),
        ("new3", test["targets"] >= 12, NAMES[12:], list(range(12, 15))),
        ("all15", np.ones(len(test["targets"]), dtype=bool), NAMES, list(range(15))),
    ):
        reports[group] = {}
        for variant, value in (
            ("raw", 1.0),
            ("joint_temperature", temperature),
            ("old_only_temperature", old_temperature),
        ):
            if variant == "old_only_temperature" and group != "old12":
                continue
            reports[group][variant] = evaluate(
                test["logits"][mask].numpy(),
                test["targets"][mask],
                names,
                temperature=value,
                report_indices=indices,
            )
    calibration = {
        "temperature": temperature,
        "old_only_temperature": old_temperature,
        "fit_objective": "unweighted mixed-validation NLL",
        "bounds": [0.05, 20],
        "model_sha256": file_hash(output / "model.pt"),
        "codes": CODES,
        "encoder_fingerprint": fingerprint,
        "calibration_ids": [row["id"] for row in calibration_data["rows"]],
    }
    write_json(output / "calibration.json", calibration)
    write_json(
        output / "report.json",
        {
            "method": method,
            "groups": reports,
            "vocabulary": CODES,
            "all15_outputs_enabled_for_every_group": True,
            "ece_bins": 15,
        },
    )
    np.savez_compressed(
        output / "predictions.npz",
        logits=test["logits"].numpy(),
        targets=test["targets"],
        ids=np.array([row["id"] for row in test["rows"]]),
        groups=np.array([row["group"] for row in test["rows"]]),
    )
    selection_old = read_rows(BASE / "manifests/validation.jsonl")
    selection_new = read_rows(run / "new_selection.jsonl")
    emit(f"gain_diagnostic_{method}")
    gain_diagnostic(
        encoder, head, selection_old + selection_new, output / "gain_diagnostic.json"
    )
    emit(f"export_{method}")
    export_bundle(
        checkpoint,
        head,
        temperature,
        output / "speechbrain_bundle",
        test["logits"],
        test["embeddings"],
        calibration,
        test["rows"][0],
    )
    write_json(
        output / "evaluation_complete.json",
        {
            "method": method,
            "test_recordings": len(test["targets"]),
            "temperature": temperature,
            "bundle_verified": True,
        },
    )


def write_comparison():
    reference = baseline_reports()
    write_json(RUN / "baseline_reproduction.json", reference)
    reports = {
        method: json.loads((RUN / method / "report.json").read_text())["groups"]
        for method in ("frozen", "partial")
    }
    rows = []
    for method, value in reference.items():
        rows.append(
            {
                "method": method,
                "group": "old12",
                "variant": "reference",
                **{
                    key: value[key]
                    for key in (
                        "n",
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "ece",
                        "nll",
                    )
                },
            }
        )
    for method, groups in reports.items():
        for group, variants in groups.items():
            for variant, value in variants.items():
                rows.append(
                    {
                        "method": method,
                        "group": group,
                        "variant": variant,
                        **{
                            key: value[key]
                            for key in (
                                "n",
                                "accuracy",
                                "macro_precision",
                                "macro_recall",
                                "macro_f1",
                                "ece",
                                "nll",
                            )
                        },
                    }
                )
    with (RUN / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_language = []
    for method, groups in reports.items():
        for language, metrics in groups["all15"]["raw"]["per_language"].items():
            old = reference["retrained_head_12"]["per_language"].get(language)
            per_language.append(
                {
                    "method": method,
                    "language": language,
                    **metrics,
                    "f1_delta_pp_vs_previous12": 100 * (metrics["f1"] - old["f1"])
                    if old
                    else "",
                    "recall_delta_pp_vs_previous12": 100
                    * (metrics["recall"] - old["recall"])
                    if old
                    else "",
                }
            )
    with (RUN / "per_language.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_language[0]))
        writer.writeheader()
        writer.writerows(per_language)
    write_extension_report(reference, reports)
    write_json(
        RUN / "comparison_complete.json",
        {"baseline_reproduced": True, "codes": CODES, "models": ["frozen", "partial"]},
    )


def write_extension_report(reference, reports):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path("reports/extension15")
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
    labels = ["Frozen encoder", "Partial encoder"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    for ax, group, title in zip(
        axes, ("old12", "new3", "all15"), ("Original 12", "New 3", "All 15")
    ):
        values = [reports[method][group]["raw"]["macro_f1"] * 100 for method in reports]
        ax.bar(labels, values, color=colors)
        for i, value in enumerate(values):
            ax.text(i, value + 1, f"{value:.2f}", ha="center")
        ax.set(title=title, ylim=(0, 106), ylabel="Macro F1 (%)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figures / f"performance.{extension}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    y = np.arange(12)
    for offset, method, color, label in zip((-0.12, 0.12), reports, colors, labels):
        deltas = [
            100
            * (
                reports[method]["old12"]["raw"]["per_language"][name]["f1"]
                - reference["retrained_head_12"]["per_language"][name]["f1"]
            )
            for name in NAMES[:12]
        ]
        ax.scatter(deltas, y + offset, color=color, label=label)
    ax.axvline(0, color="#888888", linestyle="--")
    ax.set_yticks(y, NAMES[:12])
    ax.invert_yaxis()
    ax.set(
        xlabel="F1 change versus previous 12-language model (pp)",
        title="Regression on the original languages",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figures / f"regression.{extension}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(
        1, 2, figsize=(8, 3.7), sharex=True, sharey=True, layout="constrained"
    )
    for ax, method, label in zip(axes, reports, labels):
        ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
        for variant, color, marker, name in (
            ("raw", colors[0], "o", "Raw"),
            ("joint_temperature", colors[1], "s", "Calibrated"),
        ):
            value = reports[method]["all15"][variant]
            bins = [row for row in value["reliability"] if row["count"]]
            ax.scatter(
                [row["confidence"] for row in bins],
                [row["accuracy"] for row in bins],
                s=[10 + 100 * row["count"] / value["n"] for row in bins],
                color=color,
                marker=marker,
                label=f"{name}: ECE {value['ece'] * 100:.2f}%",
            )
        ax.set(title=label, xlabel="Mean confidence", xlim=(0, 1.025), ylim=(0, 1.025))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_ylabel("Observed accuracy")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(figures / f"reliability.{extension}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    table = []
    for method, label in zip(reports, labels):
        for group in ("old12", "new3", "all15"):
            before = reports[method][group]["raw"]
            after = reports[method][group]["joint_temperature"]
            table.append(
                f"| {label} | {group} | {before['n']:,} | {100 * before['accuracy']:.2f} | {100 * before['macro_precision']:.2f} | {100 * before['macro_recall']:.2f} | {100 * before['macro_f1']:.2f} | {100 * before['ece']:.2f} | {100 * after['ece']:.2f} |"
            )
    language_rows = []
    for name in NAMES:
        values = [
            reports[method]["all15"]["raw"]["per_language"][name] for method in reports
        ]
        language_rows.append(
            f"| {name} | {values[0]['support']:,} | {values[0]['f1'] * 100:.2f} | {values[1]['f1'] * 100:.2f} |"
        )
    audit = json.loads((RUN / "data_ready.json").read_text())
    old_comparison = []
    for method, label in (
        ("original_107", "Original 107"),
        ("filtered_12", "Filtered 12"),
        ("retrained_head_12", "Previous fine-tuned 12"),
    ):
        if method in reference:
            value = reference[method]
            calibrated = reference.get(method + "_temperature", value)
            old_comparison.append(
                f"| {label} | {value['accuracy'] * 100:.2f} | {value['macro_f1'] * 100:.2f} | {calibrated['ece'] * 100:.2f} |"
            )
    for method, label in zip(reports, labels):
        value = reports[method]["old12"]["raw"]
        calibrated = reports[method]["old12"].get(
            "old_only_temperature", reports[method]["old12"]["joint_temperature"]
        )
        old_comparison.append(
            f"| {label} (15 outputs) | {value['accuracy'] * 100:.2f} | {value['macro_f1'] * 100:.2f} | {calibrated['ece'] * 100:.2f} |"
        )
    gain_rows, confusion_rows = [], []
    for method, label in zip(reports, labels):
        diagnostic = json.loads((RUN / method / "gain_diagnostic.json").read_text())
        for gain in (-12, 12):
            measurements = [
                row
                for row in diagnostic["measurements"]
                if row["language"] == "sam" and row["gain_db_requested"] == gain
            ]
            gain_rows.append(
                f"| {label} | {gain:+d} | {len(measurements)} | {np.median([row['embedding_cosine'] for row in measurements]):.6f} | {sum(row['prediction_changed'] for row in measurements)} |"
            )
        matrix = np.array(reports[method]["all15"]["raw"]["confusion_matrix"])
        pairs = sorted(
            [
                (int(matrix[i, j]), i, j)
                for i in range(15)
                for j in range(15)
                if i != j and matrix[i, j] and (i >= 12 or j >= 12 or i == 6 or j == 6)
            ],
            reverse=True,
        )
        for count, truth, predicted in pairs[:5]:
            confusion_rows.append(
                f"| {label} | {NAMES[truth]} | {NAMES[predicted]} | {count} |"
            )
    text = (
        "# Adding Samoan, Niuean and Tongan\n\n"
        + "## Experiment\n\n- Vocabulary: the original 12 languages plus `sam`, `niu`, `ton`. **Cook Islands Māori is excluded.**\n"
        + f"- Audited {audit['original_files']:,} new cuts; {audit['excluded']:,} excluded. Grouped by source recording, approximately 80% train / 5% checkpoint selection / 5% calibration / 10% test. Source grouping does not establish speaker independence.\n"
        + "- One paired run, seed 42. Both methods start from the same expanded classifier and replay original VoxLingua training data with equal class sampling.\n"
        + "- Frozen: train classifier only. Partial: additionally train ECAPA `mfa`, `asp`, and `fc`; earlier blocks and encoder batch-normalization parameters/statistics stay frozen.\n"
        + "- AdamW; head LR 1e-4, encoder LR 1e-5, effective batch 32, at most 20 epochs of 32,768 samples, patience 5. Select by 15-class validation macro F1, then balanced NLL.\n"
        + "- Mono 16 kHz, 10-second maximum center crop; native log-mel sentence mean normalization, no Samoan-specific amplification.\n"
        + "- FLEURS test evaluates the old 12; source-held-out local test cuts evaluate the new 3. Every extension evaluation retains **all 15 outputs**. Domains differ, so combined accuracy depends on dataset composition.\n"
        + "- Temperatures fit on separate FLEURS + new-language calibration data. ECE uses 15 equal-width bins. Old-only calibration comparisons are also in the CSV.\n\n"
        + "## Results\n\nAll metric values are percentages. Precision, recall and F1 are macro-averaged over the indicated group.\n\n"
        + "| Model | Group | Test n | Accuracy | Precision | Recall | F1 | ECE raw | ECE calibrated |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(table)
        + "\n\n"
        + f"- Previous 12-language head: old-language accuracy **{reference['retrained_head_12']['accuracy'] * 100:.2f}%**, macro F1 **{reference['retrained_head_12']['macro_f1'] * 100:.2f}%**. All original reference results were recomputed and checked against the saved experiment.\n"
        + "- Read original-language regressions alongside new-language gains; checkpoint selection did not impose a regression limit.\n\n"
        + "### Matched original-language comparison\n\nSame 8,519 FLEURS test recordings; temperatures fitted using the original FLEURS validation set only. All values are percentages.\n\n| Model | Accuracy | Macro F1 | Calibrated ECE |\n|---|---:|---:|---:|\n"
        + "\n".join(old_comparison)
        + "\n\n"
        + "![Performance by evaluation group](figures/performance.png)\n\n"
        + "![Original-language F1 changes](figures/regression.png)\n\n"
        + "![Calibration across all 15 languages](figures/reliability.png)\n\n*Reliability points are nonempty bins; marker area reflects sample count plus a visibility floor. Sparse bins may be noisy.*\n\n"
        + "## Per-language results\n\n| Language | Test n | Frozen F1 (%) | Partial F1 (%) |\n|---|---:|---:|---:|\n"
        + "\n".join(language_rows)
        + "\n\n"
        + "The per-language table uses the combined test set, including false positives from the new languages. The regression plot uses the unchanged original-language test set for a matched comparison.\n\n"
        + "### Confusions involving Pacific languages\n\nLargest confusion counts involving a new language or Māori (up to five per model):\n\n| Model | True language | Predicted language | Errors |\n|---|---|---|---:|\n"
        + "\n".join(confusion_rows)
        + "\n\n"
        + "### Samoan volume sensitivity\n\nValidation-only gain probe; 12 Samoan cuts spanning measured recording levels per model, with amplification capped to prevent clipping. No test data or gain probe was used to select preprocessing.\n\n| Model | Requested gain (dB) | Cuts | Median embedding cosine | Changed predictions |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(gain_rows)
        + "\n\nUniform gain robustness does not measure noise robustness. Gain limits and per-recording measurements are in each model's diagnostic JSON.\n\n"
        + "## Artifacts and limits\n\n- [Full comparison](../../artifacts/extension15/summary.csv) · [Per-language results](../../artifacts/extension15/per_language.csv).\n"
        + "- [Frozen bundle](../../artifacts/extension15/frozen/speechbrain_bundle/) · [Partial bundle](../../artifacts/extension15/partial/speechbrain_bundle/). Both contain actual evaluated encoder weights, 15 labels, folded temperature calibration, and offline relocation verification.\n"
        + "- Per-model reports include confusion matrices, raw/joint-calibrated metrics, old-only calibration, training histories, memory measurements, and gain-sensitivity diagnostics. Figures are PNG (350 dpi), PDF and SVG.\n"
        + "- One training seed; no training-variance estimate or significance claim. News-domain source-held-out results do not establish cross-speaker or deployment-domain generalization. No unknown-language or silence rejection policy was evaluated.\n"
    )
    (destination / "language_extension.md").write_text(text)
