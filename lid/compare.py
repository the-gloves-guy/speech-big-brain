"""Calibrate on FLEURS validation, then report once on held-out FLEURS test."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from .cache import compatible, disjoint, load_cache
from .data import write_json
from .languages import NAMES, selected_indices
from .metrics import evaluate, fit_temperature
from .model import new_head
from .train import head_logits


def compare(args):
    if args.bins < 1:
        raise ValueError("bins must be positive")
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    torch.set_num_threads(args.threads)
    calibration, test = load_cache(args.calibration), load_cache(args.test)
    compatible(calibration, test)
    disjoint(calibration, test)
    if (calibration["meta"]["source"], calibration["meta"]["split"]) != (
        "fleurs",
        "validation",
    ):
        raise ValueError("Temperature fitting requires the FLEURS validation split")
    if (test["meta"]["source"], test["meta"]["split"]) != ("fleurs", "test"):
        raise ValueError("Evaluation requires the FLEURS test split")
    if calibration["meta"]["dataset_revision"] != test["meta"]["dataset_revision"]:
        raise ValueError(
            "FLEURS validation and test must use the same dataset revision"
        )
    checkpoint = torch.load(args.head, map_location="cpu", weights_only=True)
    compatible(
        calibration, {"meta": checkpoint["meta"], "labels": checkpoint["labels"]}
    )
    for bundle in (calibration, test):
        if set(bundle["ids"]) & set(
            checkpoint["train_ids"] + checkpoint["validation_ids"]
        ) or set(bundle["groups"]) & set(
            checkpoint["train_groups"] + checkpoint["validation_groups"]
        ):
            raise ValueError("Head fitting data overlaps evaluation/calibration data")
    indices = selected_indices(test["labels"])
    if indices != checkpoint["indices"]:
        raise ValueError("Checkpoint uses a different language ordering")
    head = new_head(checkpoint["initial_state"], indices)
    head.load_state_dict(checkpoint["state"])
    adapted_cal = head_logits(head, calibration["embeddings"]).numpy()
    adapted_test = head_logits(head, test["embeddings"]).numpy()
    cal_targets, test_targets = calibration["targets"].numpy(), test["targets"].numpy()
    filtered_cal = calibration["logits"][:, indices].numpy()
    filtered_test = test["logits"][:, indices].numpy()
    filtered_t = fit_temperature(filtered_cal, cal_targets)
    head_t = fit_temperature(adapted_cal, cal_targets)
    baseline_t = fit_temperature(
        calibration["logits"].numpy(), np.array(indices)[cal_targets]
    )
    # Original classifier remains a 107-way distribution: out-of-set guesses count as errors.
    baseline_targets = np.array(indices)[test_targets]
    reports = {
        "original_107": evaluate(
            test["logits"].numpy(),
            baseline_targets,
            NAMES,
            n_bins=args.bins,
            report_indices=indices,
        ),
        "original_107_temperature": evaluate(
            test["logits"].numpy(),
            baseline_targets,
            NAMES,
            temperature=baseline_t,
            n_bins=args.bins,
            report_indices=indices,
        ),
    }
    variants = {
        "filtered_12": (filtered_test, 1.0),
        "filtered_12_temperature": (filtered_test, filtered_t),
        "retrained_head_12": (adapted_test, 1.0),
        "retrained_head_12_temperature": (adapted_test, head_t),
    }
    for name, (scores, temperature) in variants.items():
        reports[name] = evaluate(scores, test_targets, NAMES, temperature, args.bins)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = hashlib.sha256(Path(args.head).read_bytes()).hexdigest()
    write_json(
        output / "report.json",
        {
            "methods": reports,
            "metadata": {
                "calibration": calibration["meta"],
                "test": test["meta"],
                "head_sha256": checkpoint_hash,
                "head_best_epoch": checkpoint["best_epoch"],
                "ece_bins": args.bins,
                "ece_definition": "equal-width, top-label, sample-weighted",
                "confusion_rows": NAMES,
                "baseline_confusion_columns": test["labels"],
                "adapted_confusion_columns": NAMES,
            },
        },
    )
    write_json(
        output / "calibration.json",
        {
            "filtered_temperature": filtered_t,
            "head_temperature": head_t,
            "original_temperature": baseline_t,
            "fit_objective": "validation NLL",
            "temperature_bounds": [0.05, 20],
            "head_sha256": checkpoint_hash,
            "calibration_ids": calibration["ids"],
            "metadata": calibration["meta"],
        },
    )
    columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "ece",
        "nll",
        "brier",
        "outside_set_rate",
        "temperature",
        "n",
    ]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *columns])
        writer.writeheader()
        for name, report in reports.items():
            writer.writerow({"method": name, **{key: report[key] for key in columns}})
    with (output / "per_language.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "language", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        for name, report in reports.items():
            for language, values in report["per_language"].items():
                writer.writerow({"method": name, "language": language, **values})
    np.savez_compressed(
        output / "predictions.npz",
        ids=np.array(test["ids"]),
        targets=test_targets,
        original_targets=baseline_targets,
        original_logits=test["logits"].numpy(),
        filtered_logits=filtered_test,
        head_logits=adapted_test,
    )
    plot_reliability(reports, output / "reliability.png")
    for name, report in reports.items():
        print(
            f"{name:34} accuracy={report['accuracy']:.4f} macro-F1={report['macro_f1']:.4f} ECE={report['ece']:.4f}"
        )
    print(f"Reports saved to {output}")


def plot_reliability(reports, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(reports), figsize=(20, 4), sharex=True, sharey=True)
    for ax, (name, report) in zip(axes, reports.items()):
        bins = [row for row in report["reliability"] if row["count"]]
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.scatter(
            [row["confidence"] for row in bins],
            [row["accuracy"] for row in bins],
            s=[20 + 180 * row["count"] / report["n"] for row in bins],
        )
        ax.set(
            title=f"{name.replace('_', ' ')}\nECE={report['ece']:.3f}",
            xlabel="Mean confidence",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Bin accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
