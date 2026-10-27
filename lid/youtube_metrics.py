"""Conditional metrics: English predictions are counted and excluded explicitly."""

from collections import Counter

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .youtube_data import TARGETS


def score_segments(rows, predictions, confidence, mask=None, *, exclude_english=True):
    n = len(rows)
    if len(predictions) != n or len(confidence) != n:
        raise ValueError("Predictions and segment manifest differ in length")
    truth = np.array([row["language"] for row in rows])
    predictions = np.array(predictions)
    confidence = np.array(confidence, dtype=float)
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Invalid confidence values")
    eligible = np.ones(n, dtype=bool) if mask is None else np.array(mask, dtype=bool)
    if exclude_english:
        eligible = eligible & (predictions != "en")
    if eligible.shape != (n,):
        raise ValueError("Wrong cohort shape")
    duration = np.array([row["duration"] for row in rows])
    correct = predictions == truth
    total_seconds = float(duration.sum())
    result = {
        "segments_total": n,
        "english_predictions": int((predictions == "en").sum()),
        "english_seconds": float(duration[predictions == "en"].sum()),
        "segments_scored": int(eligible.sum()),
        "coverage": float(eligible.mean()) if n else None,
        "seconds_scored": float(duration[eligible].sum()),
        "duration_coverage": float(duration[eligible].sum()) / total_seconds
        if total_seconds
        else None,
        "correct_scored": int(correct[eligible].sum()),
        "other_errors": int((~correct[eligible]).sum()),
        "accuracy": None,
        "duration_weighted_accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "ece": None,
        "macro_languages": [],
        "per_language": {},
        "prediction_counts_all": dict(Counter(predictions.tolist())),
        "confusion_scored": {},
    }
    if eligible.any():
        result["accuracy"] = float(correct[eligible].mean())
        result["duration_weighted_accuracy"] = float(
            np.average(correct[eligible], weights=duration[eligible])
        )
        represented = [code for code in TARGETS if np.any(truth[eligible] == code)]
        precision, recall, f1, support = precision_recall_fscore_support(
            truth[eligible], predictions[eligible], labels=TARGETS, zero_division=0
        )
        result["macro_languages"] = represented
        for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
            result[f"macro_{name}"] = float(
                np.mean([values[TARGETS.index(code)] for code in represented])
            )
        for i, code in enumerate(TARGETS):
            result["per_language"][code] = {
                "precision": float(precision[i]) if support[i] else None,
                "recall": float(recall[i]) if support[i] else None,
                "f1": float(f1[i]) if support[i] else None,
                "support": int(support[i]),
            }
        assignments = np.minimum((confidence[eligible] * 15).astype(int), 14)
        result["ece"] = float(
            sum(
                np.mean(assignments == index)
                * abs(
                    correct[eligible][assignments == index].mean()
                    - confidence[eligible][assignments == index].mean()
                )
                for index in range(15)
                if np.any(assignments == index)
            )
        )
        result["confusion_scored"] = {
            code: dict(Counter(predictions[eligible & (truth == code)].tolist()))
            for code in TARGETS
        }
    # Coverage/counts are displayed even when every segment is English/abstained.
    result["by_language"] = {}
    for code in TARGETS:
        selection = truth == code
        scored = eligible & selection
        result["by_language"][code] = {
            "total": int(selection.sum()),
            "english": int(((predictions == "en") & selection).sum()),
            "scored": int(scored.sum()),
            "correct": int(correct[scored].sum()),
            "accuracy": float(correct[scored].mean()) if scored.any() else None,
            "coverage": float(scored.sum() / selection.sum())
            if selection.any()
            else None,
        }
    result["by_source"] = {}
    for source in dict.fromkeys(row["source_id"] for row in rows):
        selected = np.array([row["source_id"] == source for row in rows])
        scored = selected & eligible
        result["by_source"][source] = {
            "language": truth[selected][0],
            "total": int(selected.sum()),
            "english": int((selected & (predictions == "en")).sum()),
            "scored": int(scored.sum()),
            "accuracy": float(correct[scored].mean()) if scored.any() else None,
            "coverage": float(scored.sum() / selected.sum()),
        }
    clip_accuracies = [
        value["accuracy"]
        for value in result["by_source"].values()
        if value["accuracy"] is not None
    ]
    result["mean_source_accuracy"] = (
        float(np.mean(clip_accuracies)) if clip_accuracies else None
    )
    result["sources_scored"] = len(clip_accuracies)
    return result


def common_cohort(predictions_by_model, models=("frozen15", "partial15")):
    if not models or any(model not in predictions_by_model for model in models):
        raise ValueError("Missing model for shared comparison cohort")
    lengths = {len(predictions_by_model[model]) for model in models}
    if len(lengths) != 1:
        raise ValueError("Model prediction counts differ")
    return np.logical_and.reduce(
        [np.array(predictions_by_model[model]) != "en" for model in models]
    )
