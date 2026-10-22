"""Top-label ECE and multiclass metrics, including predictions outside the set."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def validate(logits, targets):
    logits, targets = np.asarray(logits, dtype=np.float64), np.asarray(targets)
    if logits.ndim != 2 or targets.shape != (len(logits),) or len(targets) == 0:
        raise ValueError("Expected nonempty [N,C] logits and [N] targets")
    if not np.isfinite(logits).all() or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Logits must be finite and targets integer-valued")
    if targets.min() < 0 or targets.max() >= logits.shape[1]:
        raise ValueError("Target is outside the logit columns")
    return logits, targets


def fit_temperature(logits, targets):
    logits, targets = validate(logits, targets)

    def nll(log_t):
        scaled = logits / np.exp(log_t)
        return np.mean(
            logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]
        )

    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20)), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fitting failed")
    return float(np.exp(result.x))


def reliability(probs, targets, n_bins=15):
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == targets
    # [0, 1/B), ..., [(B-1)/B, 1], including exactly 0 and 1.
    assignments = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = assignments == i
        rows.append(
            {
                "lower": i / n_bins,
                "upper": (i + 1) / n_bins,
                "count": int(mask.sum()),
                "confidence": float(confidence[mask].mean()) if mask.any() else 0.0,
                "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
            }
        )
    ece = sum(
        row["count"] / len(targets) * abs(row["accuracy"] - row["confidence"])
        for row in rows
    )
    return float(ece), rows


def evaluate(logits, targets, names, temperature=1.0, n_bins=15, report_indices=None):
    logits, targets = validate(logits, targets)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = logits / temperature
    probs = softmax(scaled, axis=1)
    predicted = probs.argmax(axis=1)
    indices = (
        list(range(len(names))) if report_indices is None else list(report_indices)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predicted, labels=indices, zero_division=0
    )
    ece, bins = reliability(probs, targets, n_bins)
    report = {
        "n": len(targets),
        "accuracy": float(accuracy_score(targets, predicted)),
        "ece": ece,
        "temperature": temperature,
        "nll": float(
            np.mean(
                logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]
            )
        ),
        "brier": float(
            np.mean(
                np.sum(probs**2, axis=1)
                - 2 * probs[np.arange(len(targets)), targets]
                + 1
            )
        ),
        "outside_set_rate": float(np.mean(~np.isin(predicted, indices))),
        "per_language": {},
        "reliability": bins,
    }
    for metric, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        report[f"macro_{metric}"] = float(values.mean())
        report[f"weighted_{metric}"] = float(np.average(values, weights=support))
    for i, name in enumerate(names):
        report["per_language"][name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    report["confusion_matrix"] = [
        [int(np.sum((targets == i) & (predicted == j))) for j in range(logits.shape[1])]
        for i in indices
    ]
    return report
