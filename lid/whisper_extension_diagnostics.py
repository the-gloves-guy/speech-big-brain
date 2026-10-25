"""Matched duration and selection-only volume probes, including the Pacific classes."""

import json
from collections import Counter

import numpy as np
from scipy.special import softmax

from .extension_data import CODES, file_hash, write_json
from .whisper_extension import ExtensionAdapter
from .youtube_evaluate import infer, write_csv


def run(study, original):
    directory = study.root / "diagnostics"
    directory.mkdir(exist_ok=True)
    rows = original.prepare_rows(
        study,
        directory,
        codes=CODES,
        selection_cohorts=["vox_validation", "new_selection"],
    )
    models = {}
    for method in ["frozen", "finetuned"]:
        adapter = ExtensionAdapter(study.root, study.root / method / "model.pt")
        scores = study.score_adapter(
            adapter,
            rows,
            directory / f"whisper15_{method}.npz",
            f"diagnostics_{method}",
        )
        adapter.release()
        models[f"whisper15_{method}"] = (
            scores,
            CODES,
            json.loads((study.root / method / "calibration.json").read_text())[
                "temperature"
            ],
        )
    # Apply the same diagnostic waveforms to the unchanged SpeechBrain checkpoints.
    sb = directory / "speechbrain"
    sb.mkdir(exist_ok=True)
    if not (sb / "complete.json").exists():
        scores = infer(rows, sb)
        write_json(
            sb / "complete.json", {"rows_sha256": file_hash(directory / "rows.jsonl")}
        )
    else:
        if json.loads((sb / "complete.json").read_text())["rows_sha256"] != file_hash(
            directory / "rows.jsonl"
        ):
            raise ValueError("Diagnostic SpeechBrain rows changed")
        meta = json.loads((sb / "model_provenance.json").read_text())
        with np.load(sb / "logits.npz") as saved:
            if saved["ids"].tolist() != [r["id"] for r in rows]:
                raise ValueError("Diagnostic IDs differ")
            scores = {
                m: (saved[m], v["codes"], v["temperature"]) for m, v in meta.items()
            }
    models.update({f"speechbrain_{m}": v for m, v in scores.items()})
    # Existing Whisper12 controls and full-encoder SpeechBrain12 control.
    from types import SimpleNamespace

    proxy = SimpleNamespace(root=study.base, config=study.config, status=study.status)
    for method in [
        "whisper_filtered",
        "whisper_frozen",
        "whisper_finetuned",
        "speechbrain_finetuned",
    ]:
        values = original.score_method(proxy, method, rows, directory / f"{method}.npz")
        temperature = json.loads(
            (study.base / method / "calibration.json").read_text()
        )["temperature"]
        models[method] = (values, CODES[:12], temperature)
    summarize(directory, rows, models)


def summarize(directory, rows, models):
    duration, gain, predictions_table = [], [], []
    for method, (scores, codes, temperature) in models.items():
        predicted = np.array([codes[i] for i in scores.argmax(1)])
        confidence = softmax(scores.astype(float) / temperature, axis=1).max(1)
        for language in ["hi", "th", "sam", "niu", "ton"]:
            for seconds in [3, 5, 10]:
                ix = [
                    i
                    for i, r in enumerate(rows)
                    if r["probe"] == "duration"
                    and r["language"] == language
                    and r["seconds"] == seconds
                ]
                if not ix:
                    continue
                duration.append(
                    {
                        "model": method,
                        "language": language,
                        "seconds": seconds,
                        "n": len(ix),
                        "in_set": language in codes,
                        "accuracy": float((predicted[ix] == language).mean())
                        if language in codes
                        else None,
                        "english": int((predicted[ix] == "en").sum()),
                        "mean_confidence": float(confidence[ix].mean()),
                        "niu_to_ton_rate": float((predicted[ix] == "ton").mean())
                        if language == "niu" and "ton" in codes
                        else None,
                        "ton_to_niu_rate": float((predicted[ix] == "niu").mean())
                        if language == "ton" and "niu" in codes
                        else None,
                        "prediction_counts": json.dumps(
                            dict(Counter(predicted[ix].tolist()))
                        ),
                    }
                )
        baseline = {
            r["parent_id"]: i
            for i, r in enumerate(rows)
            if r["probe"] == "gain" and r["gain_db"] == 0
        }
        for language in CODES:
            for db in [-12, 12]:
                ix = [
                    i
                    for i, r in enumerate(rows)
                    if r["probe"] == "gain"
                    and r["language"] == language
                    and r["gain_db"] == db
                ]
                before = [baseline[rows[i]["parent_id"]] for i in ix]
                gain.append(
                    {
                        "model": method,
                        "language": language,
                        "gain_db": db,
                        "n": len(ix),
                        "in_set": language in codes,
                        "changed_predictions": int(
                            (predicted[ix] != predicted[before]).sum()
                        ),
                        "accuracy_before": float((predicted[before] == language).mean())
                        if language in codes
                        else None,
                        "accuracy_after": float((predicted[ix] == language).mean())
                        if language in codes
                        else None,
                        "mean_applied_gain_db": float(
                            np.mean([rows[i]["gain_db_applied"] for i in ix])
                        ),
                        "mean_absolute_confidence_change": float(
                            np.abs(confidence[ix] - confidence[before]).mean()
                        ),
                    }
                )
        predictions_table.extend(
            {
                "model": method,
                "id": r["id"],
                "parent_id": r["parent_id"],
                "language": r["language"],
                "probe": r["probe"],
                "seconds": r["seconds"],
                "gain_db": r["gain_db"],
                "prediction": predicted[i],
                "confidence": float(confidence[i]),
            }
            for i, r in enumerate(rows)
        )
    write_csv(directory / "duration.csv", duration)
    write_csv(directory / "gain.csv", gain)
    write_csv(directory / "predictions.csv", predictions_table)
    write_json(
        directory / "complete.json",
        {
            "models": list(models),
            "rows": len(rows),
            "rows_sha256": file_hash(directory / "rows.jsonl"),
            "gain_only_on_selection": True,
            "positive_gain_capped_to_avoid_clipping": True,
        },
    )
