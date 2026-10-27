"""Matched duration and validation-only gain probes for six twelve-way methods."""

import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.special import softmax

from .data import waveform
from .extension_data import file_hash, read_rows, write_json
from .languages import CODES, selected_indices
from .model import load_encoder
from .speaker_pilot import bind, digest
from .speaker_study import write_csv
from .whisper_model import (
    Adapter,
    load_whisper,
    mel_batch,
    native_logits,
    selected_languages,
)

METHODS = [
    "whisper_filtered",
    "whisper_frozen",
    "whisper_finetuned",
    "speechbrain_filtered",
    "speechbrain_frozen",
    "speechbrain_finetuned",
]


def prepare_rows(study, directory, codes=None, selection_cohorts=None):
    codes = CODES if codes is None else codes
    selection_cohorts = (
        ["vox_validation"] if selection_cohorts is None else selection_cohorts
    )
    identity = {
        "config": study.config,
        "data_sha256": file_hash(study.root / "data_ready.json"),
    }
    bind(directory / "identity.json", identity)
    manifest = directory / "rows.jsonl"
    if manifest.exists():
        rows = read_rows(manifest)
        if any(file_hash(r["path"]) != r["sha256"] for r in rows):
            raise ValueError("Diagnostic waveform changed")
        return rows
    audio = directory / "audio"
    audio.mkdir(exist_ok=True)
    external = Path(study.config["external"])
    prepared = json.loads((external / "segments_ready.json").read_text())
    rows = []
    for source in prepared["source_summaries"]:
        saved = json.loads((external / f"{source['source_id']}.json").read_text())
        with sf.SoundFile(
            external / "decoded" / f"{source['source_id']}.wav"
        ) as stream:
            if stream.samplerate != 16000 or stream.channels != 1:
                raise ValueError("Duration source protocol differs")
            for interval in saved["vad_intervals_samples"]:
                left, right = interval["start"], interval["end"]
                n = (right - left) // 160000
                offset = (right - left - n * 160000) // 2
                for index in range(n):
                    start = left + offset + index * 160000
                    stream.seek(start)
                    signal = stream.read(160000, dtype="float32")
                    parent = f"{source['source_id']}-{start}"
                    for seconds in [3, 5, 10]:
                        length = seconds * 16000
                        crop = (160000 - length) // 2
                        identifier = f"{parent}-{seconds}s"
                        path = (audio / f"{identifier}.wav").resolve()
                        sf.write(
                            path, signal[crop : crop + length], 16000, subtype="FLOAT"
                        )
                        rows.append(
                            {
                                "id": identifier,
                                "parent_id": parent,
                                "path": str(path),
                                "group": source["source_id"],
                                "source_id": source["source_id"],
                                "language": source["language"],
                                "probe": "duration",
                                "seconds": seconds,
                                "gain_db": 0,
                                "sha256": file_hash(path),
                            }
                        )
    selection = [row for cohort in selection_cohorts for row in study.rows(cohort)]
    for code in codes:
        pool = sorted(
            [r for r in selection if r["language"] == code],
            key=lambda r: digest([study.config["seed"], r["id"]]),
        )[:12]
        for row in pool:
            signal = waveform(row, 10)
            for gain in [0, -12, 12]:
                factor = 10 ** (gain / 20)
                if gain > 0:
                    factor = min(factor, 0.999 / max(float(np.abs(signal).max()), 1e-8))
                identifier = f"gain-{digest(row['id'])[:20]}-{gain}db"
                path = (audio / f"{identifier}.wav").resolve()
                sf.write(path, signal * factor, 16000, subtype="FLOAT")
                rows.append(
                    {
                        "id": identifier,
                        "parent_id": row["id"],
                        "path": str(path),
                        "group": row["group"],
                        "source_id": row["group"],
                        "language": code,
                        "probe": "gain",
                        "seconds": len(signal) / 16000,
                        "gain_db": gain,
                        "gain_db_applied": float(20 * np.log10(factor)),
                        "sha256": file_hash(path),
                    }
                )
    with manifest.open("w") as stream:
        stream.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    return rows


def score_method(study, method, rows, path):
    if method.startswith("whisper"):
        checkpoint = (
            study.root / method / "model.pt"
            if method != "whisper_filtered"
            else study.root / "pretrained/tiny.pt"
        )
    else:
        checkpoint = (
            study.root / method / "model.pt"
            if method == "speechbrain_finetuned"
            else Path("artifacts/full_run/head/head.pt")
        )
    identity = {
        "config": study.config,
        "rows": digest(rows),
        "method": method,
        "weights": file_hash(checkpoint),
    }
    if path.exists():
        with np.load(path) as saved:
            if saved["identity"].item() != digest(identity) or saved[
                "ids"
            ].tolist() != [r["id"] for r in rows]:
                raise ValueError("Diagnostic score cache changed")
            return saved["logits"]
    if method == "whisper_filtered":
        model = load_whisper(study.root)
        adapter = None
    elif method == "speechbrain_filtered":
        from .extension_data import REVISION

        model, labels, _ = load_encoder("cuda", REVISION)
        indices = selected_indices(labels)
        adapter = None
    else:
        family = "whisper" if method.startswith("whisper") else "speechbrain"
        saved = (
            study.root / method / "model.pt" if method != "speechbrain_frozen" else None
        )
        adapter = Adapter(family, study.root, saved)
        model = None
    parts = path.with_suffix(".parts")
    parts.mkdir(exist_ok=True)
    bind(parts / "identity.json", identity)
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            target = parts / f"part-{start:06d}.npy"
            if target.exists():
                scores = np.load(target)
            else:
                batch = []
                for row in rows[start : start + 64]:
                    if adapter is not None:
                        scores = adapter.scores([row])
                    elif method == "whisper_filtered":
                        mels, _ = mel_batch([row])
                        encoded = model.encoder(mels)
                        logits, codes = native_logits(model, encoded)
                        scores = logits[:, selected_languages(codes)]
                    else:
                        signal = (
                            torch.from_numpy(waveform(row, 10)).unsqueeze(0).to("cuda")
                        )
                        embeddings = model.encode_batch(signal, normalize=False)
                        scores = model.mods.classifier(embeddings).reshape(1, -1)[
                            :, indices
                        ]
                    batch.append(scores.float().cpu().numpy())
                scores = np.concatenate(batch)
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite diagnostic score")
                np.save(target, scores)
                study.status(
                    "diagnostics",
                    method=method,
                    completed=start + len(scores),
                    total=len(rows),
                )
            values.append(scores)
    values = np.concatenate(values)
    np.savez_compressed(
        path, ids=[r["id"] for r in rows], logits=values, identity=digest(identity)
    )
    if adapter is not None:
        adapter.release()
    del model, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return values


def run(study):
    directory = study.root / "diagnostics"
    directory.mkdir(exist_ok=True)
    rows = prepare_rows(study, directory)
    duration, gain, prediction_rows = [], [], []
    for method in METHODS:
        scores = score_method(study, method, rows, directory / f"{method}.npz")
        prediction = np.array([CODES[i] for i in scores.argmax(axis=1)])
        if method == "speechbrain_filtered":
            temperature = json.loads(
                Path("artifacts/full_run/comparison/calibration.json").read_text()
            )["filtered_temperature"]
        elif method == "speechbrain_frozen":
            temperature = json.loads(
                Path("artifacts/full_run/comparison/calibration.json").read_text()
            )["head_temperature"]
        else:
            temperature = json.loads(
                (study.root / method / "calibration.json").read_text()
            )["temperature"]
        confidence = softmax(scores.astype(np.float64) / temperature, axis=1).max(
            axis=1
        )
        for language in ["hi", "th", "sam", "niu", "ton"]:
            for seconds in [3, 5, 10]:
                indices = [
                    i
                    for i, r in enumerate(rows)
                    if r["probe"] == "duration"
                    and r["language"] == language
                    and r["seconds"] == seconds
                ]
                if not indices:
                    continue
                duration.append(
                    {
                        "method": method,
                        "language": language,
                        "seconds": seconds,
                        "n": len(indices),
                        "in_set": language in CODES,
                        "accuracy_all": float((prediction[indices] == language).mean())
                        if language in CODES
                        else None,
                        "english_predictions": int((prediction[indices] == "en").sum()),
                        "mean_confidence": float(confidence[indices].mean()),
                        "prediction_counts": json.dumps(
                            dict(Counter(prediction[indices].tolist()))
                        ),
                    }
                )
        baseline = {
            r["parent_id"]: i
            for i, r in enumerate(rows)
            if r["probe"] == "gain" and r["gain_db"] == 0
        }
        for code in CODES:
            for db in [-12, 12]:
                indices = [
                    i
                    for i, r in enumerate(rows)
                    if r["probe"] == "gain"
                    and r["language"] == code
                    and r["gain_db"] == db
                ]
                gain.append(
                    {
                        "method": method,
                        "language": code,
                        "gain_db": db,
                        "n": len(indices),
                        "changed_predictions": sum(
                            prediction[i] != prediction[baseline[rows[i]["parent_id"]]]
                            for i in indices
                        ),
                        "mean_absolute_confidence_change": float(
                            np.mean(
                                [
                                    abs(
                                        confidence[i]
                                        - confidence[baseline[rows[i]["parent_id"]]]
                                    )
                                    for i in indices
                                ]
                            )
                        ),
                    }
                )
        prediction_rows.extend(
            {
                "method": method,
                "id": r["id"],
                "parent_id": r["parent_id"],
                "language": r["language"],
                "probe": r["probe"],
                "seconds": r["seconds"],
                "gain_db": r["gain_db"],
                "prediction": prediction[i],
                "confidence": float(confidence[i]),
            }
            for i, r in enumerate(rows)
        )
    write_csv(directory / "duration.csv", duration)
    write_csv(directory / "gain.csv", gain)
    write_csv(directory / "predictions.csv", prediction_rows)
    write_json(
        directory / "complete.json",
        {
            "methods": METHODS,
            "diagnostic_cuts": len(rows),
            "rows_sha256": file_hash(directory / "rows.jsonl"),
        },
    )
