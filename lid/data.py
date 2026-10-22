"""Local VoxLingua manifests and streamed, selected FLEURS splits."""

import hashlib
import io
import json
import re
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .languages import CODES


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_manifest(path, require_all_languages=True):
    path = Path(path).resolve()
    records = []
    seen = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["language"] not in CODES or not row.get("group"):
            raise ValueError(
                "Every record needs a supported language and a source-video group"
            )
        audio = Path(row["path"])
        audio = (
            (path.parent / audio).resolve()
            if not audio.is_absolute()
            else audio.resolve()
        )
        if not audio.is_file() or str(audio) in seen:
            raise ValueError(f"Missing or duplicate audio: {audio}")
        seen.add(str(audio))
        row.update(path=str(audio), id=str(audio))
        records.append(row)
    if not records or (
        require_all_languages and {r["language"] for r in records} != set(CODES)
    ):
        raise ValueError("Manifest must contain all 12 languages")
    return records


def split_groups(records, fraction=0.1, seed=42):
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    # Assign globally, ensuring the same video never crosses splits even if mislabeled.
    validation_groups = set()
    for code in CODES:
        groups = {r["group"] for r in records if r["language"] == code}
        if len(groups) < 2:
            raise ValueError(f"Need at least two source-video groups for {code}")
        ordered = sorted(
            groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()
        )
        validation_groups.update(
            ordered[: min(len(groups) - 1, max(1, round(len(groups) * fraction)))]
        )
    train = [r for r in records if r["group"] not in validation_groups]
    valid = [r for r in records if r["group"] in validation_groups]
    for split in (train, valid):
        if {r["language"] for r in split} != set(CODES):
            raise ValueError(
                "Shared video groups left a language empty; supply explicit grouped manifests"
            )
    return train, valid


def prepare_vox(root, output, fraction=0.1, seed=42):
    root, output = Path(root).resolve(), Path(output)
    records = []
    for code in CODES:
        for path in sorted((root / code).rglob("*")):
            if path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
                continue
            # Standard VoxLingua names start with an 11-character YouTube video ID.
            match = re.match(r"^([A-Za-z0-9_-]{11})(?:[_.-]|$)", path.stem)
            if not match:
                raise ValueError(
                    f"Cannot infer video ID from {path}; supply a JSONL manifest with group IDs"
                )
            records.append({"path": str(path), "language": code, "group": match[1]})
    train, valid = split_groups(records, fraction, seed)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", valid)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    print(f"VoxLingua: {len(train)} training / {len(valid)} validation clips")


def fleurs_records(split, revision, limit=0):
    from .fleurs import records

    yield from records(split, revision, limit)


def waveform(row, max_seconds=10.0):
    source = row.get("audio", {})
    source = (
        io.BytesIO(source["bytes"])
        if source.get("bytes") is not None
        else source.get("path", row.get("path"))
    )
    with sf.SoundFile(source) as audio:
        length = len(audio)
        if max_seconds > 0:
            length = min(length, int(max_seconds * audio.samplerate))
        audio.seek(max(0, (len(audio) - length) // 2))
        data = audio.read(length, dtype="float32", always_2d=True).mean(axis=1)
        rate = audio.samplerate
    if not len(data) or not np.isfinite(data).all():
        raise ValueError(f"Empty or nonfinite audio: {row['id']}")
    if rate != 16000:
        divisor = gcd(rate, 16000)
        data = resample_poly(data, 16000 // divisor, rate // divisor).astype(np.float32)
    # ECAPA needs enough frames for temporal convolutions.
    if len(data) < 1600:
        raise ValueError(f"Audio shorter than 0.1 seconds: {row['id']}")
    return data
