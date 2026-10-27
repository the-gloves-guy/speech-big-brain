"""Read-only audit and recording-group splits for the three-language extension."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from .languages import CODES as OLD_CODES
from .languages import NAMES as OLD_NAMES

NEW = (("Samoan", "sam"), ("Niuean", "niu"), ("Tongan", "ton"))
CODES = [*OLD_CODES, *(code for _, code in NEW)]
NAMES = [*OLD_NAMES, *(name for name, _ in NEW)]
ROOT = Path("/home/the-gloves-guy/lid-finetuning/data-scraping/pipeline")
RUN = Path("artifacts/extension15")
BASE = Path("artifacts/full_run")
REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def status(phase, **details):
    from datetime import UTC, datetime

    payload = {"phase": phase, "updated_at": datetime.now(UTC).isoformat(), **details}
    write_json(RUN / "status.json", payload)
    print(json.dumps(payload), flush=True)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_snapshot(row):
    if "bytes" in row:
        stat = Path(row["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
            raise ValueError(f"Source audio changed since audit: {row['path']}")


def split_for_group(group, seed=42):
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    return (
        "train"
        if value < 0.8
        else "selection"
        if value < 0.85
        else "calibration"
        if value < 0.9
        else "test"
    )


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    marker = RUN / "data_ready.json"
    if marker.exists():
        return json.loads(marker.read_text())
    inventory_path = RUN / "inventory.jsonl"
    if not inventory_path.exists():
        paths = [
            (name, code, path)
            for name, code in NEW
            for path in sorted((ROOT / name / "segments").glob("*.wav"))
        ]
        if not paths or any(
            not any(code == item[1] for item in paths) for _, code in NEW
        ):
            raise ValueError("Missing source recordings for a new language")
        # Freeze the inventory before auditing; subsequent scraper output is not included.
        with inventory_path.open("w") as stream:
            for name, code, path in paths:
                stat = path.stat()
                stream.write(
                    json.dumps(
                        {
                            "path": str(path),
                            "language": code,
                            "bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                        }
                    )
                    + "\n"
                )
    inventory = read_rows(inventory_path)
    journal = RUN / "audit.jsonl"
    audited = (
        {row["path"]: row for row in read_rows(journal)} if journal.exists() else {}
    )
    with journal.open("a", buffering=1) as stream:
        for number, item in enumerate(inventory, 1):
            check_snapshot(item)
            if item["path"] in audited:
                continue
            row = dict(item)
            try:
                match = re.fullmatch(
                    r"(.+)_start(\d+)_dur(\d+)", Path(row["path"]).stem
                )
                if not match:
                    raise ValueError("Unrecognized source-recording filename")
                source = match[1]
                source_hash = re.search(r"(?:^|_)([a-f0-9]{12})$", source)
                row["group"] = "new:" + (source_hash[1] if source_hash else source)
                row["sha256"] = file_hash(row["path"])
                energy, count, peak, clipped = 0.0, 0, 0.0, 0
                with sf.SoundFile(row["path"]) as audio:
                    row.update(
                        sample_rate=audio.samplerate,
                        channels=audio.channels,
                        duration=len(audio) / audio.samplerate,
                    )
                    for block in audio.blocks(
                        blocksize=16000, dtype="float32", always_2d=True
                    ):
                        signal = block.mean(axis=1)
                        if not np.isfinite(signal).all():
                            raise ValueError("Nonfinite audio")
                        energy += float(
                            np.dot(signal.astype(np.float64), signal.astype(np.float64))
                        )
                        count += len(signal)
                        peak = max(peak, float(np.abs(signal).max()))
                        clipped += int((np.abs(signal) >= 0.999).sum())
                if count == 0 or row["duration"] < 0.1 or peak < 1e-6:
                    raise ValueError("Empty, silent, or too short")
                row.update(
                    rms_dbfs=20 * np.log10(max((energy / count) ** 0.5, 1e-12)),
                    peak=peak,
                    clipped_fraction=clipped / count,
                )
                check_snapshot(item)
            except (ValueError, RuntimeError, sf.LibsndfileError) as error:
                row["excluded"] = str(error)
            stream.write(json.dumps(row) + "\n")
            audited[row["path"]] = row
            if number % 250 == 0:
                status(
                    "audit",
                    completed=number,
                    total=len(inventory),
                    percent=round(number / len(inventory) * 100, 1),
                )
    rows = [audited[item["path"]] for item in inventory]
    # Duplicate-linked source groups must never straddle partitions.
    parent = {}

    def find(group):
        parent.setdefault(group, group)
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    by_hash = {}
    for row in rows:
        if "excluded" in row:
            continue
        find(row["group"])
        by_hash.setdefault(row["sha256"], []).append(row)
    for duplicates in by_hash.values():
        if len({row["language"] for row in duplicates}) > 1:
            for row in duplicates:
                row["excluded"] = "Identical audio with conflicting labels"
        else:
            canonical = min(row["group"] for row in duplicates)
            for row in duplicates:
                parent[find(row["group"])] = find(canonical)
            for row in duplicates[1:]:
                row["excluded"] = "Exact duplicate"
    partitions = {key: [] for key in ("train", "selection", "calibration", "test")}
    for row in rows:
        if "excluded" not in row:
            row["group"] = find(row["group"])
            row["id"] = row["path"]
            partitions[split_for_group(row["group"])].append(row)
    for split, items in partitions.items():
        if {row["language"] for row in items} != {code for _, code in NEW}:
            raise ValueError(f"Missing language in {split}")
        with (RUN / f"new_{split}.jsonl").open("w") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in items)
    write_json(RUN / "quarantine.json", [row for row in rows if "excluded" in row])
    summary = {
        "codes": CODES,
        "inventory_sha256": file_hash(inventory_path),
        "original_files": len(rows),
        "excluded": sum("excluded" in row for row in rows),
        "splits": {
            split: {
                "clips": dict(Counter(row["language"] for row in items)),
                "source_groups": {
                    code: len(
                        {row["group"] for row in items if row["language"] == code}
                    )
                    for _, code in NEW
                },
            }
            for split, items in partitions.items()
        },
        "levels": {
            code: {
                "rms_dbfs_p10_p50_p90": np.percentile(
                    [
                        row["rms_dbfs"]
                        for row in rows
                        if row["language"] == code and "excluded" not in row
                    ],
                    [10, 50, 90],
                ).tolist()
            }
            for _, code in NEW
        },
    }
    write_json(marker, summary)
    status("data_ready", **summary)
    return summary
