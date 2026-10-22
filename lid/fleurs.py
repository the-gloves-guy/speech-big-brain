"""FLEURS archives on disk; no Parquet tables or in-memory audio datasets."""

import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from .languages import LANGUAGES

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def download_file(relative, revision, root):
    from huggingface_hub import hf_hub_download

    target = root / relative
    if target.is_file():
        return target
    return Path(
        hf_hub_download(
            "google/fleurs",
            relative,
            repo_type="dataset",
            revision=revision,
            local_dir=root,
            token=False,
        )
    )


def prepare_split(config, split, revision, root):
    source_split = "dev" if split == "validation" else split
    root = Path(root) / revision
    metadata = download_file(f"data/{config}/{source_split}.tsv", revision, root)
    entries = []
    with metadata.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if (
                len(row) < 2
                or Path(row[1]).name != row[1]
                or not row[1].endswith(".wav")
            ):
                raise ValueError(f"Invalid FLEURS metadata row in {metadata}")
            entries.append((row[0], row[1]))
    expected = {name for _, name in entries}
    if not entries or len(expected) != len(entries):
        raise ValueError(f"Empty or duplicate FLEURS filenames in {metadata}")
    audio_dir = root / f"audio/{config}/{source_split}"
    marker = audio_dir / "complete.json"
    if not marker.exists():
        archive = download_file(
            f"data/{config}/audio/{source_split}.tar.gz", revision, root
        )
        audio_dir.mkdir(parents=True, exist_ok=True)
        extracted = set()
        # Streaming tar reads and fixed-size copies bound memory independently of archive size.
        with tarfile.open(archive, mode="r|gz") as source:
            for member in source:
                name = Path(member.name).name
                if not member.isfile() or name not in expected:
                    continue
                if name in extracted:
                    raise ValueError(f"Duplicate FLEURS archive entry: {name}")
                target = audio_dir / name
                temporary = target.with_suffix(".wav.tmp")
                with (
                    source.extractfile(member) as audio,
                    temporary.open("wb") as output,
                ):
                    shutil.copyfileobj(audio, output, length=64 * 1024)
                temporary.replace(target)
                extracted.add(name)
        if extracted != expected:
            raise ValueError(
                f"FLEURS archive is missing {len(expected - extracted)} recordings"
            )
        with archive.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        payload = {
            "config": config,
            "split": split,
            "revision": revision,
            "count": len(entries),
            "archive_sha256": digest,
            "archive_bytes": archive.stat().st_size,
        }
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(marker)
    return entries, audio_dir


def records(split, revision, limit=0, root="data/fleurs"):
    for _, code, config in LANGUAGES:
        print(f"Preparing disk-backed FLEURS {config}/{split}", flush=True)
        entries, directory = prepare_split(config, split, revision, root)
        for utterance_id, name in entries[: limit or None]:
            path = directory / name
            if not path.is_file():
                raise ValueError(f"Missing FLEURS audio file: {path}")
            yield {
                "path": str(path),
                "language": code,
                "id": f"{config}/{split}/{name}",
                "group": f"{config}/{split}/{utterance_id}",
            }
