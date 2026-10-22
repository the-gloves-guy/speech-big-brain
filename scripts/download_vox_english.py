"""Fetch all English training shards from the dataset publisher's official mirror."""

import concurrent.futures
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "TalTechNLP/voxlingua107_wds"
REVISION = "8dfcfa5b4aa77ac630c566a997c1af095088a36b"
ROOT = Path("data/voxlingua107")
MIRROR = Path("data/voxlingua107_wds")


def shard(file):
    marker = MIRROR / (file.rfilename + ".verified.json")
    if marker.exists():
        return json.loads(marker.read_text())
    for attempt in range(20):
        try:
            path = Path(
                hf_hub_download(
                    REPO,
                    file.rfilename,
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=MIRROR,
                    token=False,
                )
            )
            break
        except Exception as exc:
            print(
                f"English {file.rfilename}: retry {attempt + 1} ({type(exc).__name__})",
                flush=True,
            )
            if attempt == 19:
                raise
            time.sleep(min(30, 2 ** min(attempt, 5)))
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if file.lfs is None or digest != file.lfs.sha256:
        raise ValueError(f"Mirror shard hash mismatch: {file.rfilename}")
    destination = ROOT / "train/en"
    destination.mkdir(parents=True, exist_ok=True)
    count, size = 0, 0
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".wav"):
                continue
            if "/" in member.name or "\\" in member.name:
                raise ValueError(f"Unexpected shard member: {member.name}")
            with (
                archive.extractfile(member) as source,
                (destination / member.name).open("wb") as output,
            ):
                shutil.copyfileobj(source, output)
            count += 1
            size += member.size
    result = {
        "file": file.rfilename,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "clips": count,
        "audio_bytes": size,
    }
    marker.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"English mirror: verified/extracted {file.rfilename}, {count} clips",
        flush=True,
    )
    return result


def main():
    marker = ROOT / "en.extracted.json"
    if marker.exists():
        print("English already extracted", flush=True)
        return
    info = HfApi(token=False).dataset_info(REPO, revision=REVISION, files_metadata=True)
    files = [
        file
        for file in info.siblings
        if file.rfilename.startswith("train/en/") and file.rfilename.endswith(".tar")
    ]
    if not files:
        raise ValueError("No English mirror shards found")
    print(
        f"English mirror: {len(files)} shards, {sum(file.size for file in files) / 1e9:.2f} GB",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(shard, files))
    marker.write_text(
        json.dumps(
            {
                "source": REPO,
                "revision": REVISION,
                "clips": sum(row["clips"] for row in results),
                "uncompressed_bytes": sum(row["audio_bytes"] for row in results),
                "sha256_checked": True,
                "shards": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "All English training shards downloaded, hash-verified, and extracted",
        flush=True,
    )


if __name__ == "__main__":
    main()
