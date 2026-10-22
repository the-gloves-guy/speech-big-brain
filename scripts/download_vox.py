"""Download and CRC-check the requested official VoxLingua107 training archives.

Resumable downloads and per-language completion markers allow safe restarts.
"""

import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CODES = ["ar", "en", "es", "fa", "hi", "id", "mi", "ps", "ru", "th", "ur", "zh"]
BASE = "https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/"
ROOT = Path("data/voxlingua107")


def download(code):
    if code == "en":
        subprocess.run(
            [sys.executable, "-u", "scripts/download_vox_english.py"],
            env={
                **os.environ,
                "HF_HUB_DISABLE_XET": "1",
                "HF_HUB_DOWNLOAD_TIMEOUT": "60",
            },
            check=True,
        )
        return None
    archives = ROOT / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    destination = archives / f"{code}.zip"
    partial = destination.with_suffix(".zip.part")
    marker = archives / f"{code}.json"
    url = BASE + f"{code}.zip"
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=8, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
            )
        ),
    )
    head = session.head(url, timeout=(20, 30), allow_redirects=True)
    head.raise_for_status()
    expected = int(head.headers["Content-Length"])
    if (
        destination.exists()
        and destination.stat().st_size == expected
        and marker.exists()
    ):
        print(f"Download already complete: {code}", flush=True)
        return destination
    started = reported = time.monotonic()
    initial_offset = partial.stat().st_size if partial.exists() else 0
    failures = 0
    while not destination.exists():
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == expected:
                partial.replace(destination)
                break
            end = min(offset + 64 * 1024**2, expected) - 1
            headers = {"Range": f"bytes={offset}-{end}"}
            downloaded = offset
            with session.get(
                url, headers=headers, stream=True, timeout=(20, 45)
            ) as response:
                response.raise_for_status()
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range")
                    != f"bytes {offset}-{end}/{expected}"
                ):
                    raise ValueError(f"Invalid resume response for {code}")
                with partial.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - reported >= 30:
                            speed = (
                                (downloaded - initial_offset) / (now - started) / 1e6
                            )
                            print(
                                f"Downloading {code}: {downloaded / 1e9:.2f}/{expected / 1e9:.2f} GB ({speed:.1f} MB/s)",
                                flush=True,
                            )
                            reported = now
            if partial.stat().st_size != end + 1:
                raise ValueError(f"Incomplete range response: {code}")
            failures = 0
        except (requests.RequestException, OSError, ValueError) as exc:
            failures += 1
            print(f"Download {code}, consecutive failure {failures}: {exc}", flush=True)
            if failures >= 20:
                raise
            time.sleep(min(30, 2 ** min(failures, 5)))
    with destination.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    marker.write_text(
        json.dumps(
            {
                "url": url,
                "bytes": expected,
                "sha256": digest,
                "etag": head.headers.get("ETag"),
                "last_modified": head.headers.get("Last-Modified"),
                "downloaded_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Downloaded {code}: {expected / 1e9:.2f} GB, SHA256={digest}", flush=True)
    return destination


def extract(archive):
    with (ROOT / f".{archive.stem}.extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _extract(archive)


def _extract(archive):
    code = archive.stem
    destination = (ROOT / "train").resolve()
    marker = ROOT / f"{code}.extracted.json"
    if marker.exists():
        print(f"Extraction already complete: {code}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        required = sum(item.file_size for item in members)
        if shutil.disk_usage(destination).free < required + 10 * 1024**3:
            raise OSError(
                f"Insufficient disk space to extract {code}: needs {required / 1e9:.1f} GB plus 10 GiB reserve"
            )
        for member in members:
            target = (destination / member.filename).resolve()
            if (
                not target.is_relative_to(destination)
                or member.filename.split("/")[0] != code
            ):
                raise ValueError(f"Unexpected archive path: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unexpected symlink: {member.filename}")
            # ZipFile extraction checks each member's CRC while reading it.
            source.extract(member, destination)
    marker.write_text(
        json.dumps(
            {
                "archive": str(archive),
                "entries": len(members),
                "uncompressed_bytes": required,
                "crc_checked": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Extracted and CRC-checked {code}: {len(members)} entries, {required / 1e9:.2f} GB",
        flush=True,
    )


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download, code) for code in CODES]
        # Extract one language at a time while other archives continue downloading.
        for future in concurrent.futures.as_completed(futures):
            archive = future.result()
            if archive is not None:
                if "--download-only" not in sys.argv:
                    extract(archive)
                else:
                    print(f"Archive ready for extraction: {archive}", flush=True)
    print("All twelve VoxLingua training language downloads are complete.", flush=True)


if __name__ == "__main__":
    main()
