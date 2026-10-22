"""Use the idle GPU as training languages finish downloading and extracting."""

import json
import re
import time
from argparse import Namespace
from pathlib import Path

from lid.cache import extract
from lid.languages import CODES

ROOT = Path("data/voxlingua107")
OUTPUT = Path("artifacts/full_run/prefetch")
MODEL = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"


def state(phase):
    payload = {"phase": phase, "model_revision": MODEL, "max_seconds": 10.0}
    temporary = OUTPUT / "session.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(OUTPUT / "session.json")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state("active")
    try:
        while True:
            for code in CODES:
                output = OUTPUT / f"{code}.pt"
                if output.exists() or not (ROOT / f"{code}.extracted.json").exists():
                    continue
                rows = []
                for path in sorted((ROOT / f"train/{code}").rglob("*.wav")):
                    match = re.match(r"^([A-Za-z0-9_-]{11})(?:[_.-]|$)", path.stem)
                    if not match:
                        raise ValueError(f"Cannot identify source video: {path}")
                    rows.append(
                        {
                            "path": str(path.resolve()),
                            "language": code,
                            "group": match[1],
                        }
                    )
                manifest = OUTPUT / f"{code}.jsonl"
                manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
                print(
                    f"Prefetching {code}: {len(rows)} frozen embeddings/logits",
                    flush=True,
                )
                extract(
                    Namespace(
                        manifest=str(manifest),
                        output=str(output),
                        split="all",
                        limit_per_language=0,
                        max_seconds=10.0,
                        device="cuda",
                        threads=2,
                        batch_size=16,
                        buffer_size=32,
                        model_revision=MODEL,
                        allow_partial_languages=True,
                    )
                )
            if all((OUTPUT / f"{code}.pt").exists() for code in CODES):
                state("complete")
                print("All frozen per-language feature caches are complete", flush=True)
                return
            time.sleep(15)
    except BaseException:
        state("failed")
        raise


if __name__ == "__main__":
    main()
