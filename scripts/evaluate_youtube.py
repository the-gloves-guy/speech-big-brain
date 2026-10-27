"""Evaluate the existing models on VAD-filtered held-out YouTube clips."""

import argparse
import fcntl
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="validation_clips")
    parser.add_argument("--output", default="artifacts/youtube_validation")
    parser.add_argument("--min-seconds", type=float, default=3.0)
    parser.add_argument(
        "--languages", nargs="+", choices=["sam", "niu", "ton", "hi", "th"]
    )
    args = parser.parse_args()
    if not 0 < args.min_seconds <= 5:
        parser.error(
            "Minimum speech duration must be in (0, 5] seconds for 10-second chunks"
        )
    if not os.environ.get("LID_MEMORY_GUARDED"):
        from lid.guard import launch

        raise SystemExit(
            launch(
                [sys.executable, "-u", __file__, *sys.argv[1:]],
                memory="3G",
                unit="lid-youtube-evaluation",
            )
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    from lid.youtube_evaluate import evaluate

    torch.set_num_threads(2)
    try:
        with open("/tmp/lid-speaker-pilot-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            evaluate(args.root, args.output, args.min_seconds, languages=args.languages)
    except BaseException as error:
        from pathlib import Path

        from lid.extension_data import write_json

        write_json(
            Path(args.output) / "status.json", {"phase": "failed", "error": str(error)}
        )
        raise


if __name__ == "__main__":
    main()
