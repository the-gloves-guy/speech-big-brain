"""Separate matched speaker pilot; always enter a kernel-limited scope."""

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "prepare",
            "validate",
            "sample",
            "ecapa",
            "wespeaker",
            "verify",
            "analyze",
            "report",
        ],
        default="all",
    )
    parser.add_argument("--config", default="configs/speaker_pilot.json")
    parser.add_argument("--root", default="artifacts/speaker_pilot")
    args = parser.parse_args()
    if not os.environ.get("LID_PILOT_GUARDED"):
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "--unit=lid-speaker-pilot",
            "-p",
            "MemoryMax=4G",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "CPUQuota=200%",
            "-p",
            "TasksMax=64",
            "env",
            "LID_PILOT_GUARDED=1",
            "PYTHONUTF8=1",
            "OMP_NUM_THREADS=2",
            "OPENBLAS_NUM_THREADS=2",
            "MKL_NUM_THREADS=2",
            "NUMBA_NUM_THREADS=2",
            "HF_HUB_DISABLE_XET=1",
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ]
        raise SystemExit(subprocess.call(command))
    Path(args.root).mkdir(parents=True, exist_ok=True)
    with open("/tmp/lid-speaker-pilot-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from lid.speaker_pilot import Pilot

        pilot = Pilot(args.config, args.root)
        try:
            stages = (
                [
                    "prepare",
                    "validate",
                    "sample",
                    "ecapa",
                    "wespeaker",
                    "verify",
                    "analyze",
                    "report",
                ]
                if args.stage == "all"
                else [args.stage]
            )
            for stage in stages:
                pilot.status(stage)
                from lid.extension_data import write_json
                from lid.speaker_study import resources

                try:
                    getattr(pilot, stage)()
                finally:
                    write_json(pilot.root / f"resources_{stage}.json", resources())
            pilot.status("complete", stages=stages)
        except Exception as error:
            pilot.status("failed", stage=stage, error=repr(error))
            raise


if __name__ == "__main__":
    main()
