"""Full-corpus provisional speaker labels and paired LID reruns."""

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "extract", "split", "train", "evaluate", "report"],
        default="all",
    )
    parser.add_argument("--root", default="artifacts/speaker_split")
    parser.add_argument("--config", default="configs/speaker_split.json")
    args = parser.parse_args()
    if not os.environ.get("LID_SPEAKER_SPLIT_GUARDED"):
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "--unit=lid-speaker-split",
            "-p",
            "MemoryMax=4G",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "CPUQuota=200%",
            "-p",
            "TasksMax=96",
            "env",
            "LID_SPEAKER_SPLIT_GUARDED=1",
            "LID_MEMORY_GUARDED=1",
            "PYTHONUTF8=1",
            "HF_HUB_DISABLE_XET=1",
            "OMP_NUM_THREADS=2",
            "OPENBLAS_NUM_THREADS=2",
            "MKL_NUM_THREADS=2",
            "NUMBA_NUM_THREADS=2",
            str(PROJECT / ".venv/bin/python"),
            "-u",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ]
        raise SystemExit(subprocess.call(command))
    from lid.speaker_split import Study

    study = Study(args.config, args.root)
    with open("/tmp/lid-speaker-pilot-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = (
            ["extract", "split", "train", "report"]
            if args.stage == "all"
            else [args.stage]
        )
        for stage in stages:
            study.status(stage)
            try:
                if stage == "extract":
                    # Pyannote remains isolated from the existing LID environment.
                    subprocess.run(
                        [
                            str(
                                PROJECT / "environments/speaker_pilot/.venv/bin/python"
                            ),
                            "-u",
                            "-c",
                            "from lid.speaker_split import Study; import sys; Study(sys.argv[1], sys.argv[2]).extract()",
                            args.config,
                            args.root,
                        ],
                        check=True,
                    )
                else:
                    getattr(study, stage)()
            except Exception as error:
                study.status("failed", stage=stage, error=repr(error))
                raise
        study.status("complete", stages=stages)


if __name__ == "__main__":
    main()
