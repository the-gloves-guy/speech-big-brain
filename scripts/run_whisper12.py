"""Resumable Whisper/SpeechBrain twelve-language study with enforced resource limits."""

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
STAGES = [
    "prepare",
    "cache_eval",
    "filtered",
    "cache_train",
    "frozen",
    "whisper_finetune",
    "speechbrain_finetune",
    "diagnostics",
    "report",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--root", default="artifacts/whisper_tiny12")
    parser.add_argument("--config", default="configs/whisper12.json")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("LID_WHISPER_GUARDED"):
        Path(args.root).mkdir(parents=True, exist_ok=True)
        command = [
            "systemd-run",
            "--user",
            "--unit=lid-whisper12",
            f"--working-directory={PROJECT}",
            "-p",
            "MemoryMax=4G",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "CPUQuota=200%",
            "-p",
            "TasksMax=96",
        ]
        if args.background:
            log = (PROJECT / args.root / "pipeline.log").resolve()
            command += [
                "-p",
                "Restart=on-failure",
                "-p",
                "RestartSec=30",
                "-p",
                "StartLimitIntervalSec=3600",
                "-p",
                "StartLimitBurst=3",
                "-p",
                f"StandardOutput=append:{log}",
                "-p",
                f"StandardError=append:{log}",
            ]
        else:
            command += ["--scope"]
        command += [
            "env",
            "LID_WHISPER_GUARDED=1",
            "PYTHONUTF8=1",
            "HF_HUB_DISABLE_XET=1",
            "OMP_NUM_THREADS=2",
            "OPENBLAS_NUM_THREADS=2",
            "MKL_NUM_THREADS=2",
            "NUMBA_NUM_THREADS=2",
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--root",
            args.root,
            "--config",
            args.config,
            "--stage",
            args.stage,
        ]
        raise SystemExit(subprocess.call(command))
    import torch

    from lid.extension_data import write_json
    from lid.speaker_study import resources
    from lid.whisper_study import Study

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    study = Study(args.config, args.root)
    with open("/tmp/lid-speaker-pilot-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for stage in STAGES if args.stage == "all" else [args.stage]:
            study.status(stage)
            try:
                getattr(study, stage)()
                write_json(study.root / f"resources_{stage}.json", resources())
            except Exception as error:
                study.status("failed", stage=stage, error=repr(error))
                raise
        study.status("complete", stages=STAGES if args.stage == "all" else [args.stage])


if __name__ == "__main__":
    main()
