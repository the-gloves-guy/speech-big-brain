"""Focused, resumable Small15 study with fresh processes for bounded host memory."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
STAGES = [
    "prepare_model",
    "preflight",
    "warm_cache",
    "warm_train",
    "warm_evaluate",
    "prepare",
    "cache_all",
    "frozen",
    "finetuned",
    "evaluate",
    "baselines",
    "diagnostics",
    "export",
    "report",
]


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def internal(args):
    import torch

    from lid.speaker_study import resources
    from lid.whisper_small import ResourceFailure, SmallStudy

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.internal_stage not in ["prepare_model", "prepare", "baselines", "report"]:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 6.5 * 1024**3 / total))
    study = SmallStudy(args.config, args.root)
    stage = args.internal_stage
    try:
        if stage == "preflight":
            study.preflight(args.candidate, args.checkpointing)
        else:
            getattr(study, stage)()
        write(study.root / f"resources_{stage}.json", resources())
    except (torch.cuda.OutOfMemoryError, ResourceFailure) as error:
        print(f"Resource preflight/step failed: {error}", flush=True)
        return 42
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--root", default="artifacts/whisper_small15")
    parser.add_argument("--config", default="configs/whisper_small15.json")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--internal-stage", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument(
        "--candidate", type=int, choices=[1, 2], default=2, help=argparse.SUPPRESS
    )
    parser.add_argument("--checkpointing", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("LID_WHISPER_SMALL_GUARDED"):
        command = [
            "systemd-run",
            "--user",
            "--unit=lid-whisper-small15",
            f"--working-directory={PROJECT}",
            "-p",
            "MemoryMax=4G",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "CPUQuota=200%",
            "-p",
            "TasksMax=256",
            "-p",
            "OOMPolicy=continue",
        ]
        if args.background:
            log = (PROJECT / root / "pipeline.log").resolve()
            command += [
                "-p",
                f"StandardOutput=append:{log}",
                "-p",
                f"StandardError=append:{log}",
            ]
        else:
            command += ["--scope"]
        command += [
            "env",
            "LID_WHISPER_SMALL_GUARDED=1",
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
        return subprocess.call(command)
    if args.internal_stage:
        return internal(args)

    def status(phase, **details):
        value = {"phase": phase, "updated_at": datetime.now(UTC).isoformat(), **details}
        write(root / "status.json", value)
        print(json.dumps(value), flush=True)

    def child(stage, *extra):
        return subprocess.call(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--root",
                args.root,
                "--config",
                args.config,
                "--internal-stage",
                stage,
                *extra,
            ]
        )

    with open("/tmp/lid-speaker-pilot-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = STAGES if args.stage == "all" else [args.stage]
        for stage in stages:
            status(stage)
            if stage == "preflight":
                marker = root / "preflight.json"
                if marker.exists():
                    previous = json.loads(marker.read_text())
                    if previous["config_sha256"] != checksum(
                        root / "configuration.json"
                    ):
                        raise ValueError(
                            "Existing Small preflight configuration changed"
                        )
                    continue
                attempts = []
                for microbatch, checkpointing in [(2, False), (1, False), (1, True)]:
                    extra = ["--candidate", str(microbatch)] + (
                        ["--checkpointing"] if checkpointing else []
                    )
                    result = child(stage, *extra)
                    attempts.append(
                        {
                            "microbatch": microbatch,
                            "gradient_checkpointing": checkpointing,
                            "returncode": result,
                        }
                    )
                    write(root / "preflight_attempts.json", attempts)
                    if result == 0:
                        break
                    if result not in [42, -9, 137]:
                        status("failed", stage=stage, returncode=result)
                        return result
                else:
                    status(
                        "failed",
                        stage=stage,
                        error="Small cannot fit resource limits after all planned fallbacks",
                    )
                    return 42
            else:
                result = child(stage)
                if result:
                    status("failed", stage=stage, returncode=result)
                    return result
        completion_markers = [
            "frozen/training_complete.json",
            "finetuned/training_complete.json",
            "frozen/evaluation_complete.json",
            "finetuned/evaluation_complete.json",
            "diagnostics/complete.json",
            "export_complete.json",
            "report_complete.json",
        ]
        if args.stage == "all":
            for marker in completion_markers:
                if not (root / marker).exists():
                    raise RuntimeError(f"Incomplete Small study: {marker}")
        complete = args.stage in ["all", "report"] and all(
            (root / marker).exists() for marker in completion_markers
        )
        status(
            "complete" if complete else "stage_complete",
            stages=STAGES if complete else stages,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
