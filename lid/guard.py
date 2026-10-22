"""Kernel-enforced limits for a job, without changing desktop-wide settings."""

import argparse
import subprocess
import sys
import uuid


def launch(command, memory="2G", unit=None):
    unit = unit or f"lid-{uuid.uuid4().hex[:10]}"
    result = subprocess.run(
        [
            "systemd-run",
            "--user",
            "--scope",
            f"--unit={unit}",
            "-p",
            f"MemoryMax={memory}",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "CPUWeight=10",
            "-p",
            "IOWeight=10",
            "env",
            "LID_MEMORY_GUARDED=1",
            "HF_HUB_DISABLE_XET=1",
            "HF_HUB_DOWNLOAD_TIMEOUT=60",
            *command,
        ],
        check=False,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", default="3G")
    parser.add_argument("--unit")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --")
    sys.exit(launch(command, args.memory, args.unit))


if __name__ == "__main__":
    main()
