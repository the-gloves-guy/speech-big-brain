"""Print compact experiment progress without loading models or tensor caches."""

import json
import re
import shutil
from pathlib import Path

RUN = Path("artifacts/full_run")
DATA = Path("data/voxlingua107")


def cached(split):
    log = RUN / f"fleurs_direct_{split}.log"
    content = log.read_text(errors="replace") if log.exists() else ""
    saved = re.findall(r"Saved (\d+) examples", content)
    counts = re.findall(r"Cached (\d+) clips", content)
    phases = re.findall(r"Preparing disk-backed FLEURS ([^\n]+)", content)
    result = {
        "complete": (RUN / f"fleurs_direct_{split}.pt").exists(),
        "recordings": int(saved[-1]) if saved else int(counts[-1]) if counts else 0,
        "phase": phases[-1] if phases else "not started",
    }
    memory = RUN / f"fleurs_direct_{split}.memory.txt"
    if memory.exists():
        maximum = re.search(
            r"Maximum resident set size \(kbytes\): (\d+)", memory.read_text()
        )
        if maximum:
            result["peak_process_ram_gib"] = round(int(maximum[1]) / 1024**2, 2)
    return result


def main():
    import sys

    if "--speakers" in sys.argv:
        import subprocess

        directory = Path("artifacts/speaker_study")
        result = {}
        for key, filename in (
            ("current", "status.json"),
            ("audit", "audit_complete.json"),
            ("split_gate", "split_gate.json"),
        ):
            path = directory / filename
            if path.exists():
                result[key] = json.loads(path.read_text())
        result["speaker_shards_completed"] = len(
            list((directory / "speaker_parts").glob("part-*.pt"))
        )
        result["geometry_complete"] = [
            method
            for method in ("frozen", "partial")
            if (directory / f"{method}_geometry_complete.json").exists()
        ]
        result["training_complete"] = [
            str(path.relative_to(directory))
            for path in sorted((directory / "runs").glob("*/*/complete.json"))
        ]
        worker = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                "lid-speaker-study.scope",
                "-p",
                "ActiveState",
                "-p",
                "MemoryCurrent",
                "-p",
                "MemoryMax",
                "-p",
                "MemorySwapMax",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        result["worker"] = dict(
            line.split("=", 1) for line in worker.stdout.splitlines() if "=" in line
        )
        print(json.dumps(result, indent=2))
        return
    if "--extension" in sys.argv:
        import subprocess

        directory = Path("artifacts/extension15")
        result = {}
        for key, name in (
            ("current", "status.json"),
            ("pipeline", "pipeline.json"),
            ("data", "data_ready.json"),
        ):
            if (directory / name).exists():
                result[key] = json.loads((directory / name).read_text())
        result["free_disk_gib"] = round(shutil.disk_usage(".").free / 1024**3, 1)
        workers = {}
        for unit in ("lid-extension15-cache.scope", "lid-extension15.scope"):
            worker = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "-p",
                    "ActiveState",
                    "-p",
                    "MemoryCurrent",
                    "-p",
                    "MemoryMax",
                    "-p",
                    "MemorySwapMax",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            workers[unit] = dict(
                line.split("=", 1) for line in worker.stdout.splitlines() if "=" in line
            )
        result["workers"] = workers
        print(json.dumps(result, indent=2))
        return
    prepared = sorted(path.stem.split(".")[0] for path in DATA.glob("*.extracted.json"))
    archives = sorted(path.stem for path in (DATA / "archives").glob("*.zip"))
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    status = (
        json.loads((RUN / "status.json").read_text())
        if (RUN / "status.json").exists()
        else {}
    )
    prefetch_log = RUN / "prefetch.log"
    feature_progress = {}
    if prefetch_log.exists():
        content = prefetch_log.read_text(errors="replace")
        phase = content.rsplit("Prefetching ", 1)[-1]
        language = re.match(r"(\w+): (\d+)", phase)
        counts = re.findall(r"Cached (\d+) clips", phase)
        feature_progress = {
            "complete_languages": sorted(
                p.stem for p in (RUN / "prefetch").glob("*.pt")
            ),
            "current_language": language[1] if language else None,
            "current_cached": int(counts[-1]) if counts else 0,
        }
    print(
        json.dumps(
            {
                "stage": status.get("phase"),
                "prepared_languages": prepared,
                "complete_zip_archives": archives,
                "fleurs_test": cached("test"),
                "fleurs_validation": cached("validation"),
                "vox_feature_caches": feature_progress,
                "available_ram_gib": round(available / 1024**3, 1),
                "free_disk_gib": round(shutil.disk_usage(".").free / 1024**3, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
