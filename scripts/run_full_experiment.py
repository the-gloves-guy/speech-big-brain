"""Run the authorized full twelve-language experiment, preserving logs/artifacts."""

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

RUN = Path("artifacts/full_run")
MODEL = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
FLEURS = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def status(phase):
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"phase": phase, "updated_at": datetime.now(UTC).isoformat()}
    (RUN / "status.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(payload, flush=True)


def stage(name, arguments, complete):
    if Path(complete).exists():
        print(f"Already completed: {name}", flush=True)
        return
    status(name)
    with (RUN / f"{name}.log").open("a") as log:
        subprocess.run(
            [sys.executable, "-u", "-m", "lid.cli", *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    finish_existing = "--finish-existing" in sys.argv
    downloader = None
    if not finish_existing:
        with Path("artifacts/vox_download.log").open("a") as download_log:
            downloader = subprocess.Popen(
                [sys.executable, "-u", "scripts/download_vox.py"],
                stdout=download_log,
                stderr=subprocess.STDOUT,
            )
    try:
        # Populate target-domain caches while VoxLingua is downloading.
        for split in ("validation", "test"):
            if finish_existing:
                continue  # These cache jobs have already been started separately.
            output = str(RUN / f"fleurs_direct_{split}.pt")
            stage(
                f"cache_fleurs_{split}",
                [
                    "cache",
                    "--fleurs",
                    "--split",
                    split,
                    "--output",
                    output,
                    "--model-revision",
                    MODEL,
                    "--dataset-revision",
                    FLEURS,
                    "--batch-size",
                    "16",
                    "--device",
                    "cuda",
                ],
                output,
            )
        status("waiting_for_voxlingua_downloads")
        if finish_existing:
            from download_vox import CODES, ROOT, extract

            while True:
                for code in CODES:
                    archive = ROOT / f"archives/{code}.zip"
                    if (
                        code != "en"
                        and archive.exists()
                        and (ROOT / f"archives/{code}.json").exists()
                        and not (ROOT / f"{code}.extracted.json").exists()
                    ):
                        status(f"extracting_{code}")
                        extract(archive)
                ready = sum(
                    (ROOT / f"{code}.extracted.json").exists() for code in CODES
                )
                test_ready = all(
                    (RUN / f"fleurs_direct_{split}.pt").exists()
                    for split in ("validation", "test")
                )
                status(f"waiting_data_{ready}_of_12_fleurs_ready_{test_ready}")
                if ready == 12 and test_ready:
                    break
                time.sleep(30)
        elif downloader.wait() != 0:
            raise RuntimeError(
                "VoxLingua download/extraction failed; see artifacts/vox_download.log"
            )
        stage(
            "prepare_vox",
            [
                "prepare-vox",
                "--root",
                "data/voxlingua107/train",
                "--output",
                str(RUN / "manifests"),
            ],
            RUN / "manifests/validation.jsonl",
        )
        for split in ("train", "validation"):
            output = str(RUN / f"vox_{split}.pt")
            stage(
                f"cache_vox_{split}",
                [
                    "cache",
                    "--manifest",
                    str(RUN / f"manifests/{split}.jsonl"),
                    "--split",
                    split,
                    "--output",
                    output,
                    "--model-revision",
                    MODEL,
                    "--batch-size",
                    "16",
                    "--device",
                    "cuda",
                ],
                output,
            )
        stage(
            "train_gpu",
            [
                "train-head",
                "--train",
                str(RUN / "vox_train.pt"),
                "--validation",
                str(RUN / "vox_validation.pt"),
                "--output",
                str(RUN / "head"),
                "--device",
                "cuda",
                "--batch-size",
                "256",
                "--epochs",
                "30",
                "--patience",
                "5",
            ],
            RUN / "head/head.pt",
        )
        stage(
            "compare",
            [
                "compare",
                "--calibration",
                str(RUN / "fleurs_direct_validation.pt"),
                "--test",
                str(RUN / "fleurs_direct_test.pt"),
                "--head",
                str(RUN / "head/head.pt"),
                "--output",
                str(RUN / "comparison"),
            ],
            RUN / "comparison/report.json",
        )
        status("complete")
    except BaseException:
        status("failed_see_logs")
        raise


if __name__ == "__main__":
    main()
