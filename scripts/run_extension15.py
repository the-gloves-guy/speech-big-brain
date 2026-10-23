"""Run/resume the authorized 15-language paired experiment in a memory guard."""

import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="all",
        choices=(
            "all",
            "audit",
            "cache",
            "frozen",
            "evaluate_frozen",
            "partial",
            "evaluate_partial",
            "comparison",
        ),
    )
    args = parser.parse_args()
    if not os.environ.get("LID_MEMORY_GUARDED"):
        from lid.guard import launch

        raise SystemExit(
            launch(
                [sys.executable, "-u", __file__, *sys.argv[1:]],
                memory="4G",
                unit="lid-extension15"
                if args.stage == "all"
                else f"lid-extension15-{args.stage}",
            )
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    from lid.extension_data import RUN, status, write_json

    RUN.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        from lid.extension_data import prepare

        prepare()
        return
    if args.stage == "cache":
        import torch

        from lid.extension_model import cache_new

        torch.set_num_threads(2)
        cache_new()
        return
    stages = [
        (
            "frozen",
            "from lid.extension_train import train; train('frozen')",
            RUN / "frozen/complete.json",
        ),
        (
            "evaluate_frozen",
            "from lid.extension_evaluate import evaluate_method; evaluate_method('frozen')",
            RUN / "frozen/evaluation_complete.json",
        ),
        (
            "partial",
            "from lid.extension_train import train; train('partial')",
            RUN / "partial/complete.json",
        ),
        (
            "evaluate_partial",
            "from lid.extension_evaluate import evaluate_method; evaluate_method('partial')",
            RUN / "partial/evaluation_complete.json",
        ),
        (
            "comparison",
            "from lid.extension_evaluate import write_comparison; write_comparison()",
            RUN / "comparison_complete.json",
        ),
    ]
    if args.stage != "all":
        stages = [stage for stage in stages if stage[0] == args.stage]
    try:
        if not (RUN / "data_ready.json").exists():
            from lid.extension_data import prepare

            prepare()
        while not (RUN / "cache_ready.json").exists():
            result = subprocess.run(
                ["systemctl", "--user", "is-active", "lid-extension15-cache.scope"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode and not (RUN / "cache_ready.json").exists():
                status("resume_feature_caching")
                with (RUN / "cache.log").open("a") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-c",
                            "import torch; torch.set_num_threads(2); from lid.extension_model import cache_new; cache_new()",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                continue
            print("Waiting for memory-protected GPU feature caching", flush=True)
            time.sleep(20)
        cache = json.loads((RUN / "cache_ready.json").read_text())
        from lid.extension_data import BASE, file_hash

        if cache["initial_head_sha256"] != file_hash(BASE / "head/head.pt"):
            raise ValueError("Original model changed after feature extraction")
        for name, code, marker in stages:
            if marker.exists():
                continue
            status(name)
            write_json(
                RUN / "pipeline.json",
                {
                    "stage": name,
                    "completed_stages": [
                        key for key, _, path in stages if path.exists()
                    ],
                    "total_stages": len(stages),
                },
            )
            with (RUN / f"{name}.log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        "import torch; torch.set_num_threads(2); " + code,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        status(
            "complete",
            codes=cache.get(
                "codes",
                [
                    "ar",
                    "en",
                    "es",
                    "fa",
                    "hi",
                    "id",
                    "mi",
                    "ps",
                    "ru",
                    "th",
                    "ur",
                    "zh",
                    "sam",
                    "niu",
                    "ton",
                ],
            ),
            results=str(RUN / "summary.csv"),
            report="reports/extension15/language_extension.md",
        )
        write_json(
            RUN / "pipeline.json",
            {
                "stage": "complete",
                "completed_stages": [name for name, _, _ in stages],
                "total_stages": len(stages),
            },
        )
    except BaseException as error:
        status(
            "failed",
            error=str(error),
            resume_command="uv run python -u scripts/run_extension15.py",
        )
        raise


if __name__ == "__main__":
    main()
