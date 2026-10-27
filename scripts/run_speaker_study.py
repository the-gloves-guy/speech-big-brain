"""Run the Pacific speaker audit and generalisation study under a RAM limit."""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "extract", "audit", "visualize", "train", "evaluate", "report"),
        default="all",
    )
    args = parser.parse_args()
    if not os.environ.get("LID_MEMORY_GUARDED"):
        from lid.guard import launch

        raise SystemExit(
            launch(
                [sys.executable, "-u", __file__, *sys.argv[1:]],
                memory="4G",
                unit="lid-speaker-study",
            )
        )
    if args.stage in ("all", "extract"):
        from lid.speaker_study import ensure_speaker_model

        ensure_speaker_model()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    import torch

    from lid.speaker_study import (
        ROOT,
        cluster,
        extract,
        graph_diagnostics,
        resources,
        status,
    )

    torch.set_num_threads(2)
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage in ("all", "extract"):
            extract()
        if args.stage in ("all", "audit"):
            cluster()
            graph_diagnostics()
        if args.stage in ("all", "visualize"):
            from lid.speaker_visualize import visualize

            visualize()
            from lid.speaker_experiments import duration_diagnostic

            duration_diagnostic()
        if args.stage in ("all", "train", "evaluate"):
            gate = json.loads((ROOT / "split_gate.json").read_text())
            if gate["feasible"]:
                from lid.speaker_experiments import run_experiments

                run_experiments(train=args.stage != "evaluate")
            else:
                status("retraining_not_feasible", reason=gate["reason"])
        if args.stage in ("all", "report"):
            graph_diagnostics()
            from lid.speaker_visualize import write_report

            write_report()
            from lid.extension_data import write_json

            write_json(
                ROOT / f"resources_{args.stage}.json",
                {
                    "stage": args.stage,
                    "scope": "current invocation only",
                    **resources(),
                },
            )
            gate = json.loads((ROOT / "split_gate.json").read_text())
            trained = list((ROOT / "runs").glob("*/*/complete.json"))
            evaluated = list((ROOT / "runs").glob("*/*/evaluation_complete.json"))
            complete = (
                (ROOT / "visualization_complete.json").exists()
                and (ROOT / "duration_complete.json").exists()
                and (not gate["feasible"] or len(evaluated) == 4)
            )
            status(
                "complete" if complete else "report_complete",
                report="reports/speaker_study/speaker_generalisation.md",
                split_feasible=gate["feasible"],
                new_models_trained=len(trained),
                retraining_skipped_reason=gate.get("reason"),
            )
    except Exception as error:
        status("failed", error=repr(error), stage=args.stage)
        raise


if __name__ == "__main__":
    main()
