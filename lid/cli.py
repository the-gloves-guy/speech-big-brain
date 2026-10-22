"""Command-line entry point; each stage can be rerun independently."""

import argparse
import importlib
import os
import sys


def lazy(module, function):
    def run(args):
        return getattr(importlib.import_module(module), function)(args)

    return run


def prepare_manifest(args):
    from .data import prepare_vox

    return prepare_vox(args.root, args.output, args.validation_fraction, args.seed)


def main():
    parser = argparse.ArgumentParser(
        description="Compare restricted-set language ID approaches"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare-vox", help="Create video-disjoint manifests from extracted VoxLingua"
    )
    prepare.add_argument(
        "--root", required=True, help="Directory containing ar/, en/, ..., zh/"
    )
    prepare.add_argument("--output", default="artifacts/manifests")
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=prepare_manifest)

    cache = sub.add_parser(
        "cache",
        help="Extract frozen embeddings and original logits with resumable shards",
    )
    source = cache.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Local VoxLingua JSONL manifest")
    source.add_argument(
        "--fleurs",
        action="store_true",
        help="Download selected FLEURS archives to disk and read individual WAV files",
    )
    cache.add_argument(
        "--split", choices=["train", "validation", "test"], required=True
    )
    cache.add_argument("--output", required=True)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--threads", type=int, default=4)
    cache.add_argument(
        "--batch-size", type=int, default=1, help="Batch only equal-length signals"
    )
    cache.add_argument(
        "--buffer-size",
        type=int,
        default=32,
        help="Maximum decoded WAVs held in memory",
    )
    cache.add_argument(
        "--max-seconds",
        type=float,
        default=10.0,
        help="Center crop; 0 keeps full audio",
    )
    cache.add_argument(
        "--limit-per-language",
        type=int,
        default=0,
        help="0 means all; nonzero is a pilot",
    )
    cache.add_argument(
        "--model-revision", default="main", help="Resolved and recorded as a commit SHA"
    )
    cache.add_argument(
        "--dataset-revision",
        default="70bb2e84b976b7e960aa89f1c648e09c59f894dd",
        help="FLEURS original audio archive revision",
    )
    cache.set_defaults(func=lazy("lid.cache", "extract"))

    fit = sub.add_parser(
        "train-head",
        help="Fine-tune the complete pretrained MLP classifier on cached embeddings",
    )
    fit.add_argument("--train", required=True)
    fit.add_argument("--validation", required=True)
    fit.add_argument("--output", default="artifacts/head")
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--patience", type=int, default=5)
    fit.add_argument("--batch-size", type=int, default=256)
    fit.add_argument("--lr", type=float, default=0.0001)
    fit.add_argument("--weight-decay", type=float, default=0.0001)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--threads", type=int, default=4)
    fit.add_argument("--device", default="cpu")
    fit.set_defaults(func=lazy("lid.train", "train"))

    report = sub.add_parser(
        "compare",
        help="Fit temperatures on FLEURS validation and evaluate on FLEURS test",
    )
    report.add_argument("--calibration", required=True)
    report.add_argument("--test", required=True)
    report.add_argument("--head", required=True)
    report.add_argument("--output", default="artifacts/comparison")
    report.add_argument("--bins", type=int, default=15)
    report.add_argument("--threads", type=int, default=4)
    report.set_defaults(func=lazy("lid.compare", "compare"))

    infer = sub.add_parser(
        "predict", help="Identify a local audio file using an adapted model"
    )
    infer.add_argument("audio")
    infer.add_argument("--method", choices=["filtered", "head"], default="filtered")
    infer.add_argument("--head", help="Required for --method head")
    infer.add_argument("--calibration", help="calibration.json produced by compare")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--threads", type=int, default=4)
    infer.set_defaults(func=lazy("lid.predict", "predict"))

    args = parser.parse_args()
    if (
        args.command == "cache"
        and args.fleurs
        and not os.environ.get("LID_MEMORY_GUARDED")
    ):
        from .guard import launch

        raise SystemExit(
            launch([sys.executable, "-m", "lid.cli", *sys.argv[1:]], memory="2G")
        )
    args.func(args)


if __name__ == "__main__":
    main()
