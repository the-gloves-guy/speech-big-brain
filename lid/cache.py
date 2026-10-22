"""Small tensor caches replace repeated GPU audio passes."""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

from .data import fleurs_records, read_manifest, waveform
from .languages import CODES
from .model import MODEL_ID, cpu_state, load_encoder


def extract(args):
    torch.set_num_threads(args.threads)
    if args.limit_per_language < 0 or args.max_seconds < 0:
        raise ValueError("Limits must be nonnegative")
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Cache already exists: {output}; choose a new output path")
    prefetch = output.parent / "prefetch"
    if args.manifest and (prefetch / "session.json").exists():
        from .prefetch import merge_prefetched

        merge_prefetched(args, prefetch)
        return
    encoder, labels, revision = load_encoder(args.device, args.model_revision)
    if args.manifest:
        partial_languages = getattr(args, "allow_partial_languages", False)
        rows = (
            read_manifest(args.manifest, require_all_languages=False)
            if partial_languages
            else read_manifest(args.manifest)
        )
        dataset_revision = None
        source_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    else:
        from huggingface_hub import HfApi

        dataset_revision = (
            HfApi(token=False)
            .dataset_info("google/fleurs", revision=args.dataset_revision)
            .sha
        )
        rows = fleurs_records(args.split, dataset_revision, args.limit_per_language)
        source_hash = None
    batch_size = getattr(args, "batch_size", 1)
    buffer_size = getattr(args, "buffer_size", 32)
    if batch_size < 1 or buffer_size < 1:
        raise ValueError("Extraction batch and buffer sizes must be positive")
    meta = {
        "model": MODEL_ID,
        "model_revision": revision,
        "max_seconds": args.max_seconds,
        "crop": "center",
        "sample_rate": 16000,
        "codes": CODES,
        "source": "voxlingua" if args.manifest else "fleurs",
        "split": args.split,
        "dataset_revision": dataset_revision,
        "manifest_sha256": source_hash,
        "limit_per_language": args.limit_per_language,
    }
    parts = output.with_suffix(output.suffix + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    context = parts / "context.json"
    if context.exists() and json.loads(context.read_text()) != meta:
        raise ValueError("Partial cache has incompatible extraction settings")
    context.write_text(json.dumps(meta, indent=2) + "\n")
    seen = set()
    counts = Counter()
    shard_paths = sorted(parts.glob("part-*.pt"))
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu", weights_only=True)
        if seen.intersection(shard["ids"]):
            raise ValueError("Duplicate examples in partial cache")
        seen.update(shard["ids"])
        counts.update(CODES[i] for i in shard["targets"].tolist())
    print(f"Starting {output}; resuming {len(seen)} examples", flush=True)
    embeddings, logits, targets, ids, groups = [], [], [], [], []
    pending = []
    reported = time.monotonic()

    def save_shard():
        if not ids:
            return
        path = parts / f"part-{len(shard_paths):06d}.pt"
        shard = {
            "embeddings": torch.cat(embeddings),
            "logits": torch.cat(logits),
            "targets": torch.tensor(targets),
            "ids": list(ids),
            "groups": list(groups),
        }
        temporary = path.with_suffix(".tmp")
        torch.save(shard, temporary)
        temporary.replace(path)
        shard_paths.append(path)
        embeddings.clear()
        logits.clear()
        targets.clear()
        ids.clear()
        groups.clear()

    def encode_pending():
        nonlocal reported
        # Equal sample counts avoid padding changing the input signal or ECAPA outputs.
        by_length = {}
        for row, signal in pending:
            by_length.setdefault(len(signal), []).append((row, signal))
        for examples in by_length.values():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                signal = torch.stack([torch.from_numpy(item[1]) for item in batch]).to(
                    args.device
                )
                embedding = encoder.encode_batch(signal, normalize=False)
                scores = encoder.mods.classifier(embedding).reshape(len(batch), -1)
                embeddings.append(embedding.reshape(len(batch), -1).float().cpu())
                logits.append(scores.float().cpu())
                for row, _ in batch:
                    ids.append(row["id"])
                    groups.append(row["group"])
                    targets.append(CODES.index(row["language"]))
                    seen.add(row["id"])
                if len(ids) >= 1024:
                    save_shard()
        pending.clear()
        if time.monotonic() - reported >= 30:
            print(f"Cached {len(seen)} clips: {dict(counts)}", flush=True)
            reported = time.monotonic()

    with torch.inference_mode():
        for row in rows:
            code = row["language"]
            if row["id"] in seen:
                continue
            if args.limit_per_language and counts[code] >= args.limit_per_language:
                continue
            pending.append((row, waveform(row, args.max_seconds)))
            counts[code] += 1
            if len(pending) >= buffer_size:
                encode_pending()
        encode_pending()
    save_shard()
    if not counts or (
        not getattr(args, "allow_partial_languages", False)
        and set(counts) != set(CODES)
    ):
        raise ValueError(f"Cache missing languages: {set(CODES) - set(counts)}")
    shards = [
        torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths
    ]
    bundle = {
        "embeddings": torch.cat([shard["embeddings"] for shard in shards]),
        "logits": torch.cat([shard["logits"] for shard in shards]),
        "targets": torch.cat([shard["targets"] for shard in shards]),
        "ids": [item for shard in shards for item in shard["ids"]],
        "groups": [item for shard in shards for item in shard["groups"]],
        "labels": labels,
        "head_state": cpu_state(encoder.mods.classifier),
        "meta": meta,
    }
    if len(bundle["ids"]) != len(set(bundle["ids"])):
        raise ValueError("Duplicate example IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(output)
    print(f"Saved {len(bundle['ids'])} examples to {output}")


def load_cache(path, require_all_languages=True):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    n = len(bundle["targets"])
    if (
        n == 0
        or bundle["embeddings"].shape != (n, 256)
        or bundle["logits"].shape != (n, 107)
    ):
        raise ValueError(f"Invalid cache shape: {path}")
    targets = set(bundle["targets"].tolist())
    if (
        bundle["meta"]["codes"] != CODES
        or not targets.issubset(set(range(12)))
        or (require_all_languages and targets != set(range(12)))
    ):
        raise ValueError(
            f"Cache must contain all selected languages in the canonical order: {path}"
        )
    if len(bundle["ids"]) != n or len(bundle["groups"]) != n:
        raise ValueError(f"Invalid cache identifiers: {path}")
    if (
        not torch.isfinite(bundle["embeddings"]).all()
        or not torch.isfinite(bundle["logits"]).all()
    ):
        raise ValueError(f"Nonfinite cache values: {path}")
    return bundle


def compatible(*bundles):
    keys = ("model", "model_revision", "max_seconds", "crop", "sample_rate", "codes")
    for bundle in bundles[1:]:
        if bundle["labels"] != bundles[0]["labels"] or any(
            bundle["meta"][key] != bundles[0]["meta"][key] for key in keys
        ):
            raise ValueError(
                "Caches use incompatible models, label orders, or audio protocols"
            )


def disjoint(a, b):
    if set(a["ids"]) & set(b["ids"]) or set(a["groups"]) & set(b["groups"]):
        raise ValueError("Data leakage: caches share examples or source groups")
