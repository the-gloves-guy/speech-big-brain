"""Assemble video-disjoint training/validation caches from frozen per-language features."""

import hashlib
import json
import time
from pathlib import Path

import torch

from .cache import compatible, load_cache
from .data import read_manifest
from .languages import CODES


def merge_prefetched(args, directory):
    directory = Path(directory)
    session = json.loads((directory / "session.json").read_text())
    if (
        args.model_revision != session["model_revision"]
        or args.max_seconds != session["max_seconds"]
    ):
        raise ValueError("Prefetched features use a different model or crop protocol")
    if args.limit_per_language:
        raise ValueError("Prefetched full-run caches cannot apply a pilot sample cap")
    while not all((directory / f"{code}.pt").exists() for code in CODES):
        state = json.loads((directory / "session.json").read_text())
        if state["phase"] == "failed":
            raise RuntimeError(
                "Embedding prefetch failed; resume scripts/prefetch_vox.py"
            )
        done = [code for code in CODES if (directory / f"{code}.pt").exists()]
        print(f"Waiting for frozen feature caches: {done}", flush=True)
        time.sleep(30)
    rows = read_manifest(args.manifest)
    wanted = {row["id"]: row for row in rows}
    tensors = {key: [] for key in ("embeddings", "logits", "targets")}
    ids, groups, first, fingerprints = [], [], None, {}
    for code in CODES:
        bundle = load_cache(directory / f"{code}.pt", require_all_languages=False)
        if set(bundle["targets"].tolist()) != {CODES.index(code)}:
            raise ValueError(f"Incorrect language in prefetched cache: {code}")
        if first is None:
            first = {
                "meta": bundle["meta"],
                "labels": bundle["labels"],
                "head_state": bundle["head_state"],
            }
        else:
            compatible(first, bundle)
        selected = [i for i, item in enumerate(bundle["ids"]) if item in wanted]
        for i in selected:
            item = bundle["ids"][i]
            if (
                wanted[item]["group"] != bundle["groups"][i]
                or wanted[item]["language"] != code
            ):
                raise ValueError(f"Source group changed for {item}")
            ids.append(item)
            groups.append(bundle["groups"][i])
        index = torch.tensor(selected, dtype=torch.long)
        for key, values in tensors.items():
            values.append(bundle[key][index])
        fingerprints[code] = bundle["meta"]["manifest_sha256"]
    if len(ids) != len(wanted) or set(ids) != set(wanted):
        raise ValueError("Prefetched features do not exactly cover the split manifest")
    meta = {
        **first["meta"],
        "split": args.split,
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "prefetch_manifest_sha256": fingerprints,
    }
    result = {
        **{key: torch.cat(value) for key, value in tensors.items()},
        "ids": ids,
        "groups": groups,
        "labels": first["labels"],
        "head_state": first["head_state"],
        "meta": meta,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(result, temporary)
    temporary.replace(output)
    print(
        f"Assembled {len(ids)} examples from frozen features into {output}", flush=True
    )
