"""Isolated, resumable speaker audit for Pacific radio adaptation."""

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .extension_data import RUN as ORIGINAL
from .extension_data import check_snapshot, file_hash, read_rows, write_json

ROOT = Path("artifacts/speaker_study")
EXTERNAL = Path("artifacts/youtube_validation_v2")
SPLITS = ("train", "selection", "calibration", "test")
LANGUAGES = ("sam", "niu", "ton")
THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)
SPEAKER_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def ensure_speaker_model():
    """Fetch only the small public model bundle; never require user credentials."""
    directory = ROOT / "speaker_model"
    files = (
        "hyperparams.yaml",
        "embedding_model.ckpt",
        "mean_var_norm_emb.ckpt",
        "classifier.ckpt",
        "label_encoder.txt",
    )
    expected = {
        "repo": "speechbrain/spkrec-ecapa-voxceleb",
        "revision": SPEAKER_REVISION,
    }
    marker = directory / "revision.json"
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError("Speaker model revision changed")
        if all((directory / name).is_file() for name in files):
            return
    from huggingface_hub import snapshot_download

    snapshot_download(
        expected["repo"],
        revision=SPEAKER_REVISION,
        token=False,
        allow_patterns=list(files),
        local_dir=str(directory),
    )
    write_json(marker, expected)


def status(phase, **values):
    payload = {"phase": phase, "updated_at": datetime.now(UTC).isoformat(), **values}
    write_json(ROOT / "status.json", payload)
    print(json.dumps(payload), flush=True)


def resources():
    """Read-only process/cgroup measurements; avoid initializing CUDA for reporting."""
    result = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            result["process_peak_ram_bytes"] = int(line.split()[1]) * 1024
    group = Path("/proc/self/cgroup").read_text().splitlines()
    for line in group:
        if line.startswith("0::"):
            path = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            for filename in ("memory.peak", "memory.max", "memory.swap.max"):
                if (path / filename).exists():
                    value = (path / filename).read_text().strip()
                    result[filename] = int(value) if value != "max" else value
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        result["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved()
    return result


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory():
    rows = []
    for split in SPLITS:
        for item in read_rows(ORIGINAL / f"new_{split}.jsonl"):
            rows.append({**item, "original_split": split, "domain": "radio"})
    for item in read_rows(EXTERNAL / "segments.jsonl"):
        rows.append({**item, "original_split": "external", "domain": "external"})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs in speaker study")
    for row in rows:
        check_snapshot(row)
        if row["domain"] == "external" and file_hash(row["path"]) != row["sha256"]:
            raise ValueError("External speech segment changed since evaluation")
    return rows


def speech_windows(intervals, size=48000):
    """Nonoverlapping 3-second windows, evenly spaced within each speech region."""
    result = []
    for interval in intervals:
        left, right = int(interval["start"]), int(interval["end"])
        count = (right - left) // size
        if count:
            # Distribute unused speech symmetrically; never join disjoint regions.
            offset = (right - left - count * size) // 2
            result.extend(
                (left + offset + i * size, left + offset + (i + 1) * size)
                for i in range(count)
            )
    return result


def extract():
    import silero_vad
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    from speechbrain.inference.speaker import EncoderClassifier

    from .data import waveform

    ROOT.mkdir(parents=True, exist_ok=True)
    rows = inventory()
    model_dir = ROOT / "speaker_model"
    model_revision = json.loads((model_dir / "revision.json").read_text())
    identity = {
        "rows_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        "speaker_model": model_revision,
        "speaker_weights_sha256": file_hash(model_dir / "embedding_model.ckpt"),
        "vad_weights_sha256": file_hash(
            Path(silero_vad.__file__).parent / "data/silero_vad.jit"
        ),
        "window_seconds": 3,
        "vad_threshold": 0.5,
        "min_silence_ms": 300,
        "speech_pad_ms": 150,
        "sample_rate": 16000,
        "protocol": "full cut; VAD; nonoverlapping fixed windows; no extra gain",
    }
    context = ROOT / "speaker_identity.json"
    if context.exists() and json.loads(context.read_text()) != identity:
        raise ValueError("Speaker cache identity changed; use a fresh study directory")
    write_json(context, identity)
    with (ROOT / "inventory.jsonl").open("w") as stream:
        stream.writelines(json.dumps(r) + "\n" for r in rows)
    parts = ROOT / "speaker_parts"
    parts.mkdir(exist_ok=True)
    if (ROOT / "speaker_ready.json").exists():
        return
    torch.set_num_threads(2)
    encoder = EncoderClassifier.from_hparams(
        source=str(model_dir.resolve()),
        savedir=str((ROOT / "speaker_loaded").resolve()),
        overrides={"pretrained_path": str(model_dir.resolve())},
        run_opts={"device": "cuda"},
    )
    encoder.eval()
    vad = load_silero_vad(onnx=False).eval()
    for start in range(0, len(rows), 128):
        target = parts / f"part-{start:06d}.pt"
        if target.exists():
            continue
        values, metadata, rejected, pending, pending_meta = [], [], [], [], []

        def flush(
            pending=pending, pending_meta=pending_meta, values=values, metadata=metadata
        ):
            if not pending:
                return
            with torch.inference_mode():
                batch = torch.stack(pending).to("cuda")
                embedding = encoder.encode_batch(batch, normalize=False).squeeze(1)
                embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
                if not torch.isfinite(embedding).all():
                    raise ValueError("Nonfinite speaker embedding")
                values.append(embedding.cpu())
            metadata.extend(pending_meta)
            pending.clear()
            pending_meta.clear()

        for index, row in enumerate(rows[start : start + 128], start):
            check_snapshot(row)
            # waveform crops to a limit: use the audited full duration here.
            if row["domain"] == "external":
                signal, rate = sf.read(row["path"], dtype="float32")
                if rate != 16000 or signal.ndim != 1:
                    raise ValueError("External audio protocol changed")
            else:
                signal = waveform(row, float(row["duration"]) + 1)
            vad.reset_states()
            with torch.inference_mode():
                intervals = get_speech_timestamps(
                    torch.from_numpy(signal),
                    vad,
                    sampling_rate=16000,
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=300,
                    speech_pad_ms=150,
                )
            windows = speech_windows(intervals)
            if not windows:
                rejected.append({"row": index, "reason": "no_contiguous_3s_speech"})
            for left, right in windows:
                pending.append(torch.from_numpy(signal[left:right].copy()))
                pending_meta.append(
                    {"row": index, "start": left / 16000, "end": right / 16000}
                )
                if len(pending) == 16:
                    flush()
        flush()
        payload = {
            "embeddings": torch.cat(values) if values else torch.empty(0, 192),
            "windows": metadata,
            "rejected": rejected,
            "identity": identity,
        }
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        status(
            "speaker_embeddings",
            completed=min(start + 128, len(rows)),
            total=len(rows),
            percent=round(100 * min(start + 128, len(rows)) / len(rows), 1),
        )
    write_json(ROOT / "speaker_ready.json", {"identity": identity, "cuts": len(rows)})


class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int64)

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return int(value)

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def connected_cut_groups(rows, speaker_sets):
    """A recording or inferred speaker may occur in only one resulting component."""
    union = UnionFind(len(rows))
    owners = {}
    for index, (row, speakers) in enumerate(zip(rows, speaker_sets)):
        keys = [("recording", row["group"]), ("duplicate", row["sha256"])]
        keys.extend(("speaker", int(speaker)) for speaker in speakers)
        for key in keys:
            if key in owners:
                union.union(index, owners[key])
            else:
                owners[key] = index
    return [union.find(i) for i in range(len(rows))]


def allocate(rows, groups):
    """Deterministic whole-component assignment; return a failed gate if impossible."""
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    totals = Counter(r["language"] for r in rows)
    sizes = {
        g: Counter(rows[i]["language"] for i in indices)
        for g, indices in members.items()
    }
    coverage = {
        code: sum(counts[code] > 0 for counts in sizes.values()) for code in LANGUAGES
    }
    if any(count < 4 for count in coverage.values()):
        return {}, {
            "feasible": False,
            "reason": "fewer than four independent groups for a language",
            "groups_per_language": coverage,
        }
    # Multiple deterministic greedy trials handle components spanning languages.
    best = None
    target = np.array([0.8, 0.05, 0.05, 0.1])[:, None] * np.array(
        [totals[c] for c in LANGUAGES]
    )
    for trial in range(256):
        rng = np.random.default_rng(42 + trial)
        order = sorted(
            members, key=lambda g: -(len(members[g]) * rng.uniform(0.8, 1.2))
        )
        counts = np.zeros((4, 3), dtype=int)
        assignment = {}
        for group in order:
            value = np.array([sizes[group][c] for c in LANGUAGES])
            scores = []
            for split in range(4):
                after = counts.copy()
                after[split] += value
                score = float(((after - target) ** 2 / np.maximum(target, 50)).sum())
                # Early coverage dominates proportions.
                score -= float(((counts == 0) & (after > 0)).sum()) * 1e8
                scores.append(score)
            selected = int(np.argmin(scores))
            counts[selected] += value
            assignment[group] = SPLITS[selected]
        if counts.min() >= 50:
            cost = float(((counts - target) ** 2 / np.maximum(target, 50)).sum())
            if best is None or cost < best[0]:
                best = cost, assignment, counts
    if best is None:
        return {}, {
            "feasible": False,
            "reason": "no allocation with at least 50 cuts per language in every partition",
            "groups_per_language": coverage,
        }
    _, assignment, counts = best
    return assignment, {
        "feasible": True,
        "groups_per_language": coverage,
        "counts": {
            split: dict(zip(LANGUAGES, map(int, count)))
            for split, count in zip(SPLITS, counts)
        },
    }


def graph_diagnostics():
    """Measure transitive chaining without asserting any pair's true identity."""
    import torch

    marker = ROOT / "speaker_cohesion.json"
    if marker.exists():
        return
    rows = read_rows(ROOT / "inventory.jsonl")
    matrices, windows = [], []
    for path in sorted((ROOT / "speaker_parts").glob("part-*.pt")):
        part = torch.load(path, map_location="cpu", weights_only=True)
        matrices.append(part["embeddings"])
        windows.extend(part["windows"])
    embeddings = torch.cat(matrices).numpy()
    with np.load(ROOT / "window_labels.npz") as saved:
        indices, labels = saved["radio_indices"], saved["labels"]
    largest, count = Counter(labels.tolist()).most_common(1)[0]
    pool = indices[labels == largest]
    rng = np.random.default_rng(42)
    left = rng.choice(pool, size=20000)
    right = rng.choice(pool, size=20000)
    different = np.array(
        [
            rows[windows[a]["row"]]["group"] != rows[windows[b]["row"]]["group"]
            for a, b in zip(left, right)
        ]
    )
    left, right = left[different][:10000], right[different][:10000]
    similarities = np.sum(embeddings[left] * embeddings[right], axis=1)
    pairs = []
    for a, b, score in zip(left, right, similarities):
        first, second = rows[windows[a]["row"]], rows[windows[b]["row"]]
        pairs.append(
            {
                "left_id": first["id"],
                "left_window_start": windows[a]["start"],
                "left_language": first["language"],
                "right_id": second["id"],
                "right_window_start": windows[b]["start"],
                "right_language": second["language"],
                "cosine": float(score),
                "same_language": first["language"] == second["language"],
            }
        )
    write_csv(
        ROOT / "speaker_cohesion_pairs.csv",
        pairs,
        fields=[
            "left_id",
            "left_window_start",
            "left_language",
            "right_id",
            "right_window_start",
            "right_language",
            "cosine",
            "same_language",
        ],
    )
    write_json(
        marker,
        {
            "largest_component_windows": count,
            "cross_recording_sample_pairs": len(pairs),
            "seed": 42,
            "fraction_below_primary_threshold": float((similarities < 0.5).mean())
            if len(pairs)
            else None,
            "cosine_p10_p50_p90": np.percentile(similarities, [10, 50, 90]).tolist()
            if len(pairs)
            else [],
            "window_labels_sha256": file_hash(ROOT / "window_labels.npz"),
            "interpretation": "Similarity diagnostics only, without labelled same/different-person truth. Low within-component pair similarity reveals transitive chaining.",
        },
    )


def cluster():
    """Exact blockwise cosine connectivity; no quadratic distance matrix is retained."""
    import torch
    from numba import njit

    @njit
    def link(similarity, parents, offset, thresholds):
        for i in range(similarity.shape[0]):
            for j in range(min(offset + i, similarity.shape[1])):
                value = similarity[i, j]
                if value < thresholds[0]:
                    continue
                for k in range(len(thresholds)):
                    if value < thresholds[k]:
                        break
                    a, b = offset + i, j
                    while parents[k, a] != a:
                        parents[k, a] = parents[k, parents[k, a]]
                        a = parents[k, a]
                    while parents[k, b] != b:
                        parents[k, b] = parents[k, parents[k, b]]
                        b = parents[k, b]
                    if a != b:
                        parents[k, max(a, b)] = min(a, b)

    if (ROOT / "audit_complete.json").exists():
        return json.loads((ROOT / "split_gate.json").read_text())
    rows = read_rows(ROOT / "inventory.jsonl")
    values, windows, rejected = [], [], []
    for path in sorted((ROOT / "speaker_parts").glob("part-*.pt")):
        part = torch.load(path, map_location="cpu", weights_only=True)
        if part["identity"] != json.loads((ROOT / "speaker_identity.json").read_text()):
            raise ValueError(
                "Speaker shard belongs to another input/model configuration"
            )
        if len(part["windows"]) != len(part["embeddings"]):
            raise ValueError("Speaker shard metadata and tensors do not align")
        values.append(part["embeddings"])
        windows.extend(part["windows"])
        rejected.extend(part["rejected"])
    embeddings = torch.cat(values)
    radio_mask = np.array([rows[w["row"]]["domain"] == "radio" for w in windows])
    radio_indices = np.flatnonzero(radio_mask)
    external_indices = np.flatnonzero(~radio_mask)
    radio = embeddings[radio_indices].to("cuda")
    parents = np.tile(np.arange(len(radio)), (len(THRESHOLDS), 1))
    # Disable TF32 so threshold comparisons use actual FP32 cosine scores.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        thresholds = np.array(THRESHOLDS, dtype=np.float32)
        for start in range(0, len(radio), 128):
            similarity = (
                (radio[start : start + 128] @ radio[: start + 128].T).cpu().numpy()
            )
            link(similarity, parents, start, thresholds)
            if start % 2048 == 0:
                status(
                    "speaker_connectivity",
                    completed=start,
                    total=len(radio),
                    percent=round(100 * start / len(radio), 1),
                )
        # External matches are diagnostic only: never union external audio into the radio groups.
        matches = []
        for start in range(0, len(external_indices), 128):
            selected = external_indices[start : start + 128]
            similarity = embeddings[selected].to("cuda") @ radio.T
            score, index = similarity.max(dim=1)
            for external, value, candidate in zip(
                selected, score.cpu().tolist(), index.cpu().tolist()
            ):
                matches.append(
                    {
                        "external_cut": rows[windows[external]["row"]]["id"],
                        "radio_cut": rows[windows[radio_indices[candidate]]["row"]][
                            "id"
                        ],
                        "cosine": value,
                        "candidate_at_0_5": value >= 0.5,
                    }
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    write_csv(ROOT / "external_speaker_matches.csv", matches)
    radio_rows = [r for r in rows if r["domain"] == "radio"]
    rejected_ids = {r["row"] for r in rejected}
    summaries = []
    for k, threshold in enumerate(THRESHOLDS):
        union = UnionFind(len(radio))
        union.parent = parents[k]
        labels = np.array([union.find(i) for i in range(len(radio))])
        speaker_counts = [Counter() for _ in radio_rows]
        for index, label in zip(radio_indices, labels):
            speaker_counts[windows[index]["row"]][int(label)] += 1
        speaker_sets = [set(counts) for counts in speaker_counts]
        primary = [
            str(min(counts, key=lambda s: (-counts[s], s)))
            if counts
            else "unknown:" + row["group"]
            for row, counts in zip(radio_rows, speaker_counts)
        ]
        components = connected_cut_groups(radio_rows, speaker_sets)
        assignments, gate = allocate(radio_rows, components)
        split_sets = defaultdict(set)
        for row, speakers in zip(radio_rows, speaker_sets):
            for speaker in speakers:
                split_sets[speaker].add(row["original_split"])
        component_counts = Counter(components)
        summary = {
            "threshold": threshold,
            "speaker_clusters": len(set(labels.tolist())),
            "cross_original_split_clusters": sum(
                len(s) > 1 for s in split_sets.values()
            ),
            "cuts_in_cross_split_clusters": sum(
                any(len(split_sets[s]) > 1 for s in ss) for ss in speaker_sets
            ),
            "connected_recording_speaker_groups": len(component_counts),
            "largest_group_cuts": max(component_counts.values()),
            "ambiguous_multiple_speaker_cuts": sum(len(ss) > 1 for ss in speaker_sets),
            **gate,
        }
        summaries.append(summary)
        if threshold == 0.5:
            write_json(ROOT / "split_gate.json", summary)
            metadata = []
            for i, (row, speakers, group) in enumerate(
                zip(radio_rows, speaker_sets, components)
            ):
                metadata.append(
                    {
                        "id": row["id"],
                        "language": row["language"],
                        "source_group": row["group"],
                        "original_split": row["original_split"],
                        "speaker_ids": json.dumps(sorted(speakers)),
                        "speaker_id": primary[i],
                        "speaker_group": group,
                        "uncertain": not speakers or len(speakers) != 1,
                        "no_3s_speech": i in rejected_ids,
                        "new_split": assignments.get(group, "unassigned"),
                    }
                )
            write_csv(ROOT / "speaker_labels.csv", metadata)
            write_json(ROOT / "group_sizes.json", dict(Counter(components)))
            np.savez_compressed(
                ROOT / "window_labels.npz", radio_indices=radio_indices, labels=labels
            )
            if gate["feasible"]:
                for split in SPLITS:
                    selected = [
                        {**row, "speaker_group": str(group), "speaker_id": primary[i]}
                        for i, (row, group) in enumerate(zip(radio_rows, components))
                        if assignments[group] == split
                    ]
                    with (ROOT / f"new_{split}.jsonl").open("w") as stream:
                        stream.writelines(json.dumps(row) + "\n" for row in selected)
                write_json(
                    ROOT / "data_ready.json",
                    {
                        "source_inventory_sha256": file_hash(ROOT / "inventory.jsonl"),
                        "speaker_identity_sha256": file_hash(
                            ROOT / "speaker_identity.json"
                        ),
                        "threshold": threshold,
                        "splits": {
                            s: file_hash(ROOT / f"new_{s}.jsonl") for s in SPLITS
                        },
                        **gate,
                    },
                )
    write_json(ROOT / "threshold_sensitivity.json", summaries)
    write_csv(
        ROOT / "threshold_sensitivity.csv",
        [
            {k: v for k, v in s.items() if k not in ("counts", "groups_per_language")}
            for s in summaries
        ],
    )
    write_json(
        ROOT / "audit_complete.json",
        {
            "radio_cuts": len(radio_rows),
            "windows": len(windows),
            "radio_windows": len(radio),
            "external_windows": len(external_indices),
            "unusable_cuts": len(rejected),
            "speaker_identity_sha256": file_hash(ROOT / "speaker_identity.json"),
        },
    )
    return json.loads((ROOT / "split_gate.json").read_text())
