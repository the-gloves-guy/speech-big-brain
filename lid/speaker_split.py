"""Provisional full-corpus speaker assignment, leakage audits and LID comparisons."""

import gc
import itertools
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .extension_data import check_snapshot, file_hash, read_rows, write_json
from .speaker_pilot import bind, digest, hierarchical
from .speaker_study import SPLITS, UnionFind, write_csv


def reference_labels(values, counts, method="average"):
    return hierarchical(values, method, counts)


def classify(values, references, labels):
    """Mean cosine to each reference group, without normalizing mixed profiles."""
    unique = sorted(set(labels.tolist()))
    scores = np.stack(
        [(values @ references[labels == label].T).mean(axis=1) for label in unique],
        axis=1,
    )
    rank = np.argsort(scores, axis=1)
    best = rank[:, -1]
    return (
        np.array(unique)[best],
        scores[np.arange(len(scores)), best],
        scores[np.arange(len(scores)), best]
        - scores[np.arange(len(scores)), rank[:, -2]],
    )


def audit_disjoint(partitions):
    sets = {}
    for split, rows in partitions.items():
        sets[split] = {
            key: {row[key] for row in rows}
            for key in ("id", "group", "sha256", "speaker_id")
        }
    for left, right in itertools.combinations(partitions, 2):
        for key in sets[left]:
            overlap = sets[left][key] & sets[right][key]
            if overlap:
                raise ValueError(
                    f"{key} leakage between {left} and {right}: {len(overlap)}"
                )
    return {
        "pairwise_overlap": {key: 0 for key in ("id", "group", "sha256", "speaker_id")},
        "scope": "provisional inferred Pacific speaker IDs; true-person independence is unverified",
    }


def allocate_speakers(rows, seed=42, minimum=50):
    """Whole global-speaker allocation, balancing cut counts without splitting identities."""
    speakers = sorted({r["speaker_id"] for r in rows})
    languages = sorted({r["language"] for r in rows})
    sizes = {
        speaker: np.array(
            [
                sum(
                    r["speaker_id"] == speaker and r["language"] == language
                    for r in rows
                )
                for language in languages
            ]
        )
        for speaker in speakers
    }
    total = np.sum(list(sizes.values()), axis=0)
    # Few readers imply at least three held-out readers per language; 80/5/5/10 is a soft size target.
    target = np.array([0.8, 0.05, 0.05, 0.1])[:, None] * total
    best = None
    # Exact one-reader-per-held-out-partition search when identities are language-local.
    # Enumerating k*(k-1)*(k-2) allocations keeps large readers eligible for training.
    if all(np.count_nonzero(size) == 1 for size in sizes.values()):
        result = {}
        for j, language in enumerate(languages):
            pool = sorted(
                (s for s in speakers if sizes[s][j]), key=lambda s: digest([seed, s])
            )
            options = None
            for held_out in itertools.permutations(pool, 3):
                counts = np.array(
                    [
                        total[j] - sum(sizes[s][j] for s in held_out),
                        *(sizes[s][j] for s in held_out),
                    ]
                )
                if counts.min() < minimum:
                    continue
                cost = float(
                    ((counts - target[:, j]) ** 2 / np.maximum(target[:, j], 50)).sum()
                )
                if options is None or cost < options[0]:
                    options = cost, held_out
            if options is None:
                raise ValueError(
                    "Cannot form four speaker-disjoint partitions with language coverage"
                )
            result.update({s: "train" for s in pool})
            result.update(zip(options[1], SPLITS[1:]))
        return result

    # Cross-language identities couple allocations. Deterministic random starts plus
    # coverage-preserving moves/swaps minimize the same cut-count objective.
    def cost(counts):
        return float(((counts - target) ** 2 / np.maximum(target, 50)).sum())

    for trial in range(256):
        rng = np.random.default_rng(seed + trial)
        assignment = {s: int(rng.integers(4)) for s in speakers}
        counts = np.zeros((4, len(languages)), dtype=int)
        for speaker, split in assignment.items():
            counts[split] += sizes[speaker]
        if counts.min() < minimum:
            continue
        for _ in range(64):
            improvement = None
            current = cost(counts)
            for speaker in speakers:
                source = assignment[speaker]
                for destination in range(4):
                    after = counts.copy()
                    after[source] -= sizes[speaker]
                    after[destination] += sizes[speaker]
                    value = cost(after)
                    if (
                        after.min() >= minimum
                        and value < current - 1e-9
                        and (improvement is None or value < improvement[0])
                    ):
                        improvement = value, {speaker: destination}, after
            for left, right in itertools.combinations(speakers, 2):
                a, b = assignment[left], assignment[right]
                if a == b:
                    continue
                after = counts.copy()
                after[a] += sizes[right] - sizes[left]
                after[b] += sizes[left] - sizes[right]
                value = cost(after)
                if (
                    after.min() >= minimum
                    and value < current - 1e-9
                    and (improvement is None or value < improvement[0])
                ):
                    improvement = value, {left: b, right: a}, after
            if improvement is None:
                break
            assignment.update(improvement[1])
            counts = improvement[2]
        value = cost(counts)
        if best is None or value < best[0]:
            best = value, assignment
    if best is None:
        raise ValueError(
            "Cannot form four speaker-disjoint partitions with language coverage"
        )
    return {speaker: SPLITS[split] for speaker, split in best[1].items()}


class Study:
    def __init__(
        self, config="configs/speaker_split.json", root="artifacts/speaker_split"
    ):
        self.config = json.loads(Path(config).read_text())
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        bind(self.root / "configuration.json", self.config)
        self.pilot = Path(self.config["pilot"])
        self.previous = Path(self.config["previous_study"])
        self.external = Path(self.config["external"])

    def status(self, phase, **details):
        payload = dict(phase=phase, updated_at=datetime.now(UTC).isoformat(), **details)
        write_json(self.root / "status.json", payload)
        print(json.dumps(payload), flush=True)

    def inventory(self):
        rows = read_rows(self.previous / "inventory.jsonl")
        return [r for r in rows if r["domain"] == "radio"]

    def extract(self):
        import torch
        from pyannote.audio import Model

        from .data import waveform
        from .speaker_study import resources

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rows = self.inventory()
        model_dir = self.pilot / "models/wespeaker"
        models = json.loads((self.pilot / "models.json").read_text())
        identity = {
            "config": self.config,
            "inventory": digest(rows),
            "model": models["wespeaker"],
            "weights": file_hash(model_dir / "pytorch_model.bin"),
            "protocol": "mono16k real center crop <=10s, no padding or gain; single waveform inference",
            "environment": file_hash("environments/speaker_pilot/uv.lock"),
        }
        expected = json.loads((self.pilot / "model_fingerprints.json").read_text())[
            "models/wespeaker/pytorch_model.bin"
        ]
        if identity["weights"] != expected:
            raise ValueError("WeSpeaker weights changed")
        bind(self.root / "embedding_identity.json", identity)
        parts = self.root / "speaker_parts"
        parts.mkdir(exist_ok=True)
        encoder = (
            Model.from_pretrained(
                str(model_dir / "pytorch_model.bin"), map_location="cpu"
            )
            .eval()
            .to("cuda")
        )
        with torch.inference_mode():
            for start in range(0, len(rows), 128):
                path = parts / f"part-{start:06d}.npz"
                batch_rows = rows[start : start + 128]
                if path.exists():
                    with np.load(path) as saved:
                        if saved["identity"].item() != digest(identity) or saved[
                            "ids"
                        ].tolist() != [r["id"] for r in batch_rows]:
                            raise ValueError("Speaker shard changed")
                    continue
                embeddings = []
                for row in batch_rows:
                    check_snapshot(row)
                    if file_hash(row["path"]) != row["sha256"]:
                        raise ValueError("Source audio changed")
                    signal = waveform(row, 10)
                    if len(signal) < int(self.config["minimum_seconds"] * 16000):
                        raise ValueError("Unexpected short cut")
                    raw = encoder(
                        torch.from_numpy(signal).reshape(1, 1, -1).to("cuda")
                    ).float()
                    if not torch.isfinite(raw).all() or raw.norm() < 1e-8:
                        raise ValueError("Invalid embedding")
                    embeddings.append(
                        torch.nn.functional.normalize(raw, dim=-1).cpu().numpy()[0]
                    )
                temp = path.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temp,
                    embeddings=np.stack(embeddings),
                    ids=[r["id"] for r in batch_rows],
                    identity=digest(identity),
                )
                temp.replace(path)
                free, total = torch.cuda.mem_get_info()
                if total - free >= 7 * 1024**3:
                    raise RuntimeError("GPU use reached 7 GiB")
                self.status(
                    "speaker_embeddings",
                    completed=start + len(batch_rows),
                    total=len(rows),
                    gpu_used_bytes=total - free,
                )
        write_json(
            self.root / "embedding_ready.json",
            {"identity": digest(identity), "cuts": len(rows), "resources": resources()},
        )

    def split(self):
        rows = self.inventory()
        identity = json.loads((self.root / "embedding_identity.json").read_text())
        values = []
        for start in range(0, len(rows), 128):
            with np.load(
                self.root / "speaker_parts" / f"part-{start:06d}.npz"
            ) as saved:
                if saved["identity"].item() != digest(identity) or saved[
                    "ids"
                ].tolist() != [r["id"] for r in rows[start : start + 128]]:
                    raise ValueError("Speaker embedding alignment changed")
                values.append(saved["embeddings"])
        values = np.concatenate(values)
        if not np.isfinite(values).all() or not np.allclose(
            np.linalg.norm(values, axis=1), 1, atol=1e-5
        ):
            raise ValueError("Invalid normalized speaker embeddings")
        sample = json.loads((self.pilot / "sample.json").read_text())
        with np.load(self.pilot / "wespeaker/10.npz") as saved:
            if saved["ids"].tolist() != [r["id"] for r in sample]:
                raise ValueError("Pilot IDs differ")
            anchors = saved["embeddings"]
        predictions = np.empty(len(rows), dtype=object)
        similarities = np.zeros(len(rows))
        margins = np.zeros(len(rows))
        anchor_labels = np.empty(len(sample), dtype=object)
        medoids = []
        for language, count in self.config["clusters"].items():
            ref = np.array(
                [i for i, r in enumerate(sample) if r["language"] == language]
            )
            labels = reference_labels(anchors[ref], count)
            labels = np.array([f"{language}:{label:02d}" for label in labels])
            anchor_labels[ref] = labels
            indices = np.array(
                [i for i, r in enumerate(rows) if r["language"] == language]
            )
            for start in range(0, len(indices), 256):
                subset = indices[start : start + 256]
                predictions[subset], similarities[subset], margins[subset] = classify(
                    values[subset], anchors[ref], labels
                )
            for label in sorted(set(labels)):
                pool = ref[labels == label]
                medoid = int(
                    pool[np.argmax((anchors[pool] @ anchors[pool].T).mean(axis=1))]
                )
                medoids.append(
                    {
                        "label": label,
                        "index": medoid,
                        "language": language,
                        "id": sample[medoid]["id"],
                        "cuts": len(pool),
                    }
                )
        # Global identities can span languages. Merge only high-similarity cross-language anchor candidates.
        union = UnionFind(len(medoids))
        cross = []
        for i, j in itertools.combinations(range(len(medoids)), 2):
            a, b = medoids[i], medoids[j]
            if a["language"] == b["language"]:
                continue
            cosine = float(anchors[a["index"]] @ anchors[b["index"]])
            merge = cosine >= self.config["cross_language_similarity"]
            cross.append(
                {
                    "left": a["label"],
                    "right": b["label"],
                    "cosine": cosine,
                    "merged": merge,
                }
            )
            if merge:
                union.union(i, j)
        global_ids = {
            m["label"]: f"global:{union.find(i):02d}" for i, m in enumerate(medoids)
        }
        write_csv(self.root / "cross_language_anchor_audit.csv", cross)
        write_json(
            self.root / "reference_groups.json",
            {
                "medoids": medoids,
                "global_ids": global_ids,
                "model": json.loads((self.pilot / "models.json").read_text())[
                    "wespeaker"
                ],
            },
        )
        candidates, excluded = [], []
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row["group"]].append(i)
        for recording, indices in groups.items():
            reliable = [
                i
                for i in indices
                if similarities[i] >= self.config["minimum_similarity"]
                and margins[i] >= self.config["minimum_margin"]
            ]
            speakers = {global_ids[predictions[i]] for i in reliable}
            if len(speakers) != 1:
                reason = (
                    "conflicting_speakers_in_recording"
                    if len(speakers) > 1
                    else "no_confident_speaker"
                )
                excluded.extend(
                    {
                        "id": rows[i]["id"],
                        "group": recording,
                        "language": rows[i]["language"],
                        "reason": reason,
                        "similarity": float(similarities[i]),
                        "margin": float(margins[i]),
                    }
                    for i in indices
                )
                continue
            speaker = next(iter(speakers))
            for i in indices:
                row = rows[i]
                if i not in reliable:
                    excluded.append(
                        {
                            "id": row["id"],
                            "group": recording,
                            "language": row["language"],
                            "reason": "uncertain_cut",
                            "similarity": float(similarities[i]),
                            "margin": float(margins[i]),
                        }
                    )
                    continue
                candidates.append(
                    dict(
                        row,
                        speaker_id=speaker,
                        language_speaker_id=predictions[i],
                        speaker_similarity=float(similarities[i]),
                        speaker_margin=float(margins[i]),
                        speaker_provisional=True,
                    )
                )
        # Exact duplicate files across recordings must share the same global identity or be excluded.
        hashes = defaultdict(set)
        for row in candidates:
            hashes[row["sha256"]].add(row["speaker_id"])
        bad_hashes = {key for key, ids in hashes.items() if len(ids) > 1}
        bad_recordings = {r["group"] for r in candidates if r["sha256"] in bad_hashes}
        kept = []
        for row in candidates:
            if row["group"] in bad_recordings:
                excluded.append(
                    {
                        "id": row["id"],
                        "group": row["group"],
                        "language": row["language"],
                        "reason": "duplicate_with_conflicting_identity",
                        "similarity": row["speaker_similarity"],
                        "margin": row["speaker_margin"],
                    }
                )
            else:
                kept.append(row)
        assignment = allocate_speakers(
            kept, self.config["seed"], self.config["minimum_split_cuts"]
        )
        partitions = {
            split: [r for r in kept if assignment[r["speaker_id"]] == split]
            for split in SPLITS
        }
        audit = audit_disjoint(partitions)
        audit.update(
            {
                "retained_cuts": len(kept),
                "excluded_cuts": len(excluded),
                "original_cuts": len(rows),
                "counts": {
                    s: dict(Counter(r["language"] for r in part))
                    for s, part in partitions.items()
                },
                "speakers": {
                    s: {
                        lang: sorted(
                            {r["speaker_id"] for r in part if r["language"] == lang}
                        )
                        for lang in self.config["clusters"]
                    }
                    for s, part in partitions.items()
                },
                "exclusion_reasons": dict(Counter(r["reason"] for r in excluded)),
            }
        )
        bind(
            self.root / "split_identity.json",
            {
                "config": self.config,
                "embeddings": digest(identity),
                "pilot_sample": file_hash(self.pilot / "sample.json"),
                "pilot_embeddings": file_hash(self.pilot / "wespeaker/10.npz"),
            },
        )
        for split, part in partitions.items():
            path = self.root / f"new_{split}.jsonl"
            content = "".join(json.dumps(r, sort_keys=True) + "\n" for r in part)
            if path.exists() and path.read_text() != content:
                raise ValueError("Existing speaker partition changed")
            path.write_text(content)
        write_csv(self.root / "excluded.csv", excluded)
        write_csv(
            self.root / "speaker_labels.csv",
            [
                {
                    "id": r["id"],
                    "language": r["language"],
                    "recording": r["group"],
                    "speaker_id": r["speaker_id"],
                    "language_speaker_id": r["language_speaker_id"],
                    "split": assignment[r["speaker_id"]],
                    "similarity": r["speaker_similarity"],
                    "margin": r["speaker_margin"],
                }
                for r in kept
            ],
        )
        write_json(self.root / "split_audit.json", audit)
        write_json(
            self.root / "data_ready.json",
            {
                "feasible": True,
                "config": self.config,
                "splits": {s: file_hash(self.root / f"new_{s}.jsonl") for s in SPLITS},
                "audit": audit,
            },
        )
        # Reuse the original order-aligned LID caches without altering the previous study.
        for name in ["inventory.jsonl", "frozen_lid_embeddings.pt"]:
            path = self.root / name
            if not path.exists():
                path.symlink_to((self.previous / name).resolve())
        self.status("speaker_splits_ready", **audit)

    def train(self):
        self.run_experiments(train=True)

    def evaluate(self):
        self.run_experiments(train=False)

    def run_experiments(self, train):
        import torch

        from . import speaker_experiments as experiments
        from .extension_evaluate import evaluate_method
        from .extension_train import train as train_model
        from .speaker_study import resources

        experiments.ROOT = self.root
        experiments.EXTERNAL = self.external
        experiments.status = self.status
        partitions = {s: read_rows(self.root / f"new_{s}.jsonl") for s in SPLITS}
        audit_disjoint(partitions)
        # Train both frozen heads first so external results are available before longer encoder updates.
        for method in ["frozen", "partial"]:
            for condition, sampling in self.config["conditions"].items():
                run = experiments.prepare_run(condition)

                def emit(phase, condition=condition, method=method, **details):
                    self.status(phase, condition=condition, method=method, **details)

                if train:
                    train_model(method, run=run, sampling=sampling, progress=emit)
                if not (run / method / "complete.json").exists():
                    raise ValueError("Training incomplete")
                evaluate_method(method, run=run, progress=emit)
                experiments.evaluate_external(run, method)
                gc.collect()
                torch.cuda.empty_cache()
                write_json(
                    self.root / f"resources_{condition}_{method}.json", resources()
                )
                self.report()

    def report(self):
        """Fixed external cohort is the primary generalisation comparison."""
        from scipy.special import softmax
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        from .extension_data import CODES
        from .youtube_metrics import score_segments

        external = read_rows(self.external / "segments.jsonl")
        with np.load(self.external / "logits.npz") as saved:
            if saved["ids"].tolist() != [r["id"] for r in external]:
                raise ValueError("External baseline ID mismatch")
            logits = {
                f"recording_uniform/{m}": saved[f"{m}15"] for m in ["frozen", "partial"]
            }
        from .extension_data import RUN as ORIGINAL

        temperatures = {
            f"recording_uniform/{m}": json.loads(
                (ORIGINAL / m / "calibration.json").read_text()
            )["temperature"]
            for m in ["frozen", "partial"]
        }
        internal = []
        for method in ["frozen", "partial"]:
            for condition in self.config["conditions"]:
                path = self.root / "runs" / condition / method
                if not (path / "external/logits.npz").exists():
                    continue
                with np.load(path / "external/logits.npz") as saved:
                    if saved["ids"].tolist() != [r["id"] for r in external]:
                        raise ValueError("External IDs changed")
                    logits[f"{condition}/{method}"] = saved["logits"]
                temperatures[f"{condition}/{method}"] = json.loads(
                    (path / "calibration.json").read_text()
                )["temperature"]
        predictions = {
            name: np.array([CODES[i] for i in values.argmax(axis=1)])
            for name, values in logits.items()
        }
        common = np.logical_and.reduce([p != "en" for p in predictions.values()])
        truth = np.array([r["language"] for r in external])
        summaries, per_language = [], []
        detailed = {}
        for name, values in logits.items():
            confidence = softmax(
                values.astype(np.float64) / temperatures[name], axis=1
            ).max(axis=1)
            conditional = score_segments(external, predictions[name], confidence)
            shared = score_segments(external, predictions[name], confidence, common)
            summaries.append(
                {
                    "model": name,
                    "total": len(external),
                    "accuracy_all": float(accuracy_score(truth, predictions[name])),
                    "macro_f1_all": float(
                        f1_score(
                            truth,
                            predictions[name],
                            labels=["sam", "niu", "ton"],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "conditional_accuracy": conditional["accuracy"],
                    "conditional_f1": conditional["macro_f1"],
                    "english_exclusions": conditional["english_predictions"],
                    "coverage": conditional["coverage"],
                    "ece_conditional": conditional["ece"],
                    "common_n": int(common.sum()),
                    "common_accuracy": shared["accuracy"],
                }
            )
            detailed[name] = {"conditional": conditional, "common": shared}
            for language in ["sam", "niu", "ton"]:
                mask = truth == language
                per_language.append(
                    {
                        "model": name,
                        "language": language,
                        "n": int(mask.sum()),
                        "accuracy_all": float(
                            (predictions[name][mask] == truth[mask]).mean()
                        ),
                        "precision": float(
                            precision_score(
                                truth,
                                predictions[name],
                                labels=[language],
                                average=None,
                                zero_division=0,
                            )[0]
                        ),
                        "recall": float(
                            recall_score(
                                truth,
                                predictions[name],
                                labels=[language],
                                average=None,
                                zero_division=0,
                            )[0]
                        ),
                        "f1": float(
                            f1_score(
                                truth,
                                predictions[name],
                                labels=[language],
                                average=None,
                                zero_division=0,
                            )[0]
                        ),
                    }
                )
        write_csv(self.root / "external_summary.csv", summaries)
        write_csv(self.root / "external_per_language.csv", per_language)
        write_json(self.root / "external_comparison.json", detailed)
        for condition in ["recording_uniform", *self.config["conditions"]]:
            for method in ["frozen", "partial"]:
                path = (
                    ORIGINAL / method
                    if condition == "recording_uniform"
                    else self.root / "runs" / condition / method
                )
                if not (path / "report.json").exists():
                    continue
                report = json.loads((path / "report.json").read_text())["groups"]
                for group in ["old12", "new3"]:
                    internal.append(
                        dict(
                            condition=condition,
                            method=method,
                            group=group,
                            **{
                                key: report[group]["raw"][key]
                                for key in ["accuracy", "macro_f1", "ece"]
                            },
                        )
                    )
        write_csv(self.root / "internal_summary.csv", internal)
        audit = json.loads((self.root / "split_audit.json").read_text())
        lines = [
            "# Speaker-controlled LID reruns",
            "",
            "Speaker labels are provisional automated labels derived from fixed WeSpeaker pilot groups: Samoan 8, Niuean 7 and Tongan 10. All source recordings and inferred global speaker IDs are kept together. Ambiguous cuts, conflicting recordings and conflicting duplicates are quarantined. This prevents overlap of the assigned IDs, but cannot certify zero overlap of real people. The original 12-language VoxLingua/FLEURS splits are unchanged.",
            "",
            "## Split audit",
            "",
            f"Retained {audit['retained_cuts']:,} of {audit['original_cuts']:,} cuts; excluded {audit['excluded_cuts']:,}. Cut IDs, source recordings, file hashes and inferred speaker IDs have zero overlap across train, checkpoint selection, calibration and test.",
            "",
            "| Partition | Samoan | Niuean | Tongan |",
            "|---|---:|---:|---:|",
        ]
        for split in SPLITS:
            lines.append(
                f"| {split} | "
                + " | ".join(
                    str(audit["counts"][split].get(lang, 0))
                    for lang in ["sam", "niu", "ton"]
                )
                + " |"
            )
        lines += [
            "",
            "## Fixed external validation",
            "",
            "Same 428 external speech segments for every model, never used for checkpoint selection, temperature fitting or label/split selection. Folder labels are provisional. All-segment accuracy counts English predictions as errors; conditional accuracy reproduces the previous English-exclusion policy. The common cohort is shared by every currently completed model and can change until all four reruns finish.",
            "",
            "| Model | All-segment accuracy | Macro F1 | Conditional accuracy | English exclusions | Common accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]

        def pct(value):
            return "undefined" if value is None else f"{100 * value:.2f}%"

        for r in summaries:
            lines.append(
                f"| {r['model']} | {pct(r['accuracy_all'])} | {pct(r['macro_f1_all'])} | {pct(r['conditional_accuracy'])} | {r['english_exclusions']} | {pct(r['common_accuracy'])} |"
            )
        lines += [
            "",
            "## Generalisation change versus the corresponding original model",
            "",
        ]
        for row in summaries:
            if row["model"].startswith("recording_uniform/"):
                continue
            method = row["model"].split("/")[-1]
            baseline = next(
                r for r in summaries if r["model"] == f"recording_uniform/{method}"
            )
            delta = 100 * (row["accuracy_all"] - baseline["accuracy_all"])
            lines.append(
                f"- {row['model']}: {delta:+.2f} percentage points in fixed external all-segment accuracy."
            )
        complete = sum(
            (self.root / "runs" / condition / method / "external/logits.npz").exists()
            for condition in self.config["conditions"]
            for method in ["frozen", "partial"]
        )
        lines += [
            "",
            f"Completed reruns: {complete}/4. One training seed, identical initialization and maximum update budgets. The existing recording-based baselines used different amounts of data and different training/selection populations, so an external difference does not isolate the causal effect of speaker splitting. The two new sampling conditions share identical partitions.",
            "",
            "Internal Pacific test populations differ from the original recording split; compare them as different difficulty levels. Old-language FLEURS results use the same test population and 15 outputs. Cross-language identities use a conservative threshold on reference medoids; true-person matches, unknown readers and multiple readers per recording can still be missed. Pretraining speaker overlap is not controlled.",
            "",
            f"Full metrics, per-language comparisons, split audit, excluded cuts, model fingerprints and checkpoints are in `{self.root}`.",
            "",
        ]
        directory = Path("reports/speaker_split")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "speaker_generalisation.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        self.status(
            "comparison_updated",
            completed_runs=complete,
            report=str(directory / "speaker_generalisation.md"),
        )
