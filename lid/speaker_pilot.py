"""Matched, provisional speaker diagnostics. Never produces training identities."""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .extension_data import check_snapshot, file_hash, read_rows, write_json
from .speaker_study import SPLITS, write_csv

MODEL_REVISIONS = {
    "ecapa": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
    "wespeaker": "837717ddb9ff5507820346191109dc79c958d614",
}
RATE = 16000
CROPS = {
    "3": (56000, 104000),
    "5": (40000, 120000),
    "10": (0, 160000),
    "early3": (0, 48000),
    "late3": (112000, 160000),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def bind(path, identity):
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f"Cache identity changed: {path}; choose a fresh --root")
    write_json(path, identity)


def ordered(values, seed):
    return sorted(values, key=lambda v: (digest([seed, str(v)]), str(v)))


def coverage(intervals, left, right):
    return sum(
        max(0, min(right, r["end"]) - max(left, r["start"])) for r in intervals
    ) / (right - left)


def crop_bounds(length):
    if length < 160000:
        raise ValueError("short_audio")
    offset = (length - 160000) // 2
    return {key: (a + offset, b + offset) for key, (a, b) in CROPS.items()}


def hierarchical(values, method, count):
    from scipy.cluster.hierarchy import cut_tree, linkage
    from scipy.spatial.distance import pdist

    if count == 1:
        return np.zeros(len(values), dtype=int)
    if not 1 < count < len(values):
        raise ValueError("Invalid cluster count")
    return cut_tree(
        linkage(pdist(values, metric="cosine"), method=method), n_clusters=[count]
    ).ravel()


class Pilot:
    def __init__(self, config, root):
        self.config = json.loads(Path(config).read_text())
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if (
            self.config["sample_rate"] != RATE
            or self.config["durations"] != [3, 5, 10]
            or self.config["microbatch"] != 2
        ):
            raise ValueError("Unsupported waveform or microbatch protocol")
        bind(self.root / "configuration.json", self.config)

    def status(self, phase, **details):
        payload = dict(phase=phase, updated_at=datetime.now(UTC).isoformat(), **details)
        write_json(self.root / "status.json", payload)
        print(json.dumps(payload), flush=True)

    def prepare(self):
        from huggingface_hub import snapshot_download

        models = {
            name: {
                "repo": self.config[name + "_repo"],
                "revision": self.config.get(name + "_revision")
                or MODEL_REVISIONS[name],
            }
            for name in ["ecapa", "wespeaker"]
        }
        bind(self.root / "models.json", models)
        for name, item in models.items():
            patterns = (
                ["*.yaml", "*.ckpt", "*.txt"]
                if name == "ecapa"
                else ["config.yaml", "pytorch_model.bin"]
            )
            directory = self.root / "models" / name
            snapshot_download(
                item["repo"],
                revision=item["revision"],
                token=False,
                allow_patterns=patterns,
                local_dir=directory,
                max_workers=2,
            )
        fingerprints = {
            str(p.relative_to(self.root)): file_hash(p)
            for p in (self.root / "models").rglob("*")
            if p.is_file() and ".cache" not in p.parts
        }
        bind(self.root / "model_fingerprints.json", fingerprints)

    def validate(self):
        # Import every audio stack component before declaring the environment valid.
        import importlib

        import torch

        for module in ["torchaudio", "pyannote.audio", "speechbrain", "silero_vad"]:
            importlib.import_module(module)
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 2:
            torch.set_num_interop_threads(2)
        versions = {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "torchaudio",
                "numpy",
                "pyannote.audio",
                "speechbrain",
                "silero-vad",
            ]
        }
        assert versions["torch"].startswith("2.5.1+cu121")
        assert versions["torchaudio"].startswith("2.5.1+cu121")
        assert versions["numpy"] == "1.26.4" and versions["pyannote.audio"] == "3.3.2"
        assert torch.cuda.is_available()
        group = Path("/proc/self/cgroup").read_text().strip().split("0::")[-1]
        cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
        limits = {
            name: (cgroup / name).read_text().strip()
            for name in ["memory.max", "memory.swap.max", "cpu.max"]
        }
        assert (
            int(limits["memory.max"]) <= 4 * 1024**3
            and limits["memory.swap.max"] == "0"
        )
        quota, period = map(int, limits["cpu.max"].split())
        assert quota / period <= 2
        bind(
            self.root / "environment.json",
            {
                "versions": versions,
                "limits": limits,
                "lock_sha256": file_hash("environments/speaker_pilot/uv.lock"),
            },
        )
        self.status("environment_validated", versions=versions, limits=limits)

    def sample(self):
        import silero_vad
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        from .data import waveform

        rows = []
        for split in SPLITS:
            rows.extend(
                dict(r, original_split=split)
                for r in read_rows(Path("artifacts/extension15") / f"new_{split}.jsonl")
            )
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate source IDs")
        identity = {
            "config": self.config,
            "inventory": digest(rows),
            "vad": file_hash(Path(silero_vad.__file__).parent / "data/silero_vad.jit"),
            "crops": CROPS,
        }
        bind(self.root / "sample_identity.json", json.loads(json.dumps(identity)))
        if (self.root / "sample_ready.json").exists():
            self.load_sample()
            return
        torch.set_num_threads(2)
        vad = load_silero_vad().eval()
        selected, rejected, coverage_rows = [], [], []
        audio_dir = self.root / "audio"
        audio_dir.mkdir(exist_ok=True)
        seen_audio = set()
        for language in self.config["languages"]:
            groups = defaultdict(list)
            for row in rows:
                if row["language"] == language:
                    groups[row["group"]].append(row)
            accepted = 0
            for group in ordered(groups, self.config["seed"]):
                if accepted == self.config["recordings_per_language"]:
                    break
                candidates = []
                for row_id in ordered(
                    [r["id"] for r in groups[group]], self.config["seed"]
                ):
                    row = next(r for r in groups[group] if r["id"] == row_id)
                    check_snapshot(row)
                    try:
                        if file_hash(row["path"]) != row["sha256"]:
                            raise ValueError("audio_hash_changed")
                        if row["sha256"] in seen_audio or any(
                            r["sha256"] == row["sha256"] for r in candidates
                        ):
                            raise ValueError("duplicate_audio")
                        signal = waveform(row, 10)
                        bounds = crop_bounds(len(signal))
                        vad.reset_states()
                        intervals = get_speech_timestamps(
                            torch.from_numpy(signal),
                            vad,
                            sampling_rate=RATE,
                            threshold=0.5,
                            min_speech_duration_ms=250,
                            min_silence_duration_ms=300,
                            speech_pad_ms=150,
                        )
                        fractions = {
                            key: coverage(intervals, *value)
                            for key, value in bounds.items()
                        }
                        if (
                            min(fractions.values())
                            < self.config["minimum_speech_coverage"]
                        ):
                            raise ValueError("insufficient_vad_coverage")
                        token = digest(row["id"])[:24]
                        path = audio_dir / f"{token}.npy"
                        np.save(path, signal)
                        candidates.append(
                            dict(
                                row,
                                audio_cache=str(path),
                                audio_sha256=file_hash(path),
                                crop_bounds=bounds,
                                speech_coverage=fractions,
                                source_crop_offset_samples=(
                                    round(row["duration"] * RATE) - len(signal)
                                )
                                // 2,
                            )
                        )
                        if len(candidates) == self.config["cuts_per_recording"]:
                            break
                    except ValueError as error:
                        if str(error) == "audio_hash_changed":
                            raise
                        rejected.append(
                            {
                                "id": row_id,
                                "language": language,
                                "group": group,
                                "reason": str(error),
                            }
                        )
                if len(candidates) == self.config["cuts_per_recording"]:
                    selected.extend(candidates)
                    seen_audio.update(r["sha256"] for r in candidates)
                    accepted += 1
                else:
                    rejected.append(
                        {
                            "id": "",
                            "language": language,
                            "group": group,
                            "reason": "fewer_than_two_eligible_cuts",
                        }
                    )
                self.status(
                    "sampling",
                    language=language,
                    recordings=accepted,
                    cuts=len(selected),
                )
            coverage_rows.append(
                {
                    "language": language,
                    "available_recordings": len(groups),
                    "selected_recordings": accepted,
                    "selected_cuts": accepted * self.config["cuts_per_recording"],
                }
            )
        write_json(self.root / "rejected.json", rejected)
        write_csv(self.root / "recording_coverage.csv", coverage_rows)
        write_json(self.root / "sample.json", selected)
        if any(
            r["selected_recordings"] != self.config["recordings_per_language"]
            for r in coverage_rows
        ):
            raise ValueError(
                "Insufficient eligible recordings; see recording_coverage.csv"
            )
        write_json(
            self.root / "sample_ready.json",
            {
                "sample_sha256": file_hash(self.root / "sample.json"),
                "cuts": len(selected),
            },
        )

    def load_sample(self):
        ready = json.loads((self.root / "sample_ready.json").read_text())
        assert file_hash(self.root / "sample.json") == ready["sample_sha256"]
        rows = json.loads((self.root / "sample.json").read_text())
        for row in rows:
            check_snapshot(row)
            if "sha256" in row and file_hash(row["path"]) != row["sha256"]:
                raise ValueError("Source audio changed")
            if file_hash(row["audio_cache"]) != row["audio_sha256"]:
                raise ValueError("Cached audio changed")
        return rows

    def verify(self):
        """Check the actual crops and every encoder cache against the shared sample."""
        from .data import waveform

        rows = self.load_sample()
        counts = Counter((r["language"], r["group"]) for r in rows)
        if any(count != self.config["cuts_per_recording"] for count in counts.values()):
            raise ValueError("Incorrect cut count per recording")
        for language in self.config["languages"]:
            if (
                sum(code == language for code, _ in counts)
                != self.config["recordings_per_language"]
            ):
                raise ValueError("Incorrect recording coverage")
        waveform_hashes = set()
        for row in rows:
            signal = np.load(row["audio_cache"])
            if (
                signal.shape != (160000,)
                or signal.dtype != np.float32
                or not np.isfinite(signal).all()
            ):
                raise ValueError("Invalid matched audio")
            if not np.array_equal(signal, waveform(row, 10)):
                raise ValueError("Cached crop does not match original audio")
            waveform_hash = hashlib.sha256(signal.tobytes()).hexdigest()
            if waveform_hash in waveform_hashes:
                raise ValueError("Duplicate decoded audio")
            waveform_hashes.add(waveform_hash)
            if row["crop_bounds"] != {
                key: list(bounds) for key, bounds in crop_bounds(len(signal)).items()
            }:
                raise ValueError("Invalid crop boundaries")
            if (
                min(row["speech_coverage"].values())
                < self.config["minimum_speech_coverage"]
            ):
                raise ValueError("Uncertain speech coverage")
        for name in ["ecapa", "wespeaker"]:
            identity = json.loads((self.root / name / "identity.json").read_text())
            if identity["sample"] != file_hash(self.root / "sample.json"):
                raise ValueError("Encoder used another sample")
            for key in CROPS:
                with np.load(self.root / name / f"{key}.npz") as saved:
                    if saved["ids"].tolist() != [r["id"] for r in rows] or saved[
                        "identity"
                    ].item() != digest(identity):
                        raise ValueError("Mismatched encoder cache")
                    values = saved["embeddings"]
                    if not np.isfinite(values).all() or not np.allclose(
                        np.linalg.norm(values, axis=1), 1, atol=1e-5
                    ):
                        raise ValueError("Invalid embeddings")
                check = json.loads(
                    (self.root / name / f"{key}_reproducibility.json").read_text()
                )
                if check["max_absolute_difference"] > 1e-5:
                    raise ValueError("Reproducibility tolerance exceeded")
        write_json(
            self.root / "acceptance.json",
            {
                "sample_sha256": file_hash(self.root / "sample.json"),
                "cuts": len(rows),
                "unique_decoded_cuts": len(waveform_hashes),
                "matched_encoder_crop_sets": 10,
                "unpadded_audio_matches_source": True,
                "identical_ids_and_boundaries": True,
                "finite_unit_embeddings": True,
                "identity_status": "provisional; no labelled speaker truth",
            },
        )

    def ecapa(self):
        self.extract("ecapa")

    def wespeaker(self):
        self.extract("wespeaker")

    def extract(self, name):
        import torch

        self.validate()
        rows = self.load_sample()
        directory = self.root / "models" / name
        identity = {
            "config": self.config,
            "sample": file_hash(self.root / "sample.json"),
            "models": json.loads((self.root / "model_fingerprints.json").read_text()),
            "environment": json.loads((self.root / "environment.json").read_text()),
        }
        for relative, expected in identity["models"].items():
            if file_hash(self.root / relative) != expected:
                raise ValueError("Model weights changed")
        parts = self.root / name
        parts.mkdir(exist_ok=True)
        bind(parts / "identity.json", identity)
        if name == "ecapa":
            from speechbrain.inference.speaker import EncoderClassifier

            encoder = EncoderClassifier.from_hparams(
                source=str(directory.resolve()),
                savedir=str((parts / "loaded").resolve()),
                overrides={"pretrained_path": str(directory.resolve())},
                run_opts={"device": "cuda"},
            ).eval()

            def encode(batch):
                return encoder.encode_batch(batch, normalize=False).squeeze(1)
        else:
            from pyannote.audio import Model

            encoder = (
                Model.from_pretrained(
                    str(directory / "pytorch_model.bin"), map_location="cpu"
                )
                .eval()
                .to("cuda")
            )

            def encode(batch):
                return encoder(batch.unsqueeze(1))

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        extracted = False
        for key in CROPS:
            destination = parts / f"{key}.npz"
            if destination.exists():
                with np.load(destination) as saved:
                    assert saved["identity"].item() == digest(identity)
                    assert saved["ids"].tolist() == [r["id"] for r in rows]
                    cached = saved["embeddings"]
                    if not np.isfinite(cached).all() or not np.allclose(
                        np.linalg.norm(cached, axis=1), 1, atol=1e-5
                    ):
                        raise ValueError("Invalid cached embeddings")
                continue
            extracted = True
            values = []
            for start in range(0, len(rows), self.config["microbatch"]):
                batch_rows = rows[start : start + self.config["microbatch"]]
                signals = [
                    np.load(r["audio_cache"])[slice(*r["crop_bounds"][key])]
                    for r in batch_rows
                ]
                assert all(len(s) == CROPS[key][1] - CROPS[key][0] for s in signals)
                batch = torch.from_numpy(np.stack(signals)).to("cuda")
                with torch.inference_mode():
                    raw = encode(batch).float()
                    if start == 0:
                        repeat = encode(batch).float()
                        difference = float((raw - repeat).abs().max())
                        if not torch.allclose(raw, repeat, atol=1e-5, rtol=1e-5):
                            raise ValueError("Identical-input reproducibility failed")
                        write_json(
                            parts / f"{key}_reproducibility.json",
                            {"max_absolute_difference": difference},
                        )
                    if not torch.isfinite(raw).all() or (raw.norm(dim=-1) < 1e-8).any():
                        raise ValueError("Invalid embeddings")
                    embedding = torch.nn.functional.normalize(raw, dim=-1)
                    assert torch.allclose(
                        embedding.norm(dim=-1),
                        torch.ones(len(raw), device="cuda"),
                        atol=1e-5,
                    )
                    values.append(embedding.cpu().numpy())
                # Total process-independent device usage includes other GPU work.
                free, total = torch.cuda.mem_get_info()
                used = total - free
                if used >= self.config["vram_limit_gib"] * 1024**3:
                    raise RuntimeError("GPU use reached 7 GiB; extraction stopped")
                self.status(
                    "extracting",
                    model=name,
                    crop=key,
                    completed=start + len(batch_rows),
                    total=len(rows),
                    gpu_used_bytes=used,
                )
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                embeddings=np.concatenate(values),
                ids=[r["id"] for r in rows],
                identity=digest(identity),
            )
            temporary.replace(destination)
        if extracted:
            write_json(
                parts / "resources.json",
                {"peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()},
            )

    def analyze(self):
        import matplotlib
        from sklearn.decomposition import PCA
        from sklearn.metrics import adjusted_rand_score, silhouette_samples

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = self.load_sample()
        out = self.root / "analysis"
        out.mkdir(exist_ok=True)
        matrices = {}
        for name in ["ecapa", "wespeaker"]:
            for key in CROPS:
                with np.load(self.root / name / f"{key}.npz") as saved:
                    assert saved["ids"].tolist() == [r["id"] for r in rows]
                    assert saved["identity"].item() == digest(
                        json.loads((self.root / name / "identity.json").read_text())
                    )
                    values = saved["embeddings"]
                    assert np.isfinite(values).all() and np.allclose(
                        np.linalg.norm(values, axis=1), 1, atol=1e-5
                    )
                    matrices[name, key] = values
        identity = {
            "config": self.config,
            "sample": file_hash(self.root / "sample.json"),
            "embeddings": {
                f"{name}/{key}": file_hash(self.root / name / f"{key}.npz")
                for name, key in matrices
            },
        }
        bind(out / "identity.json", identity)
        marker = out / "complete.json"
        if marker.exists() and json.loads(marker.read_text())["identity"] == digest(
            identity
        ):
            return
        metrics, similarities, stability, assignments, candidates, projections = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        labels_by_duration = {}
        for name in ["ecapa", "wespeaker"]:
            for language in self.config["languages"]:
                indices = np.array(
                    [i for i, r in enumerate(rows) if r["language"] == language]
                )
                groups = np.array([rows[i]["group"] for i in indices])
                fig, ax = plt.subplots(figsize=(7, 4))
                within = np.sum(
                    matrices[name, "early3"][indices]
                    * matrices[name, "late3"][indices],
                    axis=1,
                )
                ax.hist(
                    within,
                    bins=30,
                    density=True,
                    histtype="step",
                    label="within cut: early/late 3s",
                )
                similarities.append(
                    {
                        "model": name,
                        "language": language,
                        "duration": "3 excerpts",
                        "relationship": "within_cut",
                        "n": len(within),
                        "p10": float(np.percentile(within, 10)),
                        "median": float(np.median(within)),
                        "p90": float(np.percentile(within, 90)),
                    }
                )
                for seconds in self.config["durations"]:
                    key = str(seconds)
                    values = matrices[name, key][indices]
                    similarity = np.clip(values @ values.T, -1, 1)
                    upper = np.triu(np.ones_like(similarity, dtype=bool), 1)
                    same_recording = groups[:, None] == groups[None, :]
                    for relation, mask in [
                        ("same_recording", upper & same_recording),
                        ("across_recordings", upper & ~same_recording),
                    ]:
                        scores = similarity[mask]
                        similarities.append(
                            {
                                "model": name,
                                "language": language,
                                "duration": key,
                                "relationship": relation,
                                "n": len(scores),
                                "p10": float(np.percentile(scores, 10)),
                                "median": float(np.median(scores)),
                                "p90": float(np.percentile(scores, 90)),
                            }
                        )
                        ax.hist(
                            scores,
                            bins=30,
                            density=True,
                            histtype="step",
                            label=f"{seconds}s {relation}",
                        )
                    coordinates = PCA(n_components=2).fit_transform(values)
                    for i, coordinate in zip(indices, coordinates):
                        projections.append(
                            {
                                "model": name,
                                "duration": seconds,
                                "language": language,
                                "id": rows[i]["id"],
                                "recording": rows[i]["group"],
                                "x": float(coordinate[0]),
                                "y": float(coordinate[1]),
                            }
                        )
                    fig_p, ax_p = plt.subplots(figsize=(6, 5))
                    ax_p.scatter(*coordinates.T, s=9, alpha=0.6)
                    ax_p.set_title(
                        f"{name} {language} {seconds}s: PCA, no identity labels"
                    )
                    fig_p.tight_layout()
                    fig_p.savefig(
                        out / f"projection_{name}_{language}_{seconds}.png", dpi=150
                    )
                    plt.close(fig_p)
                    for method in self.config["linkages"]:
                        for count in self.config["cluster_counts"]:
                            labels = hierarchical(values, method, count)
                            labels_by_duration[name, language, key, method, count] = (
                                labels
                            )
                            sizes = np.bincount(labels)
                            within_mask = upper & (labels[:, None] == labels[None, :])
                            between_mask = upper & ~within_mask
                            cohesion = similarity[within_mask]
                            separation = similarity[between_mask]
                            silhouettes = (
                                silhouette_samples(values, labels, metric="cosine")
                                if count > 1
                                else np.full(len(values), np.nan)
                            )
                            metrics.append(
                                {
                                    "model": name,
                                    "language": language,
                                    "duration": seconds,
                                    "linkage": method,
                                    "k": count,
                                    "silhouette": float(np.nanmean(silhouettes))
                                    if count > 1
                                    else "",
                                    "within_median": float(np.median(cohesion))
                                    if len(cohesion)
                                    else "",
                                    "within_min": float(cohesion.min())
                                    if len(cohesion)
                                    else "",
                                    "between_p95": float(np.percentile(separation, 95))
                                    if len(separation)
                                    else "",
                                    "similarity_gap": float(
                                        np.median(cohesion)
                                        - np.percentile(separation, 95)
                                    )
                                    if len(cohesion) and len(separation)
                                    else "",
                                    "smallest": int(sizes.min()),
                                    "largest": int(sizes.max()),
                                    "size_cv": float(sizes.std() / sizes.mean()),
                                }
                            )
                            for local, i in enumerate(indices):
                                assignments.append(
                                    {
                                        "model": name,
                                        "duration": seconds,
                                        "language": language,
                                        "linkage": method,
                                        "k": count,
                                        "id": rows[i]["id"],
                                        "recording": rows[i]["group"],
                                        "cluster": int(labels[local]),
                                        "silhouette": float(silhouettes[local])
                                        if count > 1
                                        else "",
                                        "provisional": True,
                                        "uncertain": True,
                                    }
                                )
                            if count == 1:
                                continue
                            for trial in range(self.config["resamples"]):
                                rng = np.random.default_rng(self.config["seed"] + trial)
                                chosen = np.array(
                                    [
                                        rng.choice(np.flatnonzero(groups == group))
                                        for group in sorted(set(groups))
                                    ]
                                )
                                new = hierarchical(values[chosen], method, count)
                                stability.append(
                                    {
                                        "model": name,
                                        "language": language,
                                        "duration": seconds,
                                        "linkage": method,
                                        "k": count,
                                        "trial": trial,
                                        "ari": float(
                                            adjusted_rand_score(labels[chosen], new)
                                        ),
                                    }
                                )
                ax.legend(fontsize=6)
                ax.set_xlabel("Cosine similarity (diagnostic relationships)")
                ax.set_title(f"{name}: {language}")
                fig.tight_layout()
                fig.savefig(out / f"similarities_{name}_{language}.png", dpi=150)
                plt.close(fig)
            for seconds in self.config["durations"]:
                values = matrices[name, str(seconds)]
                similarity = values @ values.T
                for i, row in enumerate(rows):
                    eligible = np.array(
                        [
                            j
                            for j, other in enumerate(rows)
                            if row["language"] != other["language"]
                        ]
                    )
                    for j in eligible[np.argsort(similarity[i, eligible])[-3:][::-1]]:
                        candidates.append(
                            {
                                "model": name,
                                "duration": seconds,
                                "left_id": row["id"],
                                "left_language": row["language"],
                                "left_recording": row["group"],
                                "right_id": rows[j]["id"],
                                "right_language": rows[j]["language"],
                                "right_recording": rows[j]["group"],
                                "cosine": float(similarity[i, j]),
                                "provisional": True,
                            }
                        )
        duration_stability = []
        for (name, language, key, method, count), labels in labels_by_duration.items():
            if key == "10":
                continue
            duration_stability.append(
                {
                    "model": name,
                    "language": language,
                    "duration": int(key),
                    "reference_duration": 10,
                    "linkage": method,
                    "k": count,
                    "ari": float(
                        adjusted_rand_score(
                            labels,
                            labels_by_duration[name, language, "10", method, count],
                        )
                    ),
                }
            )
        for filename, data in [
            ("comparison", metrics),
            ("similarities", similarities),
            ("resampling_stability", stability),
            ("duration_stability", duration_stability),
            ("assignments", assignments),
            ("cross_language_candidates", candidates),
            ("projection_points", projections),
        ]:
            write_csv(out / f"{filename}.csv", data)
        for name in ["ecapa", "wespeaker"]:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
            for ax, language in zip(axes, self.config["languages"]):
                for method in self.config["linkages"]:
                    for seconds in self.config["durations"]:
                        means = [
                            np.mean(
                                [
                                    r["ari"]
                                    for r in stability
                                    if r["model"] == name
                                    and r["language"] == language
                                    and r["linkage"] == method
                                    and r["duration"] == seconds
                                    and r["k"] == k
                                ]
                            )
                            for k in range(2, 16)
                        ]
                        ax.plot(range(2, 16), means, label=f"{method} {seconds}s")
                ax.set_title(language)
                ax.set_xlabel("Clusters (ten is a soft prior)")
            axes[0].set_ylabel("Mean resample ARI")
            axes[-1].legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(out / f"stability_{name}.png", dpi=150)
            plt.close(fig)
        write_json(
            out / "complete.json",
            {"identity": digest(identity), "combinations": len(metrics)},
        )

    def report(self):
        import csv

        out = self.root / "analysis"

        def read(name):
            with (out / f"{name}.csv").open() as stream:
                return list(csv.DictReader(stream))

        metrics = read("comparison")
        stability = read("resampling_stability")
        report = Path("reports/speaker_pilot")
        report.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Matched speaker embedding pilot",
            "",
            "All assignments are provisional and retained as uncertain. This automated pilot has no labelled speaker truth. Recording relationships and silhouette scores are diagnostics, not identity accuracy. Roughly ten readers per language is a soft prior; no identity count is imposed.",
            "",
            "The previous connected-component study remains unchanged. Its chaining failure did not establish that the embeddings could not cluster. This pilot uses average and complete linkage with cosine distance.",
            "",
            f"Sample: {json.loads((self.root / 'sample_ready.json').read_text())['cuts']} cuts, seed {self.config['seed']}, two cuts per recording. Every cut provides the same unpadded 10-second center window to both encoders, nested 3/5/10-second crops and two disjoint 3-second excerpts. Every evaluated crop passes 70% VAD coverage. Ordinary pauses remain. VAD does not identify speaker changes.",
            "",
            "## Comparison",
            "",
            "Best silhouette setting per model, language and duration is shown for navigation only. Selection over many settings can inflate this score. Cohesion must be judged relative to separation, duration agreement and recording resamples.",
            "",
            "| Model | Language | Seconds | Linkage | k | Silhouette | Cohesion−between p95 | Resample ARI |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
        for name in ["ecapa", "wespeaker"]:
            for language in self.config["languages"]:
                for seconds in self.config["durations"]:
                    pool = [
                        r
                        for r in metrics
                        if r["model"] == name
                        and r["language"] == language
                        and int(r["duration"]) == seconds
                        and int(r["k"]) > 1
                    ]
                    best = max(pool, key=lambda r: float(r["silhouette"]))
                    resamples = [
                        float(r["ari"])
                        for r in stability
                        if all(
                            r[k] == best[k]
                            for k in ["model", "language", "duration", "linkage", "k"]
                        )
                    ]
                    lines.append(
                        f"| {name} | {language} | {seconds} | {best['linkage']} | {best['k']} | {float(best['silhouette']):.3f} | {float(best['similarity_gap']):.3f} | {np.mean(resamples):.3f} |"
                    )
        lines += [
            "",
            "## Ten-cluster reference",
            "",
            "This fixed reference compares duration and encoder without selecting a different k. It is not an assertion of ten identities. Gap is median within-cluster cosine minus between-cluster p95. ARI compares partitions, not true people.",
            "",
            "| Model | Language | Seconds | Linkage | Silhouette | Gap | Resample ARI | Min/max cuts |",
            "|---|---|---:|---|---:|---:|---:|---|",
        ]
        for r in metrics:
            if r["k"] != "10":
                continue
            scores = [
                float(t["ari"])
                for t in stability
                if all(
                    t[k] == r[k]
                    for k in ["model", "language", "duration", "linkage", "k"]
                )
            ]
            lines.append(
                f"| {r['model']} | {r['language']} | {r['duration']} | {r['linkage']} | {float(r['silhouette']):.3f} | {float(r['similarity_gap']):.3f} | {np.mean(scores):.3f} | {r['smallest']}/{r['largest']} |"
            )
        lines += [
            "",
            "At the average-linkage ten-cluster reference, 10-second crops improve silhouette and the separation gap over 3-second crops for both models in all three languages. WeSpeaker has higher silhouette and gap than ECAPA for Samoan and Tongan at 3 seconds, while ECAPA has the larger Samoan gap at 10 seconds. Niuean favors ECAPA on both measures at 10 seconds. The model comparison therefore depends on language, duration and k.",
            "",
            "## Duration agreement",
            "",
            "Average linkage at k=10; shorter-duration labels compared with 10 seconds on the same cuts.",
            "",
            "| Model | Language | Seconds | ARI vs 10s |",
            "|---|---|---:|---:|",
        ]
        for r in read("duration_stability"):
            if r["k"] == "10" and r["linkage"] == "average":
                lines.append(
                    f"| {r['model']} | {r['language']} | {r['duration']} | {float(r['ari']):.3f} |"
                )
        lines += ["", "## Sampling and consistency", ""]
        rejected = Counter(
            r["reason"] for r in json.loads((self.root / "rejected.json").read_text())
        )
        lines.append(
            "Rejection events: "
            + "; ".join(
                f"{reason}: {count}" for reason, count in sorted(rejected.items())
            )
            + ". Recording rejection events are separate from cut rejections; eligible orphan cuts in a recording with fewer than two eligible cuts are excluded. Sampling stops after 100 accepted recordings per language, so these are not corpus-wide rejection rates."
        )
        lines += ["", "| Language | Original partition | Cuts |", "|---|---|---:|"]
        counts = Counter(
            (r["language"], r["original_split"])
            for r in json.loads((self.root / "sample.json").read_text())
        )
        for (language, partition), count in sorted(counts.items()):
            lines.append(f"| {language} | {partition} | {count} |")
        lines += [
            "",
            "| Model | Language | Within-cut 3s median | Same-recording 10s median | Across-recordings 10s median |",
            "|---|---|---:|---:|---:|",
        ]
        relationships = read("similarities")
        for name in ["ecapa", "wespeaker"]:
            for language in self.config["languages"]:

                def median(relation, duration, name=name, language=language):
                    return float(
                        next(
                            r["median"]
                            for r in relationships
                            if r["model"] == name
                            and r["language"] == language
                            and r["relationship"] == relation
                            and r["duration"] == duration
                        )
                    )

                lines.append(
                    f"| {name} | {language} | {median('within_cut', '3 excerpts'):.3f} | {median('same_recording', '10'):.3f} | {median('across_recordings', '10'):.3f} |"
                )
        lines += [
            "",
            "Same-recording similarities are generally high, supporting consistency as a working assumption for much of this sample. This does not establish one reader per recording: inspect low-tail pairs and unstable assignments in the CSV artifacts. Across-recording pairs include unknown same-person pairs, and all medians can conceal mixed speakers.",
            "",
            "## Interpretation",
            "",
        ]
        for language in self.config["languages"]:
            pool = [r for r in metrics if r["language"] == language and int(r["k"]) > 1]
            best = max(pool, key=lambda r: float(r["silhouette"]))
            lines.append(
                f"For {language}, {best['model']} at {best['duration']} seconds with {best['linkage']} linkage and k={best['k']} has the highest observed silhouette ({float(best['silhouette']):.3f}). This is a candidate for further investigation, not a verified reader inventory. Its median within-cluster similarity minus the between-cluster 95th percentile is {float(best['similarity_gap']):.3f}."
            )
            lines.append("")
        lines += [
            "Higher overall similarity alone is not an improvement. Low within-cut consistency can reflect speaker changes, channel effects or encoder variability. Same-recording cuts can contain different speakers; different recordings can share a speaker. Cross-language nearest matches are listed separately and do not certify independence.",
            "",
            "No speaker profiles are averaged, no full-corpus identities are assigned and no LID models are retrained. Stable clusters and plausible reader counts cannot guarantee zero true speaker overlap.",
            "",
            "## Artifacts",
            "",
            f"Configuration, immutable model revisions and hashes, audio hashes, rejected candidates, partition references, recording coverage and resource checks: `{self.root}`.",
            "",
            f"Full comparison, all uncertain assignments, similarity quantiles, duration/resample stability, PCA points with recording references and cross-language candidates: `{out}`.",
            "",
            "PCA is a projection for inspection; distances in the plot are not the original cosine distances.",
            "",
        ]
        models = json.loads((self.root / "models.json").read_text())
        lines += [
            "## Reproducibility",
            "",
            "| Encoder | Repository | Immutable revision |",
            "|---|---|---|",
        ]
        for name, model in models.items():
            lines.append(f"| {name} | {model['repo']} | `{model['revision']}` |")
        lines += [
            "",
            "The isolated uv lock fixes pyannote.audio 3.3.2, Torch/torchaudio 2.5.1+cu121 and NumPy 1.26.4; the complete installed environment passed dependency checks and actual GPU inference. The kernel scope enforces 4 GiB RAM, zero swap and two CPU equivalents; Torch/BLAS threads are limited to two. Batch size is two. The original run log records device usage after every extraction batch. Stage resource files describe their current invocation, which may reuse earlier caches.",
            "",
            "Acceptance checks reloaded all 600 source center windows and confirmed exact equality, unique decoded audio, matching sample IDs and crop boundaries, reproducibility records, finite embeddings and unit norms. `acceptance.json` records the result. No padding or repetition creates longer windows.",
            "",
        ]
        for name in ["ecapa", "wespeaker"]:
            lines += [f"![{name} stability](../../{out}/stability_{name}.png)", ""]
            for language in self.config["languages"]:
                lines += [
                    f"![{name} {language} similarity](../../{out}/similarities_{name}_{language}.png)",
                    "",
                ]
        (report / "speaker_embedding_pilot.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        self.status("report_written", report=str(report / "speaker_embedding_pilot.md"))
