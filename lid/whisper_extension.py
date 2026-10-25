"""Original-recording-split, paired Whisper tiny Pacific extension.

No external waveform enters training, selection, or temperature fitting.
"""

import gc
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .extension_data import CODES, check_snapshot, file_hash, read_rows, write_json
from .extension_model import state_fingerprint
from .extension_train import epoch_order, scaled_optimizer_step, selection_metrics
from .model import cpu_state
from .speaker_pilot import bind, digest
from .whisper_model import Adapter, load_whisper, whisper_head
from .whisper_study import Study, save_torch


def expanded_head(old, seed=42):
    """Copy every old parameter exactly; draw only the appended output rows."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = whisper_head(old["1.weight"].shape[1] // 2)
        head[3] = torch.nn.Linear(512, len(CODES))
    state = head.state_dict()
    for key, value in old.items():
        if key.startswith("3."):
            state[key][:12].copy_(value)
        else:
            state[key].copy_(value)
    head.load_state_dict(state)
    return head


def audit_partitions(partitions):
    seen_ids, seen_groups, seen_hashes = set(), set(), set()
    for name, rows in partitions.items():
        ids = [r["id"] for r in rows]
        groups = {r["group"] for r in rows}
        hashes = {r["sha256"] for r in rows if "sha256" in r}
        if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
            raise ValueError(f"Duplicate/overlapping IDs: {name}")
        if seen_groups & groups or seen_hashes & hashes:
            raise ValueError(f"Recording/audio overlap: {name}")
        if {r["language"] for r in rows} != set(CODES[12:]):
            raise ValueError(f"Missing Pacific language: {name}")
        seen_ids.update(ids)
        seen_groups.update(groups)
        seen_hashes.update(hashes)


def audio_fingerprint(path):
    """Ignore nondeterministic WAV PEAK timestamps while hashing exact PCM values."""
    import soundfile as sf

    signal, rate = sf.read(path, dtype="float32", always_2d=True)
    return digest(
        {
            "sample_rate": rate,
            "shape": signal.shape,
            "pcm_sha256": hashlib.sha256(signal.astype("<f4").tobytes()).hexdigest(),
        }
    )


def restore_resume(path, identity, adapter, optimizer, scaler):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["identity"] != identity:
        raise ValueError("Resume checkpoint configuration/data changed")
    adapter.encoder.load_state_dict(saved["encoder"])
    adapter.head.load_state_dict(saved["head"])
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return [saved[k] for k in ["epoch", "cursor", "stale", "best", "history"]]


class ExtensionAdapter(Adapter):
    def __init__(self, root, checkpoint, finetune=False):
        self.family = "whisper"
        self.model = load_whisper(root)
        self.model.requires_grad_(False)
        self.encoder = self.model.encoder
        old = whisper_head(self.model.dims.n_audio_state).state_dict()
        self.head = expanded_head(old).to("cuda")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.head.load_state_dict(saved["head"])
        if "encoder" in saved:
            self.encoder.load_state_dict(saved["encoder"])
        self.finetune = finetune
        self.configure(False)

    def configure(self, train=False):
        self.model.eval().requires_grad_(False)
        self.encoder.requires_grad_(train and self.finetune)
        self.encoder.train(train and self.finetune)
        self.head.requires_grad_(True).train(train)


class ExtensionStudy(Study):
    adapter_class = ExtensionAdapter
    def __init__(
        self, config="configs/whisper15.json", root="artifacts/whisper_tiny15"
    ):
        super().__init__(config, root)
        self.base = Path(self.config["base"])
        self.pacific = Path(self.config["pacific"])

    def status(self, phase, **details):
        if torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            if total - free >= 7 * 1024**3:
                raise RuntimeError("Global GPU use reached 7 GiB")
        super().status(phase, **details)

    def prepare(self):
        from .youtube_data import TARGETS, prepare

        _, snapshot = prepare(
            "validation_clips", self.config["external"], languages=TARGETS
        )
        previous = json.loads(
            (
                Path(self.config["historical_external"]) / "segments_ready.json"
            ).read_text()
        )
        old_sources = {s["source_id"]: s for s in previous["configuration"]["sources"]}
        current = snapshot["configuration"]["sources"]
        for source in current:
            if (
                source["source_id"] in old_sources
                and source != old_sources[source["source_id"]]
            ):
                raise ValueError(
                    "Historical source changed; cannot claim unchanged cohort"
                )
        if not set(old_sources).issubset({s["source_id"] for s in current}):
            raise ValueError("Historical cohort is no longer complete")
        old_segments = read_rows(
            Path(self.config["historical_external"]) / "segments.jsonl"
        )
        external = read_rows(Path(self.config["external"]) / "segments.jsonl")
        by_id = {r["id"]: r for r in external}
        historical_pcm = {}
        changed_headers = 0
        for row in old_segments:
            new = by_id.get(row["id"])
            if new is None or any(
                new[k] != row[k] for k in ["start_seconds", "end_seconds", "language"]
            ):
                raise ValueError("Historical segment boundaries/audio differ")
            # libsndfile FLOAT WAVs contain a creation-time PEAK timestamp.
            pcm = audio_fingerprint(row["path"])
            if audio_fingerprint(new["path"]) != pcm:
                raise ValueError("Historical decoded audio samples differ")
            historical_pcm[row["id"]] = pcm
            changed_headers += row["sha256"] != new["sha256"]
        bind(
            self.root / "external_inventory.json",
            {
                "sources": current,
                "historical_sources": list(old_sources),
                "added_sources": [
                    s for s in current if s["source_id"] not in old_sources
                ],
                "historical_segment_ids": [r["id"] for r in old_segments],
                "historical_pcm_fingerprints": historical_pcm,
                "historical_changed_wav_headers": changed_headers,
                "snapshot_sha256": file_hash(
                    Path(self.config["external"]) / "segments_ready.json"
                ),
            },
        )
        original = json.loads((self.base / "data_ready.json").read_text())
        from .extension_data import BASE

        for split, cohort in [("train", "vox_train"), ("validation", "vox_validation")]:
            if (
                file_hash(BASE / "manifests" / f"{split}.jsonl")
                != original["sources"][cohort]
            ):
                raise ValueError("Canonical recording-based VoxLingua manifest changed")
            bundle = torch.load(
                BASE / f"vox_{split}.pt", map_location="cpu", weights_only=True
            )
            original_rows = read_rows(self.base / "cohorts" / f"{cohort}.jsonl")
            if [r["id"] for r in original_rows] != bundle["ids"] or [
                CODES.index(r["language"]) for r in original_rows
            ] != bundle["targets"].tolist():
                raise ValueError("Training/sampler order differs from SpeechBrain15")
            del bundle, original_rows
            gc.collect()
        old_training = read_rows(self.base / "cohorts/vox_train.jsonl")
        old_selection = read_rows(self.base / "cohorts/vox_validation.jsonl")
        if {r["group"] for r in old_training} & {r["group"] for r in old_selection}:
            raise ValueError("Original recording partitions overlap")
        del old_training, old_selection
        if {
            r["id"] for r in read_rows(self.base / "cohorts/fleurs_validation.jsonl")
        } & {r["id"] for r in read_rows(self.base / "cohorts/fleurs_test.jsonl")}:
            raise ValueError("FLEURS calibration/test overlap")
        partitions = {
            s: read_rows(self.pacific / f"new_{s}.jsonl")
            for s in ["train", "selection", "calibration", "test"]
        }
        audit_partitions(partitions)
        if [len(partitions[s]) for s in partitions] != [35668, 2325, 2284, 4545]:
            raise ValueError("Pacific manifest counts differ from original experiment")
        cohort_dir = self.root / "cohorts"
        cohort_dir.mkdir(exist_ok=True)
        metadata = {}
        sources = {}
        for name in [
            "vox_train",
            "vox_validation",
            "fleurs_validation",
            "fleurs_test",
            *[f"new_{s}" for s in partitions],
            "external",
        ]:
            if name in original["cohorts"] and name != "external":
                source = self.base / "cohorts" / f"{name}.jsonl"
                if file_hash(source) != original["cohorts"][name]["sha256"]:
                    raise ValueError("Whisper12 cohort changed")
                rows = read_rows(source)
            elif name.startswith("new_"):
                source = self.pacific / f"{name}.jsonl"
                rows = partitions[name[4:]]
            else:
                source = Path(self.config["external"]) / "segments.jsonl"
                rows = external
            for row in rows:
                check_snapshot(row)
            target = cohort_dir / f"{name}.jsonl"
            bind(
                cohort_dir / f"{name}.identity.json",
                {"source": str(source), "sha256": file_hash(source)},
            )
            if not target.exists():
                shutil.copyfile(source, target)
            if file_hash(source) != file_hash(target):
                raise ValueError("Copied cohort differs")
            metadata[name] = {
                "n": len(rows),
                "languages": dict(Counter(r["language"] for r in rows)),
                "sha256": file_hash(target),
            }
            sources[name] = file_hash(source)
        bind(
            self.root / "data_ready.json",
            {
                "cohorts": metadata,
                "sources": sources,
                "codes": CODES,
                "training_split": "original recording-based; no speaker labels",
                "external_snapshot": snapshot["identity"],
            },
        )
        for filename in ["model_identity.json", "environment.json"]:
            bind(self.root / filename, json.loads((self.base / filename).read_text()))
        model_identity = json.loads((self.root / "model_identity.json").read_text())
        if (
            file_hash(self.base / "pretrained" / f"{model_identity['model']}.pt")
            != model_identity["weight_sha256"]
        ):
            raise ValueError("Pretrained Whisper weights changed")
        pretrained = self.root / "pretrained"
        if not pretrained.exists():
            pretrained.symlink_to(
                (self.base / "pretrained").resolve(), target_is_directory=True
            )
        self.initialize()
        self.status("data_ready", cohorts=metadata)

    def initialize(self):
        checkpoint = self.base / "whisper_frozen/model.pt"
        old = torch.load(checkpoint, map_location="cpu", weights_only=True)
        head = expanded_head(old["head"], self.config["seed"])
        identity = {
            "base_checkpoint": file_hash(checkpoint),
            "seed": self.config["seed"],
            "codes": CODES,
            "head": state_fingerprint(head),
            "encoder": json.loads((self.root / "model_identity.json").read_text())[
                "weight_sha256"
            ],
        }
        bind(self.root / "initialization.json", identity)
        path = self.root / "initial.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            check = expanded_head(old["head"])
            check.load_state_dict(saved["head"])
            if (
                saved["identity"] != identity
                or state_fingerprint(check) != identity["head"]
            ):
                raise ValueError("Initial weights changed")
        else:
            save_torch(path, {"head": cpu_state(head), "identity": identity})

    def cache_all(self):
        (self.root / "cache").mkdir(exist_ok=True)
        for name in ["vox_train", "vox_validation", "fleurs_validation", "fleurs_test"]:
            source = self.base / "cache" / name
            saved = json.loads((source / "complete.json").read_text())
            identity = json.loads((source / "identity.json").read_text())
            if (
                identity["rows"] != file_hash(self.root / "cohorts" / f"{name}.jsonl")
                or saved["identity"] != digest(identity)
                or saved["features_sha256"] != file_hash(source / "features.npy")
            ):
                raise ValueError("Original Whisper cache is stale")
            bind(
                self.root / f"reused_{name}.json",
                {"identity": identity, "complete": saved},
            )
            target = self.root / "cache" / name
            if not target.exists():
                target.symlink_to(source.resolve(), target_is_directory=True)
        for name in [
            "new_train",
            "new_selection",
            "new_calibration",
            "new_test",
            "external",
        ]:
            self.cache(name, native=name == "external" and self.config["model"] == "tiny")

    def cache(self, cohort, native=False):
        directory = self.root / "cache" / cohort
        directory.mkdir(parents=True, exist_ok=True)
        bind(
            directory / "frontend_identity.json",
            {
                name: file_hash(Path(__file__).with_name(name))
                for name in ["whisper_model.py", "data.py"]
            },
        )
        super().cache(cohort, native)

    def combined_rows(self, split):
        return self.rows(
            "vox_train" if split == "train" else "vox_validation"
        ) + self.rows("new_train" if split == "train" else "new_selection")

    def feature_scores(self, head, cohorts):
        values = []
        with torch.inference_mode():
            for cohort in cohorts:
                features = np.load(
                    self.root / "cache" / cohort / "features.npy", mmap_mode="r"
                )
                for start in range(0, len(features), 256):
                    batch = torch.from_numpy(
                        np.array(features[start : start + 256])
                    ).to("cuda")
                    values.append(head(batch).float().cpu().numpy())
        return np.concatenate(values)

    def frozen(self):
        self.train("frozen")

    def finetuned(self):
        self.train("finetuned")

    def train(self, method):
        output = self.root / method
        output.mkdir(exist_ok=True)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        adapter = self.adapter_class(
            self.root, self.root / "initial.pt", method == "finetuned"
        )
        initial_encoder = state_fingerprint(adapter.encoder)
        identity = {
            "config": self.config,
            "data": file_hash(self.root / "data_ready.json"),
            "initialization": file_hash(self.root / "initial.pt"),
            "initial_encoder": initial_encoder,
            "initial_head": state_fingerprint(adapter.head),
            "method": method,
            "implementation": {
                name: file_hash(Path(__file__).with_name(name))
                for name in [
                    "whisper_extension.py",
                    "whisper_model.py",
                    "extension_train.py",
                    "data.py",
                ]
            },
        }
        bind(output / "identity.json", identity)
        if (output / "training_complete.json").exists():
            adapter.release()
            return
        rows = self.combined_rows("train")
        selection = self.combined_rows("selection")
        targets = torch.tensor([CODES.index(r["language"]) for r in rows])
        validation_targets = torch.tensor(
            [CODES.index(r["language"]) for r in selection]
        )
        features = None
        if method == "frozen":
            features = [
                np.load(self.root / "cache" / name / "features.npy", mmap_mode="r")
                for name in ["vox_train", "new_train"]
            ]
        adapter.configure(True)
        microbatch = 32 if method == "frozen" else self.config["microbatch"]
        if method == "finetuned":
            if self.config["model"] == "small":
                preflight = json.loads((self.root / "preflight.json").read_text())
                microbatch = preflight["microbatch"]
                adapter.gradient_checkpointing = preflight["gradient_checkpointing"]
            # Exercise real audio and the complete encoder graph before training.
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(
                    adapter.scores(rows[:microbatch]), targets[:microbatch].to("cuda")
                )
            loss.backward()
            grads = [p.grad for p in adapter.encoder.parameters() if p.requires_grad]
            if not grads or any(
                g is None or not torch.isfinite(g).all() for g in grads
            ):
                raise ValueError("Missing/nonfinite encoder gradients")
            peak = torch.cuda.max_memory_reserved()
            if peak >= 6.5 * 1024**3:
                raise RuntimeError("Whisper preflight exceeds GPU budget")
            write_json(
                output / "preflight.json",
                {
                    "microbatch": microbatch,
                    "encoder_gradients_verified": True,
                    "peak_reserved_bytes": peak,
                },
            )
            del grads, loss
            adapter.encoder.zero_grad(set_to_none=True)
            adapter.head.zero_grad(set_to_none=True)
        groups = [{"params": adapter.head.parameters(), "lr": self.config["head_lr"]}]
        if method == "finetuned":
            groups.append(
                {
                    "params": adapter.encoder.parameters(),
                    "lr": self.config["encoder_lr"],
                }
            )
        optimizer = torch.optim.AdamW(groups, weight_decay=self.config["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", init_scale=1024)
        epoch, cursor, stale, best, history = 1, 0, 0, (-1.0, float("-inf")), []
        resume = output / "resume.pt"
        if resume.exists():
            epoch, cursor, stale, best, history = restore_resume(
                resume, identity, adapter, optimizer, scaler
            )

        def checkpoint(optimizer=optimizer, scaler=scaler):
            save_torch(
                resume,
                {
                    "identity": identity,
                    "encoder": cpu_state(adapter.encoder),
                    "head": cpu_state(adapter.head),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "cursor": cursor,
                    "stale": stale,
                    "best": best,
                    "history": history,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
            )

        while epoch <= self.config["epochs"] and stale < self.config["patience"]:
            adapter.configure(True)
            order = epoch_order(targets, epoch, self.config["examples_per_epoch"])
            # Both variants, and SpeechBrain15, use this exact index order.
            bind(
                self.root / f"order_epoch_{epoch:02d}.json",
                {
                    "sha256": digest(order.tolist()),
                    "labels": digest(targets[order].tolist()),
                },
            )
            for start in range(cursor, len(order), 32):
                logical = order[start : start + 32]
                for attempt in range(8):
                    optimizer.zero_grad(set_to_none=True)
                    for subset in logical.split(microbatch):
                        with torch.autocast("cuda", dtype=torch.float16):
                            if features is None:
                                scores = adapter.scores(
                                    [rows[i] for i in subset.tolist()]
                                )
                            else:
                                batch = np.stack(
                                    [
                                        features[0][i]
                                        if i < len(features[0])
                                        else features[1][i - len(features[0])]
                                        for i in subset.tolist()
                                    ]
                                )
                                scores = adapter.head(
                                    torch.from_numpy(batch).to("cuda")
                                ).float()
                            loss = (
                                F.cross_entropy(scores, targets[subset].to("cuda"))
                                * len(subset)
                                / len(logical)
                            )
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training loss")
                        scaler.scale(loss).backward()
                    if scaled_optimizer_step(scaler, optimizer):
                        break
                else:
                    raise ValueError("AMP overflow persisted for eight attempts")
                cursor = start + len(logical)
                if cursor % 1024 == 0 or cursor == len(order):
                    checkpoint()
                    free, total = torch.cuda.mem_get_info()
                    if total - free >= 7 * 1024**3:
                        raise RuntimeError("GPU memory reached 7 GiB")
                    self.status(
                        "train",
                        method=method,
                        epoch=epoch,
                        completed=cursor,
                        total=len(order),
                        stale_epochs=stale,
                    )
            adapter.configure(False)
            if method == "frozen":
                scores = self.feature_scores(
                    adapter.head, ["vox_validation", "new_selection"]
                )
            else:
                scores = self.score_adapter(
                    adapter,
                    selection,
                    output / f"selection_epoch_{epoch:02d}.npz",
                    f"selection_{epoch}",
                )
            metrics = selection_metrics(torch.from_numpy(scores), validation_targets)
            history.append({"epoch": epoch, **metrics})
            current = (metrics["macro_f1"], -metrics["balanced_nll"])
            actual = state_fingerprint(adapter.encoder)
            if (actual == initial_encoder) != (method == "frozen"):
                raise ValueError("Encoder freeze/update verification failed")
            if current > best:
                best, stale = current, 0
                save_torch(
                    output / "model.pt",
                    {
                        "encoder": cpu_state(adapter.encoder),
                        "head": cpu_state(adapter.head),
                        "epoch": epoch,
                        "codes": CODES,
                        "identity": identity,
                        "selection_metrics": metrics,
                    },
                )
            else:
                stale += 1
            write_json(output / "history.json", history)
            self.status(
                "epoch_complete", method=method, **history[-1], stale_epochs=stale
            )
            epoch, cursor = epoch + 1, 0
            checkpoint()
        write_json(
            output / "training_complete.json",
            {
                "epochs": len(history),
                "best_macro_f1": best[0],
                "best_balanced_nll": -best[1],
                "encoder_unchanged": method == "frozen",
                "encoder_update_verified": method == "finetuned",
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        )
        del checkpoint, optimizer, scaler, rows, selection, features
        adapter.release()
        gc.collect()

    def evaluate(self):
        from .whisper_extension_results import evaluate_models

        evaluate_models(self)

    def baselines(self):
        from .whisper_extension_results import baselines

        baselines(self)

    def diagnostics(self):
        from .whisper_extension_results import diagnostics

        diagnostics(self)

    def export(self):
        from .whisper_extension_export import export_models

        export_models(self)

    def report(self):
        from .whisper_extension_results import report

        report(self)
