"""Matched original-split Whisper tiny versus SpeechBrain LID experiments."""

import gc
import importlib.metadata
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .extension_data import BASE, file_hash, read_rows, write_json
from .languages import CODES, NAMES
from .metrics import evaluate, fit_temperature
from .model import cpu_state
from .speaker_pilot import bind, digest
from .whisper_model import (
    Adapter,
    load_whisper,
    mel_batch,
    native_logits,
    pool_real_tokens,
    selected_languages,
    tokenizer_for,
    whisper_head,
)


def save_torch(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def balanced_order(targets, epoch, count, seed=42):
    generator = torch.Generator().manual_seed(seed + epoch)
    labels = torch.randint(len(CODES), (count,), generator=generator)
    result = torch.empty(count, dtype=torch.long)
    for index in range(len(CODES)):
        pool = (targets == index).nonzero().flatten()
        if not len(pool):
            raise ValueError("Missing training language")
        mask = labels == index
        result[mask] = pool[
            torch.randint(len(pool), (int(mask.sum()),), generator=generator)
        ]
    return result


class Study:
    def __init__(
        self, config="configs/whisper12.json", root="artifacts/whisper_tiny12"
    ):
        self.config = json.loads(Path(config).read_text())
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if (
            self.config["model"] not in ("tiny", "small")
            or self.config["max_seconds"] != 10
            or self.config["whisper_window_seconds"] != 30
        ):
            raise ValueError(
                "This matched protocol requires multilingual tiny or small and 10s real audio in a native 30s window"
            )
        bind(self.root / "configuration.json", self.config)

    def status(self, phase, **details):
        details = {
            key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in details.items()
        }
        data = {"phase": phase, "updated_at": datetime.now(UTC).isoformat(), **details}
        write_json(self.root / "status.json", data)
        print(json.dumps(data), flush=True)

    def rows(self, cohort):
        path = self.root / "cohorts" / f"{cohort}.jsonl"
        expected = json.loads((self.root / "data_ready.json").read_text())["cohorts"][
            cohort
        ]["sha256"]
        if file_hash(path) != expected:
            raise ValueError("Cohort manifest changed")
        return read_rows(path)

    def prepare(self):
        import whisper

        from .data import read_manifest
        from .fleurs import REVISION, records
        from .youtube_data import inventory

        versions = {
            name: importlib.metadata.version(name)
            for name in ["openai-whisper", "torch", "torchaudio", "numpy"]
        }
        if versions["openai-whisper"] != self.config["whisper_version"]:
            raise ValueError("Whisper package version changed")
        bind(
            self.root / "environment.json",
            {"versions": versions, "uv_lock_sha256": file_hash("uv.lock")},
        )
        directory = self.root / "pretrained"
        directory.mkdir(exist_ok=True)
        model = whisper.load_model("tiny", device="cpu", download_root=str(directory))
        _, codes = tokenizer_for(model)
        checksum = whisper._MODELS["tiny"].split("/")[-2]
        if file_hash(directory / "tiny.pt") != checksum:
            raise ValueError("Whisper checkpoint checksum mismatch")
        bind(
            self.root / "model_identity.json",
            {
                "model": "tiny",
                "weight_sha256": checksum,
                "download_url": whisper._MODELS["tiny"],
                "language_codes": codes,
                "dims": vars(model.dims),
                "version": versions["openai-whisper"],
            },
        )
        del model
        gc.collect()
        cohort_dir = self.root / "cohorts"
        cohort_dir.mkdir(exist_ok=True)
        sources = {}
        prepared = {}
        for name, cache in [
            ("vox_train", "vox_train.pt"),
            ("vox_validation", "vox_validation.pt"),
            ("fleurs_validation", "fleurs_direct_validation.pt"),
            ("fleurs_test", "fleurs_direct_test.pt"),
        ]:
            bundle = torch.load(BASE / cache, map_location="cpu", weights_only=True)
            ids = bundle["ids"]
            targets = bundle["targets"].tolist()
            if name.startswith("vox"):
                split = name.removeprefix("vox_")
                manifest = BASE / "manifests" / f"{split}.jsonl"
                candidates = read_manifest(manifest)
                sources[name] = file_hash(manifest)
            else:
                split = name.removeprefix("fleurs_")
                candidates = list(records(split, REVISION))
                sources[name] = REVISION
            by_id = {row["id"]: row for row in candidates}
            if set(ids) != set(by_id):
                raise ValueError(f"Original SpeechBrain IDs differ: {name}")
            rows = []
            for identifier, target in zip(ids, targets):
                row = by_id[identifier]
                if CODES[target] != row["language"]:
                    raise ValueError("Original target ordering differs")
                stat = Path(row["path"]).stat()
                rows.append(dict(row, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
            prepared[name] = rows
            del bundle, candidates, by_id
            gc.collect()
        if {r["group"] for r in prepared["vox_train"]} & {
            r["group"] for r in prepared["vox_validation"]
        }:
            raise ValueError("Original VoxLingua train/selection recording overlap")
        if {r["id"] for r in prepared["fleurs_validation"]} & {
            r["id"] for r in prepared["fleurs_test"]
        }:
            raise ValueError("FLEURS calibration/test overlap")
        external = Path(self.config["external"])
        previous = json.loads((external / "segments_ready.json").read_text())
        current = inventory(
            "validation_clips", languages=["sam", "niu", "ton", "hi", "th"]
        )
        if current != previous["configuration"]["sources"]:
            raise ValueError(
                "External clips changed; prepare a fresh matched VAD validation snapshot first"
            )
        prepared["external"] = read_rows(external / "segments.jsonl")
        for row in prepared["external"]:
            if file_hash(row["path"]) != row["sha256"]:
                raise ValueError("External speech segment changed")
        metadata = {}
        for name, rows in prepared.items():
            path = cohort_dir / f"{name}.jsonl"
            identity = digest(rows)
            marker = cohort_dir / f"{name}.identity.json"
            bind(marker, {"rows": identity})
            if not path.exists():
                with path.open("w") as stream:
                    stream.writelines(
                        json.dumps(row, sort_keys=True) + "\n" for row in rows
                    )
            if digest(read_rows(path)) != identity:
                raise ValueError("Existing cohort differs")
            metadata[name] = {
                "n": len(rows),
                "languages": dict(Counter(r["language"] for r in rows)),
                "sha256": file_hash(path),
            }
        bind(
            self.root / "data_ready.json",
            {
                "cohorts": metadata,
                "sources": sources,
                "external_snapshot": file_hash(external / "segments_ready.json"),
                "training_split": "original recording-based; no speaker labels",
                "codes": CODES,
            },
        )
        self.status("data_ready", cohorts=metadata)

    def cache_eval(self):
        for cohort in ["fleurs_validation", "fleurs_test", "external"]:
            self.cache(cohort, native=True)

    def cache_train(self):
        self.cache("vox_validation", native=True)
        self.filtered()
        self.cache("vox_train", native=False)

    def cache(self, cohort, native=False):
        directory = self.root / "cache" / cohort
        directory.mkdir(parents=True, exist_ok=True)
        rows = self.rows(cohort)
        model_identity = json.loads((self.root / "model_identity.json").read_text())
        identity = {
            "model": model_identity,
            "rows": file_hash(self.root / "cohorts" / f"{cohort}.jsonl"),
            "config": self.config,
            "native": native,
        }
        bind(directory / "identity.json", identity)
        marker = directory / "complete.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if (
                saved["identity"] != digest(identity)
                or file_hash(directory / "features.npy") != saved["features_sha256"]
            ):
                raise ValueError("Completed Whisper cache changed")
            if (
                native
                and file_hash(directory / "native_logits.npy")
                != saved["native_logits_sha256"]
            ):
                raise ValueError("Completed native Whisper logits changed")
            return
        model = load_whisper(self.root)
        width = model.dims.n_audio_state * 2
        language_codes = model_identity["language_codes"]
        parts = directory / "parts"
        parts.mkdir(exist_ok=True)
        with torch.inference_mode():
            for start in range(0, len(rows), 128):
                path = parts / f"part-{start:06d}.npz"
                selected = rows[start : start + 128]
                if path.exists():
                    with np.load(path) as saved:
                        if saved["identity"].item() != digest(identity) or saved[
                            "ids"
                        ].tolist() != [r["id"] for r in selected]:
                            raise ValueError("Whisper cache shard identity changed")
                    continue
                features, logits = [], []
                for offset in range(0, len(selected), self.config["cache_batch"]):
                    subset = selected[offset : offset + self.config["cache_batch"]]
                    mels, lengths = mel_batch(subset)
                    encoded = model.encoder(mels)
                    pooled = pool_real_tokens(encoded, lengths)
                    features.append(pooled.cpu().numpy())
                    if native:
                        scores, codes = native_logits(model, encoded)
                        if codes != language_codes:
                            raise ValueError("Whisper language ordering changed")
                        logits.append(scores.cpu().numpy())
                values = np.concatenate(features)
                scores = (
                    np.concatenate(logits)
                    if logits
                    else np.empty((len(selected), 0), dtype=np.float32)
                )
                if not np.isfinite(values).all() or not np.isfinite(scores).all():
                    raise ValueError("Nonfinite Whisper output")
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary,
                    features=values,
                    logits=scores,
                    ids=[r["id"] for r in selected],
                    identity=digest(identity),
                )
                temporary.replace(path)
                self.status(
                    "cache_whisper",
                    cohort=cohort,
                    completed=start + len(selected),
                    total=len(rows),
                )
        feature_path = directory / "features.npy"
        features = np.lib.format.open_memmap(
            feature_path.with_suffix(".tmp.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), width),
        )
        logits = (
            np.lib.format.open_memmap(
                directory / "native_logits.tmp.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(rows), len(language_codes)),
            )
            if native
            else None
        )
        for start in range(0, len(rows), 128):
            with np.load(parts / f"part-{start:06d}.npz") as saved:
                features[start : start + len(saved["features"])] = saved["features"]
                if native:
                    logits[start : start + len(saved["logits"])] = saved["logits"]
        features.flush()
        del features
        feature_path.with_suffix(".tmp.npy").replace(feature_path)
        if native:
            logits.flush()
            del logits
            (directory / "native_logits.tmp.npy").replace(
                directory / "native_logits.npy"
            )
        write_json(
            marker,
            {
                "identity": digest(identity),
                "n": len(rows),
                "features_sha256": file_hash(feature_path),
                "native_logits_sha256": file_hash(directory / "native_logits.npy")
                if native
                else None,
            },
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def filtered(self):
        codes = json.loads((self.root / "model_identity.json").read_text())[
            "language_codes"
        ]
        indices = selected_languages(codes)
        for method, selection in [
            ("whisper_native", list(range(len(codes)))),
            ("whisper_filtered", indices),
        ]:
            directory = self.root / method
            directory.mkdir(exist_ok=True)
            cohorts = ["fleurs_validation", "fleurs_test", "external"]
            if (self.root / "cache/vox_validation/native_logits.npy").exists():
                cohorts.append("vox_validation")
            for cohort in cohorts:
                data = np.load(
                    self.root / "cache" / cohort / "native_logits.npy", mmap_mode="r"
                )
                self.save_scores(
                    directory,
                    cohort,
                    data[:, selection],
                    codes if method == "whisper_native" else CODES,
                )
            self.calibrate_and_evaluate(method)
        self.report()

    def save_scores(self, directory, cohort, logits, codes):
        rows = self.rows(cohort)
        path = directory / f"{cohort}.npz"
        identity = {
            "config": self.config,
            "model_sha256": file_hash(directory / "model.pt")
            if (directory / "model.pt").exists()
            else file_hash(self.root / "pretrained/tiny.pt"),
            "rows": file_hash(self.root / "cohorts" / f"{cohort}.jsonl"),
            "codes": list(codes),
        }
        if path.exists():
            with np.load(path) as old:
                if (
                    old["identity"].item() != digest(identity)
                    or old["ids"].tolist() != [r["id"] for r in rows]
                    or not np.allclose(old["logits"], logits, atol=1e-5)
                ):
                    raise ValueError("Stored score identity differs")
            return
        np.savez_compressed(
            path,
            ids=[r["id"] for r in rows],
            logits=logits,
            codes=codes,
            identity=digest(identity),
        )

    def calibrate_and_evaluate(self, method):
        directory = self.root / method
        with np.load(directory / "fleurs_validation.npz") as saved:
            codes = saved["codes"].tolist()
            cal = saved["logits"]
        rows = self.rows("fleurs_validation")
        targets = np.array([codes.index(r["language"]) for r in rows])
        temperature = fit_temperature(cal, targets)
        identity = {
            "calibration_scores_sha256": file_hash(directory / "fleurs_validation.npz"),
            "temperature": temperature,
            "fit": "FLEURS validation NLL only",
            "calibration_ids_sha256": digest([r["id"] for r in rows]),
            "codes": codes,
        }
        bind(directory / "calibration.json", identity)
        with np.load(directory / "fleurs_test.npz") as saved:
            scores = saved["logits"]
        targets = np.array(
            [codes.index(r["language"]) for r in self.rows("fleurs_test")]
        )
        indices = [codes.index(c) for c in CODES]
        reports = {
            variant: evaluate(
                scores, targets, NAMES, temperature=t, report_indices=indices
            )
            for variant, t in [("raw", 1.0), ("calibrated", temperature)]
        }
        write_json(
            directory / "fleurs_report.json",
            {
                "method": method,
                "metrics": reports,
                "codes": codes,
                "calibration": identity,
            },
        )
        write_json(
            directory / "evaluation_complete.json",
            {"fleurs_test_n": len(targets), "temperature": temperature},
        )

    def frozen(self):
        directory = self.root / "whisper_frozen"
        directory.mkdir(exist_ok=True)
        if not (directory / "training_complete.json").exists():
            self.train_frozen(directory)
        width = json.loads((self.root / "model_identity.json").read_text())["dims"]["n_audio_state"]
        head = whisper_head(width).to("cuda")
        saved = torch.load(
            directory / "model.pt", map_location="cpu", weights_only=True
        )
        head.load_state_dict(saved["head"])
        head.eval()
        with torch.inference_mode():
            for cohort in [
                "fleurs_validation",
                "fleurs_test",
                "external",
                "vox_validation",
            ]:
                features = np.load(
                    self.root / "cache" / cohort / "features.npy", mmap_mode="r"
                )
                values = []
                for start in range(0, len(features), 256):
                    batch = torch.from_numpy(
                        np.array(features[start : start + 256])
                    ).to("cuda")
                    values.append(head(batch).cpu().numpy())
                self.save_scores(directory, cohort, np.concatenate(values), CODES)
        del head, saved
        gc.collect()
        torch.cuda.empty_cache()
        self.calibrate_and_evaluate("whisper_frozen")
        self.report()

    def train_frozen(self, directory):
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        features = np.load(self.root / "cache/vox_train/features.npy", mmap_mode="r")
        validation = np.load(
            self.root / "cache/vox_validation/features.npy", mmap_mode="r"
        )
        targets = torch.tensor(
            [CODES.index(r["language"]) for r in self.rows("vox_train")]
        )
        validation_targets = torch.tensor(
            [CODES.index(r["language"]) for r in self.rows("vox_validation")]
        )
        width = json.loads((self.root / "model_identity.json").read_text())["dims"]["n_audio_state"]
        head = whisper_head(width).to("cuda")
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=self.config["head_lr"],
            weight_decay=self.config["weight_decay"],
        )
        counts = torch.bincount(targets, minlength=12).float()
        weights = (counts.sum() / (12 * counts)).to("cuda")
        identity = {
            "config": self.config,
            "train": file_hash(self.root / "cache/vox_train/features.npy"),
            "validation": file_hash(self.root / "cache/vox_validation/features.npy"),
        }
        bind(directory / "identity.json", identity)
        epoch, cursor, stale, best, history = 1, 0, 0, float("inf"), []
        resume = directory / "resume.pt"
        if resume.exists():
            state = torch.load(resume, map_location="cpu", weights_only=False)
            if state["identity"] != identity:
                raise ValueError("Frozen resume identity changed")
            head.load_state_dict(state["head"])
            optimizer.load_state_dict(state["optimizer"])
            epoch, cursor, stale, best, history = [
                state[k] for k in ["epoch", "cursor", "stale", "best", "history"]
            ]

        def checkpoint(current_head=head, current_optimizer=optimizer):
            save_torch(
                resume,
                {
                    "identity": identity,
                    "head": cpu_state(current_head),
                    "optimizer": current_optimizer.state_dict(),
                    "epoch": epoch,
                    "cursor": cursor,
                    "stale": stale,
                    "best": best,
                    "history": history,
                },
            )

        while epoch <= self.config["frozen_epochs"] and stale < self.config["patience"]:
            head.train()
            order = torch.randperm(
                len(targets),
                generator=torch.Generator().manual_seed(self.config["seed"] + epoch),
            )
            for start in range(cursor, len(order), self.config["frozen_batch"]):
                subset = order[start : start + self.config["frozen_batch"]]
                batch = torch.from_numpy(np.array(features[subset.numpy()])).to("cuda")
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(
                    head(batch), targets[subset].to("cuda"), weight=weights
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite frozen loss")
                loss.backward()
                optimizer.step()
                cursor = start + len(subset)
                if cursor % 16384 == 0:
                    checkpoint()
                    self.status(
                        "train_whisper_frozen",
                        epoch=epoch,
                        completed=cursor,
                        total=len(order),
                        best_nll=best,
                    )
            head.eval()
            scores = []
            with torch.inference_mode():
                for start in range(0, len(validation), 256):
                    scores.append(
                        head(
                            torch.from_numpy(
                                np.array(validation[start : start + 256])
                            ).to("cuda")
                        ).cpu()
                    )
            scores = torch.cat(scores)
            nll = float(F.cross_entropy(scores, validation_targets))
            history.append(
                {
                    "epoch": epoch,
                    "validation_nll": nll,
                    "accuracy": float(
                        (scores.argmax(1) == validation_targets).float().mean()
                    ),
                }
            )
            if nll < best:
                best, stale = nll, 0
                save_torch(
                    directory / "model.pt",
                    {"head": cpu_state(head), "epoch": epoch, "identity": identity},
                )
            else:
                stale += 1
            write_json(directory / "history.json", history)
            self.status(
                "epoch_complete_whisper_frozen", **history[-1], stale_epochs=stale
            )
            epoch, cursor = epoch + 1, 0
            checkpoint()
        write_json(
            directory / "training_complete.json",
            {"epochs": len(history), "best_validation_nll": best},
        )
        del checkpoint, head, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    def whisper_finetune(self):
        self.train_encoder("whisper")

    def speechbrain_finetune(self):
        self.train_encoder("speechbrain")

    def train_encoder(self, family):
        from .extension_model import state_fingerprint
        from .extension_train import scaled_optimizer_step

        method = f"{family}_finetuned"
        directory = self.root / method
        directory.mkdir(exist_ok=True)
        if not (directory / "training_complete.json").exists():
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            warm = (
                self.root / "whisper_frozen/model.pt" if family == "whisper" else None
            )
            adapter = Adapter(family, self.root, warm)
            initial_encoder = state_fingerprint(adapter.encoder)
            rows = self.rows("vox_train")
            validation = self.rows("vox_validation")
            targets = torch.tensor([CODES.index(r["language"]) for r in rows])
            validation_targets = torch.tensor(
                [CODES.index(r["language"]) for r in validation]
            )
            identity = {
                "config": self.config,
                "data": file_hash(self.root / "data_ready.json"),
                "family": family,
                "initial_encoder": initial_encoder,
                "warm_head": file_hash(warm or BASE / "head/head.pt"),
                "policy": "all encoder parameters and classifier; SpeechBrain BN running statistics frozen",
                "selection": "VoxLingua validation NLL; FLEURS/external not used",
            }
            bind(directory / "identity.json", identity)
            adapter.configure(True)
            microbatch = None
            for candidate in [self.config["microbatch"], 1]:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    adapter.encoder.zero_grad(set_to_none=True)
                    adapter.head.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = F.cross_entropy(
                            adapter.scores(rows[:candidate]),
                            targets[:candidate].to("cuda"),
                        )
                    loss.backward()
                    gradients = [
                        p.grad
                        for p in adapter.encoder.parameters()
                        if p.grad is not None
                    ]
                    if not gradients or not all(
                        torch.isfinite(g).all() for g in gradients
                    ):
                        raise ValueError(
                            "Encoder preflight has invalid or absent gradients"
                        )
                    peak = torch.cuda.max_memory_reserved()
                    adapter.encoder.zero_grad(set_to_none=True)
                    adapter.head.zero_grad(set_to_none=True)
                    del gradients, loss
                    if peak < 6.5 * 1024**3:
                        microbatch = candidate
                        break
                except torch.cuda.OutOfMemoryError:
                    adapter.encoder.zero_grad(set_to_none=True)
                    adapter.head.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
            if microbatch is None:
                raise RuntimeError("Fine-tuning preflight exceeds the GPU budget")
            write_json(
                directory / "preflight.json",
                {
                    "microbatch": microbatch,
                    "peak_reserved_bytes": peak,
                    "encoder_gradient_verified": True,
                    "trainable_encoder_parameters": sum(
                        p.numel()
                        for p in adapter.encoder.parameters()
                        if p.requires_grad
                    ),
                },
            )
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": adapter.encoder.parameters(),
                        "lr": self.config["encoder_lr"],
                    },
                    {"params": adapter.head.parameters(), "lr": self.config["head_lr"]},
                ],
                weight_decay=self.config["weight_decay"],
            )
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            epoch, cursor, stale, best, history = 1, 0, 0, float("inf"), []
            resume = directory / "resume.pt"
            if resume.exists():
                state = torch.load(resume, map_location="cpu", weights_only=False)
                if state["identity"] != identity:
                    raise ValueError("Fine-tuning resume identity changed")
                adapter.encoder.load_state_dict(state["encoder"])
                adapter.head.load_state_dict(state["head"])
                optimizer.load_state_dict(state["optimizer"])
                scaler.load_state_dict(state["scaler"])
                epoch, cursor, stale, best, history = [
                    state[k] for k in ["epoch", "cursor", "stale", "best", "history"]
                ]
                torch.set_rng_state(state["torch_rng"])
                torch.cuda.set_rng_state_all(state["cuda_rng"])
                del state

            def checkpoint(
                current_adapter=adapter,
                current_optimizer=optimizer,
                current_scaler=scaler,
            ):
                save_torch(
                    resume,
                    {
                        "identity": identity,
                        "encoder": cpu_state(current_adapter.encoder),
                        "head": cpu_state(current_adapter.head),
                        "optimizer": current_optimizer.state_dict(),
                        "scaler": current_scaler.state_dict(),
                        "epoch": epoch,
                        "cursor": cursor,
                        "stale": stale,
                        "best": best,
                        "history": history,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                    },
                )

            while (
                epoch <= self.config["finetune_epochs"]
                and stale < self.config["patience"]
            ):
                adapter.configure(True)
                order = balanced_order(
                    targets,
                    epoch,
                    self.config["examples_per_epoch"],
                    self.config["seed"],
                )
                effective = self.config["effective_batch"]
                for start in range(cursor, len(order), effective):
                    logical = order[start : start + effective]
                    for attempt in range(8):
                        optimizer.zero_grad(set_to_none=True)
                        for subset in logical.split(microbatch):
                            with torch.autocast("cuda", dtype=torch.float16):
                                scores = adapter.scores(
                                    [rows[i] for i in subset.tolist()]
                                )
                                loss = (
                                    F.cross_entropy(scores, targets[subset].to("cuda"))
                                    * len(subset)
                                    / len(logical)
                                )
                            if not torch.isfinite(loss):
                                raise ValueError("Nonfinite fine-tuning loss")
                            scaler.scale(loss).backward()
                        if scaled_optimizer_step(scaler, optimizer):
                            break
                    else:
                        raise ValueError(
                            "Encoder gradients overflowed on eight retries"
                        )
                    cursor = start + len(logical)
                    if cursor % 4096 == 0 or cursor == len(order):
                        checkpoint()
                        free, total = torch.cuda.mem_get_info()
                        if total - free >= 7 * 1024**3:
                            raise RuntimeError("GPU use reached 7 GiB")
                        self.status(
                            "train_encoder",
                            family=family,
                            epoch=epoch,
                            completed=cursor,
                            total=len(order),
                            best_nll=best,
                            stale_epochs=stale,
                        )
                adapter.configure(False)
                scores = self.score_adapter(
                    adapter,
                    validation,
                    directory / f"selection_epoch_{epoch:02d}.npz",
                    f"{method}_selection_{epoch}",
                )
                nll = float(
                    F.cross_entropy(torch.from_numpy(scores), validation_targets)
                )
                accuracy = float(
                    (scores.argmax(axis=1) == validation_targets.numpy()).mean()
                )
                history.append(
                    {"epoch": epoch, "validation_nll": nll, "accuracy": accuracy}
                )
                if nll < best:
                    best, stale = nll, 0
                    save_torch(
                        directory / "model.pt",
                        {
                            "identity": identity,
                            "encoder": cpu_state(adapter.encoder),
                            "head": cpu_state(adapter.head),
                            "epoch": epoch,
                        },
                    )
                else:
                    stale += 1
                write_json(directory / "history.json", history)
                self.status(
                    "epoch_complete_encoder",
                    family=family,
                    **history[-1],
                    stale_epochs=stale,
                )
                epoch, cursor = epoch + 1, 0
                checkpoint()
            if state_fingerprint(adapter.encoder) == initial_encoder:
                raise ValueError("Fine-tuned encoder did not change")
            write_json(
                directory / "training_complete.json",
                {
                    "epochs": len(history),
                    "best_validation_nll": best,
                    "encoder_changed": True,
                },
            )
            del checkpoint, optimizer, scaler, rows, validation
            adapter.release()
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
        saved = torch.load(
            directory / "model.pt", map_location="cpu", weights_only=True
        )
        epoch = saved["epoch"]
        del saved
        adapter = Adapter(family, self.root, directory / "model.pt")
        for cohort in ["fleurs_validation", "fleurs_test", "external"]:
            rows = self.rows(cohort)
            scores = self.score_adapter(
                adapter, rows, directory / f"{cohort}_scores.npz", f"{method}_{cohort}"
            )
            self.save_scores(directory, cohort, scores, CODES)
        with np.load(directory / f"selection_epoch_{epoch:02d}.npz") as saved:
            self.save_scores(directory, "vox_validation", saved["logits"], CODES)
        adapter.release()
        del adapter
        self.calibrate_and_evaluate(method)
        self.report()

    def score_adapter(self, adapter, rows, path, phase):
        from .extension_model import state_fingerprint

        identity = {
            "rows": digest(rows),
            "encoder": state_fingerprint(adapter.encoder),
            "head": state_fingerprint(adapter.head),
            "config": self.config,
        }
        if path.exists():
            with np.load(path) as saved:
                if saved["identity"].item() != digest(identity) or saved[
                    "ids"
                ].tolist() != [r["id"] for r in rows]:
                    raise ValueError("Encoder evaluation cache changed")
                return saved["logits"]
        parts = path.with_suffix(".parts")
        parts.mkdir(exist_ok=True)
        bind(parts / "identity.json", identity)
        values = []
        adapter.configure(False)
        with torch.inference_mode():
            for start in range(0, len(rows), 64):
                target = parts / f"part-{start:06d}.npy"
                if target.exists():
                    scores = np.load(target)
                else:
                    scores = np.concatenate(
                        [
                            adapter.scores([row]).cpu().numpy()
                            for row in rows[start : start + 64]
                        ]
                    )
                    if not np.isfinite(scores).all():
                        raise ValueError("Nonfinite encoder evaluation")
                    np.save(target, scores)
                    self.status(
                        "evaluate_encoder",
                        cohort=phase,
                        completed=start + len(scores),
                        total=len(rows),
                    )
                values.append(scores)
        scores = np.concatenate(values)
        np.savez_compressed(
            path, logits=scores, ids=[r["id"] for r in rows], identity=digest(identity)
        )
        return scores

    def diagnostics(self):
        from .whisper_diagnostics import run

        run(self)
        self.report()

    def report(self):
        from .whisper_report import write_report

        write_report(self)
