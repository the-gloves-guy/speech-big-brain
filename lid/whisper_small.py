"""Focused Small15 study with a frozen Small12 warm-start and guarded preflight."""

import gc
import importlib.metadata
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
from .extension_train import scaled_optimizer_step
from .model import cpu_state
from .speaker_pilot import bind, digest
from .whisper_extension import ExtensionAdapter, ExtensionStudy, expanded_head
from .whisper_model import load_whisper, mel_batch, pool_real_tokens, whisper_head
from .whisper_study import Study, save_torch


class ResourceFailure(RuntimeError):
    pass


class SmallAdapter(ExtensionAdapter):
    gradient_checkpointing = False

    def embeddings(self, rows):
        mels, lengths = mel_batch(rows)
        if self.gradient_checkpointing and self.encoder.training:
            from torch.utils.checkpoint import checkpoint

            # Match AudioEncoder.forward exactly, recomputing only residual blocks.
            x = F.gelu(self.encoder.conv1(mels))
            x = F.gelu(self.encoder.conv2(x)).permute(0, 2, 1)
            if x.shape[1:] != self.encoder.positional_embedding.shape:
                raise ValueError("Small encoder input shape differs from native window")
            x = (x + self.encoder.positional_embedding).to(x.dtype)
            for block in self.encoder.blocks:
                x = checkpoint(block, x, use_reentrant=False)
            encoded = self.encoder.ln_post(x)
        else:
            encoded = self.encoder(mels)
        return pool_real_tokens(encoded, lengths)


class WarmStudy(Study):
    def __init__(self, small):
        self.root = small.base
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = {**small.config, "frozen_epochs": small.config["warm_frozen_epochs"],
                       "frozen_batch": small.config["warm_frozen_batch"],
                       "selection": "original VoxLingua validation NLL", "calibration": "FLEURS validation only"}
        bind(self.root / "configuration.json", self.config)

    def prepare(self, reference):
        from .youtube_data import TARGETS, inventory

        snapshot = Path(self.config["external"])
        prepared = json.loads((snapshot / "segments_ready.json").read_text())
        if inventory("validation_clips", languages=TARGETS) != prepared["configuration"]["sources"]:
            raise ValueError("Validation inventory changed; create a fresh matched snapshot before running Small")
        metadata, sources = {}, {}
        original = json.loads((reference / "data_ready.json").read_text())
        for cohort in ["vox_train", "vox_validation", "fleurs_validation", "fleurs_test", "external"]:
            source = reference / "cohorts" / f"{cohort}.jsonl"
            if file_hash(source) != original["cohorts"][cohort]["sha256"]:
                raise ValueError("Reference cohort changed")
            records = read_rows(source)
            for row in records:
                check_snapshot(row)
                if cohort == "external" and file_hash(row["path"]) != row["sha256"]:
                    raise ValueError("External segment changed")
            target = self.root / "cohorts" / source.name
            target.parent.mkdir(exist_ok=True)
            bind(target.with_suffix(".identity.json"), {"sha256": file_hash(source)})
            if not target.exists():
                shutil.copyfile(source, target)
            if file_hash(target) != file_hash(source):
                raise ValueError("Small warm-start cohort differs from original split")
            metadata[cohort] = {"n": len(records), "languages": dict(Counter(r["language"] for r in records)), "sha256": file_hash(target)}
        # Extension preparation audits these against canonical SpeechBrain manifests.
        from .extension_data import BASE
        for split in ["train", "validation"]:
            sources[f"vox_{split}"] = file_hash(BASE / "manifests" / f"{split}.jsonl")
        bind(self.root / "data_ready.json", {"cohorts": metadata, "sources": sources,
            "codes": CODES[:12], "external_snapshot": file_hash(snapshot / "segments_ready.json"),
            "training_split": "original recording-based; no speaker labels"})

    def train(self):
        directory = self.root / "whisper_frozen"
        directory.mkdir(exist_ok=True)
        if not (directory / "training_complete.json").exists():
            self.train_frozen(directory)

    def evaluate(self):
        from .metrics import evaluate, fit_temperature
        from .languages import NAMES

        directory = self.root / "whisper_frozen"
        model = load_whisper(self.root)
        head = whisper_head(model.dims.n_audio_state).to("cuda")
        saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        head.load_state_dict(saved["head"])
        head.eval()
        del saved, model
        values = {}
        for cohort in ["fleurs_validation", "fleurs_test", "external"]:
            self.cache(cohort, native=False)
            features = np.load(self.root / "cache" / cohort / "features.npy", mmap_mode="r")
            with torch.inference_mode():
                scores = np.concatenate([head(torch.from_numpy(np.array(features[i:i+256])).to("cuda")).cpu().numpy()
                                         for i in range(0, len(features), 256)])
            self.save_scores(directory, cohort, scores, CODES[:12])
            values[cohort] = scores
        targets = np.array([CODES.index(r["language"]) for r in self.rows("fleurs_validation")])
        temperature = fit_temperature(values["fleurs_validation"], targets)
        bind(directory / "calibration.json", {"temperature": temperature, "codes": CODES[:12],
            "fit": "FLEURS validation NLL only", "model_sha256": file_hash(directory / "model.pt"),
            "calibration_scores_sha256": file_hash(directory / "fleurs_validation.npz")})
        targets = np.array([CODES.index(r["language"]) for r in self.rows("fleurs_test")])
        write_json(directory / "fleurs_report.json", {"metrics": {variant:evaluate(values["fleurs_test"], targets, NAMES, t)
            for variant,t in [("raw",1.0),("calibrated",temperature)]}})
        write_json(directory / "evaluation_complete.json", {"fleurs_test": len(targets), "purpose": "Small12 frozen warm-start baseline; no fine-tuned Small12 control"})
        del head
        gc.collect()
        torch.cuda.empty_cache()


class SmallStudy(ExtensionStudy):
    adapter_class = SmallAdapter
    extension_prefix = "whisper_small15"

    def __init__(self, config="configs/whisper_small15.json", root="artifacts/whisper_small15"):
        super().__init__(config, root)
        if self.config["model"] != "small":
            raise ValueError("The focused Small runner requires multilingual Small")
        self.reference = Path(self.config["reference"])
        self.warm = WarmStudy(self)

    def prepare_model(self):
        import whisper

        self.warm.prepare(self.reference)
        versions = {n: importlib.metadata.version(n) for n in ["openai-whisper", "torch", "torchaudio", "numpy"]}
        if versions["openai-whisper"] != self.config["whisper_version"]:
            raise ValueError("Whisper package version changed")
        bind(self.base / "environment.json", {"versions": versions, "uv_lock_sha256": file_hash("uv.lock")})
        directory = self.base / "pretrained"
        directory.mkdir(exist_ok=True)
        url = whisper._MODELS["small"]
        checksum = url.split("/")[-2]
        path = directory / "small.pt"
        if not path.exists():
            whisper._download(url, str(directory), False)
        if file_hash(path) != checksum:
            raise ValueError("Whisper Small checkpoint checksum mismatch")
        # Official weights use torch's ZIP format: mmap avoids eagerly copying the decoder.
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        dims = saved["dims"]
        encoder_path = directory / "small_encoder.pt"
        encoder = {k.removeprefix("encoder."):v for k,v in saved["model_state_dict"].items() if k.startswith("encoder.")}
        if not encoder_path.exists():
            save_torch(encoder_path, {"dims": dims, "encoder": encoder})
        # Check extracted state against the exact source, not just its dimensions.
        extracted = torch.load(encoder_path, map_location="cpu", weights_only=True, mmap=True)
        if extracted["dims"] != dims or set(extracted["encoder"]) != set(encoder) or any(
            not torch.equal(v, extracted["encoder"][k]) for k,v in encoder.items()):
            raise ValueError("Extracted Small encoder differs from pretrained weights")
        from whisper.tokenizer import get_tokenizer
        codes = sorted(get_tokenizer(multilingual=True, num_languages=99, language="en").all_language_codes)
        bind(self.base / "model_identity.json", {"model": "small", "weight_sha256": checksum, "download_url": url,
            "dims": dims, "language_codes": codes, "version": versions["openai-whisper"],
            "encoder_checkpoint": encoder_path.name, "encoder_checkpoint_sha256": file_hash(encoder_path)})
        self.status("small_model_ready", encoder_dimensions=dims)

    def preflight(self, microbatch, checkpointing):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = load_whisper(self.base)
        head = expanded_head(whisper_head(model.dims.n_audio_state).state_dict())
        initial = self.root / "preflight_initial_head.pt"
        save_torch(initial, {"head": cpu_state(head)})
        del head, model
        gc.collect()
        torch.cuda.empty_cache()
        adapter = SmallAdapter(self.base, initial, finetune=True)
        adapter.gradient_checkpointing = checkpointing
        adapter.configure(True)
        initial_encoder = state_fingerprint(adapter.encoder)
        initial_head = state_fingerprint(adapter.head)
        optimizer = torch.optim.AdamW([
            {"params":adapter.encoder.parameters(),"lr":self.config["encoder_lr"]},
            {"params":adapter.head.parameters(),"lr":self.config["head_lr"]}],weight_decay=self.config["weight_decay"],foreach=False)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024)
        training = self.warm.rows("vox_train")
        long = [r for r in training if r.get("duration",0)>=10]
        if len(long)<32:
            raise ValueError("Preflight requires a full logical batch of ten-second training audio")
        training = long[:32]
        targets = torch.tensor([CODES.index(r["language"]) for r in training],device="cuda")
        torch.cuda.reset_peak_memory_stats()
        for attempt in range(8):
            optimizer.zero_grad(set_to_none=True)
            for start in range(0,32,microbatch):
                with torch.autocast("cuda",dtype=torch.float16):
                    scores=adapter.scores(training[start:start+microbatch])
                    loss=F.cross_entropy(scores,targets[start:start+microbatch])*len(scores)/32
                scaler.scale(loss).backward()
            grads=[p.grad for p in adapter.encoder.parameters() if p.requires_grad]
            if not grads or any(g is None for g in grads):
                raise ValueError("Absent Small encoder gradients")
            if scaled_optimizer_step(scaler,optimizer):
                break
        else:
            raise ValueError("Small preflight gradients overflowed eight times")
        if state_fingerprint(adapter.encoder)==initial_encoder or state_fingerprint(adapter.head)==initial_head:
            raise ValueError("Preflight did not update both encoder and classifier")
        temporary=self.root / "preflight_optimizer_checkpoint.pt"
        save_torch(temporary,{"encoder":cpu_state(adapter.encoder),"head":cpu_state(adapter.head),
                              "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict()})
        if not temporary.is_file() or temporary.stat().st_size<1:
            raise ValueError("Preflight checkpoint was not saved")
        free,total=torch.cuda.mem_get_info()
        peak=torch.cuda.max_memory_reserved()
        if peak>=6.5*1024**3 or total-free>=7*1024**3:
            raise ResourceFailure("Small optimizer-step/checkpoint preflight exceeds GPU budget")
        # Discard trial weights and restore exactly before recording success.
        del grads,optimizer,scaler,loss
        adapter.release()
        model=load_whisper(self.base)
        if state_fingerprint(model.encoder)!=initial_encoder:
            raise ValueError("Pretrained Small encoder could not be restored")
        write_json(self.root / "preflight.json",{"microbatch":microbatch,"gradient_checkpointing":checkpointing,
            "peak_reserved_bytes":peak,"encoder_gradients_verified":True,"optimizer_step_verified":True,
            "checkpoint_save_verified":True,"initial_weights_restored":True,
            "model_sha256":json.loads((self.base/"model_identity.json").read_text())["weight_sha256"],
            "config_sha256":file_hash(self.root/"configuration.json")})
        from .speaker_study import resources
        write_json(self.root/"resources_preflight.json",resources())

    def warm_cache(self):
        self.warm.cache("vox_validation", native=False)
        self.warm.cache("vox_train", native=False)

    def warm_train(self):
        self.warm.train()

    def warm_evaluate(self):
        self.warm.evaluate()

    def prepare(self):
        super().prepare()

    def cache_all(self):
        super().cache_all()

    def train(self, method):
        preflight=json.loads((self.root/"preflight.json").read_text())
        if preflight["config_sha256"]!=file_hash(self.root/"configuration.json") or preflight["model_sha256"]!=json.loads((self.root/"model_identity.json").read_text())["weight_sha256"]:
            raise ValueError("Small preflight is stale")
        super().train(method)

    def baselines(self):
        from .whisper_small_results import reuse_baselines
        reuse_baselines(self)

    def diagnostics(self):
        from .whisper_small_results import diagnostics
        diagnostics(self)

    def report(self):
        from .whisper_small_results import report
        report(self)
