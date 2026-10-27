"""Frozen Large v3 study using the canonical recording splits and cached features."""

import gc
import importlib.metadata
import json
import time
from pathlib import Path

import torch

from .extension_data import CODES, file_hash, write_json
from .extension_model import state_fingerprint
from .model import cpu_state
from .speaker_pilot import bind
from .whisper_extension import ExtensionAdapter, ExtensionStudy, expanded_head
from .whisper_model import load_whisper, mel_batch, pool_real_tokens, whisper_head
from .whisper_small import ResourceFailure, WarmStudy
from .whisper_study import save_torch


class CachedHeadAdapter:
    """The frozen trainer/evaluator needs a classifier, not a resident encoder."""

    def __init__(self, root, checkpoint, finetune=False):
        if finetune:
            raise ValueError("Large v3 study only supports frozen encoders")
        identity = json.loads((Path(root) / "model_identity.json").read_text())
        self.encoder = torch.nn.Identity()
        self.head = expanded_head(
            whisper_head(identity["dims"]["n_audio_state"]).state_dict()
        ).cuda()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        reference = saved.get("encoder_checkpoint_sha256")
        if reference is not None and reference != identity["encoder_checkpoint_sha256"]:
            raise ValueError("Classifier encoder reference changed")
        self.head.load_state_dict(saved["head"])
        self.configure(False)

    def configure(self, train=False):
        self.head.requires_grad_(True).train(train)

    def release(self):
        del self.head, self.encoder
        gc.collect()
        torch.cuda.empty_cache()


class LargeWarmStudy(WarmStudy):
    pass


class LargeStudy(ExtensionStudy):
    methods = ("frozen",)
    adapter_class = CachedHeadAdapter
    extension_prefix = "whisper_large_v3_15"
    comparison_models = (
        "whisper_large_v3_15_frozen",
        "speechbrain_partial15",
        "whisper_tiny15_frozen",
        "whisper_small15_frozen",
    )

    def __init__(
        self,
        config="configs/whisper_large_v3_15.json",
        root="artifacts/whisper_large_v3_15",
    ):
        super().__init__(config, root)
        if (
            self.config["model"] != "large-v3"
            or self.config["encoder_precision"] != "float32"
        ):
            raise ValueError("This experiment requires float32 multilingual Large v3")
        self.reference = Path(self.config["reference"])
        self.small_reference = Path(self.config["small_reference"])
        self.warm = LargeWarmStudy(self)

    def prepare_model(self):
        import whisper
        from whisper.tokenizer import get_tokenizer

        self.warm.prepare(self.reference)
        versions = {
            n: importlib.metadata.version(n)
            for n in ["openai-whisper", "torch", "torchaudio", "numpy"]
        }
        if versions["openai-whisper"] != self.config["whisper_version"]:
            raise ValueError("Whisper package version changed")
        bind(
            self.base / "environment.json",
            {"versions": versions, "uv_lock_sha256": file_hash("uv.lock")},
        )
        directory = self.base / "pretrained"
        directory.mkdir(exist_ok=True)
        url = whisper._MODELS["large-v3"]
        checksum = url.split("/")[-2]
        path = directory / "large-v3.pt"
        if not path.exists():
            whisper._download(url, str(directory), False)
        if file_hash(path) != checksum:
            raise ValueError("Official Large v3 checkpoint checksum mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        dims = saved["dims"]
        if (dims["n_mels"], dims["n_audio_state"], dims["n_audio_layer"]) != (
            128,
            1280,
            32,
        ):
            raise ValueError("Unexpected Large v3 architecture")
        state = {
            k.removeprefix("encoder."): v
            for k, v in saved["model_state_dict"].items()
            if k.startswith("encoder.")
        }
        encoder_path = directory / "large-v3_encoder.pt"
        if not encoder_path.exists():
            save_torch(encoder_path, {"dims": dims, "encoder": state})
        extracted = torch.load(
            encoder_path, map_location="cpu", weights_only=True, mmap=True
        )
        if (
            extracted["dims"] != dims
            or set(extracted["encoder"]) != set(state)
            or any(
                not torch.equal(v, extracted["encoder"][k]) for k, v in state.items()
            )
        ):
            raise ValueError("Extracted encoder differs from official weights")
        bind(
            self.base / "model_identity.json",
            {
                "model": "large-v3",
                "weight_sha256": checksum,
                "download_url": url,
                "dims": dims,
                "version": versions["openai-whisper"],
                "language_codes": sorted(
                    get_tokenizer(
                        multilingual=True, num_languages=100, language="en"
                    ).all_language_codes
                ),
                "encoder_checkpoint": encoder_path.name,
                "encoder_checkpoint_sha256": file_hash(encoder_path),
                "bounded_memory_loading": True,
                "encoder_precision": "float32",
            },
        )
        self.status("large_model_ready", encoder_dimensions=dims)

    def preflight(self, microbatch=2):
        torch.manual_seed(42)
        model = load_whisper(self.base)
        model.eval().requires_grad_(False)
        fingerprint = state_fingerprint(model.encoder)
        rows = self.warm.rows("vox_train")[:16]
        head = expanded_head(whisper_head(1280).state_dict()).cuda()
        before = state_fingerprint(head)
        optimizer = torch.optim.AdamW(head.parameters(), lr=self.config["head_lr"])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        features = []
        with torch.inference_mode():
            for start in range(0, len(rows), microbatch):
                mels, sizes = mel_batch(rows[start : start + microbatch], n_mels=128)
                features.append(pool_real_tokens(model.encoder(mels), sizes))
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        # Clone outside inference_mode so the classifier can save features for backward.
        batch = torch.cat(features).clone()
        loss = torch.nn.functional.cross_entropy(
            head(batch),
            torch.tensor([CODES.index(r["language"]) for r in rows], device="cuda"),
        )
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is None or not torch.isfinite(p.grad).all()
            for p in head.parameters()
        ):
            raise ValueError("Nonfinite or absent classifier gradients")
        optimizer.step()
        if (
            state_fingerprint(head) == before
            or state_fingerprint(model.encoder) != fingerprint
        ):
            raise ValueError("Classifier update or encoder freeze verification failed")
        if any(
            p.grad is not None or p.requires_grad for p in model.encoder.parameters()
        ):
            raise ValueError("Frozen encoder has gradients")
        peak = torch.cuda.max_memory_reserved()
        free, total = torch.cuda.mem_get_info()
        if peak >= 6.5 * 1024**3 or total - free >= 7 * 1024**3:
            raise ResourceFailure("Large v3 inference exceeds GPU budget")
        save_torch(
            self.root / "preflight_head.pt",
            {"head": cpu_state(head), "optimizer": optimizer.state_dict()},
        )
        write_json(
            self.root / "preflight.json",
            {
                "microbatch": microbatch,
                "peak_reserved_bytes": peak,
                "examples": len(rows),
                "seconds": seconds,
                "examples_per_second": len(rows) / seconds,
                "encoder_fingerprint": fingerprint,
                "encoder_unchanged": True,
                "classifier_step_verified": True,
                "config_sha256": file_hash(self.root / "configuration.json"),
                "encoder_checkpoint_sha256": json.loads(
                    (self.base / "model_identity.json").read_text()
                )["encoder_checkpoint_sha256"],
            },
        )

    def apply_preflight(self):
        saved = json.loads((self.root / "preflight.json").read_text())
        if (
            saved["config_sha256"] != file_hash(self.root / "configuration.json")
            or saved["encoder_checkpoint_sha256"]
            != json.loads((self.base / "model_identity.json").read_text())[
                "encoder_checkpoint_sha256"
            ]
        ):
            raise ValueError("Large v3 resource preflight is stale")
        identity = json.loads((self.base / "model_identity.json").read_text())
        if (
            file_hash(self.base / "pretrained" / identity["encoder_checkpoint"])
            != identity["encoder_checkpoint_sha256"]
        ):
            raise ValueError("Extracted Large v3 encoder checksum changed")
        # Keep immutable experiment configuration separate from measured geometry.
        self.config["cache_batch"] = saved["microbatch"]
        self.warm.config["cache_batch"] = saved["microbatch"]

    def warm_cache(self):
        self.apply_preflight()
        self.warm.cache("vox_validation", native=False)
        self.warm.cache("vox_train", native=False)

    def warm_train(self):
        self.warm.train()

    def warm_evaluate(self):
        self.apply_preflight()
        self.warm.evaluate()

    def cache_all(self):
        self.apply_preflight()
        super().cache_all()

    def frozen(self):
        self.apply_preflight()
        super().train("frozen")
        identity = json.loads((self.root / "model_identity.json").read_text())
        path = self.root / "frozen/model.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        reference = saved.get("encoder_checkpoint_sha256")
        if reference is not None and reference != identity["encoder_checkpoint_sha256"]:
            raise ValueError("Completed classifier encoder reference changed")
        if saved.get("encoder"):
            raise ValueError("Cached training unexpectedly stored encoder parameters")
        if reference is None:
            saved["encoder_checkpoint_sha256"] = identity["encoder_checkpoint_sha256"]
            save_torch(path, saved)
        result = json.loads((self.root / "frozen/training_complete.json").read_text())
        result["encoder_checkpoint_sha256"] = identity["encoder_checkpoint_sha256"]
        result["encoder_fingerprint"] = json.loads(
            (self.root / "preflight.json").read_text()
        )["encoder_fingerprint"]
        write_json(self.root / "frozen/training_complete.json", result)

    def baselines(self):
        from .whisper_large_results import baselines

        baselines(self)

    def additional_external_models(self):
        from .whisper_large_results import additional_external_models

        return additional_external_models(self)

    def diagnostics(self):
        from .whisper_large_results import diagnostics

        diagnostics(self)

    def report(self):
        from .whisper_large_results import report

        report(self)

    def export(self):
        import importlib.util

        from .whisper_extension_export import export_one, verify

        spec = importlib.util.spec_from_file_location(
            "whisper_archive", Path(__file__).parents[1] / "scripts/export_whisper12.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = Path("artifacts/deployment/whisper_large_v3_15_frozen")
        export_one(self, "frozen", output)
        if not (output / "verification.json").exists():
            verify(self, "frozen", output)
        module.archive(output)
        write_json(
            self.root / "export_complete.json",
            {"methods": ["frozen"], "offline_verification_required": True},
        )


class LargeAudioAdapter(ExtensionAdapter):
    """Real-audio inference loads the encoder; cached classifier training does not."""

    def __init__(self, root, checkpoint, finetune=False):
        if finetune:
            raise ValueError("Large v3 encoder must stay frozen")
        self.family = "whisper"
        self.model = load_whisper(root)
        self.encoder = self.model.encoder
        self.head = expanded_head(
            whisper_head(self.model.dims.n_audio_state).state_dict()
        ).cuda()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        identity = json.loads((Path(root) / "model_identity.json").read_text())
        if saved["encoder_checkpoint_sha256"] != identity["encoder_checkpoint_sha256"]:
            raise ValueError("Audio adapter encoder reference changed")
        self.head.load_state_dict(saved["head"])
        self.finetune = False
        self.configure(False)
