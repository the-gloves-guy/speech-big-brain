"""Local Whisper tiny LID and paired twelve-language encoder adapters."""

import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .data import waveform
from .extension_data import check_snapshot
from .languages import CODES
from .model import load_encoder, new_head


def selected_languages(codes):
    missing = set(CODES) - set(codes)
    if missing:
        raise ValueError(f"Whisper language vocabulary is missing {sorted(missing)}")
    return [codes.index(code) for code in CODES]


def pool_real_tokens(features, sample_counts):
    """Exclude right-padding outputs from the statistics, retaining native attention."""
    lengths = torch.div(sample_counts + 319, 320, rounding_mode="floor").clamp(
        1, features.shape[1]
    )
    mask = (
        torch.arange(features.shape[1], device=features.device)[None, :]
        < lengths[:, None]
    ).unsqueeze(-1)
    values = features.float()
    mean = (values * mask).sum(dim=1) / lengths[:, None]
    variance = ((values - mean[:, None, :]).square() * mask).sum(dim=1) / lengths[
        :, None
    ]
    return torch.cat([mean, variance.clamp_min(1e-8).sqrt()], dim=-1)


def mel_batch(rows, device="cuda"):
    import whisper

    mels, sizes = [], []
    for row in rows:
        check_snapshot(row)
        signal = waveform(row, 10)
        sizes.append(len(signal))
        # Per-waveform STFT and dynamic-range normalization are batch-independent.
        signal = torch.from_numpy(signal).to(device)
        with torch.autocast(device_type=signal.device.type, enabled=False):
            mels.append(
                whisper.log_mel_spectrogram(
                    whisper.pad_or_trim(signal.float()), n_mels=80
                )
            )
    return torch.stack(mels), torch.tensor(sizes, device=device)


def load_whisper(root, device="cuda"):
    import whisper

    identity_path = Path(root) / "model_identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    if "encoder_checkpoint" in identity:
        from whisper.model import AudioEncoder

        path = Path(root) / "pretrained" / identity["encoder_checkpoint"]
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        dims = saved["dims"]
        if dims != identity["dims"]:
            raise ValueError("Encoder checkpoint dimensions differ from identity")
        model = nn.Module()
        model.dims = SimpleNamespace(**dims)
        model.encoder = AudioEncoder(dims["n_mels"], dims["n_audio_ctx"], dims["n_audio_state"],
                                     dims["n_audio_head"], dims["n_audio_layer"])
        model.encoder.load_state_dict(saved["encoder"])
        return model.to(device).eval()
    return whisper.load_model(
        str(Path(root) / "pretrained" / f"{identity.get('model', 'tiny')}.pt"), device=device
    ).eval()


def tokenizer_for(model):
    from whisper.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(
        multilingual=True, num_languages=model.num_languages, language="en"
    )
    codes = sorted(tokenizer.all_language_codes)
    selected_languages(codes)
    return tokenizer, codes


def native_logits(model, encoded):
    tokenizer, codes = tokenizer_for(model)
    tokens = torch.full(
        (len(encoded), 1), tokenizer.sot, device=encoded.device, dtype=torch.long
    )
    by_code = dict(zip(tokenizer.all_language_codes, tokenizer.all_language_tokens))
    token_ids = [by_code[code] for code in codes]
    return model.decoder(tokens, encoded)[:, 0, token_ids].float(), codes


def whisper_head(width=384):
    return nn.Sequential(
        nn.LayerNorm(width * 2),
        nn.Linear(width * 2, 512),
        nn.LeakyReLU(),
        nn.Linear(512, len(CODES)),
    )


def freeze_batch_stats(module):
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


class Adapter:
    def __init__(self, family, root, checkpoint=None):
        self.family = family
        if family == "whisper":
            self.model = load_whisper(root)
            self.model.requires_grad_(False)
            self.encoder = self.model.encoder
            self.head = whisper_head(self.model.dims.n_audio_state).to("cuda")
        elif family == "speechbrain":
            saved = torch.load(
                "artifacts/full_run/head/head.pt", map_location="cpu", weights_only=True
            )
            self.model, _, _ = load_encoder("cuda", saved["meta"]["model_revision"])
            self.encoder = self.model.mods.embedding_model
            self.head = new_head(saved["initial_state"], saved["indices"]).to("cuda")
            self.head.load_state_dict(saved["state"])
            del saved
        else:
            raise ValueError(family)
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.head.load_state_dict(saved["head"])
            if "encoder" in saved:
                self.encoder.load_state_dict(saved["encoder"])
            del saved
        self.configure(False)

    def configure(self, train=False):
        self.model.eval().requires_grad_(False)
        self.encoder.requires_grad_(train)
        self.encoder.train(train)
        self.head.requires_grad_(True).train(train)
        freeze_batch_stats(self.encoder)
        freeze_batch_stats(self.head)

    def embeddings(self, rows):
        if self.family == "whisper":
            mels, lengths = mel_batch(rows)
            return pool_real_tokens(self.encoder(mels), lengths)
        from .extension_model import audio_batch

        signals, lengths = audio_batch(rows, "cuda")
        # Keep the fixed STFT and sentence normalization in FP32 during AMP.
        with torch.autocast("cuda", enabled=False):
            features = self.model.mods.compute_features(signals.float())
            features = self.model.mods.mean_var_norm(features, lengths)
        return self.encoder(features, lengths).squeeze(1)

    def scores(self, rows):
        embeddings = self.embeddings(rows)
        if self.family == "speechbrain":
            return self.head(embeddings.unsqueeze(1)).squeeze(1).float()
        return self.head(embeddings).float()

    def release(self):
        del self.model, self.encoder, self.head
        gc.collect()
        torch.cuda.empty_cache()
