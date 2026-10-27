"""Standalone Whisper tiny LID backend; copied into exported bundles as custom.py.

Runtime dependencies are torch, openai-whisper and SpeechBrain. No lid imports.
"""

import torch
from torch import nn
from whisper.audio import log_mel_spectrogram, pad_or_trim
from whisper.model import AudioEncoder


def load_whisper_encoder(module, path):
    """Memory-map large encoder weights instead of duplicating them in host RAM."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    module.load_state_dict(state, strict=True, assign=True)


class WhisperLIDPassThrough(nn.Module):
    def forward(self, waveforms, lengths=None):
        return waveforms


class WhisperTinyLIDEncoder(nn.Module):
    def __init__(
        self,
        n_mels=80,
        n_ctx=1500,
        n_state=384,
        n_head=6,
        n_layer=4,
        max_samples=160000,
        bounded_memory_loading=False,
    ):
        super().__init__()
        self.bounded_memory_loading = bounded_memory_loading
        if bounded_memory_loading:
            with torch.device("meta"):
                self.encoder = AudioEncoder(n_mels, n_ctx, n_state, n_head, n_layer)
        else:
            self.encoder = AudioEncoder(n_mels, n_ctx, n_state, n_head, n_layer)
        self.max_samples = int(max_samples)
        self.n_mels = n_mels

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(
            state_dict, strict=strict, assign=assign or self.bounded_memory_loading
        )

    def forward(self, waveforms, lengths=None):
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError("Expected mono waveform batch [batch, samples]")
        if lengths is None:
            lengths = torch.ones(len(waveforms), device=waveforms.device)
        if (
            lengths.shape != (len(waveforms),)
            or not torch.isfinite(lengths).all()
            or (lengths <= 0).any()
            or (lengths > 1).any()
        ):
            raise ValueError("Waveform lengths must be fractions in (0, 1]")
        mels = []
        sample_counts = []
        for signal, length in zip(waveforms, lengths):
            count = min(len(signal), round(float(length) * len(signal)))
            if count < 1600:
                raise ValueError("Audio must contain at least 0.1 seconds at 16 kHz")
            size = min(count, self.max_samples)
            start = (count - size) // 2
            signal = signal[start : start + size].float()
            with torch.autocast(device_type=signal.device.type, enabled=False):
                mels.append(
                    log_mel_spectrogram(pad_or_trim(signal), n_mels=self.n_mels)
                )
            sample_counts.append(size)
        encoded = self.encoder(torch.stack(mels))
        counts = torch.tensor(sample_counts, device=encoded.device)
        valid = torch.div(counts + 319, 320, rounding_mode="floor").clamp(
            1, encoded.shape[1]
        )
        mask = (
            torch.arange(encoded.shape[1], device=encoded.device)[None, :]
            < valid[:, None]
        ).unsqueeze(-1)
        features = encoded.float()
        mean = (features * mask).sum(dim=1) / valid[:, None]
        variance = ((features - mean[:, None, :]).square() * mask).sum(dim=1) / valid[
            :, None
        ]
        pooled = torch.cat([mean, variance.clamp_min(1e-8).sqrt()], dim=-1)
        return pooled.unsqueeze(1)


class WhisperTinyLIDClassifier(nn.Sequential):
    def __init__(self, n_state=384, hidden=512, outputs=12):
        super().__init__(
            nn.LayerNorm(n_state * 2),
            nn.Linear(n_state * 2, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, outputs),
        )

    def forward(self, embeddings):
        return super().forward(embeddings).log_softmax(dim=-1)


# Architecture-neutral names for new bundles; retain the historical Tiny API.
WhisperLIDEncoder = WhisperTinyLIDEncoder
WhisperLIDClassifier = WhisperTinyLIDClassifier
