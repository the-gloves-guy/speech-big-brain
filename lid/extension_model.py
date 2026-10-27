"""Explicit 15-way initialization, encoder policies, and fingerprinted caches."""

import gc
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .data import waveform
from .extension_data import (
    BASE,
    CODES,
    REVISION,
    RUN,
    check_snapshot,
    file_hash,
    status,
    write_json,
)
from .model import load_encoder, new_head


def load_initial(device="cuda"):
    torch.manual_seed(42)
    checkpoint = torch.load(
        BASE / "head/head.pt", map_location="cpu", weights_only=True
    )
    if checkpoint["meta"]["model_revision"] != REVISION:
        raise ValueError("Unexpected base encoder revision")
    encoder, _, _ = load_encoder(device, REVISION)
    head = new_head(checkpoint["initial_state"], checkpoint["indices"])
    head.load_state_dict(checkpoint["state"])
    original = head.out
    head.out = nn.Linear(original.in_features, len(CODES))
    with torch.no_grad():
        head.out.weight[:12].copy_(original.weight)
        head.out.bias[:12].copy_(original.bias)
    return encoder, head.to(device), checkpoint


def freeze_batch_stats(module, freeze_affine=False):
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            if freeze_affine:
                child.requires_grad_(False)


def configure(encoder, head, method):
    encoder.eval().requires_grad_(False)
    if method == "partial":
        for name in ("mfa", "asp", "fc"):
            getattr(encoder.mods.embedding_model, name).train().requires_grad_(True)
        freeze_batch_stats(encoder, freeze_affine=True)
    elif method != "frozen":
        raise ValueError(method)
    head.train().requires_grad_(True)
    freeze_batch_stats(head)


def tail_embedding(encoder, signals, lengths):
    """Detach the frozen prefix; preserve gradients through MFA, ASP and FC."""
    network = encoder.mods.embedding_model
    with torch.no_grad():
        # Keep STFT/filterbank computation in FP32 even during AMP training.
        with torch.autocast(device_type=signals.device.type, enabled=False):
            features = encoder.mods.compute_features(signals.float())
            features = encoder.mods.mean_var_norm(features, lengths)
        x, layers = features.transpose(1, 2), []
        for layer in network.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            layers.append(x)
        x = torch.cat(layers[1:], dim=1).detach()
    x = network.mfa(x)
    x = network.asp(x, lengths=lengths)
    x = network.asp_bn(x)
    return network.fc(x).transpose(1, 2)


def audio_batch(rows, device):
    signals = []
    for row in rows:
        check_snapshot(row)
        signals.append(torch.from_numpy(waveform(row, 10.0)))
    sizes = torch.tensor([len(signal) for signal in signals], device=device)
    padded = nn.utils.rnn.pad_sequence(signals, batch_first=True).to(device)
    return padded, sizes / padded.shape[1]


def state_fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_fingerprints(encoder):
    trained = {
        name for name, value in encoder.named_parameters() if value.requires_grad
    }
    return {
        name: hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, value in encoder.state_dict().items()
        if name not in trained
    }


def verify_frozen(encoder, expected):
    for name, value in encoder.state_dict().items():
        if name in expected:
            actual = hashlib.sha256(
                value.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            if actual != expected[name]:
                raise ValueError(f"Frozen parameter or buffer changed: {name}")


def cache_embeddings(encoder, rows, path, fingerprint, phase, progress=None):
    """Durable shards; never keep decoded datasets or graphs in memory."""
    path = Path(path)
    identity = {
        "encoder": fingerprint,
        "rows_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        "sample_rate": 16000,
        "crop": "center",
        "max_seconds": 10,
        "codes": CODES,
    }
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError(f"Incompatible embedding cache: {path}")
        return saved["embeddings"]
    parts = path.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    context = parts / "context.json"
    if context.exists() and json.loads(context.read_text()) != identity:
        raise ValueError(f"Incompatible partial cache: {parts}")
    write_json(context, identity)
    shards = sorted(parts.glob("part-*.pt"))
    completed = sum(
        torch.load(shard, map_location="cpu", weights_only=True).shape[0]
        for shard in shards
    )
    device = next(encoder.parameters()).device
    encoder.eval()
    with torch.inference_mode():
        for start in range(completed, len(rows), 256):
            values = []
            # No padded inference: use the identical waveform protocol as the
            # original benchmark, independent of other examples in a batch.
            for row in rows[start : start + 256]:
                signals, lengths = audio_batch([row], device)
                values.append(
                    encoder.encode_batch(signals, lengths, normalize=False)
                    .reshape(1, 256)
                    .cpu()
                )
            shard = parts / f"part-{len(shards):06d}.pt"
            temporary = shard.with_suffix(".tmp")
            torch.save(torch.cat(values), temporary)
            temporary.replace(shard)
            shards.append(shard)
            (progress or status)(
                phase,
                completed=min(start + 256, len(rows)),
                total=len(rows),
                percent=round(min(start + 256, len(rows)) / len(rows) * 100, 1),
            )
    embeddings = torch.cat(
        [torch.load(shard, map_location="cpu", weights_only=True) for shard in shards]
    )
    if embeddings.shape != (len(rows), 256) or not torch.isfinite(embeddings).all():
        raise ValueError("Invalid extracted embeddings")
    temporary = path.with_suffix(".tmp")
    torch.save({"embeddings": embeddings, "identity": identity}, temporary)
    temporary.replace(path)
    return embeddings


def old_cache(split):
    name = (
        f"vox_{split}.pt"
        if split in ("train", "validation")
        else f"fleurs_direct_{split.removeprefix('fleurs_')}.pt"
    )
    bundle = torch.load(BASE / name, map_location="cpu", weights_only=True)
    if (
        bundle["meta"]["model_revision"] != REVISION
        or bundle["meta"]["codes"] != CODES[:12]
        or bundle["meta"]["sample_rate"] != 16000
        or bundle["meta"]["max_seconds"] != 10
        or bundle["meta"]["crop"] != "center"
    ):
        raise ValueError("Incompatible original cache")
    return bundle


def head_scores(head, embeddings, device="cuda"):
    head.eval()
    # SpeechBrain enables TF32 globally. Full-precision scoring avoids apparent
    # export differences between CUDA TF32 GEMMs and standard CPU inference.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            return torch.cat(
                [
                    head(batch.to(device).unsqueeze(1)).squeeze(1).float().cpu()
                    for batch in embeddings.split(256)
                ]
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def cache_new():
    from .extension_data import read_rows

    encoder, head, _ = load_initial()
    del head
    fingerprint = state_fingerprint(encoder.mods.embedding_model)
    for split in ("train", "selection", "calibration", "test"):
        rows = read_rows(RUN / f"new_{split}.jsonl")
        cache_embeddings(
            encoder, rows, RUN / f"new_{split}.pt", fingerprint, f"cache_new_{split}"
        )
    write_json(
        RUN / "cache_ready.json",
        {
            "encoder": fingerprint,
            "initial_head_sha256": file_hash(BASE / "head/head.pt"),
        },
    )
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
