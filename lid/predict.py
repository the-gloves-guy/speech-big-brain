"""Use either adapted model on a local audio recording."""

import hashlib
import json
from pathlib import Path

import torch

from .data import waveform
from .languages import CODES, NAMES, selected_indices
from .model import load_encoder, new_head


def predict(args):
    torch.set_num_threads(args.threads)
    calibration = (
        json.loads(Path(args.calibration).read_text()) if args.calibration else None
    )
    checkpoint = (
        torch.load(args.head, map_location="cpu", weights_only=True)
        if args.head
        else None
    )
    if args.method == "head" and checkpoint is None:
        raise ValueError("Head inference requires --head")
    if checkpoint is not None and calibration is not None:
        digest = hashlib.sha256(Path(args.head).read_bytes()).hexdigest()
        if digest != calibration["head_sha256"]:
            raise ValueError("Calibration was fitted for a different head checkpoint")
    meta = (
        checkpoint["meta"]
        if checkpoint
        else (calibration["metadata"] if calibration else None)
    )
    if calibration is not None and checkpoint is not None:
        for key in ("model_revision", "max_seconds", "crop", "codes"):
            if calibration["metadata"][key] != meta[key]:
                raise ValueError(
                    "Head and calibration use incompatible encoder/preprocessing settings"
                )
    encoder, labels, revision = load_encoder(
        args.device, meta["model_revision"] if meta else "main"
    )
    indices = selected_indices(labels)
    if checkpoint and (
        checkpoint["labels"] != labels or checkpoint["indices"] != indices
    ):
        raise ValueError("Checkpoint label order differs from encoder")
    signal = waveform(
        {"path": args.audio, "id": args.audio}, meta["max_seconds"] if meta else 10.0
    )
    signal = torch.from_numpy(signal).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        embedding = encoder.encode_batch(signal, normalize=False)
        if args.method == "head":
            head = new_head(checkpoint["initial_state"], indices).to(args.device).eval()
            head.load_state_dict(checkpoint["state"])
            logits = head(embedding).reshape(-1)
        else:
            logits = encoder.mods.classifier(embedding).reshape(-1)[indices]
        temperature = (
            calibration[
                f"{'head' if args.method == 'head' else 'filtered'}_temperature"
            ]
            if calibration
            else 1.0
        )
        probs = torch.softmax(logits / temperature, dim=-1).cpu().tolist()
    winner = max(range(len(probs)), key=probs.__getitem__)
    print(
        json.dumps(
            {
                "language": NAMES[winner],
                "code": CODES[winner],
                "confidence": probs[winner],
                "method": args.method,
                "temperature": temperature,
                "model_revision": revision,
                "probabilities": dict(zip(CODES, probs)),
            },
            indent=2,
        )
    )
