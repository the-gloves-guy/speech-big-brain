"""Train the pretrained classifier MLP, keeping the audio encoder frozen."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cache import compatible, disjoint, load_cache
from .data import write_json
from .languages import selected_indices
from .model import cpu_state, new_head


def head_logits(head, embeddings, batch_size=256):
    head.eval()
    device = next(head.parameters()).device
    with torch.inference_mode():
        return torch.cat(
            [
                head(batch.to(device).unsqueeze(1)).squeeze(1).cpu()
                for batch in embeddings.split(batch_size)
            ]
        )


def train(args):
    if args.epochs < 1 or args.batch_size < 2 or args.patience < 1 or args.lr <= 0:
        raise ValueError("Use positive epochs/patience/lr and batch size >= 2")
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Small classifier matrices are faster with a modest CPU thread count.
    torch.set_num_threads(args.threads)
    train_data, validation = load_cache(args.train), load_cache(args.validation)
    compatible(train_data, validation)
    disjoint(train_data, validation)
    if any(b["meta"]["source"] != "voxlingua" for b in (train_data, validation)):
        raise ValueError(
            "Head training and checkpoint selection must use VoxLingua, not FLEURS"
        )
    if (
        train_data["meta"]["split"] != "train"
        or validation["meta"]["split"] != "validation"
    ):
        raise ValueError(
            "Training requires train and validation caches in their corresponding roles"
        )
    indices = selected_indices(train_data["labels"])
    head = new_head(train_data["head_state"], indices).to(args.device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    counts = torch.bincount(train_data["targets"], minlength=12).float()
    weights = (counts.sum() / (12 * counts)).to(args.device)
    best_loss, stale, history, best_state, best_epoch = float("inf"), 0, [], None, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(train_data["targets"]))
        batches = list(order.split(args.batch_size))
        if len(batches[-1]) == 1:
            last = batches.pop()
            batches[-1] = torch.cat([batches[-1], last])
        total_loss, total_n = 0.0, 0
        for batch in batches:
            embeddings = train_data["embeddings"][batch].to(args.device).unsqueeze(1)
            targets = train_data["targets"][batch].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            scores = head(embeddings).squeeze(1)
            loss = F.cross_entropy(scores, targets, weight=weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_n += len(batch)
        scores = head_logits(head, validation["embeddings"])
        val_loss = F.cross_entropy(scores, validation["targets"]).item()
        accuracy = (scores.argmax(1) == validation["targets"]).float().mean().item()
        row = {
            "epoch": epoch,
            "train_weighted_loss": total_loss / total_n,
            "validation_nll": val_loss,
            "validation_accuracy": accuracy,
        }
        history.append(row)
        print(row, flush=True)
        if val_loss < best_loss:
            best_loss, stale, best_state, best_epoch = (
                val_loss,
                0,
                cpu_state(head),
                epoch,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": best_state,
            "initial_state": train_data["head_state"],
            "indices": indices,
            "labels": train_data["labels"],
            "meta": train_data["meta"],
            "best_epoch": best_epoch,
            "validation_nll": best_loss,
            "train_ids": train_data["ids"],
            "train_groups": train_data["groups"],
            "validation_ids": validation["ids"],
            "validation_groups": validation["groups"],
            "settings": {k: v for k, v in vars(args).items() if k != "func"},
        },
        output / "head.pt",
    )
    write_json(output / "history.json", history)
    print(f"Saved best head (epoch {best_epoch}) to {output / 'head.pt'}")
