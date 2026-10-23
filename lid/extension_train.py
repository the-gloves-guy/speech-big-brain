"""Resumable, class-balanced paired adaptation with guarded encoder gradients."""

import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .extension_data import (
    BASE,
    CODES,
    NAMES,
    RUN,
    file_hash,
    read_rows,
    status,
    write_json,
)
from .extension_model import (
    audio_batch,
    cache_embeddings,
    configure,
    frozen_fingerprints,
    head_scores,
    load_initial,
    old_cache,
    state_fingerprint,
    tail_embedding,
    verify_frozen,
)
from .model import cpu_state

SETTINGS = {
    "seed": 42,
    "epochs": 20,
    "patience": 5,
    "examples_per_epoch": 32768,
    "effective_batch_size": 32,
    "head_lr": 1e-4,
    "encoder_lr": 1e-5,
    "weight_decay": 1e-4,
    "max_seconds": 10,
    "codes": CODES,
    "selection": "15-class macro F1, then class-balanced NLL",
    "batch_norm_running_stats": "frozen",
    "encoder_batch_norm_affine": "frozen",
}


def save_torch(path, payload):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def labels_for(rows):
    return torch.tensor(
        [CODES.index(row["language"]) for row in rows], dtype=torch.long
    )


def align_old(rows, bundle):
    by_path = {str(Path(row["path"]).resolve()): row for row in rows}
    if set(by_path) != set(bundle["ids"]):
        raise ValueError("Original manifest and cache IDs differ")
    result = [by_path[item] for item in bundle["ids"]]
    if not torch.equal(labels_for(result), bundle["targets"]):
        raise ValueError("Original cache targets and manifest differ")
    return result


def epoch_order(targets, epoch, count=32768, speaker_groups=None):
    generator = torch.Generator().manual_seed(42 + epoch)
    classes = torch.randint(len(CODES), (count,), generator=generator)
    result = torch.empty(count, dtype=torch.long)
    for index in range(len(CODES)):
        pool = (targets == index).nonzero().flatten()
        if not len(pool):
            raise ValueError(f"Missing training class {CODES[index]}")
        mask = classes == index
        if speaker_groups is not None and index >= 12:
            groups = {}
            for item in pool.tolist():
                groups.setdefault(speaker_groups[item], []).append(item)
            pools = [torch.tensor(items) for _, items in sorted(groups.items())]
            selected = torch.randint(
                len(pools), (int(mask.sum()),), generator=generator
            )
            sampled = torch.empty(len(selected), dtype=torch.long)
            for group_index, group_pool in enumerate(pools):
                group_mask = selected == group_index
                sampled[group_mask] = group_pool[
                    torch.randint(
                        len(group_pool), (int(group_mask.sum()),), generator=generator
                    )
                ]
            result[mask] = sampled
        else:
            result[mask] = pool[
                torch.randint(len(pool), (int(mask.sum()),), generator=generator)
            ]
    return result


def selection_metrics(logits, targets):
    predictions = logits.argmax(1)
    losses = F.cross_entropy(logits, targets, reduction="none")
    return {
        "macro_f1": float(
            f1_score(
                targets.numpy(),
                predictions.numpy(),
                labels=list(range(len(CODES))),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_nll": float(
            torch.stack([losses[targets == i].mean() for i in range(len(CODES))]).mean()
        ),
        "old_accuracy": float(
            (predictions[targets < 12] == targets[targets < 12]).float().mean()
        ),
        "new_accuracy": float(
            (predictions[targets >= 12] == targets[targets >= 12]).float().mean()
        ),
    }


def scaled_optimizer_step(scaler, optimizer):
    """Never change weights on overflow; lower AMP scale for a same-batch retry."""
    scaler.unscale_(optimizer)
    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=False)
    if not torch.isfinite(norm):
        scaler.update(new_scale=scaler.get_scale() * 0.5)
        return False
    scaler.step(optimizer)
    scaler.update()
    return True


def preflight(encoder, head, run=RUN):
    configure(encoder, head, "partial")
    for microbatch in (2, 1):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            signals = torch.zeros(microbatch, 160000, device="cuda")
            lengths = torch.ones(microbatch, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = head(tail_embedding(encoder, signals, lengths)).squeeze(1)
                loss = F.cross_entropy(
                    logits.float(),
                    torch.zeros(microbatch, dtype=torch.long, device="cuda"),
                )
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            encoder.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            del signals, lengths, logits, loss
            torch.cuda.empty_cache()
            if peak < 7 * 1024**3:
                write_json(
                    run / "partial_preflight.json",
                    {
                        "microbatch": microbatch,
                        "effective_batch_size": 32,
                        "peak_reserved_gib": peak / 1024**3,
                        "passed": True,
                    },
                )
                return microbatch
        except torch.cuda.OutOfMemoryError:
            encoder.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(
        "Encoder-tail preflight exceeds the 7 GiB GPU budget even at microbatch 1"
    )


def train(method, run=RUN, sampling="uniform", progress=None):
    run = Path(run)
    if sampling not in ("uniform", "speaker_balanced"):
        raise ValueError("Unknown sampling policy")
    emit = progress or status
    output = run / method
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        return
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    encoder, head, _ = load_initial()
    initial_head_fingerprint = state_fingerprint(head)
    initial_encoder_fingerprint = state_fingerprint(encoder.mods.embedding_model)
    paired_configuration = run / "frozen/configuration.json"
    if method == "partial" and paired_configuration.exists():
        paired = json.loads(paired_configuration.read_text())
        if (paired["initial_head"], paired["initial_encoder"]) != (
            initial_head_fingerprint,
            initial_encoder_fingerprint,
        ):
            raise ValueError("Paired models do not have identical initial weights")
    configure(encoder, head, method)
    expected_frozen = frozen_fingerprints(encoder)
    training_rows = read_rows(run / "new_train.jsonl")
    validation_rows = read_rows(run / "new_selection.jsonl")
    old_train = old_cache("train")
    old_validation = old_cache("validation")
    old_training_rows = align_old(read_rows(BASE / "manifests/train.jsonl"), old_train)
    old_validation_rows = align_old(
        read_rows(BASE / "manifests/validation.jsonl"), old_validation
    )
    targets = torch.cat([old_train["targets"], labels_for(training_rows)])
    validation_targets = torch.cat(
        [old_validation["targets"], labels_for(validation_rows)]
    )
    all_training_rows = old_training_rows + training_rows
    speaker_groups = None
    if sampling == "speaker_balanced":
        speaker_groups = ["old"] * len(old_training_rows) + [
            row["speaker_id"] for row in training_rows
        ]
    all_validation_rows = old_validation_rows + validation_rows
    embeddings = None
    validation_embeddings = None
    if method == "frozen":
        new_train = cache_embeddings(
            encoder,
            training_rows,
            run / "new_train.pt",
            initial_encoder_fingerprint,
            "cache_new_train",
            progress=emit,
        )
        new_validation = cache_embeddings(
            encoder,
            validation_rows,
            run / "new_selection.pt",
            initial_encoder_fingerprint,
            "cache_new_selection",
            progress=emit,
        )
        embeddings = torch.cat([old_train["embeddings"], new_train])
        validation_embeddings = torch.cat(
            [old_validation["embeddings"], new_validation]
        )
        del new_train, new_validation
    del old_train, old_validation
    gc.collect()
    identity = {
        "settings": SETTINGS,
        "method": method,
        "initial_head": initial_head_fingerprint,
        "initial_encoder": initial_encoder_fingerprint,
        "base_checkpoint_sha256": file_hash(BASE / "head/head.pt"),
        "new_data_sha256": file_hash(run / "data_ready.json"),
        **({"sampling": sampling} if run != RUN else {}),
    }
    groups = [{"params": list(head.parameters()), "lr": SETTINGS["head_lr"]}]
    if method == "partial":
        groups.append(
            {
                "params": [
                    parameter
                    for parameter in encoder.parameters()
                    if parameter.requires_grad
                ],
                "lr": SETTINGS["encoder_lr"],
            }
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=SETTINGS["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    epoch, cursor, stale, history = 1, 0, 0, []
    best_f1, best_nll, best_epoch = -1.0, float("inf"), 0
    resume = output / "resume.pt"
    microbatch = 32 if method == "frozen" else preflight(encoder, head, run)
    if resume.exists():
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume checkpoint configuration or data changed")
        encoder.mods.embedding_model.load_state_dict(checkpoint["encoder"])
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch, cursor, stale, history, best_f1, best_nll, best_epoch = [
            checkpoint[key]
            for key in (
                "epoch",
                "cursor",
                "stale",
                "history",
                "best_f1",
                "best_nll",
                "best_epoch",
            )
        ]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint
    accumulation = 32 // microbatch

    def save_resume():
        save_torch(
            resume,
            {
                "identity": identity,
                "encoder": cpu_state(encoder.mods.embedding_model),
                "head": cpu_state(head),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "cursor": cursor,
                "stale": stale,
                "history": history,
                "best_f1": best_f1,
                "best_nll": best_nll,
                "best_epoch": best_epoch,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
        )

    trainable_names = [
        name
        for name, parameter in encoder.named_parameters()
        if parameter.requires_grad
    ]
    write_json(
        output / "configuration.json",
        {
            **identity,
            "microbatch": microbatch,
            "gradient_accumulation": accumulation,
            "trainable_encoder_parameters": trainable_names,
            "train_clips": len(targets),
            "selection_clips": len(validation_targets),
        },
    )
    while epoch <= SETTINGS["epochs"] and stale < SETTINGS["patience"]:
        configure(encoder, head, method)
        order = epoch_order(
            targets, epoch, SETTINGS["examples_per_epoch"], speaker_groups
        )
        for start in range(cursor, len(order), 32):
            for attempt in range(8):
                optimizer.zero_grad(set_to_none=True)
                for subset in order[start : start + 32].split(microbatch):
                    with torch.autocast("cuda", dtype=torch.float16):
                        if method == "frozen":
                            embedding = embeddings[subset].to("cuda").unsqueeze(1)
                        else:
                            signals, lengths = audio_batch(
                                [all_training_rows[i] for i in subset.tolist()], "cuda"
                            )
                            embedding = tail_embedding(encoder, signals, lengths)
                        logits = head(embedding).squeeze(1)
                        loss = (
                            F.cross_entropy(logits.float(), targets[subset].to("cuda"))
                            / accumulation
                        )
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    scaler.scale(loss).backward()
                if scaled_optimizer_step(scaler, optimizer):
                    break
                print(
                    json.dumps(
                        {
                            "amp_retry": attempt + 1,
                            "method": method,
                            "epoch": epoch,
                            "cursor": start,
                            "reduced_scale": scaler.get_scale(),
                        }
                    ),
                    flush=True,
                )
            else:
                raise ValueError(
                    "Nonfinite gradients persisted after eight reduced-scale retries"
                )
            cursor = start + 32
            if cursor % 4096 == 0:
                save_resume()
                emit(
                    f"train_{method}",
                    epoch=epoch,
                    max_epochs=SETTINGS["epochs"],
                    completed=cursor,
                    total=len(order),
                    percent=round(cursor / len(order) * 100, 1),
                    best_epoch=best_epoch,
                    stale_epochs=stale,
                    patience=SETTINGS["patience"],
                    peak_vram_gib=round(torch.cuda.max_memory_reserved() / 1024**3, 2),
                )
        if method == "partial":
            encoder.eval()
            fingerprint = state_fingerprint(encoder.mods.embedding_model)
            validation_embeddings = cache_embeddings(
                encoder,
                all_validation_rows,
                output / f"selection_epoch_{epoch:02d}.pt",
                fingerprint,
                f"selection_{method}_epoch_{epoch}",
                progress=emit,
            )
        scores = head_scores(head, validation_embeddings)
        metrics = selection_metrics(scores, validation_targets)
        history.append({"epoch": epoch, **metrics})
        verify_frozen(encoder, expected_frozen)
        if (metrics["macro_f1"], -metrics["balanced_nll"]) > (best_f1, -best_nll):
            best_f1, best_nll, best_epoch, stale = (
                metrics["macro_f1"],
                metrics["balanced_nll"],
                epoch,
                0,
            )
            save_torch(
                output / "model.pt",
                {
                    "format_version": 2,
                    "identity": identity,
                    "encoder": cpu_state(encoder.mods.embedding_model),
                    "head": cpu_state(head),
                    "codes": CODES,
                    "names": NAMES,
                    "epoch": epoch,
                    "selection_metrics": metrics,
                    "preprocessing": {
                        "sample_rate": 16000,
                        "max_seconds": 10,
                        "crop": "center",
                        "waveform_normalization": "none",
                    },
                },
            )
        else:
            stale += 1
        write_json(output / "history.json", history)
        emit(
            f"epoch_complete_{method}",
            epoch=epoch,
            best_epoch=best_epoch,
            stale_epochs=stale,
            **metrics,
        )
        epoch, cursor = epoch + 1, 0
        save_resume()
    verify_frozen(encoder, expected_frozen)
    write_json(
        output / "complete.json",
        {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "best_macro_f1": best_f1,
            "frozen_parameters_verified": True,
            "peak_reserved_vram_gib": torch.cuda.max_memory_reserved() / 1024**3,
        },
    )
