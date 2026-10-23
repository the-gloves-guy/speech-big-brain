"""Invariants that protect the old benchmark and encoder during adaptation."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from lid.extension_data import CODES, check_snapshot, split_for_group
from lid.extension_model import (
    configure,
    frozen_fingerprints,
    tail_embedding,
    verify_frozen,
)
from lid.extension_train import epoch_order, selection_metrics


def test_registry_excludes_rar_and_preserves_old_indices():
    from lid.languages import CODES as old

    assert CODES == [*old, "sam", "niu", "ton"]
    assert len(CODES) == 15
    assert "rar" not in CODES


def test_source_split_keeps_recordings_together_and_is_reproducible():
    first = [split_for_group(f"recording-{i}") for i in range(1000)]
    assert first == [split_for_group(f"recording-{i}") for i in range(1000)]
    assert set(first) == {"train", "selection", "calibration", "test"}
    assert 750 < first.count("train") < 850


def test_inventory_changes_are_rejected(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"original")
    stat = path.stat()
    row = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    check_snapshot(row)
    path.write_bytes(b"changed audio")
    with pytest.raises(ValueError, match="changed since audit"):
        check_snapshot(row)


def test_balanced_sampling_is_identical_for_both_methods():
    targets = torch.tensor([i for i in range(15) for _ in range(i + 1)])
    first = epoch_order(targets, 1, 15000)
    assert torch.equal(first, epoch_order(targets, 1, 15000))
    assert not torch.equal(first, epoch_order(targets, 2, 15000))
    counts = torch.bincount(targets[first], minlength=15)
    assert counts.min() > 850 and counts.max() < 1150


def test_amp_overflow_preserves_weights_and_allows_retry():
    from lid.extension_train import scaled_optimizer_step

    parameter = nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    scaler.scale(parameter.square().sum()).backward()
    parameter.grad.fill_(float("inf"))
    assert not scaled_optimizer_step(scaler, optimizer)
    assert parameter.item() == 2
    assert scaler.get_scale() == 4
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter.square().sum()).backward()
    assert scaled_optimizer_step(scaler, optimizer)
    assert parameter.item() < 2


def test_macro_selection_gives_new_languages_equal_weight():
    targets = torch.arange(15).repeat_interleave(2)
    logits = torch.eye(15)[targets] * 10
    metrics = selection_metrics(logits, targets)
    assert metrics["macro_f1"] == 1
    assert metrics["old_accuracy"] == metrics["new_accuracy"] == 1


class PassthroughNorm(nn.Module):
    def forward(self, features, lengths):
        return features


def test_real_ecapa_tail_backpropagates_only_into_selected_layers():
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    torch.set_num_threads(2)
    network = ECAPA_TDNN(
        input_size=5,
        channels=[8, 8, 8, 8, 24],
        lin_neurons=4,
        attention_channels=8,
        se_channels=8,
    )
    encoder = nn.Module()
    encoder.mods = nn.ModuleDict(
        {
            "embedding_model": network,
            "compute_features": nn.Identity(),
            "mean_var_norm": PassthroughNorm(),
        }
    )
    head = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 15))
    configure(encoder, head, "partial")
    expected = frozen_fingerprints(encoder)
    features = torch.randn(2, 40, 5)
    lengths = torch.ones(2)
    network.eval()
    with torch.no_grad():
        direct = network(features, lengths)
    actual = tail_embedding(encoder, features, lengths)
    torch.testing.assert_close(actual, direct)
    loss = head(actual.squeeze(1)).square().mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in network.blocks.parameters())
    assert any(parameter.grad is not None for parameter in network.mfa.parameters())
    assert any(parameter.grad is not None for parameter in network.asp.parameters())
    assert all(parameter.grad is not None for parameter in network.fc.parameters())
    optimizer = torch.optim.SGD(
        [p for p in encoder.parameters() if p.requires_grad], lr=0.01
    )
    optimizer.step()
    verify_frozen(encoder, expected)
    assert not any(
        child.training
        for child in encoder.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    )


def test_audited_partitions_are_disjoint_when_available():
    from lid.extension_data import RUN, read_rows

    if not (RUN / "data_ready.json").exists():
        pytest.skip("No actual extension audit in this checkout")
    seen_groups, seen_hashes = set(), set()
    for split in ("train", "selection", "calibration", "test"):
        rows = read_rows(RUN / f"new_{split}.jsonl")
        assert {row["language"] for row in rows} == {"sam", "niu", "ton"}
        groups = {row["group"] for row in rows}
        hashes = {row["sha256"] for row in rows}
        assert not seen_groups.intersection(groups)
        assert not seen_hashes.intersection(hashes)
        seen_groups.update(groups)
        seen_hashes.update(hashes)
    summary = json.loads((RUN / "data_ready.json").read_text())
    assert (
        sum(sum(split["clips"].values()) for split in summary["splits"].values())
        == summary["original_files"] - summary["excluded"]
    )


def test_report_builds_tables_and_publication_figures(tmp_path, monkeypatch):
    from lid import extension_evaluate as module
    from lid.extension_data import NAMES
    from lid.metrics import evaluate

    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(module, "RUN", run)
    (run / "data_ready.json").write_text(
        json.dumps({"original_files": 100, "excluded": 0})
    )
    targets = np.arange(15)
    logits = np.eye(15) * 5
    reports = {}
    for method in ("frozen", "partial"):
        directory = run / method
        directory.mkdir()
        measurements = [
            {
                "language": "sam",
                "gain_db_requested": gain,
                "embedding_cosine": 0.999,
                "prediction_changed": False,
            }
            for gain in (-12, 12)
        ]
        (directory / "gain_diagnostic.json").write_text(
            json.dumps({"measurements": measurements})
        )
        reports[method] = {}
        for group, selection, indices in (
            ("all15", slice(None), list(range(15))),
            ("old12", slice(0, 12), list(range(12))),
            ("new3", slice(12, 15), list(range(12, 15))),
        ):
            value = evaluate(
                logits[selection],
                targets[selection],
                [NAMES[i] for i in indices],
                report_indices=indices,
            )
            reports[method][group] = {"raw": value, "joint_temperature": value}
    reference = {
        "retrained_head_12": evaluate(np.eye(12) * 5, np.arange(12), NAMES[:12])
    }
    module.write_extension_report(reference, reports)
    report = tmp_path / "reports/extension15/language_extension.md"
    assert report.exists()
    assert "Cook Islands Māori is excluded" in report.read_text()
    assert len(list(report.parent.glob("figures/*.pdf"))) == 3
    assert len(list(report.parent.glob("figures/*.svg"))) == 3
