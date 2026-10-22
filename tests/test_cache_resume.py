from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lid import cache
from lid.languages import CODES


def test_equal_length_batching_and_resume_after_final_write_failure(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "source.jsonl"
    manifest.write_text("manifest fingerprint")
    records = [
        {"id": f"{code}-{i}", "language": code, "group": f"video-{code}-{i}"}
        for code in CODES
        for i in range(2)
    ]
    calls = []
    classifier = torch.nn.Linear(256, 107)

    def encode_batch(signal, normalize):
        assert normalize is False
        calls.append(tuple(signal.shape))
        return signal.mean(1).reshape(-1, 1, 1).expand(-1, 1, 256)

    encoder = SimpleNamespace(
        encode_batch=encode_batch, mods=SimpleNamespace(classifier=classifier)
    )
    monkeypatch.setattr(
        cache, "load_encoder", lambda *a: (encoder, list(range(107)), "sha")
    )
    monkeypatch.setattr(cache, "read_manifest", lambda path: records)
    monkeypatch.setattr(
        cache,
        "waveform",
        lambda row, limit: np.ones(
            1600 if row["id"].endswith("0") else 3200, dtype=np.float32
        ),
    )
    output = tmp_path / "cache.pt"
    args = Namespace(
        output=str(output),
        manifest=str(manifest),
        limit_per_language=0,
        max_seconds=10.0,
        device="cpu",
        threads=2,
        batch_size=8,
        model_revision="sha",
        split="train",
    )
    original_save = torch.save

    def interrupted_save(value, path):
        if path == output.with_suffix(".pt.tmp"):
            raise RuntimeError("simulated interruption")
        original_save(value, path)

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        cache.extract(args)
    assert not output.exists()
    assert {shape[1] for shape in calls} == {1600, 3200}
    assert max(shape[0] for shape in calls) == 8
    count = len(calls)
    monkeypatch.setattr(torch, "save", original_save)
    cache.extract(args)
    assert len(calls) == count  # The resumed run performs no repeated audio inference.
    saved = cache.load_cache(output)
    assert set(saved["ids"]) == {row["id"] for row in records}
    assert saved["embeddings"].shape == (24, 256)
    assert torch.all(saved["embeddings"] == 1)
