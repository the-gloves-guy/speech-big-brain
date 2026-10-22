import hashlib
import json
from argparse import Namespace

import torch

from lid.cache import disjoint, load_cache
from lid.languages import CODES
from lid.prefetch import merge_prefetched


def test_prefetch_merge_preserves_features_labels_and_group_disjointness(tmp_path):
    directory = tmp_path / "prefetch"
    directory.mkdir()
    (directory / "session.json").write_text(
        json.dumps(
            {"phase": "complete", "model_revision": "test-sha", "max_seconds": 10.0}
        )
    )
    train, valid = [], []
    expected = {}
    meta = {
        "model": "test-model",
        "model_revision": "test-sha",
        "max_seconds": 10.0,
        "crop": "center",
        "sample_rate": 16000,
        "codes": CODES,
        "source": "voxlingua",
        "split": "all",
        "dataset_revision": None,
        "limit_per_language": 0,
    }
    labels = [f"{code}: label" for code in CODES] + [f"other{i}" for i in range(95)]
    for index, code in enumerate(CODES):
        rows = []
        for split in ("train", "validation"):
            audio = tmp_path / f"{code}-{split}.wav"
            audio.touch()
            row = {
                "path": str(audio),
                "language": code,
                "group": f"{code}-{split}-video",
            }
            rows.append(row)
            (train if split == "train" else valid).append(row)
        features = torch.randn(2, 256)
        for row, feature in zip(rows, features):
            expected[row["path"]] = feature
        torch.save(
            {
                "embeddings": features,
                "logits": torch.randn(2, 107),
                "targets": torch.full((2,), index, dtype=torch.long),
                "ids": [row["path"] for row in rows],
                "groups": [row["group"] for row in rows],
                "labels": labels,
                "head_state": {},
                "meta": {**meta, "manifest_sha256": f"source-{code}"},
            },
            directory / f"{code}.pt",
        )
    caches = []
    for split, rows in (("train", train), ("validation", valid)):
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        output = tmp_path / f"{split}.pt"
        merge_prefetched(
            Namespace(
                manifest=str(manifest),
                output=str(output),
                split=split,
                model_revision="test-sha",
                max_seconds=10.0,
                limit_per_language=0,
            ),
            directory,
        )
        result = load_cache(output)
        assert (
            result["meta"]["manifest_sha256"]
            == hashlib.sha256(manifest.read_bytes()).hexdigest()
        )
        assert result["targets"].tolist() == list(range(12))
        for item, feature in zip(result["ids"], result["embeddings"]):
            torch.testing.assert_close(feature, expected[item])
        caches.append(result)
    disjoint(*caches)
