import json
from argparse import Namespace

import numpy as np
import pytest
import soundfile as sf
import torch

from lid.cache import disjoint
from lid.compare import compare
from lid.data import split_groups, waveform
from lid.languages import CODES, LANGUAGES, NAMES, selected_indices
from lid.model import cpu_state, new_head
from lid.train import head_logits, train


def pretrained_head():
    from speechbrain.lobes.models.Xvector import Classifier

    torch.manual_seed(1)
    head = Classifier(
        input_shape=[None, None, 256], lin_blocks=1, lin_neurons=512, out_neurons=107
    )
    head.softmax = torch.nn.Identity()
    return head.eval()


def test_head_initialization_exactly_matches_selected_pretrained_logits():
    torch.set_num_threads(2)
    original = pretrained_head()
    indices = [4, 0, 50, 3, 6, 20, 70, 90, 5, 30, 40, 100]
    reduced = new_head(cpu_state(original), indices).eval()
    embeddings = torch.randn(4, 1, 256)
    with torch.inference_mode():
        torch.testing.assert_close(
            reduced(embeddings), original(embeddings)[:, :, indices]
        )


def test_group_split_is_disjoint_and_reproducible():
    rows = [
        {"language": code, "group": f"{code}-{g}", "path": f"{code}-{g}-{i}"}
        for code in CODES
        for g in range(5)
        for i in range(3)
    ]
    first, second = split_groups(rows)
    assert not {r["group"] for r in first} & {r["group"] for r in second}
    assert len(first) + len(second) == len(rows)
    assert (first, second) == split_groups(rows)


def test_overlap_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        disjoint(
            {"ids": ["a"], "groups": ["video"]}, {"ids": ["b"], "groups": ["video"]}
        )


def test_fleurs_uses_disk_archives_and_reuses_extracted_audio(tmp_path, monkeypatch):
    import io
    import tarfile

    from lid import fleurs

    metadata = tmp_path / "dev.tsv"
    metadata.write_text("17\tclip.wav\ttranscription\n")
    audio = io.BytesIO()
    sf.write(audio, np.ones(3200, dtype=np.float32) * 0.25, 16000, format="WAV")
    payload = audio.getvalue()
    archive = tmp_path / "dev.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("dev/clip.wav")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))
    requested = []

    def download(relative, revision, root):
        assert revision == "immutable-sha"
        requested.append(relative)
        return metadata if relative.endswith(".tsv") else archive

    monkeypatch.setattr(fleurs, "download_file", download)
    monkeypatch.setattr(fleurs, "LANGUAGES", LANGUAGES[:1])
    rows = list(fleurs.records("validation", "immutable-sha", root=tmp_path / "data"))
    assert len(rows) == 1 and "audio" not in rows[0]
    assert rows[0]["id"] == "ar_eg/validation/clip.wav"
    assert rows[0]["group"] == "ar_eg/validation/17"
    assert waveform(rows[0]).mean() == pytest.approx(0.25, abs=1e-4)
    assert (
        list(fleurs.records("validation", "immutable-sha", root=tmp_path / "data"))
        == rows
    )
    assert sum(path.endswith(".tar.gz") for path in requested) == 1


def test_center_crop_and_resampling(tmp_path):
    path = tmp_path / "audio.wav"
    signal = np.zeros((24000 * 4, 2), dtype=np.float32)
    signal[24000:72000] = 0.5
    sf.write(path, signal, 24000)
    result = waveform({"path": str(path), "id": "sample"}, max_seconds=1)
    assert result.shape == (16000,)
    assert result[100:-100].mean() == pytest.approx(0.5, abs=1e-3)


def test_training_calibration_reporting_end_to_end(tmp_path):
    """Synthetic data verifies execution and leakage guards; it is not a benchmark."""
    torch.set_num_threads(2)
    original = pretrained_head()
    labels = [f"{code}: {name}" for code, name in zip(CODES, NAMES)] + [
        f"other{i}" for i in range(95)
    ]
    meta = {
        "model": "test-model",
        "model_revision": "test-sha",
        "max_seconds": 10.0,
        "crop": "center",
        "sample_rate": 16000,
        "codes": CODES,
        "dataset_revision": "dataset-sha",
    }
    paths = {}
    for split, source, n in [
        ("train", "voxlingua", 25),
        ("validation", "voxlingua", 24),
        ("validation", "fleurs", 24),
        ("test", "fleurs", 24),
    ]:
        name = f"{source}_{split}"
        features = torch.randn(n, 256)
        data = {
            "embeddings": features,
            "logits": head_logits(original, features),
            "targets": torch.arange(n) % 12,
            "labels": labels,
            "head_state": cpu_state(original),
            "ids": [f"{name}/clip{i}" for i in range(n)],
            "groups": [f"{name}/group{i}" for i in range(n)],
            "meta": {**meta, "source": source, "split": split},
        }
        paths[name] = tmp_path / f"{name}.pt"
        torch.save(data, paths[name])
    output = tmp_path / "head"
    train(
        Namespace(
            train=str(paths["voxlingua_train"]),
            validation=str(paths["voxlingua_validation"]),
            output=str(output),
            epochs=2,
            batch_size=24,
            patience=2,
            lr=1e-4,
            weight_decay=1e-4,
            device="cpu",
            threads=2,
            seed=42,
        )
    )
    report_dir = tmp_path / "report"
    compare(
        Namespace(
            calibration=str(paths["fleurs_validation"]),
            test=str(paths["fleurs_test"]),
            head=str(output / "head.pt"),
            output=str(report_dir),
            bins=15,
            threads=2,
        )
    )
    report = json.loads((report_dir / "report.json").read_text())
    assert len(report["methods"]) == 6
    assert (
        report["methods"]["filtered_12"]["accuracy"]
        == report["methods"]["filtered_12_temperature"]["accuracy"]
    )
    assert (report_dir / "reliability.png").is_file()
    with np.load(report_dir / "predictions.npz") as predictions:
        assert predictions["head_logits"].shape == (24, 12)
    assert selected_indices(labels) == list(range(12))
