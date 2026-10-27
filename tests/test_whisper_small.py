import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from whisper.model import AudioEncoder

from lid.extension_data import CODES, file_hash, write_json
from lid.whisper_extension import expanded_head
from lid.whisper_model import load_whisper, whisper_head
from lid.whisper_small import SmallAdapter, SmallStudy


def test_expanded_small_classifier_preserves_all_old_parameters():
    original = whisper_head(768)
    first = expanded_head(original.state_dict())
    second = expanded_head(original.state_dict())
    assert first[1].in_features == 1536
    for key, value in original.state_dict().items():
        torch.testing.assert_close(
            first.state_dict()[key][:12]
            if key.startswith("3.")
            else first.state_dict()[key],
            value,
            atol=0,
            rtol=0,
        )
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    assert first(torch.zeros(2, 1536)).shape == (2, 15)


@pytest.fixture
def encoder_root(tmp_path):
    root = tmp_path / "small"
    directory = root / "pretrained"
    directory.mkdir(parents=True)
    dims = {
        "n_mels": 80,
        "n_audio_ctx": 1500,
        "n_audio_state": 8,
        "n_audio_head": 2,
        "n_audio_layer": 1,
    }
    encoder = AudioEncoder(80, 1500, 8, 2, 1)
    path = directory / "small_encoder.pt"
    torch.save({"dims": dims, "encoder": encoder.state_dict()}, path)
    write_json(
        root / "model_identity.json",
        {
            "model": "small",
            "dims": dims,
            "encoder_checkpoint": path.name,
            "encoder_checkpoint_sha256": file_hash(path),
        },
    )
    return root, dims, encoder


def test_encoder_only_loading_uses_dimensions_and_rejects_stale_checkpoint(
    encoder_root,
):
    root, dims, encoder = encoder_root
    loaded = load_whisper(root, "cpu")
    assert not hasattr(loaded, "decoder")
    assert vars(loaded.dims) == dims
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, loaded.encoder.state_dict()[key])
    path = root / "pretrained/small_encoder.pt"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        load_whisper(root, "cpu")


def test_checkpointed_encoder_matches_forward_and_gradients(encoder_root, monkeypatch):
    import lid.whisper_small as module

    _, _, encoder = encoder_root
    first = object.__new__(SmallAdapter)
    second = object.__new__(SmallAdapter)
    first.encoder = encoder.train()
    second.encoder = copy.deepcopy(encoder).train()
    first.gradient_checkpointing = False
    second.gradient_checkpointing = True
    mels = torch.randn(2, 80, 3000)
    lengths = torch.tensor([160000, 1600])
    monkeypatch.setattr(module, "mel_batch", lambda rows: (mels, lengths))
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        one = first.embeddings([{}, {}])
        two = second.embeddings([{}, {}])
        torch.testing.assert_close(one, two)
        one.square().sum().backward()
        two.square().sum().backward()
        for p, q in zip(first.encoder.parameters(), second.encoder.parameters()):
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)
    finally:
        torch.set_num_threads(threads)


def test_small_study_is_isolated_and_rejects_changed_config(tmp_path):
    config = json.loads(Path("configs/whisper_small15.json").read_text())
    config["base"] = str(tmp_path / "warm")
    path = tmp_path / "config.json"
    write_json(path, config)
    study = SmallStudy(path, tmp_path / "small15")
    assert study.warm.config["frozen_epochs"] == 30
    assert study.warm.config["frozen_batch"] == 256
    write_json(path, dict(config, seed=43))
    with pytest.raises(ValueError, match="identity changed"):
        SmallStudy(path, study.root)


def test_export_yaml_loads_non_tiny_dimensions_offline(
    encoder_root, tmp_path, monkeypatch
):
    import socket

    from speechbrain.inference.classifiers import EncoderClassifier

    from lid.whisper_extension_export import export_one

    root, dims, encoder = encoder_root
    source = root / "frozen"
    source.mkdir()
    head = expanded_head(whisper_head(8).state_dict())
    torch.save(
        {"encoder": encoder.state_dict(), "head": head.state_dict(), "epoch": 1},
        source / "model.pt",
    )
    write_json(source / "evaluation_complete.json", {})
    write_json(
        source / "calibration.json",
        {
            "codes": CODES,
            "temperature": 1.4,
            "model_sha256": file_hash(source / "model.pt"),
        },
    )
    output = tmp_path / "bundle"
    export_one(SimpleNamespace(root=root), "frozen", output)

    def denied(*a, **kw):
        raise AssertionError("Network access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    model = EncoderClassifier.from_hparams(
        source=str(output), savedir=str(tmp_path / "loaded"), run_opts={"device": "cpu"}
    )
    assert (
        model.mods.embedding_model.encoder.conv1.out_channels == dims["n_audio_state"]
    )
    assert model.mods.classifier[1].in_features == 16
    assert model.mods.classifier[3].out_features == 15
    metadata = json.loads((output / "export.json").read_text())
    assert metadata["model"] == "small"
    assert metadata["dims"] == dims


def test_comparator_rejects_changed_ids_nonfinite_scores_and_duplicate_labels(tmp_path):
    from lid.whisper_small_results import read_scores

    path = tmp_path / "scores.npz"
    values = np.zeros((2, 15))
    np.savez(path, ids=["a", "b"], logits=values, codes=CODES)
    assert read_scores(path, ["a", "b"])[1] == CODES
    with pytest.raises(ValueError, match="IDs differ"):
        read_scores(path, ["b", "a"])
    values[0, 0] = np.nan
    np.savez(path, ids=["a", "b"], logits=values, codes=CODES)
    with pytest.raises(ValueError, match="score matrix"):
        read_scores(path, ["a", "b"])
    np.savez(path, ids=["a", "b"], logits=np.zeros((2, 15)), codes=[*CODES[:14], "sam"])
    with pytest.raises(ValueError, match="vocabulary"):
        read_scores(path, ["a", "b"])


def test_small_final_report_builds_from_complete_fixture(tmp_path, monkeypatch):
    import csv
    import shutil

    import lid.whisper_small_report as module
    from lid.whisper_small_report import build
    from lid.youtube_evaluate import write_csv

    project = Path(__file__).resolve().parents[1]
    reference = project / "artifacts/whisper_tiny15"
    root = tmp_path / "artifacts/whisper_small15_fixture"
    root.mkdir(parents=True)
    for method in ["frozen", "finetuned"]:
        folder = root / method
        folder.mkdir()
        for name in [
            "benchmarks.json",
            "history.json",
            "summary.csv",
            "per_language.csv",
        ]:
            shutil.copyfile(reference / method / name, folder / name)
    shutil.copyfile(reference / "data_ready.json", root / "data_ready.json")
    write_json(root / "preflight.json", {"fixture": True, "microbatch": 2})
    external = json.loads(
        (project / "artifacts/youtube_validation_v4/matched_report.json").read_text()
    )
    for method in ["frozen", "finetuned"]:
        values = external["models"][f"whisper15_{method}"]
        external["models"][f"whisper_small15_{method}"] = values
        external["models"][f"whisper_tiny15_{method}"] = values
    write_json(root / "external/matched_report.json", external)
    for name in [
        "matched_summary.csv",
        "matched_per_language.csv",
        "matched_per_source.csv",
    ]:
        shutil.copyfile(
            project / "artifacts/youtube_validation_v4" / name, root / "external" / name
        )
    (root / "diagnostics").mkdir()
    for name in ["duration.csv", "gain.csv"]:
        with (reference / "diagnostics" / name).open() as stream:
            values = list(csv.DictReader(stream))
        for row in values:
            row["model"] = row["model"].replace("whisper15_", "whisper_small15_")
        write_csv(root / "diagnostics" / name, values)
    write_json(
        root / "diagnostics/complete.json",
        {"duration_languages": ["hi", "th", "sam", "ton"]},
    )
    copied = root / "frozen/figures/combined_test_joint_confusion.png"
    copied.parent.mkdir()
    shutil.copyfile(
        reference / "frozen/figures/combined_test_joint_confusion.png", copied
    )
    external_plot = root / "external/figures/external_confusion.png"
    external_plot.parent.mkdir()
    shutil.copyfile(copied, external_plot)
    monkeypatch.chdir(tmp_path)
    for method in ["frozen", "finetuned"]:
        directory = Path(f"artifacts/deployment/whisper_small15_{method}")
        directory.mkdir(parents=True)
        write_json(directory / "export.json", {"fixture": True})
        write_json(
            directory / "verification.json",
            {
                "n": 14400,
                "prediction_disagreements": 0,
                "max_absolute_log_probability_error": 0.00001,
            },
        )
        archive = directory.with_suffix(".tar.gz")
        archive.write_bytes(b"fixture")
        Path(str(archive) + ".sha256").write_text(
            file_hash(archive) + "  " + archive.name
        )

    def fake_pdf(html_path, pdf_path):
        pdf_path.write_bytes(b"fixture PDF")
        return (
            "Whisper Small15 Complete plot appendix "
            + str(copied)
            + " "
            + str(external_plot)
        )

    monkeypatch.setattr(module, "render_pdf", fake_pdf)
    study = SimpleNamespace(
        root=root,
        reference=reference,
        base=project / "artifacts/whisper_tiny12",
        pacific=project / "artifacts/extension15",
    )
    build(study)
    output = tmp_path / "reports" / root.name
    report = (output / "whisper_small15_analysis.html").read_text()
    assert "no matched fine-tuned Small12 regression" in report
    assert "Missing matched duration languages" in report
    assert "data:image/png;base64," in report
    assert (root / "report_complete.json").exists()
    manifest = json.loads((output / "report_manifest.json").read_text())
    assert manifest["plots_embedded"] == 2


def test_pdf_renderer_preserves_browser_failure_details(tmp_path, monkeypatch):
    import lid.whisper_small_results as module

    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/google-chrome")

    def failed(command, **kwargs):
        assert "--renderer-process-limit=1" in command
        assert kwargs["check"] is False
        return SimpleNamespace(
            returncode=-5, stderr="pthread_create: Resource temporarily unavailable"
        )

    monkeypatch.setattr(module.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="pthread_create"):
        module.render_pdf(tmp_path / "report.html", tmp_path / "report.pdf")
