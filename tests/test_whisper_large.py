"""Large v3 frontend, bounded loading, classifier-only checkpoints and reporting."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from whisper.model import AudioEncoder

from lid.extension_data import CODES, file_hash, write_json
from lid.whisper_export_backend import WhisperLIDEncoder
from lid.whisper_extension import expanded_head
from lid.whisper_model import load_whisper, whisper_head


@pytest.fixture
def large_fixture(tmp_path):
    root = tmp_path / "large"
    (root / "pretrained").mkdir(parents=True)
    dims = {
        "n_mels": 128,
        "n_audio_ctx": 1500,
        "n_audio_state": 8,
        "n_audio_head": 2,
        "n_audio_layer": 1,
    }
    encoder = AudioEncoder(128, 1500, 8, 2, 1)
    path = root / "pretrained/encoder.pt"
    torch.save({"dims": dims, "encoder": encoder.state_dict()}, path)
    identity = {
        "model": "large-v3",
        "dims": dims,
        "bounded_memory_loading": True,
        "encoder_checkpoint": path.name,
        "encoder_checkpoint_sha256": file_hash(path),
    }
    write_json(root / "model_identity.json", identity)
    return root, encoder, identity


def test_bounded_encoder_load_and_128_channel_variable_length_forward(large_fixture):
    root, encoder, _ = large_fixture
    model = load_whisper(root, "cpu")
    assert not hasattr(model, "decoder")
    backend = WhisperLIDEncoder(
        n_mels=128, n_state=8, n_head=2, n_layer=1, bounded_memory_loading=True
    )
    backend.load_state_dict(
        {"encoder." + k: v for k, v in encoder.state_dict().items()}
    )
    backend.eval()
    assert all(p.device.type == "cpu" for p in backend.parameters())
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(
            model.encoder.state_dict()[key], value, atol=0, rtol=0
        )
    signal = torch.randn(2, 4800)
    signal[1, 1600:] = 0
    with torch.inference_mode():
        batch = backend(signal, torch.tensor([1.0, 1 / 3]))
        one = backend(signal[1, :1600])
    assert batch.shape == (2, 1, 16) and torch.isfinite(batch).all()
    torch.testing.assert_close(batch[1:], one, atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match="at least"):
        backend(torch.zeros(1599))


def test_cached_adapter_never_loads_encoder_and_checks_reference(
    large_fixture, monkeypatch
):
    import lid.whisper_large as module
    from lid.whisper_large import CachedHeadAdapter

    root, _, identity = large_fixture
    head = expanded_head(whisper_head(8).state_dict())
    checkpoint = root / "head.pt"
    torch.save(
        {
            "head": head.state_dict(),
            "encoder_checkpoint_sha256": identity["encoder_checkpoint_sha256"],
        },
        checkpoint,
    )
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(
        module,
        "load_whisper",
        lambda *a, **k: pytest.fail("Encoder loaded during cached training"),
    )
    adapter = CachedHeadAdapter(root, checkpoint)
    assert not list(adapter.encoder.parameters())
    assert adapter.head(torch.randn(2, 16)).shape == (2, 15)
    with pytest.raises(ValueError, match="only supports"):
        CachedHeadAdapter(root, checkpoint, True)
    torch.save(
        {"head": head.state_dict(), "encoder_checkpoint_sha256": "changed"}, checkpoint
    )
    with pytest.raises(ValueError, match="reference changed"):
        CachedHeadAdapter(root, checkpoint)


def test_large_configuration_and_preflight_reject_stale_state(tmp_path):
    from lid.whisper_large import LargeStudy

    config = json.loads(Path("configs/whisper_large_v3_15.json").read_text())
    config["base"] = str(tmp_path / "warm")
    path = tmp_path / "config.json"
    write_json(path, config)
    study = LargeStudy(path, tmp_path / "large15")
    assert study.methods == ("frozen",)
    assert study.warm.config["frozen_epochs"] == 30
    write_json(
        study.base / "model_identity.json", {"encoder_checkpoint_sha256": "expected"}
    )
    write_json(
        study.root / "preflight.json",
        {"config_sha256": "wrong", "encoder_checkpoint_sha256": "expected"},
    )
    with pytest.raises(ValueError, match="stale"):
        study.apply_preflight()
    config["seed"] = 43
    write_json(path, config)
    with pytest.raises(ValueError, match="identity changed"):
        LargeStudy(path, study.root)


def test_classifier_reference_export_loads_offline_without_duplicate_model(
    large_fixture, tmp_path, monkeypatch
):
    import socket

    from speechbrain.inference.classifiers import EncoderClassifier

    import lid.whisper_model as module
    from lid.whisper_extension_export import export_one

    root, encoder, identity = large_fixture
    folder = root / "frozen"
    folder.mkdir()
    head = expanded_head(whisper_head(8).state_dict())
    torch.save(
        {
            "head": head.state_dict(),
            "encoder": {},
            "encoder_checkpoint_sha256": identity["encoder_checkpoint_sha256"],
            "epoch": 1,
        },
        folder / "model.pt",
    )
    write_json(folder / "evaluation_complete.json", {})
    write_json(
        folder / "calibration.json",
        {
            "codes": CODES,
            "temperature": 1.7,
            "model_sha256": file_hash(folder / "model.pt"),
        },
    )
    monkeypatch.setattr(
        module, "load_whisper", lambda root: SimpleNamespace(encoder=encoder)
    )
    export_one(SimpleNamespace(root=root), "frozen", tmp_path / "bundle")

    def denied(*a, **k):
        raise AssertionError("Network access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    model = EncoderClassifier.from_hparams(
        source=str(tmp_path / "bundle"),
        savedir=str(tmp_path / "loaded"),
        run_opts={"device": "cpu"},
    )
    assert model.mods.embedding_model.n_mels == 128
    assert model.mods.embedding_model.bounded_memory_loading
    torch.testing.assert_close(model.mods.classifier[3].weight, head[3].weight / 1.7)
    with torch.inference_mode():
        scores = model.classify_batch(torch.randn(2, 4800), torch.tensor([1.0, 0.5]))[0]
    assert scores.reshape(2, 15).isfinite().all()


def test_large_report_with_completed_fixture(tmp_path, monkeypatch):
    import shutil

    import lid.whisper_large_report as module
    from lid.whisper_large import LargeStudy
    from lid.whisper_large_results import csv_rows
    from lid.youtube_evaluate import write_csv

    project = Path(__file__).resolve().parents[1]
    small = project / "artifacts/whisper_small15"
    root = tmp_path / "artifacts/large_fixture"
    (root / "frozen").mkdir(parents=True)
    for name in ["benchmarks.json", "history.json", "summary.csv", "per_language.csv"]:
        shutil.copyfile(small / "frozen" / name, root / "frozen" / name)
    shutil.copyfile(small / "data_ready.json", root / "data_ready.json")
    write_json(root / "preflight.json", {"fixture": True})
    external = json.loads((small / "external/matched_report.json").read_text())
    external["models"]["whisper_large_v3_15_frozen"] = external["models"][
        "whisper_small15_frozen"
    ]
    external["models"] = {
        n: external["models"][n] for n in LargeStudy.comparison_models
    }
    write_json(root / "external/matched_report.json", external)
    for name in [
        "matched_predictions.csv",
        "matched_summary.csv",
        "matched_per_language.csv",
        "matched_per_source.csv",
    ]:
        values = csv_rows(small / "external" / name)
        values += [
            {**r, "model": "whisper_large_v3_15_frozen"}
            for r in values
            if r["model"] == "whisper_small15_frozen"
        ]
        write_csv(
            root / "external" / name,
            [r for r in values if r["model"] in LargeStudy.comparison_models],
        )
    (root / "diagnostics").mkdir()
    shutil.copyfile(
        small / "diagnostics/complete.json", root / "diagnostics/complete.json"
    )
    for name in ["duration.csv", "gain.csv"]:
        values = csv_rows(small / "diagnostics" / name)
        values += [
            {**r, "model": "whisper_large_v3_15_frozen"}
            for r in values
            if r["model"] == "whisper_small15_frozen"
        ]
        write_csv(root / "diagnostics" / name, values)
    monkeypatch.chdir(tmp_path)
    bundle = Path("artifacts/deployment/whisper_large_v3_15_frozen")
    bundle.mkdir(parents=True)
    write_json(
        bundle / "verification.json", {"n": 14400, "prediction_disagreements": 0}
    )
    archive = bundle.with_suffix(".tar.gz")
    archive.write_bytes(b"fixture")
    Path(str(archive) + ".sha256").write_text(file_hash(archive))

    def fake_pdf(source, destination):
        destination.write_bytes(b"PDF fixture")
        return "Whisper Large v3 Complete plot appendix " + " ".join(
            str(p)
            for f in [project / "artifacts/whisper_tiny15", small]
            for p in (f / "frozen/figures").glob("*.png")
        )

    monkeypatch.setattr(module, "render_pdf", fake_pdf)
    module.build(
        SimpleNamespace(
            root=root,
            comparison_models=LargeStudy.comparison_models,
            pacific=project / "artifacts/extension15",
            reference=project / "artifacts/whisper_tiny15",
            small_reference=small,
            base=project / "artifacts/whisper_small12",
            rows=lambda c: [None] * 1336,
        )
    )
    report = (
        tmp_path / "reports/large_fixture/whisper_large_v3_15_analysis.html"
    ).read_text()
    assert "Directional confusions" in report and "128 mel channels" in report
    assert (root / "report_complete.json").exists()


def test_cached_training_resume_matches_uninterrupted_weights(tmp_path, monkeypatch):
    """Exercise the real extension checkpoint loop on CPU, with no audio encoder."""
    from contextlib import nullcontext

    import numpy as np

    import lid.whisper_extension as module
    from lid.whisper_large import CachedHeadAdapter

    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0] == "cuda":
            args = ("cpu", *args[1:])
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(
        torch.amp,
        "GradScaler",
        lambda *a, **k: SimpleNamespace(
            scale=lambda loss: loss, state_dict=dict, load_state_dict=lambda state: None
        ),
    )

    def step(scaler, optimizer):
        optimizer.step()
        return True

    monkeypatch.setattr(module, "scaled_optimizer_step", step)
    torch.manual_seed(42)
    initial = expanded_head(whisper_head(8).state_dict()).state_dict()
    features = np.random.default_rng(42).standard_normal((15, 16)).astype("float32")
    rows = [{"id": str(i), "language": code} for i, code in enumerate(CODES)]

    def make(name):
        study = object.__new__(module.ExtensionStudy)
        study.root = tmp_path / name
        study.root.mkdir()
        study.config = {
            "seed": 42,
            "model": "large-v3",
            "head_lr": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 2,
            "patience": 5,
            "examples_per_epoch": 2048,
        }
        study.adapter_class = CachedHeadAdapter
        study.rows = lambda cohort: (
            rows[:12] if cohort.startswith("vox_") else rows[12:]
        )
        write_json(study.root / "data_ready.json", {})
        write_json(
            study.root / "model_identity.json",
            {"dims": {"n_audio_state": 8}, "encoder_checkpoint_sha256": "fixed"},
        )
        torch.save({"head": initial}, study.root / "initial.pt")
        for cohort, values in [
            ("vox_train", features[:12]),
            ("new_train", features[12:]),
            ("vox_validation", features[:12]),
            ("new_selection", features[12:]),
        ]:
            path = study.root / "cache" / cohort
            path.mkdir(parents=True)
            np.save(path / "features.npy", values)
        study.status = lambda *a, **k: None
        return study

    control = make("control")
    control.train("frozen")
    resumed = make("resumed")

    def interrupted(phase, **details):
        if phase == "train" and details["completed"] == 1024:
            raise RuntimeError("simulated interruption")

    resumed.status = interrupted
    with pytest.raises(RuntimeError, match="simulated interruption"):
        resumed.train("frozen")
    state = torch.load(resumed.root / "frozen/resume.pt", weights_only=False)
    assert state["cursor"] == 1024 and not state["encoder"]
    resumed.status = lambda *a, **k: None
    resumed.train("frozen")
    expected = torch.load(control.root / "frozen/model.pt", weights_only=True)
    actual = torch.load(resumed.root / "frozen/model.pt", weights_only=True)
    for name, value in expected["head"].items():
        torch.testing.assert_close(actual["head"][name], value, atol=0, rtol=0)
    assert json.loads((resumed.root / "frozen/history.json").read_text()) == json.loads(
        (control.root / "frozen/history.json").read_text()
    )
