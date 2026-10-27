import json

import numpy as np
import pytest
import torch

from lid.extension_data import CODES, write_json
from lid.extension_train import epoch_order
from lid.whisper_extension import ExtensionStudy, audit_partitions, expanded_head
from lid.whisper_extension_results import grouped_metrics
from lid.whisper_model import whisper_head


def test_expansion_copies_hidden_layers_and_original_rows_exactly():
    old = whisper_head()
    with torch.no_grad():
        for p in old.parameters():
            p.normal_()
    first = expanded_head(old.state_dict())
    torch.manual_seed(123)
    second = expanded_head(old.state_dict())
    for key, value in old.state_dict().items():
        actual = first.state_dict()[key]
        torch.testing.assert_close(
            actual[:12] if key.startswith("3.") else actual, value, atol=0, rtol=0
        )
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.equal(
        first[3].weight[12:], expanded_head(old.state_dict(), seed=43)[3].weight[12:]
    )
    assert first(torch.randn(2, 768)).shape == (2, 15)


def test_original_logits_preserved_with_new_outputs_enabled():
    old = whisper_head().eval()
    new = expanded_head(old.state_dict()).eval()
    x = torch.randn(4, 768)
    torch.testing.assert_close(new(x)[:, :12], old(x))
    assert new(x).softmax(-1)[:, :12].sum(-1).max() < 1


def test_shared_sampler_resumes_by_cursor_without_changing_draws():
    targets = torch.arange(15).repeat_interleave(7)
    first = epoch_order(targets, 1)
    cursor = 1024
    resumed = epoch_order(targets, 1)[cursor:]
    assert torch.equal(torch.cat([first[:cursor], resumed]), first)
    assert set(targets[first].tolist()) == set(range(15))
    assert not torch.equal(first, epoch_order(targets, 2))


def partitions():
    return {
        split: [
            {
                "id": f"{split}:{c}",
                "group": f"{split}:{c}",
                "sha256": f"{split}:{c}",
                "language": c,
            }
            for c in CODES[12:]
        ]
        for split in ["train", "selection", "calibration", "test"]
    }


@pytest.mark.parametrize("key", ["id", "group", "sha256"])
def test_partition_overlap_rejected(key):
    data = partitions()
    audit_partitions(data)
    data["test"][0][key] = data["calibration"][0][key]
    with pytest.raises(ValueError, match="overlap"):
        audit_partitions(data)


def test_grouped_metrics_do_not_renormalize_away_new_language_errors():
    logits = np.zeros((15, 15))
    logits[np.arange(15), np.arange(15)] = 5
    logits[0, 14] = 10
    rows = [{"language": code} for code in CODES]
    result = grouped_metrics(logits, rows, CODES, 1.0)
    assert result["old12"]["accuracy"] == pytest.approx(11 / 12)
    assert result["new3"]["accuracy"] == 1
    assert len(result["old12"]["confusion_matrix"][0]) == 15
    assert result["old12"]["outside_set_rate"] == pytest.approx(1 / 12)


def test_changed_configuration_rejects_stale_study(tmp_path):
    config = {
        "model": "tiny",
        "max_seconds": 10,
        "whisper_window_seconds": 30,
        "base": "old",
        "pacific": "original",
        "seed": 42,
    }
    path = tmp_path / "config.json"
    write_json(path, config)
    ExtensionStudy(path, tmp_path / "run")
    write_json(path, dict(config, seed=43))
    with pytest.raises(ValueError, match="identity changed"):
        ExtensionStudy(path, tmp_path / "run")


def test_fifteen_way_export_folds_temperature_once():
    from lid.whisper_export_backend import WhisperTinyLIDClassifier

    original = expanded_head(whisper_head().state_dict()).eval()
    state = {k: v.clone() for k, v in original.state_dict().items()}
    state["3.weight"] /= 2.1
    state["3.bias"] /= 2.1
    exported = WhisperTinyLIDClassifier(outputs=15).eval()
    exported.load_state_dict(state)
    x = torch.randn(3, 768)
    torch.testing.assert_close(exported(x), (original(x) / 2.1).log_softmax(-1))


def test_interruption_restores_optimizer_and_rng_exactly(tmp_path):
    from types import SimpleNamespace

    from lid.whisper_extension import restore_resume
    from lid.whisper_study import save_torch

    def setup():
        adapter = SimpleNamespace(
            encoder=torch.nn.Linear(3, 4), head=torch.nn.Linear(4, 15)
        )
        optimizer = torch.optim.AdamW(
            [*adapter.encoder.parameters(), *adapter.head.parameters()], lr=0.001
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return adapter, optimizer, scaler

    def step(adapter, optimizer):
        x = torch.randn(5, 3)
        optimizer.zero_grad()
        adapter.head(adapter.encoder(x)).square().sum().backward()
        optimizer.step()

    torch.manual_seed(42)
    adapter, optimizer, scaler = setup()
    step(adapter, optimizer)
    path = tmp_path / "resume.pt"
    identity = {"data": "original", "seed": 42}
    save_torch(
        path,
        {
            "identity": identity,
            "encoder": adapter.encoder.state_dict(),
            "head": adapter.head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": [],
            "epoch": 1,
            "cursor": 32,
            "stale": 0,
            "best": (0.5, -0.6),
            "history": [],
        },
    )
    step(adapter, optimizer)
    resumed, opt2, scaler2 = setup()
    assert restore_resume(path, identity, resumed, opt2, scaler2)[:2] == [1, 32]
    step(resumed, opt2)
    for first, second in [
        (adapter.encoder, resumed.encoder),
        (adapter.head, resumed.head),
    ]:
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="configuration/data changed"):
        restore_resume(path, {"data": "changed"}, resumed, opt2, scaler2)


def test_matched_external_tables_and_english_shared_cohort(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import lid.whisper_extension_results as module

    output = tmp_path / "external"
    output.mkdir()
    root = tmp_path / "study"
    root.mkdir()
    rows = [
        {
            "id": f"id{i}",
            "language": code,
            "source_id": f"source{i}",
            "source": f"source{i}.wav",
            "start_seconds": 0,
            "end_seconds": 5,
            "duration": 5,
        }
        for i, code in enumerate(["hi", "niu", "ton"])
    ]
    write_json(
        root / "external_inventory.json",
        {"historical_segment_ids": ["id0"], "added_sources": []},
    )
    models = ["frozen15", "partial15", "original107", "filtered12", "previous12"]
    logits = np.zeros((3, 15), dtype=np.float32)
    logits[np.arange(3), [4, 13, 14]] = 5
    modified = logits.copy()
    modified[0, 1] = 10
    np.savez(
        output / "logits.npz",
        ids=[r["id"] for r in rows],
        **{m: modified if m == "partial15" else logits for m in models},
    )
    write_json(
        output / "model_provenance.json",
        {m: {"codes": CODES, "temperature": 1.0} for m in models},
    )
    for method in [
        "whisper_native",
        "whisper_filtered",
        "whisper_frozen",
        "whisper_finetuned",
        "speechbrain_finetuned",
        "frozen",
        "finetuned",
    ]:
        folder = (
            root / method
            if method in ["frozen", "finetuned"]
            else root / "baselines" / method
        )
        folder.mkdir(parents=True)
        codes = CODES if method in ["frozen", "finetuned"] else CODES[:12]
        np.savez(
            folder / "external.npz",
            ids=[r["id"] for r in rows],
            logits=logits[:, : len(codes)],
            codes=codes,
        )
        write_json(folder / "calibration.json", {"temperature": 1.0})
    monkeypatch.setattr(module, "plots", lambda *args: None)
    study = SimpleNamespace(
        root=root, config={"external": str(output)}, rows=lambda name: rows
    )
    summary = module.external_results(study)
    shared = next(
        r
        for r in summary
        if r["model"] == "whisper15_frozen"
        and r["cohort"] == "complete"
        and r["policy"] == "shared"
    )
    assert shared["segments_scored"] == 2
    assert shared["coverage"] == pytest.approx(2 / 3)
    assert (output / "matched_per_language.csv").exists()
    report = json.loads((output / "matched_report.json").read_text())
    assert report["models"]["whisper_frozen"]["complete"]["all"][
        "out_of_set_languages"
    ] == ["niu", "ton"]
    # Reordered IDs must never silently pair different segments.
    folder = root / "frozen"
    np.savez(
        folder / "external.npz", ids=["id1", "id0", "id2"], logits=logits, codes=CODES
    )
    with pytest.raises(ValueError, match="segment IDs differ"):
        module.external_results(study)


def test_relocated_export_loads_offline_and_handles_short_variable_audio(
    tmp_path, monkeypatch
):
    import shutil
    import socket
    from types import SimpleNamespace

    from speechbrain.inference.classifiers import EncoderClassifier

    from lid.extension_data import file_hash
    from lid.whisper_export_backend import WhisperTinyLIDEncoder
    from lid.whisper_extension_export import export_one

    root = tmp_path / "study"
    source = root / "frozen"
    source.mkdir(parents=True)
    encoder = WhisperTinyLIDEncoder()
    head = expanded_head(whisper_head().state_dict())
    torch.save(
        {
            "encoder": encoder.encoder.state_dict(),
            "head": head.state_dict(),
            "epoch": 1,
        },
        source / "model.pt",
    )
    write_json(source / "evaluation_complete.json", {"synthetic_fixture": True})
    write_json(
        source / "calibration.json",
        {
            "codes": CODES,
            "temperature": 2.0,
            "model_sha256": file_hash(source / "model.pt"),
        },
    )
    output = tmp_path / "export"
    export_one(SimpleNamespace(root=root), "frozen", output)
    relocated = tmp_path / "relocated"
    shutil.move(output, relocated)

    def denied(*args, **kwargs):
        raise AssertionError("Export attempted network access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = EncoderClassifier.from_hparams(
            source=str(relocated),
            savedir=str(tmp_path / "loaded"),
            run_opts={"device": "cpu"},
        )
        signals = torch.randn(2, 8000) * 0.01
        lengths = torch.tensor([1.0, 0.2])
        with torch.inference_mode():
            actual, _, _, labels = model.classify_batch(signals, lengths)
            expected = (head(encoder(signals, lengths).squeeze(1)) / 2).log_softmax(-1)
        torch.testing.assert_close(actual.reshape(2, 15), expected)
        assert torch.isfinite(actual).all()
        assert len(labels) == 2
        with pytest.raises(ValueError, match="0.1 seconds"):
            model.classify_batch(torch.zeros(1, 1599))
    finally:
        torch.set_num_threads(threads)


def test_pcm_fingerprint_ignores_wav_timestamp_but_detects_changed_audio(tmp_path):
    import soundfile as sf

    from lid.extension_data import file_hash
    from lid.whisper_extension import audio_fingerprint

    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    signal = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    sf.write(first, signal, 16000, subtype="FLOAT")
    raw = bytearray(first.read_bytes())
    peak = raw.index(b"PEAK")
    raw[peak + 12 : peak + 16] = (1234).to_bytes(4, "little")
    second.write_bytes(raw)
    assert file_hash(first) != file_hash(second)
    assert audio_fingerprint(first) == audio_fingerprint(second)
    sf.write(second, signal * 0.5, 16000, subtype="FLOAT")
    assert audio_fingerprint(first) != audio_fingerprint(second)


def test_final_reports_include_regression_and_speechbrain_comparison(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import lid.whisper_extension_results as module
    from lid.languages import NAMES as OLD_NAMES
    from lid.youtube_evaluate import write_csv

    monkeypatch.chdir(tmp_path)
    root, base, pacific = [tmp_path / name for name in ["study", "base", "pacific"]]
    logits = np.eye(15) * 5
    results = grouped_metrics(logits, [{"language": c} for c in CODES], CODES, 1.0)
    for method, control in [
        ("frozen", "whisper_frozen"),
        ("finetuned", "whisper_finetuned"),
    ]:
        folder = root / method
        folder.mkdir(parents=True)
        variants = {v: results for v in ["raw", "joint", "old_only"]}
        write_json(
            folder / "benchmarks.json",
            {"fleurs_test": variants, "combined_test": variants},
        )
        write_csv(
            folder / "summary.csv",
            [
                {
                    "method": method,
                    "cohort": "combined_test",
                    "variant": "joint",
                    "group": "all15",
                    **{
                        k: results["all15"][k]
                        for k in ["n", "accuracy", "macro_f1", "nll", "brier", "ece"]
                    },
                }
            ],
        )
        previous = json.loads(json.dumps(results["old12"]))
        previous["per_language"] = {
            name: previous["per_language"][c] for c, name in zip(CODES[:12], OLD_NAMES)
        }
        write_json(
            base / control / "fleurs_report.json", {"metrics": {"calibrated": previous}}
        )
    for method in ["frozen", "partial"]:
        write_json(
            pacific / method / "report.json",
            {
                "groups": {
                    g: {"raw": v, "joint_temperature": v} for g, v in results.items()
                }
            },
        )
    diagnostic = root / "diagnostics"
    diagnostic.mkdir()
    (diagnostic / "gain.csv").write_text("model,language\n")
    (diagnostic / "duration.csv").write_text("model,language\n")
    write_json(root / "external_inventory.json", {"added_sources": []})
    monkeypatch.setattr(module, "external_results", lambda study: [])
    module.report(SimpleNamespace(root=root, base=base, pacific=pacific))
    text = (tmp_path / "reports/whisper_tiny15/whisper_extension.md").read_text()
    assert "SpeechBrain15 test comparison" in text
    assert "Samoan volume sensitivity" in text
    assert (root / "original_language_per_class_regression.csv").exists()
    assert (root / "report_complete.json").exists()
