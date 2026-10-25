import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import softmax

from lid.extension_data import file_hash, write_json
from lid.languages import CODES
from lid.whisper_model import (
    native_logits,
    pool_real_tokens,
    selected_languages,
    whisper_head,
)
from lid.whisper_study import Study, balanced_order


def test_pool_excludes_padding_and_has_correct_gradients():
    values = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0], [300.0, 400.0]]], requires_grad=True
    )
    pooled = pool_real_tokens(values, torch.tensor([640]))
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0, 1.0, 1.0]]))
    pooled.sum().backward()
    assert torch.count_nonzero(values.grad[:, :2]) > 0
    assert torch.count_nonzero(values.grad[:, 2:]) == 0


def test_native_token_extraction_discards_nonlanguage_tokens(monkeypatch):
    import lid.whisper_model as module

    codes = [*CODES, "de"]
    tokens = list(range(2, 2 + len(codes)))
    tokenizer = SimpleNamespace(
        sot=0, all_language_tokens=tokens, all_language_codes=codes
    )
    monkeypatch.setattr(module, "tokenizer_for", lambda model: (tokenizer, codes))
    raw = torch.arange(2 * 20).reshape(2, 1, 20).float()
    raw[:, :, 1] = 10000

    def decoder(sot, encoded):
        assert sot.tolist() == [[0], [0]]
        return raw

    result, result_codes = native_logits(
        SimpleNamespace(decoder=decoder), torch.zeros(2, 1500, 384)
    )
    assert result_codes == codes
    torch.testing.assert_close(result, raw[:, 0, tokens])
    assert result.max() < 10000


def test_filtering_matches_probability_renormalisation_and_language_order():
    codes = list(reversed([*CODES, "de", "fr"]))
    indices = selected_languages(codes)
    assert [codes[i] for i in indices] == CODES
    logits = np.random.default_rng(42).normal(size=(4, len(codes)))
    full = softmax(logits, axis=1)[:, indices]
    np.testing.assert_allclose(
        softmax(logits[:, indices], axis=1), full / full.sum(axis=1, keepdims=True)
    )
    with pytest.raises(ValueError, match="missing"):
        selected_languages(["en", "hi"])


def test_paired_balanced_draws_are_deterministic_and_include_all_classes():
    targets = torch.arange(12).repeat(3)
    order = balanced_order(targets, 2, 4096)
    assert torch.equal(order, balanced_order(targets, 2, 4096))
    assert set(targets[order].tolist()) == set(range(12))
    assert not torch.equal(order, balanced_order(targets, 3, 4096))


def test_whisper_head_outputs_twelve_logits_and_backpropagates():
    head = whisper_head()
    inputs = torch.randn(3, 768, requires_grad=True)
    scores = head(inputs)
    assert scores.shape == (3, 12)
    scores.sum().backward()
    assert torch.isfinite(inputs.grad).all()


def test_multilingual_model_required_and_changed_configuration_rejected(tmp_path):
    config = tmp_path / "config.json"
    original = {
        "model": "tiny",
        "max_seconds": 10,
        "whisper_window_seconds": 30,
        "seed": 42,
    }
    write_json(config, original)
    study = Study(config, tmp_path / "run")
    write_json(config, dict(original, seed=43))
    with pytest.raises(ValueError, match="Cache identity changed"):
        Study(config, study.root)
    write_json(config, dict(original, model="tiny.en"))
    with pytest.raises(ValueError, match="multilingual tiny"):
        Study(config, tmp_path / "bad")


def test_cohort_snapshot_mutation_is_rejected(tmp_path):
    config = tmp_path / "config.json"
    write_json(
        config, {"model": "tiny", "max_seconds": 10, "whisper_window_seconds": 30}
    )
    study = Study(config, tmp_path / "run")
    (study.root / "cohorts").mkdir()
    path = study.root / "cohorts/test.jsonl"
    path.write_text(json.dumps({"id": "a"}) + "\n")
    write_json(
        study.root / "data_ready.json",
        {"cohorts": {"test": {"sha256": file_hash(path)}}},
    )
    assert study.rows("test") == [{"id": "a"}]
    path.write_text(json.dumps({"id": "b"}) + "\n")
    with pytest.raises(ValueError, match="Cohort manifest changed"):
        study.rows("test")


def test_language_columns_stable_across_python_hash_seeds():
    import os
    import subprocess
    import sys

    code = "import json; from types import SimpleNamespace; from lid.whisper_model import tokenizer_for; print(json.dumps(tokenizer_for(SimpleNamespace(num_languages=99))[1]))"
    results = []
    for seed in ["1", "2"]:
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        results.append(json.loads(result.stdout))
    assert results[0] == results[1] == sorted(results[0])
    assert set(CODES).issubset(results[0])


@pytest.fixture
def diagnostic_study(tmp_path):
    import soundfile as sf

    root = tmp_path / "study"
    root.mkdir()
    write_json(root / "data_ready.json", {"test": "synthetic diagnostic fixture"})
    external = tmp_path / "external"
    (external / "decoded").mkdir(parents=True)
    signal = np.linspace(-0.8, 0.8, 160000, dtype=np.float32)
    summaries = []
    for language in ["hi", "sam"]:
        source = f"{language}-source"
        sf.write(external / "decoded" / f"{source}.wav", signal, 16000, subtype="FLOAT")
        write_json(
            external / f"{source}.json",
            {"vad_intervals_samples": [{"start": 0, "end": 160000}]},
        )
        summaries.append({"source_id": source, "language": language})
    write_json(external / "segments_ready.json", {"source_summaries": summaries})
    selection = tmp_path / "selection.wav"
    sf.write(selection, signal, 16000, subtype="FLOAT")
    rows = [
        {
            "id": f"{code}:selection",
            "path": str(selection),
            "group": f"{code}:rec",
            "language": code,
        }
        for code in CODES
    ]
    study = SimpleNamespace(
        root=root,
        config={"seed": 42, "external": str(external)},
        rows=lambda name: rows,
        status=lambda *a, **kw: None,
    )
    return study, signal


def test_diagnostic_crops_are_nested_real_audio_and_positive_gain_is_capped(
    diagnostic_study,
):
    import soundfile as sf

    from lid.whisper_diagnostics import prepare_rows

    study, signal = diagnostic_study
    directory = study.root / "diagnostics"
    directory.mkdir()
    rows = prepare_rows(study, directory)
    for row in [r for r in rows if r["probe"] == "duration" and r["language"] == "hi"]:
        actual, rate = sf.read(row["path"], dtype="float32")
        length = row["seconds"] * 16000
        left = (160000 - length) // 2
        np.testing.assert_array_equal(actual, signal[left : left + length])
        assert rate == 16000
    for row in [r for r in rows if r["probe"] == "gain" and r["gain_db"] == 12]:
        actual, _ = sf.read(row["path"], dtype="float32")
        assert np.abs(actual).max() <= 0.999001
        assert row["gain_db_applied"] < 12


def test_diagnostic_summary_does_not_score_pacific_as_valid_twelve_way_accuracy(
    diagnostic_study, monkeypatch, tmp_path
):
    import csv

    import lid.whisper_diagnostics as module

    study, _ = diagnostic_study
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "artifacts/full_run/comparison/calibration.json"
    path.parent.mkdir(parents=True)
    write_json(path, {"filtered_temperature": 1.0, "head_temperature": 1.0})
    for method in module.METHODS:
        folder = study.root / method
        folder.mkdir()
        write_json(folder / "calibration.json", {"temperature": 1.0})

    def score(study, method, rows, path):
        result = np.zeros((len(rows), 12), dtype=np.float32)
        for i, row in enumerate(rows):
            result[
                i, CODES.index(row["language"]) if row["language"] in CODES else 0
            ] = 4
        return result

    monkeypatch.setattr(module, "score_method", score)
    module.run(study)
    with (study.root / "diagnostics/duration.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(
        r["accuracy_all"] == "" and r["in_set"] == "False"
        for r in rows
        if r["language"] == "sam"
    )
    assert all(float(r["accuracy_all"]) == 1 for r in rows if r["language"] == "hi")
    assert (study.root / "diagnostics/complete.json").exists()
