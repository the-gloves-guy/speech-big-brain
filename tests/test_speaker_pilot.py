import json

import numpy as np
import pytest

from lid.speaker_pilot import (
    CROPS,
    Pilot,
    bind,
    coverage,
    crop_bounds,
    hierarchical,
    ordered,
)


def test_sampling_order_is_deterministic_and_input_order_independent():
    assert ordered(["a", "b", "c"], 42) == ordered(["c", "a", "b"], 42)
    assert set(ordered(["a", "b", "c"], 42)) == {"a", "b", "c"}


def test_cache_rejects_changed_configuration(tmp_path):
    path = tmp_path / "identity.json"
    bind(path, {"seed": 42})
    bind(path, {"seed": 42})
    with pytest.raises(ValueError, match="Cache identity changed"):
        bind(path, {"seed": 43})


def test_nested_crops_no_padding_or_repeated_audio():
    with pytest.raises(ValueError, match="short_audio"):
        crop_bounds(159999)
    bounds = crop_bounds(176000)
    signal = np.arange(176000)
    for key, (a, b) in bounds.items():
        assert b - a == CROPS[key][1] - CROPS[key][0]
        assert len(np.unique(signal[a:b])) == b - a
    assert bounds["early3"][1] <= bounds["late3"][0]
    assert bounds["10"][0] <= bounds["5"][0] <= bounds["3"][0]


def test_vad_coverage_keeps_pauses_and_rejects_insufficient_speech():
    intervals = [{"start": 0, "end": 40000}, {"start": 80000, "end": 160000}]
    assert coverage(intervals, 0, 160000) == 0.75
    assert coverage(intervals, 40000, 120000) == 0.5


@pytest.mark.parametrize("method", ["average", "complete"])
def test_linkage_does_not_chain_dissimilar_endpoints(method):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    angles = np.deg2rad([0, 40, 80])
    values = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    distances = pdist(values, metric="cosine")
    assert distances[0] < 0.3 and distances[2] < 0.3 and distances[1] > 0.8
    labels = fcluster(linkage(distances, method=method), 0.3, criterion="distance")
    assert labels[0] != labels[2]
    assert len(set(hierarchical(values, method, 2))) == 2


def test_audio_mutation_rejected(tmp_path):
    from lid.extension_data import file_hash, write_json

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"sample_rate": 16000, "durations": [3, 5, 10], "microbatch": 2})
    )
    pilot = Pilot(config, tmp_path / "pilot")
    audio = tmp_path / "audio.npy"
    np.save(audio, np.zeros(160000, dtype=np.float32))
    write_json(
        pilot.root / "sample.json",
        [{"audio_cache": str(audio), "audio_sha256": file_hash(audio)}],
    )
    write_json(
        pilot.root / "sample_ready.json",
        {"sample_sha256": file_hash(pilot.root / "sample.json")},
    )
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Cached audio changed"):
        pilot.load_sample()


@pytest.mark.parametrize("mismatch", [False, True])
def test_acceptance_checks_source_crops_and_matched_encoder_ids(tmp_path, mismatch):
    import soundfile as sf

    from lid.data import waveform
    from lid.extension_data import file_hash, write_json
    from lid.speaker_pilot import digest

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "sample_rate": 16000,
                "durations": [3, 5, 10],
                "microbatch": 2,
                "languages": ["sam"],
                "cuts_per_recording": 2,
                "recordings_per_language": 1,
                "minimum_speech_coverage": 0.7,
            }
        )
    )
    pilot = Pilot(config, tmp_path / "pilot")
    rows = []
    for i in range(2):
        source = tmp_path / f"{i}.wav"
        sf.write(source, np.full(160000, 0.1 * i, dtype=np.float32), 16000)
        row = {
            "id": str(i),
            "path": str(source),
            "language": "sam",
            "group": "recording",
        }
        audio = tmp_path / f"{i}.npy"
        np.save(audio, waveform(row, 10))
        rows.append(
            dict(
                row,
                audio_cache=str(audio),
                audio_sha256=file_hash(audio),
                crop_bounds={k: list(v) for k, v in CROPS.items()},
                speech_coverage={k: 1 for k in CROPS},
            )
        )
    write_json(pilot.root / "sample.json", rows)
    identity = {"sample": file_hash(pilot.root / "sample.json")}
    write_json(pilot.root / "sample_ready.json", {"sample_sha256": identity["sample"]})
    for name in ["ecapa", "wespeaker"]:
        folder = pilot.root / name
        folder.mkdir()
        write_json(folder / "identity.json", identity)
        for key in CROPS:
            ids = (
                ["wrong", "1"]
                if mismatch and name == "wespeaker" and key == "10"
                else ["0", "1"]
            )
            np.savez(
                folder / f"{key}.npz",
                embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
                ids=ids,
                identity=digest(identity),
            )
            write_json(
                folder / f"{key}_reproducibility.json", {"max_absolute_difference": 0.0}
            )
    if mismatch:
        with pytest.raises(ValueError, match="Mismatched encoder cache"):
            pilot.verify()
    else:
        pilot.verify()
        assert (
            json.loads((pilot.root / "acceptance.json").read_text())[
                "unique_decoded_cuts"
            ]
            == 2
        )
