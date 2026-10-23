"""Checks for leakage prevention, feasibility, and sampling behavior."""

import json
from collections import Counter

import numpy as np
import torch

from lid.extension_train import epoch_order
from lid.speaker_study import (
    LANGUAGES,
    SPLITS,
    allocate,
    connected_cut_groups,
    speech_windows,
)


def test_windows_do_not_join_distant_speech_or_keep_short_regions():
    windows = speech_windows(
        [{"start": 0, "end": 47000}, {"start": 80000, "end": 190000}]
    )
    assert windows == [(87000, 135000), (135000, 183000)]
    assert all(end - start == 48000 for start, end in windows)


def test_speaker_recording_and_duplicate_links_are_transitive():
    rows = [
        {"group": "a", "sha256": "1"},
        {"group": "b", "sha256": "2"},
        {"group": "b", "sha256": "3"},
        {"group": "c", "sha256": "3"},
        {"group": "d", "sha256": "4"},
    ]
    groups = connected_cut_groups(rows, [{10}, {10, 20}, {30}, {40}, set()])
    assert len(set(groups[:4])) == 1
    assert groups[4] != groups[0]


def test_allocation_rejects_language_with_too_few_groups():
    rows = [{"language": code} for code in LANGUAGES for _ in range(200)]
    groups = [code for code in LANGUAGES for _ in range(200)]
    assigned, gate = allocate(rows, groups)
    assert not assigned and not gate["feasible"]
    assert all(n == 1 for n in gate["groups_per_language"].values())


def test_allocation_is_deterministic_whole_group_and_covers_languages():
    rows, groups = [], []
    for code in LANGUAGES:
        for group in range(12):
            rows.extend({"language": code} for _ in range(50))
            groups.extend(f"{code}:{group}" for _ in range(50))
    assignment, gate = allocate(rows, groups)
    assert gate["feasible"]
    assert allocate(rows, groups) == (assignment, gate)
    assert all(
        set(counts) == set(LANGUAGES) and min(counts.values()) >= 50
        for counts in gate["counts"].values()
    )
    assert set(assignment.values()) == set(SPLITS)


def test_speaker_balancing_changes_speaker_exposure_without_changing_class_draws():
    targets = torch.tensor([*range(12), *([12] * 101), 13, 14])
    speakers = ["old"] * 12 + ["prolific"] * 100 + ["rare", "other", "another"]
    uniform = epoch_order(targets, 1, 32768)
    balanced = epoch_order(targets, 1, 32768, speakers)
    assert torch.equal(targets[uniform], targets[balanced])
    assert torch.equal(balanced, epoch_order(targets, 1, 32768, speakers))
    selected = balanced[targets[balanced] == 12].tolist()
    counts = Counter(speakers[i] for i in selected)
    assert 0.45 < counts["rare"] / len(selected) < 0.55
    assert (
        np.mean([speakers[i] == "rare" for i in uniform[targets[uniform] == 12]]) < 0.03
    )
    assert int(balanced.min()) >= 0 and int(balanced.max()) < len(targets)


def test_run_preparation_aligns_reused_embeddings_to_new_manifests(
    tmp_path, monkeypatch
):
    from lid import speaker_experiments as experiments

    original, root = tmp_path / "original", tmp_path / "study"
    original.mkdir()
    root.mkdir()
    monkeypatch.setattr(experiments, "ORIGINAL", original)
    monkeypatch.setattr(experiments, "ROOT", root)
    source_rows = [
        {"id": f"cut{i}", "language": "sam", "speaker_id": str(i)} for i in range(4)
    ]
    (root / "inventory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows)
    )
    values = torch.arange(4).float().unsqueeze(1).expand(-1, 256).clone()
    torch.save(
        {"embeddings": values, "identity": {}}, root / "frozen_lid_embeddings.pt"
    )
    for i, split in enumerate(SPLITS):
        (root / f"new_{split}.jsonl").write_text(json.dumps(source_rows[3 - i]) + "\n")
        torch.save({"identity": {"encoder": "fixed"}}, original / f"new_{split}.pt")
    (root / "data_ready.json").write_text(json.dumps({"feasible": True}))
    run = experiments.prepare_run("disjoint_uniform")
    for i, split in enumerate(SPLITS):
        saved = torch.load(run / f"new_{split}.pt", weights_only=True)
        assert torch.equal(saved["embeddings"], values[3 - i : 4 - i])
        assert saved["identity"]["encoder"] == "fixed"
    # Resuming does not reorder the cached rows.
    assert experiments.prepare_run("disjoint_uniform") == run


def test_failed_feasibility_gate_skips_training(tmp_path, monkeypatch):
    from lid import speaker_study
    from scripts.run_speaker_study import main

    monkeypatch.setattr(speaker_study, "ROOT", tmp_path)
    monkeypatch.setenv("LID_MEMORY_GUARDED", "1")
    for name in ("HF_HUB_OFFLINE", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr("sys.argv", ["run_speaker_study.py", "--stage", "train"])
    (tmp_path / "split_gate.json").write_text(
        json.dumps({"feasible": False, "reason": "too few groups"})
    )
    main()
    assert (
        json.loads((tmp_path / "status.json").read_text())["phase"]
        == "retraining_not_feasible"
    )
    assert not (tmp_path / "runs").exists()


def test_speaker_associations_exclude_unknown_multiple_speaker_and_noise_points(
    tmp_path, monkeypatch
):
    import csv

    from lid import speaker_visualize
    from lid.speaker_study import write_csv

    monkeypatch.setattr(speaker_visualize, "ROOT", tmp_path)
    points = [
        {
            "speaker": speaker,
            "domain": domain,
            "uncertain_speaker": uncertain,
            "language": "sam",
            "recording": str(i),
        }
        for i, (speaker, domain, uncertain) in enumerate(
            [
                ("a", "radio", "False"),
                ("b", "radio", "False"),
                ("unknown", "radio", "True"),
                ("external_unknown", "external", "True"),
                ("a", "radio", "True"),
                ("c", "radio", "False"),
            ]
        )
    ]
    write_csv(tmp_path / "frozen_points.csv", points)
    for minimum in (25, 50, 100):
        np.save(tmp_path / f"frozen_clusters_{minimum}.npy", [0, 1, 0, 1, 0, -1])
    (tmp_path / "frozen_geometry_complete.json").write_text(json.dumps({}))
    speaker_visualize.refresh_associations()
    with (tmp_path / "frozen_cluster_associations.csv").open() as stream:
        values = list(csv.DictReader(stream))
    assert all(
        int(row["n"]) == 2 and float(row["nmi"]) == 1
        for row in values
        if row["attribute"] == "speaker"
    )
    assert all(row["nmi"] == "" for row in values if row["attribute"] == "language")
