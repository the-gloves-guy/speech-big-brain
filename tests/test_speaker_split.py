import numpy as np
import pytest

from lid.speaker_split import allocate_speakers, audit_disjoint, classify


def row(i, speaker, language="sam"):
    return {
        "id": str(i),
        "group": f"rec:{i}",
        "sha256": f"hash:{i}",
        "speaker_id": speaker,
        "language": language,
    }


@pytest.mark.parametrize("key", ["id", "group", "sha256", "speaker_id"])
def test_leakage_audit_rejects_each_shared_identifier(key):
    a, b = row(0, "a"), row(1, "b")
    b[key] = a[key]
    with pytest.raises(ValueError, match=f"{key} leakage"):
        audit_disjoint({"train": [a], "test": [b]})


def test_whole_speaker_allocation_is_deterministic_and_covers_languages():
    rows = [
        row(f"{language}:{speaker}:{i}", f"{language}:{speaker}", language)
        for language in ["sam", "niu", "ton"]
        for speaker in range(7)
        for i in range(3)
    ]
    assigned = allocate_speakers(rows, minimum=1)
    assert assigned == allocate_speakers(list(reversed(rows)), minimum=1)
    partitions = {
        s: [r for r in rows if assigned[r["speaker_id"]] == s]
        for s in ["train", "selection", "calibration", "test"]
    }
    audit_disjoint(partitions)
    assert all(
        {r["language"] for r in part} == {"sam", "niu", "ton"}
        for part in partitions.values()
    )


def test_allocation_rejects_too_few_speakers():
    with pytest.raises(ValueError, match="Cannot form four"):
        allocate_speakers([row(i, f"speaker:{i}") for i in range(3)], minimum=1)


def test_reference_classification_reports_ambiguity():
    references = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array(["a", "a", "b"])
    predicted, similarity, margin = classify(
        np.array([[1.0, 0.0], [0.7, 0.7]]), references, labels
    )
    assert predicted[0] == "a" and similarity[0] == 1 and margin[0] == 1
    assert margin[1] == 0


def test_allocation_keeps_large_speakers_in_training_when_small_holdouts_suffice():
    rows = [
        row(f"{speaker}:{i}", str(speaker))
        for speaker, size in enumerate([60, 20, 5, 5, 5, 5])
        for i in range(size)
    ]
    assignment = allocate_speakers(rows, minimum=1)
    assert assignment["0"] == assignment["1"] == "train"
    assert sum(assignment[r["speaker_id"]] == "train" for r in rows) >= 80
