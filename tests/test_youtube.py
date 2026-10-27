"""Segmentation and exclusions must not inflate an empty evaluation cohort."""

import numpy as np
import pytest

from lid.youtube_data import inventory, speech_chunks
from lid.youtube_metrics import common_cohort, score_segments


def rows(languages):
    return [
        {
            "id": str(i),
            "language": language,
            "duration": 3 + i,
            "source_id": f"clip-{i // 2}",
        }
        for i, language in enumerate(languages)
    ]


def test_inventory_includes_added_wav_clips_alongside_mp3(tmp_path):
    for directory in ("Samoan", "Nuiean", "Tongan"):
        target = tmp_path / directory
        target.mkdir()
        for suffix in (".mp3", ".WAV"):
            (target / f"source{suffix}").write_bytes(f"{directory}:{suffix}".encode())
        (target / "notes.txt").write_text("not audio")
    result = inventory(tmp_path)
    assert len(result) == 6
    assert {row["language"] for row in result} == {"sam", "niu", "ton"}
    assert sum(row["source"].endswith(".WAV") for row in result) == 3


def test_balanced_chunks_preserve_long_speech_and_drop_short_regions():
    kept, dropped = speech_chunks(
        [{"start": 0, "end": 20000}, {"start": 32000, "end": 195200}], 48000, 160000
    )
    assert len(dropped) == 1
    assert len(kept) == 2
    assert kept[0]["start"] == 32000 and kept[-1]["end"] == 195200
    assert kept[0]["end"] == kept[1]["start"]
    assert all(48000 <= value["end"] - value["start"] <= 160000 for value in kept)


def test_segmentation_rejects_overlapping_vad_boundaries():
    with pytest.raises(ValueError, match="overlap"):
        speech_chunks([{"start": 0, "end": 80000}, {"start": 70000, "end": 150000}])


def test_english_is_counted_but_does_not_enter_scores():
    data = rows(["sam", "sam", "niu", "ton"])
    result = score_segments(data, ["sam", "en", "niu", "mi"], [0.9, 0.99, 0.8, 0.95])
    assert result["english_predictions"] == 1
    assert result["segments_scored"] == 3
    assert result["coverage"] == 0.75
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["other_errors"] == 1
    assert result["per_language"]["sam"]["recall"] == 1
    assert "en" not in result["confusion_scored"]["sam"]
    assert result["by_language"]["sam"]["english"] == 1


def test_all_english_is_undefined_not_perfect():
    result = score_segments(rows(["sam", "niu", "ton"]), ["en"] * 3, [0.9] * 3)
    assert result["segments_scored"] == 0
    assert result["english_predictions"] == 3
    assert result["accuracy"] is None
    assert result["macro_f1"] is None
    assert result["ece"] is None
    assert result["coverage"] == 0


def test_common_cohort_excludes_union_of_english_predictions():
    predictions = {
        "frozen15": ["sam", "en", "niu", "ton"],
        "partial15": ["en", "sam", "niu", "ton"],
    }
    common = common_cohort(predictions)
    np.testing.assert_array_equal(common, [False, False, True, True])
    data = rows(["sam", "sam", "niu", "ton"])
    first = score_segments(data, predictions["frozen15"], [0.9] * 4, common)
    second = score_segments(data, predictions["partial15"], [0.9] * 4, common)
    assert first["segments_scored"] == second["segments_scored"] == 2
    assert first["accuracy"] == second["accuracy"] == 1
    assert first["macro_languages"] == ["niu", "ton"]


def test_old_model_without_target_outputs_scores_zero_not_unknown_correct():
    result = score_segments(rows(["sam", "niu", "ton"]), ["mi", "mi", "id"], [0.8] * 3)
    assert result["accuracy"] == 0
    assert result["macro_f1"] == 0
    assert result["segments_scored"] == 3


def test_duration_weighting_and_source_balance_are_separate():
    data = rows(["sam", "sam", "sam"])
    result = score_segments(data, ["sam", "mi", "sam"], [0.9] * 3)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["mean_source_accuracy"] == 0.75
    assert result["duration_weighted_accuracy"] == pytest.approx(8 / 12)


def test_inventory_requires_requested_hindi_and_thai_and_includes_them(tmp_path):
    for directory in ["Samoan", "Nuiean", "Tongan", "Hindi", "Thai"]:
        folder = tmp_path / directory
        folder.mkdir()
        (folder / "audio.wav").write_bytes(directory.encode())
    data = inventory(tmp_path, languages=["sam", "niu", "ton", "hi", "th"])
    assert len(data) == 5
    assert {r["language"] for r in data} == {"sam", "niu", "ton", "hi", "th"}
    (tmp_path / "Thai/audio.wav").unlink()
    with pytest.raises(ValueError, match="requested languages"):
        inventory(tmp_path, languages=["sam", "niu", "ton", "hi", "th"])


def test_hindi_and_thai_enter_macro_metrics_and_wrong_outputs_remain_errors():
    result = score_segments(rows(["hi", "th"]), ["sam", "th"], [0.9, 0.8])
    assert result["accuracy"] == 0.5
    assert result["macro_languages"] == ["hi", "th"]
    assert result["macro_recall"] == 0.5
    assert result["per_language"]["hi"]["recall"] == 0
    assert result["per_language"]["th"]["recall"] == 1


def test_all_segment_policy_counts_english_as_error_without_exclusions():
    result = score_segments(
        rows(["hi", "th"]), ["hi", "en"], [0.9, 0.9], exclude_english=False
    )
    assert result["segments_scored"] == 2 and result["coverage"] == 1
    assert result["english_predictions"] == 1
    assert result["accuracy"] == 0.5
    empty = score_segments(
        rows(["hi", "th"]), ["en", "en"], [0.9, 0.9], exclude_english=False
    )
    assert empty["accuracy"] == 0 and empty["macro_f1"] == 0


def test_group_common_cohort_uses_supported_models_only():
    from lid.youtube_report import summarize_groups

    data = rows(["hi", "th", "sam", "ton"])
    predicted = {"new15": ["hi", "th", "sam", "ton"], "old12": ["hi", "en", "hi", "hi"]}
    confidence = {name: [0.9] * 4 for name in predicted}
    codes = {"new15": ["hi", "th", "sam", "niu", "ton"], "old12": ["hi", "th", "en"]}
    result = summarize_groups(data, predicted, confidence, codes)
    assert result["original_sample"]["old12"]["supported"]
    assert result["original_sample"]["new15"]["all_segments"]["segments_total"] == 2
    assert result["original_sample"]["new15"]["common_cohort"]["segments_scored"] == 1
    assert result["original_sample"]["old12"]["common_cohort"]["segments_scored"] == 1
    assert not result["pacific"]["old12"]["supported"]
    assert result["pacific"]["old12"]["all_segments"] is None
    assert result["pacific"]["new15"]["common_cohort"]["segments_scored"] == 2
