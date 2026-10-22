"""Verify a relocated SpeechBrain export offline against the evaluated head."""

import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from speechbrain.inference.classifiers import EncoderClassifier

from lid.data import waveform
from lid.model import new_head


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent.parent
    run = root / "artifacts/full_run"
    bundle = run / "speechbrain_bundle"
    checkpoint = torch.load(run / "head/head.pt", map_location="cpu", weights_only=True)
    calibration = json.loads((bundle / "calibration.json").read_text())
    head = new_head(checkpoint["initial_state"], checkpoint["indices"]).eval()
    head.load_state_dict(checkpoint["state"])
    data = torch.load(
        run / "fleurs_direct_test.pt", map_location="cpu", weights_only=True
    )
    maximum_error = 0.0
    matches = 0
    # Copy actual files (not cache symlinks) to prove the bundle is relocatable.
    with tempfile.TemporaryDirectory(prefix="lid-bundle-check-") as temporary:
        relocated = Path(temporary) / "model"
        shutil.copytree(bundle, relocated)
        model = EncoderClassifier.from_hparams(
            source=str(relocated),
            savedir=str(Path(temporary) / "loaded"),
            run_opts={"device": "cpu"},
        )
        expected_labels = [checkpoint["labels"][i] for i in checkpoint["indices"]]
        assert [
            model.hparams.label_encoder.ind2lab[i] for i in range(12)
        ] == expected_labels
        temperature = calibration["head_temperature"]
        with torch.inference_mode():
            for batch in data["embeddings"].split(256):
                embedding = batch.unsqueeze(1)
                expected = torch.log_softmax(head(embedding) / temperature, dim=-1)
                actual = model.mods.classifier(embedding)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                maximum_error = max(
                    maximum_error, (actual.exp() - expected.exp()).abs().max().item()
                )
                matches += (actual.argmax(-1) == expected.argmax(-1)).sum().item()
            audio = next((root / "data/fleurs").rglob("*.wav"))
            signal = torch.from_numpy(
                waveform({"path": str(audio), "id": str(audio)}, 10.0)
            ).unsqueeze(0)
            log_probs, _score, index, labels = model.classify_batch(signal)
            embedding = model.encode_batch(signal, normalize=False)
            expected = torch.log_softmax(
                head(embedding).squeeze(1) / temperature, dim=-1
            )
            torch.testing.assert_close(log_probs, expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(1))
            assert labels == [expected_labels[index.item()]]
            # Also exercise the unchanged classify_file interface on a real file.
            file_log_probs, _, _, file_labels = model.classify_file(str(audio))
            assert file_log_probs.shape == (1, 12)
            torch.testing.assert_close(file_log_probs.exp().sum(-1), torch.ones(1))
    result = {
        "offline_relocated_load": "passed",
        "cached_examples": len(data["embeddings"]),
        "matching_predictions": matches,
        "maximum_probability_error": maximum_error,
        "classify_batch": "passed",
        "classify_file": "passed",
        "file_labels": file_labels,
    }
    (bundle / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
