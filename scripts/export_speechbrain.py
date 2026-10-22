"""Export the trained head as a relocatable, standard SpeechBrain bundle."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from speechbrain.dataio.encoder import CategoricalEncoder

from lid.languages import CODES, selected_indices
from lid.model import MODEL_ID, new_head


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(head_path, calibration_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    calibration = json.loads(Path(calibration_path).read_text())
    if digest(head_path) != calibration["head_sha256"]:
        raise ValueError("Calibration belongs to a different checkpoint")
    for key in ("model_revision", "max_seconds", "crop", "codes"):
        if checkpoint["meta"][key] != calibration["metadata"][key]:
            raise ValueError(f"Calibration metadata mismatch: {key}")
    if checkpoint["meta"]["codes"] != CODES:
        raise ValueError("Unexpected target language ordering")
    if selected_indices(checkpoint["labels"]) != checkpoint["indices"]:
        raise ValueError("Unexpected pretrained label ordering")
    temperature = float(calibration["head_temperature"])
    if not 0 < temperature < float("inf"):
        raise ValueError("Invalid calibration temperature")
    # Export only from the immutable, already-downloaded model snapshot.
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=checkpoint["meta"]["model_revision"],
            local_files_only=True,
        )
    )
    head = new_head(checkpoint["initial_state"], checkpoint["indices"])
    head.load_state_dict(checkpoint["state"])
    weights = dict(head.state_dict())
    # Training replaced SB's Linear wrapper with torch.nn.Linear. Restore its
    # state-dict naming and fold logit / T into the final affine projection.
    for parameter in ("weight", "bias"):
        weights[f"out.w.{parameter}"] = weights.pop(f"out.{parameter}") / temperature
    yaml = (snapshot / "hyperparams.yaml").read_text()
    yaml = yaml.replace("pretrained_path: speechbrain/lang-id-voxlingua107-ecapa", "")
    yaml = yaml.replace("out_n_neurons: 107", "out_n_neurons: 12")
    # Omit explicit paths: Pretrainer resolves filenames against from_hparams'
    # source, so moving the bundle needs neither overrides nor base-model access.
    yaml = yaml.split("    paths:\n", 1)[0]
    if "pretrained_path" in yaml or "out_n_neurons: 12" not in yaml:
        raise ValueError("Unexpected upstream hyperparameter template")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".sb-export-") as tmp:
        bundle = Path(tmp) / "bundle"
        bundle.mkdir()
        shutil.copyfile(
            snapshot / "embedding_model.ckpt", bundle / "embedding_model.ckpt"
        )
        torch.save(weights, bundle / "classifier.ckpt")
        labels = CategoricalEncoder()
        labels.update_from_iterable(
            checkpoint["labels"][index] for index in checkpoint["indices"]
        )
        # With default_source, Pretrainer fetches every loadable as NAME.ckpt,
        # including the categorical encoder (whose contents remain plain text).
        labels.save(bundle / "label_encoder.ckpt")
        shutil.copyfile(bundle / "label_encoder.ckpt", bundle / "label_encoder.txt")
        (bundle / "hyperparams.yaml").write_text(yaml)
        shutil.copyfile(calibration_path, bundle / "calibration.json")
        metadata = {
            "base_model": MODEL_ID,
            "base_revision": checkpoint["meta"]["model_revision"],
            "head_sha256": digest(head_path),
            "best_epoch": checkpoint["best_epoch"],
            "codes": CODES,
            "temperature": temperature,
            "temperature_folded_into_classifier": True,
            "preprocessing": {
                key: checkpoint["meta"][key] for key in ("max_seconds", "crop")
            },
            "files_sha256": {
                path.name: digest(path) for path in sorted(bundle.iterdir())
            },
        }
        (bundle / "export.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (bundle / "README.md").write_text(
            "# Fine-tuned VoxLingua 12-language classifier\n\n"
            "Load with standard SpeechBrain 1.0.3:\n\n"
            "```python\n"
            "from speechbrain.inference.classifiers import EncoderClassifier\n"
            "model = EncoderClassifier.from_hparams(\n"
            "    source='/path/to/this/bundle',\n"
            "    run_opts={'device': 'cuda'},\n"
            ")\n"
            "log_probs, score, index, labels = model.classify_file('audio.wav')\n"
            "probabilities = log_probs.exp()\n"
            "```\n\n"
            "Contains the frozen original ECAPA encoder and trained 12-way head. "
            "Temperature calibration is already folded into the classifier; "
            "do not apply it again. Returns log-probabilities, like the base model.\n\n"
            "Language order: " + ", ".join(CODES) + ".\n\n"
            "Evaluation used mono 16 kHz audio, center-cropped to at most 10 seconds. "
            "Standard SpeechBrain resamples files but does not apply this crop; "
            "use the same crop when reproducing reported metrics. "
            "For classify_batch, supply mono 16 kHz waveforms.\n\n"
            "The directory is self-contained and can be moved without overrides. "
            "export.json records provenance and checksums.\n"
        )
        bundle.rename(output)
    print(f"Exported calibrated SpeechBrain bundle to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="artifacts/full_run/head/head.pt")
    parser.add_argument(
        "--calibration", default="artifacts/full_run/comparison/calibration.json"
    )
    parser.add_argument("--output", default="artifacts/full_run/speechbrain_bundle")
    args = parser.parse_args()
    export(args.head, args.calibration, args.output)
