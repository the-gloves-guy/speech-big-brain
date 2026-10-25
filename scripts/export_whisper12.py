"""Export the evaluated Whisper encoder/classifier with the standard SpeechBrain API."""

import argparse
import importlib.metadata
import json
import shutil
import sys
import tarfile
from pathlib import Path

import torch
from speechbrain.dataio.encoder import CategoricalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lid.extension_data import file_hash, write_json
from lid.languages import CODES, NAMES


def export(root, output):
    root, output = Path(root), Path(output)
    source = root / "whisper_finetuned"
    if not (source / "evaluation_complete.json").exists():
        raise ValueError("Fine-tuned Whisper evaluation has not completed")
    if output.exists():
        raise ValueError("Export directory already exists; choose a fresh destination")
    saved = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    calibration = json.loads((source / "calibration.json").read_text())
    identity = json.loads((root / "model_identity.json").read_text())
    temperature = float(calibration["temperature"])
    if not 0 < temperature < float("inf") or calibration["codes"] != CODES:
        raise ValueError("Wrong calibration or language order")
    output.mkdir(parents=True)
    torch.save(
        {"encoder." + key: value for key, value in saved["encoder"].items()},
        output / "embedding_model.ckpt",
    )
    head = {key: value.clone() for key, value in saved["head"].items()}
    head["3.weight"] /= temperature
    head["3.bias"] /= temperature
    torch.save(head, output / "classifier.ckpt")
    labels = CategoricalEncoder()
    labels.update_from_iterable([f"{code}: {name}" for code, name in zip(CODES, NAMES)])
    labels.expect_len(12)
    labels.save(output / "label_encoder.ckpt")
    shutil.copyfile(output / "label_encoder.ckpt", output / "label_encoder.txt")
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "lid/whisper_export_backend.py",
        output / "custom.py",
    )
    dims = identity["dims"]
    yaml = f"""# Standalone twelve-language Whisper tiny encoder/classifier.
# custom.py is fetched automatically by EncoderClassifier.from_hparams.
sample_rate: 16000
out_n_neurons: 12

compute_features: !new:torch.nn.Identity
mean_var_norm: !new:custom.WhisperLIDPassThrough
embedding_model: !new:custom.WhisperTinyLIDEncoder
    n_mels: {dims["n_mels"]}
    n_ctx: {dims["n_audio_ctx"]}
    n_state: {dims["n_audio_state"]}
    n_head: {dims["n_audio_head"]}
    n_layer: {dims["n_audio_layer"]}
    max_samples: 160000
classifier: !new:custom.WhisperTinyLIDClassifier
    n_state: {dims["n_audio_state"]}
    hidden: 512
    outputs: !ref <out_n_neurons>
label_encoder: !new:speechbrain.dataio.encoder.CategoricalEncoder

modules:
    compute_features: !ref <compute_features>
    mean_var_norm: !ref <mean_var_norm>
    embedding_model: !ref <embedding_model>
    classifier: !ref <classifier>

pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    loadables:
        embedding_model: !ref <embedding_model>
        classifier: !ref <classifier>
        label_encoder: !ref <label_encoder>
"""
    (output / "hyperparams.yaml").write_text(yaml, encoding="utf-8")
    write_json(output / "calibration.json", calibration)
    metadata = {
        "model": "Whisper tiny encoder + twelve-language classifier",
        "method": "whisper_finetuned",
        "epoch": saved["epoch"],
        "source_checkpoint_sha256": file_hash(source / "model.pt"),
        "pretrained_whisper_sha256": identity["weight_sha256"],
        "training_split": "original recording-based twelve-language training set; no speaker separation",
        "codes": CODES,
        "names": NAMES,
        "temperature": temperature,
        "temperature_folded_into_classifier": True,
        "preprocessing": {
            "sample_rate": 16000,
            "mono": True,
            "real_audio_max_seconds": 10,
            "crop": "center",
            "whisper_window_seconds": 30,
            "pooling": "mean and population std of real audio positions",
            "vad_included": False,
        },
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ["speechbrain", "openai-whisper", "torch", "torchaudio"]
        },
        "files_sha256": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
    }
    write_json(output / "export.json", metadata)
    (output / "README.md").write_text(
        """# Whisper tiny twelve-language LID

```python
from speechbrain.inference.classifiers import EncoderClassifier

model = EncoderClassifier.from_hparams(
    source="/path/to/whisper_tiny12_finetuned",
    savedir="pretrained_models/whisper_tiny12_lid",
    run_opts={"device": "cuda"},  # "cpu" also works
)
log_probs, score, index, labels = model.classify_file("audio.wav")
probabilities = log_probs.exp()
```

Keep the entire folder together, including `custom.py`. SpeechBrain automatically
loads the bundled Whisper frontend/encoder backend from the YAML. No lid project
code, pretrained model download, training dataset or network access is required.
Install the tested SpeechBrain and openai-whisper versions listed in `export.json`.

The classifier supports ar, en, es, fa, hi, id, mi, ps, ru, th, ur, zh.
Temperature calibration is folded into the final layer; do not apply it twice.
`classify_batch` and `classify_file` return log probabilities like the original
SpeechBrain model. `classify_batch` accepts mono 16 kHz waveforms and optional
relative lengths for padded batches. `classify_file` uses standard SpeechBrain
resampling and mono conversion.

The backend reproduces the benchmark: center-crop to at most ten real seconds,
right-zero-pad to Whisper's thirty-second input, native FP32 log-mel frontend,
and mean/std pooling over real audio positions. Cropping is automatic in this
bundle. VAD and unknown-language rejection are not included.

This exports the trained language-ID encoder and classifier, with no unused text
decoder. It uses SpeechBrain's `EncoderClassifier.from_hparams` interface. Native
`whisper.load_model` and transcription APIs expect a different encoder/decoder
checkpoint and do not load this custom language-ID head.

The selected epoch is recorded in `export.json`; selection uses VoxLingua NLL, with calibration fit
only on FLEURS validation. The test data were not used to choose the checkpoint.
For strict CUDA benchmark reproducibility disable TF32 for CUDA matmul and cuDNN;
the verification record states the precision used.
""",
        encoding="utf-8",
    )
    print(json.dumps({"bundle": str(output), "epoch": saved["epoch"]}), flush=True)
    return output


def archive(output):
    output = Path(output)
    files = sorted(
        p for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{file_hash(p)}  {p.name}\n" for p in files), encoding="ascii"
    )
    path = output.with_suffix(".tar.gz")
    temporary = path.with_name(path.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as stream:
        stream.add(output, arcname=output.name)
    temporary.replace(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{file_hash(path)}  {path.name}\n", encoding="ascii"
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/whisper_tiny12")
    parser.add_argument(
        "--output", default="artifacts/deployment/whisper_tiny12_finetuned"
    )
    args = parser.parse_args()
    export(args.root, args.output)
