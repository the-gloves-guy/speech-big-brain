"""Standalone, decoder-free SpeechBrain exports and relocated offline verification."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .extension_data import CODES, NAMES, file_hash, write_json


def export_one(study, method, output):
    from speechbrain.dataio.encoder import CategoricalEncoder

    source = study.root / method
    identity_path = study.root / "model_identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    model_name = identity.get("model", "tiny")
    dims = identity.get(
        "dims",
        {
            "n_mels": 80,
            "n_audio_ctx": 1500,
            "n_audio_state": 384,
            "n_audio_head": 6,
            "n_audio_layer": 4,
        },
    )
    if not (source / "evaluation_complete.json").exists():
        raise ValueError("Cannot export an unevaluated model")
    saved = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    calibration = json.loads((source / "calibration.json").read_text())
    if (
        calibration["model_sha256"] != file_hash(source / "model.pt")
        or calibration["codes"] != CODES
    ):
        raise ValueError("Export calibration/checkpoint mismatch")
    if output.exists():
        meta = json.loads((output / "export.json").read_text())
        if meta["source_checkpoint_sha256"] != calibration["model_sha256"]:
            raise ValueError("Export destination belongs to another checkpoint")
        for name, checksum in meta["files_sha256"].items():
            if file_hash(output / name) != checksum:
                raise ValueError("Existing export changed")
        if (
            identity.get("bounded_memory_loading")
            and not (output / "verification.json").exists()
        ):
            # Repair only an unverified bundle's loader; evaluated weights and
            # calibration remain byte-identical, and completed exports stay intact.
            shutil.copyfile(
                Path(__file__).with_name("whisper_export_backend.py"),
                output / "custom.py",
            )
            yaml_path = output / "hyperparams.yaml"
            yaml = yaml_path.read_text()
            if "custom_hooks:" not in yaml:
                yaml = yaml.replace(
                    "pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer\n",
                    "pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer\n    custom_hooks:\n        embedding_model: !name:custom.load_whisper_encoder\n",
                )
                yaml_path.write_text(yaml)
            meta["memory_mapped_encoder_loading"] = True
            meta["files_sha256"] = {
                name: file_hash(output / name) for name in meta["files_sha256"]
            }
            write_json(output / "export.json", meta)
        return
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        # Rebuild only our explicitly incomplete export; never modify completed bundles.
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    if identity.get("bounded_memory_loading"):
        from .model import cpu_state
        from .whisper_model import load_whisper

        if (
            saved.get("encoder_checkpoint_sha256")
            != identity["encoder_checkpoint_sha256"]
        ):
            raise ValueError("Frozen classifier references a different encoder")
        model = load_whisper(study.root)
        encoder_state = cpu_state(model.encoder)
        torch.save(
            {"encoder." + k: v for k, v in encoder_state.items()},
            temporary / "embedding_model.ckpt",
        )
        del encoder_state, model
        torch.cuda.empty_cache()
    else:
        torch.save(
            {"encoder." + k: v for k, v in saved["encoder"].items()},
            temporary / "embedding_model.ckpt",
        )
    head = {k: v.clone() for k, v in saved["head"].items()}
    temperature = calibration["temperature"]
    head["3.weight"] /= temperature
    head["3.bias"] /= temperature
    torch.save(head, temporary / "classifier.ckpt")
    labels = CategoricalEncoder()
    labels.update_from_iterable([f"{code}: {name}" for code, name in zip(CODES, NAMES)])
    labels.expect_len(15)
    labels.save(temporary / "label_encoder.ckpt")
    shutil.copyfile(temporary / "label_encoder.ckpt", temporary / "label_encoder.txt")
    shutil.copyfile(
        Path(__file__).with_name("whisper_export_backend.py"), temporary / "custom.py"
    )
    load_hook = (
        "    custom_hooks:\n        embedding_model: !name:custom.load_whisper_encoder\n"
        if identity.get("bounded_memory_loading")
        else ""
    )
    (temporary / "hyperparams.yaml").write_text(f"""sample_rate: 16000
out_n_neurons: 15
compute_features: !new:torch.nn.Identity
mean_var_norm: !new:custom.WhisperLIDPassThrough
embedding_model: !new:custom.WhisperLIDEncoder
    n_mels: {dims["n_mels"]}
    n_ctx: {dims["n_audio_ctx"]}
    n_state: {dims["n_audio_state"]}
    n_head: {dims["n_audio_head"]}
    n_layer: {dims["n_audio_layer"]}
    bounded_memory_loading: {str(identity.get("bounded_memory_loading", False)).lower()}
classifier: !new:custom.WhisperLIDClassifier
    n_state: {dims["n_audio_state"]}
    outputs: !ref <out_n_neurons>
label_encoder: !new:speechbrain.dataio.encoder.CategoricalEncoder
modules:
    compute_features: !ref <compute_features>
    mean_var_norm: !ref <mean_var_norm>
    embedding_model: !ref <embedding_model>
    classifier: !ref <classifier>
pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
{load_hook}    loadables:
        embedding_model: !ref <embedding_model>
        classifier: !ref <classifier>
        label_encoder: !ref <label_encoder>
""")
    (
        temporary / "README.md"
    ).write_text(f"""# Whisper {model_name} fifteen-language LID ({method})

```python
from speechbrain.inference.classifiers import EncoderClassifier
model = EncoderClassifier.from_hparams(
    source="/path/to/{output.name}",
    savedir="/tmp/whisper15_{method}_loaded",
    run_opts={{"device": "cpu"}},  # cuda also supported
)
log_probs, score, index, labels = model.classify_file("audio.wav")
# classify_batch accepts mono 16 kHz padded waveforms and relative lengths.
```

Keep custom.py with the checkpoint files. No project imports or transcription
decoder are required. Temperature is folded once into the final classifier;
do not apply another temperature. The fifteen output labels, in order, are
{", ".join(CODES)}. Preprocessing is mono 16 kHz, at most ten real seconds
(center crop), native thirty-second padding, real-token mean/std pooling.
VAD is not bundled. Calibrated on FLEURS validation plus Pacific calibration.
Selected on combined VoxLingua/Pacific selection macro F1, then balanced NLL.
Original recording-based splits do not establish speaker independence.
Runtime versions and checksums are recorded in export.json.
""")
    write_json(temporary / "calibration.json", calibration)
    write_json(
        temporary / "export.json",
        {
            "model": model_name,
            "dims": dims,
            "source_checkpoint_sha256": calibration["model_sha256"],
            "method": method,
            "epoch": saved["epoch"],
            "codes": CODES,
            "names": NAMES,
            "temperature": temperature,
            "temperature_folded_into_classifier": True,
            "runtime_versions": {
                n: importlib.metadata.version(n)
                for n in ["torch", "speechbrain", "openai-whisper", "torchaudio"]
            },
            "files_sha256": {
                p.name: file_hash(p) for p in temporary.iterdir() if p.is_file()
            },
        },
    )
    temporary.replace(output)


def verify(study, method, output):
    """Fresh interpreter, relocated bundle, network disabled, no lid imports."""
    source = study.root / method
    with tempfile.TemporaryDirectory(prefix="whisper15-offline-") as workspace:
        workspace = Path(workspace)
        relocated = workspace / "bundle"
        shutil.copytree(output, relocated)
        cohorts = {}
        for cohort in ["fleurs_test", "new_test", "external"]:
            with np.load(source / f"{cohort}.npz") as saved:
                expected = (
                    saved["logits"]
                    / json.loads((source / "calibration.json").read_text())[
                        "temperature"
                    ]
                )
                np.save(
                    workspace / f"{cohort}.npy",
                    torch.from_numpy(expected).log_softmax(-1).numpy(),
                )
            cohorts[cohort] = [
                {"id": r["id"], "path": str(Path(r["path"]).resolve(strict=True))}
                for r in study.rows(cohort)
            ]
        write_json(workspace / "cohorts.json", cohorts)
        code = """import json, socket, sys
from pathlib import Path
import numpy as np
import torch, torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
def denied(*a, **kw):
    raise RuntimeError("Network access forbidden during export verification")
socket.socket.connect = denied
root = Path(sys.argv[1])
model = EncoderClassifier.from_hparams(source=str(root / "bundle"), savedir=str(root / "loaded"), run_opts={"device": "cuda"})
maximum, n, disagreement = 0.0, 0, 0
cohorts = json.loads((root / "cohorts.json").read_text())
with torch.inference_mode():
    for cohort, rows in cohorts.items():
        expected = np.load(root / f"{cohort}.npy")
        for start in range(0, len(rows), 2):
            signals = []
            for row in rows[start:start+2]:
                signal, rate = torchaudio.load(row["path"])
                signal = signal.mean(0)
                if rate != 16000:
                    signal = torchaudio.functional.resample(signal, rate, 16000)
                size = min(len(signal), 160000)
                left = (len(signal)-size)//2
                signals.append(signal[left:left+size])
            lengths = torch.tensor([len(s) for s in signals])
            batch = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
            actual = model.classify_batch(batch, lengths/lengths.max())[0].reshape(len(signals), 15).cpu().numpy()
            reference = expected[start:start+len(signals)]
            maximum = max(maximum, float(np.max(np.abs(actual-reference))))
            disagreement += int((actual.argmax(1) != reference.argmax(1)).sum())
            np.testing.assert_allclose(actual, reference, atol=0.005, rtol=0.001)
            n += len(signals)
        print(json.dumps({"cohort": cohort, "verified": len(rows)}), flush=True)
    # File loading and CPU inference use the same offline relocated bundle.
    row = cohorts["external"][0]
    gpu_file = model.classify_file(row["path"])[0].cpu()
    cpu = EncoderClassifier.from_hparams(source=str(root / "bundle"), savedir=str(root / "cpu_loaded"), run_opts={"device": "cpu"})
    cpu_file = cpu.classify_file(row["path"])[0]
    torch.testing.assert_close(cpu_file, gpu_file, atol=0.005, rtol=0.001)
if disagreement:
    raise ValueError(f"Export prediction disagreement: {disagreement}")
(root / "verified.json").write_text(json.dumps({"n": n, "max_absolute_log_probability_error": maximum, "prediction_disagreements": disagreement, "relocated": True, "offline": True, "cpu_file_and_gpu_batch": True}))
"""
        script = workspace / "verify.py"
        script.write_text(code)
        env = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONPATH="",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
        )
        subprocess.run(
            [sys.executable, "-u", str(script), str(workspace)],
            cwd=workspace,
            env=env,
            check=True,
        )
        write_json(
            output / "verification.json",
            json.loads((workspace / "verified.json").read_text()),
        )


def export_models(study):
    # This import uses the existing archive helper without changing Whisper12 exports.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "whisper12_export", Path(__file__).parents[1] / "scripts/export_whisper12.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for method in ["frozen", "finetuned"]:
        model_name = json.loads((study.root / "model_identity.json").read_text()).get(
            "model", "tiny"
        )
        output = Path(f"artifacts/deployment/whisper_{model_name}15_{method}")
        export_one(study, method, output)
        if not (output / "verification.json").exists():
            verify(study, method, output)
        module.archive(output)
    write_json(
        study.root / "export_complete.json",
        {"methods": ["frozen", "finetuned"], "offline_verification_required": True},
    )
