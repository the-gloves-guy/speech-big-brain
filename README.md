# Restricted-language speech identification

Compare `speechbrain/lang-id-voxlingua107-ecapa` in a deployment containing
Arabic, English, Spanish, Persian, Hindi, Indonesian, Māori, Pushto, Russian,
Thai, Urdu, and Mandarin Chinese.

The pipeline reports the original 107-class model, its restricted 12-class
distribution, and a retrained 12-class classifier. All three models are
evaluated before and after temperature scaling. Results are generated from real
data; no benchmark scores are included in this repository yet.

## Setup

```bash
uv sync --dev
uv run lid --help
uv run pytest
```

Python 3.11 or 3.12 is required. PyTorch and torchaudio are pinned together to
2.5.1 with CUDA 12.1 wheels, compatible with the RTX 3070 and a recent NVIDIA
driver. The initial installation downloads several GB of dependencies. Dataset
and model downloads need internet access. The uv lockfile records resolved
dependencies. `uv run lid cache --device cpu ...` supports CPU extraction too.

## Experiment design

1. **Filter and rescale:** select the twelve allowed columns of the original
   **logits**, then calculate `softmax(selected_logits / T)`. `T=1` is ordinary
   restriction and renormalization. Fit one positive temperature by minimizing
   FLEURS validation negative log likelihood, without changing test predictions.
   A scalar temperature changes confidence but cannot change the winning class.
2. **Retrain the head:** freeze ECAPA and extract its 256-dimensional embeddings.
   Copy the pretrained classifier MLP, retain its hidden layers, and initialize
   its final projection with the twelve relevant pretrained rows. Fine-tune the
   complete classifier, including its batch normalization layers, using only
   the selected VoxLingua107 training languages. The encoder stays frozen.
   Use a video-disjoint VoxLingua holdout for early stopping. The default loss
   weights each class inversely to its training count. Fit a separate temperature
   on FLEURS validation for a fair calibration comparison.
3. **Evaluate:** run all six variants on the same held-out FLEURS test examples.
   Precision, recall and F1 are reported per language, macro averaged, and support
   weighted. Accuracy is overall top-1 accuracy. For the original model, predictions
   outside the selected languages count as errors. Its macro metrics average over
   the twelve target languages; its confidence and ECE retain all 107 classes.

ECE uses fifteen equal-width bins over maximum predicted probability. Each bin's
absolute confidence–accuracy gap is weighted by its sample count. Empty bins have
zero contribution. ECE depends on the bin choice and sample composition; NLL and
multiclass Brier score are also saved. Compare calibration with the same bin count.

FLEURS validation is used for calibration, so this measures target-domain
calibration rather than a completely untuned cross-domain transfer. FLEURS test
must never be used to choose temperatures, checkpoints, languages, or hyperparameters.
For deployment conclusions, repeat across training seeds and evaluate deployment
recordings as well. FLEURS read speech and VoxLingua web speech have different
acoustic and language distributions.

## 1. Prepare selected VoxLingua107 training data

Obtain the original VoxLingua107 **training** audio for the twelve ISO codes below
from the [dataset distribution](https://bark.phon.ioc.ee/voxlingua107/).
The [SpeechBrain recipe](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxLingua107)
describes the source data. Download only the selected language archives and extract
them locally; this repository deliberately does not download the entire 6,628-hour
corpus. Point `--root` at a directory containing `ar/`, `en/`, …, `zh/`.

```bash
uv run lid prepare-vox --root /path/to/voxlingua107/train
```

This creates `artifacts/manifests/train.jsonl` and `validation.jsonl`, holding
out about 10% of source videos per language with seed 42. Standard filenames must
begin with an 11-character YouTube ID followed by a separator. For renamed data,
supply your own two manifests. Each JSONL line must look like:

```json
{"path": "/absolute/path/clip.wav", "language": "en", "group": "source-video-id"}
```

Relative paths resolve against the manifest directory. Each manifest must contain
all twelve languages. Groups must identify the same original video across clips;
do not substitute a separate group for each segment. Training rejects overlapping
paths or groups between the two caches. Corpus provenance is the user's
responsibility: a local manifest cannot prove its files are VoxLingua training data.

## 2. Cache embeddings and original logits

Start with a balanced pilot of 1,000 training clips and 100 validation clips per
language, then remove `--limit-per-language` to process the full selected corpus.
The cap takes the first examples in manifest order; it is a resource limit,
not a random sampling strategy. Use a fresh output path for each experiment.

```bash
uv run lid cache --manifest artifacts/manifests/train.jsonl --split train --limit-per-language 1000 --output artifacts/cache/vox_train.pt
uv run lid cache --manifest artifacts/manifests/validation.jsonl --split validation --limit-per-language 100 --output artifacts/cache/vox_validation.pt

uv run python scripts/run_guarded.py --memory 2G -- uv run lid cache --fleurs --split validation --buffer-size 16 --threads 2 --output artifacts/cache/fleurs_validation.pt
uv run python scripts/run_guarded.py --memory 2G -- uv run lid cache --fleurs --split test --buffer-size 16 --threads 2 --output artifacts/cache/fleurs_test.pt
```

Extraction defaults to CUDA, FP32, one clip at a time, no gradients, and a
deterministic center crop of at most **10 seconds**. This is a cropped-audio
evaluation, not a full-utterance benchmark. All methods receive the same audio.
Use `--batch-size 16` to batch equal-length signals (without padding); a 32-clip
buffer groups matching lengths. The length cap limits ECAPA activation memory on an 8 GB GPU. If necessary use
`--max-seconds 5` consistently for every cache. `--max-seconds 0` processes full
utterances and can exceed GPU memory. CPU extraction is available with
`--device cpu`. Audio is mixed to mono and resampled to 16 kHz.

FLEURS uses its original `.tar.gz` audio archives and `.tsv` metadata. Archives are
downloaded to disk under `data/fleurs/<revision>/`, then extracted with fixed-size
copies. WAV files are opened individually. Parquet/Arrow audio streaming is not
used: it caused excessive RAM use on this machine even with small read batches.
Only validation and test are needed; FLEURS train is not used for head training.
This needs additional disk space for archives and extracted audio.

`lid cache --fleurs` automatically starts inside a **2 GiB** memory-limited scope.
Use `scripts/run_guarded.py`, as above, to select a different job limit. On Linux this requires a
working systemd user manager and applies a kernel-enforced RAM cap with zero swap
for the job. If the cap is exceeded, the job is terminated; it does not run without
the requested guard if setup fails. CPU and I/O priority are reduced too. A small
allocation test verified that an over-limit child is killed. There is no automatic
OOM retry. Extraction saves
atomic intermediate shards in `<output>.parts` and automatically resumes them
when the same command is rerun with compatible settings. The final cache is
written atomically after extraction succeeds.

The replacement reader completed all selected FLEURS test recordings (8,519) with
1.40 GiB peak process RAM and validation recordings (4,172) with 1.30 GiB, both
with zero swaps and exit status 0. Per-run `/usr/bin/time -v` measurements are in
`artifacts/full_run/fleurs_direct_{test,validation}.memory.txt`.

For the full run, `scripts/prefetch_vox.py` caches each prepared training language
while other archives are still downloading. These are frozen inference features;
the final video-disjoint split is applied when assembling the training and
validation caches, before any classifier fitting. Partial language caches are not
accepted directly by training or evaluation. Inspect lightweight live progress
with `python3 scripts/progress.py`.

Model and dataset branch names resolve to commit SHAs recorded in each cache.
For repeated runs, pass those SHAs via `--model-revision` and
`--dataset-revision`; cache compatibility checks reject different encoder revisions,
language orders, or crop protocols. Keep dataset revisions the same across FLEURS
validation and test. Caches contain embeddings, original logits, IDs, groups,
labels, model weights for head initialization, and extraction metadata.

| Language | VoxLingua code | FLEURS config |
| --- | --- | --- |
| Arabic | ar | ar_eg |
| English | en | en_us |
| Spanish | es | es_419 |
| Persian | fa | fa_ir |
| Hindi | hi | hi_in |
| Indonesian | id | id_id |
| Māori | mi | mi_nz |
| Pushto / Pashto | ps | ps_af |
| Russian | ru | ru_ru |
| Thai | th | th_th |
| Urdu | ur | ur_pk |
| Mandarin Chinese | zh | cmn_hans_cn |

“Chinese” here means Mandarin, not Cantonese or every Sinitic language. Arabic is
evaluated using FLEURS `ar_eg`; Spanish uses `es_419`. These choices constrain
what conclusions the benchmark supports.

## 3. Train only the classifier

```bash
uv run lid train-head --train artifacts/cache/vox_train.pt --validation artifacts/cache/vox_validation.pt --output artifacts/head
```

The small classifier trains on CPU by default, using batch size 256, AdamW,
learning rate 1e-4, a maximum of 30 epochs, and patience 5. Choose `--device cuda`
if desired. The best checkpoint minimizes unweighted VoxLingua validation NLL.
Training history and the head checkpoint are saved in the output directory.
No FLEURS data is used for checkpoint selection.

## 4. Calibrate and compare

```bash
uv run lid compare --calibration artifacts/cache/fleurs_validation.pt --test artifacts/cache/fleurs_test.pt --head artifacts/head/head.pt --output artifacts/comparison
```

Outputs include:

- `summary.csv`: accuracy, macro/weighted precision, recall and F1, ECE, NLL,
  Brier score, outside-set prediction rate and temperature for each method.
- `per_language.csv`: precision, recall, F1 and support by language and method.
- `report.json`: metrics, reliability bins, confusion matrices and provenance.
  Baseline confusion matrices have 12 target rows and 107 prediction columns;
  adapted matrices have 12 rows and 12 columns. Label orders are included.
- `reliability.png`: calibration diagrams; marker size reflects bin population.
- `calibration.json`: fitted temperatures, calibration example IDs and head hash.
- `predictions.npz`: test IDs, targets and logits for further analysis.

Output paths must be new to avoid overwriting earlier experiments. Temperatures
are constrained to [0.05, 20]. A fitted value near a boundary deserves inspection.
All metrics are fractions in [0,1] except NLL, temperature, and counts.

## 5. Predict a deployment recording

```bash
uv run lid predict recording.wav --method filtered --calibration artifacts/comparison/calibration.json
uv run lid predict recording.wav --method head --head artifacts/head/head.pt --calibration artifacts/comparison/calibration.json
```

Use `--device cpu` for CPU inference. Omit `--calibration` to use `T=1`.
The same preprocessing and encoder revision are loaded from the artifacts.
Output JSON includes probabilities, the winning language, and confidence.
For offline deployment, download the model once, copy the Hub snapshot cache and
the head/calibration artifacts, and set `HF_HUB_OFFLINE=1`. Calibrated inference
loads the recorded model SHA directly rather than looking up a mutable branch.
These are closed-set predictions: neither adapted method recognizes unknown
languages or silence as a separate class. If those occur in production, evaluate
an explicit rejection policy on separate in-set and out-of-set recordings.

## Standalone SpeechBrain bundle

The completed run is exported to `artifacts/full_run/speechbrain_bundle/`.
Copy the entire directory to deploy it. It includes the frozen ECAPA encoder,
fine-tuned 12-language classifier, labels, calibration metadata, and provenance.
It needs SpeechBrain and its dependencies, but no project-specific Python code
or access to the original model on Hugging Face.

```python
from speechbrain.inference.classifiers import EncoderClassifier

model = EncoderClassifier.from_hparams(
    source="artifacts/full_run/speechbrain_bundle",
    run_opts={"device": "cuda"},  # or "cpu"
)
log_probs, score, index, labels = model.classify_file("recording.wav")
probabilities = log_probs.exp()
```

Temperature calibration is folded into the final classifier weights. Do not
apply the temperature again. Return values follow the original SpeechBrain
interface, with 12 outputs ordered as `ar, en, es, fa, hi, id, mi, ps, ru, th, ur, zh`.
Labels retain the original `code: Language` format.

The reported evaluation used mono 16 kHz recordings center-cropped to at most
10 seconds. Standard `classify_file()` resamples but does not crop; prepare the
same crop to reproduce evaluation preprocessing. `classify_batch()` expects
mono 16 kHz waveforms.

To export another trained checkpoint into a new destination:

```bash
uv run python scripts/run_guarded.py --memory 2G -- \
  uv run python scripts/export_speechbrain.py \
  --head artifacts/full_run/head/head.pt \
  --calibration artifacts/full_run/comparison/calibration.json \
  --output artifacts/new_speechbrain_bundle
```

The exporter uses the pinned model snapshot already cached locally. It checks
calibration/checkpoint compatibility and records file checksums in `export.json`.
The completed bundle's `verification.json` records an offline relocated load,
prediction parity on all 8,519 test embeddings, and real-audio checks through
both `classify_batch()` and `classify_file()`.

## Sources

- [Pretrained model card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
- [Pretrained architecture](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/blob/main/hyperparams.yaml)
- [FLEURS dataset](https://huggingface.co/datasets/google/fleurs)
- [Temperature scaling and ECE](https://proceedings.mlr.press/v70/guo17a.html)
