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

## Offline deployment notebook

Open [notebooks/offline_deployment_demo.ipynb](notebooks/offline_deployment_demo.ipynb)
for loading Whisper Small frozen, SpeechBrain partial15 and Whisper Tiny frozen
directly from their extracted deployment folders. It shows the supported language
set and output order, single-file inference, probabilities, and padded batch
inference with real audio lengths. It defaults to CPU and uses no project imports,
model downloads or internet services.

Prepare notebook tooling before disconnecting with `uv sync --group notebook`,
then select `.venv/bin/python` as the notebook kernel in your editor. Set the
deployment and audio paths in the configuration cell. In this checkout the demo
automatically finds the three deployment folders and existing local v4 speech
segments; when moving it elsewhere, supply your own local audio. No install cells
are run in the notebook, and existing deployment folders are left unchanged.

To reproduce the real-kernel checks without internet access:

```bash
uv run --offline --group notebook python scripts/check_offline_notebook.py
uv run --offline --group notebook python scripts/check_offline_notebook.py --reverse-model-order
```

These checks use an empty Hugging Face cache and block Python network connections
inside the kernel. They also verify that all three models expose fifteen languages
and perform single-file and unequal-length batch inference. Add `--write` to save
the executed example outputs into the notebook.

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

## Three-language extension (15 languages total)

The extension adds **Samoan (`sam`), Niuean (`niu`), and Tongan (`ton`)** to the
original 12. Cook Islands Māori / Rarotongan is excluded. Existing 12-language
artifacts and commands are preserved.

Run or resume the complete pipeline:

```bash
uv run python -u scripts/run_extension15.py
```

The runner automatically enters a systemd scope with a 4 GiB RAM limit,
swap disabled, and reduced CPU/I/O priority. It runs GPU work sequentially.
Specific stages can be run with `--stage audit`, `cache`, `frozen`,
`evaluate_frozen`, `partial`, `evaluate_partial`, or `comparison`.

Watch live progress without loading tensors:

```bash
watch -n 10 python3 scripts/progress.py --extension
```

The output includes the current phase, per-epoch/per-cache progress, selected
epoch, pipeline milestones, worker state, memory limits and audited split sizes.
An epoch percentage is progress through that epoch, not an overall ETA.

Artifacts and logs live in `artifacts/extension15/`. The final report is
`reports/extension15/language_extension.md`; both trained models receive their
own `speechbrain_bundle/` inside the `frozen/` and `partial/` output directories.
The final `status.json` phase becomes `complete` only after evaluation, bundle
verification, and report generation. On a normal exception it records `failed`;
worker state also exposes interruptions or kernel termination.

Protocol:

- Snapshot and audit all WAV cuts from the three supplied external segment
  directories. Originals are read only; new scraper output is not added midway.
- Approximate 80/5/5/10 train/selection/calibration/test assignment by hashed
  source-recording group, seed 42. Cuts and exact-duplicate-linked sources do
  not cross partitions. Source independence does not prove speaker independence.
- Initialize both methods identically from the uncalibrated 12-language head,
  preserve its 12 output rows, and append three new rows in `sam, niu, ton` order.
- Sample the 15 training classes uniformly, including original VoxLingua replay.
  Each logical epoch draws 32,768 examples; the effective batch size is 32.
- Frozen method: classifier training from cached embeddings. Partial method:
  additionally train encoder `mfa`, `asp`, and `fc`, detaching earlier blocks.
  Encoder batch-normalization parameters and statistics remain frozen; classifier
  normalization affine parameters train, with running statistics frozen.
- AdamW, head LR 1e-4, encoder-tail LR 1e-5, weight decay 1e-4, gradient norm
  limit 5, AMP, at most 20 epochs and patience 5. Select by 15-class validation
  macro F1, breaking ties by class-balanced NLL. No regression limit is imposed.
- Partial training starts with microbatch 2 and accumulation 16. A real backward
  preflight must reserve less than 7 GiB VRAM; otherwise use microbatch 1 with
  accumulation 32. Frozen parameters/buffers are checked for changes each epoch.
- New-model evaluation retains all 15 outputs on both the original FLEURS test
  set and new-language test cuts. Joint calibration uses FLEURS validation and
  the reserved new calibration groups; an additional old-only calibration fit
  supports comparison with the original protocol. No test-set checkpoint tuning.
- The baseline six evaluations are recomputed from saved logits and checked
  against their original metrics. Encoder-tail changes invalidate original
  embeddings; partial-model validation and evaluation features are recomputed.
- Waveforms remain mono 16 kHz with a maximum 10-second center crop and no
  added amplitude normalization. Validation-only gain probes quantify whether
  quieter Samoan recordings are sensitive to uniform volume changes.

Training checkpoints resume from completed optimizer steps (saved every 4,096
examples and at epoch boundaries). Feature extraction resumes from durable
256-example shards. Changed audit snapshots, encoder weights, cache contexts,
or model/data identities are rejected rather than silently reused.

## Held-out YouTube validation with VAD

Future experiments use the original recording-based training sets and the canonical
models under `artifacts/full_run` and `artifacts/extension15`. Speaker-separation
training is stopped and is historical; do not resume it unless explicitly requested.
The persistent preference is recorded in `AGENTS.md`.

```bash
uv run python scripts/evaluate_youtube.py --min-seconds 3 --languages sam niu ton hi th --output artifacts/youtube_validation_v3
```

This runs under a 3 GiB systemd RAM cap with swap disabled. It reads WAVs and MP3s from
`validation_clips/Samoan`, `validation_clips/Tongan`, `validation_clips/Nuiean`,
`validation_clips/Hindi`, and `validation_clips/Thai` (the spelling `Niuean` is also
accepted). The explicit `--languages` option requires audio for every requested
language; omit it for a Pacific corpus with optional Hindi/Thai sources. Originals and model weights are read
only. Decode one file at a time; Silero VAD 6.2.3 runs on CPU. Speech regions
shorter than 3 seconds are discarded, while longer regions are split evenly
into non-overlapping segments of at most 10 seconds. Durations include VAD
padding. No added level normalization or test-set fitting is performed.

Both 15-language models are evaluated. Original-107, filtered-12 and previous
fine-tuned-12 models are included as reference controls. They support Hindi and
Thai but lack the three Pacific outputs; unsupported group comparisons are marked
accordingly. Group results separate Hindi/Thai from Pacific languages. All-segment
metrics count English predictions as errors against the folder labels; the original
English-excluded metrics and same-cohort comparisons are also retained. In-domain
FLEURS/radio comparisons use the same checkpoint weights, with differing audio
populations and protocols explicitly identified.

For each model, top-1 English predictions are counted (including seconds and
source/language breakdowns) and excluded from accuracy, precision, recall, F1
and ECE. Other output languages remain errors. Coverage is always reported;
an empty scoring cohort has undefined metrics. A second comparison uses the
same segments for both 15-language models, excluding the union of their
English predictions. Macro scores average over target languages with retained
examples. Source-balanced and duration-weighted accuracy are also exported.

These are provisional directory-label metrics. Predicted English may be an
actual code-switch or a Pacific-language error, and true English can be missed.
With few source clips per language, many slices do not constitute many
independent speakers or videos; short/long extracts can also overlap. The report describes these limitations and a timestamped
review queue supports manual annotation.

- Expanded report and figures: `reports/youtube_validation_v3/held_out_evaluation.md`.
- Current expanded-corpus artifacts: `artifacts/youtube_validation_v3/`, including group/domain comparisons, full predictions and
  probabilities, model/source/VAD hashes, VAD boundaries, extracted WAVs,
  conditional and common-cohort metrics, per-source results and annotation CSVs.
- Runtime log: `artifacts/youtube_evaluation_v3.log`; phase: the artifact directory's
  `status.json`. Cached embeddings are fingerprinted by model and segment data.
- Use a fresh `--output` directory when changing source clips or VAD/minimum
  duration settings. Report regeneration reads the saved results without fitting.
- The earlier Pacific-only expanded corpus remains in `artifacts/youtube_validation_v2/`.
- The original seven-clip artifacts remain in `artifacts/youtube_validation/`;
  their report and figures are preserved in `reports/youtube_validation_previous_7clips/`.

## Pacific speaker and embedding study (historical)

Speaker-separated training is stopped. The commands below describe the historical
study; future LID experiments use the original recording-based splits.

```bash
uv run python scripts/run_speaker_study.py
uv run python scripts/progress.py --speakers
```

- Outputs: `artifacts/speaker_study/`; report and offline interactive explorers:
  `reports/speaker_study/`. The runner enforces 4 GiB host RAM with swap disabled.
- Stages: `extract`, `audit`, `visualize`, `train`, `evaluate`, `report`, or `all`.
  Sharded extraction and fingerprinted caches allow resumption.
- The revision-pinned public speaker model is downloaded anonymously on first run;
  subsequent model execution is offline. Original audio and experiment outputs remain intact.
- Automatic speaker-similarity groups are not verified identities. Cosine thresholds
  0.4–0.8 are audited; the primary 0.5 grouping links recordings and potential speakers.
- Retraining runs only if four partitions can each retain all three languages,
  an independent connected group, and at least 50 cuts per language. Otherwise the
  audit and visualizations finish and the report explains why training stopped.
- If feasible: compare uniform and speaker-balanced sampling on the same new split,
  with frozen and partial encoders, seed 42, original-language replay and FLEURS
  regression checks. Reuse the existing two recording-split baselines.
- UMAP covers Pacific radio and fixed external segments only, separately for the
  base and existing partial encoder. Clustering uses PCA-reduced embeddings rather
  than 2D plot coordinates. Interactive views include audio references.

## Sources

- [Pretrained model card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
- [Pretrained architecture](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/blob/main/hyperparams.yaml)
- [FLEURS dataset](https://huggingface.co/datasets/google/fleurs)
- [Temperature scaling and ECE](https://proceedings.mlr.press/v70/guo17a.html)

## Whisper tiny twelve-language comparison

The current study uses multilingual **Whisper tiny**, pinned to
`openai-whisper==20250625` and its registered SHA-256 checkpoint. It retains the
original 204,234 VoxLingua training clips, 23,607 recording-held-out checkpoint
selection clips, 4,172 FLEURS calibration clips and 8,519 FLEURS test clips,
including their original IDs and ordering. No speaker separation is used.

```bash
uv sync --locked
uv run python scripts/run_whisper12.py --background
uv run python scripts/progress.py --whisper
systemctl --user status lid-whisper12.service
```

The persistent user service runs sequential GPU jobs with 4 GiB host RAM,
zero swap, two CPU equivalents and two Torch/BLAS threads. Training has a
microbatch-two backward preflight, falls back to one if necessary, and enforces
a sub-7 GiB GPU budget. Checkpoints resume optimizer steps; feature and prediction
caches resume from immutable, fingerprinted shards. Changing the configuration,
model or cohort requires a fresh root. Manual service stop prevents automatic
restart; failures have bounded retries.

Approaches:

- **Filtering:** original first-token Whisper language logits, filtered to the
  twelve selected codes and renormalized with softmax. Report native-language
  Whisper as a separate reference. Canonical language columns are explicitly
  sorted by code to avoid tokenizer iteration-order changes between processes.
- **Frozen encoder:** concatenate mean and population standard deviation of real
  encoder positions, followed by LayerNorm and a 512-unit/twelve-output classifier.
  Padding positions are excluded from pooling. Match the existing SpeechBrain
  head's weighted-loss/full-epoch schedule; head architecture and initialization
  still differ between families.
- **Full encoder fine-tuning:** start from the best frozen head and pretrained
  encoder, update all encoder parameters plus the head. An additional genuine
  twelve-output SpeechBrain full-encoder model uses the same example draws and
  update budget; the historical fifteen-output partial-tail model is not used as
  an equivalent twelve-output reference. SpeechBrain BN running statistics stay
  fixed for microbatch stability.

All models receive at most ten real seconds of mono 16 kHz audio. Whisper adds
its required right-zero-padding to thirty seconds; it does not receive additional
real speech. Frontend calculation stays FP32 during AMP encoder training. Model
selection uses only VoxLingua validation NLL. Temperature fits use only FLEURS
validation, never FLEURS test or external clips.

FLEURS comparisons include accuracy, macro/weighted precision, recall and F1,
raw/calibrated ECE, NLL, Brier scores, out-of-set guesses and per-language confusion
matrices. Existing SpeechBrain results are recomputed and checked from archived
reference logits. The latest external VAD snapshot is shared exactly: Hindi/Thai
have recognition metrics, English-excluded and all-segment policies plus a shared
cohort. Pacific clips are **out-of-set diagnostics**, since the requested twelve
outputs cannot represent Samoan, Niuean or Tongan. No forced-choice accuracy is
presented as Pacific recognition performance.

After training, compare nested 3/5/10-second excerpts from matched real ten-second
speech windows and a ±12 dB, clipping-capped gain probe using only original
VoxLingua checkpoint-selection audio. Diagnostic CSVs retain source references.

- Configuration: `configs/whisper12.json`.
- Artifacts and progress: `artifacts/whisper_tiny12/`; log: `pipeline.log`.
- Evolving report: `reports/whisper_tiny12/whisper_comparison.md`.
- Real-model implementation checks: `artifacts/whisper_tiny12/implementation_preflight.json`.
- The separate `artifacts/whisper_tiny12_smoke/` cohorts verify implementation only;
  their deliberately tiny training results are excluded from the comparison.
- Stages: `prepare`, `cache_eval`, `filtered`, `cache_train`, `frozen`,
  `whisper_finetune`, `speechbrain_finetune`, `diagnostics`, `report`, or `all`.

All original SpeechBrain artifacts, previous validation outputs and stopped
speaker-split results are preserved.

### Completed Whisper export

The selected fine-tuned Whisper checkpoint is
`artifacts/whisper_tiny12/whisper_finetuned/model.pt` (epoch 8). Its standalone
SpeechBrain-compatible export is `artifacts/deployment/whisper_tiny12_finetuned/`,
with a single portable `whisper_tiny12_finetuned.tar.gz` archive alongside it.

```python
from speechbrain.inference.classifiers import EncoderClassifier
model = EncoderClassifier.from_hparams(
    source="artifacts/deployment/whisper_tiny12_finetuned",
    savedir="pretrained_models/whisper_tiny12_lid",
    run_opts={"device": "cuda"},
)
log_probs, score, index, labels = model.classify_file("audio.wav")
```

Keep `custom.py` and all model files together. The bundled backend reproduces the
Whisper frontend, automatic maximum-ten-second center crop, real-token pooling,
and trained twelve-language head. Calibration is already folded into classifier
weights. Runtime needs SpeechBrain and openai-whisper, but no project code or
pretrained model download. This is the LID encoder/classifier export; native
Whisper transcription loaders expect a different checkpoint with a text decoder.

The export was verified offline after relocation against all 8,519 FLEURS test
clips and 1,243 external validation segments, plus CPU, file and padded-batch API
checks. `verification.json`, `export.json`, `SHA256SUMS` and the archive checksum
record these checks. To reproduce the folder export use
`scripts/export_whisper12.py --output /a/fresh/directory`.

### Whisper tiny Pacific extension

The separate `configs/whisper15.json` study appends `sam`, `niu`, `ton` to the
original twelve labels. Both options start from the selected frozen Whisper12
classifier with identical seeded new output rows. `frozen` trains the classifier;
`finetuned` updates the entire Whisper audio encoder and classifier. Both use the
original recording-based VoxLingua/Pacific splits and the SpeechBrain15 balanced
sample order, 32,768 draws per epoch, effective batch 32, at most 20 epochs and
patience 5. Selection uses combined 15-class macro F1, then balanced NLL.

```bash
uv run python scripts/run_whisper15.py --background
uv run python scripts/progress.py --whisper15
```

Stages are `prepare`, `baselines`, `cache_all`, `frozen`, `finetuned`, `evaluate`,
`diagnostics`, `export`, `report`, or `all` (default). The runner enforces 4 GiB
host memory, zero swap and two CPU threads through systemd, shares the existing
single-GPU lock and limits GPU use below 7 GiB. Completed cache shards and training
state are resumable. After a failure, inspect `artifacts/whisper_tiny15/pipeline.log`,
resolve the error, clear the failed unit with
`systemctl --user reset-failed lid-whisper15.service`, and rerun the same command.
Configuration, source manifests, initial weights and implementation fingerprints
must match; changed inputs require fresh output directories.

Validation v4 freezes all 24 current source files into identical segments for
all twelve benchmark models. Its 1,336 segments include the 1,243 historical v3
segments and 93 added Niuean segments. Historical comparisons verify exact
decoded PCM samples and timestamps: float WAV creation timestamps can change
file hashes without changing samples. External clips remain evaluation-only;
joint temperature fitting uses only FLEURS validation and Pacific calibration.
All 15 outputs remain enabled in original-language regression comparisons.

On successful completion, the runner writes
`reports/whisper_tiny15/whisper_extension.md` and
`reports/youtube_validation_v4/held_out_evaluation.md`, plus detailed artifacts,
histories, provenance, duration/gain diagnostics and PNG/PDF/SVG figures. Both
deployment bundles under `artifacts/deployment/whisper_tiny15_{frozen,finetuned}`
must pass relocated offline checks against complete test/external predictions
before archives and checksums are finalized. Their bundled custom backend supports
`EncoderClassifier.from_hparams(...)` without a transcription decoder, with joint
temperature calibration folded once into the classifier. The study is complete
only after all training, evaluation, diagnostics, export verification and report
stages succeed; a running service does not indicate completed results.

### Focused Whisper Small15 study

```bash
uv run python scripts/run_whisper_small15.py --background
uv run python scripts/progress.py --whisper-small15
```

`configs/whisper_small15.json` uses multilingual Small and fresh output directories
`artifacts/whisper_small12` (frozen warm-start) and `artifacts/whisper_small15`
(paired fifteen-language study). The warm-start copies the original recording
manifests and trains a Small twelve-language classifier; it reuses no Tiny weights
or embeddings. Both Small15 variants start from this same selected classifier.
The focused scope excludes native/filtered Small controls and a fine-tuned
Small12 control. Comparisons with the frozen Small12 warm-start are explicitly
labelled, while completed Tiny and SpeechBrain controls are reused only after
matching cohort IDs, fingerprints, checkpoints and calibration provenance.

The resource preflight covers a full optimizer step and checkpoint save, trying
microbatch 2, then 1, then 1 with encoder gradient checkpointing if necessary.
The effective batch stays 32 and the original ten-second real-audio/thirty-second
Whisper preprocessing remains fixed. The runner uses separate stage processes
under one 4 GiB, zero-swap, two-CPU systemd unit and the shared single-GPU lock.
If a stage fails, inspect `artifacts/whisper_small15/pipeline.log`, resolve the
failure, clear `lid-whisper-small15.service` with `systemctl --user reset-failed`,
and rerun the same command. Verified cache shards and training state resume;
changed identities are rejected rather than silently reused.

Stages are `prepare_model`, `preflight`, `warm_cache`, `warm_train`,
`warm_evaluate`, `prepare`, `cache_all`, `frozen`, `finetuned`, `evaluate`,
`baselines`, `diagnostics`, `export`, `report`, or `all` (default).
Final self-contained HTML/PDF reports and all figures are written under
`reports/whisper_small15`; standalone calibrated deployment folders, archives,
loading examples and checksums use `artifacts/deployment/whisper_small15_*`.
Reports require completed test/external predictions and relocated offline export
verification. A running worker or implemented report builder is not a finished
Small experiment.

## Whisper Large v3 frozen study

```bash
uv run python scripts/run_whisper_large_v3_15.py --background
uv run python scripts/progress.py --whisper-large15
```

The Large v3 experiment trains a frozen twelve-language warm-start and expands
its classifier to fifteen languages using the original recording splits and
matched seed-42 balanced orders. It updates no encoder parameters. Feature
extraction uses float32, 128 mel channels, native thirty-second padding and the
same ten-second centre crop and real-token mean/std pooling as the Small study.
The encoder is loaded without the transcription decoder; training checkpoints
contain the classifier and a fingerprinted encoder reference, rather than a
duplicate encoder. Existing Tiny, Small and SpeechBrain artifacts are preserved.

Stages are `prepare_model`, `preflight`, `warm_cache`, `warm_train`,
`warm_evaluate`, `prepare`, `cache_all`, `frozen`, `evaluate`, `baselines`,
`diagnostics`, `export`, `report`, or `all`. The runner enforces 4 GiB host RAM,
zero swap, two CPU threads and the shared single-GPU lock. The actual-checkpoint
preflight tries inference batch two, then one; it verifies frozen encoder
weights and a classifier optimizer step. A failed stage leaves atomic cache
shards and resumable training state. Clear a failed service with
`systemctl --user reset-failed lid-whisper-large-v3-15.service`, then rerun the
same command. Configuration, model, frontend and data identity changes are
rejected rather than silently reused.

Outputs use `artifacts/whisper_large_v3_12` and
`artifacts/whisper_large_v3_15`. The full evaluation compares SpeechBrain partial,
Tiny frozen and Small frozen using verified matching test/external inputs and
their unchanged calibration temperatures. External audio remains evaluation-only.
Self-contained HTML/PDF analysis and PNG/PDF/SVG plots are written under
`reports/whisper_large_v3_15`. The verified offline bundle and archive use
`artifacts/deployment/whisper_large_v3_15_frozen`. The offline demo notebook
includes an optional configuration for this bundle once export completes.
Completion requires all training, evaluation, diagnostics, export verification
and report markers; a running cache stage is not a completed study.
