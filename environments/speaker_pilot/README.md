# Matched speaker pilot

From the repository root:

```bash
uv sync --project environments/speaker_pilot --locked
environments/speaker_pilot/.venv/bin/python -u scripts/run_speaker_pilot.py --stage all > artifacts/speaker_pilot.log 2>&1
```

The runner enters a systemd user scope with `MemoryMax=4G`, `MemorySwapMax=0`, `CPUQuota=200%`, and two Torch/BLAS threads. A nonblocking process lock prevents concurrent pilot jobs. Microbatch size is two. GPU usage is measured after each batch; reaching 7 GiB stops extraction. Run only one GPU workload at a time.

Stages are `prepare`, `validate`, `sample`, `ecapa`, `wespeaker`, `verify`, `analyze`, and `report`. The default is `all`. Each stage can be restarted. Completed crop caches are reused only when their input/configuration/model/environment identity matches. A changed configuration requires a fresh `--root`. Model revisions are pinned in the configuration or `MODEL_REVISIONS` defaults, written to `models.json`, and checked on every restart; model files are hashed. ECAPA is pinned in the configuration. WeSpeaker defaults to immutable revision `837717ddb9ff5507820346191109dc79c958d614`.

Sampling orders all original recordings and cuts by a seed-42 hash, independently of manifest order, pooling all four original partitions. It selects the first 100 eligible recordings per language and the first two eligible cuts per recording. Short cuts and any crop with less than 70% VAD speech are rejected. Each accepted recording has the same inclusion opportunity in the deterministic shuffle; eligibility constraints can bias the sample. Counts by partition and recording references remain in the sample manifest. Failure to obtain the full sample stops the run and writes coverage and rejection diagnostics.

Every crop is cut from one cached real 10-second center window. No padding, repetition, speech concatenation, or extra gain is used. Stereo sources are averaged to mono, resampling uses the repository polyphase loader, and both encoders receive identical 16 kHz float32 waveforms. `sample.json` records source and cached audio hashes, source offsets, per-crop boundaries, and speech coverage. `early3` and `late3` never overlap. No profile averaging or downstream training occurs.

Analysis clusters cuts separately within language and duration. One cluster is a degenerate reference; 2–15 clusters are explored without imposing ten. Resampling uses ten seeds and one cut per recording, comparing labels to the full-data labels restricted to those cuts using adjusted Rand index (ARI). Duration ARI compares each shorter crop with 10 seconds. PCA coordinates are for viewing only; CSV files preserve full recording/cut references. Cross-language top-three nearest candidates are separate diagnostics. All cluster assignments remain provisional and uncertain.

The generated report is `reports/speaker_pilot/speaker_embedding_pilot.md`. The previous `artifacts/speaker_study` and its report are preserved.

The `verify` stage reloads all original center windows, rejects duplicate decoded audio, checks all crop boundaries and VAD coverage, and verifies encoder sample IDs, cache identities, unit norms and reproducibility records. Results are written to `acceptance.json`. Stage resource files describe the current invocation; a resumed cached stage does not repeat the original GPU work. The original extraction measurements remain in the run log.
