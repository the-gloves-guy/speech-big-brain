# Matched speaker embedding pilot

All assignments are provisional and retained as uncertain. This automated pilot has no labelled speaker truth. Recording relationships and silhouette scores are diagnostics, not identity accuracy. Roughly ten readers per language is a soft prior; no identity count is imposed.

The previous connected-component study remains unchanged. Its chaining failure did not establish that the embeddings could not cluster. This pilot uses average and complete linkage with cosine distance.

Sample: 600 cuts, seed 42, two cuts per recording. Every cut provides the same unpadded 10-second center window to both encoders, nested 3/5/10-second crops and two disjoint 3-second excerpts. Every evaluated crop passes 70% VAD coverage. Ordinary pauses remain. VAD does not identify speaker changes.

## Comparison

Best silhouette setting per model, language and duration is shown for navigation only. Selection over many settings can inflate this score. Cohesion must be judged relative to separation, duration agreement and recording resamples.

| Model | Language | Seconds | Linkage | k | Silhouette | Cohesion−between p95 | Resample ARI |
|---|---|---:|---|---:|---:|---:|---:|
| ecapa | sam | 3 | average | 8 | 0.430 | 0.290 | 0.999 |
| ecapa | sam | 5 | average | 8 | 0.533 | 0.363 | 0.998 |
| ecapa | sam | 10 | average | 8 | 0.630 | 0.423 | 1.000 |
| ecapa | niu | 3 | average | 6 | 0.394 | 0.197 | 1.000 |
| ecapa | niu | 5 | average | 6 | 0.486 | 0.278 | 1.000 |
| ecapa | niu | 10 | average | 7 | 0.591 | 0.349 | 1.000 |
| ecapa | ton | 3 | average | 7 | 0.338 | 0.201 | 0.978 |
| ecapa | ton | 5 | average | 12 | 0.420 | 0.260 | 0.973 |
| ecapa | ton | 10 | average | 12 | 0.519 | 0.308 | 0.991 |
| wespeaker | sam | 3 | average | 7 | 0.490 | 0.350 | 0.985 |
| wespeaker | sam | 5 | average | 7 | 0.571 | 0.425 | 1.000 |
| wespeaker | sam | 10 | average | 7 | 0.643 | 0.468 | 1.000 |
| wespeaker | niu | 3 | average | 7 | 0.462 | 0.233 | 1.000 |
| wespeaker | niu | 5 | average | 7 | 0.547 | 0.282 | 1.000 |
| wespeaker | niu | 10 | average | 7 | 0.639 | 0.335 | 1.000 |
| wespeaker | ton | 3 | average | 10 | 0.396 | 0.258 | 0.997 |
| wespeaker | ton | 5 | average | 10 | 0.478 | 0.309 | 0.997 |
| wespeaker | ton | 10 | average | 10 | 0.565 | 0.373 | 1.000 |

## Ten-cluster reference

This fixed reference compares duration and encoder without selecting a different k. It is not an assertion of ten identities. Gap is median within-cluster cosine minus between-cluster p95. ARI compares partitions, not true people.

| Model | Language | Seconds | Linkage | Silhouette | Gap | Resample ARI | Min/max cuts |
|---|---|---:|---|---:|---:|---:|---|
| ecapa | sam | 3 | average | 0.382 | 0.264 | 1.000 | 2/104 |
| ecapa | sam | 3 | complete | 0.382 | 0.264 | 0.881 | 2/104 |
| ecapa | sam | 5 | average | 0.496 | 0.331 | 1.000 | 2/104 |
| ecapa | sam | 5 | complete | 0.496 | 0.331 | 1.000 | 2/104 |
| ecapa | sam | 10 | average | 0.610 | 0.387 | 1.000 | 2/104 |
| ecapa | sam | 10 | complete | 0.610 | 0.387 | 1.000 | 2/104 |
| ecapa | niu | 3 | average | 0.284 | 0.099 | 0.933 | 1/88 |
| ecapa | niu | 3 | complete | 0.239 | 0.060 | 0.742 | 2/76 |
| ecapa | niu | 5 | average | 0.357 | 0.113 | 0.973 | 2/89 |
| ecapa | niu | 5 | complete | 0.253 | 0.017 | 0.704 | 2/64 |
| ecapa | niu | 10 | average | 0.438 | 0.117 | 0.978 | 2/88 |
| ecapa | niu | 10 | complete | 0.329 | 0.035 | 0.735 | 2/65 |
| ecapa | ton | 3 | average | 0.336 | 0.207 | 0.947 | 2/50 |
| ecapa | ton | 3 | complete | 0.330 | 0.208 | 0.907 | 2/50 |
| ecapa | ton | 5 | average | 0.417 | 0.262 | 0.959 | 2/50 |
| ecapa | ton | 5 | complete | 0.409 | 0.260 | 0.939 | 2/50 |
| ecapa | ton | 10 | average | 0.517 | 0.311 | 0.966 | 2/50 |
| ecapa | ton | 10 | complete | 0.500 | 0.304 | 0.968 | 2/50 |
| wespeaker | sam | 3 | average | 0.447 | 0.270 | 1.000 | 2/104 |
| wespeaker | sam | 3 | complete | 0.330 | 0.132 | 0.849 | 2/93 |
| wespeaker | sam | 5 | average | 0.545 | 0.317 | 1.000 | 2/104 |
| wespeaker | sam | 5 | complete | 0.545 | 0.317 | 0.940 | 2/104 |
| wespeaker | sam | 10 | average | 0.641 | 0.342 | 1.000 | 2/104 |
| wespeaker | sam | 10 | complete | 0.641 | 0.342 | 1.000 | 2/104 |
| wespeaker | niu | 3 | average | 0.315 | 0.075 | 0.866 | 1/92 |
| wespeaker | niu | 3 | complete | 0.274 | 0.046 | 0.810 | 2/86 |
| wespeaker | niu | 5 | average | 0.377 | 0.070 | 0.941 | 2/88 |
| wespeaker | niu | 5 | complete | 0.359 | 0.065 | 0.877 | 2/89 |
| wespeaker | niu | 10 | average | 0.413 | 0.084 | 0.950 | 2/85 |
| wespeaker | niu | 10 | complete | 0.302 | 0.042 | 0.843 | 2/86 |
| wespeaker | ton | 3 | average | 0.396 | 0.258 | 0.997 | 2/50 |
| wespeaker | ton | 3 | complete | 0.379 | 0.257 | 0.960 | 2/50 |
| wespeaker | ton | 5 | average | 0.478 | 0.309 | 0.997 | 2/50 |
| wespeaker | ton | 5 | complete | 0.458 | 0.310 | 0.989 | 2/50 |
| wespeaker | ton | 10 | average | 0.565 | 0.373 | 1.000 | 2/50 |
| wespeaker | ton | 10 | complete | 0.521 | 0.370 | 0.989 | 2/50 |

At the average-linkage ten-cluster reference, 10-second crops improve silhouette and the separation gap over 3-second crops for both models in all three languages. WeSpeaker has higher silhouette and gap than ECAPA for Samoan and Tongan at 3 seconds, while ECAPA has the larger Samoan gap at 10 seconds. Niuean favors ECAPA on both measures at 10 seconds. The model comparison therefore depends on language, duration and k.

## Duration agreement

Average linkage at k=10; shorter-duration labels compared with 10 seconds on the same cuts.

| Model | Language | Seconds | ARI vs 10s |
|---|---|---:|---:|
| ecapa | sam | 3 | 1.000 |
| ecapa | sam | 5 | 1.000 |
| ecapa | niu | 3 | 0.966 |
| ecapa | niu | 5 | 0.964 |
| ecapa | ton | 3 | 0.971 |
| ecapa | ton | 5 | 0.995 |
| wespeaker | sam | 3 | 1.000 |
| wespeaker | sam | 5 | 1.000 |
| wespeaker | niu | 3 | 0.906 |
| wespeaker | niu | 5 | 0.917 |
| wespeaker | ton | 3 | 1.000 |
| wespeaker | ton | 5 | 1.000 |

## Sampling and consistency

Rejection events: fewer_than_two_eligible_cuts: 118; insufficient_vad_coverage: 75; short_audio: 1260. Recording rejection events are separate from cut rejections; eligible orphan cuts in a recording with fewer than two eligible cuts are excluded. Sampling stops after 100 accepted recordings per language, so these are not corpus-wide rejection rates.

| Language | Original partition | Cuts |
|---|---|---:|
| niu | calibration | 16 |
| niu | selection | 10 |
| niu | test | 24 |
| niu | train | 150 |
| sam | calibration | 10 |
| sam | selection | 12 |
| sam | test | 22 |
| sam | train | 156 |
| ton | calibration | 8 |
| ton | selection | 20 |
| ton | test | 22 |
| ton | train | 150 |

| Model | Language | Within-cut 3s median | Same-recording 10s median | Across-recordings 10s median |
|---|---|---:|---:|---:|
| ecapa | sam | 0.683 | 0.874 | 0.248 |
| ecapa | niu | 0.645 | 0.841 | 0.323 |
| ecapa | ton | 0.661 | 0.857 | 0.215 |
| wespeaker | sam | 0.735 | 0.893 | 0.219 |
| wespeaker | niu | 0.735 | 0.881 | 0.399 |
| wespeaker | ton | 0.718 | 0.874 | 0.191 |

Same-recording similarities are generally high, supporting consistency as a working assumption for much of this sample. This does not establish one reader per recording: inspect low-tail pairs and unstable assignments in the CSV artifacts. Across-recording pairs include unknown same-person pairs, and all medians can conceal mixed speakers.

## Interpretation

For sam, wespeaker at 10 seconds with average linkage and k=7 has the highest observed silhouette (0.643). This is a candidate for further investigation, not a verified reader inventory. Its median within-cluster similarity minus the between-cluster 95th percentile is 0.468.

For niu, wespeaker at 10 seconds with average linkage and k=7 has the highest observed silhouette (0.639). This is a candidate for further investigation, not a verified reader inventory. Its median within-cluster similarity minus the between-cluster 95th percentile is 0.335.

For ton, wespeaker at 10 seconds with average linkage and k=10 has the highest observed silhouette (0.565). This is a candidate for further investigation, not a verified reader inventory. Its median within-cluster similarity minus the between-cluster 95th percentile is 0.373.

Higher overall similarity alone is not an improvement. Low within-cut consistency can reflect speaker changes, channel effects or encoder variability. Same-recording cuts can contain different speakers; different recordings can share a speaker. Cross-language nearest matches are listed separately and do not certify independence.

No speaker profiles are averaged, no full-corpus identities are assigned and no LID models are retrained. Stable clusters and plausible reader counts cannot guarantee zero true speaker overlap.

## Artifacts

Configuration, immutable model revisions and hashes, audio hashes, rejected candidates, partition references, recording coverage and resource checks: `artifacts/speaker_pilot`.

Full comparison, all uncertain assignments, similarity quantiles, duration/resample stability, PCA points with recording references and cross-language candidates: `artifacts/speaker_pilot/analysis`.

PCA is a projection for inspection; distances in the plot are not the original cosine distances.

## Reproducibility

| Encoder | Repository | Immutable revision |
|---|---|---|
| ecapa | speechbrain/spkrec-ecapa-voxceleb | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` |
| wespeaker | pyannote/wespeaker-voxceleb-resnet34-LM | `837717ddb9ff5507820346191109dc79c958d614` |

The isolated uv lock fixes pyannote.audio 3.3.2, Torch/torchaudio 2.5.1+cu121 and NumPy 1.26.4; the complete installed environment passed dependency checks and actual GPU inference. The kernel scope enforces 4 GiB RAM, zero swap and two CPU equivalents; Torch/BLAS threads are limited to two. Batch size is two. The original run log records device usage after every extraction batch. Stage resource files describe their current invocation, which may reuse earlier caches.

Acceptance checks reloaded all 600 source center windows and confirmed exact equality, unique decoded audio, matching sample IDs and crop boundaries, reproducibility records, finite embeddings and unit norms. `acceptance.json` records the result. No padding or repetition creates longer windows.

![ecapa stability](../../artifacts/speaker_pilot/analysis/stability_ecapa.png)

![ecapa sam similarity](../../artifacts/speaker_pilot/analysis/similarities_ecapa_sam.png)

![ecapa niu similarity](../../artifacts/speaker_pilot/analysis/similarities_ecapa_niu.png)

![ecapa ton similarity](../../artifacts/speaker_pilot/analysis/similarities_ecapa_ton.png)

![wespeaker stability](../../artifacts/speaker_pilot/analysis/stability_wespeaker.png)

![wespeaker sam similarity](../../artifacts/speaker_pilot/analysis/similarities_wespeaker_sam.png)

![wespeaker niu similarity](../../artifacts/speaker_pilot/analysis/similarities_wespeaker_niu.png)

![wespeaker ton similarity](../../artifacts/speaker_pilot/analysis/similarities_wespeaker_ton.png)
