# Speaker-controlled LID reruns

Speaker labels are provisional automated labels derived from fixed WeSpeaker pilot groups: Samoan 8, Niuean 7 and Tongan 10. All source recordings and inferred global speaker IDs are kept together. Ambiguous cuts, conflicting recordings and conflicting duplicates are quarantined. This prevents overlap of the assigned IDs, but cannot certify zero overlap of real people. The original 12-language VoxLingua/FLEURS splits are unchanged.

## Split audit

Retained 43,663 of 44,822 cuts; excluded 1,159. Cut IDs, source recordings, file hashes and inferred speaker IDs have zero overlap across train, checkpoint selection, calibration and test.

| Partition | Samoan | Niuean | Tongan |
|---|---:|---:|---:|
| train | 11893 | 13884 | 10664 |
| selection | 854 | 219 | 318 |
| calibration | 136 | 403 | 882 |
| test | 1535 | 487 | 2388 |

## Fixed external validation

Same 428 external speech segments for every model, never used for checkpoint selection, temperature fitting or label/split selection. Folder labels are provisional. All-segment accuracy counts English predictions as errors; conditional accuracy reproduces the previous English-exclusion policy. The common cohort is shared by every currently completed model and can change until all four reruns finish.

| Model | All-segment accuracy | Macro F1 | Conditional accuracy | English exclusions | Common accuracy |
|---|---:|---:|---:|---:|---:|
| recording_uniform/frozen | 68.22% | 71.59% | 68.87% | 4 | 68.88% |
| recording_uniform/partial | 70.56% | 74.11% | 71.23% | 4 | 71.50% |
| disjoint_uniform/frozen | 47.66% | 48.49% | 48.23% | 5 | 48.22% |
| disjoint_balanced/frozen | 58.88% | 64.24% | 59.72% | 6 | 59.86% |
| disjoint_uniform/partial | 52.80% | 55.84% | 53.43% | 5 | 53.44% |

## Generalisation change versus the corresponding original model

- disjoint_uniform/frozen: -20.56 percentage points in fixed external all-segment accuracy.
- disjoint_balanced/frozen: -9.35 percentage points in fixed external all-segment accuracy.
- disjoint_uniform/partial: -17.76 percentage points in fixed external all-segment accuracy.

Completed reruns: 3/4. One training seed, identical initialization and maximum update budgets. The existing recording-based baselines used different amounts of data and different training/selection populations, so an external difference does not isolate the causal effect of speaker splitting. The two new sampling conditions share identical partitions.

Internal Pacific test populations differ from the original recording split; compare them as different difficulty levels. Old-language FLEURS results use the same test population and 15 outputs. Cross-language identities use a conservative threshold on reference medoids; true-person matches, unknown readers and multiple readers per recording can still be missed. Pretraining speaker overlap is not controlled.

Full metrics, per-language comparisons, split audit, excluded cuts, model fingerprints and checkpoints are in `artifacts/speaker_split`.

## Training stopped

The remaining speaker-balanced partial-encoder run was stopped at the user’s request. Three of four reruns completed external evaluation; the final run is incomplete and has no final external result. Completed results and resumable checkpoints are preserved. The worker service is stopped and will not restart automatically.
