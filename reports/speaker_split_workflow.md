# Speaker-controlled LID experiment workflow

Run from the repository root:

```bash
.venv/bin/python -u scripts/run_speaker_split.py --stage all > artifacts/speaker_split.log 2>&1
.venv/bin/python scripts/progress.py --speaker-split
```

The full pipeline is resumable and isolated from the previous speaker study and original LID experiments. It enforces 4 GiB RAM, no swap, two CPU equivalents and two BLAS/Torch threads. A shared GPU process lock prevents simultaneous pilot/speaker-split jobs. Speaker extraction runs in the locked pyannote environment; LID training uses the original project environment.

`configs/speaker_split.json` fixes 8 Samoan, 7 Niuean and 10 Tongan provisional speaker groups. The pilot's 10-second WeSpeaker embeddings are clustered with average linkage. All 44,822 source cuts receive WeSpeaker embeddings using their real center crops of at most ten seconds. The closest reference group is selected by mean cosine similarity to its constituent reference cuts, with minimum similarity 0.35 and minimum winning margin 0.05. These thresholds are fixed before external scoring. Shorter cuts are used at their actual lengths without padding. Low-confidence cuts are excluded. If confident cuts within one source recording support conflicting global identities, the entire recording is quarantined. Exact duplicate conflicts are also quarantined.

Cross-language reference medoids with cosine similarity at least 0.75 are assigned one global identity before partition allocation. This is a heuristic audit, not proof that all cross-language matches have been identified. No external validation audio is used to set labels, thresholds, training splits, checkpoints or calibration temperatures.

Whole inferred identities are allocated deterministically to train, checkpoint selection, calibration and test. Cut proportions target 80/5/5/10, but the small number and uneven sizes of inferred speakers can substantially change the achieved proportions. Every partition must contain at least 50 cuts per Pacific language. An explicit audit rejects any shared cut ID, recording ID, file hash or inferred global speaker ID across partitions. Label and exclusion CSVs preserve original recording references and partitions. The original 12-language training/FLEURS data retain their previous splits; their speaker independence is not established here.

Four new runs use the existing initialization, optimizer, maximum update budget and checkpoint criterion: classifier-only (`frozen`) and encoder-tail (`partial`) training, each with uniform and speaker-balanced Pacific sampling. Class balancing and original-language replay remain unchanged. Both classifier-only runs complete before encoder-tail training begins. Evaluation retains all fifteen LID outputs. Temperature calibration uses the dedicated speaker-held-out calibration partition plus the original FLEURS calibration set.

The primary generalisation comparison uses the same 428 external speech segments and existing recording-based baseline models. Report both all-segment accuracy (including English predictions as errors), macro F1, per-language metrics, conditional English-excluded metrics and a shared cohort. Internal Pacific test scores use a different population from the original recording split and cannot establish an improvement by direct subtraction. Different training populations and quarantined cuts also prevent attributing external changes solely to speaker splitting. The two new sampling conditions share exactly the same partitions. One training seed is used.

Outputs are under `artifacts/speaker_split`; the evolving results report is `reports/speaker_split/speaker_generalisation.md`. `--stage extract`, `split`, `train`, `evaluate` and `report` allow stage-specific resumption. The report updates after each completed model. The previous speaker study and original model checkpoints are preserved. Provisional speaker independence does not establish independence of real people or eliminate speaker overlap from encoder pretraining.

The current long-running experiment is hosted by the persistent user service `lid-speaker-split.service`, independent of this chat session. It uses the same resource limits and resumes existing checkpoints. Failures are retried after 30 seconds, with at most three starts per hour. Inspect it with `systemctl --user status lid-speaker-split.service` or the progress command above. The service runs both remaining encoder-tail conditions and refreshes the comparison report after each model completes.
