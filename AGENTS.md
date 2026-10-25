# Experiment preferences

- Use the original recording-based training datasets and splits for all future LID experiments: `artifacts/full_run` for the original twelve languages and `artifacts/extension15/new_{train,selection,calibration,test}.jsonl` for the Pacific extension.
- Do not use speaker-separated manifests, inferred speaker labels, or speaker-balanced sampling for future training unless the user explicitly changes this instruction. The speaker-split study is historical and its worker must remain stopped.
- The canonical 15-language models are `artifacts/extension15/frozen` and `artifacts/extension15/partial`. Validation should use these original-split models, plus the original/filtered/fine-tuned twelve-language controls where appropriate.
- Preserve previous experiments. Use a fresh output directory when validation audio or preprocessing changes; do not fit models, checkpoints or calibration temperatures on the external validation clips.
