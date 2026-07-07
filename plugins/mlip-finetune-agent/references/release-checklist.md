# ELoRA Release Checklist

Use this checklist before packaging an ELoRA fine-tuned MLIP model.

## Required Files

- Final `.model` file.
- `config.yaml`.
- `train-command.sh`.
- `dataset-manifest.md`.
- `evaluation.md`.
- ELoRA setup/check report.
- Training logs when available.
- `README.md` for intended use.
- `release-manifest.md` for traceability.

## Required Metadata

- Release name and date.
- Model architecture and mode: ELoRA MACE or ELoRA ScaleShiftMACE.
- uv environment path and Python version.
- ELoRA source refs: `ELoRA@main` and `ELoRA@MACE_ELoRA`.
- ELoRA marker evidence: `LoRA_weight`, `ELoRA_weights`, rank `16`, alpha `16`.
- Foundation model filename and source URL.
- Dataset name, source URL, split counts, and split seed.
- Energy/force reference keys.
- Evaluation metrics and test split.
- Python, PyTorch, CUDA, MACE, and ELoRA versions when discoverable.
- Known limitations and intended application domain.

## External Publishing

Do not upload to a remote registry by default. If the user explicitly requests GitHub, Hugging Face, object storage, or another registry, first verify credentials and confirm the exact target repository or bucket.
