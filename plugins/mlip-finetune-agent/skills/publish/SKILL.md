---
name: MLIP-Finetune-Publish
description: Use when the user asks to publish, package, release, export, archive, or share a fine-tuned MLIP model and its training artifacts.
version: 0.1.0
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep
---

Package a fine-tuned MLIP model release.

## Goal

Create a release directory containing the ELoRA-trained model, configuration, command history, dataset provenance, evaluation metrics, ELoRA environment evidence, and usage notes. Do not upload externally unless the user explicitly requests a destination and credentials are configured.

## Inputs

- Final model path.
- Run directory, normally `mlip_runs/<run_id>/`.
- Evaluation report path.
- Dataset manifest path.
- Release name, defaulting to the model stem.

## Workflow

1. Verify the model file exists.
2. Verify the run directory contains `config.yaml`, `train-command.sh`, and `evaluation.md`.
3. Read `../../references/release-checklist.md`.
4. Create `mlip_releases/<release_name>/`.
5. Copy or reference:
   - final model file
   - `config.yaml`
   - `train-command.sh`
   - `dataset-manifest.md`
   - `evaluation.md`
   - ELoRA setup/check report if available
   - relevant training logs if available
6. Write `mlip_releases/<release_name>/release-manifest.md`.
7. Write a short `README.md` with intended system, dataset, metrics, and inference/evaluation command.
8. If the user asks for an archive, create `mlip_releases/<release_name>.tar.gz`.

## Release Rules

- Include original dataset sources and licenses when known.
- Include foundation model filename and source URL when known.
- Include active environment notes: `.venv-elora`, PyTorch/CUDA, MACE/ELoRA versions, `ELoRA@main`, `ELoRA@MACE_ELoRA`, and LoRA marker verification.
- Do not claim the model is production-ready unless evaluation metrics and intended use are documented.
- Do not publish to GitHub, Hugging Face, object storage, or a model registry without explicit user instruction.
