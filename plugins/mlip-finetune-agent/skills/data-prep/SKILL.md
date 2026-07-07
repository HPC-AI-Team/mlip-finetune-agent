---
name: MLIP-Finetune-Data-Prep
description: Use when the user asks to prepare, inspect, validate, split, clean, or document datasets for MACE/ELoRA MLIP fine-tuning.
version: 0.1.0
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep
---

Prepare dataset inputs for an MLIP fine-tuning run.

## Goal

Turn a user-selected dataset into a documented MACE-compatible training input plan. Preserve source datasets and write all derived artifacts under `mlip_runs/<run_id>/`.

## Inputs To Discover

- Dataset name or path.
- Whether train/valid/test files already exist.
- Whether the target system is organic or inorganic.
- Available reference properties in XYZ metadata, especially energy, forces, stress, cell, and pbc.
- Desired split policy. Default to train/valid/test with seed `123`, validation fraction `0.1`, and no synthetic test split unless the user requests one or a test file is available.

## Workflow

1. Read `../../references/datasets.md` and the local `docs/test-model-and-dataset.md`.
2. Locate candidate data under `/hdd/mlip-finetune/datasets` unless the user provides a path.
3. Inspect file names, sizes, and extensions. Prefer `.xyz` or extended XYZ for MACE.
4. If ASE is installed, inspect the first few frames to identify arrays and info keys.
5. If a single source XYZ needs splitting, write derived files only under `mlip_runs/<run_id>/dataset/`.
6. Record provenance in `mlip_runs/<run_id>/dataset-manifest.md`.
7. Record the intended training file paths in `mlip_runs/<run_id>/config.yaml`.

## Dataset Rules

- Never edit or overwrite source data under `/hdd/mlip-finetune/datasets`.
- Keep generated splits deterministic with seed `123`.
- If multiple energy or force keys are present, inspect and report them instead of silently choosing.
- If stress is available and the target application needs elastic or EOS behavior, note that stress-weighted training may be needed.
- If pbc/cell metadata is missing for inorganic periodic systems, warn before training.

## Expected Artifacts

- `mlip_runs/<run_id>/dataset-manifest.md`
- `mlip_runs/<run_id>/config.yaml`
- Optional derived files:
  - `mlip_runs/<run_id>/dataset/train.xyz`
  - `mlip_runs/<run_id>/dataset/valid.xyz`
  - `mlip_runs/<run_id>/dataset/test.xyz`

## Next Step

After data preparation, recommend `/mlip-finetune:fine-tune` with the generated `mlip_runs/<run_id>/config.yaml`.
