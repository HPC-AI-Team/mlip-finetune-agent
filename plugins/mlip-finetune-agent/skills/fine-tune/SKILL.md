---
name: MLIP-Finetune-Fine-Tune
description: Use when the user asks to fine-tune with ELoRA, train an MLIP with low-rank adaptation, generate an ELoRA MACE command, or start an ELoRA MLIP job.
version: 0.1.0
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep
---

Generate or run an ELoRA MACE fine-tuning command.

## Goal

Create a reproducible ELoRA fine-tuning run directory from dataset inputs and the templates in `../../references/training-templates.md`.

## Required Inputs

- `run_id`, or create one like `elora-YYYYMMDD-HHMMSS-<dataset>-<mode>`.
- Dataset paths: `train_file`, plus either `valid_file` or `valid_fraction`; optional `test_file`.
- Mode: `organic` or `inorganic`.
- Foundation model path.
- Device, default `cuda` when available, otherwise `cpu`.
- Whether to generate the command only or actually launch training.

## Defaults

- Organic mode:
  - Foundation model: `/hdd/mlip-finetune/models/MACE-OFF23_medium.model`
  - Model: `MACE`
  - `r_max=5.0`
  - `max_L=1`
  - `max_num_epochs=500`
  - `default_dtype=float64`
- Inorganic mode:
  - Foundation model: `/hdd/mlip-finetune/models/2024-01-07-mace-128-L2_epoch-199.model`
  - Model: `ScaleShiftMACE`
  - `r_max=6.0`
  - `max_L=2`
  - `max_num_epochs=200`
  - `interaction_first=RealAgnosticResidualInteractionBlock`
  - `interaction=RealAgnosticResidualInteractionBlock`
- Shared defaults:
  - `E0s=average`
  - `loss=ef`
  - `energy_weight=1`
  - `forces_weight=1000`
  - `lr=0.005`
  - `weight_decay=1e-8`
  - `batch_size=5`
  - `valid_batch_size=5`
  - `scheduler_patience=5`
  - `ema`
  - `ema_decay=0.995`
  - `error_table=TotalRMSE`
  - `seed=123`
  - `clip_grad=100`
  - `save_cpu`
- ELoRA implementation:
  - executable: target environment `bin/mace_run_train`, default `.venv-elora/bin/mace_run_train`
  - e3nn LoRA rank: `16`
  - e3nn LoRA alpha: `16`
  - plain PyPI `mace-torch` is not an acceptable substitute for an ELoRA run.

## Workflow

1. Read the run config from `mlip_runs/<run_id>/config.yaml` if present.
2. Verify dataset and foundation model paths exist.
3. Verify target environment `bin/mace_run_train` is available and `check_elora_env.py` passes.
4. Pick the matching template from `../../references/training-templates.md`.
5. Write:
   - `mlip_runs/<run_id>/config.yaml`
   - `mlip_runs/<run_id>/train-command.sh`
   - `mlip_runs/<run_id>/README.md`
6. Do not start training unless the user explicitly asks to run it.
7. If starting training, state the expected output directory and log file before launching.

## Safety

Training can run for hours and consume GPU memory. Command generation is the default. Treat "train it", "run fine-tuning", or "start the job" as explicit permission to launch; otherwise stop after writing the command and ask the user to review it.

## Next Step

After training finishes, recommend `/mlip-finetune:evaluate` with the final `.model` path and the test XYZ path.
