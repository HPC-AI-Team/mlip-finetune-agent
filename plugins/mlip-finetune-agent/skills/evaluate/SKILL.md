---
name: MLIP-Finetune-Evaluate
description: Use when the user asks to evaluate, test, benchmark, compare accuracy, calculate RMSE, or summarize metrics for a fine-tuned MACE/ELoRA MLIP model.
version: 0.1.0
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep
---

Evaluate a fine-tuned MLIP model and write an accuracy report.

## Goal

Measure model accuracy on a held-out XYZ file and record energy/force metrics in `mlip_runs/<run_id>/evaluation.md`.

## Required Inputs

- Trained model path, normally a `.model` file.
- Test or validation XYZ path.
- `run_id` or output directory.
- Device, defaulting to the device used during training when known.

## Workflow

1. Verify target environment `bin/mace_eval_configs` is available.
2. Verify the model and test XYZ paths exist.
3. Inspect the training logs for MACE `TotalRMSE` tables. Prefer official training/test tables when available.
4. For standalone evaluation, prepare:
   ```bash
   <target-prefix>/bin/mace_eval_configs \
       --configs="<test.xyz>" \
       --model="<model.model>" \
       --output="mlip_runs/<run_id>/eval/predictions.xyz" \
       --device="<device>"
   ```
5. If reference energy/force keys are present, compute RMSE from the reference and predicted properties. Inspect keys first; common datasets differ in names.
6. Write `mlip_runs/<run_id>/evaluation.md` using `../../references/evaluation-report-template.md`.
7. If comparing baseline and ELoRA runs, report metrics side by side with the same test split and clearly label baseline as optional comparison.

## Metrics

Report at minimum:

- Energy RMSE, preferably meV/atom.
- Force RMSE, preferably meV/A.
- Number of evaluated configurations.
- Number of atoms or atom count range.
- Dataset split and source.
- Model file and command used.
- ELoRA environment path and LoRA marker verification.

## Failure Behavior

If reference properties cannot be identified, do not invent metrics. Save the prediction command and state which XYZ keys were found. Ask the user to identify the reference energy and force keys before computing RMSE.
