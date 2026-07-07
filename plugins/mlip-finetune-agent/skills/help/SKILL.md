---
name: MLIP-Finetune-Help
description: Use when the user asks for ELoRA MLIP plugin help, available commands, supported workflow, or how to use the ELoRA fine-tuning plugin.
version: 0.1.0
---

Show concise usage help for the ELoRA MLIP Fine-Tune Agent plugin.

## Overview

This plugin guides Codex through ELoRA low-rank MLIP fine-tuning. It can install missing ELoRA pieces into a user-specified existing environment or create an uv-managed `.venv-elora`, then verifies ELoRA-modified `e3nn` and MACE, prepares run artifacts, generates commands, evaluates results, and packages releases. Plain PyPI `mace-torch` is not the main path.

## Available Commands

| Command | Description |
| --- | --- |
| `/mlip-finetune:setup` | Install into an existing environment or create `.venv-elora` with uv, then validate ELoRA. |
| `/mlip-finetune:data-prep` | Inspect datasets, validate XYZ metadata, plan or create train/valid/test splits, and record provenance. |
| `/mlip-finetune:fine-tune` | Generate organic or inorganic target-environment `bin/mace_run_train` ELoRA commands. |
| `/mlip-finetune:evaluate` | Run or prepare accuracy evaluation and write an energy/force RMSE report. |
| `/mlip-finetune:publish` | Package model, config, metrics, dataset provenance, and command history into a release directory. |
| `/mlip-finetune:help` | Show this help. |

## Recommended Workflow

1. Run `/mlip-finetune:setup` to install and verify the ELoRA target environment.
2. Run `/mlip-finetune:data-prep` for the target dataset.
3. Run `/mlip-finetune:fine-tune` to generate the training command.
4. Start training only after the command and paths are reviewed.
5. Run `/mlip-finetune:evaluate` on the final model and test set.
6. Run `/mlip-finetune:publish` to package the model release.

## Important Defaults

- Environment: existing `ELORA_PYTHON_BIN` / `ELORA_CONDA_ENV`, or fresh `.venv-elora`
- Users do not need to clone ELoRA manually; setup installs the Git branches with uv.
- ELoRA sources: `git+https://github.com/hyjwpk/ELoRA.git@main` and `git+https://github.com/hyjwpk/ELoRA.git@MACE_ELoRA`
- Model cache: `/hdd/mlip-finetune/models`
- Dataset cache: `/hdd/mlip-finetune/datasets`
- Recommended initial datasets: `3BPA` and `AcAc`
- Run artifacts: `mlip_runs/<run_id>/`
- Release artifacts: `mlip_releases/<model_name>/`
- ELoRA rank/alpha in current code: `r=16`, `alpha=16`

## References

Use these plugin references when answering detailed workflow questions:

- `../../references/datasets.md`
- `../../references/training-templates.md`
- `../../references/evaluation-report-template.md`
- `../../references/release-checklist.md`
