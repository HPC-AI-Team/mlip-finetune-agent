---
name: MLIP-Finetune-Setup
description: Use when the user asks to install, set up, check, validate, or diagnose the uv ELoRA environment for MLIP fine-tuning.
version: 0.1.0
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep
---

Install or validate the ELoRA MLIP fine-tuning environment.

## Goal

Support two setup paths: install missing ELoRA pieces into a user-specified existing conda/uv/Python environment, or create `.venv-elora` from scratch with `uv`. Both paths must verify that the target runtime is ELoRA-modified MACE, not plain PyPI `mace-torch`.

## Install Flow

When the user asks to set up/install the plugin environment, choose one of these modes.

### Existing Environment

Use this mode when the user already has CUDA PyTorch or a slow network. Install only missing ELoRA/MACE pieces into the specified environment:

```bash
ELORA_SETUP_MODE=existing \
ELORA_CONDA_ENV=<conda-env-name> \
ELORA_SKIP_TORCH=1 \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

If the target conda environment is already active, this is also valid:

```bash
conda activate <conda-env-name>
ELORA_SETUP_MODE=existing \
ELORA_SKIP_TORCH=1 \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

or:

```bash
ELORA_SETUP_MODE=existing \
ELORA_PYTHON_BIN=/path/to/env/bin/python \
ELORA_SKIP_TORCH=1 \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

### Fresh uv Environment

Use this mode for a clean, reproducible environment:

```bash
ELORA_SETUP_MODE=fresh \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

### Auto Mode

`ELORA_SETUP_MODE=auto` is the default. It uses existing mode when `ELORA_PYTHON_BIN` or `ELORA_CONDA_ENV` is set; otherwise it creates or reuses `.venv-elora`.

The user does not need to manually clone ELoRA. The script installs the required Git branches with uv.

The script must:

1. Resolve the target Python from `ELORA_PYTHON_BIN`, `ELORA_CONDA_ENV`, or `.venv-elora`.
2. Create `.venv-elora` with `uv venv` only in fresh mode when needed.
3. Install CUDA PyTorch using `PYTORCH_INDEX_URL`, default `https://download.pytorch.org/whl/cu126`, unless torch is already importable or `ELORA_SKIP_TORCH=1`.
4. Install common MACE dependencies with `uv pip --python <target-python>`.
5. Install ELoRA e3nn:
   ```bash
   uv pip install --python <target-python> "git+https://github.com/hyjwpk/ELoRA.git@main"
   ```
6. Install ELoRA MACE:
   ```bash
   uv pip install --python <target-python> "git+https://github.com/hyjwpk/ELoRA.git@MACE_ELoRA"
   ```
7. Run `plugins/mlip-finetune-agent/scripts/patch_elora_mace.py` with the resolved Python. The patch is idempotent and makes ELoRA/MACE foundation-weight copying skip LoRA-only parameters.
8. Run `plugins/mlip-finetune-agent/scripts/check_elora_env.py` with the resolved Python and prefix.

Use `ELORA_UV_RETRIES` and `ELORA_UV_RETRY_DELAY` for poor network conditions. The setup script retries `uv pip install` commands by default.

## Checks

1. Read local sources of truth:
   - `docs/test-model-and-dataset.md`
   - `ELoRA/README.md`
   - `plugins/mlip-finetune-agent/references/training-templates.md`
2. Check expected local paths:
   - `/hdd/mlip-finetune/models`
   - `/hdd/mlip-finetune/datasets`
3. Check uv and ELoRA runtime availability:
   - `uv --version`
   - `<target-python> --version`
   - `<target-python> -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`
   - `<target-prefix>/bin/mace_run_train --help`
   - `<target-prefix>/bin/mace_eval_configs --help`
   - `nvidia-smi` when CUDA is expected
4. Verify ELoRA is really installed:
   - `e3nn.o3.Linear` source contains `LoRA_weight`
   - tensor product source contains `ELoRA_weights`
   - MACE CLI resolves under the target Python prefix
5. Check candidate model files under `/hdd/mlip-finetune/models`:
   - `2024-01-07-mace-128-L2_epoch-199.model`
   - `MACE-OFF23_medium.model`
6. Verify model files are real weights, not HTML downloads or Git LFS pointer text. Use size checks, first-byte checks, and a lightweight `torch.load(..., map_location="cpu")` check.
7. Check candidate datasets under `/hdd/mlip-finetune/datasets`, especially `3BPA` and `AcAc`.

## Environment Interpretation

- ELoRA low-rank tuning is the main path and must use the validated target environment.
- Baseline/full-parameter MACE is optional comparison only.
- ELoRA and baseline share the same command shape; the installed environment determines whether low-rank ELoRA modules are active.
- Current ELoRA code hard-codes rank `r=16` and alpha `16` inside modified e3nn layers.

## Output

Report:

- Whether each required path exists.
- Which model artifacts were found.
- Whether each model artifact is loadable by PyTorch/MACE.
- Which dataset candidates were found.
- Whether `mace_run_train` and `mace_eval_configs` are available under the target environment.
- Whether ELoRA LoRA markers were found in installed e3nn.
- Whether CUDA is visible.
- The recommended next command, normally `/mlip-finetune:data-prep`.

When asked to save the report, write `mlip_runs/setup-report.md`.

## Failure Behavior

If uv, CUDA PyTorch, ELoRA e3nn, or MACE_ELoRA installation fails, stop and report the failing command. Do not silently fall back to plain `mace-torch`.
