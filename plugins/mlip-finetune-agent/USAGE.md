# ELoRA MLIP Fine-Tune Plugin Usage

This plugin uses ELoRA to fine-tune MACE-style MLIP models. It does not require users to clone the ELoRA repository manually. During setup, the plugin installs:

- `git+https://github.com/hyjwpk/ELoRA.git@main` for the ELoRA-modified `e3nn`
- `git+https://github.com/hyjwpk/ELoRA.git@MACE_ELoRA` for the ELoRA MACE CLI

## Prerequisites

- `uv`
- CUDA driver and a visible GPU for GPU training
- A Python environment with CUDA PyTorch, or permission to create a fresh uv environment
- Real model and dataset files, by default:
  - `/hdd/mlip-finetune/models/2024-01-07-mace-128-L2_epoch-199.model`
  - `/hdd/mlip-finetune/datasets/BOTNet-datasets/dataset_3BPA/train_300K.xyz`

## Setup

Recommended for this machine, using the existing `pytorch` conda environment:

```bash
ELORA_SETUP_MODE=existing \
ELORA_CONDA_ENV=pytorch \
ELORA_SKIP_TORCH=1 \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

If `pytorch` is already active:

```bash
conda activate pytorch
ELORA_SETUP_MODE=existing \
ELORA_SKIP_TORCH=1 \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

For an existing uv or Python virtual environment:

```bash
ELORA_SETUP_MODE=existing \
ELORA_PYTHON_BIN=/path/to/env/bin/python \
ELORA_SKIP_TORCH=1 \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

For a clean uv-managed environment:

```bash
ELORA_SETUP_MODE=fresh \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

Use `ELORA_FORCE_E3NN_INSTALL=1` or `ELORA_FORCE_MACE_INSTALL=1` when you want to reinstall the ELoRA branches even if compatible packages already appear to exist.

Setup also applies an idempotent compatibility patch to the installed ELoRA MACE package. The patch makes foundation model weight copying match parameters by name and skip LoRA-only parameters, which is needed when fine-tuning older MACE foundation weights under the ELoRA e3nn branch.

For unstable networks, increase retries:

```bash
ELORA_UV_RETRIES=8 \
ELORA_UV_RETRY_DELAY=30 \
ELORA_SETUP_MODE=existing \
ELORA_CONDA_ENV=pytorch \
ELORA_SKIP_TORCH=1 \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

## Workflow

1. `setup`

   Installs or validates the target ELoRA runtime. It checks Python, CUDA PyTorch, MACE CLIs, ELoRA markers in `e3nn`, the foundation model file, and dataset roots.

2. `data-prep`

   Reads real XYZ data, verifies energy and force keys, creates deterministic train/valid/test splits, and writes provenance under `mlip_runs/<run_id>/`.

3. `fine-tune`

   Runs target-environment `bin/mace_run_train` with ELoRA-modified MACE and explicit `--lora=True --lora_rank=16 --lora_alpha=16`. The default E2E smoke run uses one epoch on a small real-data subset.

4. `evaluate`

   Runs target-environment `bin/mace_eval_configs`, writes predictions, and records energy/force accuracy evidence.

5. `publish`

   Copies the trained model, config, command, dataset manifest, evaluation report, and release manifest to `mlip_releases/<run_id>/`.

## Real E2E Test

Run the full GPU/data/model test with the `pytorch` conda environment:

```bash
ELORA_SETUP_MODE=existing \
ELORA_CONDA_ENV=pytorch \
ELORA_SKIP_TORCH=1 \
python -m unittest tests.test_elora_gpu_e2e -v
```

The test creates:

- `mlip_runs/<run_id>/dataset/`
- `mlip_runs/<run_id>/models/<model_name>.model`
- `mlip_runs/<run_id>/results/predictions.xyz`
- `mlip_releases/<run_id>/`

## Static Tests

```bash
python -m unittest tests.test_mlip_finetune_plugin -v
```

These tests validate the plugin manifest, skill contracts, setup scripts, templates, and documentation.
