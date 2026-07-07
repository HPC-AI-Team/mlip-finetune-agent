# ELoRA MLIP Fine-Tune Agent

Repo-local Codex plugin for MLIP fine-tuning with ELoRA low-rank adaptation. The main path is not plain PyPI `mace-torch`; it installs the ELoRA-modified `e3nn` and MACE branches with `uv`.

## Commands

| Command | Purpose |
| --- | --- |
| `/mlip-finetune:setup` | Install missing ELoRA pieces into an existing environment, or create a fresh `.venv-elora` with `uv`. |
| `/mlip-finetune:data-prep` | Inspect real XYZ datasets, create deterministic splits, and record provenance. |
| `/mlip-finetune:fine-tune` | Generate or run target-environment `bin/mace_run_train` ELoRA fine-tuning commands. |
| `/mlip-finetune:evaluate` | Evaluate an ELoRA-tuned model with target-environment `bin/mace_eval_configs`. |
| `/mlip-finetune:publish` | Package ELoRA model artifacts, metrics, provenance, and environment evidence. |
| `/mlip-finetune:help` | Show plugin usage and workflow guidance. |

## Setup

The setup workflow is implemented by:

```bash
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

The setup script supports two installation paths. Use the existing-environment path when downloads are slow and the machine already has CUDA PyTorch installed.

Users do not need to manually download or clone the ELoRA repository. The setup script installs the required ELoRA branches directly with `uv pip install` from Git.

Existing conda environment:

```bash
ELORA_SETUP_MODE=existing \
ELORA_CONDA_ENV=pytorch \
ELORA_SKIP_TORCH=1 \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

Already active conda environment:

```bash
conda activate pytorch
ELORA_SETUP_MODE=existing \
ELORA_SKIP_TORCH=1 \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

Existing Python or uv environment:

```bash
ELORA_SETUP_MODE=existing \
ELORA_PYTHON_BIN=/path/to/env/bin/python \
ELORA_SKIP_TORCH=1 \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

Fresh uv environment:

```bash
ELORA_SETUP_MODE=fresh \
bash plugins/mlip-finetune-agent/scripts/setup_elora_env.sh
```

`ELORA_SETUP_MODE=auto` is the default. It uses an existing environment when `ELORA_PYTHON_BIN` or `ELORA_CONDA_ENV` is set; otherwise it creates or reuses `.venv-elora`.

The script installs:

- CUDA PyTorch from `https://download.pytorch.org/whl/cu126`
- ELoRA e3nn from `git+https://github.com/hyjwpk/ELoRA.git@main`
- ELoRA MACE from `git+https://github.com/hyjwpk/ELoRA.git@MACE_ELoRA`

For slow or unstable networks, tune `ELORA_UV_RETRIES` and `ELORA_UV_RETRY_DELAY`; defaults are `5` attempts and `20` seconds.

The script also applies an idempotent ELoRA/MACE compatibility patch so foundation-weight copying skips LoRA-only parameters. It finishes by running `check_elora_env.py`, which verifies CUDA, MACE CLIs, ELoRA LoRA markers in `e3nn`, model file validity, and dataset availability.

## Default Local Paths

- Models: `/hdd/mlip-finetune/models`
- Datasets: `/hdd/mlip-finetune/datasets`
- Run artifacts: `mlip_runs/<run_id>/`
- Release artifacts: `mlip_releases/<run_id>/`
- ELoRA environment: existing conda/Python env, or `.venv-elora` in fresh uv mode
- ELoRA implementation: rank `r=16`, alpha `16` in the modified e3nn modules

The plugin skills verify paths before using them. They do not launch long training jobs unless the user explicitly asks to run training.

## Tests

Static plugin tests:

```bash
python -m unittest tests.test_mlip_finetune_plugin -v
```

Real GPU/data/model E2E test using the `pytorch` conda environment:

```bash
ELORA_SETUP_MODE=existing \
ELORA_CONDA_ENV=pytorch \
ELORA_SKIP_TORCH=1 \
python -m unittest tests.test_elora_gpu_e2e -v
```

See `USAGE.md` for the full user workflow.
