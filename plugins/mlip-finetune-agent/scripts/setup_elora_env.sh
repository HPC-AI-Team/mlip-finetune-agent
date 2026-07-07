#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

UV_BIN="${UV_BIN:-uv}"
VENV_DIR="${ELORA_VENV_DIR:-$ROOT_DIR/.venv-elora}"
SETUP_MODE="${ELORA_SETUP_MODE:-auto}"
CONDA_BIN="${CONDA_BIN:-conda}"
PYTHON_VERSION="${ELORA_PYTHON:-3.10}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_PACKAGES="${ELORA_TORCH_PACKAGES:-torch torchvision torchaudio}"
UV_RETRIES="${ELORA_UV_RETRIES:-5}"
UV_RETRY_DELAY="${ELORA_UV_RETRY_DELAY:-20}"
COMMON_PACKAGES=(
  numpy scipy matplotlib ase opt_einsum prettytable pandas pyyaml h5py torch-ema
  matscipy torchmetrics python-hostlist configargparse GitPython lmdb orjson tqdm
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

uv_pip_install() {
  local attempt=1
  while true; do
    if "$UV_BIN" pip install --python "$PYTHON_BIN" "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$UV_RETRIES" ]; then
      return 1
    fi
    echo "uv pip install failed on attempt $attempt/$UV_RETRIES; retrying in ${UV_RETRY_DELAY}s" >&2
    sleep "$UV_RETRY_DELAY"
    attempt=$((attempt + 1))
  done
}

python_has_torch() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
}

python_has_elora_e3nn() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import inspect
import e3nn.nn._fc as fc_mod
import e3nn.o3._linear as linear_mod
import e3nn.o3._tensor_product._tensor_product as tp_mod
linear_src = inspect.getsource(linear_mod.Linear)
tp_src = inspect.getsource(tp_mod.TensorProduct)
fc_src = inspect.getsource(fc_mod._Layer)
ok = (
    "LoRA_weight" in linear_src
    and "ELoRA_weights" in linear_src
    and "LoRA_weight" in tp_src
    and "ELoRA_weights" in tp_src
    and "LoRA_weight" in fc_src
    and "self.r = 16" in linear_src
    and "self.alpha = 16" in linear_src
)
raise SystemExit(0 if ok else 1)
PY
}

resolve_existing_python() {
  if [ -n "${ELORA_PYTHON_BIN:-}" ]; then
    PYTHON_BIN="$ELORA_PYTHON_BIN"
  elif [ -n "${ELORA_CONDA_ENV:-}" ]; then
    PYTHON_BIN="$("$CONDA_BIN" run -n "$ELORA_CONDA_ENV" python -c 'import sys; print(sys.executable)')"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    echo "Using active conda environment at $CONDA_PREFIX"
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  elif [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  else
    die "existing setup mode needs ELORA_PYTHON_BIN, ELORA_CONDA_ENV, an active conda env, or ELORA_VENV_DIR with bin/python"
  fi
  [ -x "$PYTHON_BIN" ] || die "target Python is not executable: $PYTHON_BIN"
}

resolve_fresh_python() {
  if [ -x "$VENV_DIR/bin/python" ]; then
    echo "Reusing existing uv environment at $VENV_DIR"
  else
    echo "Creating fresh ELoRA uv environment at $VENV_DIR"
    "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
}

case "$SETUP_MODE" in
  auto)
    if [ -n "${ELORA_PYTHON_BIN:-}" ] || [ -n "${ELORA_CONDA_ENV:-}" ]; then
      echo "ELORA_SETUP_MODE=auto selected existing environment"
      resolve_existing_python
    else
      echo "ELORA_SETUP_MODE=auto selected fresh uv environment"
      resolve_fresh_python
    fi
    ;;
  existing)
    echo "ELORA_SETUP_MODE=existing selected"
    resolve_existing_python
    ;;
  fresh)
    echo "ELORA_SETUP_MODE=fresh selected"
    resolve_fresh_python
    ;;
  *)
    die "ELORA_SETUP_MODE must be one of: auto, existing, fresh"
    ;;
esac

PYTHON_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
echo "Target Python: $PYTHON_BIN"
echo "Target prefix: $PYTHON_PREFIX"

if [ "${ELORA_SKIP_TORCH:-0}" = "1" ]; then
  echo "ELORA_SKIP_TORCH=1; checking existing torch"
  python_has_torch || die "ELORA_SKIP_TORCH=1 but target Python cannot import torch"
elif [ "${ELORA_FORCE_TORCH_INSTALL:-0}" = "1" ] || ! python_has_torch; then
  echo "Installing CUDA PyTorch from $PYTORCH_INDEX_URL"
  uv_pip_install --index-url "$PYTORCH_INDEX_URL" $TORCH_PACKAGES
else
  echo "Torch is already importable in target environment; skipping large torch download"
fi

echo "Installing common MACE dependencies"
uv_pip_install "${COMMON_PACKAGES[@]}"

if [ "${ELORA_FORCE_E3NN_INSTALL:-0}" = "1" ] || ! python_has_elora_e3nn; then
  echo "Installing ELoRA e3nn from hyjwpk/ELoRA@main"
  uv_pip_install "git+https://github.com/hyjwpk/ELoRA.git@main"
else
  echo "ELoRA-modified e3nn is already installed; skipping e3nn git install"
fi

if [ "${ELORA_FORCE_MACE_INSTALL:-0}" = "1" ] || [ ! -x "$PYTHON_PREFIX/bin/mace_run_train" ]; then
  echo "Installing ELoRA MACE from hyjwpk/ELoRA@MACE_ELoRA"
  uv_pip_install "git+https://github.com/hyjwpk/ELoRA.git@MACE_ELoRA"
else
  echo "MACE CLI already exists in target environment; set ELORA_FORCE_MACE_INSTALL=1 to reinstall MACE_ELoRA"
fi

echo "Applying ELoRA MACE compatibility patch"
"$PYTHON_BIN" "$SCRIPT_DIR/patch_elora_mace.py"

echo "Validating ELoRA environment"
CHECK_ARGS=(
  --expected-python "$PYTHON_BIN"
  --expected-prefix "$PYTHON_PREFIX"
)
if [ "${ELORA_ALLOW_MISSING_ASSETS:-0}" = "1" ]; then
  CHECK_ARGS+=(--allow-missing-assets)
fi
if [ "${ELORA_ALLOW_CPU:-0}" = "1" ]; then
  CHECK_ARGS+=(--allow-cpu)
fi
"$PYTHON_BIN" "$SCRIPT_DIR/check_elora_env.py" "${CHECK_ARGS[@]}"

echo "ELoRA environment is ready: $PYTHON_PREFIX"
