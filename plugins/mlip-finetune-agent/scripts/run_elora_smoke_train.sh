#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

VENV_DIR="${ELORA_VENV_DIR:-$ROOT_DIR/.venv-elora}"
CONDA_BIN="${CONDA_BIN:-conda}"
RUN_ID="${ELORA_RUN_ID:-elora-real-integration-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${ELORA_RUN_DIR:-$ROOT_DIR/mlip_runs/$RUN_ID}"
MODEL_NAME="${ELORA_MODEL_NAME:-elora_real_integration}"
SOURCE_XYZ="${ELORA_SOURCE_XYZ:-/hdd/mlip-finetune/datasets/BOTNet-datasets/dataset_3BPA/train_300K.xyz}"
FOUNDATION_MODEL="${ELORA_FOUNDATION_MODEL:-/hdd/mlip-finetune/models/2024-01-07-mace-128-L2_epoch-199.model}"
DEVICE="${ELORA_DEVICE:-cuda}"
E0S="${ELORA_E0S:-foundation}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [ -n "${ELORA_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$ELORA_PYTHON_BIN"
elif [ -n "${ELORA_CONDA_ENV:-}" ]; then
  PYTHON_BIN="$("$CONDA_BIN" run -n "$ELORA_CONDA_ENV" python -c 'import sys; print(sys.executable)')"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [ -x "$VENV_DIR/bin/python" ]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
else
  die "No ELoRA Python found. Run setup first, or set ELORA_PYTHON_BIN / ELORA_CONDA_ENV / active conda env / ELORA_VENV_DIR."
fi

[ -x "$PYTHON_BIN" ] || die "target Python is not executable: $PYTHON_BIN"
PYTHON_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
MACE_RUN_TRAIN="${ELORA_MACE_RUN_TRAIN:-$PYTHON_PREFIX/bin/mace_run_train}"
MACE_EVAL_CONFIGS="${ELORA_MACE_EVAL_CONFIGS:-$PYTHON_PREFIX/bin/mace_eval_configs}"
[ -x "$MACE_RUN_TRAIN" ] || die "mace_run_train is not executable: $MACE_RUN_TRAIN"
[ -x "$MACE_EVAL_CONFIGS" ] || die "mace_eval_configs is not executable: $MACE_EVAL_CONFIGS"

"$PYTHON_BIN" "$SCRIPT_DIR/check_elora_env.py" \
  --expected-python "$PYTHON_BIN" \
  --expected-prefix "$PYTHON_PREFIX" \
  --foundation-model "$FOUNDATION_MODEL" \
  --dataset-root "$(dirname "$(dirname "$SOURCE_XYZ")")"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_real_xyz_subset.py" \
  --source-xyz "$SOURCE_XYZ" \
  --run-dir "$RUN_DIR"

mkdir -p "$RUN_DIR/models" "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/checkpoints"

cat > "$RUN_DIR/config.yaml" <<EOF
run_id: "$RUN_ID"
mode: "elora-inorganic-smoke"
stack: "ELoRA"
source_xyz: "$SOURCE_XYZ"
foundation_model: "$FOUNDATION_MODEL"
environment: "$PYTHON_PREFIX"
python: "$PYTHON_BIN"
python_prefix: "$PYTHON_PREFIX"
mace_run_train: "$MACE_RUN_TRAIN"
energy_key: "REF_energy"
forces_key: "REF_forces"
E0s: "$E0S"
device: "$DEVICE"
max_num_epochs: 1
lora_rank: 16
lora_alpha: 16
EOF

CMD=(
  "$MACE_RUN_TRAIN"
  "--name=$MODEL_NAME"
  "--work_dir=$RUN_DIR"
  "--model_dir=$RUN_DIR/models"
  "--log_dir=$RUN_DIR/logs"
  "--results_dir=$RUN_DIR/results"
  "--checkpoints_dir=$RUN_DIR/checkpoints"
  "--train_file=$RUN_DIR/dataset/train.xyz"
  "--valid_file=$RUN_DIR/dataset/valid.xyz"
  "--test_file=$RUN_DIR/dataset/test.xyz"
  "--energy_key=REF_energy"
  "--forces_key=REF_forces"
  "--E0s=$E0S"
  "--foundation_model=$FOUNDATION_MODEL"
  "--foundation_model_elements=True"
  "--foundation_filter_elements=False"
  "--multiheads_finetuning=False"
  "--model=ScaleShiftMACE"
  "--interaction_first=RealAgnosticResidualInteractionBlock"
  "--interaction=RealAgnosticResidualInteractionBlock"
  "--num_interactions=2"
  "--correlation=3"
  "--max_ell=3"
  "--r_max=6.0"
  "--max_L=2"
  "--num_channels=128"
  "--num_radial_basis=10"
  "--MLP_irreps=16x0e"
  "--scaling=rms_forces_scaling"
  "--loss=ef"
  "--energy_weight=1"
  "--forces_weight=1000"
  "--lora=True"
  "--lora_rank=16"
  "--lora_alpha=16"
  "--amsgrad"
  "--lr=0.001"
  "--weight_decay=1e-8"
  "--batch_size=1"
  "--valid_batch_size=1"
  "--lr_factor=0.8"
  "--scheduler_patience=1"
  "--max_num_epochs=1"
  "--error_table=TotalRMSE"
  "--device=$DEVICE"
  "--seed=123"
  "--clip_grad=100"
  "--save_cpu"
  "--num_workers=0"
  "--plot=False"
)

{
  printf '#!/usr/bin/env bash\n'
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$RUN_DIR/train-command.sh"

"${CMD[@]}" 2>&1 | tee "$RUN_DIR/elora-train-output.log"

"$MACE_EVAL_CONFIGS" \
  --configs="$RUN_DIR/dataset/test.xyz" \
  --model="$RUN_DIR/models/$MODEL_NAME.model" \
  --output="$RUN_DIR/results/predictions.xyz" \
  --device="$DEVICE" \
  --default_dtype=float64 \
  --batch_size=1

LOG_FILE="$(find "$RUN_DIR/logs" -maxdepth 1 -name "${MODEL_NAME}_run-*.log" | sort | tail -n 1)"
{
  echo "# ELoRA MLIP Evaluation"
  echo
  echo "- Run ID: \`$RUN_ID\`"
  echo "- Model: \`$RUN_DIR/models/$MODEL_NAME.model\`"
  echo "- Dataset: \`$SOURCE_XYZ\`"
  echo "- Foundation model: \`$FOUNDATION_MODEL\`"
  echo "- Python: \`$PYTHON_BIN\`"
  echo "- Environment: \`$PYTHON_PREFIX\`"
  echo "- Device: \`$DEVICE\`"
  echo "- LoRA rank/alpha: \`16 / 16\`"
  echo
  echo "## Training Log Tail"
  echo
  echo '```text'
  tail -n 40 "$LOG_FILE"
  echo '```'
} > "$RUN_DIR/evaluation.md"

RELEASE_DIR="$ROOT_DIR/mlip_releases/$RUN_ID"
mkdir -p "$RELEASE_DIR"
cp "$RUN_DIR/models/$MODEL_NAME.model" "$RELEASE_DIR/"
cp "$RUN_DIR/config.yaml" "$RELEASE_DIR/"
cp "$RUN_DIR/train-command.sh" "$RELEASE_DIR/"
cp "$RUN_DIR/dataset-manifest.md" "$RELEASE_DIR/"
cp "$RUN_DIR/evaluation.md" "$RELEASE_DIR/"
cat > "$RELEASE_DIR/release-manifest.md" <<EOF
# ELoRA Release Manifest

- Run ID: \`$RUN_ID\`
- Foundation model: \`$FOUNDATION_MODEL\`
- Fine-tuned model: \`$RUN_DIR/models/$MODEL_NAME.model\`
- Dataset: \`$SOURCE_XYZ\`
- Python: \`$PYTHON_BIN\`
- Environment: \`$PYTHON_PREFIX\`
- Device: \`$DEVICE\`
- ELoRA refs: \`ELoRA@main\`, \`ELoRA@MACE_ELoRA\`
- LoRA rank/alpha: \`16 / 16\`
EOF

echo "$RUN_DIR"
