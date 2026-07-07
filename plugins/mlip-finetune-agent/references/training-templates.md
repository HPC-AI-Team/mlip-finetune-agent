# ELoRA Training Templates

These templates come from `ELoRA/README.md` and must be run with the validated target ELoRA environment. The command shape matches MACE, but the executable must come from the target environment, such as `<target-prefix>/bin/mace_run_train` or fresh-mode `.venv-elora/bin/mace_run_train`, where `e3nn` and MACE are installed from ELoRA branches.

Current ELoRA implementation facts from the local `ELoRA/` code:

- `e3nn.o3.Linear` adds `LoRA_weight`
- `e3nn.o3.TensorProduct` reports `ELoRA_weights`
- `e3nn.nn.FullyConnectedNet` layers add `LoRA_weight`
- LoRA rank `r=16`
- LoRA alpha `16`

## Inorganic Template

```bash
.venv-elora/bin/mace_run_train \
    --name="<run_name>" \
    --work_dir="mlip_runs/<run_id>" \
    --model_dir="mlip_runs/<run_id>/models" \
    --log_dir="mlip_runs/<run_id>/logs" \
    --results_dir="mlip_runs/<run_id>/results" \
    --checkpoints_dir="mlip_runs/<run_id>/checkpoints" \
    --train_file="mlip_runs/<run_id>/dataset/train.xyz" \
    --valid_file="mlip_runs/<run_id>/dataset/valid.xyz" \
    --test_file="mlip_runs/<run_id>/dataset/test.xyz" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --E0s="average" \
    --foundation_model="/hdd/mlip-finetune/models/2024-01-07-mace-128-L2_epoch-199.model" \
    --foundation_model_elements=True \
    --foundation_filter_elements=False \
    --multiheads_finetuning=False \
    --model="ScaleShiftMACE" \
    --interaction_first="RealAgnosticResidualInteractionBlock" \
    --interaction="RealAgnosticResidualInteractionBlock" \
    --num_interactions=2 \
    --correlation=3 \
    --max_ell=3 \
    --r_max=6.0 \
    --max_L=2 \
    --num_channels=128 \
    --num_radial_basis=10 \
    --MLP_irreps="16x0e" \
    --scaling="rms_forces_scaling" \
    --loss="ef" \
    --energy_weight=1 \
    --forces_weight=1000 \
    --lora=True \
    --lora_rank=16 \
    --lora_alpha=16 \
    --amsgrad \
    --lr=0.005 \
    --weight_decay=1e-8 \
    --batch_size=5 \
    --valid_batch_size=5 \
    --lr_factor=0.8 \
    --scheduler_patience=5 \
    --ema \
    --ema_decay=0.995 \
    --max_num_epochs=200 \
    --error_table="TotalRMSE" \
    --device="<device>" \
    --seed=123 \
    --clip_grad=100 \
    --save_cpu \
    --num_workers=0 \
    --plot=False
```

## Organic Template

```bash
.venv-elora/bin/mace_run_train \
    --name="<run_name>" \
    --work_dir="mlip_runs/<run_id>" \
    --model_dir="mlip_runs/<run_id>/models" \
    --log_dir="mlip_runs/<run_id>/logs" \
    --results_dir="mlip_runs/<run_id>/results" \
    --checkpoints_dir="mlip_runs/<run_id>/checkpoints" \
    --train_file="mlip_runs/<run_id>/dataset/train.xyz" \
    --valid_fraction=0.1 \
    --test_file="mlip_runs/<run_id>/dataset/test.xyz" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --E0s="average" \
    --foundation_model="/hdd/mlip-finetune/models/MACE-OFF23_medium.model" \
    --foundation_model_elements=True \
    --foundation_filter_elements=False \
    --multiheads_finetuning=False \
    --model="MACE" \
    --loss="ef" \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=1 \
    --correlation=3 \
    --r_max=5.0 \
    --lr=0.005 \
    --forces_weight=1000 \
    --energy_weight=1 \
    --lora=True \
    --lora_rank=16 \
    --lora_alpha=16 \
    --weight_decay=1e-8 \
    --clip_grad=100 \
    --batch_size=5 \
    --valid_batch_size=5 \
    --max_num_epochs=500 \
    --scheduler_patience=5 \
    --ema \
    --ema_decay=0.995 \
    --error_table="TotalRMSE" \
    --default_dtype="float64" \
    --device="<device>" \
    --seed=123 \
    --save_cpu \
    --num_workers=0 \
    --plot=False
```

## Run Config Shape

Use this YAML shape for `mlip_runs/<run_id>/config.yaml`:

```yaml
run_id: "elora-<run_id>"
mode: "organic"
dataset:
  name: "<dataset>"
  train_file: "mlip_runs/<run_id>/dataset/train.xyz"
  valid_file: "mlip_runs/<run_id>/dataset/valid.xyz"
  valid_fraction: 0.1
  test_file: "mlip_runs/<run_id>/dataset/test.xyz"
  energy_key: "REF_energy"
  forces_key: "REF_forces"
model:
  foundation_model: "/hdd/mlip-finetune/models/MACE-OFF23_medium.model"
  output_name: "elora_<dataset>"
training:
  executable: "<target-prefix>/bin/mace_run_train"
  device: "cuda"
  seed: 123
  batch_size: 5
  max_num_epochs: 500
  error_table: "TotalRMSE"
  lora: true
  lora_rank: 16
  lora_alpha: 16
environment:
  python_prefix: "<target-prefix>"
  default_fresh_venv: ".venv-elora"
  stack: "mace_elora"
  e3nn_source: "git+https://github.com/hyjwpk/ELoRA.git@main"
  mace_source: "git+https://github.com/hyjwpk/ELoRA.git@MACE_ELoRA"
  lora_rank: 16
  lora_alpha: 16
```
