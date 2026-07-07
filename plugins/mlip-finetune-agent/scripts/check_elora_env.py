#!/usr/bin/env python
import argparse
import inspect
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/hdd/mlip-finetune/models/2024-01-07-mace-128-L2_epoch-199.model")
DEFAULT_DATASET_ROOT = Path("/hdd/mlip-finetune/datasets")


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def resolve_env_cli(prefix, name):
    candidate = prefix / "bin" / name
    if candidate.is_file():
        return candidate
    found = shutil.which(name)
    return Path(found) if found is not None else None


def check_model_file(path):
    if not path.exists():
        fail(f"foundation model not found: {path}")
    if path.stat().st_size < 10 * 1024 * 1024:
        fail(f"foundation model is too small to be a real MACE weight file: {path}")
    head = path.read_bytes()[:256]
    lowered = head.lower()
    if b"<!doctype html" in lowered or b"<html" in lowered:
        fail(f"foundation model is an HTML page, not a torch model: {path}")
    if b"version https://git-lfs.github.com/spec" in lowered:
        fail(f"foundation model is a Git LFS pointer, not a torch model: {path}")

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch

    torch.load(path, map_location="cpu")


def check_elora_markers():
    import e3nn
    import e3nn.nn._fc as fc_mod
    import e3nn.o3._linear as linear_mod
    import e3nn.o3._tensor_product._tensor_product as tp_mod

    linear_src = inspect.getsource(linear_mod.Linear)
    tp_src = inspect.getsource(tp_mod.TensorProduct)
    fc_src = inspect.getsource(fc_mod._Layer)
    required = {
        "Linear.LoRA_weight": "LoRA_weight" in linear_src,
        "Linear.ELoRA_weights": "ELoRA_weights" in linear_src,
        "TensorProduct.LoRA_weight": "LoRA_weight" in tp_src,
        "TensorProduct.ELoRA_weights": "ELoRA_weights" in tp_src,
        "FC.LoRA_weight": "LoRA_weight" in fc_src,
        "rank_16": "self.r = 16" in linear_src and "self.r = 16" in tp_src,
        "alpha_16": "self.alpha = 16" in linear_src and "self.alpha = 16" in tp_src,
    }
    missing = [name for name, ok in required.items() if not ok]
    if missing:
        fail("installed e3nn is not the ELoRA-modified version; missing " + ", ".join(missing))
    return e3nn


def main():
    parser = argparse.ArgumentParser(description="Validate an ELoRA MLIP Python environment.")
    parser.add_argument("--foundation-model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--expected-python")
    parser.add_argument("--expected-prefix")
    parser.add_argument("--allow-missing-assets", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    prefix = Path(sys.prefix).resolve()
    if args.expected_python is not None:
        expected_python = Path(args.expected_python).resolve()
        actual_python = Path(sys.executable).resolve()
        if actual_python != expected_python:
            fail(f"expected Python {expected_python}, got {actual_python}")
    if args.expected_prefix is not None:
        expected_prefix = Path(args.expected_prefix).resolve()
        if prefix != expected_prefix:
            fail(f"expected Python prefix {expected_prefix}, got {prefix}")

    train_cli = resolve_env_cli(prefix, "mace_run_train")
    eval_cli = resolve_env_cli(prefix, "mace_eval_configs")
    if train_cli is None:
        fail("mace_run_train is not available in the target environment")
    if eval_cli is None:
        fail("mace_eval_configs is not available in the target environment")
    if not str(train_cli.resolve()).startswith(str(prefix)):
        fail(f"mace_run_train does not come from target environment {prefix}: {train_cli}")
    if not str(eval_cli.resolve()).startswith(str(prefix)):
        fail(f"mace_eval_configs does not come from target environment {prefix}: {eval_cli}")

    import ase
    import mace
    import torch

    e3nn = check_elora_markers()

    if not args.allow_cpu and not torch.cuda.is_available():
        fail("CUDA is not available")

    if not args.allow_missing_assets:
        check_model_file(Path(args.foundation_model))
        dataset_root = Path(args.dataset_root)
        if not dataset_root.exists():
            fail(f"dataset root not found: {dataset_root}")

    print("ELoRA environment: ok")
    print(f"python: {sys.executable}")
    print(f"prefix: {prefix}")
    print(f"torch: {torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"mace: {getattr(mace, '__version__', 'unknown')} {mace.__file__}")
    print(f"e3nn: {getattr(e3nn, '__version__', 'unknown')} {e3nn.__file__}")
    print(f"ase: {ase.__version__}")
    print("ELoRA e3nn markers: LoRA_weight, ELoRA_weights, r=16, alpha=16")
    print(f"mace_run_train: {train_cli}")
    print(f"mace_eval_configs: {eval_cli}")


if __name__ == "__main__":
    main()
