#!/usr/bin/env python
import argparse
from pathlib import Path

from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(description="Prepare a tiny real XYZ split for ELoRA smoke tests.")
    parser.add_argument("--source-xyz", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-size", type=int, default=4)
    parser.add_argument("--valid-size", type=int, default=2)
    parser.add_argument("--test-size", type=int, default=2)
    args = parser.parse_args()

    source = Path(args.source_xyz)
    run_dir = Path(args.run_dir)
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    total = args.train_size + args.valid_size + args.test_size
    frames = read(source, index=f":{total}")
    if len(frames) != total:
        raise SystemExit(f"expected {total} frames, got {len(frames)} from {source}")

    for atoms in frames:
        atoms.info["REF_energy"] = float(atoms.get_potential_energy())
        atoms.arrays["REF_forces"] = atoms.get_forces()
        atoms.calc = None

    train = frames[: args.train_size]
    valid = frames[args.train_size : args.train_size + args.valid_size]
    test = frames[args.train_size + args.valid_size :]

    write(dataset_dir / "train.xyz", train, format="extxyz")
    write(dataset_dir / "valid.xyz", valid, format="extxyz")
    write(dataset_dir / "test.xyz", test, format="extxyz")

    (run_dir / "dataset-manifest.md").write_text(
        "\n".join(
            [
                "# Dataset Manifest",
                "",
                "- Dataset: BOTNet real XYZ subset",
                f"- Source XYZ: `{source}`",
                f"- Train configurations: {len(train)}",
                f"- Validation configurations: {len(valid)}",
                f"- Test configurations: {len(test)}",
                "- Energy key: `REF_energy`",
                "- Forces key: `REF_forces`",
                "- Source data is copied, not modified.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(dataset_dir)


if __name__ == "__main__":
    main()
