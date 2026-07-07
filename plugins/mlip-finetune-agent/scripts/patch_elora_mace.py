#!/usr/bin/env python
from pathlib import Path


MARKER = "ELoRA compatibility patch: name-matched foundation copy"

OLD_BLOCK = """        for (_, param_1), (_, param_2) in zip(
            model.interactions[i].conv_tp_weights.named_parameters(),
            model_foundations.interactions[i].conv_tp_weights.named_parameters(),
        ):
            if param_1.shape == param_2.shape:
                param_1.data.copy_(param_2.data)
            else:
                param_1.data.copy_(param_2.data[: (num_radial + 2 * num_species), ...])
"""

NEW_BLOCK = """        # ELoRA compatibility patch: name-matched foundation copy.
        # ELoRA e3nn adds LoRA parameters to FC layers, so zip(named_parameters())
        # can pair a new LoRA parameter with a foundation base weight.
        foundation_conv_tp_weights = dict(
            model_foundations.interactions[i].conv_tp_weights.named_parameters()
        )
        for name, param_1 in model.interactions[i].conv_tp_weights.named_parameters():
            if "LoRA" in name or "lora_" in name:
                continue
            param_2 = foundation_conv_tp_weights.get(name)
            if param_2 is None:
                continue
            if param_1.shape == param_2.shape:
                param_1.data.copy_(param_2.data)
            else:
                sliced = param_2.data[: (num_radial + 2 * num_species), ...]
                if param_1.shape == sliced.shape:
                    param_1.data.copy_(sliced)
"""


def main():
    import mace.tools.finetuning_utils as finetuning_utils

    path = Path(finetuning_utils.__file__)
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"ELoRA MACE compatibility patch already present: {path}")
        return
    if OLD_BLOCK not in text:
        print(
            "ELoRA MACE compatibility patch target block not found; "
            f"leaving file unchanged: {path}"
        )
        return
    path.write_text(text.replace(OLD_BLOCK, NEW_BLOCK), encoding="utf-8")
    print(f"Applied ELoRA MACE compatibility patch: {path}")


if __name__ == "__main__":
    main()
