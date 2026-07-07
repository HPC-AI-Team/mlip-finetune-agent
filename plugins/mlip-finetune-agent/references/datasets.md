# Dataset Reference

Local source: `docs/test-model-and-dataset.md`.

## Default Local Paths

- Models: `/hdd/mlip-finetune/models`
- Datasets: `/hdd/mlip-finetune/datasets`

## Foundation Models

| Use | Filename | Source |
| --- | --- | --- |
| Inorganic MACE foundation model | `2024-01-07-mace-128-L2_epoch-199.model` | `https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model` |
| Organic MACE-OFF foundation model | `MACE-OFF23_medium.model` | `https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_medium.model` |

Model files must be real torch weights. A GitHub HTML page, Git LFS pointer, or small text file is not valid even if it has a `.model` suffix.

## Dataset Catalog

| Dataset | Size | Source |
| --- | ---: | --- |
| rMD17 | 1,000,000 | `https://dx.doi.org/10.6084/m9.figshare.12672038` |
| 3BPA | 13,997 | `https://github.com/davkovacs/BOTNet-datasets` |
| AcAc | 6,263 | `https://github.com/davkovacs/BOTNet-datasets` |
| SSE-PBE | 15,774 | `https://www.aissquare.com/datasets/detail?pageType=datasets&id=146` |
| H2O-PD | 48,419 | `https://www.aissquare.com/datasets/detail?pageType=datasets&id=137` |
| Ag/Au-PBE | 17,508 | `https://www.aissquare.com/datasets/detail?pageType=datasets&id=152` |
| Al/Mg/Cu | 25,397 | `https://www.aissquare.com/datasets/detail?pageType=datasets&id=139` |
| Cu | 15,366 | `https://www.aissquare.com/datasets/detail?pageType=datasets&id=132` |
| Sn | 6,725 | `https://www.aissquare.com/datasets/detail?pageType=datasets&id=129` |
| Ti | 10,528 | `https://www.aissquare.com/datasets/detail?pageType=datasets&id=133` |
| V | 15,673 | `https://www.aissquare.com/datasets/detail?pageType=datasets&id=135` |
| W | 44,397 | `https://www.aissquare.com/datasets/detail?pageType=datasets&id=136` |
| HfO2 | 28,577 | `https://www.aissquare.com/datasets/detail?pageType=datasets&id=145` |

## Recommended First Targets

Start with `3BPA` or `AcAc` for ELoRA experiments. They are small enough for iteration and were explicitly recommended in the local project notes. If `MACE-OFF23_medium.model` is not a real weight file, use the valid MACE-MP foundation model for smoke tests until the MACE-OFF weight is downloaded correctly.

## Provenance Fields

Record these in `mlip_runs/<run_id>/dataset-manifest.md`:

- Dataset name and source URL.
- Local source path.
- Source file checksums when practical.
- Split policy and random seed.
- Number of configurations per split.
- Reference property keys used for energy, forces, stress, cell, and pbc.
- Known license or access constraints.
