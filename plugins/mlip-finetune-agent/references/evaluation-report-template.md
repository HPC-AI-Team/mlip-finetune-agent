# ELoRA Evaluation Report Template

Write this file as `mlip_runs/<run_id>/evaluation.md`.

```markdown
# ELoRA MLIP Evaluation Report

## Summary

- Run ID:
- Model:
- Dataset split:
- Evaluation date:
- Device:
- Evaluation command:
- ELoRA environment: `.venv-elora`
- ELoRA source refs: `ELoRA@main`, `ELoRA@MACE_ELoRA`
- LoRA rank/alpha: `16` / `16`

## Metrics

| Metric | Value | Unit | Notes |
| --- | ---: | --- | --- |
| Configurations |  | count |  |
| Energy RMSE |  | meV/atom |  |
| Force RMSE |  | meV/A |  |
| Relative force RMSE |  | percent | optional |

## Dataset

- Dataset name:
- Source URL:
- Local path:
- Reference energy key:
- Reference force key:
- Split policy:

## Model

- Foundation model:
- Fine-tuned model:
- Mode: organic/inorganic
- Environment: ELoRA `.venv-elora`
- ELoRA marker check:
- Training command: `mlip_runs/<run_id>/train-command.sh`

## Notes

- Known limitations:
- Follow-up experiments:
```
