# Scripts index

Operational scripts for WSSIS / SWSIS. **Primary metric everywhere: COCO instance-segmentation AP** (mask AP, plus AP50/AP75/AP_S/M/L for the student).

| Path | Purpose |
|------|---------|
| [RUNBOOK.md](RUNBOOK.md) | **Step-by-step** remote GPU setup and full pipeline |
| [CHECKLIST.md](CHECKLIST.md) | **Status checklist** — prep, experiments, eval, report deliverables |
| [setup/](setup/) | Conda env, Kaggle data, SAM weights |
| [prep/run_p0.sh](prep/run_p0.sh) | P0: splits → SAM cache → Stage-1 GNN |
| [eval/](eval/) | Teacher AP eval on val set |
| [experiments/](experiments/) | Per-experiment runners + run-all |
| [upload_exp_1c_hf.py](upload_exp_1c_hf.py) | Upload Exp 1C demo weights to Hugging Face |

## Quick commands

```bash
export WSSIS_REPO_ROOT=$PWD PYTHONPATH=$PWD WSSIS_RUN_ID=wssis_main

bash scripts/setup/00_create_conda_env.sh
bash scripts/setup/01_download_data.sh
bash scripts/prep/run_p0.sh --run-id $WSSIS_RUN_ID

python scripts/experiments/run_exp_1c.py --run-id $WSSIS_RUN_ID
bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID
bash scripts/eval/run_experiment_eval.sh 1C --run-id $WSSIS_RUN_ID
```

See [CHECKLIST.md](CHECKLIST.md) before writing the project report.
