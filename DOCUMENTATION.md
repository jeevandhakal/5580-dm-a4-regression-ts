# Electricity Time-Series Regression - Documentation

This document describes the current project structure, execution flow, and recommended commands.

## 1) Project Summary

This project predicts electricity consumption (regression on time-ordered data) using:
- Classical models: Linear Regression, Holt-Winters, ARIMA
- ML models: SVR, Decision Tree, XGBoost, LightGBM
- Neural models: Dense NN variants
- Hyperparameter optimization: Optuna

Dataset:
- `data/electricity_prediction.csv`
- 7 columns: `Hour_1` ... `Hour_6`, `Target`

## 2) Current Repository Structure

```text
.
├── README.md
├── DOCUMENTATION.md
├── pyproject.toml
├── uv.lock
├── data/
├── results/
├── report/
└── src/
    ├── workflow/
    │   ├── notebooks/
    │   │   ├── 1_connection_preprocessing.ipynb
    │   │   ├── 2_linear_regression_baseline.ipynb
    │   │   ├── 3_advanced_models_and_eda.ipynb
    │   │   ├── 4_hyperparameter_tuning.ipynb
    │   │   ├── 5_analyze_tuning_results.ipynb
    │   │   └── 6_cuda_gpu_setup_reference.ipynb
    │   ├── scripts/
    │   │   ├── 1_extract_best_params.py
    │   │   ├── 2_final_retrain_and_evaluate.py
    │   │   └── utils.py
    │   ├── notebook_utils.py
    │   ├── QUICK_REFERENCE.md
    │   ├── PATH_CONSTANTS_GUIDE.md
    │   └── README.md
    └── gpu_setup/
        ├── README.md
        ├── tools/
        ├── scripts/
        └── docs/
```

## 3) Setup (uv-first)

Run from project root:

```bash
uv sync
```

This installs/updates dependencies into `.venv` according to `pyproject.toml` and `uv.lock`.

## 4) Quick Start (Production)

```bash
uv sync
uv run src/gpu_setup/tools/verify_setup.py
uv run src/workflow/scripts/1_extract_best_params.py
uv run src/workflow/scripts/2_final_retrain_and_evaluate.py
```

Outputs:
- `results/extract_params.json`
- `results/final_test_results.csv`
- `results/final_df_results.pkl`

## 5) Notebook Workflow (Exploratory)

```bash
uv sync
jupyter lab
```

Run notebooks in order:
1. `src/workflow/notebooks/1_connection_preprocessing.ipynb`
2. `src/workflow/notebooks/2_linear_regression_baseline.ipynb`
3. `src/workflow/notebooks/3_advanced_models_and_eda.ipynb`
4. `src/workflow/notebooks/4_hyperparameter_tuning.ipynb`
5. `src/workflow/notebooks/5_analyze_tuning_results.ipynb`
6. `src/workflow/notebooks/6_cuda_gpu_setup_reference.ipynb` (optional)

## 6) GPU Verification and Recovery

```bash
uv sync
uv run src/gpu_setup/tools/verify_setup.py
uv run src/gpu_setup/tools/diagnose_cuda.py
uv run src/gpu_setup/tools/gpu_recovery.py --cleanup
bash src/gpu_setup/scripts/cleanup_broken_links.sh
bash src/gpu_setup/scripts/fix_cuda_paths.sh
```

Notes:
- `nvidia-smi` health alone is not enough; runtime libraries/toolchain must also be visible to TensorFlow.
- Workflow scripts auto-configure NVIDIA runtime paths from `.venv` before TensorFlow import.

## 7) Path Management

Path constants are centralized in:
- `src/workflow/scripts/utils.py`

For notebooks, use:
- `src/workflow/notebook_utils.py`
- `src/workflow/PATH_CONSTANTS_GUIDE.md`

This avoids hard-coded absolute paths.

## 8) Main Result Artifacts

Common files in `results/`:
- `final_test_results.csv`
- `optuna_trials_final.csv`
- `extract_params.json`
- `electricity_study.pkl`
- plots in `results/plots/` (and other chart files)

## 9) Troubleshooting

If setup or runtime fails:

```bash
uv sync
uv run src/gpu_setup/tools/verify_setup.py
uv run src/gpu_setup/tools/diagnose_cuda.py
```

If script imports/path fail:
- Ensure commands are run from project root.
- Re-run `uv sync`.

If GPU is unavailable:
- CPU mode is supported; workflow still runs.

## 10) References

- High-level usage: `README.md`
- Workflow details: `src/workflow/README.md`
- GPU details: `src/gpu_setup/README.md`
- Legacy historical notes: `GPU_FIX_WALKTHROUGH.md`
