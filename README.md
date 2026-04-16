# Electricity Time-Series Regression

**Machine Learning Assignment**: Time-series regression for electricity consumption prediction.

- **Status**: ✅ Complete and operational
- **Python Version**: 3.12+
- **Main Framework**: TensorFlow, XGBoost, LightGBM, scikit-learn, Optuna

## Quick Start (4 commands)

```bash
# 1. Install/sync dependencies
uv sync

# 2. Verify setup works
uv run src/gpu_setup/tools/verify_setup.py

# 3. Run production pipeline
uv run src/workflow/scripts/1_extract_best_params.py
uv run src/workflow/scripts/2_final_retrain_and_evaluate.py

# 4. Check results
cat results/final_test_results.csv
```


---

## What This Project Does

Given **6 hours of past electricity consumption**, predict the **next hour's consumption**.

- **Input**: 6 historical hourly values + engineered features
- **Output**: Predicted consumption  
- **Models**: 9 algorithms trained and compared
- **Best Model**: Usually LightGBM or XGBoost (~9-10% MAPE)

---

## Installation

### Using `uv` (Recommended)

```bash
uv sync
```

### Using pip + venv

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install tensorflow xgboost lightgbm scikit-learn pandas numpy jupyter optuna
```

---

## How to Run

### Option 1: Production Pipeline (Fastest)

**Extract best hyperparameters and train final models:**

```bash
# Ensure dependencies are synced (run once per environment change)
uv sync

# Run extraction
uv run src/workflow/scripts/1_extract_best_params.py

# Run final training and evaluation
uv run src/workflow/scripts/2_final_retrain_and_evaluate.py

# Check results
cat results/final_test_results.csv
```

**Time**: ~10 minutes total

### Option 2: Full Exploratory Workflow (Jupyter)

**Run notebooks in order:**

```bash
# Ensure dependencies are synced
uv sync

jupyter lab

# Then open and run these notebooks in order:
# 1. src/workflow/notebooks/1_connection_preprocessing.ipynb       (5 min)
# 2. src/workflow/notebooks/2_linear_regression_baseline.ipynb     (10 min)
# 3. src/workflow/notebooks/3_advanced_models_and_eda.ipynb        (30 min)
# 4. src/workflow/notebooks/4_hyperparameter_tuning.ipynb          (2-10 hrs)
# 5. src/workflow/notebooks/5_analyze_tuning_results.ipynb         (5 min)
# 6. src/workflow/notebooks/6_cuda_gpu_setup_reference.ipynb       (optional)
```

**Time**: ~3-12 hours (depending on tuning)

### Quick Setup Verification

```bash
uv sync
uv run src/gpu_setup/tools/verify_setup.py
```

This checks all dependencies and GPU status.

---

## 9 Models Evaluated

| Model | Type | Speed | Accuracy |
|-------|------|-------|----------|
| LinearRegression | Baseline | ⚡ | ⭐⭐ |
| HoltWinters | Classical | ⭐ | ⭐⭐ |
| ARIMA | Classical | ⭐ | ⭐⭐ |
| SVR | ML | 🐢 | ⭐⭐⭐ |
| RegressionTree | Tree | ⭐ | ⭐⭐ |
| XGBoost | Boosting | ⭐⭐ (GPU: ⚡) | ⭐⭐⭐⭐ |
| LightGBM | Boosting | ⭐⭐ (GPU: ⚡) | ⭐⭐⭐⭐ |
| NN_1_Layer | Neural | ⭐⭐ | ⭐⭐⭐ |
| NN_3_Layer | Neural | ⭐⭐ | ⭐⭐⭐ |

---

## GPU Setup

### Check GPU Status

```bash
uv run src/gpu_setup/tools/diagnose_cuda.py
```

### Your Project Automatically:
- ✅ Works on CPU (always)
- ✅ Uses GPU if available (5-10x faster)
- ✅ Falls back to CPU if GPU fails

### See Also

- **`src/gpu_setup/README.md`** - GPU package overview
- **`src/gpu_setup/docs/TROUBLESHOOTING.md`** - GPU troubleshooting
- **`DOCUMENTATION.md`** - Complete documentation

---

## Results

### Expected Performance

```
Best Model MAPE: ~9-10%
RMSE: 35-40 units
MAE: 25-30 units

CPU Training: 5-10 min
GPU Training: 1-2 min
```

### Output Files

- `results/final_test_results.csv` - Model metrics
- `results/optuna_trials_final.csv` - Tuning history
- `results/plots/` - Visualizations

---

## Troubleshooting

### "GPU not detected"
Normal. Your project works on CPU automatically.

### "CUDA not found" errors
See: `src/gpu_setup/docs/TROUBLESHOOTING.md`

### "Broken symlinks" errors
Run: `bash src/gpu_setup/scripts/cleanup_broken_links.sh`

### Need Help?
- Quick: `uv run src/gpu_setup/tools/verify_setup.py`
- Full: See `DOCUMENTATION.md`

---

## Documentation

- **`DOCUMENTATION.md`** - Complete project documentation
- **`src/workflow/README.md`** - Workflow package documentation  
- **`src/workflow/QUICK_REFERENCE.md`** - Quick reference guide
- **`src/workflow/PATH_CONSTANTS_GUIDE.md`** - Path constants for notebooks
- **`src/gpu_setup/README.md`** - GPU package overview
- **`src/gpu_setup/docs/TROUBLESHOOTING.md`** - GPU troubleshooting

---

## Next Steps

```bash
# Quick: Verify setup and run production pipeline
uv sync
uv run src/gpu_setup/tools/verify_setup.py
uv run src/workflow/scripts/1_extract_best_params.py
uv run src/workflow/scripts/2_final_retrain_and_evaluate.py
cat results/final_test_results.csv

# Or: Full exploration with Jupyter
jupyter lab
# Open: src/workflow/notebooks/1_connection_preprocessing.ipynb
```

---

## Project Structure

```
a4_regression_ts/
├── README.md                          # This file
├── DOCUMENTATION.md                   # Complete documentation
├── GPU_FIX_WALKTHROUGH.md            # GPU setup reference (RTX 4070)
├── pyproject.toml                     # Dependencies
├── uv.lock                            # Locked versions
│
├── src/                               # Source code
│   ├── workflow/                      # ML Workflow (PRIMARY)
│   │   ├── README.md                  # Workflow guide
│   │   ├── QUICK_REFERENCE.md         # Quick start
│   │   ├── PATH_CONSTANTS_GUIDE.md    # Path usage in notebooks
│   │   ├── notebook_utils.py          # Utilities for notebooks
│   │   ├── notebooks/                 # Jupyter notebooks (numbered)
│   │   │   ├── 1_connection_preprocessing.ipynb
│   │   │   ├── 2_linear_regression_baseline.ipynb
│   │   │   ├── 3_advanced_models_and_eda.ipynb
│   │   │   ├── 4_hyperparameter_tuning.ipynb
│   │   │   ├── 5_analyze_tuning_results.ipynb
│   │   │   └── 6_cuda_gpu_setup_reference.ipynb
│   │   └── scripts/                   # Production scripts (numbered)
│   │       ├── 1_extract_best_params.py
│   │       ├── 2_final_retrain_and_evaluate.py
│   │       └── utils.py               # Path constants & utilities
│   │
│   └── gpu_setup/                     # GPU Setup Package
│       ├── README.md
│       ├── tools/
│       │   ├── verify_setup.py
│       │   ├── diagnose_cuda.py
│       │   └── gpu_recovery.py
│       ├── scripts/
│       │   ├── fix_cuda_paths.sh
│       │   └── cleanup_broken_links.sh
│       └── docs/
│           ├── QUICK_START.md
│           ├── TROUBLESHOOTING.md
│           └── SETUP_DETAILS.md
│
├── data/                              # Data files
│   └── electricity_prediction.csv
│
├── results/                           # Generated results
│   ├── final_test_results.csv
│   ├── optuna_trials_final.csv
│   ├── extract_params.json
│   └── plots/
│
└── report/                            # LaTeX report (optional)
```

---

## Project Info

- **Python**: 3.12+
- **Status**: ✅ Complete and ready to use
- **GPU**: Optional (works with or without)
- **Dependencies**: TensorFlow, XGBoost, LightGBM, scikit-learn, Optuna

---

**See `DOCUMENTATION.md` for complete information**
---

## Model Performance

All 9 models are trained and compared on held-out test data:

- **LightGBM** ~9.88% MAPE (best accuracy)
- **XGBoost** ~10.39% MAPE (fast and accurate)
- **SVR** ~10.59% MAPE
- **1-Layer NN** ~10.67% MAPE
- **3-Layer NN** ~10.76% MAPE
- **Decision Tree** ~10.71% MAPE
- **Linear Regression** ~11.49% MAPE
- **Holt-Winters** ~54.88% MAPE
- **ARIMA** ~63.36% MAPE

---

## Report to Code Mapping

The LaTeX report references code and results:

- **Notebooks**: `src/workflow/notebooks/`
- **Scripts**: `src/workflow/scripts/`
- **Results**: `results/final_test_results.csv` and `results/optuna_trials_final.csv`
- **Plots**: Auto-generated in `results/plots/`

For more details, see `DOCUMENTATION.md`.

---

## Important Notes

- ✅ Project uses **chronological splitting** (time-aware, not random)
- ✅ **Path constants** defined in `src/workflow/scripts/utils.py` (no hard-coded paths)
- ✅ **Workflow organized** with numbered notebooks and scripts (execution order clear)
- ✅ **GPU support** automatic (fallback to CPU if unavailable)

---

**Questions?** See `DOCUMENTATION.md` or `src/workflow/README.md`
