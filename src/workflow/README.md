# Workflow Package - Organized ML Pipeline

This package contains the complete machine learning workflow organized in logical execution order.

## Overview

The workflow is organized into **notebooks** for exploration and analysis, and **scripts** for final pipeline execution.

---

## 📓 Notebooks (Exploration & Analysis)

Run these notebooks in order using Jupyter:

```bash
# Sync dependencies first
uv sync

jupyter lab
# Then open notebooks in this order:
```

### 1. `1_connection_preprocessing.ipynb`
**Data Connection & Exploration**
- Load and explore the electricity dataset
- Understand data structure and distributions
- Initial EDA (plots, statistics)
- **Time**: ~5 minutes

### 2. `2_linear_regression_baseline.ipynb`
**Baseline Models**
- Linear regression on raw features
- Holt-Winters exponential smoothing
- ARIMA time series model
- Compare classical approaches
- **Time**: ~10 minutes

### 3. `3_advanced_models_and_eda.ipynb`
**Advanced Analysis & Feature Engineering**
- Deep exploratory data analysis
- Time series decomposition
- ACF/PACF analysis
- Advanced feature engineering (lags, cyclical, rolling stats)
- XGBoost & LightGBM on engineered features
- LSTM neural network
- **Time**: ~30 minutes

### 4. `4_hyperparameter_tuning.ipynb`
**Optuna Hyperparameter Optimization**
- Define objective function for 9 model types
- Run Optuna study with 100 trials
- Bayesian/TPE search over parameter space
- Save tuning results and study object
- **Time**: ~2-10 hours (depending on system)

### 5. `5_analyze_tuning_results.ipynb`
**Post-Tuning Analysis**
- Visualize tuning results
- Plot optimization history
- Analyze parameter importance
- Extract best models and parameters
- **Time**: ~5 minutes

### 6. `6_cuda_gpu_setup_reference.ipynb`
**GPU/CUDA Setup Reference** (Optional)
- GPU configuration notes
- CUDA path setup documentation
- Environment variable guidance
- Troubleshooting GPU issues

---

## 🐍 Scripts (Pipeline Execution)

Run these scripts in order from the command line:

### 1. `1_extract_best_params.py`
**Extract Best Hyperparameters**
```bash
# Sync dependencies first
uv sync

uv run src/workflow/scripts/1_extract_best_params.py
```
- Reads `results/optuna_trials_final.csv`
- Extracts best parameters per model type
- Saves to `results/extract_params.json`
- **Time**: ~10 seconds

### 2. `2_final_retrain_and_evaluate.py`
**Final Model Training & Evaluation**
```bash
# Sync dependencies first
uv sync

uv run src/workflow/scripts/2_final_retrain_and_evaluate.py
```
- Loads data with full feature engineering
- Trains all 9 models with best hyperparameters
- Evaluates on test set
- Saves metrics to `results/final_test_results.csv`
- **Time**: ~5-10 minutes (CPU), ~1-2 minutes (GPU)

### Supporting Module

- `utils.py` - Timing decorator and utility functions (imported by scripts)

---

## 📊 Complete Workflow

### Data Science Workflow (All Notebooks):
```
Load Data
    ↓
Baseline Models (Classical)
    ↓
Advanced EDA + Feature Engineering
    ↓
Hyperparameter Tuning (100 trials)
    ↓
Analyze Tuning Results
```

### Production Pipeline (Scripts Only):
```
Extract Best Parameters
    ↓
Retrain All Models
    ↓
Evaluate on Test Set
    ↓
Save Results
```

---

## 🚀 Quick Start

### Option A: Full Exploration (Interactive)
```bash
# 1. Start Jupyter
uv sync
jupyter lab

# 2. Run notebooks in order: 1→2→3→4→5
# (Notebook 6 is optional reference)

# 3. Time: ~3-12 hours depending on tuning
```

### Option B: Quick Pipeline (Automated)
```bash
# 1. Run extraction script
uv sync
uv run src/workflow/scripts/1_extract_best_params.py

# 2. Run final training & evaluation
uv run src/workflow/scripts/2_final_retrain_and_evaluate.py

# 3. Check results
cat results/final_test_results.csv

# Time: ~10 minutes
```

### Option C: Development (Notebooks → Scripts)
```bash
# 1. Run notebooks 1-5 for experimentation
uv sync
jupyter lab

# 2. Then run scripts for final pipeline
uv run src/workflow/scripts/1_extract_best_params.py
uv run src/workflow/scripts/2_final_retrain_and_evaluate.py
```

---

## 📁 Directory Structure

```
workflow/
├── README.md (this file)
├── notebooks/
│   ├── 1_connection_preprocessing.ipynb
│   ├── 2_linear_regression_baseline.ipynb
│   ├── 3_advanced_models_and_eda.ipynb
│   ├── 4_hyperparameter_tuning.ipynb
│   ├── 5_analyze_tuning_results.ipynb
│   └── 6_cuda_gpu_setup_reference.ipynb
└── scripts/
    ├── 1_extract_best_params.py
    ├── 2_final_retrain_and_evaluate.py
    └── utils.py
```

---

## 📈 Expected Outputs

### After Notebooks:
- Plots saved to `results/` (PNG, PDF)
- Optuna study saved to `results/electricity_study.pkl`
- Trial results saved to `results/optuna_trials_final.csv`

### After Scripts:
- Final metrics: `results/final_test_results.csv`
- Parameters JSON: `results/extract_params.json`
- Results pickle: `results/final_df_results.pkl`

---

## ⏱️ Time Estimates

| Step | Time | Notes |
|------|------|-------|
| Notebook 1 | ~5 min | Quick EDA |
| Notebook 2 | ~10 min | Baseline models |
| Notebook 3 | ~30 min | Advanced analysis |
| Notebook 4 | ~2-10 hrs | Optuna tuning (long!) |
| Notebook 5 | ~5 min | Results analysis |
| Script 1 | ~10 sec | Extract params |
| Script 2 | ~10 min | Retrain models |
| **Total** | **~3-12 hrs** | Depends on tuning |

---

## 💡 Tips

1. **Skip Notebook 4 if short on time**: You can jump directly to scripts if tuning was already done
2. **GPU helpful**: Notebook 4 (tuning) and Script 2 benefit greatly from GPU
3. **Environment**: Activate Python environment before running:
   ```bash
   cd /home/bhavik/Dropbox/edu/smu/winter/data_mining/a4_regression_ts
   uv sync  # or pip install
   ```

---

## 🔧 Configuration

### GPU/CPU Mode
Edit `2_final_retrain_and_evaluate.py` line 35:
```python
USE_GPU = True  # Set to False for CPU-only mode
```

### Hyperparameters
Notebook 4 (`4_hyperparameter_tuning.ipynb`) has configurable settings:
- `n_trials` - Number of Optuna trials (default: 100)
- `timeout` - Time limit in seconds (optional)
- Model-specific hyperparameter ranges

---

## 📚 Files Outside Workflow

For reference, the following package remains in `src/`:
- `gpu_setup/` - GPU verification/recovery tools and helper scripts

---

## ✅ Workflow Status

- ✅ Organized with numeric prefixes
- ✅ Logical execution order
- ✅ Complete documentation
- ✅ Both exploratory and production pipelines
- ✅ Ready to use

---

**Start with Notebook 1 for exploration or Script 1 for production!**

