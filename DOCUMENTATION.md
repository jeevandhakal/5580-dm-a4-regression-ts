# Project documentation: Electricity time-series regression

This guide explains what the **5580-dm-a4-regression-ts** project does, how the pieces fit together, and how to run or extend it. It is written for readers who know basic programming but are new to this repository.

---

## 1. Project overview

### What this project is

This is a **data mining / machine learning assignment** focused on **regression** (predicting a number) on **time-series** data. The main dataset is hourly **electricity consumption**: each row has six past hourly values and a seventh column—the value to predict (`Target`).

### What problem it solves

**Problem:** Given recent electricity usage patterns, predict the next period’s consumption as accurately as possible.

**Why it matters:** Utilities and grid operators forecast demand to plan generation and avoid waste. Even a course-scale project mirrors that idea: build models, compare them, and report error in business-friendly terms (especially **MAPE**—see [Key concepts](#4-key-concepts-used)).

### Why it might be useful

- Learn a full workflow: load data → explore → engineer features → train many model families → tune hyperparameters → evaluate on held-out time.
- See **classical stats** (Holt-Winters, ARIMA) next to **modern ML** (XGBoost, LightGBM, neural nets).
- Outputs (CSVs, plots, optional LaTeX report) support a written report or presentation.

---

## 2. How the project works (high level)

### Overall flow

1. **Data** lives in `data/electricity_prediction.csv` (no header row; seven numeric columns).
2. **Notebooks and scripts** in `code/` load that file, name columns (`Hour_1` … `Hour_6`, `Target`), and optionally add **lags**, **cyclical hour features**, and **rolling statistics**.
3. Models are trained on **earlier** rows and tested on **later** rows (**chronological split**—no random shuffle, so the future is not leaked into training).
4. **Hyperparameter tuning** (Optuna) tries many settings and model types; results are saved under `results/`.
5. **Final evaluation** retrains selected configurations and writes metrics (MAPE, RMSE, MAE, timing) to CSV.

### Main components and how they interact

| Piece | Role |
|--------|------|
| `data/electricity_prediction.csv` | Raw hourly features + target (~139k rows). |
| `code/connection_preprocessing.ipynb` | Connects to data, explores and preprocesses (path relative to `code/`). |
| `code/linear_regression.ipynb` | Baseline: **linear regression**, **Holt-Winters**, **ARIMA**; MAPE-focused. |
| `code/linear_regression_advanced.ipynb` | “Phase 2”: deeper EDA (decomposition, ACF/PACF), **XGBoost**, **LSTM**. |
| `code/hyperparameter_tuning.ipynb` | **Optuna** study over nine model types; saves trials and study object. |
| `code/utils.py` | `time_operation` decorator to measure wall-clock time (used in tuning). |
| `code/extract_params.py` | Reads `results/optuna_trials_final.csv` and prints best parameters per model (intended after tuning). |
| `code/final_retrain.py` | Retrains all models with fixed “best” params; writes `final_test_results.csv`. |
| `code/verify_gpu.py` / `code/gpu_health_check.py` | Check TensorFlow + GPU; optional on CPU-only setups. |
| `results/` | Plots, Optuna exports, final metrics, pickles. |
| `report/` | LaTeX source for an academic-style write-up (separate from Python execution). |

### Data flow (step by step)

1. **Read CSV** with `header=None`; assign column names.
2. **Feature engineering** (in advanced pipelines): e.g. `Hour_Sin` / `Hour_Cos` from row index mod 24, `Lag_24`, `Lag_168`, rolling mean/std; then **drop rows with NaNs** from shifting.
3. **Split by time**: e.g. 70% train, 15% validation, 15% test (exact ratios may vary by notebook).
4. **Scale** features/targets with `StandardScaler` where needed (many sklearn/NN models expect similar scales).
5. **Train** on train (+ sometimes validation), **predict** on test.
6. **Metrics**: MAPE, RMSE, MAE (and sometimes timing in milliseconds).

---

## 3. Code structure breakdown

### Repository layout (meaningful parts only)

```
5580-dm-a4-regression-ts/
├── pyproject.toml          # Python 3.12+, dependencies (Jupyter, sklearn, TF, Optuna, XGBoost, …)
├── uv.lock                 # Locked dependency versions (if you use `uv`)
├── data/
│   └── electricity_prediction.csv   # Main dataset
├── code/
│   ├── linear_regression.ipynb           # Baseline models + simple EDA
│   ├── linear_regression_advanced.ipynb  # EDA + XGBoost + LSTM
│   ├── connection_preprocessing.ipynb    # Data connection / preprocessing exploration
│   ├── hyperparameter_tuning.ipynb       # Optuna: multi-model tuning
│   ├── tuning_results.ipynb              # Visualize / analyze tuning outputs
│   ├── cuda.ipynb                        # CUDA / GPU notes (environment-specific)
│   ├── final_retrain.py                  # Batch retrain + test metrics → results/
│   ├── extract_params.py                 # Summarize best Optuna params from CSV
│   ├── utils.py                          # Timing helper
│   ├── verify_gpu.py                     # Quick TF GPU check
│   ├── gpu_health_check.py               # Deeper GPU / XLA checks
│   ├── run_health_check.sh               # Shell wrapper (Linux) for gpu_health_check
│   └── fix_cuda_paths.sh                 # Linux CUDA library bridging (see GPU_FIX_WALKTHROUGH.md)
├── results/                # Generated outputs (CSVs, PNGs, PKLs)
├── report/                 # LaTeX report (chapters, figures)
├── GPU_FIX_WALKTHROUGH.md  # How GPU issues were fixed on one Linux + RTX setup
├── AS4.md                  # Study notes (supervised learning, MAPE, time series)
└── DOCUMENTATION.md        # This file
```

### What important files do

- **`pyproject.toml`** — Declares the environment: Jupyter, pandas, scikit-learn, statsmodels, TensorFlow, XGBoost, LightGBM, Optuna, plotting libraries, etc.
- **`code/utils.py`** — Wraps a function so each call returns `(result, duration_ms)` using high-resolution timers.
- **`code/final_retrain.py`** — Single script that loads data, applies the same `prepare_data` logic as tuning (cyclical hours, lags, rolling stats), splits, scales, fits nine model types, and saves **`results/final_test_results.csv`** and **`results/final_df_results.pkl`**.

**Important:** `final_retrain.py` and `extract_params.py` currently use a **hard-coded absolute path** (`/home/bhavik/...`). On your machine you must change those paths to your project root (see [Common issues](#8-common-issues--fixes)).

---

## 4. Key concepts used

### Supervised regression

You have **inputs** (past hours, engineered features) and a **numeric target** (next-hour consumption). The model learns a mapping from inputs to target. This is **regression**, not classification (no discrete “labels” like spam/not spam).

### Time series

Data points are **ordered in time**. The project assumes **one row per hour** in order. Splits are **chronological** so the model is evaluated on “future” data, similar to real forecasting.

### Feature engineering

- **Lags:** Values from 24 or 168 steps ago capture **daily** and **weekly** patterns.
- **Cyclical encoding (sin/cos for hour):** Maps “hour 23” near “hour 0” so the model sees them as neighbors on a clock, not as far-apart integers.
- **Rolling mean/std:** Summarize **recent level** and **volatility**.

### Classical forecasting models

- **Holt-Winters (triple exponential smoothing):** Good for series with **trend** and **seasonality** (here, seasonal period 24 for hourly data).
- **ARIMA(p,d,q):** Uses past values and past errors to predict; `p`, `d`, `q` control how much memory the model has.

### Machine learning models

- **Linear regression:** Simple, fast baseline; assumes roughly linear relationships after scaling.
- **Decision tree / SVR:** Nonlinear; SVR is expensive on large data (this project may train SVR on a **subset** for speed).
- **XGBoost / LightGBM:** **Gradient boosting** on trees; strong tabular performance; project code can use **GPU** (`device='cuda'` / `device='gpu'`).
- **Neural networks (Dense / LSTM):** Stacks of layers that learn nonlinear patterns; need more tuning and compute.

### Hyperparameter tuning (Optuna)

**Hyperparameters** are settings you choose before training (tree depth, ARIMA orders, neural net width, etc.). **Optuna** runs many **trials**, each suggesting parameters, and keeps the setup that minimizes validation error (here, **MAPE**).

### Error metrics

- **MAPE (Mean Absolute Percentage Error):** Average of \|actual − predicted\| / \|actual\|, in percent. Easy to explain (“about 10% off on average”).
- **RMSE / MAE:** Absolute error in the same units as consumption; RMSE penalizes large mistakes more.

---

## 5. What the project produces (results)

### After running notebooks

Typical artifacts (exact filenames depend on what you execute):

- **Plots** under `results/` or `results/plots/`: time series, decomposition, ACF/PACF, correlation heatmaps, model comparison charts, Optuna optimization history, etc.
- **`results/optuna_trials_final.csv`** — One row per Optuna trial with parameters and objective value.
- **`results/electricity_study.pkl`** — Serialized Optuna study (reload for more analysis).
- **`results/extract_params.json`** — Written by `final_retrain.py` in its current form (best-parameter snapshot).

### After `final_retrain.py`

- **`results/final_test_results.csv`** — Table of models vs **MAPE**, **RMSE**, **MAE**, **Time_ms** on the held-out test slice.

Example shape (your numbers may differ slightly):

| Model           | MAPE (approx.) | Notes                          |
|-----------------|----------------|--------------------------------|
| LightGBM        | ~9.9%          | Often strong on this pipeline  |
| XGBoost         | ~10.4%         | GPU-accelerated in script      |
| LinearRegression| ~11.5%         | Simple baseline                |

Holt-Winters and ARIMA rows may look much worse on this **feature-based test design** because those cells are trained on the **scaled target series only**, while the test slice is aligned with the engineered feature matrix—document your evaluation setup clearly in a report.

### LaTeX report (`report/`)

Building `report/main.tex` produces a PDF write-up (figures, methodology, results). That is optional for “running the ML code” but part of the full assignment deliverable.

---

## 6. How to run the project (step by step)

### Prerequisites

- **Python 3.12 or newer** (`pyproject.toml` requires `>=3.12`).
- **Git** (optional) to clone; you already have the folder.
- **Jupyter** — run notebooks in VS Code, Cursor, or classic Jupyter Lab/Notebook.
- **Optional: NVIDIA GPU** — Speeds up XGBoost, LightGBM, and TensorFlow. The project can be explored on CPU, but you may need to **change device settings** in code (see [Common issues](#8-common-issues--fixes)).

### Installation (recommended: `uv`)

From the **project root** (folder that contains `pyproject.toml`):

```powershell
# If you use uv (https://github.com/astral-sh/uv)
uv sync
```

That creates/uses `.venv` and installs dependencies from `pyproject.toml` / `uv.lock`.

**Alternative (pip + venv):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

If `pip install -e .` fails because there is no package layout, install from the file:

```powershell
pip install jupyter pandas numpy scikit-learn matplotlib seaborn plotly kaleido statsmodels optuna xgboost lightgbm tensorflow
```

(Align versions with `pyproject.toml` when possible.)

### Running Jupyter

```powershell
cd "c:\path\to\5580-dm-a4-regression-ts"
.\.venv\Scripts\Activate.ps1
jupyter lab
```

Open notebooks from the `code/` folder. Most notebooks set:

```python
data_path = Path().cwd().parent / "data" / "electricity_prediction.csv"
results_path = Path().cwd().parent / "results"
```

So the **current working directory should be `code/`** when those cells run (normal if you open the notebook from `code/`).

### Suggested order for beginners

1. **`connection_preprocessing.ipynb`** — Confirm the CSV loads and you understand columns.
2. **`linear_regression.ipynb`** — Linear regression + Holt-Winters + ARIMA; smaller conceptual surface.
3. **`linear_regression_advanced.ipynb`** — Decomposition, XGBoost, LSTM.
4. **`hyperparameter_tuning.ipynb`** — Long-running; uses GPU if available; produces `optuna_trials_final.csv`.
5. **`extract_params.py`** — After tuning, to inspect best parameters (fix paths first).
6. **`final_retrain.py`** — Final benchmark table (fix paths and GPU settings if needed).

### GPU checks (optional)

```powershell
python code\verify_gpu.py
```

On Linux, `bash code/run_health_check.sh` runs the fuller check (expects `.venv` layout from that script).

### Expected output examples

- Notebooks: printed metrics, plots inline, files under `results/`.
- `final_retrain.py`: lines like `✅ XGBoost: MAPE=10.39%, RMSE=38.07, Time=1178.44ms` and a final printed table; **`results/final_test_results.csv`** updated.

---

## 7. How to experiment / extend the project

Simple, safe experiments:

| Idea | What to change | What you might see |
|------|----------------|-------------------|
| Train/test ratio | `train_size` / `val_size` in `final_retrain.py` or notebook splits | More training data can lower variance; less test data makes scores noisier. |
| Seasonal period | `seasonal_periods=24` in Holt-Winters | Wrong period hurts seasonal models; 24 fits hourly-with-daily pattern. |
| Optuna trials | `n_trials=100` in `hyperparameter_tuning.ipynb` | More trials → better search, longer runtime. |
| Neural net training | `epochs`, `batch_size` in Keras `fit` | Underfitting vs overfitting; training time. |
| Tree models | `n_estimators` for XGBoost/LightGBM | Often better with more trees until diminishing returns. |
| Features | Add/remove lags or rolling windows in `prepare_data` | Can improve or destabilize; watch MAPE on validation. |

**Try different data:** Replacing `electricity_prediction.csv` with another hourly series (same column shape) lets you test whether conclusions generalize—keep the same seven-column format or update column logic.

**Report / figures:** Regenerate plots after changes; `report/` figures are often copies of `results/` outputs for the PDF.

---

## 8. Common issues & fixes

### `FileNotFoundError` for the CSV

- Run notebooks with **cwd = `code/`**, or change `data_path` to an absolute path to your `data/electricity_prediction.csv`.

### Hard-coded paths in `final_retrain.py` / `extract_params.py`

- Replace `root_path = Path('/home/bhavik/...')` with something like:

  ```python
  root_path = Path(__file__).resolve().parent.parent
  ```

  and use `root_path / "data" / "electricity_prediction.csv"`, `root_path / "results"`, etc.

### GPU / CUDA errors (TensorFlow, XGBoost, LightGBM)

- Install a **GPU-enabled** stack matching your driver, or temporarily set **CPU**:
  - XGBoost: e.g. `tree_method='hist'` without `device='cuda'`, or `device='cpu'`.
  - LightGBM: `device='cpu'`.
  - TensorFlow: will fall back to CPU if no GPU; first run may download or compile kernels.

See **`GPU_FIX_WALKTHROUGH.md`** for one documented Linux + TensorFlow + cuDNN mismatch fix (symbolic links and `LD_LIBRARY_PATH`). On Windows, use official installs or WSL2; paths differ from that doc.

### `ModuleNotFoundError: utils` in Jupyter

- Start Jupyter from project root **or** add `code` to `sys.path`, or run the notebook’s first cells after opening from `code/` so `from utils import time_operation` resolves.

### SVR or tuning “too slow”

- The design uses a **10% random subset** for SVR during tuning and retrain—document that if you report results.

### Holt-Winters / ARIMA vs tree models on the same test set

- Classical models use **univariate** scaled `y`; tree/linear models use the **full feature matrix**. Comparisons are still informative but **methodologically asymmetric**—worth one sentence in a report.

---

## Quick reference: dependency highlights

From `pyproject.toml`: **Jupyter**, **pandas**, **numpy**, **scikit-learn**, **statsmodels**, **TensorFlow**, **XGBoost**, **LightGBM**, **Optuna**, **matplotlib**, **seaborn**, **plotly**, **ruff**, **ty**, **nvidia-cudnn-cu12** (for GPU stacks).

---

*This document describes the repository as of the assignment tree; if you refactor paths or models, update the “run” and “issues” sections to match.*
