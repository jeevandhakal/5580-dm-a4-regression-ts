import os
import warnings
import logging
import json
import time
import glob

import numpy as np
import pandas as pd
import joblib

import xgboost as xgb
import lightgbm as lgb

# --- IMPORT CONSTANTS & UTILS ---
from utils import (
    verify_directories,
    ROOT_DIR,
    DATA_FILE,
    EXTRACT_PARAMS_JSON,
    FINAL_RESULTS_CSV,
    FINAL_RESULTS_PKL,
)

# --- GPU RUNTIME PATH SETUP (before TensorFlow import) ---
def _append_flag(env_var: str, flag: str) -> None:
    current = os.environ.get(env_var, "")
    if flag not in current:
        os.environ[env_var] = f"{current} {flag}".strip()


def _configure_gpu_runtime_paths() -> dict:
    venv_site = next((ROOT_DIR / ".venv" / "lib").glob("python*/site-packages"), None)
    if not venv_site:
        return {"nvidia_lib_dirs": [], "nvcc_bin": None, "libdevice_dir": None}

    nvidia_root = venv_site / "nvidia"
    nvidia_lib_dirs = sorted(glob.glob(str(nvidia_root / "*" / "lib")))

    # 1) Shared library search path
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = ["/usr/lib/x86_64-linux-gnu", *nvidia_lib_dirs]
    if existing_ld:
        ld_parts.append(existing_ld)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    # 2) ptxas/nvlink path from nvidia-cuda-nvcc-cu12 wheel
    nvcc_bin = nvidia_root / "cuda_nvcc" / "bin"
    if nvcc_bin.exists():
        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{nvcc_bin}:{existing_path}"

    # 3) XLA libdevice location
    libdevice_dir = nvidia_root / "cuda_nvcc" / "nvvm" / "libdevice"
    if libdevice_dir.exists():
        # XLA expects the CUDA root directory, not the libdevice file path.
        cuda_root = libdevice_dir.parent.parent  # .../nvidia/cuda_nvcc/nvvm -> .../nvidia/cuda_nvcc
        _append_flag("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={cuda_root}")

    # 4) Safety fallback if ptxas toolchain still fails on a host
    _append_flag("XLA_FLAGS", "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true")

    return {
        "nvidia_lib_dirs": nvidia_lib_dirs,
        "nvcc_bin": str(nvcc_bin) if nvcc_bin.exists() else None,
        "libdevice_dir": str(libdevice_dir) if libdevice_dir.exists() else None,
    }


GPU_RUNTIME_INFO = _configure_gpu_runtime_paths()

import tensorflow as tf  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import mean_squared_error, mean_absolute_error  # noqa: E402

# --- SILENCE ENVIRONMENT ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

if GPU_RUNTIME_INFO["nvcc_bin"]:
    print(f"GPU toolchain path: {GPU_RUNTIME_INFO['nvcc_bin']}")
if GPU_RUNTIME_INFO["libdevice_dir"]:
    print(f"GPU libdevice path: {GPU_RUNTIME_INFO['libdevice_dir']}")


def prepare_data(df):
    """Create engineered features and target for model training."""
    X = df.copy()
    series = X['Target']

    # Cyclical Features
    hour_series = X.index % 24
    X['Hour_Sin'] = np.sin(2 * np.pi * hour_series / 24)
    X['Hour_Cos'] = np.cos(2 * np.pi * hour_series / 24)

    # Lags & Rolling Stats
    X['Lag_24'] = series.shift(24)
    X['Lag_168'] = series.shift(168)
    X['Rolling_Mean_6'] = series.shift(1).rolling(window=6).mean()
    X['Rolling_Std_24'] = series.shift(1).rolling(window=24).std()

    X = X.dropna()
    y = X['Target']
    X = X.drop(columns=['Target'])

    return X, y

# --- 1. VERIFY ENVIRONMENT & LOAD DATA ---
verify_directories()

df = pd.read_csv(DATA_FILE, header=None)
column_names = [f'Hour_{i}' for i in range(1, 7)] + ['Target']
df.columns = column_names

X_eng, y_eng = prepare_data(df)

# --- 2. CHRONOLOGICAL SPLIT & COMBINE ---
# 70% Train, 15% Val, 15% Test
train_size = int(len(X_eng) * 0.70)
val_size = int(len(X_eng) * 0.15)

# Train+Validation set (85%)
X_train_val = X_eng.iloc[:train_size + val_size]
y_train_val = y_eng.iloc[:train_size + val_size]

# Test set (15%)
X_test = X_eng.iloc[train_size + val_size:]
y_test = y_eng.iloc[train_size + val_size:]

# --- 3. SCALING ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_val_scaled = scaler_X.fit_transform(X_train_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_val_scaled = scaler_y.fit_transform(y_train_val.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# --- 4. BEST PARAMETERS ---
with open(EXTRACT_PARAMS_JSON, 'w') as f:
    best_params = {
      "LinearRegression": {},
      "HoltWinters": {},
      "ARIMA": {"d": 0, "p": 0, "q": 3},
      "SVR": {"SVR_C": 0.6839107352478422, "svr_kernel": "rbf"},
      "RegressionTree": {"dt_depth": 11},
      "XGBoost": {},
      "LightGBM": {},
      "NN_1_Layer": {"u0_NN_1_Layer": 251},
      "NN_3_Layer": {"u0_NN_3_Layer": 117, "u1_NN_3_Layer": 173, "u2_NN_3_Layer": 247}
    }
    json.dump(best_params, f)

# --- 5. RETRAINING & EVALUATION ---
results_registry = {}

def calculate_metrics(y_true, y_pred, model_name, duration_ms):
    # Calculations are done on ORIGINAL SCALE as requested by "Professional precision"
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    results_registry[model_name] = {
        "MAPE": mape,
        "RMSE": rmse,
        "MAE": mae,
        "Time_ms": duration_ms
    }
    print(f"✅ {model_name}: MAPE={mape:.2f}%, RMSE={rmse:.2f}, Time={duration_ms:.2f}ms")

# Loop through models
for model_name, params in best_params.items():
    print(f"Processing {model_name}...")
    
    if model_name == 'LinearRegression':
        start = time.perf_counter()
        model = LinearRegression().fit(X_train_val_scaled, y_train_val_scaled)
        preds_scaled = model.predict(X_test_scaled)
        duration_ms = (time.perf_counter() - start) * 1000
        
    elif model_name == 'HoltWinters':
        start = time.perf_counter()
        # HoltWinters expects a univariate series. 
        # Note: seasonal_periods=24 as in the original tuning
        model = ExponentialSmoothing(y_train_val_scaled, trend='add', seasonal='add', seasonal_periods=24).fit()
        preds_scaled = model.forecast(len(y_test_scaled))
        duration_ms = (time.perf_counter() - start) * 1000

    elif model_name == 'ARIMA':
        start = time.perf_counter()
        model = ARIMA(y_train_val_scaled, order=(int(params['p']), int(params['d']), int(params['q']))).fit()
        preds_scaled = model.forecast(steps=len(y_test_scaled))
        duration_ms = (time.perf_counter() - start) * 1000

    elif model_name == 'SVR':
        start = time.perf_counter()
        # SVR on full dataset is O(N^2) or O(N^3), taking hours. 
        # Using 10% subset as per the tuning strategy for final evaluation.
        idx = np.random.choice(len(X_train_val_scaled), int(len(X_train_val_scaled)*0.1), replace=False)
        model = SVR(kernel=params['svr_kernel'], C=params['SVR_C']).fit(X_train_val_scaled[idx], y_train_val_scaled[idx])
        preds_scaled = model.predict(X_test_scaled)
        duration_ms = (time.perf_counter() - start) * 1000

    elif model_name == 'RegressionTree':
        start = time.perf_counter()
        model = DecisionTreeRegressor(max_depth=int(params['dt_depth'])).fit(X_train_val_scaled, y_train_val_scaled)
        preds_scaled = model.predict(X_test_scaled)
        duration_ms = (time.perf_counter() - start) * 1000

    elif model_name == 'XGBoost':
        start = time.perf_counter()
        model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=500).fit(X_train_val_scaled, y_train_val_scaled)
        preds_scaled = model.predict(X_test_scaled)
        duration_ms = (time.perf_counter() - start) * 1000

    elif model_name == 'LightGBM':
        start = time.perf_counter()
        model = lgb.LGBMRegressor(device='gpu', n_estimators=500, verbose=-1).fit(X_train_val_scaled, y_train_val_scaled)
        preds_scaled = model.predict(X_test_scaled)
        duration_ms = (time.perf_counter() - start) * 1000

    elif 'NN' in model_name:
        start = time.perf_counter()
        num_layers = 1 if '1_Layer' in model_name else 3
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(X_train_val_scaled.shape[1],))])
        for i in range(num_layers):
            units = int(params[f'u{i}_{model_name}'])
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_val_scaled, y_train_val_scaled, epochs=30, batch_size=1024, verbose=0)
        preds_scaled = model.predict(X_test_scaled, verbose=0).flatten()
        duration_ms = (time.perf_counter() - start) * 1000

    # Inverse scale predictions for actual error calculation
    y_pred = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_true = y_test.values
    
    calculate_metrics(y_true, y_pred, model_name, duration_ms)

# --- 6. FINAL RESULTS ---
df_results = pd.DataFrame(results_registry).T
df_results.to_csv(FINAL_RESULTS_CSV)
print("\n--- FINAL TEST RESULTS ---")
print(df_results)

joblib.dump(df_results, FINAL_RESULTS_PKL)
