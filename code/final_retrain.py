
import os
import warnings
import logging
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# --- SILENCE ENVIRONMENT ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# --- UTILS ---
def time_operation(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration_ms = (end - start) * 1000
        return result, duration_ms
    return wrapper

def prepare_data(df):
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

# --- 1. LOAD DATA ---
root_path = Path('/home/bhavik/Dropbox/edu/smu/winter/data_mining/a4_regression_ts')
data_path = root_path / "data" / "electricity_prediction.csv"
results_path = root_path / "results"

df = pd.read_csv(data_path, header=None)
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
with open(root_path / 'extract_params.json', 'w') as f:
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
df_results.to_csv(results_path / "final_test_results.csv")
print("\n--- FINAL TEST RESULTS ---")
print(df_results)

joblib.dump(df_results, results_path / "final_df_results.pkl")
