import pandas as pd
import numpy as np
import json

df = pd.read_csv('/home/bhavik/Dropbox/edu/smu/winter/data_mining/a4_regression_ts/results/optuna_trials_final.csv')
model_types = [
    'LinearRegression', 'HoltWinters', 'ARIMA', 'SVR', 
    'RegressionTree', 'XGBoost', 'LightGBM', 
    'NN_1_Layer', 'NN_3_Layer'
]

best_params = {}

for mt in model_types:
    mt_df = df[df['params_model_type'] == mt]
    if not mt_df.empty:
        best_trial = mt_df.loc[mt_df['value'].idxmin()]
        params = {}
        for col in mt_df.columns:
            if col.startswith('params_') and col != 'params_model_type':
                val = best_trial[col]
                if pd.notna(val):
                    # Clean the parameter name
                    pname = col.replace('params_', '')
                    params[pname] = val
        best_params[mt] = params

print(json.dumps(best_params, indent=2))
