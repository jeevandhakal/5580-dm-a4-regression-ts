# Workflow Quick Reference

## TL;DR - Get Started Now

### For Exploration:
```bash
cd /home/bhavik/Dropbox/edu/smu/winter/data_mining/a4_regression_ts
uv sync
jupyter lab
# Open: src/workflow/notebooks/1_connection_preprocessing.ipynb
# Then: 2→3→4→5 in order
```

### For Production:
```bash
cd /home/bhavik/Dropbox/edu/smu/winter/data_mining/a4_regression_ts
uv sync
uv run src/workflow/scripts/1_extract_best_params.py
uv run src/workflow/scripts/2_final_retrain_and_evaluate.py
cat results/final_test_results.csv
```

---

## Execution Order

### Notebooks (Exploratory)
1. **1_connection_preprocessing** - Load & explore data (5 min)
2. **2_linear_regression_baseline** - Classical models (10 min)
3. **3_advanced_models_and_eda** - Advanced features & ML (30 min)
4. **4_hyperparameter_tuning** - Optuna optimization (2-10 hrs) ⚠️ LONG
5. **5_analyze_tuning_results** - Visualize results (5 min)
6. **6_cuda_gpu_setup_reference** - GPU info (optional)

### Scripts (Production)
1. **1_extract_best_params.py** - Get best hyperparameters (10 sec)
2. **2_final_retrain_and_evaluate.py** - Train & evaluate (10 min)

---

## File Locations

```
src/workflow/
├── README.md                          ← Full documentation
├── QUICK_REFERENCE.md                 ← This file
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

## Quick Facts

- ✅ **Organized**: Numbered files indicate execution order
- ✅ **Complete**: 6 notebooks + 2 scripts cover full pipeline
- ✅ **Documented**: Each notebook/script has clear docstrings
- ✅ **Flexible**: Run all, or just production scripts
- ✅ **GPU-Ready**: Scripts support GPU acceleration

---

## Common Commands

```bash
# From project root
cd /home/bhavik/Dropbox/edu/smu/winter/data_mining/a4_regression_ts

# Sync dependencies
uv sync

# Start Jupyter
jupyter lab

# Run extraction script
uv run src/workflow/scripts/1_extract_best_params.py

# Run final training
uv run src/workflow/scripts/2_final_retrain_and_evaluate.py

# Check GPU status
uv run src/gpu_setup/tools/verify_setup.py

# See full documentation
cat src/workflow/README.md
```

---

## Expected Results

After running everything:
```
results/
├── final_test_results.csv     (metrics)
├── extract_params.json        (best params)
├── final_df_results.pkl       (results)
├── optuna_trials_final.csv    (tuning history)
└── plots/                     (visualizations)
```

---

## Tips

1. **Skip tuning if needed**: If tuning already done, jump to scripts
2. **Use GPU**: Sets training ~5-10x faster
3. **Check dependencies**: Run `uv sync` first
4. **GPU help**: See `src/gpu_setup/docs/TROUBLESHOOTING.md`

---

**Pick a starting point above and get going!** 🚀

