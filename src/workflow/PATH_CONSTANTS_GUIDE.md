# Path Constants Guide for Notebooks

This guide explains how to use the path constants in Jupyter notebooks.

## Setup (Add This to First Cell)

```python
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path('../../scripts').resolve()))

from utils import (
    ROOT_DIR, DATA_DIR, DATA_FILE, RESULTS_DIR,
    OPTUNA_TRIALS_CSV, OPTUNA_STUDY_PKL, EXTRACT_PARAMS_JSON,
    verify_directories
)

# Verify environment
verify_directories()
```

## Available Path Constants

### Project Directories
- `ROOT_DIR` - Project root directory
- `SRC_DIR` - src/ directory
- `WORKFLOW_DIR` - workflow/ directory
- `NOTEBOOKS_DIR` - notebooks/ directory (current)
- `SCRIPTS_DIR` - scripts/ directory
- `DATA_DIR` - data/ directory
- `RESULTS_DIR` - results/ directory
- `REPORT_DIR` - report/ directory
- `GPU_SETUP_DIR` - gpu_setup/ directory

### Data Files
- `DATA_FILE` - Main dataset (electricity_prediction.csv)
- `ELECTRICITY_PICKLE` - Preprocessed pickle file

### Results Files
- `FINAL_RESULTS_CSV` - final_test_results.csv
- `FINAL_RESULTS_PKL` - final_df_results.pkl
- `OPTUNA_TRIALS_CSV` - optuna_trials_final.csv
- `OPTUNA_STUDY_PKL` - electricity_study.pkl
- `EXTRACT_PARAMS_JSON` - extract_params.json

## Example Usage in Notebooks

### Loading Data
```python
# Instead of:
# df = pd.read_csv('/home/bhavik/Dropbox/.../electricity_prediction.csv')

# Use:
import pandas as pd
df = pd.read_csv(DATA_FILE)
```

### Saving Results
```python
import matplotlib.pyplot as plt

# Instead of:
# plt.savefig('/home/bhavik/Dropbox/.../results/plot.png')

# Use:
plot_path = RESULTS_DIR / 'my_plot.png'
plt.savefig(plot_path)
```

### Saving Pickle Files
```python
import joblib

# Instead of:
# joblib.dump(data, '/home/bhavik/Dropbox/.../results/data.pkl')

# Use:
joblib.dump(data, RESULTS_DIR / 'my_data.pkl')
```

### Creating New Files in Results
```python
# Create new result files
new_file = RESULTS_DIR / 'my_new_results.csv'
df.to_csv(new_file, index=False)

print(f"Saved to: {new_file}")  # Prints full path
```

## Notebook Setup Template

Copy this to the first cell of each notebook:

```python
# ===== SETUP PATHS & UTILITIES =====
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path('../../scripts').resolve()))

from utils import (
    ROOT_DIR, DATA_DIR, DATA_FILE, RESULTS_DIR, REPORT_DIR,
    OPTUNA_TRIALS_CSV, OPTUNA_STUDY_PKL, EXTRACT_PARAMS_JSON,
    FINAL_RESULTS_CSV, FINAL_RESULTS_PKL,
    verify_directories, time_operation
)

# Verify environment
verify_directories()

print(f"✓ Paths loaded")
print(f"  Root: {ROOT_DIR}")
print(f"  Data: {DATA_FILE}")
print(f"  Results: {RESULTS_DIR}")
```

## Important Notes

1. **Path Type**: All paths are `pathlib.Path` objects, which work seamlessly with pandas, matplotlib, and other libraries
2. **Absolute Paths**: All paths are absolute, so notebooks can be run from any directory
3. **Cross-Platform**: Using `pathlib` ensures paths work on Windows, Mac, and Linux
4. **String Conversion**: If you need a string path, use `str(path)` (e.g., `str(DATA_FILE)`)

## Troubleshooting

**ImportError when importing utils?**
```python
# Make sure you're running from a notebook in:
# src/workflow/notebooks/

# The path should resolve to:
# src/workflow/scripts/utils.py
```

**FileNotFoundError?**
```python
# Add this to debug:
print(DATA_FILE)  # Should print full path to file
print(DATA_FILE.exists())  # Should print True

# If False, the file is missing
```

---

**Use these constants in all notebooks for consistent, maintainable path handling!**

