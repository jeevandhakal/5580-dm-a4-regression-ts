"""
Notebook utility functions for importing paths and constants.

Use this in notebooks to get access to all project paths and utilities.

Example:
    import sys
    sys.path.append('../../scripts')
    from notebook_utils import *

    # Now you have access to:
    # ROOT_DIR, DATA_FILE, RESULTS_DIR, etc.
"""
import sys
from pathlib import Path

# Add scripts directory to path for importing utils
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import all constants and utilities from utils.py
from utils import (
    # Path constants
    ROOT_DIR,
    SRC_DIR,
    WORKFLOW_DIR,
    NOTEBOOKS_DIR,
    SCRIPTS_DIR,
    DATA_DIR,
    RESULTS_DIR,
    REPORT_DIR,
    GPU_SETUP_DIR,
    # Data files
    DATA_FILE,
    ELECTRICITY_PICKLE,
    # Results files
    FINAL_RESULTS_CSV,
    FINAL_RESULTS_PKL,
    OPTUNA_TRIALS_CSV,
    OPTUNA_STUDY_PKL,
    EXTRACT_PARAMS_JSON,
    # Functions
    verify_directories,
    time_operation,
)

__all__ = [
    "ROOT_DIR",
    "SRC_DIR",
    "WORKFLOW_DIR",
    "NOTEBOOKS_DIR",
    "SCRIPTS_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "REPORT_DIR",
    "GPU_SETUP_DIR",
    "DATA_FILE",
    "ELECTRICITY_PICKLE",
    "FINAL_RESULTS_CSV",
    "FINAL_RESULTS_PKL",
    "OPTUNA_TRIALS_CSV",
    "OPTUNA_STUDY_PKL",
    "EXTRACT_PARAMS_JSON",
    "verify_directories",
    "time_operation",
]

