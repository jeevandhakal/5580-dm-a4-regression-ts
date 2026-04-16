#!/usr/bin/env python3
"""
Utility functions and constants for the workflow package.

This module defines:
- Path constants for accessing project directories
- Utility functions for timing and other operations
"""
import time
import functools
from pathlib import Path

# ===== PATH CONSTANTS =====
# Calculate ROOT_DIR relative to this file location
# utils.py is at: project_root/src/workflow/scripts/utils.py
# So we go up 3 levels: scripts -> workflow -> src -> root
UTILS_DIR = Path(__file__).resolve().parent
ROOT_DIR = UTILS_DIR.parent.parent.parent  # Up 3 levels to project root

# Project directories
SRC_DIR = ROOT_DIR / "src"
WORKFLOW_DIR = SRC_DIR / "workflow"
NOTEBOOKS_DIR = WORKFLOW_DIR / "notebooks"
SCRIPTS_DIR = WORKFLOW_DIR / "scripts"
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
REPORT_DIR = ROOT_DIR / "report"
GPU_SETUP_DIR = SRC_DIR / "gpu_setup"

# Data files
DATA_FILE = DATA_DIR / "electricity_prediction.csv"
ELECTRICITY_PICKLE = DATA_DIR / "electricity_data_split.pkl"

# Results files
FINAL_RESULTS_CSV = RESULTS_DIR / "final_test_results.csv"
FINAL_RESULTS_PKL = RESULTS_DIR / "final_df_results.pkl"
OPTUNA_TRIALS_CSV = RESULTS_DIR / "optuna_trials_final.csv"
OPTUNA_STUDY_PKL = RESULTS_DIR / "electricity_study.pkl"
EXTRACT_PARAMS_JSON = RESULTS_DIR / "extract_params.json"

# Verify critical directories exist
def verify_directories():
    """Verify that all required directories exist."""
    required_dirs = [ROOT_DIR, DATA_DIR, RESULTS_DIR, SRC_DIR, WORKFLOW_DIR]
    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Required directory not found: {directory}")


# ===== UTILITY FUNCTIONS =====
def time_operation(func):
    """Decorator to measure execution time of a function.

    Returns:
        tuple: (result, duration_ms) where duration_ms is execution time in milliseconds
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration_ms = (end - start) * 1000
        return result, duration_ms
    return wrapper
