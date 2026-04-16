"""
Notebook utility exports for paths/constants used in workflow notebooks.

Preferred usage (package context):
    from src.workflow import notebook_utils as nu

Fallback usage (notebook/direct file context) is also supported.
"""

from pathlib import Path
import sys


def _import_utils_symbols():
    """Import constants/functions from workflow scripts utils.

    - First, try package-relative import for clean static analysis.
    - Fallback to direct path import for ad-hoc notebook execution contexts.
    """
    try:
        # Package context: src.workflow.notebook_utils
        from .scripts.utils import (  # type: ignore
            ROOT_DIR,
            SRC_DIR,
            WORKFLOW_DIR,
            NOTEBOOKS_DIR,
            SCRIPTS_DIR,
            DATA_DIR,
            RESULTS_DIR,
            REPORT_DIR,
            GPU_SETUP_DIR,
            DATA_FILE,
            ELECTRICITY_PICKLE,
            FINAL_RESULTS_CSV,
            FINAL_RESULTS_PKL,
            OPTUNA_TRIALS_CSV,
            OPTUNA_STUDY_PKL,
            EXTRACT_PARAMS_JSON,
            verify_directories,
            time_operation,
        )
        return {
            "ROOT_DIR": ROOT_DIR,
            "SRC_DIR": SRC_DIR,
            "WORKFLOW_DIR": WORKFLOW_DIR,
            "NOTEBOOKS_DIR": NOTEBOOKS_DIR,
            "SCRIPTS_DIR": SCRIPTS_DIR,
            "DATA_DIR": DATA_DIR,
            "RESULTS_DIR": RESULTS_DIR,
            "REPORT_DIR": REPORT_DIR,
            "GPU_SETUP_DIR": GPU_SETUP_DIR,
            "DATA_FILE": DATA_FILE,
            "ELECTRICITY_PICKLE": ELECTRICITY_PICKLE,
            "FINAL_RESULTS_CSV": FINAL_RESULTS_CSV,
            "FINAL_RESULTS_PKL": FINAL_RESULTS_PKL,
            "OPTUNA_TRIALS_CSV": OPTUNA_TRIALS_CSV,
            "OPTUNA_STUDY_PKL": OPTUNA_STUDY_PKL,
            "EXTRACT_PARAMS_JSON": EXTRACT_PARAMS_JSON,
            "verify_directories": verify_directories,
            "time_operation": time_operation,
        }
    except Exception:
        # Fallback for notebook/direct script execution.
        scripts_dir = Path(__file__).resolve().parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from utils import (  # type: ignore
            ROOT_DIR,
            SRC_DIR,
            WORKFLOW_DIR,
            NOTEBOOKS_DIR,
            SCRIPTS_DIR,
            DATA_DIR,
            RESULTS_DIR,
            REPORT_DIR,
            GPU_SETUP_DIR,
            DATA_FILE,
            ELECTRICITY_PICKLE,
            FINAL_RESULTS_CSV,
            FINAL_RESULTS_PKL,
            OPTUNA_TRIALS_CSV,
            OPTUNA_STUDY_PKL,
            EXTRACT_PARAMS_JSON,
            verify_directories,
            time_operation,
        )
        return {
            "ROOT_DIR": ROOT_DIR,
            "SRC_DIR": SRC_DIR,
            "WORKFLOW_DIR": WORKFLOW_DIR,
            "NOTEBOOKS_DIR": NOTEBOOKS_DIR,
            "SCRIPTS_DIR": SCRIPTS_DIR,
            "DATA_DIR": DATA_DIR,
            "RESULTS_DIR": RESULTS_DIR,
            "REPORT_DIR": REPORT_DIR,
            "GPU_SETUP_DIR": GPU_SETUP_DIR,
            "DATA_FILE": DATA_FILE,
            "ELECTRICITY_PICKLE": ELECTRICITY_PICKLE,
            "FINAL_RESULTS_CSV": FINAL_RESULTS_CSV,
            "FINAL_RESULTS_PKL": FINAL_RESULTS_PKL,
            "OPTUNA_TRIALS_CSV": OPTUNA_TRIALS_CSV,
            "OPTUNA_STUDY_PKL": OPTUNA_STUDY_PKL,
            "EXTRACT_PARAMS_JSON": EXTRACT_PARAMS_JSON,
            "verify_directories": verify_directories,
            "time_operation": time_operation,
        }


_symbols = _import_utils_symbols()

# Explicit exports for static analyzers and editor symbol resolution.
ROOT_DIR = _symbols["ROOT_DIR"]
SRC_DIR = _symbols["SRC_DIR"]
WORKFLOW_DIR = _symbols["WORKFLOW_DIR"]
NOTEBOOKS_DIR = _symbols["NOTEBOOKS_DIR"]
SCRIPTS_DIR = _symbols["SCRIPTS_DIR"]
DATA_DIR = _symbols["DATA_DIR"]
RESULTS_DIR = _symbols["RESULTS_DIR"]
REPORT_DIR = _symbols["REPORT_DIR"]
GPU_SETUP_DIR = _symbols["GPU_SETUP_DIR"]
DATA_FILE = _symbols["DATA_FILE"]
ELECTRICITY_PICKLE = _symbols["ELECTRICITY_PICKLE"]
FINAL_RESULTS_CSV = _symbols["FINAL_RESULTS_CSV"]
FINAL_RESULTS_PKL = _symbols["FINAL_RESULTS_PKL"]
OPTUNA_TRIALS_CSV = _symbols["OPTUNA_TRIALS_CSV"]
OPTUNA_STUDY_PKL = _symbols["OPTUNA_STUDY_PKL"]
EXTRACT_PARAMS_JSON = _symbols["EXTRACT_PARAMS_JSON"]
verify_directories = _symbols["verify_directories"]
time_operation = _symbols["time_operation"]

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

