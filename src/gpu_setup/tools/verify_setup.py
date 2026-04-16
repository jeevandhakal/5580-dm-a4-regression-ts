#!/usr/bin/env python3
"""Verify project setup and dependencies."""
import sys
import os
import py_compile
from pathlib import Path
import glob

# Auto-wire NVIDIA runtime libraries from the uv venv before importing TensorFlow.
def _configure_gpu_runtime_paths() -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    venv_site = next((repo_root / ".venv" / "lib").glob("python*/site-packages"), None)
    if not venv_site:
        return []
    nvidia_lib_dirs = sorted(glob.glob(str(venv_site / "nvidia" / "*" / "lib")))
    if not nvidia_lib_dirs:
        return []

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = ["/usr/lib/x86_64-linux-gnu", *nvidia_lib_dirs]
    if existing:
        parts.append(existing)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    return nvidia_lib_dirs

wired_libs = _configure_gpu_runtime_paths()

print("=" * 70)
print("PROJECT SETUP VERIFICATION")
print("=" * 70)

if wired_libs:
    print(f"\nGPU runtime paths wired: {len(wired_libs)} NVIDIA lib dirs")

# 1. Test imports
print("\n1. Testing imports...")
try:
    import tensorflow as tf
    print(f"   ✓ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"   ✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    import xgboost as xgb
    print(f"   ✓ XGBoost {xgb.__version__}")
except ImportError as e:
    print(f"   ✗ XGBoost import failed: {e}")

try:
    import lightgbm as lgb
    print(f"   ✓ LightGBM {lgb.__version__}")
except ImportError as e:
    print(f"   ✗ LightGBM import failed: {e}")

# 2. Test Python syntax
print("\n2. Checking Python files...")
try:
    # Check for workflow scripts (new location)
    script_path = Path(__file__).parent.parent.parent / 'workflow' / 'scripts' / '2_final_retrain_and_evaluate.py'
    py_compile.compile(str(script_path), doraise=True)
    print("   ✓ Workflow scripts syntax OK")
except (py_compile.PyCompileError, FileNotFoundError) as e:
    print(f"   ⚠ Warning: Could not verify workflow scripts: {e}")

# 3. Check GPU
print("\n3. GPU Status:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✓ GPU Available: {len(gpus)} device(s)")
    for gpu in gpus:
        print(f"      - {gpu}")
else:
    print("   ℹ No GPU detected (CPU mode will be used)")

# 4. Check data
print("\n4. Data Files:")
data_file = Path(__file__).parent.parent.parent.parent / "data" / "electricity_prediction.csv"
if data_file.exists():
    size_mb = data_file.stat().st_size / (1024 * 1024)
    print(f"   ✓ electricity_prediction.csv ({size_mb:.1f} MB)")
else:
    print(f"   ✗ Data file not found: {data_file}")

# 5. Check results directory
results_dir = Path(__file__).parent.parent.parent.parent / "results"
if results_dir.exists():
    print("   ✓ Results directory exists")
else:
    print("   ℹ Results directory doesn't exist (will be created on run)")

print("\n" + "=" * 70)
print("✓ SETUP VERIFICATION COMPLETE")
print("=" * 70)
print("\nYou can now run:")
print("  uv run src/workflow/scripts/1_extract_best_params.py")
print("  uv run src/workflow/scripts/2_final_retrain_and_evaluate.py")
print("  jupyter lab  # Then open: src/workflow/notebooks/")
print("\nFor more info:")
print("  cat README.md")
print("  cat src/workflow/QUICK_REFERENCE.md")

