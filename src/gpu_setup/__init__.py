#!/usr/bin/env python3
"""
GPU Setup Package - Master Configuration and Installation Tool

This module orchestrates all GPU setup and recovery operations.
"""

import sys
from pathlib import Path

def main():
    """Display GPU setup options and status."""
    print("=" * 70)
    print("GPU SETUP PACKAGE - Master Tool")
    print("=" * 70)
    print()
    print("Available tools:")
    print()
    print("1. Quick Setup Check:")
    print("   python3 src/gpu_setup/tools/verify_setup.py")
    print()
    print("2. Full GPU Diagnosis:")
    print("   python3 src/gpu_setup/tools/diagnose_cuda.py")
    print()
    print("3. GPU Recovery & Configuration:")
    print("   python3 src/gpu_setup/tools/gpu_recovery.py [--cleanup]")
    print()
    print("4. Cleanup Broken Symlinks:")
    print("   bash src/gpu_setup/scripts/cleanup_broken_links.sh")
    print()
    print("5. CUDA Path Fixer:")
    print("   bash src/gpu_setup/scripts/fix_cuda_paths.sh")
    print()
    print("Documentation:")
    print("   - src/gpu_setup/README.md")
    print("   - src/gpu_setup/docs/TROUBLESHOOTING.md")
    print("   - src/gpu_setup/docs/QUICK_START.md")
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()

