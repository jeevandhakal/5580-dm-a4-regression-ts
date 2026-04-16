"""
Workflow package - Organized ML pipeline

This package contains the complete machine learning workflow for electricity
consumption prediction, organized in logical execution order.

Structure:
    - notebooks/ - Jupyter notebooks for exploration and analysis
    - scripts/ - Python scripts for final pipeline execution
    - utils/ - Utility functions (utils.py)

Usage:
    1. Exploratory: Start with notebooks in order (1→2→3→4→5)
    2. Production: Run scripts in order (1→2)
    3. Development: Mix notebooks and scripts as needed
"""

__version__ = "1.0.0"
__all__ = [
    "notebooks",
    "scripts",
]

