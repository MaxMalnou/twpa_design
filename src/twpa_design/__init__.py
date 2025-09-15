# twpa_design/__init__.py
"""TWPA Design Package"""

import os
from pathlib import Path

__version__ = "0.1.0"

# Package directory
PACKAGE_DIR = Path(__file__).parent

# All directories are inside the package now
NETLISTS_DIR = PACKAGE_DIR / "netlists"
RESULTS_DIR = PACKAGE_DIR / "results"
DESIGNS_DIR = PACKAGE_DIR / "designs"
EXTERNAL_PACKAGES_DIR = PACKAGE_DIR / "external_packages"
EXAMPLES_DIR = PACKAGE_DIR / "examples"
NOTEBOOKS_DIR = PACKAGE_DIR / "notebooks"

# Ensure working directories exist
for dir_path in [NETLISTS_DIR, RESULTS_DIR, DESIGNS_DIR]:
    dir_path.mkdir(exist_ok=True)

# For future use when you're ready to separate user/example directories
def initialize_workspace(base_dir=None):
    """
    Initialize TWPA workspace directories.
    For now, this just ensures directories exist.
    
    Parameters
    ----------
    base_dir : str or Path, optional
        Base directory for workspace. (Not used yet during development)
    """
    # During development, just ensure directories exist
    for dir_path in [NETLISTS_DIR, RESULTS_DIR, DESIGNS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Using package directories:")
    print(f"  - Designs: {DESIGNS_DIR}")
    print(f"  - Netlists: {NETLISTS_DIR}")
    print(f"  - Results: {RESULTS_DIR}")