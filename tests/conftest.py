"""Shared fixtures for twpa_design tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def package_dir():
    """Path to the twpa_design package source."""
    return Path(__file__).parent.parent / "src" / "twpa_design"


@pytest.fixture
def results_dir(package_dir):
    """Path to the results/ folder with reference .npz files."""
    return package_dir / "results"


@pytest.fixture
def netlists_dir(package_dir):
    """Path to the netlists/ folder."""
    return package_dir / "netlists"


@pytest.fixture
def examples_dir(package_dir):
    """Path to the examples/ folder."""
    return package_dir / "examples"


@pytest.fixture
def designs_dir(package_dir):
    """Path to the designs/ folder."""
    return package_dir / "designs"
