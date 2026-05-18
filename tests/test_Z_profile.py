"""Tests for the impedance-taper profile helpers.

Both `compute_linear_Z_profile` and `compute_klopfenstein_Z_profile` must
satisfy the same endpoint contract: Z(0) = Z_env, Z(taper_cells - 1) = Z_TWPA,
center cells at Z_TWPA, mirrored on the right. These tests pin those properties
plus monotonicity inside the taper.
"""
import numpy as np
import pytest

from twpa_design.helper_functions import (
    compute_linear_Z_profile,
    compute_klopfenstein_Z_profile,
)


@pytest.mark.parametrize("compute", [
    compute_linear_Z_profile,
    compute_klopfenstein_Z_profile,
])
def test_endpoint_match(compute):
    """Z(0) = Z_env and Z(taper_cells-1) = Z_TWPA exactly."""
    Z = compute(Ntot_cell=1000, taper_cells=100, Z_env=50.0, Z_TWPA=200.0)
    assert Z.shape == (1000,)
    np.testing.assert_allclose(Z[0], 50.0, rtol=1e-10)
    np.testing.assert_allclose(Z[99], 200.0, rtol=1e-10)
    # Mirror endpoint
    np.testing.assert_allclose(Z[-1], 50.0, rtol=1e-10)
    np.testing.assert_allclose(Z[-100], 200.0, rtol=1e-10)


@pytest.mark.parametrize("compute", [
    compute_linear_Z_profile,
    compute_klopfenstein_Z_profile,
])
def test_center_uniform(compute):
    """All center cells (between the two tapers) sit exactly at Z_TWPA."""
    Z = compute(Ntot_cell=500, taper_cells=80, Z_env=50.0, Z_TWPA=200.0)
    # Cells 80 through 419 inclusive should be Z_TWPA
    assert np.allclose(Z[80:420], 200.0, rtol=1e-10)


@pytest.mark.parametrize("compute", [
    compute_linear_Z_profile,
    compute_klopfenstein_Z_profile,
])
def test_monotonic_in_taper(compute):
    """Inside each taper, Z is monotonically increasing/decreasing."""
    Z = compute(Ntot_cell=400, taper_cells=60, Z_env=50.0, Z_TWPA=200.0)
    # Left taper: monotone non-decreasing
    assert np.all(np.diff(Z[:60]) >= -1e-12)
    # Right taper: monotone non-increasing
    assert np.all(np.diff(Z[-60:]) <= 1e-12)


@pytest.mark.parametrize("compute", [
    compute_linear_Z_profile,
    compute_klopfenstein_Z_profile,
])
def test_no_taper_when_endpoints_match(compute):
    """If Z_env == Z_TWPA, profile is uniform at Z_TWPA."""
    Z = compute(Ntot_cell=100, taper_cells=20, Z_env=50.0, Z_TWPA=50.0)
    assert np.allclose(Z, 50.0)


@pytest.mark.parametrize("compute", [
    compute_linear_Z_profile,
    compute_klopfenstein_Z_profile,
])
def test_mirror_symmetry(compute):
    """Z is symmetric about the device center."""
    Z = compute(Ntot_cell=300, taper_cells=50, Z_env=50.0, Z_TWPA=200.0)
    np.testing.assert_allclose(Z, Z[::-1], rtol=1e-10)


def test_klopfenstein_auto_A():
    """When A is None, it's auto-computed from max_ripple."""
    # For Z_env=50, Z_TWPA=200: Gamma_0 = 0.5*ln(4) ≈ 0.693
    # With Gamma_m = 0.05: A = arccosh(0.693/0.05) ≈ 3.32
    Z = compute_klopfenstein_Z_profile(
        Ntot_cell=400, taper_cells=80, Z_env=50.0, Z_TWPA=200.0,
        A=None, max_ripple=0.05,
    )
    assert Z[0] == pytest.approx(50.0, rel=1e-10)
    assert Z[79] == pytest.approx(200.0, rel=1e-10)


def test_klopfenstein_explicit_A():
    """User-specified A is honored."""
    Z1 = compute_klopfenstein_Z_profile(
        Ntot_cell=400, taper_cells=80, Z_env=50.0, Z_TWPA=200.0, A=2.0,
    )
    Z2 = compute_klopfenstein_Z_profile(
        Ntot_cell=400, taper_cells=80, Z_env=50.0, Z_TWPA=200.0, A=5.0,
    )
    # Different A → different profiles inside the taper
    assert not np.allclose(Z1[:80], Z2[:80])
    # But both endpoints land exactly
    assert Z1[0] == pytest.approx(50.0, rel=1e-10)
    assert Z2[0] == pytest.approx(50.0, rel=1e-10)
    assert Z1[79] == pytest.approx(200.0, rel=1e-10)
    assert Z2[79] == pytest.approx(200.0, rel=1e-10)


def test_klopfenstein_vs_linear_midpoint():
    """At the midpoint of the taper, Klopfenstein and linear differ.

    For Klopfenstein at the midpoint of the taper, Z(midpoint) = sqrt(Z_env·Z_TWPA)
    (geometric mean). For linear, Z(midpoint) = (Z_env + Z_TWPA)/2 (arithmetic
    mean). Both equal the respective means up to clipping at the endpoints.
    """
    taper = 101  # odd so midpoint is unique cell index = 50
    Z_lin = compute_linear_Z_profile(
        Ntot_cell=400, taper_cells=taper, Z_env=50.0, Z_TWPA=200.0,
    )
    Z_klop = compute_klopfenstein_Z_profile(
        Ntot_cell=400, taper_cells=taper, Z_env=50.0, Z_TWPA=200.0,
    )
    mid = taper // 2
    # Linear midpoint: arithmetic mean
    np.testing.assert_allclose(Z_lin[mid], (50.0 + 200.0) / 2, rtol=1e-10)
    # Klopfenstein midpoint: geometric mean (φ(0, A) = 0 → Z = sqrt(Z_env·Z_TWPA))
    np.testing.assert_allclose(Z_klop[mid], np.sqrt(50.0 * 200.0), rtol=1e-10)
