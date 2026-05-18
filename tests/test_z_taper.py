"""Smoke test: Z-taper-only edge case (impedance transformer without Floquet)."""
from twpa_design.atl_twpa_designer import ATLTWPADesigner


def test_z_taper_only():
    """Z taper alone (no Floquet) produces a valid linear response."""
    d = ATLTWPADesigner({
        'Z0_TWPA_ohm': 100,
        'Z_profile': 'klopfenstein',
        'Z_taper_width': 0.3,
        'floquet_taper': False,
        'stopbands_config_GHz': {27: {'max': 4}},
        'Ntot_cell': 196,
        'fmax_GHz': 15,
        'f_step_MHz': 50,
    }, verbose=False)
    d.run_initial_calculations()
    d.calculate_derived_quantities()
    d.calculate_linear_response()
    import numpy as np
    assert d.Z_taper is True
    assert d.floquet_taper is False
    assert d.floquet_weights[d.Ntot_cell // 2] == 1.0
    assert np.isclose(d.Z_percell[0], 50)
    assert np.isclose(d.Z_percell[d.Ntot_cell // 2], 100)
    assert d.floquet_linear_varies is True
    assert d.S21 is not None


def test_z_taper_explicit_disable_raises():
    """Explicitly disabling Z_taper with mismatched impedances raises."""
    import pytest
    with pytest.raises(ValueError, match="impedance step"):
        ATLTWPADesigner({
            'Z0_TWPA_ohm': 100,
            'Z_taper': False,
        }, verbose=False)


def test_z_taper_auto_disabled_when_matched():
    """Z_taper auto-resolves to False when impedances match."""
    d = ATLTWPADesigner({
        'Z0_TWPA_ohm': 50,
    }, verbose=False)
    assert d.Z_taper is False


def test_floquet_and_z_taper_independent_widths():
    """Two tapers with different widths produce overlapping per-cell regions."""
    d = ATLTWPADesigner({
        'Z0_TWPA_ohm': 100,
        'Z_taper_width': 0.2,
        'Z_profile': 'linear',
        'floquet_taper': True,
        'floquet_taper_width': 0.5,
        'taper_cutoff': False,
        'stopbands_config_GHz': {27: {'max': 4}},
        'Ntot_cell': 196,
        'fmax_GHz': 15,
        'f_step_MHz': 50,
    }, verbose=False)
    d.run_initial_calculations()
    d.calculate_derived_quantities()
    # Per-cell numeric region = union of the two tapers ⇒ width follows floquet (wider).
    expected_width = int(0.5 * 196 / 2)
    Ncpersc = d.Ncpersc_cell
    expected_aligned = round(expected_width / Ncpersc) * Ncpersc
    assert d.width == expected_aligned, f"width {d.width} != expected {expected_aligned}"
