"""Test that all example and module files have valid Python syntax.

This catches syntax errors introduced during editing without needing
to run the full examples (which may require Julia).
"""

import ast
import pytest
from pathlib import Path


def _get_py_files(directory: Path):
    """Collect all .py files in a directory."""
    return sorted(directory.glob("*.py"))


class TestSyntax:
    """Verify all .py files parse without syntax errors."""

    def test_module_syntax(self, package_dir):
        """All modules in src/twpa_design/ should parse."""
        for py_file in _get_py_files(package_dir):
            source = py_file.read_text(encoding='utf-8')
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")

    def test_example_syntax(self, examples_dir):
        """All examples should parse."""
        for py_file in _get_py_files(examples_dir):
            source = py_file.read_text(encoding='utf-8')
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")

    def test_design_syntax(self, designs_dir):
        """All design files should parse."""
        for py_file in _get_py_files(designs_dir):
            source = py_file.read_text(encoding='utf-8')
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")

    def test_netlist_syntax(self, netlists_dir):
        """All netlist files should parse."""
        for py_file in _get_py_files(netlists_dir):
            source = py_file.read_text(encoding='utf-8')
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")


class TestExampleFilesExist:
    """Verify expected example files are present."""

    EXPECTED_EXAMPLES = [
        'atl_twpa_designer_example.py',
        'atl_twpa_plotter_example.py',
        'netlist_JC_builder_example.py',
        'julia_wrapper_example.py',
        'julia_plotter_example.py',
        'diplexer_twpa_example.py',
        'twpa_twpa_example.py',
    ]

    def test_expected_examples_exist(self, examples_dir):
        existing = {f.name for f in _get_py_files(examples_dir)}
        for name in self.EXPECTED_EXAMPLES:
            assert name in existing, f"Missing example: {name}"


class TestFloquetTaper:
    """Verify Floquet taper feature works end-to-end."""

    def test_floquet_designer_gaussian(self):
        """Floquet taper with Gaussian profile runs without errors."""
        from twpa_design.atl_twpa_designer import ATLTWPADesigner
        import numpy as np
        d = ATLTWPADesigner({
            'floquet_taper': True,
            'floquet_profile': 'gaussian',
            'floquet_taper_width': 0.3,
            'stopbands_config_GHz': {27: {'max': 4}},
            'Ntot_cell': 196,
            'fmax_GHz': 15,
            'f_step_MHz': 50,
        }, verbose=False)
        d.run_initial_calculations()
        d.calculate_derived_quantities()
        d.calculate_linear_response()
        d.calculate_phase_matching()
        assert d.S21 is not None
        assert hasattr(d, 'floquet_weights')
        assert d.floquet_weights.shape[0] == d.Ntot_cell
        assert np.allclose(d.floquet_weights, d.floquet_weights[::-1])

    def test_floquet_designer_tukey(self):
        """Floquet taper with Tukey profile has center matrix_power shortcut."""
        from twpa_design.atl_twpa_designer import ATLTWPADesigner
        d = ATLTWPADesigner({
            'floquet_taper': True,
            'floquet_profile': 'tukey',
            'floquet_taper_width': 0.3,
            'stopbands_config_GHz': {27: {'max': 4}},
            'Ntot_cell': 196,
            'fmax_GHz': 15,
            'f_step_MHz': 50,
        }, verbose=False)
        d.run_initial_calculations()
        d.calculate_derived_quantities()
        assert d.width > 0
        assert d.n_periodic_sc > 0

    def test_floquet_mutual_exclusivity(self):
        """Floquet taper rejects conflicting apodization settings."""
        from twpa_design.atl_twpa_designer import ATLTWPADesigner
        import pytest
        with pytest.raises(ValueError, match="mutually exclusive"):
            ATLTWPADesigner({
                'floquet_taper': True,
                'window_type': 'tukey',
                'alpha': 0.1,
            }, verbose=False)

    def test_floquet_netlist_roundtrip(self):
        """Design with Floquet exports and builds a netlist."""
        from twpa_design.atl_twpa_designer import ATLTWPADesigner
        from twpa_design.netlist_JC_builder import (
            load_design_parameters, prepare_workspace_variables,
            build_netlist, NetlistConfig
        )
        import tempfile
        d = ATLTWPADesigner({
            'floquet_taper': True,
            'floquet_profile': 'tukey',
            'floquet_taper_width': 0.3,
            'stopbands_config_GHz': {27: {'max': 4}},
            'Ntot_cell': 196,
            'fmax_GHz': 15,
            'f_step_MHz': 50,
        }, verbose=False)
        d.run_initial_calculations()
        d.calculate_derived_quantities()
        d.calculate_linear_response()
        d.calculate_phase_matching()
        tmpdir = tempfile.mkdtemp()
        design_file = d.export_parameters(output_dir=tmpdir)
        config = NetlistConfig(use_taylor_insteadof_JJ=True)
        design_data = load_design_parameters(design_file)
        prepared = prepare_workspace_variables(design_data, config)
        builder, stats = build_netlist(prepared)
        assert stats['total_components'] > 0

    def test_default_no_floquet(self):
        """Default config (no Floquet) still works unchanged."""
        from twpa_design.atl_twpa_designer import ATLTWPADesigner
        d = ATLTWPADesigner({
            'stopbands_config_GHz': {27: {'max': 4}},
            'Ntot_cell': 196,
            'fmax_GHz': 15,
            'f_step_MHz': 50,
        }, verbose=False)
        d.run_initial_calculations()
        d.calculate_derived_quantities()
        d.calculate_linear_response()
        assert not hasattr(d, 'floquet_weights') or d.floquet_weights is None


class TestNetlistFilesExist:
    """Verify key netlist files are present."""

    EXPECTED_NETLISTS = [
        '4wm_jtwpa_2002cells_01.py',
    ]

    def test_expected_netlists_exist(self, netlists_dir):
        existing = {f.name for f in _get_py_files(netlists_dir)}
        for name in self.EXPECTED_NETLISTS:
            assert name in existing, f"Missing netlist: {name}"
