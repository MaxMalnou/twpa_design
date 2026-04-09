"""Tests for julia_wrapper module (no Julia required).

Tests cover:
- TWPAResults dataclass: S_fund/S_harmonic structure, backward-compat properties
- Save/load: new format and legacy format backward compatibility
- TWPASimulationConfig validation
"""

import pytest
import numpy as np
import tempfile
import os
from twpa_design.julia_wrapper import TWPAResults, TWPASimulationConfig


# ============================================================================
# TWPAResults dataclass
# ============================================================================

class TestTWPAResults:
    """Test TWPAResults S-matrix structure and properties."""

    def _make_results(self, n_ports=2, n_modes=5, n_freqs=100):
        """Create a TWPAResults with random data."""
        return TWPAResults(
            frequencies_GHz=np.linspace(4, 12, n_freqs),
            S_fund=np.random.rand(n_ports, n_ports, n_freqs),
            S_harmonic=np.random.rand(n_modes, n_ports, n_ports, n_freqs),
            quantum_efficiency=np.ones(n_freqs),
            commutation_error=np.zeros(n_freqs),
            modes=[(i,) for i in range(-n_modes//2, n_modes//2 + 1)][:n_modes],
            config=TWPASimulationConfig(signal_port=1, output_port=2),
            port_count=n_ports,
            port_numbers=list(range(1, n_ports + 1)),
        )

    def test_n_ports(self):
        r = self._make_results(n_ports=4)
        assert r.n_ports == 4

    def test_s_param(self):
        r = self._make_results(n_ports=4, n_freqs=50)
        s31 = r.s_param(3, 1)
        assert s31.shape == (50,)
        np.testing.assert_array_equal(s31, r.S_fund[2, 0, :])

    def test_s_harmonic_method(self):
        r = self._make_results(n_ports=4, n_modes=5, n_freqs=50)
        sh = r.s_harmonic(2, 3, 1)
        assert sh.shape == (50,)
        np.testing.assert_array_equal(sh, r.S_harmonic[2, 2, 0, :])

    def test_s_harmonic_none_raises(self):
        r = self._make_results()
        r.S_harmonic = None
        with pytest.raises(ValueError):
            r.s_harmonic(0, 1, 1)

    def test_backward_compat_S21(self):
        r = self._make_results(n_ports=2, n_freqs=50)
        np.testing.assert_array_equal(r.S21, r.S_fund[1, 0, :])

    def test_backward_compat_S11(self):
        r = self._make_results(n_ports=2, n_freqs=50)
        np.testing.assert_array_equal(r.S11, r.S_fund[0, 0, :])

    def test_backward_compat_idler_response(self):
        r = self._make_results(n_ports=2, n_modes=5, n_freqs=50)
        idler = r.idler_response
        # Should be S_harmonic[:, output_port_idx, signal_port_idx, :]
        np.testing.assert_array_equal(idler, r.S_harmonic[:, 1, 0, :])

    def test_backward_compat_idler_none(self):
        r = self._make_results()
        r.S_harmonic = None
        # Should return zeros, not raise
        idler = r.idler_response
        assert idler.shape == r.frequencies_GHz.shape


# ============================================================================
# Save and load
# ============================================================================

class TestSaveLoad:
    """Test save/load round-trip and backward compatibility."""

    def _make_results(self, n_ports=2):
        n_freqs = 50
        n_modes = 3
        return TWPAResults(
            frequencies_GHz=np.linspace(4, 12, n_freqs),
            S_fund=np.random.rand(n_ports, n_ports, n_freqs),
            S_harmonic=np.random.rand(n_modes, n_ports, n_ports, n_freqs),
            quantum_efficiency=np.ones(n_freqs),
            commutation_error=np.zeros(n_freqs),
            modes=[(-1,), (0,), (1,)],
            netlist_name='test_netlist',
            config=TWPASimulationConfig(
                pump_freq_GHz=8.63,
                signal_port=1,
                output_port=2,
            ),
            port_count=n_ports,
            port_numbers=list(range(1, n_ports + 1)),
        )

    def test_save_load_roundtrip(self):
        r = self._make_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.npz')
            r.save(filename=path)
            loaded, meta = TWPAResults.load(path)

            np.testing.assert_array_almost_equal(loaded.S_fund, r.S_fund)
            np.testing.assert_array_almost_equal(loaded.S_harmonic, r.S_harmonic)
            assert loaded.port_count == r.port_count
            assert loaded.port_numbers == r.port_numbers

    def test_save_load_4port(self):
        r = self._make_results(n_ports=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_4port.npz')
            r.save(filename=path)
            loaded, meta = TWPAResults.load(path)

            assert loaded.n_ports == 4
            np.testing.assert_array_almost_equal(loaded.S_fund, r.S_fund)

    def test_load_legacy_format(self):
        """Simulate loading an old-format .npz file with S11/S12/S21/S22 keys."""
        n_freqs = 50
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'legacy.npz')

            # Create legacy-format file manually
            import json
            metadata = {
                'pump_freq_GHz': 8.63,
                'signal_port': 1,
                'output_port': 2,
                'pump_current_A': 2.7e-6,
                'freq_start_GHz': 4.0,
                'freq_stop_GHz': 12.0,
                'freq_step_GHz': 0.1,
            }
            np.savez_compressed(
                path,
                frequencies_GHz=np.linspace(4, 12, n_freqs),
                S11=np.random.rand(n_freqs),
                S12=np.random.rand(n_freqs),
                S21=np.random.rand(n_freqs),
                S22=np.random.rand(n_freqs),
                quantum_efficiency=np.ones(n_freqs),
                commutation_error=np.zeros(n_freqs),
                idler_response=np.random.rand(3, n_freqs),
                backward_idler_response=np.random.rand(3, n_freqs),
                modes=np.array([(-1,), (0,), (1,)]),
                metadata=np.array(json.dumps(metadata)),
            )

            loaded, meta = TWPAResults.load(path)

            # Should reconstruct as 2-port
            assert loaded.n_ports == 2
            assert loaded.S_fund.shape == (2, 2, n_freqs)
            # S_harmonic should be reconstructed from idler data
            assert loaded.S_harmonic is not None
            assert loaded.S_harmonic.shape[0] == 3  # 3 modes

    def test_load_linear_mode(self):
        """Linear mode results have S_harmonic=None."""
        r = self._make_results()
        r.S_harmonic = None
        r.modes = [(0,)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'linear.npz')
            r.save(filename=path)
            loaded, meta = TWPAResults.load(path)
            assert loaded.S_harmonic is None


# ============================================================================
# Load reference results from results/ folder
# ============================================================================

class TestReferenceResults:
    """Load existing .npz reference files and verify structure."""

    def _load_if_exists(self, results_dir, filename):
        path = results_dir / filename
        if not path.exists():
            pytest.skip(f"Reference file not found: {filename}")
        return TWPAResults.load(str(path))

    def test_load_jtwpa_reference(self, results_dir):
        results, meta = self._load_if_exists(
            results_dir, '4wm_jtwpa_2002cells_01_pump8.63GHz_01.npz'
        )
        assert results.frequencies_GHz.shape[0] > 0
        assert results.S_fund.shape[0] >= 2  # at least 2 ports
        assert results.S_fund.shape[2] == results.frequencies_GHz.shape[0]

    def test_load_ktwpa_reference(self, results_dir):
        results, meta = self._load_if_exists(
            results_dir, '4wm_ktwpa_5004cells_01_pump9.10GHz_01.npz'
        )
        assert results.frequencies_GHz.shape[0] > 0


# ============================================================================
# TWPASimulationConfig
# ============================================================================

class TestSimulationConfig:
    """Test TWPASimulationConfig validation and defaults."""

    def test_default_config(self):
        config = TWPASimulationConfig()
        assert config.signal_port == 1
        assert config.output_port == 2
        assert config.pump_port == 1

    def test_frequency_array(self):
        config = TWPASimulationConfig(
            freq_start_GHz=4.0,
            freq_stop_GHz=12.0,
            freq_step_GHz=0.1,
        )
        freqs = config.frequency_array()
        assert freqs[0] == pytest.approx(4.0e9)
        assert freqs[-1] <= 12.0e9 + 0.1e9
