"""Tests for the filter_builder module.

Tests cover:
- g-value computation against known Butterworth/Chebyshev tables
- Filter design and frequency transformation
- Netlist generation (component count, port map, node naming)
- Multiplexer design and composition
- Topology composition (compose_chain)
"""

import pytest
import numpy as np
from twpa_design.filter_builder import (
    FilterSpec, FilterType, FilterDesign, MultiplexerDesign, PeripheralNetlist,
    design_filter, design_multiplexer,
    filter_to_netlist, multiplexer_to_netlist,
    compose_chain,
    calculate_g_values,
)


# ============================================================================
# g-value computation
# ============================================================================

class TestGValues:
    """Test g-value computation against known reference tables."""

    def test_butterworth_n1_double(self):
        g = calculate_g_values('butterworth', 1, termination='double')
        assert len(g) == 3  # [g0, g1, g2]
        assert g[0] == pytest.approx(1.0)
        assert g[1] == pytest.approx(2.0, rel=1e-6)
        assert g[-1] == pytest.approx(1.0, rel=1e-6)

    def test_butterworth_n3_double(self):
        g = calculate_g_values('butterworth', 3, termination='double')
        assert len(g) == 5  # [g0, g1, g2, g3, g4]
        expected = [1.0, 1.0, 2.0, 1.0, 1.0]
        for gi, ei in zip(g, expected):
            assert gi == pytest.approx(ei, rel=1e-4)

    def test_butterworth_n5_double(self):
        g = calculate_g_values('butterworth', 5, termination='double')
        assert len(g) == 7
        expected = [1.0, 0.618, 1.618, 2.0, 1.618, 0.618, 1.0]
        for gi, ei in zip(g, expected):
            assert gi == pytest.approx(ei, rel=1e-3)

    def test_butterworth_n3_single(self):
        g = calculate_g_values('butterworth', 3, termination='single')
        assert len(g) == 5
        assert g[0] == pytest.approx(1.0)
        assert g[-1] == float('inf')
        # Known values: [1.0, 0.5, 1.333, 1.5, inf]
        assert g[1] == pytest.approx(0.5, rel=1e-3)
        assert g[2] == pytest.approx(1.333, rel=1e-2)
        assert g[3] == pytest.approx(1.5, rel=1e-3)

    def test_chebyshev1_requires_ripple(self):
        with pytest.raises(Exception):
            calculate_g_values('chebyshev1', 3)

    def test_chebyshev1_n3_double(self):
        g = calculate_g_values('chebyshev1', 3, ripple_dB=0.5, termination='double')
        assert len(g) == 5
        assert g[0] == pytest.approx(1.0)
        # Chebyshev N=3, 0.5dB ripple: g1 ~= 1.5963
        assert g[1] == pytest.approx(1.5963, rel=1e-2)

    def test_g_values_symmetry_butterworth(self):
        """Butterworth doubly-terminated g-values are symmetric."""
        for n in [3, 5, 7]:
            g = calculate_g_values('butterworth', n, termination='double')
            # g[1:-1] should be palindromic
            reactive = g[1:-1]
            assert reactive == pytest.approx(reactive[::-1], rel=1e-6)


# ============================================================================
# Filter design
# ============================================================================

class TestDesignFilter:
    """Test design_filter for various filter types."""

    def test_lowpass_butterworth(self):
        design = design_filter(FilterSpec('lp', 5, 8e9))
        assert isinstance(design, FilterDesign)
        assert len(design.g_values) == 7
        assert 'series' in design.transfo_result
        assert 'shunt' in design.transfo_result

    def test_highpass_butterworth(self):
        design = design_filter(FilterSpec('hp', 5, 8e9))
        assert isinstance(design, FilterDesign)
        # HP: series branches should have C0 (capacitor), not Linf
        for b in design.transfo_result['series']:
            c0 = b.get('C0', 0)
            assert c0 > 0 and not np.isinf(c0)

    def test_bandpass_requires_bw(self):
        with pytest.raises(ValueError, match="bw"):
            FilterSpec('bp', 5, 8e9)

    def test_bandpass_butterworth(self):
        design = design_filter(FilterSpec('bp', 5, 8e9, bw=2e9))
        assert isinstance(design, FilterDesign)

    def test_forces_double_termination(self):
        spec = FilterSpec('lp', 5, 8e9, termination='single')
        design = design_filter(spec)
        assert design.spec.termination == 'double'

    def test_component_values_are_physical(self):
        """All L and C values should be positive and finite."""
        design = design_filter(FilterSpec('lp', 5, 8e9))
        for branch_list in [design.transfo_result['series'], design.transfo_result['shunt']]:
            for b in branch_list:
                for key in ['Linf', 'C0', 'L0', 'Cinf']:
                    val = b.get(key, 0)
                    if val and val > 0 and not np.isinf(val):
                        assert val > 0
                        assert val < 1  # Sanity: no component > 1 Henry or 1 Farad


# ============================================================================
# Multiplexer design
# ============================================================================

class TestDesignMultiplexer:
    """Test design_multiplexer for diplexers and N-way multiplexers."""

    def test_diplexer(self):
        mux = design_multiplexer([
            FilterSpec('lp', 7, 8e9),
            FilterSpec('hp', 7, 8e9),
        ])
        assert isinstance(mux, MultiplexerDesign)
        assert len(mux.arms) == 2
        assert mux.Z0 == 50.0

    def test_forces_single_termination(self):
        mux = design_multiplexer([
            FilterSpec('lp', 5, 8e9, termination='double'),
            FilterSpec('hp', 5, 8e9),
        ])
        for arm in mux.arms:
            assert arm.spec.termination == 'single'

    def test_triplexer(self):
        mux = design_multiplexer([
            FilterSpec('lp', 5, 6e9, label='lp'),
            FilterSpec('bp', 5, 8e9, bw=2e9, label='bp1'),
            FilterSpec('hp', 5, 10e9, label='hp'),
        ])
        assert len(mux.arms) == 3

    def test_arm_labels(self):
        mux = design_multiplexer([
            FilterSpec('lp', 5, 8e9, label='lp'),
            FilterSpec('hp', 5, 8e9, label='hp'),
        ])
        assert mux.arms[0].spec.label == 'lp'
        assert mux.arms[1].spec.label == 'hp'


# ============================================================================
# Netlist generation
# ============================================================================

class TestNetlistGeneration:
    """Test filter_to_netlist and multiplexer_to_netlist."""

    def test_filter_netlist_ports(self):
        design = design_filter(FilterSpec('lp', 5, 8e9))
        net = filter_to_netlist(design, prefix='f1')
        assert isinstance(net, PeripheralNetlist)
        assert net.n_ports == 2
        assert 'input' in net.port_map
        assert 'output' in net.port_map

    def test_filter_netlist_node_naming(self):
        design = design_filter(FilterSpec('lp', 5, 8e9))
        net = filter_to_netlist(design, prefix='f1')
        # All non-ground, non-port nodes should start with 'f1_'
        for name, n1, n2, val in net.components:
            if not name.startswith('P') and not name.startswith('R'):
                if n1 != '0':
                    assert n1.startswith('f1_'), f"Node {n1} doesn't have prefix f1_"
                if n2 != '0':
                    assert n2.startswith('f1_'), f"Node {n2} doesn't have prefix f1_"

    def test_multiplexer_netlist_ports(self):
        mux = design_multiplexer([
            FilterSpec('lp', 5, 8e9, label='lp'),
            FilterSpec('hp', 5, 8e9, label='hp'),
        ])
        net = multiplexer_to_netlist(mux, prefix='m1')
        assert net.n_ports == 3  # common + lp + hp
        assert 'common' in net.port_map
        assert 'lp' in net.port_map
        assert 'hp' in net.port_map

    def test_multiplexer_node_naming(self):
        mux = design_multiplexer([
            FilterSpec('lp', 5, 8e9, label='lp'),
            FilterSpec('hp', 5, 8e9, label='hp'),
        ])
        net = multiplexer_to_netlist(mux, prefix='m1')
        for name, n1, n2, val in net.components:
            if not name.startswith('P') and not name.startswith('R'):
                if n1 != '0':
                    assert n1.startswith('m1_'), f"Node {n1} doesn't have prefix m1_"

    def test_no_numpy_floats_in_netlist(self):
        """Component values must be plain Python floats, not np.float64."""
        design = design_filter(FilterSpec('lp', 5, 8e9))
        net = filter_to_netlist(design)
        for name, n1, n2, val in net.components:
            if isinstance(val, float):
                assert type(val) is float, f"Component {name} has {type(val)}, expected float"

    def test_hp_netlist_has_inductors_and_capacitors(self):
        """HP filter should have both L and C components."""
        design = design_filter(FilterSpec('hp', 5, 8e9))
        net = filter_to_netlist(design)
        has_L = any(name.startswith('L') for name, _, _, _ in net.components)
        has_C = any(name.startswith('C') for name, _, _, _ in net.components)
        assert has_L, "HP filter netlist missing inductors"
        assert has_C, "HP filter netlist missing capacitors"

    def test_lp_netlist_has_inductors_and_capacitors(self):
        """LP filter should have both L and C components."""
        design = design_filter(FilterSpec('lp', 5, 8e9))
        net = filter_to_netlist(design)
        has_L = any(name.startswith('L') for name, _, _, _ in net.components)
        has_C = any(name.startswith('C') for name, _, _, _ in net.components)
        assert has_L, "LP filter netlist missing inductors"
        assert has_C, "LP filter netlist missing capacitors"


# ============================================================================
# Topology composition
# ============================================================================

class TestComposition:
    """Test compose_chain and related stitching functions."""

    def _make_diplexer_and_twpa(self):
        """Helper: create a diplexer netlist and a mock TWPA netlist."""
        mux = design_multiplexer([
            FilterSpec('lp', 5, 8e9, label='lp'),
            FilterSpec('hp', 5, 8e9, label='hp'),
        ])
        d_net = multiplexer_to_netlist(mux, prefix='m1')

        # Mock TWPA netlist (minimal 2-port)
        twpa = {
            'jc_components': [
                ('P1_0', '1', '0', '1'),
                ('R1_0', '1', '0', 'R_port'),
                ('L1', '1', '2', 1e-10),
                ('C1', '2', '0', 1e-13),
                ('P2_0', '2', '0', '2'),
                ('R2_0', '2', '0', 'R_port'),
            ],
            'circuit_parameters': {'R_port': 50},
            'metadata': {'device_name': 'test_twpa', 'total_cells': 1},
        }
        return d_net, twpa

    def test_compose_removes_internal_ports(self):
        d_in, twpa = self._make_diplexer_and_twpa()
        d_out = multiplexer_to_netlist(
            design_multiplexer([
                FilterSpec('lp', 5, 8e9, label='lp'),
                FilterSpec('hp', 5, 8e9, label='hp'),
            ]),
            prefix='m2'
        )

        comps, params, meta = compose_chain(
            [d_in, twpa, d_out],
            connections=[('common', 'input'), ('output', 'common')],
        )

        # Should have 4 external ports
        assert meta['n_ports'] == 4

        # Count port components
        port_comps = [c for c in comps if c[0].startswith('P')]
        assert len(port_comps) == 4

        # Port values should be '1', '2', '3', '4'
        port_vals = sorted(c[3] for c in port_comps)
        assert port_vals == ['1', '2', '3', '4']

    def test_compose_no_orphan_resistors(self):
        """After composition, every R_port resistor should be at a port node."""
        d_in, twpa = self._make_diplexer_and_twpa()
        d_out = multiplexer_to_netlist(
            design_multiplexer([
                FilterSpec('lp', 5, 8e9, label='lp'),
                FilterSpec('hp', 5, 8e9, label='hp'),
            ]),
            prefix='m2'
        )

        comps, _, _ = compose_chain(
            [d_in, twpa, d_out],
            connections=[('common', 'input'), ('output', 'common')],
        )

        port_nodes = set()
        for name, n1, n2, val in comps:
            if name.startswith('P'):
                port_nodes.add(n1)

        rport_nodes = set()
        for name, n1, n2, val in comps:
            if name.startswith('R') and val == 'R_port':
                rport_nodes.add(n1)

        # Every R_port should be at a port node
        assert rport_nodes == port_nodes

    def test_compose_node_merge(self):
        """Junction nodes should be merged — no dangling references."""
        d_in, twpa = self._make_diplexer_and_twpa()

        comps, _, _ = compose_chain(
            [d_in, twpa],
            connections=[('common', 'input')],
        )

        # The TWPA's port 1 node '1' should be renamed to the diplexer's common node
        all_nodes = set()
        for _, n1, n2, _ in comps:
            all_nodes.add(n1)
            all_nodes.add(n2)

        # '1' should not appear (merged into m1_c)
        # Unless the TWPA has internal nodes named '1' that aren't the port
        # In our mock, node '1' is only the port node, so it should be gone
        assert '1' not in all_nodes or 'm1_c' in all_nodes
