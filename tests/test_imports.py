"""Test that all package modules import without errors."""


def test_import_package():
    import twpa_design
    assert hasattr(twpa_design, '__version__')


def test_import_atl_twpa_designer():
    from twpa_design.atl_twpa_designer import ATLTWPADesigner
    assert ATLTWPADesigner is not None


def test_import_netlist_jc_builder():
    from twpa_design.netlist_JC_builder import NetlistConfig, build_netlist_from_config
    assert NetlistConfig is not None


def test_import_julia_wrapper():
    from twpa_design.julia_wrapper import TWPASimulator, TWPASimulationConfig, TWPAResults
    assert TWPASimulator is not None
    assert TWPASimulationConfig is not None
    assert TWPAResults is not None


def test_import_filter_builder():
    from twpa_design.filter_builder import (
        FilterSpec, FilterDesign, MultiplexerDesign, PeripheralNetlist,
        design_filter, design_multiplexer,
        filter_to_netlist, multiplexer_to_netlist,
        compose_chain, save_peripheral_netlist,
        calculate_g_values,
    )
    assert FilterSpec is not None
    assert design_filter is not None
    assert compose_chain is not None


def test_import_helper_functions():
    from twpa_design.helper_functions import (
        filter_transfo_Foster1,
        filter_transfo_Foster2,
        should_have_zero_at_zero,
    )
    assert filter_transfo_Foster1 is not None


def test_import_plots_params():
    from twpa_design.plots_params import blue, red, linewidth, fontsize
    assert linewidth > 0
