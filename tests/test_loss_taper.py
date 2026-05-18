"""Dielectric loss is applied uniformly across tapered designs.

Covers:
- Symbolic center-supercell capacitors (handled at save time)
- Inline `CTLsec_<idx>` shunt caps in the taper region (`is_windowed=True` path)
- Inline filter shunt caps in the taper region (`force_numeric=True` path,
  previously missed by the loss wrapper).
"""
import os
import re
import tempfile


def _build_loss_netlist(custom_params, loss_tangent=2e-4):
    """Run the designer + netlist builder with dielectric loss enabled,
    write to a temp file, and return the netlist text plus the designer."""
    from twpa_design.atl_twpa_designer import ATLTWPADesigner
    from twpa_design.netlist_JC_builder import (
        load_design_parameters, prepare_workspace_variables,
        build_netlist, save_netlist_to_file, NetlistConfig,
    )

    d = ATLTWPADesigner(custom_params, verbose=False)
    d.run_initial_calculations()
    d.calculate_derived_quantities()
    d.calculate_linear_response()
    d.calculate_phase_matching()

    tmpdir = tempfile.mkdtemp()
    design_file = d.export_parameters(output_dir=tmpdir)
    config = NetlistConfig(
        use_taylor_insteadof_JJ=True,
        enable_dielectric_loss=True,
        loss_tangent=loss_tangent,
    )
    design_data = load_design_parameters(design_file)
    prepared = prepare_workspace_variables(design_data, config)
    builder, _ = build_netlist(prepared)

    out_path = os.path.join(tmpdir, "loss_test.py")
    save_netlist_to_file(builder, out_path, {
        'device_name': 'loss_test',
        'total_cells': d.Ntot_cell,
        'cells_per_supercell': d.Ncpersc_cell,
        'num_supercells': d.Nsc_cell,
        'dispersion_type': d.dispersion_type,
        'nonlinearity': d.nonlinearity,
        'dielectric_loss_enabled': True,
        'loss_tangent': loss_tangent,
    })
    with open(out_path) as f:
        return f.read(), d


def test_loss_applied_in_taper_and_center_periodic():
    """Periodic dispersion (no filters): loss must reach both inline taper caps
    and the symbolic center supercell parameters."""
    text, _ = _build_loss_netlist({
        'floquet_taper': True,
        'floquet_profile': 'gaussian',
        'floquet_taper_width': 0.3,
        'stopbands_config_GHz': {27: {'max': 4}},
        'Ntot_cell': 196,
        'fmax_GHz': 15,
        'f_step_MHz': 50,
    })

    components_section, params_section = text.split("circuit_parameters", 1)
    inline_loss = components_section.count("/(1+2.000000e-04im)")
    param_loss = params_section.count("/(1+2.000000e-04im)")

    assert inline_loss > 0, "taper-region CTLsec caps must carry inline loss"
    assert param_loss > 0, "center symbolic C* params must carry loss at save time"


def test_loss_applied_to_taper_filter_caps_dispersion_both():
    """`dispersion='both'`: filter shunt caps emitted inline in taper cells
    (via the `force_numeric=True` path) must also carry the loss wrapper.
    This is the path that was missed before the create_symbolic_value fix."""
    text, d = _build_loss_netlist({
        'floquet_taper': True,
        'floquet_taper_width': 0.5,
        'f_zeros_GHz': 9,
        'f_poles_GHz': 8.85,
        'stopbands_config_GHz': {27: {'max': 4}},
        'Ntot_cell': 200,
        'fmax_GHz': 15,
        'f_step_MHz': 50,
        'select_one_form': 'L',
        'Foster_form_L': 2,
        'Foster_form_C': 1,
    })

    components_section = text.split("circuit_parameters", 1)[0]
    filter_cap_pattern = re.compile(
        r"\('C(?:0|i|inf)(?:CF|LF)[12]_(\d+)'.*?,\s*'([^']+)'\s*\)")
    taper_filter_caps = []
    for line in components_section.splitlines():
        m = filter_cap_pattern.search(line)
        if not m:
            continue
        idx = int(m.group(1))
        value_str = m.group(2)
        is_inline_numeric = "e-" in value_str or "e+" in value_str
        if not is_inline_numeric:
            continue
        is_taper = (idx < d.width) or (idx >= d.Ntot_cell - d.width)
        if is_taper:
            taper_filter_caps.append((idx, value_str))

    assert len(taper_filter_caps) > 0, "no inline taper filter caps found — bad fixture"
    missing = [(idx, val) for idx, val in taper_filter_caps if "/(1+" not in val]
    assert not missing, (
        f"{len(missing)} taper filter caps emitted without dielectric loss "
        f"(first: cell {missing[0][0]}, value {missing[0][1]!r})")
