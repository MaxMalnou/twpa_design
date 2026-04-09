"""
Example: TWPA-TWPA Cascade with Diplexers

This example demonstrates a cascaded TWPA-TWPA topology:
  diplexer_in -> TWPA1 -> diplexer_center -> TWPA2 -> diplexer_out

The center diplexer is special:
  - Its LP arms are terminated with 50 Ohm loads (signal path ends)
  - Its HP arm is a doubly open-terminated filter connecting the two
    inner common ports (pump/idler passes through)

The resulting device has 4 external ports:
  Port 1: Input diplexer - LP arm (signal input)
  Port 2: Input diplexer - HP arm (pump + idler input)
  Port 3: Output diplexer - LP arm (signal output)
  Port 4: Output diplexer - HP arm (pump + idler output)

The two internal LP ports (signal path between TWPAs) are terminated
with 50 Ohm loads, so there is no signal path between the two TWPAs.
The pump/idler propagates through both TWPAs via the center HP filter.

Prerequisites:
  - TWPA netlist: 4wm_jtwpa_2002cells_01.py (in netlists/ folder)
"""

import numpy as np
import importlib.util
import os

from twpa_design.filter_builder import (
    FilterSpec,
    design_filter,
    design_multiplexer,
    design_double_open_filter,
    build_reflectionless_filter,
    multiplexer_to_netlist,
    compose_chain,
    save_peripheral_netlist,
    plot_peripheral_response,
)
from twpa_design import NETLISTS_DIR

# Configure plotting
import twpa_design.plots_params as plots_params
plots_params.USE_LATEX = True
plots_params.configure_matplotlib(True)

# ============================================================================
# Step 1: Design the peripheral filters
# ============================================================================
print("=" * 60)
print("Step 1: Designing filters")
print("=" * 60)

fc_Hz = 8.3e9  # Crossover frequency

# Input and output diplexers (standard singly-terminated arms)
diplexer_design = design_multiplexer(
    arm_specs=[
        FilterSpec(response='lp', order=25, fc=fc_Hz, approx='butterworth', label='lp'),
        FilterSpec(response='hp', order=25, fc=fc_Hz, approx='butterworth', label='hp'),
    ],
    Z0=50.0,
    label='diplexer',
)
print(f"Input/Output diplexer: order {diplexer_design.arms[0].spec.order}, fc={fc_Hz/1e9:.1f} GHz")

# Center HP filter (doubly open-terminated, for pump/idler path)
# Use same order as diplexer arms (25) for direct comparison.
# Odd order so both ends are the same element type (series cap).
center_hp = design_double_open_filter(
    response='hp',
    total_order=25,
    fc=fc_Hz,
    approx='butterworth',
    Z0=50.0,
    n_edge=3,
    label='hp_center',
)
print(f"Center HP filter: order {center_hp.spec.order}, fc={fc_Hz/1e9:.1f} GHz")
print(f"  g-values (edges): {center_hp.g_values[1]:.3f}, {center_hp.g_values[2]:.3f}, {center_hp.g_values[3]:.3f}")

# LP dump filter for the signal path (absorbed into 50 Ohm loads)
# Use same order as diplexer LP arms (25) for direct comparison.
# Singly terminated: 50 Ohm load on one end, reactive (open) on the junction end.
from twpa_design.filter_builder import calculate_g_values, frequency_transfo, FilterDesign
lp_dump_spec = FilterSpec('lp', order=25, fc=fc_Hz, approx='butterworth', termination='single')
g_lp_single = calculate_g_values('butterworth', 25, termination='single')
g_for_transfo = list(g_lp_single)
if g_for_transfo[-1] == float('inf'):
    g_for_transfo[-1] = 1.0
lp_transfo = frequency_transfo(
    g_for_transfo, [], [], Z0_ohm=50.0, fc=fc_Hz/1e9,
    foster_form=1, zero_at_zero=True, units='GHz'
)
lp_dump = FilterDesign(spec=lp_dump_spec, g_values=g_lp_single, transfo_result=lp_transfo)
print(f"LP dump filter: order {lp_dump.spec.order}, fc={fc_Hz/1e9:.1f} GHz (singly terminated)")

# Print comparison: diplexer LP arm vs dump LP
print(f"\n--- Comparison: diplexer LP arm vs center LP dump ---")
dip_lp = diplexer_design.arms[0].transfo_result  # LP arm
dump_lp = lp_dump.transfo_result
print(f"Diplexer LP series[0] Linf: {dip_lp['series'][0].get('Linf', 0):.4e}")
print(f"Dump LP    series[0] Linf: {dump_lp['series'][0].get('Linf', 0):.4e}")
print(f"Diplexer LP series[-1] Linf: {dip_lp['series'][-1].get('Linf', 0):.4e}")
print(f"Dump LP    series[-1] Linf: {dump_lp['series'][-1].get('Linf', 0):.4e}")

print(f"\n--- Comparison: diplexer HP arm vs center HP ---")
dip_hp = diplexer_design.arms[1].transfo_result  # HP arm
ctr_hp = center_hp.transfo_result
print(f"Diplexer HP series[0] C0: {dip_hp['series'][0].get('C0', 0):.4e}")
print(f"Center HP  series[0] C0: {ctr_hp['series'][0].get('C0', 0):.4e}")
print(f"Diplexer HP series[-1] C0: {dip_hp['series'][-1].get('C0', 0):.4e}")
print(f"Center HP  series[-1] C0: {ctr_hp['series'][-1].get('C0', 0):.4e}")

# ============================================================================
# Step 2: Plot standalone responses
# ============================================================================
print("\nStep 2: Plotting standalone filter responses")
print("=" * 60)

f_plot = np.linspace(0.1, 20, 2001)
fig, axes, _ = plot_peripheral_response(diplexer_design, f=f_plot, units='GHz')

# ============================================================================
# Step 3: Generate netlists
# ============================================================================
print("\nStep 3: Generating netlists")
print("=" * 60)

# Input diplexer
d_in = multiplexer_to_netlist(diplexer_design, prefix='m1')
print(f"Input diplexer (m1): {d_in.n_ports} ports, {len(d_in.components)} components")

# Center section: reflectionless HP filter (HP through-path + LP dump loads)
center_block = build_reflectionless_filter(
    through_design=center_hp,
    dump_design=lp_dump,
    Z0=50.0,
    prefix='nrf',
)
print(f"Center reflectionless filter (nrf): {center_block.n_ports} ports, {len(center_block.components)} components")

# Output diplexer
d_out = multiplexer_to_netlist(diplexer_design, prefix='m2')
print(f"Output diplexer (m2): {d_out.n_ports} ports, {len(d_out.components)} components")

# ============================================================================
# Step 4: Load TWPA netlists
# ============================================================================
print("\nStep 4: Loading TWPA netlists")
print("=" * 60)

twpa_netlist_name = "4wm_jtwpa_2002cells_01"
twpa_file = os.path.join(str(NETLISTS_DIR), f"{twpa_netlist_name}.py")

spec = importlib.util.spec_from_file_location("twpa_netlist", twpa_file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Use same TWPA for both stages
twpa1 = {
    'jc_components': mod.jc_components,
    'circuit_parameters': dict(mod.circuit_parameters),
    'metadata': dict(mod.metadata),
}
twpa2 = {
    'jc_components': list(mod.jc_components),
    'circuit_parameters': dict(mod.circuit_parameters),
    'metadata': dict(mod.metadata),
}

print(f"TWPA: {twpa_netlist_name}")
print(f"  Components: {len(twpa1['jc_components'])}")

# ============================================================================
# Step 5: Compose the TWPA-TWPA topology
# ============================================================================
print("\nStep 5: Composing TWPA-TWPA topology")
print("=" * 60)

# Topology: d_in -> TWPA1 -> center_block -> TWPA2 -> d_out
# The center block is a reflectionless HP filter: pump/idler passes through
# the HP path, while signal is absorbed by the LP dump loads.
jc_components, circuit_parameters, metadata = compose_chain(
    blocks=[d_in, twpa1, center_block, twpa2, d_out],
    connections=[
        ('common', 'input'),   # d_in common -> TWPA1 input
        ('output', 'input'),   # TWPA1 output -> center block input
        ('output', 'input'),   # center block output -> TWPA2 input
        ('output', 'common'),  # TWPA2 output -> d_out common
    ],
)

print(f"Composed circuit:")
print(f"  Total components: {len(jc_components)}")
print(f"  Total parameters: {len(circuit_parameters)}")
print(f"  External ports: {metadata.get('n_ports', '?')}")
print(f"  Port roles: {metadata.get('port_roles', {})}")

# ============================================================================
# Step 6: Save the composed netlist
# ============================================================================
print("\nStep 6: Saving composed netlist")
print("=" * 60)

output_name = "twpa_twpa_4wm_jtwpa_2002cells_01"
output_file = os.path.join(str(NETLISTS_DIR), f"{output_name}.py")

metadata['device_name'] = output_name
metadata['topology'] = 'd_in -> TWPA1 -> reflectionless_hp -> TWPA2 -> d_out'
metadata['diplexer_order'] = 25
metadata['center_hp_order'] = center_hp.spec.order
metadata['lp_dump_order'] = lp_dump.spec.order
metadata['fc_GHz'] = fc_Hz / 1e9
metadata['twpa_netlist'] = twpa_netlist_name

save_peripheral_netlist(jc_components, circuit_parameters, metadata, output_file)

print(f"\nPort assignment:")
print(f"  Port 1: LP in  (signal input)")
print(f"  Port 2: HP in  (pump + idler input)")
print(f"  Port 3: LP out (signal output)")
print(f"  Port 4: HP out (pump + idler output)")

# ============================================================================
# Step 7: Simulate with JosephsonCircuits.jl
# ============================================================================
print("\n" + "=" * 60)
print("Step 7: Running Julia simulation")
print("=" * 60)

from twpa_design.julia_wrapper import TWPASimulator, TWPASimulationConfig

simulator = TWPASimulator()

sim_config = TWPASimulationConfig(
    freq_start_GHz=4.0,
    freq_stop_GHz=12.0,
    freq_step_GHz=0.05,
    pump_freq_GHz=8.55,
    pump_current_A=2.6e-6,
    pump_port=2,            # HP in
    signal_port=1,          # LP in
    output_port=3,          # LP out
    Npumpharmonics=8,
    Nmodulationharmonics=4,
)

try:
    results = simulator.run_full_simulation(
        netlist_name=output_name,
        config=sim_config,
        verbose=True,
        save_results=False,
        show_plot=False,
    )

    # ============================================================================
    # Step 8: Plot results
    # ============================================================================
    print("\n" + "=" * 60)
    print("Step 8: Plotting results")
    print("=" * 60)

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    from twpa_design.plots_params import (
        blue, red, green, orange, purple, black,
        linewidth, fontsize, fontsize_title, fontsize_legend
    )

    fig = plt.figure(figsize=(8.6/2.54, 5))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.5)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    freq = results.frequencies_GHz

    # Panel 1: Key S-parameters
    ax1.plot(freq, 10*np.log10(results.s_param(3, 1)), color=tuple(blue), linewidth=linewidth)
    ax1.plot(freq, 10*np.log10(results.s_param(1, 3)), color=tuple(red), linewidth=linewidth)
    ax1.plot(freq, 10*np.log10(results.s_param(4, 2)), color=tuple(orange), linewidth=linewidth)
    ax1.plot(freq, 10*np.log10(results.s_param(1, 1)), color=tuple(green), linewidth=linewidth, alpha=0.7)
    ax1.plot(freq, 10*np.log10(results.s_param(3, 3)), color=tuple(green), linewidth=linewidth, alpha=0.7, linestyle='--')
    ax1.axvline(sim_config.pump_freq_GHz, color=tuple(purple), linestyle=':', alpha=0.5)

    s31_handle = Line2D([0], [0], color=tuple(blue), linewidth=linewidth, label=r'$|S_{31}|$')
    s13_handle = Line2D([0], [0], color=tuple(red), linewidth=linewidth, label=r'$|S_{13}|$')
    s42_handle = Line2D([0], [0], color=tuple(orange), linewidth=linewidth, label=r'$|S_{42}|$')
    s11_handle = Line2D([0], [0], color=tuple(green), linewidth=linewidth, alpha=0.7, label=r'$|S_{11}|$')
    s33_handle = Line2D([0], [0], color=tuple(green), linewidth=linewidth, alpha=0.7, linestyle='--', label=r'$|S_{33}|$')
    pump_handle = Line2D([0], [0], color=tuple(purple), linestyle=':', alpha=0.5,
                         label=rf'$f_a = {sim_config.pump_freq_GHz}$ GHz')

    ax1.legend(
        handles=[s31_handle, s13_handle, s42_handle, s11_handle, s33_handle, pump_handle],
        loc='lower right',
        fontsize=fontsize_legend,
        frameon=True,
        fancybox=True,
        framealpha=0.7,
        facecolor='white',
        edgecolor='gray',
        borderpad=0.3,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.5,
    )
    ax1.set_ylabel(r'$|S|$-parameters [dB]', fontsize=fontsize)
    ax1.set_title('TWPA-TWPA', fontsize=fontsize_title)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-30, 30)
    ax1.set_xlim(sim_config.freq_start_GHz, sim_config.freq_stop_GHz)
    ax1.tick_params(axis='both', labelsize=fontsize)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    # Panel 2: Quantum efficiency
    ax2.plot(freq, results.quantum_efficiency, color=tuple(blue), linewidth=linewidth)
    ax2.axhline(1.0, color=tuple(black), linestyle='--', alpha=0.7, label='Ideal')
    ax2.set_xlabel('frequency [GHz]', fontsize=fontsize)
    ax2.set_ylabel(r'$\mathsf{QE}/\mathsf{QE}_\mathsf{ideal}$', fontsize=fontsize)
    ax2.set_title('Quantum Efficiency', fontsize=fontsize_title)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.7, 1.05)
    ax2.set_yticks([0.7, 0.8, 0.9, 1])
    ax2.set_xlim(sim_config.freq_start_GHz, sim_config.freq_stop_GHz)
    ax2.tick_params(axis='both', labelsize=fontsize)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    # Save results and plot
    save_plot = True
    results.save(config=sim_config)
    if save_plot:
        from twpa_design import RESULTS_DIR
        from twpa_design.helper_functions import filecounter
        plot_pattern = f"{RESULTS_DIR}/{output_name}_pump{sim_config.pump_freq_GHz:.2f}GHz_*.svg"
        plot_filename, plot_number = filecounter(plot_pattern)
        fig.savefig(plot_filename, format='svg', bbox_inches='tight')
        print(f"📈 Plot saved to: {plot_filename}")

    plt.show()

    print(f"\n🎯 TWPA-TWPA simulation complete!")

except Exception as e:
    print(f"\n❌ Simulation failed: {e}")
    print("The composed netlist was saved successfully — you can try")
    print("running the simulation manually with different parameters.")
    raise
