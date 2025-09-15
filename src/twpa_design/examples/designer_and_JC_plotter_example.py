"""
Example: Load and Plot Saved Julia Simulation Results

This script demonstrates how to load previously saved TWPA simulation results
from .npz files and regenerate plots. It also shows how to create custom
plots using the loaded data.
"""
from twpa_design.atl_twpa_designer import ATLTWPADesigner
from twpa_design.julia_wrapper import TWPAResults
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Configure LaTeX rendering for plots (set to False to disable LaTeX)
USE_LATEX = True

# Apply the LaTeX setting
import twpa_design.plots_params as plots_params
plots_params.USE_LATEX = USE_LATEX
plots_params.configure_matplotlib(USE_LATEX)

# Import plotting parameters
from twpa_design.plots_params import linewidth, fontsize, black, blue, darkblue, red, green, orange, purple, pink, brown, yellow

# Load saved simulation results from .npz file (looks in results/ folder by default)
filename_designer = "b_jtwpa_01.npz"  # Replace with your actual filename
filename_JC = "b_jtwpa_2000cells_01_pump14.98GHz_01.npz"  # Replace with your actual filename


# Load results and metadata
data_designer = ATLTWPADesigner.load_data(filename_designer)
results, metadata = TWPAResults.load(filename_JC)

# Extract parameters from data
dispersion_type = data_designer.get('dispersion_type', 'filter')
Ncpersc_cell = data_designer.get('Ncpersc_cell', 1)
WM = data_designer.get('WM', '4WM')


# Import additional plotting parameters from the original plot method
from matplotlib import gridspec
from matplotlib.lines import Line2D
from twpa_design.plots_params import fontsize_legend, fontsize_title

# Recreate the config from metadata (same as in load_and_plot)
from twpa_design.julia_wrapper import TWPASimulationConfig
config = TWPASimulationConfig(
    freq_start_GHz=results.frequencies_GHz[0],
    freq_stop_GHz=results.frequencies_GHz[-1],
    pump_freq_GHz=metadata.get('pump_freq_GHz', 0),
    pump_current_A=metadata.get('pump_current_A', 0),
    signal_port=metadata.get('signal_port', 1),
    output_port=metadata.get('output_port', 2)
)

# Create figure with 4x1 layout (exact same as original)
fig_custom = plt.figure(figsize=(8.6/2.54, 7))
gs = gridspec.GridSpec(4, 1, figure=fig_custom, height_ratios=[1, 2, 1, 1], hspace=0.5)
ax1 = fig_custom.add_subplot(gs[0])
ax2 = fig_custom.add_subplot(gs[1])
ax3 = fig_custom.add_subplot(gs[2])
ax4 = fig_custom.add_subplot(gs[3])

# Plot 1: phase reponse
ax1.plot(data_designer['f_GHz'], data_designer['k_radpercell'], color=black, linewidth=linewidth)

# ax1.set_xlim((min(data['f_GHz']), max(data['f_GHz'])))
ax1.set_xlim(config.freq_start_GHz, config.freq_stop_GHz)

ax1.set_ylim(-np.pi/4, np.pi/4)
            
ax1.set_yticks([-np.pi/4, 0, np.pi/4])
ax1.set_yticklabels(('$-\\frac{\\pi}{4}$','0','$\\frac{\\pi}{4}$'), fontsize=fontsize)
ax1.set_ylabel(r'$k$ [rad/cell]', fontsize=fontsize)
ax1.tick_params(axis='both', labelsize=fontsize)
ax1.grid(True)
ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))


# Plot 2: S-parameters (exact same code as _plot_s_parameters)
ax2.plot(results.frequencies_GHz, 10*np.log10(results.S21), color=blue, linewidth=linewidth)
ax2.plot(results.frequencies_GHz, 10*np.log10(results.S12), color=red, linewidth=linewidth)
ax2.plot(results.frequencies_GHz, 10*np.log10(results.S11), color=green, linewidth=linewidth, alpha=0.7)
ax2.plot(results.frequencies_GHz, 10*np.log10(results.S22), color=orange, linewidth=linewidth, alpha=0.7)

# Add vertical line at pump frequency
# ax2.axvline(config.pump_freq_GHz, color=purple, linestyle=':', alpha=0.5)

# Create legend handles
s21_handle = Line2D([0], [0], color=tuple(blue), linewidth=linewidth, label=r'$|S_{21}|$')
s12_handle = Line2D([0], [0], color=tuple(red), linewidth=linewidth, label=r'$|S_{12}|$')
s11_handle = Line2D([0], [0], color=tuple(green), linewidth=linewidth, label=r'$|S_{11}|$')
s22_handle = Line2D([0], [0], color=tuple(orange), linewidth=linewidth, label=r'$|S_{22}|$')
# pump_handle = Line2D([0], [0], color=purple, linestyle=':', 
#                    label=rf'$f_a = {config.pump_freq_GHz}$ GHz' + '\n' + 
#                            rf'$I_a = {config.pump_current_A*1e6:.1f}$ $\mu$A')

ax2.legend(
    # handles=[s21_handle, s11_handle, pump_handle, s12_handle, s22_handle],
    handles=[s21_handle, s11_handle, s12_handle, s22_handle],
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
    ncol=2,
    columnspacing=0.5  # -3.5
)
ax2.set_ylabel(r'$|S|$-parameters [dB]', fontsize=fontsize)
# ax2.set_title('S-Parameters', fontsize=fontsize_title)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-30, 30)
ax2.set_xlim(config.freq_start_GHz, config.freq_stop_GHz)
ax2.tick_params(axis='both', labelsize=fontsize)
ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

# Define helper function for mode labels
def get_mode_label(n: int, pump_freq_GHz: float) -> str:
    """Universal labeling for mode (n,) where frequency = fs + n*fa"""
    if n == 0:
        return "fs"
    elif n > 0:
        if n == 1:
            return r"$f_a + f_s$"
        else:
            return rf"${n}f_a + f_s$"
    else:  # n < 0
        if n == -1:
            return r"$f_a - f_s$"
        else:
            return rf"${-n}f_a - f_s$"

# Define colors for idler plotting
idler_colors = [blue, pink, brown, yellow, purple]

# Plot 3: Idlers
max_mode_order_to_plot = 2  # Set to match the original call
if results.idler_response.ndim > 1 and results.idler_response.shape[0] > 1:
    idler_data = results.idler_response
    
    # Determine shape and orientation
    if idler_data.shape[0] == len(results.frequencies_GHz):
        num_modes = idler_data.shape[1]
        freq_axis = 0
    else:
        num_modes = idler_data.shape[0]
        freq_axis = 1
    
    # Get mode information
    if results.modes is not None:
        modes = results.modes
    else:
        modes = None
    
    # Create mode mapping
    n_to_mode_idx = {}
    if modes is not None:
        for mode_idx in range(num_modes):
            mode_n = modes[mode_idx][0]
            n_to_mode_idx[mode_n] = mode_idx
    
    # Create plot order
    if len(n_to_mode_idx) > 0:
        available_modes = sorted(n_to_mode_idx.keys())
        plot_order = []
        max_abs_n = min(max(abs(n) for n in available_modes if n != 0), 
                       max_mode_order_to_plot)
        
        for i in range(1, max_abs_n + 1):
            if -i in n_to_mode_idx:
                plot_order.append(-i)
            if i in n_to_mode_idx:
                plot_order.append(i)
        
        # Plot forward idlers
        for i, n in enumerate(plot_order):
            if n not in n_to_mode_idx:
                continue
                
            mode_idx = n_to_mode_idx[n]
            
            if freq_axis == 0:
                idler_response_dB = 10*np.log10(np.abs(idler_data[:, mode_idx]) + 1e-12)
            else:
                idler_response_dB = 10*np.log10(np.abs(idler_data[mode_idx, :]) + 1e-12)
            
            label = get_mode_label(n, config.pump_freq_GHz)
            color = idler_colors[i % len(idler_colors)]
            ax3.plot(results.frequencies_GHz, idler_response_dB, 
                   linewidth=linewidth, color=color, label=label)
            
ax3.legend(
    loc='center right',
    fontsize=fontsize_legend,
    frameon=True,
    fancybox=True,
    framealpha=0.7,
    facecolor='white',
    edgecolor='gray',
    borderpad=0.3,
    handlelength=1.5,
    handletextpad=0.5,
    columnspacing=1.0
)


# Backward idlers (exact same code as _plot_backward_idlers)
if results.backward_idler_response is not None and results.backward_idler_response.ndim > 1:
    backward_idler_data = results.backward_idler_response
    
    # Use same mode mapping as forward idlers
    if results.modes is not None:
        modes = results.modes
        n_to_mode_idx = {}
        for mode_idx in range(len(modes)):
            mode_n = modes[mode_idx][0]
            n_to_mode_idx[mode_n] = mode_idx
        
        # Same plot order as forward idlers
        if len(n_to_mode_idx) > 0:
            available_modes = sorted(n_to_mode_idx.keys())
            plot_order = []
            max_abs_n = min(max(abs(n) for n in available_modes if n != 0), 
                           max_mode_order_to_plot)
            
            for i in range(1, max_abs_n + 1):
                if -i in n_to_mode_idx:
                    plot_order.append(-i)
                if i in n_to_mode_idx:
                    plot_order.append(i)
            
            # Determine data orientation
            if backward_idler_data.shape[0] == len(results.frequencies_GHz):
                freq_axis = 0
            else:
                freq_axis = 1
            
            # Plot backward idlers
            for i, n in enumerate(plot_order):
                if n not in n_to_mode_idx:
                    continue
                    
                mode_idx = n_to_mode_idx[n]
                
                if freq_axis == 0:
                    idler_response_dB = 10*np.log10(np.abs(backward_idler_data[:, mode_idx]) + 1e-12)
                else:
                    idler_response_dB = 10*np.log10(np.abs(backward_idler_data[mode_idx, :]) + 1e-12)
                
                label = get_mode_label(n, config.pump_freq_GHz)
                color = idler_colors[i % len(idler_colors)]
                ax3.plot(results.frequencies_GHz, idler_response_dB, 
                       linewidth=linewidth, color=color, alpha=0.5)
                        
# ax3.set_ylabel(rf'$S_{{{config.output_port}{config.signal_port}}}$ [dB]', fontsize=fontsize)
# ax3.set_ylabel(rf'$S_{{\omega_i\omega_s}}$ [dB]', fontsize=fontsize)
ax3.set_ylabel(rf'$|S_{{{config.output_port}{config.signal_port},{config.signal_port}{config.output_port}}}|(\mathrm{{sig}}\rightarrow\mathrm{{idl}})$ [dB]', fontsize=fontsize)
# ax3.set_title('Idlers', fontsize=fontsize_title)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(config.freq_start_GHz, config.freq_stop_GHz)
ax3.set_ylim(-40, 30)
ax3.set_yticks([-40, -20, 0, 20])
ax3.tick_params(axis='both', labelsize=fontsize)
ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

# Plot 4: Quantum efficiency (exact same code as _plot_quantum_efficiency)
ax4.plot(results.frequencies_GHz, results.quantum_efficiency, color=blue, linewidth=linewidth)
ax4.axhline(1.0, color=black, linestyle='--', alpha=0.7, label='Ideal')
ax4.set_xlabel('frequency [GHz]', fontsize=fontsize)
ax4.set_ylabel(r'$\mathsf{QE}/\mathsf{QE}_\mathsf{ideal}$', fontsize=fontsize)
# ax4.set_title('Quantum Efficiency', fontsize=fontsize_title)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0.7, 1.05)
ax4.set_yticks([0.7, 0.8, 0.9, 1])
ax4.set_xlim(config.freq_start_GHz, config.freq_stop_GHz)
ax4.tick_params(axis='both', labelsize=fontsize)
ax4.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

# Print information (same as original)
netlist_name = metadata.get('netlist_name', 'Unknown')
print(f"Custom plot created for: {netlist_name}")
print(f"Pump: {config.pump_freq_GHz:.2f} GHz @ {config.pump_current_A*1e6:.1f} µA")

# Display the plots (keeps them open)
plt.show()