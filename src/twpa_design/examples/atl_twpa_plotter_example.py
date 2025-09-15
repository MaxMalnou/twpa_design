"""
Example: Load and Plot Saved TWPA Design Data

This script demonstrates how to load previously saved TWPA design data
from .npz files and regenerate plots. It also shows how to create custom
plots using the loaded data.
"""

from twpa_design.atl_twpa_designer import ATLTWPADesigner
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
from twpa_design.plots_params import linewidth, fontsize, black, blue, darkblue, purple

# Load saved data from .npz file (looks in designs/ folder by default)
filename = "b_jtwpa_01.npz"  # Replace with your actual filename
data = ATLTWPADesigner.load_data(filename)

# ============================================================================
# Option 1: Use the standard plotting methods
# ============================================================================
# print("Generating standard plots...")
# fig_linear, fig_phase = ATLTWPADesigner.plot_from_data(data=data)

# ============================================================================
# Option 2: Create custom linear response plot with frequency markers
# ============================================================================
print("Creating custom linear response plot with frequency markers...")

linewidth = 1

xmin = 4
xmax = 16

# Extract parameters from data
dispersion_type = data.get('dispersion_type', 'filter')
Ncpersc_cell = data.get('Ncpersc_cell', 1)
WM = data.get('WM', '4WM')

# Create custom linear response plot
fig = plt.figure(figsize=(8.6/2.54, 3))
fig.subplots_adjust(hspace=0.3)

# First subplot - S21 with frequency markers (like phase matching plot)
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(data['f_GHz'], 20*np.log10(np.abs(data['S21'])), color='k', linewidth=linewidth)

# Add frequency markers if phase matching data exists
if 'fa_GHz' in data:
    linewidth_vert_lines = 1.5
    ax1.axvline(x=data['fa_GHz'], color=purple, linewidth=linewidth_vert_lines, label='$f_a$')
    ax1.axvline(x=data['fs_GHz'], color=blue, linewidth=linewidth_vert_lines, label='$f_s$')
    ax1.axvline(x=data['fi_GHz'], color=darkblue, linewidth=linewidth_vert_lines, label='$f_i$')
    
    # Add higher order pump frequency
    if WM == '3WM':
        ax1.axvline(x=2*data['fa_GHz'], color=purple, linewidth=linewidth_vert_lines, 
                   linestyle='--', label='$2f_a$')
    elif WM == '4WM':
        ax1.axvline(x=3*data['fa_GHz'], color=purple, linewidth=linewidth_vert_lines,
                   linestyle='--', label='$3f_a$')
    
    # ax1.legend(loc='best')

# ax1.set_xlim((min(data['f_GHz']), max(data['f_GHz'])))
ax1.set_xlim((xmin, xmax))
ax1.axhline(y=0, color='k', linestyle='-')
ax1.set_ylim((-10, 2))
ax1.set_ylabel(r'$|S_{21}|^2$ [dB]', fontsize=fontsize)
ax1.grid(True)
ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax1.tick_params(axis='both', labelsize=fontsize)

# Second subplot - k (same as original)
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(data['f_GHz'], data['k_radpercell'], color=black, linewidth=linewidth)

# ax2.set_xlim((min(data['f_GHz']), max(data['f_GHz'])))
ax2.set_xlim((xmin, xmax))

ax2.set_ylim(-np.pi/4, np.pi/4)            
ax2.set_yticks([-np.pi/4, 0, np.pi/4])
ax2.set_yticklabels(('$-\\frac{\\pi}{4}$','0','$\\frac{\\pi}{4}$'), fontsize=fontsize)
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=fontsize)

ax2.set_ylabel(r'$k$ [rad/cell]', fontsize=fontsize)
ax2.set_xlabel('frequency [GHz]', fontsize=fontsize)

plt.tight_layout()

# Add title to distinguish from standard plot
device_name = data.get('device_name', 'Unknown Device')

print(f"Device: {device_name}")
if 'fa_GHz' in data:
    print(f"Pump: {data['fa_GHz']:.3f} GHz, Signal: {data['fs_GHz']:.3f} GHz, Idler: {data['fi_GHz']:.3f} GHz")

# Display all plots (keeps them open)
plt.show()


ymin = 0
ymax = 16

# Create custom linear response plot
fig2 = plt.figure(figsize=(8.6/2.54, 1.5))
fig2.subplots_adjust(hspace=0.3)

# Second subplot - k (same as original)
ax = fig2.add_subplot(111)
ax.plot(data['k_radpercell'],data['f_GHz'], color=black, linewidth=linewidth)
ax.plot(-data['k_radpercell'],data['f_GHz'], '--', color=black, linewidth=linewidth)

# Add arrows pointing to fa, fs, and fi frequencies if they exist
if 'fa_GHz' in data:
    # Find the k values at these frequencies by interpolation
    k_at_fa = np.interp(data['fa_GHz'], data['f_GHz'], data['k_radpercell'])
    k_at_fs = np.interp(data['fs_GHz'], data['f_GHz'], data['k_radpercell'])
    k_at_fi = np.interp(data['fi_GHz'], data['f_GHz'], data['k_radpercell'])
    
    # Add arrows from origin (0,0) to the points on the curve
    # Arrow to pump frequency (negative k)
    ax.annotate('', xy=(-k_at_fa, data['fa_GHz']), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=purple, lw=1.5, 
                               shrinkA=0, shrinkB=-1))
    
    # Arrow to signal frequency (positive k)
    ax.annotate('', xy=(k_at_fs, data['fs_GHz']), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=blue, lw=1.5,
                               shrinkA=0, shrinkB=-1))
    
    # Arrow to idler frequency (positive k)
    ax.annotate('', xy=(k_at_fi, data['fi_GHz']), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=darkblue, lw=1.5,
                               shrinkA=0, shrinkB=-1))


# ax.set_xlim((min(data['f_GHz']), max(data['f_GHz'])))
ax.set_ylim((ymin, ymax))
ax.set_yticks((0,8,16))
ax.set_yticklabels(('$0$','$8$','$16$'), fontsize=fontsize)

ax.set_xlim(-np.pi/8, np.pi/8)            
ax.set_xticks([-np.pi/8, 0, np.pi/8])
ax.set_xticklabels(('$-\\frac{\\pi}{8}$', '0', '$\\frac{\\pi}{8}$'), fontsize=fontsize)
            
ax.set_ylabel('frequency [GHz]', fontsize=fontsize)
ax.set_xlabel(r'$k$ [rad/cell]', fontsize=fontsize)
ax.grid(True)
ax.tick_params(axis='both', labelsize=fontsize)

ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

plt.tight_layout()

# Add title to distinguish from standard plot
device_name = data.get('device_name', 'Unknown Device')

print(f"Device: {device_name}")
if 'fa_GHz' in data:
    print(f"Pump: {data['fa_GHz']:.3f} GHz, Signal: {data['fs_GHz']:.3f} GHz, Idler: {data['fi_GHz']:.3f} GHz")

# Display all plots (keeps them open)
plt.show()



