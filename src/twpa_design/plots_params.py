# PLOT parameters
# Configure plotting parameters
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# LaTeX rendering toggle - can be overridden by user scripts
USE_LATEX = True

# import matplotlib.cm as cm
# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.patches import FancyBboxPatch
# from matplotlib.gridspec import GridSpec

fontsize = 10
fontsize_legend = 8
fontsize_title = 8 
linewidth=1
markersize = 4

def configure_matplotlib(use_latex=None):
    """Configure matplotlib parameters based on LaTeX setting.
    
    Parameters:
    -----------
    use_latex : bool, optional
        Whether to use LaTeX rendering. If None, uses the global USE_LATEX setting.
    """
    if use_latex is None:
        use_latex = USE_LATEX
    
    base_params = {
        # Font handling and embedding
        'pdf.fonttype': 42,        # Enable TrueType font embedding
        'ps.fonttype': 42,         # Enable TrueType font embedding
        'svg.fonttype': 'path',    # Convert fonts to paths if saving as SVG
        'figure.dpi' : 200,
        'font.size': fontsize,
        'axes.titlesize': fontsize,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize,
        'lines.linewidth': linewidth,
        'xtick.direction': 'in',  # Set x-ticks inside
        'ytick.direction': 'in',   # Set y-ticks inside
        'legend.handlelength': 1,  # Set legend handle length
        'legend.frameon': False    # Remove legend frame
    }
    
    if use_latex:
        latex_params = {
            'text.usetex': True,
            'font.family': 'sans-serif',            
            'text.latex.preamble': '\\renewcommand{\\familydefault}{\\sfdefault}',
        }
    else:
        latex_params = {
            'text.usetex': False,
            'font.family': 'sans-serif',
        }
    
    # Combine parameters and apply
    plt.rcParams.update({**base_params, **latex_params})

# Initialize matplotlib with default settings
configure_matplotlib()



def mask_jumps(x, y, threshold=np.pi*3/2):
   # Mask for jumps
   dx = np.abs(np.diff(x))
   mask = np.zeros_like(x, dtype=bool)
   mask[1:] = dx > threshold
   
   # Mask at k=0 and k=±π
   # mask |= np.abs(x) < 1e-30  # k=0
   # mask |= np.abs(np.abs(x) - np.pi) < 1e-30  # k=±π
   
   return np.ma.masked_array(y, mask)

   

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return np.array([int(hex_code[i:i+2], 16) for i in (0, 2, 4)]) / 255

def rgb_to_hex(rgb_arr):
    return '#%02x%02x%02x' % tuple(np.array(rgb_arr * 255, dtype=int))


# Function to create line segments avoiding discontinuities
def plot_band_with_segments(ax, x_data, y_data, color, threshold=np.pi/8):
    # Find the discontinuities
    dx = np.diff(x_data)
    dy = np.diff(y_data)
    jumps = np.where(np.abs(dx) < threshold)[0]
    
    # Split the data at discontinuities
    split_indices = np.concatenate(([0], jumps + 1, [len(x_data)]))
    
    # Plot each continuous segment separately
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i+1]
        if end_idx > start_idx:  # Ensure the segment has data points
            ax.plot(x_data[start_idx:end_idx], y_data[start_idx:end_idx], color=color)


def plot_continuous_segments(ax, x_data, y_data, color, max_step=np.pi/8):
    """
    Plot only segments where consecutive points are within max_step of each other.
    """
    dy = np.diff(y_data)
    continuous = np.abs(dy) <= max_step
    
    # Find start and end of continuous regions
    continuous_padded = np.concatenate(([False], continuous, [False]))
    starts = np.where(np.diff(continuous_padded.astype(int)) == 1)[0]
    ends = np.where(np.diff(continuous_padded.astype(int)) == -1)[0]
    
    # Plot each continuous region
    for start, end in zip(starts, ends):
        ax.plot(x_data[start:end+1], y_data[start:end+1], color=color)  # Note the end+1!


# Colors
black = hex_to_rgb('#000000')
blue = hex_to_rgb('#387EB9')
darkblue = hex_to_rgb('#393B7A')
green = hex_to_rgb('#4DAF49')
orange = hex_to_rgb('#F57E20')
red = hex_to_rgb('#E21F26')
purple = hex_to_rgb('#98509F')
pink = hex_to_rgb('#EE83B5')
brown = hex_to_rgb('#A65627')
yellow = hex_to_rgb('#E6B850')
gray = hex_to_rgb('#999999')
darkgray = hex_to_rgb('#7F7F7F')


# Create lighter versions by mixing with white
def lighten_color(color, factor=0.5):
    """Creates a lighter version of the given color by mixing with white"""
    white = np.array([1.0, 1.0, 1.0])
    return color * (1 - factor) + white * factor

# Create lighter versions of your colors
light_green = lighten_color(green, 0.7)  # 70% lighter
light_red = lighten_color(red, 0.7)      # 70% lighter
light_blue = lighten_color(blue, 0.7)      # 70% lighter
light_orange = lighten_color(orange, 0.7)      # 70% lighter
light_purple = lighten_color(purple, 0.7)      # 70% lighter