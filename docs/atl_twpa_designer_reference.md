# ATL TWPA Designer Module - Complete Reference

This module implements the Artificial Transmission Line (ATL) methodology for designing traveling-wave parametric amplifiers with engineered dispersion through filters and/or periodic modulation.

## Quick Start

```python
from twpa_design.atl_twpa_designer import ATLTWPADesigner
import numpy as np

# Design a 4WM JTWPA with filter dispersion
designer = ATLTWPADesigner(
    custom_params={
        'f_zeros_GHz': 9,           # Single number
        'f_poles_GHz': 8.85,        # Single number
        'Ic_JJ_uA': 5,
        'Ia0_uA': 3,
        'Ntot_cell': 2000
    },
    verbose=True
)

# Run complete design workflow
results = designer.run_design(interactive=True, save_results=True)
```

## Main Class: ATLTWPADesigner

### Constructor
```python
designer = ATLTWPADesigner(
    custom_params=None,    # dict: Override default parameters
    verbose=True          # bool: Print detailed output
)
```

### Complete Parameter List with Defaults

```python
# Basic parameters
'device_name': '4wm_jtwpa'           # str: Device identifier
'f_step_MHz': 5                      # float: Frequency step for plots
'fmax_GHz': 30                        # float: Maximum frequency for plots

# Filter parameters (dispersion type automatically derived)
# If only f_zeros_GHz/f_poles_GHz specified -> 'filter'
# If only stopbands_config_GHz specified -> 'periodic'
# If both specified -> 'both'
'f_zeros_GHz': []                    # list/number: Filter zero frequencies
'f_poles_GHz': []                    # list/number: Filter pole frequencies
'zero_at_zero': True                 # bool: True for low-pass (zero at DC), False for high-pass (pole at DC)
                                     # Only affects pure LP/HP filters when both f_zeros_GHz and f_poles_GHz are empty
'fc_filter_GHz': 500                 # float: Filter cutoff frequency
'fc_TLsec_GHz': 500                  # float: TL section cutoff frequency
'Foster_form_L': 1                   # int: Foster form for L filter (1 or 2)
'Foster_form_C': 1                   # int: Foster form for C filter (1 or 2)
'select_one_form': 'C'               # str: 'L', 'C', or 'both'

# Periodic modulation parameters
'stopbands_config_GHz': {}           # dict: {freq_GHz: {'min': val, 'max': val}}
                                     # Example: {27: {'max': 4}, 16: {'min': 1}}
'force_zero_phase': True             # bool: True for cosine-only modulation (default, backward compatible)
                                     # False for cosine+sine modulation (Hermitian, may reduce peak deviation)
'window_type': 'boxcar'             # str: 'boxcar', 'tukey', or 'hann'
'alpha': 0.0                         # float: Window parameter (0-1)
'n_filters_per_sc': 1                # int: Filters per supercell

# Nonlinearity type
'nonlinearity': 'JJ'                 # str: 'JJ' or 'KI'
'Id_uA': 0                           # float: DC current bias

# JJ-specific parameters
'jj_structure_type': 'jj'            # str: 'jj' or 'rf_squid'
'Ic_JJ_uA': 5                        # float: Critical current
'fJ_GHz': 40                         # float: Plasma frequency
'n_jj_struct': 1                     # int: Number of JJ structures in series per cell

# RF-SQUID additional parameters
'beta_L': np.inf                     # float: Participation ratio
'phi_dc': 0                          # float: DC flux bias (in radians)

# KI-specific parameters
'Istar_uA': 100                      # float: Nonlinearity scale current
'L0_pH': 100                         # float: Inductance per cell

# Phase-matching parameters
'WM': '4WM'                          # str: '3WM' or '4WM'
'dir_prop_PA': 'forw'                # str: 'forw' or 'back'
'Ia0_uA': 1                          # float: Pump amplitude
'detsigidlGHz': 2                    # float: Signal-idler detuning
'fa_min_GHz': 0                      # float: Min pump frequency search
'fa_max_GHz': 30                     # float: Max pump frequency search

# TWPA line parameters
'Ntot_cell': 2000                    # int: Total number of cells
'nTLsec': 0                          # int: TL sections between filters
'Z0_TWPA_ohm': 50                    # float: Characteristic impedance
```

### Dispersion Type and Stopbands Configuration

The dispersion type is automatically derived from your parameter inputs:
- **'filter'**: Only `f_zeros_GHz` and/or `f_poles_GHz` specified
- **'periodic'**: Only `stopbands_config_GHz` specified  
- **'both'**: Both filter and stopband parameters specified

#### Stopbands Configuration

Use the dictionary format to define stopband frequencies and their edges:

```python
'stopbands_config_GHz': {
    27: {'max': 4},           # Stopband at 27 GHz, upper edge at +4 GHz
    16: {'min': 1},           # Stopband at 16 GHz, lower edge at -1 GHz  
}
```

For each stopband frequency, specify either `'min'` or `'max'` (not both):
- `'min'`: Lower edge width (how far below center frequency)
- `'max'`: Upper edge width (how far above center frequency)

#### Filter Zeros and Poles Input Formats

The filter parameters accept flexible input formats for user convenience:

```python
# All these formats work:
'f_zeros_GHz': 9,             # Single number
'f_zeros_GHz': [9, 8.5],      # List of multiple values  
'f_zeros_GHz': [],            # Empty list (no zeros)

'f_poles_GHz': 8.85,          # Single number
'f_poles_GHz': [8.85, 9.2],   # List of multiple values
'f_poles_GHz': [],            # Empty list (no poles)
```

All formats are automatically converted to numpy arrays internally for calculations.

---

## Primary Methods

### run_design
```python
results = designer.run_design(
    interactive=True,      # bool: Pause at each plot for user input
    save_results=False,    # bool: Export parameters and save plot data at end
    save_plots=False,      # bool: Save phase matching plot as SVG
    output_dir=None        # str: Output directory (None uses package's designs/ folder)
)
```
Runs the complete TWPA design workflow. Returns comprehensive results dictionary.

**Parameters:**
- `interactive` (bool): If True, pause at each plot for user input. If False, run without stopping
- `save_results` (bool): If True, exports both design parameters (.py file) and plot data (.npz file)
- `save_plots` (bool): If True, saves the phase matching plot as SVG
- `output_dir` (str or Path, optional): Custom output directory. If None, uses package's designs/ folder

**Automatic Saving:** When `save_results=True`, both `export_parameters()` and `save_data()` are called automatically.

### run_initial_calculations
```python
designer.run_initial_calculations()
```
Step 1: Calculate nonlinearity parameters, Taylor coefficients, and filter components.

### calculate_derived_quantities
```python
designer.calculate_derived_quantities()
```
Step 2: Calculate effective impedances and phase velocities.

### calculate_linear_response
```python
designer.calculate_linear_response()
```
Step 3: Calculate S-parameters and dispersion relation.

### calculate_phase_matching
```python
designer.calculate_phase_matching()
```
Step 4: Find phase-matched pump frequency and calculate gain bandwidth.

---

## Plotting Methods

### plot_linear_response
```python
designer.plot_linear_response()
```
Plots S21 magnitude in dB and dispersion relation (k vs f).

### plot_phase_matching
```python
designer.plot_phase_matching()
```
Plots 2x2 grid: S-parameters with pump/signal/idler, k(f), Z_Bloch, and phase mismatch.

### plot_modulation_profile
```python
designer.plot_modulation_profile()
```
For periodic structures, plots the spatial modulation profile with windowing.

---

## Export and Analysis Methods

### export_parameters
```python
filename = designer.export_parameters(filename=None, output_dir=None)
```
Exports all design parameters to a Python file.

**Parameters:**
- `filename` (str, optional): Custom filename. If None, auto-generates based on device_name
- `output_dir` (str or Path, optional): Output directory. If None, uses package's designs/ folder

### get_results
```python
results = designer.get_results()
```
Returns comprehensive dictionary with all calculated parameters.

### get_config
```python
config = designer.get_config()
```
Returns configuration dictionary suitable for recreating the design.

### get_circuit_parameters
```python
circuit = designer.get_circuit_parameters()
```
Returns circuit parameters for netlist generation.

### get_characteristics
```python
chars = designer.get_characteristics()
```
Returns key performance characteristics.

### print_parameters
```python
designer.print_parameters()
```
Prints formatted summary of all parameters.

---

## Data Save/Load Methods

### save_data
```python
filename = designer.save_data(filename=None, output_dir=None)
```
Saves all data needed to regenerate linear response and phase matching plots to a compressed NumPy (.npz) file.

**Parameters:**
- `filename` (str, optional): Output filename. If None, auto-generates based on device_name
- `output_dir` (str or Path, optional): Output directory. If None, uses package's designs/ folder

**Returns:**
- `str`: Path to saved .npz file

**Saved Data:**
- Configuration parameters (nonlinearity, WM, etc.)
- Frequency arrays and S-parameters for linear response plots  
- Phase matching data (delta_kPA, delta_betaPA, pump/signal/idler frequencies)
- All original design parameters for completeness

### load_data (class method)
```python
data = ATLTWPADesigner.load_data(filename)
```
Loads previously saved design data from a .npz file.

**Parameters:**
- `filename` (str): Path to .npz file (looks in designs/ folder by default)

**Returns:**
- `dict`: Dictionary containing all saved data

### plot_from_data (class method)
```python
fig_linear, fig_phase = ATLTWPADesigner.plot_from_data(
    data=None,           # dict: Data from load_data() 
    filename=None,       # str: Path to .npz file (alternative to data)
    plot_linear=True,    # bool: Generate linear response plot
    plot_phase=True      # bool: Generate phase matching plot
)
```
Regenerates plots from saved data using the original plotting methods.

**Parameters:**
- `data` (dict, optional): Data dictionary from `load_data()`. If None, must provide filename
- `filename` (str, optional): Path to .npz file to load. Ignored if data is provided
- `plot_linear` (bool): Whether to plot linear response (default: True)
- `plot_phase` (bool): Whether to plot phase matching (default: True)

**Returns:**
- `tuple`: (fig_linear, fig_phase) - matplotlib figure objects or None if not plotted

**Example Usage:**
```python
# Save data during design workflow
designer = ATLTWPADesigner(custom_params={...})
results = designer.run_design(save_results=True)  # Automatically saves .npz file

# Later: Load and regenerate plots
data = ATLTWPADesigner.load_data("device_name_01.npz")
fig_linear, fig_phase = ATLTWPADesigner.plot_from_data(data=data)
plt.show()  # Display plots
```

**Note:** The `run_design()` method with `save_results=True` automatically calls `save_data()` in addition to `export_parameters()`, so both .py design files and .npz plot data are saved.

---

## Complete Usage Examples

### Example 1: 4WM JTWPA with Filter Dispersion
```python
from twpa_design.atl_twpa_designer import ATLTWPADesigner
import numpy as np

designer = ATLTWPADesigner(
    custom_params={
        'device_name': '4wm_jtwpa_custom',
        'f_zeros_GHz': 9,
        'f_poles_GHz': 8.85,
        'fc_filter_GHz': 200,
        'Foster_form_C': 1,
        'select_one_form': 'C',
        # JJ parameters
        'nonlinearity': 'JJ',
        'jj_structure_type': 'jj',
        'Ic_JJ_uA': 5,
        'fJ_GHz': 40,
        # Phase matching
        'WM': '4WM',
        'dir_prop_PA': 'forw',
        'Ia0_uA': 3,
        'detsigidlGHz': 3,
        'fa_min_GHz': 7.75,
        'fa_max_GHz': 8.75,
        # Line parameters
        'Ntot_cell': 2000,
        'nTLsec': 10,
        'Z0_TWPA_ohm': 50
    },
    verbose=True
)

results = designer.run_design(interactive=True, save_results=True)
```

### Example 2: 3WM Broadband JTWPA with RF-SQUID
```python
designer = ATLTWPADesigner(
    custom_params={
        'device_name': 'b_jtwpa',
        'f_zeros_GHz': 10,
        'f_poles_GHz': 3,
        'Foster_form_L': 1,
        'Foster_form_C': 1,
        'select_one_form': 'both',
        # RF-SQUID parameters
        'nonlinearity': 'JJ',
        'jj_structure_type': 'rf_squid',
        'Ic_JJ_uA': 2,
        'fJ_GHz': 4000,
        'beta_L': 0.4,
        'phi_dc': np.pi/2,  # Kerr-free point
        # 3WM backward pumping
        'WM': '3WM',
        'dir_prop_PA': 'back',
        'Ia0_uA': 3,
        'detsigidlGHz': 0.25,
        'fa_min_GHz': 10,
        'fa_max_GHz': 20,
        # Line
        'Ntot_cell': 2000
    }
)

results = designer.run_design()
```

### Example 3: KTWPA with Combined Dispersion
```python
designer = ATLTWPADesigner(
    custom_params={
        'device_name': '4wm_ktwpa',
        # Filter parameters (creates 'both' dispersion type with stopbands)
        'f_zeros_GHz': 9.8,
        'f_poles_GHz': 9.7,
        'Foster_form_L': 2,
        'select_one_form': 'L',
        # Periodic modulation (new dict format)
        'stopbands_config_GHz': {
            27: {'max': 4}  # Only max specified, min will be None
        },
        'window_type': 'tukey',
        'alpha': 0.125,
        # Kinetic inductance
        'nonlinearity': 'KI',
        'Istar_uA': 100,
        'L0_pH': 100,
        # Phase matching
        'WM': '4WM',
        'Ia0_uA': 30,
        'detsigidlGHz': 3,
        'fa_min_GHz': 8.6,
        'fa_max_GHz': 9.6,
        # Line
        'Ntot_cell': 5000
    }
)

results = designer.run_design()
```

### Example 4: Step-by-Step Design Process
```python
# Create designer
designer = ATLTWPADesigner(verbose=True)

# Step 1: Initial calculations
designer.run_initial_calculations()
print(f"L0 = {designer.L0_H*1e12:.2f} pH")
print(f"c3_taylor = {designer.c3_taylor:.4f}")

# Step 2: Derived quantities
designer.calculate_derived_quantities()
print(f"Nonlinear coefficient: xi = {designer.xi_perA2:.2e} /A²")

# Step 3: Linear response
designer.calculate_linear_response()
designer.plot_linear_response()

# Check S21 at specific frequency
f_check = 8.0  # GHz
idx = np.argmin(np.abs(designer.f_GHz - f_check))
s21_dB = 20*np.log10(np.abs(designer.S21[idx]))
print(f"S21 at {f_check} GHz: {s21_dB:.2f} dB")

# Step 4: Phase matching
designer.calculate_phase_matching()
designer.plot_phase_matching()

# Check phase-matched pump
print(f"Phase-matched pump: {designer.fa_GHz:.3f} GHz")
print(f"Signal: {designer.fs_GHz:.3f} GHz")
print(f"Idler: {designer.fi_GHz:.3f} GHz")
```

### Example 5: Parameter Sweep
```python
# Sweep critical current to find optimal value
Ic_values = [3, 4, 5, 6, 7]  # μA
results_list = []

for Ic in Ic_values:
    designer = ATLTWPADesigner(
        custom_params={
            'Ic_JJ_uA': Ic,
            'Ia0_uA': 0.5 * Ic,  # Scale pump with Ic
            # ... other parameters
        },
        verbose=False
    )
    
    designer.run_initial_calculations()
    designer.calculate_derived_quantities()
    
    results_list.append({
        'Ic': Ic,
        'xi': designer.xi_perA2,
        'c3': designer.c3_taylor
    })

# Find optimal
for res in results_list:
    print(f"Ic={res['Ic']}μA: xi={res['xi']:.2e}/A²")
```

### Example 5: Export and Reload Design
```python
# Create and export design
designer = ATLTWPADesigner(custom_params={...})
results = designer.run_design()

# Export to file
filename = designer.export_parameters()
print(f"Exported to: {filename}")

# Later, reload the design
import importlib.util
spec = importlib.util.spec_from_file_location("design", filename)
design_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design_module)

# Recreate designer from saved parameters
new_designer = ATLTWPADesigner(
    custom_params=vars(design_module),
    verbose=True
)
```

### Example 6: Analyzing Phase Matching
```python
designer = ATLTWPADesigner(custom_params={...})
designer.run_initial_calculations()
designer.calculate_derived_quantities()
designer.calculate_linear_response()
designer.calculate_phase_matching()

# Access phase matching data
pump_freqs = designer.f_GHz[designer.ind_fa_min:designer.ind_fa_max]
phase_mismatch = designer.delta_betaPA

# Find zero crossings
zero_crossings = np.where(np.diff(np.sign(phase_mismatch)))[0]
if len(zero_crossings) > 0:
    perfect_match_freq = pump_freqs[zero_crossings[0]]
    print(f"Perfect phase match at: {perfect_match_freq:.3f} GHz")

# Calculate bandwidth
threshold = 0.1  # rad/cell
bw_mask = np.abs(phase_mismatch) < threshold
if np.any(bw_mask):
    bw_freqs = pump_freqs[bw_mask]
    bandwidth = bw_freqs[-1] - bw_freqs[0]
    print(f"Phase-matching bandwidth: {bandwidth:.2f} GHz")
```

### Example 7: Using Custom Output Directory (Workspace)
```python
import os
from twpa_design.atl_twpa_designer import ATLTWPADesigner

# Get current script directory for workspace
workspace_dir = os.path.dirname(os.path.abspath(__file__))

# Design TWPA
designer = ATLTWPADesigner(
    custom_params={
        'device_name': 'my_custom_jtwpa',
        'f_zeros_GHz': 9,
        'f_poles_GHz': 8.85,
        'Ic_JJ_uA': 5,
        'Ia0_uA': 3,
        'Ntot_cell': 2000
    },
    verbose=True
)

# Run design and save to workspace folder instead of package folder
results = designer.run_design(
    interactive=True,
    save_results=True,
    save_plots=True,
    output_dir=workspace_dir  # Saves to your workspace!
)

# Or manually control saving
results = designer.run_design(interactive=True, save_results=False)
designer.export_parameters(output_dir=workspace_dir)
designer.save_data(output_dir=workspace_dir)
```


---

## Key Calculated Parameters

### Nonlinearity Parameters
- `L0_H`: Effective linear inductance
- `c1_taylor` to `c4_taylor`: Taylor expansion coefficients
- `epsilon_perA`: Linear phase shift per ampere
- `xi_perA2`: Nonlinear phase shift per ampere squared

### Filter Components (if applicable)
- `LinfLF1_H`, `C0LF1_F`, etc.: Foster form filter values
- `ind_zero_atl`, `ind_pole_atl`: ATL indices for zeros/poles


### Dispersion
- `k_radpercell`: Wave vector array
- `vph_m_per_s`: Phase velocity array
- `Zbloch_ohm`: Bloch impedance array

### Phase Matching
- `fa_GHz`: Phase-matched pump frequency
- `fs_GHz`, `fi_GHz`: Signal and idler frequencies
- `delta_betaPA`: Phase mismatch array

### S-Parameters
- `S11`, `S12`, `S21`, `S22`: Complex S-parameter arrays

---

## Physical Models

### JJ Nonlinearity
```
L(φ) = L_J / cos(φ/n)
Taylor: L(φ) ≈ L₀(1 + c₁φ + c₂φ²/2! + c₃φ³/3! + c₄φ⁴/4!)
```

### RF-SQUID
```
L(φ) = L_g / (1 + β_L cos(φ_dc))
With full Taylor expansion including c3 and c4
```

### Kinetic Inductance
```
L(I) = L₀(1 + (I/I*)²)
```

### Phase Matching Conditions
- **4WM Forward**: Δk = 2k_a - k_s - k_i
- **4WM Backward**: Δk = -2k_a - k_s - k_i
- **3WM Forward**: Δk = k_a - k_s - k_i (where ω_a = ω_s + ω_i)
- **3WM Backward**: Δk = -k_a - k_s - k_i

---

## Design Guidelines

### Filter Design
- Place zero slightly above pole for positive dispersion
- Wider zero-pole separation → stronger dispersion
- Foster form 1: Zero at DC
- Foster form 2: Pole at DC

### Periodic Modulation
- Stopband at 3×pump for 4WM suppression
- Stopband at 2×pump for 3WM suppression
- Use windowing to reduce ripples

### Nonlinearity Selection
- JJ: Josephson junction nonlinearity, fixed characteristics
- RF-SQUID: Tunable with flux bias, can reach Kerr-free point
- KI: Kinetic inductance nonlinearity, current-dependent

### Impedance Matching
- Keep Bloch impedance near 50Ω at signal frequency
- Avoid impedance spikes at pump frequency

---

## Common Issues and Solutions

### No Phase Matching Found
```python
# Expand search range
designer.fa_min_GHz = 0
designer.fa_max_GHz = 50
designer.calculate_phase_matching()
```

### Poor Gain Bandwidth
```python
# Adjust dispersion
# Move zero closer to pole for gentler dispersion
designer.f_zeros_GHz = 9.0   # Single number (automatically converted)
designer.f_poles_GHz = 8.9   # Single number (automatically converted)
```

### Numerical Instabilities
```python
# For periodic structures with sharp features
designer.window_type = 'tukey'
designer.alpha = 0.2  # Smooth transitions
```

### Export Missing Parameters
```python
# Get all parameters
all_params = {
    **designer.get_config(),
    **designer.get_circuit_parameters(),
    **designer.get_characteristics()
}
```

---

## Output Files

### Design Export Format
Creates Python file in designs/ folder:
```python
# designs/device_name_XX.py
device_name = '4wm_jtwpa'
# ... all parameters
```

### Results Dictionary
```python
results = designer.get_results()
# Contains: config, circuit, characteristics, 
# linear_response, phase_matching
```

---

## Notes

- Frequencies in GHz, inductances in H, capacitances in F
- Current in μA, impedance in Ω
- Phase in radians, wave vector in rad/cell
- Not implemented: SNAIL, DC-SQUID structures

---

*This module implements the methodology from [relevant papers/theory].*