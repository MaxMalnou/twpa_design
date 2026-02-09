# Julia Wrapper Module - Complete Reference

This module provides a Python interface to JosephsonCircuits.jl for simulating TWPA circuits using harmonic balance analysis.

**Solver Modes:**
- **Nonlinear mode** (default): Uses `hbsolve` for full nonlinear analysis with pump
- **Linear mode**: Uses `hblinsolve` for faster linear S-parameter analysis without pump

## Quick Start

```python
from twpa_design.julia_wrapper import TWPASimulator, TWPASimulationConfig, TWPAResults

# Simple usage - everything in one call
simulator = TWPASimulator()
results = simulator.run_full_simulation(
    netlist_name="b_jtwpa_2000cells_01",
    config=TWPASimulationConfig(
        freq_start_GHz=6.0,
        freq_stop_GHz=10.0,
        pump_freq_GHz=16.12,
        pump_current_A=0.6e-6
    ),
    verbose=True,
    save_results=True  # Auto-saves both data and plot
)
```

**Note:** Julia configuration (local fork vs GitHub fork vs registered version) is controlled by constants in `julia_setup.py`:
- `USE_LOCAL_FORK`: Toggle local development vs remote
- `USE_GITHUB_FORK`: Toggle GitHub fork vs registered (when USE_LOCAL_FORK=False)

## Classes

### TWPASimulator

Main simulation orchestrator class.

#### Constructor
```python
simulator = TWPASimulator()
```

No parameters needed. Julia configuration (local fork vs GitHub fork vs registered version) is controlled by constants in `julia_setup.py`:
- `USE_LOCAL_FORK = False`: Use remote version (default for users)
- `USE_GITHUB_FORK = True`: Use GitHub fork with Taylor expansion (default)

For local development with hot-reloading, set `USE_LOCAL_FORK = True` in `julia_setup.py`.

#### Methods

##### run_full_simulation (Recommended - All-in-one)
```python
results = simulator.run_full_simulation(
    netlist_name,           # str: Netlist file name
    config,                 # TWPASimulationConfig object
    verbose=True,           # bool: Print detailed progress
    force_julia_reinit=False,  # bool: Force Julia restart
    save_results=True,      # bool: Auto-save data and plot
    show_plot=True,         # bool: Display plot (default: True)
    max_mode_order_to_plot=2,  # int: Max idler mode order to plot
    output_dir=None         # str: Output directory (None uses package's results/ folder)
)
```
Combines setup_julia, load_netlist, build_circuit, run_simulation, and optionally saves/displays results.

##### setup_julia
```python
simulator.setup_julia(
    force_reinit=False      # Force full cleanup and reinstallation
)
```
Initialize Julia environment. Called automatically by run_full_simulation.

Configuration is controlled in `julia_setup.py` - no parameters needed.

##### load_netlist
```python
simulator.load_netlist(
    netlist_name,           # str: Name without .py extension
    netlist_dir=None        # str: Directory (None uses package default)
)
```
Load netlist from file. If netlist_dir is None, uses package's netlists/ directory.

##### build_circuit
```python
simulator.build_circuit()
```
Build circuit in Julia from loaded netlist. Must call load_netlist first.

##### run_simulation
```python
results = simulator.run_simulation(config)  # Returns TWPAResults
```
Run harmonic balance simulation. Must call build_circuit first.

##### force_julia_reinstall (Class method)
```python
TWPASimulator.force_julia_reinstall()
```
Complete cleanup and reinstallation of Julia packages.

---

### TWPASimulationConfig

Configuration for harmonic balance simulation.

#### Constructor with All Parameters
```python
config = TWPASimulationConfig(
    # Frequency sweep parameters
    freq_start_GHz=4.0,           # float: Start frequency (default: 4.0)
    freq_stop_GHz=10.0,           # float: Stop frequency (default: 10.0) 
    freq_step_GHz=0.1,            # float: Frequency step (default: 0.1)
    
    # Pump parameters
    pump_freq_GHz=8.0,            # float: Pump frequency (default: 8.0)
    pump_current_A=1e-6,          # float: Pump amplitude (default: 1e-6)
    pump_port=1,                  # int: Pump injection port (default: 1)
    
    # Signal/output ports
    signal_port=1,                # int: Signal input port (default: 1)
    output_port=2,                # int: Output port (default: 2)
    
    # DC bias - Current mode
    enable_dc_bias=False,         # bool: Enable DC bias (default: False)
    dc_current_A=None,            # float: DC current in Amperes
    dc_port=1,                    # int: DC bias port (default: 1)
    
    # DC bias - Flux mode (alternative to current mode)
    dc_flux_bias=None,            # float: Flux as fraction of Φ₀
    mutual_inductance_H=None,     # float: Mutual inductance in Henries
    
    # Harmonic balance parameters
    Npumpharmonics=20,            # int: Pump harmonics (default: 20)
    Nmodulationharmonics=None,    # int: Modulation harmonics (auto: 10 for nonlinear, 0 for linear)

    # Nonlinear mixing (auto-set based on solver_mode if None)
    enable_three_wave_mixing=None,   # bool: Enable 3WM (auto: False nonlinear, None linear = Julia default false)
    enable_four_wave_mixing=None,    # bool: Enable 4WM (auto: True nonlinear, None linear = Julia default true)

    # Solver mode selection
    solver_mode="nonlinear",         # str: "nonlinear" (hbsolve) or "linear" (hblinsolve)

    # Numerical solver parameters
    iterations=None,              # int: Max iterations (default: 1000)
    ftol=None,                    # float: Function tolerance (default: 1e-8)
    switchofflinesearchtol=None,  # float: Linesearch switch (default: 1e-5)
    alphamin=None,                # float: Min step size (default: 1e-4)
    sorting="name",               # str: Node sorting ("name", "number", "none")

    # Data storage options
    store_signal_nodeflux=False   # bool: Store signal nodeflux for harmonics plotting
)
```

#### Methods

##### frequency_array
```python
freqs_Hz = config.frequency_array()  # Returns numpy array of frequencies in Hz
```

##### get_sources
```python
sources = config.get_sources()  # Returns list of source dictionaries
```

##### get_solver_options
```python
options = config.get_solver_options()  # Returns dict of solver options
```

##### print_config
```python
config.print_config()  # Print formatted configuration summary
```

---

### TWPAResults

Container for simulation results with analysis and visualization.

#### Attributes

**S-parameter and Mode Results:**
- `frequencies_GHz`: numpy array of frequencies
- `S11`, `S12`, `S21`, `S22`: S-parameter arrays (power)
- `quantum_efficiency`: QE/QE_ideal array
- `commutation_error`: 1-CM array
- `idler_response`: Forward idler S-parameters
- `backward_idler_response`: Backward idler S-parameters (optional)
- `modes`: List of mode tuples
- `config`: Stored TWPASimulationConfig (optional)
- `netlist_name`: Stored netlist name (optional)

**Harmonics Spatial Data (for plotting power along the line):**
- `pump_nodeflux`: numpy array of pump nodeflux (shape: num_pump_harmonics × num_nodes). Always available.
- `signal_nodeflux`: numpy array of signal/idler nodeflux. Requires `store_signal_nodeflux=True` in config.
- `num_pump_harmonics`: int, number of pump harmonics
- `num_nodes`: int, number of nodes in the circuit
- `total_cells`: int, total number of cells in the TWPA
- `pump_freq_Hz`: float, pump frequency in Hz (used for power calculations)
- `main_line_node_indices`: numpy array of indices into the sorted nodeflux arrays selecting only main-line nodes (nodes on every path from port 1 to port 2). Side-branch nodes (internal to filters) are excluded. Computed automatically; used by `plot_harmonics()` for cleaner spatial plots.

#### Methods

##### save
```python
# Auto-generate filename from config
filename = results.save()

# With explicit filename
filename = results.save(filename="my_results.npz")

# With custom metadata
filename = results.save(
    filename=None,
    metadata={"custom_key": "value"},
    config=sim_config,        # Used for auto-naming
    use_filecounter=True,     # Auto-increment filename
    output_dir=None           # Output directory (None uses package's results/ folder)
)

# Save to custom directory
filename = results.save(config=sim_config, output_dir="/path/to/workspace")
```
Saves results to .npz file. Returns filename.

##### plot
```python
# Basic plot using stored config
fig = results.plot()

# With all options
fig = results.plot(
    config=sim_config,        # TWPASimulationConfig (or uses stored)
    netlist_name="netlist",   # For filename generation
    save_path="figure.svg",   # Explicit save path
    auto_save=False,          # Auto-save with incremented name
    show_plot=True,           # Display the plot
    max_mode_order_to_plot=2, # Max idler mode order to plot
    output_dir=None           # Output directory (None uses package's results/ folder)
)

# Save plot to custom directory
fig = results.plot(config=sim_config, auto_save=True, output_dir="/path/to/workspace")
```
Creates 4-panel figure (S-params, forward/backward idlers, QE). Returns matplotlib Figure.

##### load (Class method)
```python
# Load from results directory (auto-finds path)
results, metadata = TWPAResults.load("b_jtwpa_2000cells_01_pump16.12GHz_01")

# Load with full path
results, metadata = TWPAResults.load("/path/to/results.npz")
```
Returns tuple of (TWPAResults object, metadata dict).

##### load_and_plot (Class method)
```python
# Load and immediately plot
results, fig = TWPAResults.load_and_plot(
    filename="results_01",
    show_plot=True,
    auto_save=False,
    save_path=None,
    max_mode_order_to_plot=2  # Max idler mode order to plot
)
```
Convenience method to load and plot in one call.

##### plot_harmonics
```python
fig = results.plot_harmonics(
    ax=None,                    # Optional matplotlib Axes (creates new figure if None)
    config=None,                # TWPASimulationConfig (uses stored config if None)
    max_pump_harmonic=3,        # Max pump harmonic to plot (1 = fundamental only)
    max_signal_mode_order=2,    # Max signal/idler mode order to plot
    signal_freq_GHz=None,       # Signal frequency for idler plotting (required if signal_nodeflux exists)
    position_normalized=False,  # If True, x-axis shows 0-1; if False, shows cell numbers
    save_path=None,             # Optional path to save figure
    show_plot=True              # Display the plot
)
```
Plot pump and signal/idler power evolution along the transmission line.

**Colors:**
- Pump: purple (fundamental), pink/yellow/green for harmonics
- Signal: blue (always plotted on top)
- Idlers: orange (first), darkblue/red/brown for higher orders

**Requirements:**
- Pump harmonics always available
- Signal/idler harmonics require `store_signal_nodeflux=True` in simulation config

##### plot_pump_harmonics
```python
fig = results.plot_pump_harmonics(
    ax=None,
    config=None,
    max_pump_harmonic=3,
    position_normalized=False,
    save_path=None,
    show_plot=True
)
```
Convenience wrapper for `plot_harmonics` with pump-only plotting (no signal/idler).

##### plot_signal_harmonics
```python
fig = results.plot_signal_harmonics(
    ax=None,
    config=None,
    max_signal_mode_order=2,
    signal_freq_GHz=None,       # Required
    position_normalized=False,
    save_path=None,
    show_plot=True
)
```
Convenience wrapper for `plot_harmonics` with signal/idler-only plotting.
Raises `ValueError` if signal nodeflux data is not available.

---

## Helper Functions

### flux_bias_config
```python
from twpa_design.julia_wrapper import flux_bias_config

# Create DC bias config for Φ₀/3 with 2.2 pH mutual inductance
dc_config = flux_bias_config(
    flux_over_phi0=1/3,      # Flux as fraction of Φ₀
    mutual_inductance_pH=2.2  # Mutual inductance in picoHenries
)

# Use in TWPASimulationConfig
config = TWPASimulationConfig(
    **dc_config,
    freq_start_GHz=5.0,
    # ... other parameters
)
```

---

## Complete Usage Examples

### Example 1: Standard Workflow
```python
# Step-by-step approach
simulator = TWPASimulator()
simulator.setup_julia()
simulator.load_netlist("4wm_jtwpa_2002cells_01")
simulator.build_circuit()

config = TWPASimulationConfig(
    freq_start_GHz=4.0,
    freq_stop_GHz=12.0,
    pump_freq_GHz=8.63,
    pump_current_A=2.7e-6
)

results = simulator.run_simulation(config)
results.save()
fig = results.plot(auto_save=True)
```

### Example 2: Quick One-Liner
```python
# Everything in one call
results = TWPASimulator().run_full_simulation(
    "b_jtwpa_2000cells_01",
    TWPASimulationConfig(pump_freq_GHz=16.12, pump_current_A=0.6e-6),
    save_results=True,
    max_mode_order_to_plot=3  # Plot up to 3rd order idlers
)
```

### Example 3: Load and Re-analyze Previous Results
```python
# Load saved results
results, metadata = TWPAResults.load("ktwpa_pump9.1GHz_01")

# Access data
gain_dB = 10*np.log10(results.S21)
peak_gain = np.max(gain_dB)

# Re-plot with different settings
fig = results.plot(
    show_plot=True,
    save_path="new_plot.svg"
)

# Extract metadata
print(f"Original pump: {metadata['pump_freq_GHz']} GHz")
print(f"Original power: {metadata['pump_current_A']*1e6} μA")
```

### Example 4: Batch Processing
```python
simulator = TWPASimulator()

# Run multiple pump powers
pump_powers = [0.5e-6, 1.0e-6, 1.5e-6, 2.0e-6]
all_results = []

for pump_power in pump_powers:
    config = TWPASimulationConfig(
        pump_freq_GHz=8.63,
        pump_current_A=pump_power
    )

    results = simulator.run_full_simulation(
        "jtwpa_2002cells_01",
        config,
        verbose=False,  # Quiet mode for batch
        save_results=True
    )
    all_results.append(results)

# Compare gains
for i, results in enumerate(all_results):
    max_gain = np.max(10*np.log10(results.S21))
    print(f"Pump {pump_powers[i]*1e6:.1f} μA: {max_gain:.1f} dB")
```

### Example 5: Custom DC Bias
```python
# Using flux bias
config = TWPASimulationConfig(
    **flux_bias_config(1/3, 2.2),  # Φ₀/3 with 2.2 pH
    freq_start_GHz=5.0,
    freq_stop_GHz=10.0,
    pump_freq_GHz=7.5,
    pump_current_A=1e-6
)

# Or direct current bias
config = TWPASimulationConfig(
    enable_dc_bias=True,
    dc_current_A=10e-6,
    dc_port=1,
    # ... other parameters
)
```

### Example 6: Reflection vs Transmission Amplifier
```python
# Transmission amplifier (standard TWPA)
config_trans = TWPASimulationConfig(
    signal_port=1,
    output_port=2,  # Different ports
    pump_port=1
)

# Reflection amplifier (PA)
config_refl = TWPASimulationConfig(
    signal_port=1,
    output_port=1,  # Same port
    pump_port=1
)
```

### Example 7: Advanced Solver Settings
```python
# For difficult convergence cases
config = TWPASimulationConfig(
    pump_freq_GHz=9.1,
    pump_current_A=13e-6,
    iterations=2000,           # Increase max iterations
    ftol=1e-4,                # Relax tolerance
    switchofflinesearchtol=0,  # Disable linesearch switching
    alphamin=1e-6,            # Smaller min step
    sorting="number"          # Try different node sorting
)
```

### Example 8: Controlling Idler Plot Detail
```python
# Plot only fundamental idlers (mode orders ±1)
results = simulator.run_full_simulation(
    "jtwpa_2002cells_01",
    config,
    max_mode_order_to_plot=1,  # Only ±1 idlers
    save_results=True
)

# Plot more idler detail (up to ±4 mode orders)
results.plot(
    max_mode_order_to_plot=4,  # Show ±1, ±2, ±3, ±4 idlers
    auto_save=True
)

# Default behavior (up to ±2 mode orders)
results.plot()  # Equivalent to max_mode_order_to_plot=2
```

### Example 9: Using Custom Output Directory (Workspace)
```python
import os
from twpa_design.julia_wrapper import TWPASimulator, TWPASimulationConfig

# Get workspace directory
workspace_dir = os.path.dirname(os.path.abspath(__file__))

# Run simulation and save to workspace
simulator = TWPASimulator()
results = simulator.run_full_simulation(
    netlist_name="4wm_jtwpa_2002cells_01",
    config=TWPASimulationConfig(
        freq_start_GHz=6.0,
        freq_stop_GHz=10.0,
        pump_freq_GHz=8.63,
        pump_current_A=2.7e-6
    ),
    save_results=True,
    output_dir=workspace_dir  # Save to workspace instead of package folder!
)

# Or manually control saving
results = simulator.run_full_simulation(
    "4wm_jtwpa_2002cells_01",
    config,
    save_results=False  # Don't auto-save
)
# Save to workspace manually
results.save(config=config, output_dir=workspace_dir)
results.plot(config=config, auto_save=True, output_dir=workspace_dir)
```

### Example 10: Linear S-parameter Analysis (No Pump)
```python
from twpa_design.julia_wrapper import TWPASimulator, TWPASimulationConfig

# Linear analysis mode - simple! Just specify solver_mode="linear"
# Sensible defaults are applied automatically:
#   - Nmodulationharmonics=0 (DC + fundamental only)
#   - enable_three_wave_mixing=None (uses Julia default: false)
#   - enable_four_wave_mixing=None (uses Julia default: true)
simulator = TWPASimulator()
results = simulator.run_full_simulation(
    netlist_name="4wm_jtwpa_2002cells_01",
    config=TWPASimulationConfig(
        freq_start_GHz=4.0,
        freq_stop_GHz=12.0,
        freq_step_GHz=0.1,
        solver_mode="linear"  # That's it! Defaults handle the rest
    ),
    save_results=True
)

# Useful for:
# - Checking linear transmission/reflection without pump
# - Faster baseline calculations
# - Debugging circuit connectivity
# - Comparing with nonlinear results

# For nonlinear analysis with pump (default):
results_nonlinear = simulator.run_full_simulation(
    netlist_name="4wm_jtwpa_2002cells_01",
    config=TWPASimulationConfig(
        freq_start_GHz=4.0,
        freq_stop_GHz=12.0,
        pump_freq_GHz=8.0,
        pump_current_A=1e-6,
        solver_mode="nonlinear"  # Use hbsolve (default)
    )
)

# Advanced: Override linear mode defaults if needed for special cases
results_advanced = simulator.run_full_simulation(
    netlist_name="4wm_jtwpa_2002cells_01",
    config=TWPASimulationConfig(
        freq_start_GHz=4.0,
        freq_stop_GHz=12.0,
        solver_mode="linear",
        Nmodulationharmonics=2,  # Override: include more harmonics
        enable_four_wave_mixing=True  # Override: enable 4WM modes
    ),
    save_results=True
)
# User-specified values take precedence over mode defaults!
```

### Example 11: Plotting Pump and Signal Harmonics Along the Line
```python
from twpa_design.julia_wrapper import TWPASimulator, TWPASimulationConfig, TWPAResults

# Run simulation with signal nodeflux storage enabled
simulator = TWPASimulator()
results = simulator.run_full_simulation(
    netlist_name="4wm_ktwpa_5004cells_01",
    config=TWPASimulationConfig(
        freq_start_GHz=4.0,
        freq_stop_GHz=10.0,
        pump_freq_GHz=9.1,
        pump_current_A=13e-6,
        store_signal_nodeflux=True  # Required for signal/idler harmonics!
    ),
    save_results=True
)

# Plot both pump and signal harmonics on same figure
results.plot_harmonics(
    max_pump_harmonic=3,        # Plot 1st, 2nd, 3rd pump harmonics
    max_signal_mode_order=2,    # Plot signal and first 2 idler orders
    signal_freq_GHz=7.0         # Select which signal frequency to plot
)

# Plot pump harmonics only (always available, no special config needed)
results.plot_pump_harmonics(max_pump_harmonic=3)

# Plot signal/idler harmonics only
results.plot_signal_harmonics(
    max_signal_mode_order=2,
    signal_freq_GHz=7.0
)

# Use normalized position (0-1) instead of cell numbers
results.plot_harmonics(
    max_pump_harmonic=2,
    max_signal_mode_order=1,
    signal_freq_GHz=7.0,
    position_normalized=True  # X-axis shows 0-1 instead of cell numbers
)

# Load saved results and replot harmonics
results_loaded, metadata = TWPAResults.load("4wm_ktwpa_5004cells_01_pump9.10GHz_01")
if results_loaded.pump_nodeflux is not None:
    results_loaded.plot_harmonics(
        max_pump_harmonic=3,
        max_signal_mode_order=2,
        signal_freq_GHz=7.0
    )
```

---

## Common Issues and Solutions

### Julia Not Found
```python
# Force reinstall
TWPASimulator.force_julia_reinstall()

# Or specify Julia path in environment
import os
os.environ['JULIA_PATH'] = '/path/to/julia/bin'
```

### Convergence Issues
```python
# Relax tolerances and increase iterations
config = TWPASimulationConfig(
    ftol=1e-4,
    iterations=2000,
    Npumpharmonics=10,  # Reduce harmonics if needed
    Nmodulationharmonics=5
)
```

### Memory Issues
```python
# Reduce frequency points
config = TWPASimulationConfig(
    freq_step_GHz=0.1,  # Instead of 0.01
    Npumpharmonics=10,  # Reduce harmonics
    Nmodulationharmonics=5
)
```

---

## Physical Constants
- `FLUX_QUANTUM = 2.067833848e-15` Wb (Φ₀)

## Default Values Summary
- Frequency: 4-10 GHz, 0.1 GHz steps
- Pump: 8 GHz, 1 μA, port 1
- Ports: signal=1, output=2
- Harmonics: pump=20, modulation=10
- Mixing: 3WM=off, 4WM=on
- Solver: 1000 iterations, ftol=1e-8

---

*Note: This module requires JosephsonCircuits.jl (Julia package) to be installed.*