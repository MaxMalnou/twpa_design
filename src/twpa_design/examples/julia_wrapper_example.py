"""
Example: Julia Wrapper for TWPA Simulation

This example demonstrates how to use the julia_wrapper module
to run nonlinear circuit simulations using JosephsonCircuits.jl
"""

from twpa_design.julia_wrapper import TWPASimulator, TWPASimulationConfig, TWPAResults
import numpy as np

# Configure LaTeX rendering for plots (set to False to disable LaTeX)
USE_LATEX = True

# Apply the LaTeX setting
import twpa_design.plots_params as plots_params
plots_params.USE_LATEX = USE_LATEX
plots_params.configure_matplotlib(USE_LATEX)

"""
Parameters that can be customized for different simulations.
The unspecified parameters will be set to their default values.

TWPASimulationConfig parameters
-------------------------------
# Frequency range for signal sweep
freq_start_GHz: Start frequency for signal sweep. default: 4.0
freq_stop_GHz: Stop frequency for signal sweep. default: 10.0
freq_step_GHz: Frequency step for sweep. default: 0.1

# Pump parameters
pump_freq_GHz: Pump frequency. default: 8.0
pump_current_A: Pump current amplitude. default: 1e-6
pump_port: Port for pump injection. default: 1

# Signal parameters
signal_port: Port for signal injection. default: 1
output_port: Port for output. default: 2

# DC bias parameters - Current mode
enable_dc_bias: Enable DC bias analysis. default: False
dc_current_A: DC current in Amperes. default: None
dc_port: Port for DC bias injection. default: 1

# DC bias parameters - Flux mode  
dc_flux_bias: DC flux bias as fraction of Φ₀. default: None
mutual_inductance_H: Mutual inductance in Henries. default: None

# Harmonic balance settings
Npumpharmonics: Number of pump harmonics. default: 20
Nmodulationharmonics: Number of modulation harmonics. default: 10

# Additional solver settings
enable_three_wave_mixing: Enable 3WM analysis. default: False
enable_four_wave_mixing: Enable 4WM analysis. default: True

# Numerical solver parameters (JosephsonCircuits.jl defaults)
iterations: Maximum solver iterations. default: 1000
ftol: Function tolerance for convergence. default: 1e-8
switchofflinesearchtol: Tolerance for switching off linesearch. default: 1e-5
alphamin: Minimum step size for linesearch. default: 1e-4
sorting: Node sorting method ('name', 'number', 'none'). default: 'name'

Available netlist files:
- '4wm_jtwpa_2002cells_01.py': Standard 4-wave mixing JTWPA (2002 cells)
- 'b_jtwpa_1000cells_01.py': Broadband JTWPA with rf-SQUID nonlinearity (500 cells)  
- '4wm_ktwpa_5004cells_01.py': 4WM KTWPA with kinetic inductance (5004 cells)
"""

SIMULATION_CONFIGS = {
    'jtwpa': {
        'netlist_name': '4wm_jtwpa_2002cells_01',
        'config': TWPASimulationConfig(
            freq_start_GHz=4.0,
            freq_stop_GHz=12.0,
            freq_step_GHz=0.01,
            pump_freq_GHz=8.63,
            pump_current_A=2.7e-6,
            Npumpharmonics=20,
            Nmodulationharmonics=10
        )
    },
    'b_jtwpa': {
        'netlist_name': 'b_jtwpa_2000cells_01',
        'config': TWPASimulationConfig(
            freq_start_GHz=4, # 4
            freq_stop_GHz=12.01, #12.01
            freq_step_GHz=0.01, #0.01
            pump_freq_GHz=14.985, # 15
            pump_current_A=1.1e-6, # 2.3e-6
            pump_port=2,
            signal_port=1,
            output_port=2,
            enable_three_wave_mixing=True,
            enable_four_wave_mixing=True,
            Npumpharmonics=20,
            Nmodulationharmonics=10
        )
    },     
    'ktwpa': {
        'netlist_name': '4wm_ktwpa_5004cells_01',
        'config': TWPASimulationConfig(
            freq_start_GHz=4.0,
            freq_stop_GHz=14.0,
            freq_step_GHz=0.01,
            pump_freq_GHz=9.1,
            pump_current_A=22e-6,
            Npumpharmonics=10,
            Nmodulationharmonics=10,
            iterations=1000,
            ftol=1e-5  # Increased tolerance for convergence
        )
    }
}

simulation_type = "jtwpa"  # Choose: 'jtwpa', 'b_jtwpa', 'b_jtwpa_v2', 'ktwpa'

# Get the config for the chosen simulation
if simulation_type in SIMULATION_CONFIGS:
    sim_setup = SIMULATION_CONFIGS[simulation_type]
    netlist_name = sim_setup['netlist_name']
    sim_config = sim_setup['config']
else:
    raise ValueError(f"Unknown simulation type: {simulation_type}")

print(f"Running simulation with configuration: {simulation_type}")
print(f"Netlist: {netlist_name}")
print("\n" + "="*60)

# === Setup Simulator ===
# Julia configuration (local/GitHub fork/registered) is controlled in julia_setup.py
simulator = TWPASimulator()

# === Run Complete Simulation Workflow ===
try:
    # Run everything in one method call with verbose output and auto-save
    results = simulator.run_full_simulation(
        netlist_name=netlist_name,
        config=sim_config,        
        verbose=True,        # Set to False for minimal output
        save_results=True,    # Automatically saves both data (.npz) and plot (.svg)
        max_mode_order_to_plot=4
    )
    
    print(f"\n🎯 Simulation '{simulation_type}' completed successfully!")

except Exception as e:
    print(f"\n❌ Simulation failed: {e}")
    raise