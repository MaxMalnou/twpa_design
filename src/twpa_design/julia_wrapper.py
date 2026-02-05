"""
TWPA Julia Simulation Wrapper

This module handles the Julia simulation of nonlinear circuits using 
JosephsonCircuits.jl package. It provides a Python interface to:
- Load netlists
- Configure simulations
- Run harmonic balance analysis
- Extract and save results
"""

import numpy as np
import os
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.figure import Figure 
from matplotlib.axes import Axes

# Import configuration
from . import julia_setup
from . import NETLISTS_DIR, RESULTS_DIR  # Add this line

from .helper_functions import filecounter


# Import plot parameters from package
from .plots_params import (
    blue as blue_np, red as red_np, green as green_np, orange as orange_np,
    purple as purple_np, brown as brown_np, black as black_np, gray as gray_np,
    darkblue as darkblue_np, pink as pink_np, yellow as yellow_np,
    linewidth, fontsize, fontsize_legend, fontsize_title
)

# Convert colors to tuples for matplotlib
blue = tuple(blue_np)
red = tuple(red_np)
green = tuple(green_np)
orange = tuple(orange_np)
purple = tuple(purple_np)
brown = tuple(brown_np)
black = tuple(black_np)
gray = tuple(gray_np)
darkblue = tuple(darkblue_np)
pink = tuple(pink_np)
yellow = tuple(yellow_np)

# Plot configuration
PLOT_CONFIG = {
    'linewidth': linewidth,
    'fontsize': fontsize,
    'fontsize_legend': fontsize_legend,
    'fontsize_title': fontsize_title,
    'max_mode_order': 2,
    'colors': [blue, red, green, orange, purple, brown]
}

def get_mode_label(n: int, pump_freq_GHz: float) -> str:
    """
    Universal labeling for mode (n,) where frequency = fs + n*fa
    When fs + n*fa < 0, the physical frequency is |n|*fa - fs
    """
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


# Physical constants
FLUX_QUANTUM = 2.067833848e-15  # Wb (Φ₀)


@dataclass
class TWPASimulationConfig:
    """Configuration for TWPA simulation
    
    Note: Physical properties like dielectric loss are part of the netlist,
    not simulation configuration. This class only contains parameters that 
    control how the simulation is run, not what is being simulated.
    """
    
    # Frequency range for signal sweep
    freq_start_GHz: float = 4.0
    freq_stop_GHz: float = 10.0
    freq_step_GHz: float = 0.1
    
    # Pump parameters
    pump_freq_GHz: float = 8
    pump_current_A: float = 1e-6  # 1 μA
    pump_port: int = 1  # Port for pump injection
    
    # Signal parameters
    signal_port: int = 1  # Port for signal injection
    output_port: int = 2  # Port for output
    
    # DC bias parameters - Current mode
    enable_dc_bias: bool = False  # This controls the 'dc' parameter in hbsolve
    dc_current_A: Optional[float] = None  # DC current in Amperes
    dc_port: int = 1  # Port for DC bias injection
    
    # DC bias parameters - Flux mode
    dc_flux_bias: Optional[float] = None  # DC flux bias as fraction of Φ₀ (e.g., 1/3)
    mutual_inductance_H: Optional[float] = None  # Mutual inductance in Henries
    
    # Harmonic balance settings
    Npumpharmonics: int = 20
    Nmodulationharmonics: Optional[int] = None  # Auto-set based on solver_mode if None

    # Additional solver settings
    enable_three_wave_mixing: Optional[bool] = None  # Auto-set based on solver_mode if None
    enable_four_wave_mixing: Optional[bool] = None   # Auto-set based on solver_mode if None

    # Solver mode selection
    solver_mode: str = "nonlinear"  # "nonlinear" (hbsolve) or "linear" (hblinsolve)

    # Numerical solver parameters (with JosephsonCircuits.jl defaults)
    iterations: Optional[int] = None  # Default: 1000
    ftol: Optional[float] = None  # Default: 1e-8
    switchofflinesearchtol: Optional[float] = None  # Default: 1e-5
    alphamin: Optional[float] = None  # Default: 1e-4
    sorting: str = "name"  # Default in hbsolve is "number", but we use "name" for consistency

    # Data storage options
    store_signal_nodeflux: bool = False  # If True, store signal nodeflux for harmonics plotting

    def __post_init__(self):
        """Validate and apply mode-specific defaults"""
        # Validate solver_mode
        if self.solver_mode not in ["nonlinear", "linear"]:
            raise ValueError(f"solver_mode must be 'nonlinear' or 'linear', got '{self.solver_mode}'")

        # Apply mode-specific defaults (only if user didn't specify)
        if self.solver_mode == "linear":
            # Linear mode defaults
            if self.Nmodulationharmonics is None:
                self.Nmodulationharmonics = 0  # Just DC + fundamental
            # Leave enable_three_wave_mixing and enable_four_wave_mixing as None
            # to use Julia's defaults (threewavemixing=false, fourwavemixing=true)
        else:  # nonlinear mode
            # Nonlinear mode defaults
            if self.Nmodulationharmonics is None:
                self.Nmodulationharmonics = 10  # Standard for nonlinear
            if self.enable_three_wave_mixing is None:
                self.enable_three_wave_mixing = False  # Default: disabled
            if self.enable_four_wave_mixing is None:
                self.enable_four_wave_mixing = True   # Default: enabled

        if self.enable_dc_bias:
            self._compute_dc_bias()
    
    def _compute_dc_bias(self):
        """Compute DC current from flux if needed"""
        if self.dc_flux_bias is not None and self.mutual_inductance_H is not None:
            # Calculate DC current from flux
            calculated_current = self.dc_flux_bias * FLUX_QUANTUM / self.mutual_inductance_H
            
            if self.dc_current_A is not None:
                # Both specified - warn user
                print(f"⚠️  Warning: Both dc_current_A and dc_flux_bias specified.")
                print(f"   Using flux-based calculation: {calculated_current*1e6:.1f} μA")
                print(f"   (Ignoring specified current: {self.dc_current_A*1e6:.1f} μA)")
            
            self.dc_current_A = calculated_current
            
        elif self.dc_flux_bias is not None and self.mutual_inductance_H is None:
            raise ValueError("dc_flux_bias specified but mutual_inductance_H is missing!")
        
        elif self.dc_current_A is None:
            # No DC bias specified at all
            self.dc_current_A = 0.0
    
    def frequency_array(self) -> np.ndarray:
        """Generate frequency array in Hz"""
        freqs_GHz = np.arange(self.freq_start_GHz, self.freq_stop_GHz + self.freq_step_GHz, self.freq_step_GHz)
        return freqs_GHz * 1e9
    
    def get_sources(self) -> List[Dict]:
        """Generate source configuration for JosephsonCircuits.jl"""
        sources = []
        
        # Add DC bias if enabled
        if self.enable_dc_bias and self.dc_current_A != 0:
            sources.append({
                'mode': (0,),  # DC is mode 0
                'port': self.dc_port,
                'current': self.dc_current_A
            })
        
        # Add pump (always added if pump current is non-zero)
        if self.pump_current_A != 0:
            sources.append({
                'mode': (1,),  # Pump is mode 1
                'port': self.pump_port,
                'current': self.pump_current_A
            })
        
        return sources
    
    def get_solver_options(self) -> Dict:
        """Get additional solver options"""
        options = {}
        
        # Boolean options for special analysis modes
        if self.enable_dc_bias:
            options['dc'] = True
            
        if self.enable_three_wave_mixing:
            options['threewavemixing'] = True
            
        if self.enable_four_wave_mixing:
            options['fourwavemixing'] = True
        
        # Numerical solver parameters if specified (otherwise use hbsolve defaults)
        if self.iterations is not None:
            options['iterations'] = self.iterations
            
        if self.ftol is not None:
            options['ftol'] = self.ftol
            
        if self.switchofflinesearchtol is not None:
            options['switchofflinesearchtol'] = self.switchofflinesearchtol
            
        if self.alphamin is not None:
            options['alphamin'] = self.alphamin
            
        # Sorting is always specified (we default to "name" for consistency)
        options['sorting'] = f':{self.sorting}'
            
        return options
    
    def print_config(self):
        """Print simulation configuration"""
        print("=== Simulation Configuration ===")
        print(f"Signal frequency range: {self.freq_start_GHz} - {self.freq_stop_GHz} GHz")
        print(f"Frequency step: {self.freq_step_GHz} GHz")
        print(f"\nPump Configuration:")
        print(f"  Frequency: {self.pump_freq_GHz} GHz")
        print(f"  Current: {self.pump_current_A*1e6:.1f} μA")
        print(f"  Port: {self.pump_port}")
        print(f"\nSignal Configuration:")
        print(f"  Port: {self.signal_port}")
        print(f"  Output port: {self.output_port}")
        
        if self.enable_dc_bias:
            print(f"\nDC Bias Configuration:")
            if self.dc_current_A is not None:
                print(f"  Current: {self.dc_current_A*1e6:.1f} μA")
            print(f"  Port: {self.dc_port}")
            
            if self.dc_flux_bias is not None:
                print(f"  Flux bias: {self.dc_flux_bias:.3f} Φ₀")
                if self.dc_flux_bias == 1/3:
                    print(f"  (Φ₀/3 - optimal for SQUID modulation)")
                    if self.mutual_inductance_H is not None:
                        print(f"  Mutual inductance: {self.mutual_inductance_H*1e12:.1f} pH")
        
        print(f"\nHarmonic Balance:")
        print(f"  Pump harmonics: {self.Npumpharmonics}")
        print(f"  Modulation harmonics: {self.Nmodulationharmonics}")
        
        if self.enable_three_wave_mixing or self.enable_four_wave_mixing != True:
            print(f"\nNonlinear Mixing:")
            if self.enable_three_wave_mixing:
                print(f"  Three-wave mixing: Enabled")
            if not self.enable_four_wave_mixing:
                print(f"  Four-wave mixing: Disabled (non-standard!)")


# Helper function for easy flux-based DC bias calculation
def flux_bias_config(flux_over_phi0: float, mutual_inductance_pH: float) -> dict:
    """
    Helper to create DC bias configuration from flux
    
    Args:
        flux_over_phi0: Desired flux as fraction of Φ₀ (e.g., 1/3)
        mutual_inductance_pH: Mutual inductance in picoHenries
    
    Returns:
        Dictionary of config parameters
    
    Example:
        config = TWPASimulationConfig(
            **flux_bias_config(1/3, 2.2),  # Φ₀/3 with 2.2 pH
            freq_start_GHz=5.0,
            # ... other parameters
        )
    """
    return {
        'enable_dc_bias': True,
        'dc_flux_bias': flux_over_phi0,
        'mutual_inductance_H': mutual_inductance_pH * 1e-12
    }


def build_julia_sources_string(config: TWPASimulationConfig) -> str:
    """Build the sources array string for Julia evaluation"""
    sources = config.get_sources()
    
    if not sources:
        return "sources = []"
    
    source_strings = []
    for source in sources:
        mode_str = f"mode={source['mode']}"
        port_str = f"port={source['port']}"
        current_str = f"current={source['current']}"
        source_strings.append(f"({mode_str},{port_str},{current_str})")
    
    return f"sources = [{','.join(source_strings)}]"


def build_hbsolve_string(config: TWPASimulationConfig) -> str:
    """Build the hbsolve command string with all options"""

    # Basic command
    cmd = "sol = hbsolve(ws, wp, sources, Nmodulationharmonics, Npumpharmonics, circuit, circuitdefs"

    # Add optional parameters
    options = config.get_solver_options()

    # Add returnnodeflux if signal nodeflux storage is requested
    if config.store_signal_nodeflux:
        options['returnnodeflux'] = True

    if options:
        option_strings = []
        for key, value in options.items():
            if isinstance(value, bool):
                option_strings.append(f"{key}={str(value).lower()}")
            elif isinstance(value, str):
                option_strings.append(f"{key}={value}")  # For sorting=:name
            elif isinstance(value, (int, float)):
                option_strings.append(f"{key}={value}")
            else:
                option_strings.append(f"{key}={value}")
        
        # Sort to ensure consistent order
        option_strings.sort()
        cmd += ", " + ", ".join(option_strings)
    
    cmd += ")"

    return cmd


def build_hblinsolve_string(config: TWPASimulationConfig) -> str:
    """Build the hblinsolve command string for linear analysis

    hblinsolve defaults: threewavemixing=false, fourwavemixing=true
    Only add parameters when user explicitly sets them and they differ from defaults.
    """

    cmd = "sol = hblinsolve(ws, circuit, circuitdefs"
    cmd += ", Nmodulationharmonics=Nmodulationharmonics"

    # Only add threewavemixing if explicitly set to True (default is False)
    if config.enable_three_wave_mixing is not None and config.enable_three_wave_mixing:
        cmd += ", threewavemixing=true"

    # Only add fourwavemixing if explicitly set to False (default is True)
    if config.enable_four_wave_mixing is not None and not config.enable_four_wave_mixing:
        cmd += ", fourwavemixing=false"

    cmd += f", sorting={config.get_solver_options()['sorting']}"
    cmd += ", returnS=true"
    cmd += ")"
    return cmd


@dataclass
class TWPAResults:
    """Container for TWPA simulation results"""
    frequencies_GHz: np.ndarray
    S11: np.ndarray
    S12: np.ndarray
    S21: np.ndarray
    S22: np.ndarray
    quantum_efficiency: np.ndarray
    commutation_error: np.ndarray
    idler_response: np.ndarray
    backward_idler_response: Optional[np.ndarray] = None
    modes: Optional[List] = None
    netlist_name: Optional[str] = None
    config: Optional[TWPASimulationConfig] = None

    # Pump harmonics data (always available for nonlinear solver)
    pump_nodeflux: Optional[np.ndarray] = None  # Shape: (num_pump_harmonics, num_nodes)
    num_pump_harmonics: Optional[int] = None
    pump_freq_Hz: Optional[float] = None  # Pump frequency in Hz for power calculation

    # Signal harmonics data (requires store_signal_nodeflux=True)
    signal_nodeflux: Optional[np.ndarray] = None  # Shape varies with modes/ports/freqs

    # Shared spatial info
    num_nodes: Optional[int] = None
    total_cells: Optional[int] = None  # From netlist metadata

    def save(self, filename: Optional[str] = None,
                metadata: Optional[dict] = None,
                config: Optional[TWPASimulationConfig] = None,
                use_filecounter: bool = True,
                output_dir: Optional[str] = None) -> str:
        """Save results to file with flexible naming options

        Args:
            filename: Explicit filename. If None, auto-generates from config
            metadata: Custom metadata dict. If None and config provided, generates from config
            config: Simulation configuration (used for auto-naming and metadata)
            use_filecounter: If True and filename is None, uses filecounter for auto-naming
            output_dir: Output directory. If None, uses package's results/ folder

        Returns:
            str: The actual filename where results were saved

        Examples:
            # Auto-name with filecounter
            results.save(config=sim_config)

            # Explicit filename
            results.save("my_results.npz")

            # Auto-name without filecounter
            results.save(config=sim_config, use_filecounter=False)

            # Save to custom directory
            results.save(config=sim_config, output_dir="/path/to/workspace")
        """
        from pathlib import Path

        # Use provided config or fall back to stored config
        config_to_use = config or self.config

        # Determine filename
        if filename is None:
            # Auto-generate filename
            if not config_to_use:
                raise ValueError("No config available for auto-naming. Provide filename or config.")
            if not self.netlist_name:
                raise ValueError("No netlist_name available for auto-naming")

            # Use custom output_dir or default RESULTS_DIR
            if output_dir is None:
                results_dir = str(RESULTS_DIR)
            else:
                results_dir = str(Path(output_dir))
                Path(results_dir).mkdir(parents=True, exist_ok=True)

            # At this point, config_to_use is guaranteed to be not None
            assert config_to_use is not None  # Help Pylance understand

            if use_filecounter:
                # Create pattern with wildcard
                filename_pattern = f"{results_dir}/{self.netlist_name}_pump{config_to_use.pump_freq_GHz:.2f}GHz_*.npz"
                filename, file_number = filecounter(filename_pattern)
            else:
                # Simple filename without counter
                filename = f"{results_dir}/{self.netlist_name}_pump{config_to_use.pump_freq_GHz:.2f}GHz.npz"
                file_number = None
        else:
            file_number = None
        
        # Generate metadata if not provided
        if metadata is None and config_to_use is not None:
            metadata = {
                'netlist_name': self.netlist_name,
                'pump_freq_GHz': config_to_use.pump_freq_GHz,
                'pump_current_A': config_to_use.pump_current_A,
                'freq_start_GHz': config_to_use.freq_start_GHz,
                'freq_stop_GHz': config_to_use.freq_stop_GHz,
                'freq_step_GHz': config_to_use.freq_step_GHz,
                'pump_port': config_to_use.pump_port,
                'signal_port': config_to_use.signal_port,
                'output_port': config_to_use.output_port,
                'Npumpharmonics': config_to_use.Npumpharmonics,
                'Nmodulationharmonics': config_to_use.Nmodulationharmonics,
                'enable_dc_bias': config_to_use.enable_dc_bias,
                'dc_current_A': config_to_use.dc_current_A if config_to_use.enable_dc_bias else None,
                'dc_port': config_to_use.dc_port,
                'enable_three_wave_mixing': config_to_use.enable_three_wave_mixing,
                'enable_four_wave_mixing': config_to_use.enable_four_wave_mixing,
            }
            
            if file_number is not None:
                metadata['file_number'] = file_number
        
        # Save the data
        save_dict = {
            'frequencies_GHz': self.frequencies_GHz,
            'S11': self.S11,
            'S12': self.S12,
            'S21': self.S21,
            'S22': self.S22,
            'quantum_efficiency': self.quantum_efficiency,
            'commutation_error': self.commutation_error,
            'idler_response': self.idler_response,
            'backward_idler_response': self.backward_idler_response,
            'modes': self.modes,
            'netlist_name': self.netlist_name,  # Save this in the data too
            # Pump harmonics data
            'pump_nodeflux': self.pump_nodeflux,
            'num_pump_harmonics': self.num_pump_harmonics,
            'pump_freq_Hz': self.pump_freq_Hz,
            # Signal harmonics data
            'signal_nodeflux': self.signal_nodeflux,
            # Spatial info
            'num_nodes': self.num_nodes,
            'total_cells': self.total_cells
        }
        
        # Add metadata if provided
        if metadata is not None:
            save_dict['metadata'] = np.array(json.dumps(metadata))
        
        np.savez_compressed(filename, **save_dict)
        print(f"Results saved to: {filename}")
        
        if file_number is not None:
            print(f"✅ Results saved as file #{file_number:02d}")
        
        return filename
        
    @classmethod
    def load(cls, filename: str) -> Tuple['TWPAResults', dict]:
        """Load TWPA results and metadata from file
        
        Args:
            filename: Input filename (with or without path and extension)
            
        Returns:
            tuple: (TWPAResults instance, metadata dict)
        """
        # Handle filename
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        # If no path specified, assume results directory
        if os.path.sep not in filename and '/' not in filename:
            filename = os.path.join(str(RESULTS_DIR), filename)
        
        # Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Load data
        data = np.load(filename, allow_pickle=True)
        
        # Load metadata if present
        metadata = {}
        if 'metadata' in data:
            metadata = json.loads(str(data['metadata']))
        
        # Reconstruct config from metadata if available
        config = None
        if metadata and all(key in metadata for key in ['pump_freq_GHz', 'signal_port', 'output_port']):
            config = cls._config_from_metadata(metadata)
        
        # Helper to safely get array or None
        def get_array_or_none(key):
            if key in data:
                val = data[key]
                if val is not None and (not hasattr(val, 'shape') or val.shape != ()):
                    return val
            return None

        # Create TWPAResults instance
        results = cls(
            frequencies_GHz=data['frequencies_GHz'],
            S11=data['S11'],
            S12=data['S12'],
            S21=data['S21'],
            S22=data['S22'],
            quantum_efficiency=data['quantum_efficiency'],
            commutation_error=data['commutation_error'],
            idler_response=data['idler_response'],
            backward_idler_response=get_array_or_none('backward_idler_response'),
            modes=data['modes'].tolist() if 'modes' in data else None,
            netlist_name=data.get('netlist_name', metadata.get('netlist_name', None)),
            config=config,  # Reconstructed config
            # Pump harmonics data
            pump_nodeflux=get_array_or_none('pump_nodeflux'),
            num_pump_harmonics=int(data['num_pump_harmonics']) if 'num_pump_harmonics' in data and data['num_pump_harmonics'] is not None else None,
            pump_freq_Hz=float(data['pump_freq_Hz']) if 'pump_freq_Hz' in data and data['pump_freq_Hz'] is not None else None,
            # Signal harmonics data
            signal_nodeflux=get_array_or_none('signal_nodeflux'),
            # Spatial info
            num_nodes=int(data['num_nodes']) if 'num_nodes' in data and data['num_nodes'] is not None else None,
            total_cells=int(data['total_cells']) if 'total_cells' in data and data['total_cells'] is not None else None
        )
        
        return results, metadata
    

    def plot(self, config: Optional[TWPASimulationConfig] = None,  # Make optional
         netlist_name: Optional[str] = None,
         save_path: Optional[str] = None,
         auto_save: bool = False,  # Change default to False
         show_plot: bool = True,
         max_mode_order_to_plot: int = 2,
         output_dir: Optional[str] = None) -> Figure:
        """Plot TWPA simulation results in a 4x1 layout

        Args:
            config: Simulation configuration. If None, uses stored config from results
            netlist_name: Name of the netlist (for filename generation when saving)
            save_path: Optional explicit path to save the figure
            auto_save: If True and save_path is None, automatically saves with incremented filename
            show_plot: Whether to display the plot (default: True)
            max_mode_order_to_plot: Maximum mode order to plot for idlers (default: 2)
            output_dir: Output directory for saved plots. If None, uses package's results/ folder

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Use provided config or fall back to stored config
        config_to_use = config or self.config
        if not config_to_use:
            raise ValueError("No config available. Provide config parameter or ensure results have stored config.")

        # Help Pylance understand config_to_use is not None
        assert config_to_use is not None

        # Determine amplifier type
        is_reflection_amp = (config_to_use.signal_port == config_to_use.output_port)
        amp_type = "Reflection Amplifier (PA)" if is_reflection_amp else "Transmission Amplifier (TWPA)"
        
        # Determine which S-parameter represents the gain
        if is_reflection_amp:
            if config_to_use.signal_port == 1:
                gain_param = "S11"
                gain_data = self.S11
            else:
                gain_param = "S22"
                gain_data = self.S22
        else:
            if config_to_use.signal_port == 1 and config_to_use.output_port == 2:
                gain_param = "S21"
                gain_data = self.S21
            elif config_to_use.signal_port == 2 and config_to_use.output_port == 1:
                gain_param = "S12"
                gain_data = self.S12
            else:
                gain_param = "S21"
                gain_data = self.S21
        
        print(f"=== Plotting Amplifier Performance ===")
        print(f"Amplifier type: {amp_type}")
        print(f"Gain parameter: {gain_param}")

        # Check if linear mode - only plot S-parameters
        is_linear_mode = (config_to_use.solver_mode == "linear" and
                         config_to_use.Nmodulationharmonics == 0)

        if not is_linear_mode:
            # Check commutation errors (only for nonlinear mode)
            cm_errors = np.abs(self.commutation_error)
            print(f"Commutation relation |1-CM|:")
            print(f"  Min: {np.min(cm_errors):.2e}")
            print(f"  Max: {np.max(cm_errors):.2e}")
            print(f"  Mean: {np.mean(cm_errors):.2e}")
            if np.max(cm_errors) > 0.01:
                print("  ⚠️ Warning: Maximum error > 1% - check energy conservation")

        # Create figure layout based on mode
        if is_linear_mode:
            # Linear mode: only S-parameters plot
            print("Linear mode: Plotting S-parameters only")
            fig = plt.figure(figsize=(8.6/2.54, 3.5))
            ax1 = fig.add_subplot(1, 1, 1)

            # Plot S-parameters
            self._plot_s_parameters(ax1, config_to_use)
        else:
            # Nonlinear mode: full 4-panel plot
            fig = plt.figure(figsize=(8.6/2.54, 7))
            gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[2, 1, 1, 1], hspace=0.5)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            ax4 = fig.add_subplot(gs[3])

            # Plot 1: S-parameters
            self._plot_s_parameters(ax1, config_to_use)

            # Plot 2: Forward idlers
            self._plot_forward_idlers(ax2, config_to_use, max_mode_order_to_plot)

            # Plot 3: Backward idlers
            self._plot_backward_idlers(ax3, config_to_use, max_mode_order_to_plot)

            # Plot 4: Quantum efficiency
            self._plot_quantum_efficiency(ax4, config_to_use)
        
        # Handle saving
        if save_path:
            # Use explicit path
            plt.savefig(save_path, format='svg', bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        elif auto_save:
            # Determine which netlist name to use
            name_to_use = netlist_name if netlist_name else self.netlist_name

            if name_to_use:
                # Use custom output_dir or default RESULTS_DIR
                from pathlib import Path
                if output_dir is None:
                    results_dir = str(RESULTS_DIR)
                else:
                    results_dir = str(Path(output_dir))
                    Path(results_dir).mkdir(parents=True, exist_ok=True)

                # Create filename pattern with wildcard for counter
                filename_pattern = f"{results_dir}/{name_to_use}_pump{config_to_use.pump_freq_GHz:.2f}GHz_*.svg"
                
                # Get filename with incremented counter
                filename, file_number = filecounter(filename_pattern)
                
                plt.savefig(filename, format='svg', bbox_inches='tight')
                print(f"Figure saved to: {filename}")
                print(f"✅ Figure saved as file #{file_number:02d}")
            else:
                print("⚠️  Warning: auto_save=True but no netlist_name available.")
                print("   Provide netlist_name parameter or ensure results have stored netlist_name.")
        
        if show_plot:
            plt.show()
        
        return fig

    def _plot_s_parameters(self, ax: Axes, config: TWPASimulationConfig):
        """Plot S-parameters subplot"""
        ax.plot(self.frequencies_GHz, 10*np.log10(self.S21), color=blue, linewidth=linewidth)
        ax.plot(self.frequencies_GHz, 10*np.log10(self.S12), color=red, linewidth=linewidth)
        ax.plot(self.frequencies_GHz, 10*np.log10(self.S11), color=green, linewidth=linewidth, alpha=0.7)
        ax.plot(self.frequencies_GHz, 10*np.log10(self.S22), color=orange, linewidth=linewidth, alpha=0.7)
        
        # Add vertical line at pump frequency
        ax.axvline(config.pump_freq_GHz, color=purple, linestyle=':', alpha=0.5)
        
        # Create legend handles
        s21_handle = Line2D([0], [0], color=blue, linewidth=linewidth, label=r'$|S_{21}|$')
        s12_handle = Line2D([0], [0], color=red, linewidth=linewidth, label=r'$|S_{12}|$')
        s11_handle = Line2D([0], [0], color=green, linewidth=linewidth, alpha=0.7, label=r'$|S_{11}|$')
        s22_handle = Line2D([0], [0], color=orange, linewidth=linewidth, alpha=0.7, label=r'$|S_{22}|$')
        pump_handle = Line2D([0], [0], color=purple, linestyle=':', alpha=0.5, 
                            label=rf'$f_a = {config.pump_freq_GHz}$ GHz' + '\n' + 
                                  rf'$I_a = {config.pump_current_A*1e6:.1f}$ $\mu$A')
        
        ax.legend(
            handles=[s21_handle, s11_handle, pump_handle, s12_handle, s22_handle],
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
            columnspacing=-3.5
        )
        
        ax.set_ylabel(r'$|S|$-parameters [dB]', fontsize=fontsize)
        ax.set_title('Signal', fontsize=fontsize_title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-30, 30)
        ax.set_xlim(config.freq_start_GHz, config.freq_stop_GHz)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    def _plot_forward_idlers(self, ax: Axes, config: TWPASimulationConfig, max_mode_order: int = 2):
        """Plot forward idlers subplot"""
        if self.idler_response.ndim > 1 and self.idler_response.shape[0] > 1:
            idler_data = self.idler_response
            
            # Determine shape and orientation
            if idler_data.shape[0] == len(self.frequencies_GHz):
                num_modes = idler_data.shape[1]
                freq_axis = 0
            else:
                num_modes = idler_data.shape[0]
                freq_axis = 1
            
            # Get mode information
            if self.modes is not None:
                modes = self.modes
            else:
                modes = None
            
            # Create mode mapping
            n_to_mode_idx = {}
            if modes is not None:
                for mode_idx in range(num_modes):
                    mode_n = modes[mode_idx][0]
                    n_to_mode_idx[mode_n] = mode_idx
            
            # Create plot order
            available_modes = sorted(n_to_mode_idx.keys())
            plot_order = []
            max_abs_n = min(max(abs(n) for n in available_modes if n != 0), 
                           max_mode_order)
            
            for i in range(1, max_abs_n + 1):
                if -i in n_to_mode_idx:
                    plot_order.append(-i)
                if i in n_to_mode_idx:
                    plot_order.append(i)
            
            # Plot forward idlers
            colors = PLOT_CONFIG['colors']
            for i, n in enumerate(plot_order):
                if n not in n_to_mode_idx:
                    continue
                    
                mode_idx = n_to_mode_idx[n]
                
                if freq_axis == 0:
                    idler_response_dB = 10*np.log10(np.abs(idler_data[:, mode_idx]) + 1e-12)
                else:
                    idler_response_dB = 10*np.log10(np.abs(idler_data[mode_idx, :]) + 1e-12)
                
                label = get_mode_label(n, config.pump_freq_GHz)
                color = colors[i % len(colors)]
                ax.plot(self.frequencies_GHz, idler_response_dB, 
                       linewidth=linewidth, color=color, label=label)
        
        # ax.set_ylabel(rf'$S_{{{config.output_port}{config.signal_port}}}$ [dB]', fontsize=fontsize)
        ax.set_ylabel(rf'$|S_{{{config.output_port}{config.signal_port}}}|(\mathrm{{sig}}\rightarrow\mathrm{{idl}})$ [dB]', fontsize=fontsize)
        ax.set_title('Forward Idlers', fontsize=fontsize_title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(config.freq_start_GHz, config.freq_stop_GHz)
        ax.set_ylim(-30, 30)
        ax.set_yticks([-20, 0, 20])
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    def _plot_backward_idlers(self, ax: Axes, config: TWPASimulationConfig, max_mode_order: int = 2):
        """Plot backward idlers subplot"""
        if self.backward_idler_response is not None and self.backward_idler_response.ndim > 1:
            backward_idler_data = self.backward_idler_response
            
            # Use same mode mapping as forward idlers
            if self.modes is not None:
                modes = self.modes
                n_to_mode_idx = {}
                for mode_idx in range(len(modes)):
                    mode_n = modes[mode_idx][0]
                    n_to_mode_idx[mode_n] = mode_idx
                
                # Same plot order as forward idlers
                available_modes = sorted(n_to_mode_idx.keys())
                plot_order = []
                max_abs_n = min(max(abs(n) for n in available_modes if n != 0), 
                               max_mode_order)
                
                for i in range(1, max_abs_n + 1):
                    if -i in n_to_mode_idx:
                        plot_order.append(-i)
                    if i in n_to_mode_idx:
                        plot_order.append(i)
                
                # Determine data orientation
                if backward_idler_data.shape[0] == len(self.frequencies_GHz):
                    freq_axis = 0
                else:
                    freq_axis = 1
                
                # Plot backward idlers
                colors = PLOT_CONFIG['colors']
                for i, n in enumerate(plot_order):
                    if n not in n_to_mode_idx:
                        continue
                        
                    mode_idx = n_to_mode_idx[n]
                    
                    if freq_axis == 0:
                        idler_response_dB = 10*np.log10(np.abs(backward_idler_data[:, mode_idx]) + 1e-12)
                    else:
                        idler_response_dB = 10*np.log10(np.abs(backward_idler_data[mode_idx, :]) + 1e-12)
                    
                    label = get_mode_label(n, config.pump_freq_GHz)
                    color = colors[i % len(colors)]
                    ax.plot(self.frequencies_GHz, idler_response_dB, 
                           linewidth=linewidth, color=color, label=label)
                
                ax.legend(
                    loc='center right',
                    fontsize=fontsize_legend,
                    frameon=True,
                    fancybox=True,
                    framealpha=0.7,
                    facecolor='white',
                    edgecolor='gray',
                    borderpad=0.3,
                    columnspacing=1.0,
                    handlelength=1.5,
                    handletextpad=0.5,
                    borderaxespad=0.5
                )
                ax.set_ylim(-40, 0)
            else:
                ax.text(0.5, 0.5, 'Backward idler data not available', 
                       transform=ax.transAxes, ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'Backward idler data not available', 
                   transform=ax.transAxes, ha='center', va='center')
        
        # ax.set_ylabel(rf'$S_{{{config.signal_port}{config.output_port}}}$ [dB]', fontsize=fontsize)
        ax.set_ylabel(rf'$|S_{{{config.signal_port}{config.output_port}}}|(\mathrm{{sig}}\rightarrow\mathrm{{idl}})$ [dB]', fontsize=fontsize)
        ax.set_title('Backward Idlers', fontsize=fontsize_title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(config.freq_start_GHz, config.freq_stop_GHz)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    def _plot_quantum_efficiency(self, ax: Axes, config: TWPASimulationConfig):
        """Plot quantum efficiency subplot"""
        ax.plot(self.frequencies_GHz, self.quantum_efficiency, color=blue, linewidth=linewidth)
        ax.axhline(1.0, color=black, linestyle='--', alpha=0.7, label='Ideal')
        ax.set_xlabel('frequency [GHz]', fontsize=fontsize)
        ax.set_ylabel(r'$\mathsf{QE}/\mathsf{QE}_\mathsf{ideal}$', fontsize=fontsize)
        ax.set_title('Quantum Efficiency', fontsize=fontsize_title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.7, 1.05)
        ax.set_yticks([0.7, 0.8, 0.9, 1])
        ax.set_xlim(config.freq_start_GHz, config.freq_stop_GHz)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    def plot_pump_harmonics(
        self,
        ax: Optional[Axes] = None,
        config: Optional[TWPASimulationConfig] = None,
        max_pump_harmonic: int = 3,
        position_normalized: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Figure:
        """Plot pump harmonic power distribution along the TWPA.

        Convenience wrapper around plot_harmonics() for pump-only plots.
        """
        return self.plot_harmonics(
            ax=ax, config=config,
            max_pump_harmonic=max_pump_harmonic,
            max_signal_mode_order=0,
            position_normalized=position_normalized,
            save_path=save_path, show_plot=show_plot
        )

    def plot_signal_harmonics(
        self,
        ax: Optional[Axes] = None,
        config: Optional[TWPASimulationConfig] = None,
        max_mode_order: int = 2,
        signal_freq_GHz: Optional[float] = None,
        position_normalized: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Figure:
        """Plot signal/idler power distribution along the TWPA.

        Convenience wrapper around plot_harmonics() for signal-only plots.
        """
        return self.plot_harmonics(
            ax=ax, config=config,
            max_pump_harmonic=0,
            max_signal_mode_order=max_mode_order,
            signal_freq_GHz=signal_freq_GHz,
            position_normalized=position_normalized,
            save_path=save_path, show_plot=show_plot
        )

    def plot_harmonics(
        self,
        ax: Optional[Axes] = None,
        config: Optional[TWPASimulationConfig] = None,
        max_pump_harmonic: int = 3,
        max_signal_mode_order: int = 2,
        signal_freq_GHz: Optional[float] = None,
        position_normalized: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Figure:
        """Plot pump and signal/idler power distribution along the TWPA.

        Displays spatial evolution of pump harmonics (solid lines) and
        signal/idler modes (dashed lines) along the device.

        Colors: pump = purple (lighter for higher harmonics),
        signal = blue (lighter for higher orders),
        idler = red (lighter for higher orders).

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure.
            config: Simulation configuration. If None, uses stored config.
            max_pump_harmonic: Number of pump harmonics to plot (0 to disable).
            max_signal_mode_order: Max |n| for signal modes to plot (0 to disable).
            signal_freq_GHz: Signal frequency in GHz for spatial plot. Uses the
                closest available frequency. If None, uses the center of the band.
            position_normalized: If True, x-axis is 0-1. If False (default), cell numbers.
            save_path: Optional path to save the figure.
            show_plot: Whether to display the plot.

        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        # Use provided config or fall back to stored config
        config_to_use = config or self.config
        if not config_to_use:
            raise ValueError(
                "No config available. Provide config parameter or ensure "
                "results have stored config."
            )

        # Physical constants
        phi0 = 3.29105976e-16  # Reduced flux quantum (V*s)
        Z0 = 50.0  # Characteristic impedance (Ohms)
        w_pump = 2 * np.pi * config_to_use.pump_freq_GHz * 1e9

        # Determine number of nodes and total cells
        if self.pump_nodeflux is not None and self.pump_nodeflux.ndim > 1:
            num_nodes = self.pump_nodeflux.shape[1]
        elif self.signal_nodeflux is not None and self.signal_nodeflux.ndim == 5:
            num_nodes = self.signal_nodeflux.shape[1]
        elif self.num_nodes is not None:
            num_nodes = self.num_nodes
        else:
            raise ValueError("No nodeflux data available.")

        total_cells = self.total_cells if self.total_cells else num_nodes

        # Create position array
        if position_normalized:
            position = np.linspace(0, 1, num_nodes)
            xlabel = 'Normalized position'
        else:
            position = np.linspace(1, total_cells, num_nodes)
            xlabel = 'Cell number'

        # Create figure if no axes provided
        create_new_figure = ax is None
        if create_new_figure:
            fig = plt.figure(figsize=(8.6/2.54, 3.5))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = ax.get_figure()

        # Color scheme:
        #   Pump: purple (fundamental), then brown, darkblue, ... for higher harmonics
        #   Signal (n=0): blue — plotted last so it's on top
        #   First idler: orange, then red, green, pink, ... for higher orders
        pump_colors = [purple, brown, darkblue, pink]
        idler_colors = [orange, red, green, pink, yellow]
        signal_color = blue

        legend_handles = []

        # === Pump harmonics (solid lines) ===
        if max_pump_harmonic > 0 and self.pump_nodeflux is not None:
            nodeflux_pump = self.pump_nodeflux
            num_harmonics = nodeflux_pump.shape[0] if nodeflux_pump.ndim > 1 else 1
            harmonics_to_plot = min(max_pump_harmonic, num_harmonics)

            for h in range(harmonics_to_plot):
                harmonic_number = h + 1

                if nodeflux_pump.ndim > 1:
                    flux_harmonic = nodeflux_pump[h, :]
                else:
                    flux_harmonic = nodeflux_pump

                w_harmonic = harmonic_number * w_pump
                I_magnitude = np.abs(flux_harmonic) * w_harmonic * phi0 / Z0
                power_watts = 0.5 * I_magnitude**2 * Z0
                power_dBm = 10 * np.log10(power_watts * 1000 + 1e-30)

                freq_GHz = harmonic_number * config_to_use.pump_freq_GHz
                if harmonic_number == 1:
                    harm_label = r"$1\times f_p$"
                else:
                    harm_label = rf"${harmonic_number}\times f_p$"
                full_label = f"{harm_label} ({freq_GHz:.1f} GHz)"

                color = pump_colors[h % len(pump_colors)]
                line, = ax.plot(
                    position, power_dBm,
                    color=color, linewidth=linewidth,
                    linestyle='-', label=full_label
                )
                legend_handles.append(line)

        # === Signal/idler modes (dashed lines) ===
        # Signal (n=0) is plotted last so it appears on top
        if max_signal_mode_order > 0 and self.signal_nodeflux is not None:
            nodeflux_sig = self.signal_nodeflux

            if self.modes is None:
                raise ValueError("Mode information not available in results.")

            modes = self.modes
            n_to_mode_idx = {}
            for mode_idx in range(len(modes)):
                mode_n = modes[mode_idx][0]
                n_to_mode_idx[mode_n] = mode_idx

            # Signal mode is (0,): f_s + 0*f_a = f_s
            signal_mode_n = 0
            if signal_mode_n not in n_to_mode_idx:
                raise ValueError(f"Signal mode (0,) not found in modes: {modes}")
            input_mode_idx = n_to_mode_idx[signal_mode_n]

            # Input port (0-based)
            input_port_idx = config_to_use.signal_port - 1

            # Get frequency array and find closest index
            freq_array = config_to_use.frequency_array()  # in Hz
            freq_array_GHz = freq_array * 1e-9

            if signal_freq_GHz is None:
                # Default: center of the band
                signal_freq_GHz = (config_to_use.freq_start_GHz + config_to_use.freq_stop_GHz) / 2

            freq_index = int(np.argmin(np.abs(freq_array_GHz - signal_freq_GHz)))
            actual_freq_GHz = freq_array_GHz[freq_index]
            freq_signal_Hz = freq_array[freq_index]

            if abs(actual_freq_GHz - signal_freq_GHz) > config_to_use.freq_step_GHz:
                print(f"  Note: Requested {signal_freq_GHz:.2f} GHz, using closest: {actual_freq_GHz:.2f} GHz")

            # Validate freq_index against nodeflux shape
            if nodeflux_sig.ndim == 5:
                n_freqs = nodeflux_sig.shape[4]
                num_nodes_sig = nodeflux_sig.shape[1]
            elif nodeflux_sig.ndim == 3:
                n_freqs = nodeflux_sig.shape[2]
                num_nodes_sig = nodeflux_sig.shape[0] // len(modes)
            else:
                raise ValueError(f"Unexpected signal nodeflux shape: {nodeflux_sig.shape}")

            if freq_index >= n_freqs:
                raise ValueError(f"freq_index={freq_index} out of range (0 to {n_freqs-1})")

            # Build plot order: idlers and higher modes first, signal (n=0) LAST
            available_modes = sorted(n_to_mode_idx.keys())
            max_abs_n = min(max(abs(n) for n in available_modes), max_signal_mode_order)

            # Collect idlers/higher modes (everything except signal n=0)
            plot_order = []
            for i in range(1, max_abs_n + 1):
                if -i in n_to_mode_idx:
                    plot_order.append(-i)
                if i in n_to_mode_idx:
                    plot_order.append(i)
            # Signal (n=0) goes last so it's drawn on top
            if 0 in n_to_mode_idx:
                plot_order.append(0)

            # Helper to compute and plot one mode
            def _plot_signal_mode(mode_n, color, legend_list):
                output_mode_idx = n_to_mode_idx[mode_n]

                if nodeflux_sig.ndim == 5:
                    flux_spatial = nodeflux_sig[output_mode_idx, :, input_mode_idx, input_port_idx, freq_index]
                elif nodeflux_sig.ndim == 3:
                    num_modes_raw = len(modes)
                    row_start = output_mode_idx * num_nodes_sig
                    row_end = row_start + num_nodes_sig
                    n_ports = nodeflux_sig.shape[1] // num_modes_raw
                    col = input_mode_idx * n_ports + input_port_idx
                    flux_spatial = nodeflux_sig[row_start:row_end, col, freq_index]

                w_mode = 2 * np.pi * freq_signal_Hz + mode_n * w_pump
                I_magnitude = np.abs(flux_spatial) * np.abs(w_mode) * phi0 / Z0
                power_watts = 0.5 * I_magnitude**2 * Z0
                power_dBm = 10 * np.log10(power_watts * 1000 + 1e-30)

                label = get_mode_label(mode_n, config_to_use.pump_freq_GHz)
                freq_mode_GHz = np.abs(freq_signal_Hz * 1e-9 + mode_n * config_to_use.pump_freq_GHz)
                full_label = f"{label} ({freq_mode_GHz:.1f} GHz)"

                line, = ax.plot(
                    position, power_dBm,
                    color=color, linewidth=linewidth,
                    label=full_label
                )
                legend_list.append(line)

            # Assign colors: idlers get idler_colors, signal gets signal_color
            idler_idx = 0
            for mode_n in plot_order:
                if mode_n == 0:
                    _plot_signal_mode(mode_n, signal_color, legend_handles)
                else:
                    color = idler_colors[idler_idx % len(idler_colors)]
                    _plot_signal_mode(mode_n, color, legend_handles)
                    idler_idx += 1

        # === Styling ===
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel('Power [dBm]', fontsize=fontsize)

        # Build title
        has_pump = max_pump_harmonic > 0 and self.pump_nodeflux is not None
        has_signal = max_signal_mode_order > 0 and self.signal_nodeflux is not None
        if has_pump and has_signal:
            title = f'Harmonics along line ($f_s$ = {actual_freq_GHz:.2f} GHz)'
        elif has_signal:
            title = f'Signal Harmonics ($f_s$ = {actual_freq_GHz:.2f} GHz)'
        else:
            title = 'Pump Harmonics'
        ax.set_title(title, fontsize=fontsize_title)

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=fontsize)

        if position_normalized:
            ax.set_xlim(0, 1)
        else:
            ax.set_xlim(1, total_cells)

        ax.legend(
            handles=legend_handles,
            loc='best',
            fontsize=fontsize_legend,
            frameon=True,
            fancybox=True,
            framealpha=0.7,
            facecolor='white',
            edgecolor='gray',
            borderpad=0.3,
            handlelength=1.5,
            handletextpad=0.5,
            borderaxespad=0.5
        )

        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

        if create_new_figure:
            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format='svg', bbox_inches='tight')
            print(f"Harmonics plot saved to: {save_path}")

        if show_plot and create_new_figure:
            plt.show()

        return fig

    @classmethod
    def load_and_plot(cls, filename: str, show_plot: bool = True, 
                    auto_save: bool = False, save_path: Optional[str] = None,
                    max_mode_order_to_plot: int = 2) -> Tuple['TWPAResults', Figure]:
        """Load results from file and plot them
        
        Args:
            filename: Input filename (with or without path and extension)
            show_plot: Whether to display the plot
            auto_save: Whether to auto-save the figure
            save_path: Optional path to save the figure
            max_mode_order_to_plot: Maximum mode order to plot for idlers (default: 2)
            
        Returns:
            tuple: (TWPAResults instance, matplotlib Figure)
        """
        # Load results and metadata
        results, metadata = cls.load(filename)
        
        # Recreate a minimal config for plotting
        config = cls._config_from_metadata(metadata)
        
        # If netlist_name wasn't stored in results, get it from metadata
        if results.netlist_name is None and 'netlist_name' in metadata:
            results.netlist_name = metadata['netlist_name']
        
        # Plot the results
        fig = results.plot(config, show_plot=show_plot, auto_save=auto_save, save_path=save_path, max_mode_order_to_plot=max_mode_order_to_plot)
        
        # Print summary
        print("\n📊 Results loaded and ready for plotting")
        print(f"   Netlist: {metadata.get('netlist_name', 'Unknown')}")
        print(f"   Pump: {metadata.get('pump_freq_GHz', 0):.2f} GHz @ {metadata.get('pump_current_A', 0)*1e6:.1f} μA")
        
        return results, fig

    @staticmethod
    def _config_from_metadata(metadata: dict) -> TWPASimulationConfig:
        """Recreate a TWPASimulationConfig from saved metadata
        
        Only includes fields needed for plotting
        """
        return TWPASimulationConfig(
            freq_start_GHz=metadata.get('freq_start_GHz', 4.0),
            freq_stop_GHz=metadata.get('freq_stop_GHz', 12.0),
            freq_step_GHz=metadata.get('freq_step_GHz', 0.1),
            pump_freq_GHz=metadata.get('pump_freq_GHz', 8.0),
            pump_current_A=metadata.get('pump_current_A', 1e-6),
            pump_port=metadata.get('pump_port', 1),
            signal_port=metadata.get('signal_port', 1),
            output_port=metadata.get('output_port', 2),
        )

##################################################################################

class TWPASimulator:
    """Main class for running TWPA simulations using JosephsonCircuits.jl"""

    def __init__(self):
        """Initialize TWPA simulator.

        Julia configuration (local fork vs GitHub fork vs registered) is controlled
        by constants in julia_setup.py: USE_LOCAL_FORK and USE_GITHUB_FORK
        """
        self.jl: Optional[Any] = None
        self.julia_ready = False
        
        # Netlist data
        self.netlist_loaded = False
        self.jc_components = []
        self.circuit_parameters = {}
        self.metadata = {}
        self.netlist_has_loss = False
        self.netlist_loss_tangent = 0.0
        self.component_count = 0
        
        # Circuit state
        self.circuit_ready = False
        
    ##################################################################################        

    def setup_julia(self, force_reinit: bool = False):
        """Setup Julia environment.

        Julia configuration is controlled by julia_setup.py constants:
        - USE_LOCAL_FORK: Toggle local development vs remote
        - USE_GITHUB_FORK: Toggle GitHub fork vs registered (when USE_LOCAL_FORK=False)

        Args:
            force_reinit: If True, force full cleanup and reinstallation
        """
        try:
            print("🚀 Setting up Julia...")

            # Get Julia instance (configuration handled by julia_setup.py)
            self.jl = julia_setup.get_julia_for_session(
                force_reinit=force_reinit
            )

            assert self.jl is not None, "Failed to get Julia instance"

            # Load JosephsonCircuits
            self.jl.eval("using JosephsonCircuits")

            # Verify and report what version is being used
            actual_path = self.jl.eval("pathof(JosephsonCircuits)")
            print(f"✅ SUCCESS! Using JosephsonCircuits from: {actual_path}")

            if 'external_packages' in str(actual_path):
                print("🔥 Hot-reloading enabled - edit .jl files and changes auto-reload!")
            else:
                # Check if it's from GitHub by checking git_source
                try:
                    is_github_fork = self.jl.eval(f"""
                        import Pkg
                        deps = Pkg.dependencies()
                        jc_uuid = Base.UUID("{julia_setup._JOSEPHSON_CIRCUITS_UUID}")
                        if haskey(deps, jc_uuid)
                            dep_info = deps[jc_uuid]
                            dep_info.is_tracking_repo && !isnothing(dep_info.git_source) && occursin("MaxMalnou", dep_info.git_source)
                        else
                            false
                        end
                    """)
                    if is_github_fork:
                        print("📦 Using GitHub fork with Taylor expansion feature")
                    else:
                        print("📦 Using registered version")
                except Exception as e:
                    print("📦 Using registered version")

            self.julia_ready = True
                    
        except Exception as e:
            print(f"❌ Julia setup failed: {e}")
            self.julia_ready = False
            raise RuntimeError(f"Julia setup failed: {e}")
        
        if not self.julia_ready:
            raise RuntimeError("Julia setup failed. Check error messages above.")

    ##################################################################################    
        
    def load_netlist(self, netlist_name: str, netlist_dir: Optional[str] = None):
        """Load netlist from file (Cell 4).
        
        Args:
            netlist_name: Name of netlist file (with or without .py extension)
            netlist_dir: Directory containing netlists (default: uses package netlists directory)
        """
        if not self.julia_ready:
            raise RuntimeError("Julia not ready. Call setup_julia() first.")
        
        # Use package default if not specified
        if netlist_dir is None:
            netlist_dir = str(NETLISTS_DIR)
    
        print("Loading flattened netlist from file...")
        
        # Handle filename
        if not netlist_name.endswith('.py'):
            netlist_name += '.py'
        
        netlist_file = os.path.join(netlist_dir, netlist_name)
        
        if not os.path.exists(netlist_file):
            print(f"✗ Flattened netlist not found: {netlist_file}")
            print(f"Available files in {netlist_dir}:")
            if os.path.exists(netlist_dir):
                py_files = [f for f in os.listdir(netlist_dir) if f.endswith('.py')]
                if py_files:
                    for f in py_files:
                        print(f"  {f}")
                else:
                    print("  No .py files found")
            else:
                print(f"  Directory {netlist_dir} does not exist")
                print(f"  Run the netlist converter first!")
            raise FileNotFoundError(f"Netlist not found: {netlist_file}")
        
        # Load the Python file containing components and parameters
        namespace = {}
        with open(netlist_file, 'r') as f:
            exec(f.read(), namespace)
        
        # Extract all needed data from the flattened netlist file
        self.jc_components = namespace.get('jc_components', [])
        self.circuit_parameters = namespace.get('circuit_parameters', {})
        self.metadata = namespace.get('metadata', {})
        
        print(f"✓ Loaded netlist from: {netlist_file}")
        print(f"✓ Components: {len(self.jc_components)}")
        print(f"✓ Parameters: {len(self.circuit_parameters)}")
        
        # Detect dielectric loss
        self.netlist_has_loss, self.netlist_loss_tangent = self._detect_dielectric_loss()
        
        if self.netlist_has_loss:
            print(f"✓ Dielectric loss detected: tan δ = {self.netlist_loss_tangent}")
        else:
            print("✓ No dielectric loss in netlist")
        
        # Display metadata info
        if self.metadata:
            print(f"✓ Metadata: {self.metadata.get('total_cells', 0)} total cells")
        
        # Display sample components
        self._display_sample_components()
        
        # Store component count
        self.component_count = len(self.jc_components)
        
        # Store netlist name for later use (without .py extension)
        self.current_netlist_name = netlist_name.replace('.py', '')

        # Set loaded flag
        self.netlist_loaded = True

    def _detect_dielectric_loss(self) -> Tuple[bool, float]:
        """Detect if dielectric loss is present in circuit parameters."""
        # First check metadata (preferred)
        if self.metadata.get('dielectric_loss_enabled', False):
            return True, self.metadata.get('loss_tangent', 0.0)
        
        # Fallback: auto-detect from parameters
        for param, value in self.circuit_parameters.items():
            if (param.startswith('C') and 
                isinstance(value, str) and 
                'im' in value and 
                '/' in value):
                # Extract loss tangent from expression like "7.66e-14/(1+3e-3im)"
                import re
                match = re.search(r'\(1\+([0-9.e-]+)im\)', value)
                if match:
                    return True, float(match.group(1))
        
        return False, 0.0

    def _display_sample_components(self):
        """Display sample components for verification."""
        print("\nSample components:")
        # First 5
        for i, (name, node1, node2, value) in enumerate(self.jc_components[:5]):
            print(f"  {name}: {node1} -> {node2} = {value}")
        print(f" ... ")
        # Last 5
        for i, (name, node1, node2, value) in enumerate(self.jc_components[-5:]):
            print(f"  {name}: {node1} -> {node2} = {value}")
        
        print(f"total number of components: {len(self.jc_components)}")
        
        # Display parameters
        print(f"\nCircuit parameters:")
        for param, value in list(self.circuit_parameters.items())[:10]:  # Show first 10
            if isinstance(value, str):
                print(f"  {param}: {value}")  # Complex values with loss
            else:
                print(f"  {param}: {value:.6e}")
        if len(self.circuit_parameters) > 10:
            print(f"  ... and {len(self.circuit_parameters) - 10} more parameters")
        
    ##################################################################################

    def build_circuit(self):
        """Build circuit in Julia (Cell 5)."""
        if not self.netlist_loaded:
            raise RuntimeError("Netlist not loaded. Call load_netlist() first.")
        
        assert self.jl is not None, "Julia not initialized"
        
        print("Building circuit in Julia...")
        
        # Extract unique symbolic variables (for debugging)
        variables = self._extract_variables()
        
        # Initialize Julia circuit
        print("Setting up circuit...")
        
        # Convert Python boolean to Julia boolean string
        julia_has_loss = "true" if self.netlist_has_loss else "false"
        
        julia_setup_code = f"""
        # Initialize circuit - use Any type to handle poly strings
        circuit = Tuple{{String,String,String,Any}}[]
        
        # Create circuitdefs with appropriate type
        if {julia_has_loss}
            circuitdefs = Dict{{Symbol,ComplexF64}}()
        else
            circuitdefs = Dict{{Symbol,Float64}}()
        end
        
        # Setup complete flag
        setup_complete = true
        """
        
        try:
            self.jl.eval(julia_setup_code)
            setup_ok = self.jl.eval('setup_complete')
            if setup_ok:
                print(f"✓ Circuit initialization complete")
                print(f"  Found {len(variables)} unique variable names")
            else:
                raise Exception("Setup verification failed")
        except Exception as e:
            print(f"✗ Julia setup failed: {e}")
            self.circuit_ready = False
            return
        
        # Add components
        print("Adding components to circuit...")
        component_count, nl_count, failed_components = self._add_components_to_circuit()
        
        if failed_components == 0:
            print(f"✓ Successfully added {component_count} components")
            if nl_count > 0:
                print(f"  Including {nl_count} NL (nonlinear) elements")
        else:
            print(f"⚠️ Component addition had issues")
        
        # Build circuit parameters
        print("Setting parameter values...")
        if self._set_circuit_parameters():
            final_circuit_length = self.jl.eval('length(circuit)')
            print(f"✓ Circuit built successfully!")
            print(f"  Total components: {final_circuit_length}")
            if nl_count > 0:
                print(f"  NL elements: {nl_count}")
            print(f"  Parameters set: {len(self.circuit_parameters)}")
            if self.netlist_has_loss:
                print(f"  Dielectric loss: tan δ = {self.netlist_loss_tangent}")
            self.circuit_ready = True
        else:
            self.circuit_ready = False

    # Helper methods for build_circuit
    def _parse_spice_value(self, value_str):
        """Parse SPICE-style values with units"""
        if not isinstance(value_str, str):
            return float(value_str)
        
        unit_multipliers = {
            'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 
            'm': 1e-3, 'k': 1e3, 'meg': 1e6, 'g': 1e9,
        }
        
        value_str = value_str.lower().strip()
        for unit, multiplier in unit_multipliers.items():
            if value_str.endswith(unit):
                number_part = value_str[:-len(unit)]
                try:
                    return float(number_part) * multiplier
                except ValueError:
                    continue
        return float(value_str)

    def _clean_node_name(self, node):
        """Clean node names for Julia compatibility"""
        node_clean = node.replace('.', '_').replace('-', '_')
        if node_clean == '0':
            return '0'  # Ground
        elif node_clean.lower() in ['nin', 'nout']:
            return node_clean
        else:
            return node_clean

    def _should_include_component(self, name, value):
        """Determine if component should be included"""
        if name.startswith(('V', 'I')):
            return False  # Skip sources
        if name.startswith(('R', 'L', 'C', 'B', 'P', 'K', 'NL')):
            return True   # Include passive components, JJs, NL elements, and ports
        return False

    def _extract_variables_from_poly(self, poly_str):
        """Extract symbolic variables from poly string"""
        vars_found = []
        if isinstance(poly_str, str) and poly_str.startswith("poly "):
            # Remove "poly " and split by comma and space
            parts = poly_str[5:].replace(',', ' ').split()
            for part in parts:
                part = part.strip()
                if part:  # Skip empty strings
                    # Try to parse as float, if it fails it's a variable
                    try:
                        float(part)
                    except ValueError:
                        vars_found.append(part)
        return vars_found

    def _format_value_for_julia(self, value):
        """Format component value for Julia"""
        if isinstance(value, str) and value.startswith("poly "):
            # Return the poly string as is, wrapped in quotes
            return f'"{value}"'
        elif value in self.circuit_parameters:
            # For symbolic parameters, check if it's a complex value (has loss)
            param_value = self.circuit_parameters[value]
            if isinstance(param_value, str) and 'im' in param_value:
                # It's already a complex expression, use it directly
                return param_value
            else:
                # Regular numeric value
                return str(param_value)
        elif value in ['1', '2']:  # Port numbers
            return value
        elif isinstance(value, str) and ('im' in value or 'im)' in value):
            # Inline complex value or expression (e.g., "1.38e-12/(1+2e-4im)")
            # Return as-is for Julia to evaluate
            return value
        else:
            try:
                float_val = float(value)
                return str(float_val)
            except:
                return f'"{value}"'

    def _extract_variables(self):
        """Extract unique symbolic variables"""
        variables = set()
        
        # First, collect all parameter names
        param_vars = set()
        for param_name in self.circuit_parameters.keys():
            if param_name.replace('_', '').replace('.', '').isalnum():
                param_vars.add(param_name)
        
        # Then check component values
        poly_vars = set()
        for _, _, _, value in self.jc_components:
            if isinstance(value, str):
                if value.startswith("poly "):
                    # Extract variables from poly string
                    extracted = self._extract_variables_from_poly(value)
                    poly_vars.update(extracted)
                elif value in self.circuit_parameters:
                    param_vars.add(value)
                elif value == "Lj":
                    poly_vars.add("Lj")
        
        # Combine all variables
        variables = param_vars | poly_vars
        variables = sorted(list(variables))
        
        print(f"Parameter variables: {sorted(param_vars)}")
        print(f"Poly string variables: {sorted(poly_vars)}")
        print(f"All variables: {variables}")
        
        return variables

    def _add_components_to_circuit(self):

        assert self.jl is not None, "Julia not initialized"

        """Add components to Julia circuit"""
        component_additions = []
        component_count = 0
        nl_count = 0
        
        for name, node1, node2, value in self.jc_components:
            if self._should_include_component(name, value):
                value_expr = self._format_value_for_julia(value)
                node1_clean = self._clean_node_name(node1)
                node2_clean = self._clean_node_name(node2)
                
                component_additions.append(
                    f'push!(circuit,("{name}","{node1_clean}","{node2_clean}",{value_expr}));'
                )
                component_count += 1
                
                if name.startswith("NL"):
                    nl_count += 1
        
        # Execute all component additions in one go
        failed_components = 0
        if component_additions:
            all_components_code = f"""
            # Add all components
            {chr(10).join(component_additions)}
            # Set completion flag
            components_added = true
            """
            
            try:
                self.jl.eval(all_components_code)
                components_ok = self.jl.eval('components_added')
                if not components_ok:
                    failed_components = component_count
            except Exception as e:
                print(f"⚠️ Component addition failed: {e}")
                failed_components = component_count
        
        return component_count, nl_count, failed_components

    def _set_circuit_parameters(self):        
        """Set circuit parameters in Julia"""
        assert self.jl is not None, "Julia not initialized"

        params_lines = []
        
        # Add each parameter directly - the netlist already has loss applied
        for param, value in self.circuit_parameters.items():
            if isinstance(value, str):
                # Complex value (already includes loss) or expression
                params_lines.append(f'circuitdefs[:{param}] = {value}')
            else:
                # Regular numeric value
                params_lines.append(f'circuitdefs[:{param}] = {value}')
        
        params_lines.append("circuit_build_complete = true")
        params_code = '\n'.join(params_lines)
        
        # Execute the parameter setting
        try:
            self.jl.eval(params_code)
            build_complete = self.jl.eval('circuit_build_complete')
            return build_complete
        except Exception as e:
            print(f"✗ Parameter setting failed: {e}")
            return False
        
    @staticmethod
    def force_julia_reinstall():
        """Force a full Julia cleanup and reinstallation on next setup."""
        from . import julia_setup
        julia_setup.reset_julia_session()
        print("💡 Next Julia setup will perform full cleanup and reinstallation")
        
    ##################################################################################        

    def run_simulation(self, config: TWPASimulationConfig) -> TWPAResults:
        """Run harmonic balance simulation (Cell 6).
        
        Args:
            config: Simulation configuration (either internal or user-facing)
            
        Returns:
            TWPAResults object containing simulation results
        """
        if not self.circuit_ready:
            raise RuntimeError("Circuit not built. Call build_circuit() first.")
        
        # Ensure Julia is available
        if self.jl is None:
            raise RuntimeError("Julia not initialized. Call setup_julia() first.")
                
        
        print("=== Running Harmonic Balance Simulation ===")
        
        # Determine amplifier type
        is_reflection_amp = (config.signal_port == config.output_port)
        amp_type = "Reflection (PA)" if is_reflection_amp else "Transmission (TWPA)"
        gain_param = "S11" if is_reflection_amp else "S21"
        
        print(f"Detected amplifier type: {amp_type}")
        print(f"Will analyze gain using: {gain_param}")
        
        assert self.jl is not None, "Julia not initialized"
        
        try:
            print("\nSetting up simulation parameters...")
            
            # Create frequency array in Julia
            freqs_Hz = config.frequency_array()
            freq_step_Hz = freqs_Hz[1] - freqs_Hz[0]
            self.jl.eval(f'ws = 2*pi*collect({freqs_Hz[0]}:{freq_step_Hz}:{freqs_Hz[-1]})')

            # Set harmonic parameters
            self.jl.eval(f'Nmodulationharmonics = ({config.Nmodulationharmonics},)')

            # Solver-specific setup
            if config.solver_mode == "nonlinear":
                # Set pump parameters
                self.jl.eval(f'wp = (2*pi*{config.pump_freq_GHz}*1e9,)')

                # Build sources array
                sources_str = build_julia_sources_string(config)
                self.jl.eval(sources_str)

                # Set pump harmonics
                self.jl.eval(f'Npumpharmonics = ({config.Npumpharmonics},)')

                print("✓ Simulation parameters configured (nonlinear mode)")
                print(f"  Frequency points: {len(freqs_Hz)}")
                print(f"  Frequency range: {freqs_Hz[0]/1e9:.1f} - {freqs_Hz[-1]/1e9:.1f} GHz")

                # Display source configuration
                sources = config.get_sources()
                print(f"\n  Source configuration:")
                for source in sources:
                    mode_type = "DC" if source['mode'] == (0,) else f"Pump @ {config.pump_freq_GHz} GHz"
                    print(f"    {mode_type}: {source['current']*1e6:.1f} μA on port {source['port']}")

                if config.enable_three_wave_mixing or config.enable_four_wave_mixing:
                    print(f"  Nonlinear mixing enabled")

            else:  # linear mode
                print("✓ Simulation parameters configured (linear mode)")
                print(f"  Frequency points: {len(freqs_Hz)}")
                print(f"  Frequency range: {freqs_Hz[0]/1e9:.1f} - {freqs_Hz[-1]/1e9:.1f} GHz")
                print(f"  Mode: Linear S-parameter analysis (no pump)")

            # Display dielectric loss info (applies to both modes)
            if self.netlist_has_loss:
                print(f"  Dielectric loss: tan δ = {self.netlist_loss_tangent}")

            print("\nRunning harmonic balance solver...")
            print("⏱️  This may take several minutes for large circuits...")

            # Clear any previous warnings
            self.jl.eval("JosephsonCircuits.clear_warning_log()")

            # Build and run the appropriate solver command
            if config.solver_mode == "nonlinear":
                solver_cmd = build_hbsolve_string(config)
            else:
                solver_cmd = build_hblinsolve_string(config)

            # Print start time
            import time
            start_time = time.time()
            print(f"Started at {time.strftime('%H:%M:%S')}")

            # Run with Julia's @time macro - this will print timing info
            self.jl.eval(f'@time {solver_cmd}')
            
            # Print total elapsed time
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"✓ Total time: {mins:02d}:{secs:02d}")

            # Check for warnings
            warnings = self.jl.eval("JosephsonCircuits.get_warning_log()")
            if warnings:
                print("\n⚠️  Solver Warnings:")
                for warning in warnings:
                    print(f"   {warning}")
                print()

            print("✓ Harmonic balance simulation completed!")
            print("Extracting results...")
            
            # Extract results
            results = self._extract_results(config)
            
            print("✓ Results extracted successfully!")
            
            # Calculate and display key metrics
            self._display_quick_results(results, config, amp_type, gain_param)
            
            return results
            
        except Exception as e:
            print(f"✗ Simulation failed: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check circuit parameters are reasonable")
            print("2. Verify port numbers exist in your circuit")
            print("3. For DC bias, ensure the circuit supports it")
            print("4. For dielectric loss, check that capacitor values remain physical")
            print("5. Check Julia console for detailed error messages")
            raise RuntimeError(f"Simulation failed: {e}")

    def _extract_results(self, config: TWPASimulationConfig) -> TWPAResults:
        """Extract simulation results from Julia."""
        assert self.jl is not None, "Julia not initialized"

        # Extract frequency array
        frequencies_GHz = np.array(self.jl.eval('ws./(2*pi*1e9)'))

        # Determine result path based on solver mode
        # hbsolve returns HB(nonlinear, linearized) -> access via sol.linearized
        # hblinsolve returns LinearizedHB directly -> access via sol
        result_path = "sol" if config.solver_mode == "linear" else "sol.linearized"

        # Helper function to extract S-parameters
        def extract_s_param(out_port, out_mode, in_port, in_mode):
            assert self.jl is not None, "Failed to get Julia instance"
            return np.array(self.jl.eval(f'''
            abs2.({result_path}.S(
                outputmode={out_mode},
                outputport={out_port},
                inputmode={in_mode},
                inputport={in_port},
                freqindex=:))
            '''))
        
        print("  Extracting S-parameters...")
        S21 = extract_s_param(config.output_port, (0,), config.signal_port, (0,))
        S12 = extract_s_param(config.signal_port, (0,), config.output_port, (0,))
        S11 = extract_s_param(config.signal_port, (0,), config.signal_port, (0,))
        S22 = extract_s_param(config.output_port, (0,), config.output_port, (0,))

        # In linear mode, skip QE/CM/idler extraction (not computed)
        if config.solver_mode == "linear" and config.Nmodulationharmonics == 0:
            print("  Linear mode: Using placeholder values for QE, CM, and idlers")
            QE = np.ones_like(S21)  # Placeholder
            CM_error = np.zeros_like(S21)  # Placeholder
            idler = np.zeros_like(S21)
            backward_idler = None
            modes = [(0,)]  # Only fundamental mode
        else:
            # Quantum efficiency
            print("  Extracting quantum efficiency...")
            try:
                QE = np.array(self.jl.eval(f'''
                {result_path}.QE((0,),{config.output_port},(0,),{config.signal_port},:)./
                {result_path}.QEideal((0,),{config.output_port},(0,),{config.signal_port},:)
                '''))
            except:
                print("    Warning: Could not extract QE")
                QE = np.ones_like(S21)

            # Commutation relation error
            print("  Extracting commutation relation error...")
            try:
                CM_error = np.array(self.jl.eval(f'1 .- {result_path}.CM((0,),{config.output_port},:)'))
            except:
                print("    Warning: Could not extract CM")
                CM_error = np.zeros_like(S21)

            # Idler response
            print("  Extracting idler response...")
            try:
                idler = np.array(self.jl.eval(f"abs2.({result_path}.S(:,{config.output_port},(0,),{config.signal_port},:)')"))
            except:
                print("    Warning: Could not extract full idler response, using simplified version")
                idler = np.zeros_like(S21)

            # Backward idler response
            print("  Extracting backward idler response...")
            try:
                backward_idler = np.array(self.jl.eval(f"abs2.({result_path}.S(:,{config.signal_port},(0,),{config.output_port},:)')"))
            except:
                print("    Warning: Could not extract backward idler response")
                backward_idler = None

            # Get modes
            try:
                modes = list(self.jl.eval(f"{result_path}.modes"))
            except:
                modes = None

        # Helper: get node sort index from Julia keyed array or from netlist
        def _get_node_sort_idx(jl_nodeflux_expr, num_nodes):
            """Get sort index to reorder nodes from alphabetical to spatial order.

            With sorting="name" in the solver, nodes are alphabetically ordered
            (e.g. "1","10","100",...,"2","20",...). We need numerical order for
            spatial plots. Tries Julia axiskeys first, falls back to netlist.
            """
            # Try getting node names from Julia keyed array
            try:
                node_names = list(self.jl.eval(
                    f'collect(string.(JosephsonCircuits.AxisKeys.axiskeys({jl_nodeflux_expr}, 2)))'
                ))
                sort_idx = np.argsort([int(n) for n in node_names])
                print(f"    Node order from keyed array: reordering {num_nodes} nodes to spatial order")
                return sort_idx
            except Exception as e1:
                print(f"    Note: axiskeys not available ({e1}), trying netlist fallback...")

            # Fallback: extract node names from netlist components
            try:
                all_nodes = set()
                for _, node1, node2, _ in self.jc_components:
                    if node1 != '0':
                        all_nodes.add(node1)
                    if node2 != '0':
                        all_nodes.add(node2)
                # The keyed array has nodes sorted by sorting="name" (alphabetical)
                nodes_alpha = sorted(all_nodes)  # alphabetical, same as Julia
                if len(nodes_alpha) == num_nodes:
                    sort_idx = np.argsort([int(n) for n in nodes_alpha])
                    print(f"    Node order from netlist: reordering {num_nodes} nodes to spatial order")
                    return sort_idx
                else:
                    print(f"    Warning: netlist has {len(nodes_alpha)} nodes vs {num_nodes} in nodeflux")
            except Exception as e2:
                print(f"    Warning: Could not determine node order: {e2}")

            return None

        # Extract pump nodeflux (always available for nonlinear solver)
        pump_nodeflux = None
        num_pump_harmonics = None
        num_nodes = None
        pump_freq_Hz = None
        node_sort_idx = None
        if config.solver_mode == "nonlinear":
            print("  Extracting pump nodeflux...")
            try:
                pump_nodeflux = np.array(self.jl.eval('sol.nonlinear.nodeflux'))
                num_nodes = pump_nodeflux.shape[1] if pump_nodeflux.ndim > 1 else 1
                num_pump_harmonics = pump_nodeflux.shape[0] if pump_nodeflux.ndim > 1 else 1
                pump_freq_Hz = config.pump_freq_GHz * 1e9
                print(f"    Found {num_pump_harmonics} pump harmonics across {num_nodes} nodes")

                # Reorder nodes to spatial order
                node_sort_idx = _get_node_sort_idx('sol.nonlinear.nodeflux', num_nodes)
                if node_sort_idx is not None:
                    pump_nodeflux = pump_nodeflux[:, node_sort_idx]
            except Exception as e:
                print(f"    Warning: Could not extract pump nodeflux: {e}")

        # Extract signal nodeflux (only if store_signal_nodeflux=True)
        signal_nodeflux = None
        if config.store_signal_nodeflux:
            print("  Extracting signal nodeflux...")
            try:
                signal_nodeflux = np.array(self.jl.eval(f'{result_path}.nodeflux'))
                print(f"    Signal nodeflux shape: {signal_nodeflux.shape}")

                # Reorder node dimension (dim 1) to spatial order
                if signal_nodeflux.ndim == 5:
                    num_nodes_sig = signal_nodeflux.shape[1]
                    sig_sort_idx = node_sort_idx  # reuse if same size
                    if sig_sort_idx is None or len(sig_sort_idx) != num_nodes_sig:
                        sig_sort_idx = _get_node_sort_idx(
                            f'{result_path}.nodeflux', num_nodes_sig
                        )
                    if sig_sort_idx is not None:
                        signal_nodeflux = signal_nodeflux[:, sig_sort_idx, :, :, :]
            except Exception as e:
                print(f"    Warning: Could not extract signal nodeflux: {e}")

        # Get total_cells from netlist metadata
        total_cells = self.metadata.get('total_cells', num_nodes)

        return TWPAResults(
            frequencies_GHz=frequencies_GHz,
            S11=S11, S12=S12, S21=S21, S22=S22,
            quantum_efficiency=QE,
            commutation_error=CM_error,
            idler_response=idler,
            backward_idler_response=backward_idler,
            modes=modes,
            netlist_name=self.current_netlist_name,
            config=config,
            pump_nodeflux=pump_nodeflux,
            num_pump_harmonics=num_pump_harmonics,
            pump_freq_Hz=pump_freq_Hz,
            signal_nodeflux=signal_nodeflux,
            num_nodes=num_nodes,
            total_cells=total_cells
        )

    def _display_quick_results(self, results: TWPAResults, config: TWPASimulationConfig, 
                            amp_type: str, gain_param: str):
        """Display quick summary of results."""
        # Determine correct gain data
        if config.signal_port == config.output_port:  # Reflection amp
            gain_data = results.S11 if config.signal_port == 1 else results.S22
        else:  # Transmission amp
            if config.signal_port == 1 and config.output_port == 2:
                gain_data = results.S21
            else:
                gain_data = results.S12
        
        max_gain_dB = 10*np.log10(np.max(gain_data))
        max_gain_idx = np.argmax(gain_data)
        peak_freq = results.frequencies_GHz[max_gain_idx]
        
        # Find 3dB bandwidth
        gain_dB = 10*np.log10(gain_data)
        gain_3db = max_gain_dB - 3
        above_3db = gain_dB > gain_3db
        bandwidth_GHz = 0
        if np.any(above_3db):
            freq_3db = results.frequencies_GHz[above_3db]
            bandwidth_GHz = freq_3db[-1] - freq_3db[0]
        
        print(f"\n🎯 Quick Results Preview ({amp_type}):")
        print(f"   Max {gain_param} gain: {max_gain_dB:.1f} dB @ {peak_freq:.3f} GHz")
        print(f"   3dB bandwidth: {bandwidth_GHz:.2f} GHz")
        print(f"   Max quantum efficiency: {np.max(results.quantum_efficiency):.3f}")
        print(f"   Min commutation error: {np.min(np.abs(results.commutation_error)):.2e}")   

    def run_full_simulation(self, netlist_name: str, config: TWPASimulationConfig,
                           verbose: bool = True, force_julia_reinit: bool = False,
                           save_results: bool = True, show_plot: bool = True,
                           max_mode_order_to_plot: int = 2,
                           output_dir: Optional[str] = None) -> TWPAResults:
        """
        Complete TWPA simulation workflow in one method.

        This method combines setup_julia(), load_netlist(), build_circuit(),
        run_simulation(), and optionally saves both data and plots with comprehensive
        error handling and optional verbose output.

        Args:
            netlist_name: Name of netlist file (without .py extension)
            config: Simulation configuration
            verbose: If True, print detailed progress information. If False, only show errors and final results
            force_julia_reinit: If True, force Julia reinitialization
            save_results: If True, automatically save both data (.npz) and plot (.svg) files
            show_plot: If True, display the plot (default: True)
            max_mode_order_to_plot: Maximum mode order to plot for idlers (default: 2)
            output_dir: Output directory for saved files. If None, uses package's results/ folder

        Returns:
            TWPAResults object containing simulation results

        Raises:
            RuntimeError: If any step fails
            FileNotFoundError: If netlist file not found
            ValueError: If configuration is invalid
        """
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"    TWPA Full Simulation Workflow")
            print(f"    Netlist: {netlist_name}")
            print(f"{'='*60}")
        
        try:
            # Step 1: Setup Julia
            if verbose:
                print(f"\n[1/4] Setting up Julia...")
            
            if not self.julia_ready or force_julia_reinit:
                self.setup_julia(force_reinit=force_julia_reinit)
                if verbose:
                    print(f"✅ Julia setup complete")
            else:
                if verbose:
                    print(f"✅ Julia already ready")
            
            # Step 2: Load Netlist
            if verbose:
                print(f"\n[2/4] Loading netlist: {netlist_name}")
            
            self.load_netlist(netlist_name)
            
            if verbose:
                print(f"✅ Netlist loaded successfully")
                print(f"   Components: {self.component_count}")
                print(f"   Parameters: {len(self.circuit_parameters)}")
                if self.netlist_has_loss:
                    print(f"   Dielectric loss: tan δ = {self.netlist_loss_tangent}")
                else:
                    print(f"   No dielectric loss")
            
            # Step 3: Build Circuit
            if verbose:
                print(f"\n[3/4] Building circuit in Julia...")
            
            self.build_circuit()
            
            if verbose:
                print(f"✅ Circuit built successfully")
                # Count nonlinear elements
                nl_count = sum(1 for name, _, _, _ in self.jc_components 
                              if name.startswith("NL") and self._should_include_component(name, "dummy"))
                if nl_count > 0:
                    print(f"   Including {nl_count} nonlinear elements")
            
            # Step 4: Run Simulation
            if verbose:
                print(f"\n[4/4] Running harmonic balance simulation...")
                print(f"⏱️  This may take several minutes for large circuits...")
                config.print_config()
                print()
            
            results = self.run_simulation(config)
            
            # Success summary
            max_gain = np.max(results.S21)
            max_gain_dB = 10*np.log10(max_gain)
            
            if verbose:
                print(f"\n🎯 Simulation completed successfully!")
            else:
                print(f"✅ Simulation complete: Max S21 gain = {max_gain_dB:.1f} dB")
            
            # Save results if requested
            if save_results:
                if verbose:
                    print(f"\n💾 Saving results...")

                # Save data (.npz file)
                data_filename = results.save(config=config, output_dir=output_dir)
                if verbose:
                    print(f"📊 Data saved to: {data_filename}")

                # Save plot (.svg file) and optionally display it
                results.plot(config=config, netlist_name=netlist_name, auto_save=True, show_plot=show_plot, max_mode_order_to_plot=max_mode_order_to_plot, output_dir=output_dir)
                if verbose:
                    if show_plot:
                        print(f"📈 Plot saved and displayed")
                    else:
                        print(f"📈 Plot saved (not displayed)")
                else:
                    print(f"💾 Results and plot saved successfully")
            elif show_plot:
                # Show plot without saving
                results.plot(config=config, netlist_name=netlist_name, auto_save=False, show_plot=True, max_mode_order_to_plot=max_mode_order_to_plot)
                if verbose:
                    print(f"📈 Plot displayed (not saved)")
            
            return results
            
        except Exception as e:
            error_msg = f"❌ Full simulation failed at step: {str(e)}"
            if verbose:
                print(f"\n{error_msg}")
                print(f"\nTroubleshooting tips:")
                print(f"1. Check that netlist file exists in netlists directory")
                print(f"2. Verify simulation parameters are reasonable")
                print(f"3. Check Julia installation and package dependencies")
                print(f"4. Try force_julia_reinit=True if Julia seems corrupted")
                print(f"5. Check available memory for large circuits")
            else:
                print(error_msg)
            raise RuntimeError(error_msg) from e

    
