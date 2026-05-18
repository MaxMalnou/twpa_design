# twpa_design/netlist.py
"""Netlist builder for JosephsonCircuits.jl format.

This module contains the JCNetlistBuilder class that creates netlists
compatible with Kevin O'Brien's JosephsonCircuits.jl package.

NOTE: Incomplete implementations
---------------------------------
The following features are not yet implemented:
1. Traditional SNAIL structures (line 540) - raises NotImplementedError
2. Traditional DC-SQUID structures (line 571) - raises NotImplementedError

Currently supported structures:
- Josephson Junctions (JJ)
- RF-SQUIDs
- Kinetic Inductance (KI) elements
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import re
import importlib.util
import os
from datetime import datetime
from pathlib import Path

from .helper_functions import check_flat
from . import DESIGNS_DIR, NETLISTS_DIR

# ============= NETLIST CONFIGURATION =============
@dataclass
class NetlistConfig:
    """Configuration for netlist building.

    This handles the build settings for netlist generation:
    - Which design file to load
    - Taylor expansion settings
    - Loss settings
    - Override options
    - Output directory
    """
    # Input file from notebook 1 (ATL_TWPA_lite)
    design_file: str = "b_jtwpa_01.py"  # File from designs/ folder

    # Build configuration
    use_taylor_insteadof_JJ: bool = False
    enable_dielectric_loss: bool = False
    loss_tangent: float = 2e-4
    use_linear_in_window: bool = True

    # Override for total cells if needed
    Ntot_cell_override: Optional[int] = None

    # Output directory (None uses package's netlists/ folder)
    output_dir: Optional[str] = None

    @property
    def design_path(self):
        """Full path to design file."""
        return str(DESIGNS_DIR / self.design_file)
    
    def check_input_file(self):
        """Check if design file exists."""
        if not Path(self.design_path).exists():
            raise FileNotFoundError(f"Design file not found: {self.design_path}")
        
# ============= COMPONENT DEFINITIONS =============

@dataclass
class JCComponent:
    """Represents a component in JC format"""
    name: str
    node1: str
    node2: str
    value: str  # Can be numeric or symbolic
    comp_type: str = ""  # 'R', 'L', 'C', 'Lj', 'NL', 'P'
    
    def __post_init__(self):
        """Auto-detect component type if not provided"""
        if not self.comp_type:
            if self.name.startswith('Lj'):
                self.comp_type = 'Lj'
            elif self.name.startswith('NL'):
                self.comp_type = 'NL'
            elif self.name.startswith('P'):
                self.comp_type = 'P'
            else:
                self.comp_type = self.name[0].upper()

class JCNetlistBuilder:
    """Builder class for JosephsonCircuits netlists"""
    
    def __init__(self):
        # Component storage
        self.components: List[JCComponent] = []
        self.circuit_parameters: Dict[str, float] = {}
        
        # Node management
        self.node_counter: int = 1
        
        # Component naming
        self.used_names: set = set()
        
        # Standard parameters
        self.circuit_parameters['R_port'] = 50.0
        self.PHI0 = 2.067833848e-15  # Flux quantum in Wb
        
        # JJ parameters (will be set from workspace)
        self.Lj_value: Optional[float] = None
        self.Cj_value: Optional[float] = None
        
        # Workspace data storage (for access by methods)
        self.workspace_data: Dict[str, Any] = {}

        # When True, all component values are written as inline numeric instead of symbolic
        self.force_numeric: bool = False

        # Set by _taper_filter while building a Floquet-taper filter cell, consumed by
        # add_inductance to build a per-cell numeric Taylor poly for KI (and per-cell
        # numeric Lj for JJ) instead of using the shared symbolic L0/c1/c2.
        self._taper_cell_idx: Optional[int] = None
    
    def get_new_node(self) -> str:
        """Get a new sequential node number"""
        node = str(self.node_counter)
        self.node_counter += 1
        return node
    
    def make_unique_name(self, base_name: str) -> str:
        """Ensure component name is unique by adding suffix if needed"""
        if base_name not in self.used_names:
            self.used_names.add(base_name)
            return base_name
        
        counter = 1
        while f"{base_name}_{counter}" in self.used_names:
            counter += 1
        
        unique_name = f"{base_name}_{counter}"
        self.used_names.add(unique_name)
        return unique_name
    
    def add_component(self, name: str, node1: str, node2: str, value: str):
        """Add a component to the netlist"""
        unique_name = self.make_unique_name(name)
        component = JCComponent(unique_name, str(node1), str(node2), str(value))
        self.components.append(component)
        return component
    
    def set_jj_parameters(self, Ic_JJ_uA: Optional[float], CJ_F: Optional[float]):
        """Set Josephson junction parameters from critical current and capacitance"""
        if Ic_JJ_uA is not None:
            Ic_A = Ic_JJ_uA * 1e-6
            self.Lj_value = self.PHI0 / (2 * np.pi * Ic_A)
            self.circuit_parameters['Lj'] = self.Lj_value
            
        if CJ_F is not None and CJ_F > 0 and not np.isinf(CJ_F):
            self.Cj_value = CJ_F
            self.circuit_parameters['Cj'] = self.Cj_value
    
    def load_workspace_data(self, workspace_vars: Dict[str, Any]):
        """Load workspace variables for use in building"""
        self.workspace_data = workspace_vars.copy()
    
    def reset(self):
        """Reset the builder to initial state"""
        self.components = []
        self.circuit_parameters = {'R_port': 50.0}
        self.node_counter = 1
        self.used_names = set()
        self.Lj_value = None
        self.Cj_value = None
        self.workspace_data = {}
        self.force_numeric = False
    
    def get_netlist_tuples(self) -> List[Tuple[str, str, str, str]]:
        """Convert components to tuple format for output"""
        return [(c.name, c.node1, c.node2, c.value) for c in self.components]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the netlist"""
        stats = {
            'total_components': len(self.components),
            'total_parameters': len(self.circuit_parameters),
            'total_nodes': self.node_counter - 1,
        }
        
        # Count component types
        for comp in self.components:
            comp_type = f'num_{comp.comp_type}'
            stats[comp_type] = stats.get(comp_type, 0) + 1
        
        return stats
    
    @staticmethod
    def parse_spice_value(value_str):
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
    
    def create_symbolic_value(self, numeric_value, component_type, component_name,
                            Ncpersc_cell=None, ind_g_C_with_filters=None, dispersion_type=None):
        """Create symbolic parameter and store value.
        When self.force_numeric is True, returns inline numeric string instead."""
        if self.force_numeric:
            # Dielectric loss applies to capacitors (junction caps go through a
            # direct circuit_parameters['Cj'] write, never this path). The
            # symbolic-parameters writer applies the same wrapper to C* params
            # at save time; this branch handles inline numeric caps in the
            # taper region uniformly.
            enable_loss = self.workspace_data.get('enable_dielectric_loss', False)
            loss_tan = self.workspace_data.get('loss_tangent', 0.0)
            if component_type == 'C' and enable_loss and loss_tan > 0:
                return f"{numeric_value:.6e}/(1+{loss_tan:.6e}im)"
            return f"{numeric_value:.6e}"

        # Ensure component_name is a string
        component_name = str(component_name)

        # Special handling for remainder inductances in filters
        if component_type == 'L' and '_rem' in component_name and any(f in component_name for f in ['LF', 'CF']):
            import re
            # Extract the filter type and create appropriate remainder name
            match = re.search(r'L(0|inf)(LF|CF)([12]).*_rem', component_name)
            if match:
                symbol = f'L{match.group(1)}{match.group(2)}{match.group(3)}_rem'
                if symbol not in self.circuit_parameters:
                    self.circuit_parameters[symbol] = numeric_value
                return symbol
        
        # For modulated capacitors, use modulo to reference supercell pattern
        if component_type == 'C' and 'TLsec' in component_name and Ncpersc_cell is not None:
            # Extract cell number from component name
            import re
            match = re.search(r'TLsec_(\d+)', component_name)
            if match:
                cell_num = int(match.group(1))
                # Use modulo to get position within supercell
                cell_in_supercell = cell_num % Ncpersc_cell
                
                # First, check if we already have a parameter with this exact value
                tolerance = 1e-20  # Tight tolerance for capacitors
                for param, val in self.circuit_parameters.items():
                    if param.startswith('C_TL') and isinstance(val, float) and abs(val - numeric_value) < tolerance:
                        # Found existing parameter with same value - reuse it
                        return param
                
                # No existing parameter with this value - create new one with supercell index
                symbol = f"C_TL_{cell_in_supercell}"
                
                # Only create parameter if it doesn't exist
                if symbol not in self.circuit_parameters:
                    self.circuit_parameters[symbol] = numeric_value
                elif abs(self.circuit_parameters[symbol] - numeric_value) > tolerance:
                    # This symbol exists but with a different value
                    # This means we have multiple different values mapping to the same supercell position
                    # Need to create a unique symbol
                    counter = 0
                    while f"{symbol}_{counter}" in self.circuit_parameters:
                        counter += 1
                    symbol = f"{symbol}_{counter}"
                    self.circuit_parameters[symbol] = numeric_value
                    
                return symbol
        
        # For filter components, use their specific names
        # BUT skip this for Lg (geometric inductance) components
        if any(filter_type in component_name for filter_type in ['LF1', 'LF2', 'CF1', 'CF2']) and not component_name.startswith('Lg'):
            # Skip this special handling for Lg components
            if component_type == 'L' and component_name.startswith('Lg'):
                # Fall through to regular inductance handling
                pass
            else:
                import re
            
            # For 'both' dispersion type, map filter position to supercell pattern
            if dispersion_type == 'both' and Ncpersc_cell is not None and ind_g_C_with_filters is not None:
                # Extract the cell index from component name (e.g., "0CF1_8" -> 8)
                cell_idx_match = re.search(r'_(\d+)$', component_name)
                if cell_idx_match:
                    cell_idx = int(cell_idx_match.group(1))
                    
                    # Find which filter in the pattern this corresponds to
                    # ind_g_C_with_filters contains the positions of filters in first supercell
                    filter_positions_in_sc = [pos % Ncpersc_cell for pos in ind_g_C_with_filters]
                    cell_pos_in_sc = cell_idx % Ncpersc_cell
                    
                    if cell_pos_in_sc in filter_positions_in_sc:
                        # This is a filter position - find which one (0, 1, etc.)
                        filter_num = filter_positions_in_sc.index(cell_pos_in_sc)
                    else:
                        filter_num = None
                        
                    # Remove the cell index for pattern matching
                    component_name_base = re.sub(r'_\d+$', '', component_name)
                else:
                    filter_num = None
                    component_name_base = component_name
            else:
                filter_num = None
                component_name_base = component_name
            
            # Updated patterns to match component names WITHOUT the cell index
            patterns = [
                # Inductors (these have L prefix)
                (r'L(inf|0)LF([12])', lambda m: f'L{m.group(1)}LF{m.group(2)}_{filter_num}' if filter_num is not None else f'L{m.group(1)}LF{m.group(2)}'),
                (r'L(inf|0)CF([12])', lambda m: f'L{m.group(1)}CF{m.group(2)}_{filter_num}' if filter_num is not None else f'L{m.group(1)}CF{m.group(2)}'),
                (r'LiLF([12])_(\d+)', lambda m: f'LiLF{m.group(1)}_{m.group(2)}_{filter_num}' if filter_num is not None else f'LiLF{m.group(1)}_{m.group(2)}'),
                (r'LiCF([12])_(\d+)', lambda m: f'LiCF{m.group(1)}_{m.group(2)}_{filter_num}' if filter_num is not None else f'LiCF{m.group(1)}_{m.group(2)}'),
                # Capacitors (these DON'T have C prefix in the name passed)
                (r'(inf|0)LF([12])', lambda m: f'C{m.group(1)}LF{m.group(2)}_{filter_num}' if filter_num is not None else f'C{m.group(1)}LF{m.group(2)}'),
                (r'(inf|0)CF([12])', lambda m: f'C{m.group(1)}CF{m.group(2)}_{filter_num}' if filter_num is not None else f'C{m.group(1)}CF{m.group(2)}'),
                (r'iLF([12])_(\d+)', lambda m: f'CiLF{m.group(1)}_{m.group(2)}_{filter_num}' if filter_num is not None else f'CiLF{m.group(1)}_{m.group(2)}'),
                (r'iCF([12])_(\d+)', lambda m: f'CiCF{m.group(1)}_{m.group(2)}_{filter_num}' if filter_num is not None else f'CiCF{m.group(1)}_{m.group(2)}'),
            ]
            
            # Match against the base name (without cell index)
            for pattern, formatter in patterns:
                match = re.search(pattern, component_name_base)
                if match:
                    symbol = formatter(match)                
                    # Only create parameter if it doesn't exist
                    if symbol not in self.circuit_parameters:
                        self.circuit_parameters[symbol] = numeric_value
                    return symbol
        
        # For NL components in filters, also use special naming
        if component_type == 'L' and 'NL' in component_name and any(f in component_name for f in ['LF', 'CF']):
            import re
            # Extract the filter type
            match = re.search(r'NL(0|inf)(LF|CF)([12])', component_name)
            if match:
                symbol = f'L{match.group(1)}{match.group(2)}{match.group(3)}'
                if symbol not in self.circuit_parameters:
                    self.circuit_parameters[symbol] = numeric_value
                return symbol
        
        # For other components, use the existing logic
        # Extract base name for the parameter
        base_name = component_name.split('_')[0] if '_' in component_name else component_name
        
        # Check if we already have this exact value (with tighter tolerance for capacitors)
        tolerance = 1e-20 if component_type == 'C' else 1e-18
        for param, val in self.circuit_parameters.items():
            if isinstance(val, float) and abs(val - numeric_value) < tolerance:
                return param
        
        # Create new symbolic name
        if component_type == 'L':
            if 'g' in base_name.lower():
                symbol = f"Lg_{len([k for k in self.circuit_parameters if k.startswith('Lg_')])}"
            else:
                symbol = f"L_{len([k for k in self.circuit_parameters if k.startswith('L_')])}"
        elif component_type == 'C':
            if 'g' in base_name.lower():
                symbol = f"Cg_{len([k for k in self.circuit_parameters if k.startswith('Cg_')])}"
            else:
                symbol = f"C_{len([k for k in self.circuit_parameters if k.startswith('C_')])}"
        elif component_type == 'R':
            symbol = "R_port" if abs(numeric_value - 50.0) < 1e-6 else f"R_{len([k for k in self.circuit_parameters if k.startswith('R_')])}"
        else:
            symbol = f"{component_type}_{len([k for k in self.circuit_parameters if k.startswith(f'{component_type}_')])}"
        
        self.circuit_parameters[symbol] = numeric_value
        return symbol
    
    def get_used_parameters(self) -> Dict[str, float]:
        """
        Extract only the parameters that are actually referenced in the netlist.
        This ensures we only include parameters that are used.
        """
        used_params = {}
        
        # First, collect all symbolic parameter names used in components
        for comp in self.components:
            value = comp.value
            
            # Skip numeric values and special component types
            if value in ['Lj', 'Cj'] or value.replace('.', '').replace('e', '').replace('-', '').replace('+', '').isdigit():
                continue
            
            # Skip port numbers
            if comp.comp_type == 'P':
                continue
                
            # Handle polynomial expressions
            if value.startswith('poly '):
                # Extract parameter names from polynomial expression
                # Remove 'poly' and split by comma, then strip whitespace
                poly_content = value[5:]  # Remove 'poly '
                parts = [p.strip() for p in poly_content.split(',')]
                for part in parts:
                    # Add each parameter that's actually in the poly string
                    if part in self.circuit_parameters:
                        used_params[part] = self.circuit_parameters[part]
            else:
                # Regular symbolic parameter
                if value in self.circuit_parameters:
                    used_params[value] = self.circuit_parameters[value]
        
        # Special handling for Lj and Cj - only include if components use them
        if any(comp.value == 'Lj' for comp in self.components) and self.Lj_value is not None:
            used_params['Lj'] = self.Lj_value
            
        if any(comp.value == 'Cj' for comp in self.components) and self.Cj_value is not None:
            used_params['Cj'] = self.Cj_value
        
        return used_params
    
    def get_taylor_poly_string(self, base_param='L0'):
        """
        Build polynomial string based on available Taylor coefficients.
        Only includes coefficients that are actually defined.
        """
        poly_parts = [base_param]  # Start with base parameter

        # Check which coefficients exist and add them
        for coeff in ['c1', 'c2', 'c3', 'c4']:
            taylor_val = self.workspace_data.get(f'{coeff}_taylor')
            if taylor_val is not None:
                poly_parts.append(coeff)
                # Add to circuit parameters if not already there
                if coeff not in self.circuit_parameters:
                    self.circuit_parameters[coeff] = taylor_val
            else:
                # Stop at first missing coefficient (can't skip coefficients in polynomial)
                break

        # Join with commas, then prepend 'poly '
        return 'poly ' + ', '.join(poly_parts)

    def _build_per_cell_taylor_str(self, cell_idx):
        """Build a Taylor poly string with per-cell numeric values for cell cell_idx.

        Mirrors the format used by `add_inductance_floquet`: pulls `L0_H` and
        `cN_taylor` arrays from `workspace_data['floquet_cell_params']` and
        formats them as `poly L0_val, c1_val, c2_val, ...`. Used for KI in
        Floquet-taper filter cells (where the call chain reaches `add_inductance`
        instead of `add_inductance_floquet`, but per-cell numeric values are still
        required so each cell's Z(n)-dependent L0 and Taylor coefficients land in
        the netlist correctly).
        """
        floquet_params = self.workspace_data.get('floquet_cell_params', {})
        L0_arr = floquet_params.get('L0_H')
        if L0_arr is None:
            # Fall back to the workspace's center scalar if no per-cell array is available.
            L0_cell = float(self.workspace_data.get('L0_H', 0.0))
        else:
            L0_cell = float(L0_arr[cell_idx])

        parts = [f"{L0_cell:.6e}"]
        for coeff in ['c1_taylor', 'c2_taylor', 'c3_taylor', 'c4_taylor']:
            c_arr = floquet_params.get(coeff)
            if c_arr is None:
                # Try the workspace scalar fallback before stopping.
                scalar = self.workspace_data.get(coeff)
                if scalar is None:
                    break
                parts.append(f"{float(scalar):.6e}")
            elif hasattr(c_arr, '__len__'):
                parts.append(f"{float(c_arr[cell_idx]):.6e}")
            else:
                parts.append(f"{float(c_arr):.6e}")
        return 'poly ' + ', '.join(parts)
    
    def add_bare_jj(self, component_name, node1, node2, n_jj_struct, use_taylor=False,
                    Lj_numeric=None, Cj_numeric=None):
        """
        Add bare Josephson junction(s) or Taylor equivalent (no geometric inductance).
        Each structure includes its CJ capacitor.

        Parameters:
        - component_name: Base name for components
        - node1, node2: Circuit nodes
        - n_jj_struct: Number of JJs in series
        - use_taylor: Use Taylor expansion if True
        - Lj_numeric: If provided, use this numeric value instead of symbolic 'Lj'
        - Cj_numeric: If provided, use this numeric value instead of symbolic 'Cj'
        """
        lj_val = f"{Lj_numeric:.6e}" if Lj_numeric is not None else 'Lj'
        cj_val = f"{Cj_numeric:.6e}" if Cj_numeric is not None else 'Cj'
        has_cj = (Cj_numeric is not None and Cj_numeric > 0) if Cj_numeric is not None else (self.Cj_value is not None and self.Cj_value > 0)

        if use_taylor:
            L0_H = self.workspace_data.get('L0_H')
            if L0_H is None:
                raise ValueError("L0_H must be defined for Taylor expansion")

            if 'L0' not in self.circuit_parameters:
                self.circuit_parameters['L0'] = L0_H

            taylor_str = self.get_taylor_poly_string()

            if n_jj_struct == 1:
                nl_name = self.make_unique_name(f"NL{component_name}")
                self.components.append(JCComponent(nl_name, str(node1), str(node2), taylor_str))

                if has_cj:
                    cj_name = self.make_unique_name(f"Cj{component_name}")
                    self.components.append(JCComponent(cj_name, str(node1), str(node2), cj_val))
            else:
                current_node = node1
                for i in range(n_jj_struct):
                    next_node = self.get_new_node() if i < n_jj_struct - 1 else node2

                    nl_name = self.make_unique_name(f"NL{component_name}_{i+1}")
                    self.components.append(JCComponent(nl_name, str(current_node), str(next_node), taylor_str))

                    if has_cj:
                        cj_name = self.make_unique_name(f"Cj{component_name}_{i+1}")
                        self.components.append(JCComponent(cj_name, str(current_node), str(next_node), cj_val))

                    current_node = next_node
        else:
            # Traditional bare JJ(s)
            if n_jj_struct == 1:
                lj_name = self.make_unique_name(f"Lj{component_name}")
                self.components.append(JCComponent(lj_name, str(node1), str(node2), lj_val))
                
                if has_cj:
                    cj_name = self.make_unique_name(f"Cj{component_name}")
                    self.components.append(JCComponent(cj_name, str(node1), str(node2), cj_val))
            else:
                current_node = node1
                for i in range(n_jj_struct):
                    next_node = self.get_new_node() if i < n_jj_struct - 1 else node2

                    lj_name = self.make_unique_name(f"Lj{component_name}_{i+1}")
                    self.components.append(JCComponent(lj_name, str(current_node), str(next_node), lj_val))

                    if has_cj:
                        cj_name = self.make_unique_name(f"Cj{component_name}_{i+1}")
                        self.components.append(JCComponent(cj_name, str(current_node), str(next_node), cj_val))

                    current_node = next_node

    
    def add_rf_squid(self, component_name, node1, node2, n_jj_struct, Lg_H, use_taylor=False,
                    Lj_numeric=None, Cj_numeric=None, Cjx_numeric=None):
        """
        Add rf-SQUID(s) (JJ in parallel with geometric inductance) or Taylor equivalent.
        Each rf-SQUID unit includes its own CJ capacitor.

        Parameters:
        - component_name: Base name for the components
        - node1, node2: Circuit nodes
        - n_jj_struct: Number of rf-SQUIDs in series
        - Lg_H: Geometric inductance in Henries (None or np.inf means no Lg)
        - use_taylor: If True, use Taylor expansion instead of JJ
        - Lj_numeric: If provided, use this numeric value instead of symbolic 'Lj'
        - Cj_numeric: If provided, use this numeric value instead of symbolic 'Cj'
        - Cjx_numeric: If provided and > 0, add an extra shunt capacitor in parallel
                       with the rf-SQUID (used for rf_squid_constant_plasma option).
        """
        has_Lg = Lg_H is not None and not np.isinf(Lg_H) and Lg_H > 0
        lj_val = f"{Lj_numeric:.6e}" if Lj_numeric is not None else 'Lj'
        cj_val = f"{Cj_numeric:.6e}" if Cj_numeric is not None else 'Cj'
        has_cj = (Cj_numeric is not None and Cj_numeric > 0) if Cj_numeric is not None else (self.Cj_value is not None and self.Cj_value > 0)
        has_cjx = Cjx_numeric is not None and Cjx_numeric > 0 and not np.isinf(Cjx_numeric)

        if use_taylor:
            L0_H = self.workspace_data.get('L0_H')
            if L0_H is None:
                raise ValueError("L0_H must be defined to use Taylor expansion for rf-SQUID")

            if 'L0' not in self.circuit_parameters:
                self.circuit_parameters['L0'] = L0_H

            taylor_str = self.get_taylor_poly_string()

            if n_jj_struct == 1:
                nl_name = self.make_unique_name(f"NL{component_name}")
                self.components.append(JCComponent(nl_name, str(node1), str(node2), taylor_str))

                if has_cj:
                    cj_name = self.make_unique_name(f"Cj{component_name}")
                    self.components.append(JCComponent(cj_name, str(node1), str(node2), cj_val))
            else:
                current_node = node1
                for i in range(n_jj_struct):
                    next_node = self.get_new_node() if i < n_jj_struct - 1 else node2

                    nl_name = self.make_unique_name(f"NL{component_name}_{i+1}")
                    self.components.append(JCComponent(nl_name, str(current_node), str(next_node), taylor_str))

                    if has_cj:
                        cj_name = self.make_unique_name(f"Cj{component_name}_{i+1}")
                        self.components.append(JCComponent(cj_name, str(current_node), str(next_node), cj_val))

                    current_node = next_node
        else:
            if n_jj_struct == 1:
                lj_name = self.make_unique_name(f"Lj{component_name}")
                self.components.append(JCComponent(lj_name, str(node1), str(node2), lj_val))

                if has_Lg:
                    lg_name = self.make_unique_name(f"Lg{component_name}")
                    lg_symbol = self.create_symbolic_value(Lg_H, 'L', f"Lg{component_name}")
                    self.components.append(JCComponent(lg_name, str(node1), str(node2), lg_symbol))

                if has_cj:
                    cj_name = self.make_unique_name(f"Cj{component_name}")
                    self.components.append(JCComponent(cj_name, str(node1), str(node2), cj_val))

                if has_cjx:
                    cjx_name = self.make_unique_name(f"Cjx{component_name}")
                    self.components.append(JCComponent(cjx_name, str(node1), str(node2), f"{Cjx_numeric:.6e}"))
            else:
                current_node = node1
                for i in range(n_jj_struct):
                    next_node = self.get_new_node() if i < n_jj_struct - 1 else node2

                    lj_name = self.make_unique_name(f"Lj{component_name}_{i+1}")
                    self.components.append(JCComponent(lj_name, str(current_node), str(next_node), lj_val))

                    if has_Lg:
                        lg_name = self.make_unique_name(f"Lg{component_name}_{i+1}")
                        lg_symbol = self.create_symbolic_value(Lg_H, 'L', f"Lg{component_name}_{i+1}")
                        self.components.append(JCComponent(lg_name, str(current_node), str(next_node), lg_symbol))

                    if has_cj:
                        cj_name = self.make_unique_name(f"Cj{component_name}_{i+1}")
                        self.components.append(JCComponent(cj_name, str(current_node), str(next_node), cj_val))

                    if has_cjx:
                        cjx_name = self.make_unique_name(f"Cjx{component_name}_{i+1}")
                        self.components.append(JCComponent(cjx_name, str(current_node), str(next_node), f"{Cjx_numeric:.6e}"))

                    current_node = next_node

    def add_snail(self, component_name, node1, node2, n_large, alpha, use_taylor=False):
        """
        Add a SNAIL element or its Taylor equivalent.
        
        Parameters:
        - n_large: Number of large JJs
        - alpha: Size ratio of small JJ (Ic_small = alpha * Ic_large)
        - use_taylor: Use Taylor expansion if True
        """
        if use_taylor:
            # Use Taylor expansion for the entire SNAIL
            nl_name = self.make_unique_name(f"NL{component_name}")
            
            # Use pre-calculated L0_H (includes bias effects for SNAIL)
            L0_H = self.workspace_data.get('L0_H')
            if L0_H is None:
                raise ValueError("L0_H must be defined for Taylor expansion")
            
            # Use shared parameters - same for all structures
            if 'L0' not in self.circuit_parameters:
                self.circuit_parameters['L0'] = L0_H
            
            # Create the NL element with Taylor expansion (only includes defined coefficients)
            taylor_str = self.get_taylor_poly_string()

            self.components.append(JCComponent(nl_name, str(node1), str(node2), taylor_str))
        else:
            # Traditional SNAIL implementation
            # This would need to be customized based on your SNAIL model
            raise NotImplementedError("Traditional SNAIL not yet implemented")
    
    def add_dc_squid(self, component_name, node1, node2, n_jj, loop_inductance, asymmetry, use_taylor=False):
        """
        Add a DC-SQUID element or its Taylor equivalent.
        
        Parameters:
        - n_jj: Number of JJs (typically 2)
        - loop_inductance: Geometric inductance of the loop
        - asymmetry: JJ asymmetry parameter (0 = symmetric)
        - use_taylor: Use Taylor expansion if True
        """
        if use_taylor:
            # Use Taylor expansion for the entire DC-SQUID
            nl_name = self.make_unique_name(f"NL{component_name}")
            
            # Use pre-calculated L0_H (includes bias and flux effects)
            L0_H = self.workspace_data.get('L0_H')
            if L0_H is None:
                raise ValueError("L0_H must be defined for Taylor expansion")
            
            # Use shared parameters - same for all structures
            if 'L0' not in self.circuit_parameters:
                self.circuit_parameters['L0'] = L0_H
            
            # Create the NL element with Taylor expansion (only includes defined coefficients)
            taylor_str = self.get_taylor_poly_string()

            self.components.append(JCComponent(nl_name, str(node1), str(node2), taylor_str))
        else:
            # Traditional DC-SQUID implementation
            raise NotImplementedError("Traditional DC-SQUID not yet implemented")
        
    def add_jj_structure(self, component_name, node1, node2, structure_params,
                        Lj_numeric=None, Cj_numeric=None, Cjx_numeric=None):
        """
        Dispatcher method for different JJ structures.

        Parameters:
        - component_name: Base name for components
        - node1, node2: Circuit nodes
        - structure_params: Dict with structure-specific parameters including 'type'
        - Lj_numeric: If provided, use this numeric Lj value instead of symbolic 'Lj'
        - Cj_numeric: If provided, use this numeric Cj value instead of symbolic 'Cj'
        - Cjx_numeric: If provided and > 0, add an extra shunt cap (rf_squid only).
        """
        structure_type = structure_params.get('type', 'rf_squid')
        use_taylor = self.workspace_data.get('use_taylor_insteadof_JJ', False)

        if structure_type == 'jj':
            self.add_bare_jj(
                component_name, node1, node2,
                structure_params.get('n_jj_struct', 1),
                use_taylor,
                Lj_numeric=Lj_numeric, Cj_numeric=Cj_numeric
            )
        elif structure_type == 'rf_squid':
            self.add_rf_squid(
                component_name, node1, node2,
                structure_params.get('n_jj_struct', 1),
                structure_params.get('Lg_H'),
                use_taylor,
                Lj_numeric=Lj_numeric, Cj_numeric=Cj_numeric, Cjx_numeric=Cjx_numeric
            )
        elif structure_type == 'snail':
            self.add_snail(
                component_name, node1, node2,
                structure_params.get('n_large', 3),
                structure_params.get('alpha', 0.5),
                use_taylor
            )
        elif structure_type == 'dc_squid':
            self.add_dc_squid(
                component_name, node1, node2,
                structure_params.get('n_jj', 2),
                structure_params.get('loop_inductance'),
                structure_params.get('asymmetry', 0.0),
                use_taylor
            )
        else:
            raise ValueError(f"Unknown JJ structure type: {structure_type}")
    
    def add_inductance(self, component_name, node1, node2, nonlinearity, L_H, L_rem_H,
                      Lg_H, epsilon_perA, xi_perA2, n_jj_struct,
                      Lj_numeric=None, Cj_numeric=None, Cjx_numeric=None):
        """Add inductance to JC netlist (handles JJ, KI, and linear inductances).

        When Lj_numeric/Cj_numeric are provided, numeric values are used instead of
        symbolic 'Lj'/'Cj' references. Used for Floquet taper cells.
        Cjx_numeric: optional extra shunt cap in parallel with the rf_squid
        (used for rf_squid_constant_plasma option).
        """
        # Determine if we need an intermediate node for remainder
        need_remainder = L_rem_H != 0
        final_node = node2
        if need_remainder:
            intermediate_node = self.get_new_node()
            main_output_node = intermediate_node
        else:
            main_output_node = node2
        
        if nonlinearity == 'JJ':
            # Get JJ structure configuration
            jj_structure_type = self.workspace_data.get('jj_structure_type', 'rf_squid')
            
            # Build structure parameters based on type
            structure_params = {
                'type': jj_structure_type,
                'n_jj_struct': n_jj_struct,
                'Lg_H': Lg_H,
            }
            
            # Add structure-specific parameters if they exist
            if jj_structure_type == 'snail':
                structure_params['n_large'] = self.workspace_data.get('snail_n_large', 3)
                structure_params['alpha'] = self.workspace_data.get('snail_alpha', 0.5)
            elif jj_structure_type == 'dc_squid':
                structure_params['n_jj'] = self.workspace_data.get('dc_squid_n_jj', 2)
                structure_params['loop_inductance'] = self.workspace_data.get('dc_squid_loop_L')
                structure_params['asymmetry'] = self.workspace_data.get('dc_squid_asymmetry', 0.0)
            
            # Use the dispatcher
            self.add_jj_structure(component_name, node1, main_output_node, structure_params,
                                 Lj_numeric=Lj_numeric, Cj_numeric=Cj_numeric, Cjx_numeric=Cjx_numeric)
            
            # CJ capacitors are now added within add_bare_jj and add_rf_squid
                    
        elif nonlinearity == 'KI':
            # Create NL (nonlinear) inductor with Taylor coefficients
            nl_name = self.make_unique_name(f"NL{component_name}")

            # In a Floquet-taper filter cell, the calling chain
            # (_taper_filter → add_filtered_stages → add_foster*_L_stage) sets
            # self._taper_cell_idx so we can emit a per-cell *numeric* Taylor poly
            # (L0 and Taylor coefficients vary with the cell's Z(n), since KI
            # decouples L0 from the nonlinearity strength). Using the shared
            # symbolic L0/c1/c2 here would freeze every NL element in the line to
            # whichever cell was processed first — that was the source of the
            # filter-resonance smearing observed when Z0_TWPA_ohm != Z0_ohm.
            if self._taper_cell_idx is not None:
                taylor_str = self._build_per_cell_taylor_str(self._taper_cell_idx)
                self.components.append(JCComponent(nl_name, str(node1), str(main_output_node), taylor_str))
            else:
                # Symbolic path (non-taper / non-Floquet cells)
                if 'L0' not in self.circuit_parameters:
                    self.circuit_parameters['L0'] = L_H

                if 'c1' not in self.circuit_parameters:
                    if 'c1_taylor' in self.workspace_data:
                        self.circuit_parameters['c1'] = self.workspace_data['c1_taylor']
                    else:
                        self.circuit_parameters['c1'] = epsilon_perA

                if 'c2' not in self.circuit_parameters:
                    if 'c2_taylor' in self.workspace_data:
                        self.circuit_parameters['c2'] = self.workspace_data['c2_taylor']
                    else:
                        self.circuit_parameters['c2'] = xi_perA2

                if 'c3' not in self.circuit_parameters and 'c3_taylor' in self.workspace_data:
                    self.circuit_parameters['c3'] = self.workspace_data['c3_taylor']

                if 'c4' not in self.circuit_parameters and 'c4_taylor' in self.workspace_data:
                    self.circuit_parameters['c4'] = self.workspace_data['c4_taylor']

                taylor_str = self.get_taylor_poly_string()
                self.components.append(JCComponent(nl_name, str(node1), str(main_output_node), taylor_str))
            
        elif nonlinearity == 'lin':
            l_name = self.make_unique_name(f"L{component_name}")
            l_symbol = self.create_symbolic_value(L_H, 'L', l_name)
            self.components.append(JCComponent(l_name, str(node1), str(main_output_node), l_symbol))
        
        # Add remainder inductance if needed (for all cases)
        if need_remainder:
            rem_name = self.make_unique_name(f"L{component_name}_rem")
            rem_symbol = self.create_symbolic_value(L_rem_H, 'L', rem_name)
            self.components.append(JCComponent(rem_name, str(intermediate_node), str(final_node), rem_symbol))
    
    
    def add_inductance_floquet(self, component_name, node1, node2, cell_idx,
                              nonlinearity, n_jj_struct):
        """Add inductance for a Floquet-tapered cell with per-cell numeric values.

        Uses per-cell L0_H, Taylor coefficients, and remainder from
        the reconstructed Floquet arrays in workspace_data.

        For jj_structure_type='rf_squid', also reads per-cell Lj_H (the JJ
        kinetic inductance, distinct from the effective parallel L0) and the
        optional Cj_extra_F array (used when rf_squid_constant_plasma=True;
        added as an extra Cjx{component_name} shunt cap to keep the local
        plasma frequency constant along the taper).
        """
        floquet_params = self.workspace_data.get('floquet_cell_params', {})
        L0_arr = floquet_params.get('L0_H')
        if L0_arr is None:
            raise ValueError("Floquet cell parameters not found in workspace_data")

        L0_cell = float(L0_arr[cell_idx])
        rem_arr = floquet_params.get('LTLsec_rem_H')
        rem_cell = float(rem_arr[cell_idx]) if rem_arr is not None else 0.0

        need_remainder = rem_cell > 1e-20
        final_node = node2
        if need_remainder:
            intermediate_node = self.get_new_node()
            main_output_node = intermediate_node
        else:
            main_output_node = node2

        # Build per-cell numeric poly string
        c1 = floquet_params.get('c1_taylor')
        c2 = floquet_params.get('c2_taylor')
        c3 = floquet_params.get('c3_taylor')
        c4 = floquet_params.get('c4_taylor')

        parts = [f"{L0_cell:.6e}"]
        for c_arr in [c1, c2, c3, c4]:
            if c_arr is not None and hasattr(c_arr, '__len__'):
                parts.append(f"{float(c_arr[cell_idx]):.6e}")
            elif c_arr is not None:
                parts.append(f"{float(c_arr):.6e}")
            else:
                break
        taylor_str = "poly " + ", ".join(parts)

        if nonlinearity == 'JJ':
            jj_structure_type = self.workspace_data.get('jj_structure_type', 'jj')
            use_taylor = self.workspace_data.get('use_taylor_insteadof_JJ', False)

            if use_taylor:
                # Taylor expansion: NL poly string (works for both bare JJ and rf_squid)
                nl_name = self.make_unique_name(f"NL{component_name}")
                self.components.append(JCComponent(nl_name, str(node1), str(main_output_node), taylor_str))
            else:
                # Circuit-level: Lj (numeric) + Lg if rf_squid.
                # For bare JJ: Lj = L0_H = LJ0 * w(n).
                # For rf_squid: Lj = Lj_dyn = LJ0 / w(n) (the JJ kinetic inductance,
                # NOT the effective parallel-combined L0). Lg is added in parallel below
                # so the parallel combination gives the correct effective inductance.
                Lj_arr = floquet_params.get('Lj_H')
                lj_cell = float(Lj_arr[cell_idx]) if Lj_arr is not None else L0_cell
                lj_name = self.make_unique_name(f"Lj{component_name}")
                self.components.append(JCComponent(lj_name, str(node1), str(main_output_node), f"{lj_cell:.6e}"))
                if jj_structure_type == 'rf_squid':
                    Lg_H = self.workspace_data.get('Lg_H_value')
                    if Lg_H is not None and not np.isinf(Lg_H) and Lg_H > 0:
                        lg_name = self.make_unique_name(f"Lg{component_name}")
                        lg_symbol = self.create_symbolic_value(Lg_H, 'L', f"Lg{component_name}")
                        self.components.append(JCComponent(lg_name, str(node1), str(main_output_node), lg_symbol))

            # Per-cell CJ (numeric)
            CJ_arr = floquet_params.get('CJ_F')
            if CJ_arr is not None:
                cj_val = float(CJ_arr[cell_idx])
                if cj_val > 0 and not np.isinf(cj_val):
                    cj_name = self.make_unique_name(f"Cj{component_name}")
                    self.components.append(JCComponent(cj_name, str(node1), str(main_output_node), f"{cj_val:.6e}"))

            # Optional extra shunt cap in parallel with the rf_squid (Cj + Lj || Lg),
            # used to keep the rf_squid plasma frequency constant along the line.
            CJx_arr = floquet_params.get('Cj_extra_F')
            if CJx_arr is not None:
                cjx_val = float(CJx_arr[cell_idx])
                if cjx_val > 0 and not np.isinf(cjx_val):
                    cjx_name = self.make_unique_name(f"Cjx{component_name}")
                    self.components.append(JCComponent(cjx_name, str(node1), str(main_output_node), f"{cjx_val:.6e}"))

        elif nonlinearity == 'KI':
            nl_name = self.make_unique_name(f"NL{component_name}")
            self.components.append(JCComponent(nl_name, str(node1), str(main_output_node), taylor_str))

        elif nonlinearity == 'lin':
            l_name = self.make_unique_name(f"L{component_name}")
            self.components.append(JCComponent(l_name, str(node1), str(main_output_node), f"{L0_cell:.6e}"))

        if need_remainder:
            rem_name = self.make_unique_name(f"L{component_name}_rem")
            self.components.append(JCComponent(rem_name, str(intermediate_node), str(final_node), f"{rem_cell:.6e}"))

    def add_capacitor(self, component_name, node1, node2, cap_value, Ncpersc_cell=None,
                     is_windowed=False, ind_g_C_with_filters=None, dispersion_type=None):
        """Add capacitor to JC netlist"""
        c_name = self.make_unique_name(f"C{component_name}")        
        
        # For windowed cells, write value directly inline
        if is_windowed and 'TLsec' in component_name:
            # Write numeric value directly, with dielectric loss if enabled
            enable_loss = self.workspace_data.get('enable_dielectric_loss', False)
            loss_tan = self.workspace_data.get('loss_tangent', 0.0)
            if enable_loss and loss_tan > 0:
                self.components.append(JCComponent(c_name, str(node1), str(node2), f"{cap_value:.6e}/(1+{loss_tan:.6e}im)"))
            else:
                self.components.append(JCComponent(c_name, str(node1), str(node2), f"{cap_value:.6e}"))
        else:
            # Use symbolic value as before
            c_symbol = self.create_symbolic_value(cap_value, 'C', component_name, 
                                                 Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
            self.components.append(JCComponent(c_name, str(node1), str(node2), c_symbol))
    
    def add_port(self, node, port_num):
        """Add port component to JC netlist"""
        port_name = f"P{node}_0"
        self.components.append(JCComponent(port_name, node, "0", str(port_num)))
    
    def add_resistor(self, component_name, node1, node2, value, symbolic_name=None):
        """Add resistor to JC netlist"""
        r_name = self.make_unique_name(component_name)
        
        if symbolic_name is None:
            symbolic_name = self.create_symbolic_value(value, 'R', component_name)
        else:
            self.circuit_parameters[symbolic_name] = value
        
        self.components.append(JCComponent(r_name, str(node1), str(node2), symbolic_name))

    
    def add_foster1_L_stage(self, start_node, k, k_idx, nonlinearity, n_jj_struct, n_poles,
                            LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                            Lg_H, epsilon_perA, xi_perA2,
                            Ncpersc_cell=None, ind_g_C_with_filters=None, dispersion_type=None,
                            Lj_numeric=None, Cj_numeric=None, Cjx_numeric=None):
        """Add Foster 1 L stage components inline and return output node"""

        current_node = start_node
        p = 1

        # Handle LinfLF1
        if check_flat(LinfLF1_H, k_idx) != 0:
            if check_flat(C0LF1_F, k_idx) != np.inf or (n_poles > 0 and CiLF1_F[k_idx, n_poles-1] != np.inf):
                next_node = self.get_new_node()
                self.add_inductance(f'infLF1_{k}', current_node, next_node, nonlinearity,
                                  check_flat(LinfLF1_H, k_idx), check_flat(LinfLF1_rem_H, k_idx),
                                  Lg_H, epsilon_perA, xi_perA2, n_jj_struct,
                                  Lj_numeric=Lj_numeric, Cj_numeric=Cj_numeric, Cjx_numeric=Cjx_numeric)
                current_node = next_node
            else:
                output_node = self.get_new_node()
                self.add_inductance(f'infLF1_{k}', current_node, output_node, nonlinearity,
                                  check_flat(LinfLF1_H, k_idx), check_flat(LinfLF1_rem_H, k_idx),
                                  Lg_H, epsilon_perA, xi_perA2, n_jj_struct,
                                  Lj_numeric=Lj_numeric, Cj_numeric=Cj_numeric, Cjx_numeric=Cjx_numeric)
                return output_node
            p += 1
        
        # Handle C0LF1
        if check_flat(C0LF1_F, k_idx) != np.inf:
            if n_poles > 0 and CiLF1_F[k_idx, n_poles-1] != np.inf:
                # Create next intermediate node
                next_node = self.get_new_node()
                self.add_capacitor(f'0LF1_{k}', current_node, next_node, check_flat(C0LF1_F, k_idx), 
                                 Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
                current_node = next_node
            else:
                # Connects to output
                output_node = self.get_new_node()
                self.add_capacitor(f'0LF1_{k}', current_node, output_node, check_flat(C0LF1_F, k_idx), 
                                 Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
                return output_node
            p += 1
        
        # Handle parallel LC pairs (all except the last)
        for m in range(n_poles - 1):
            if CiLF1_F[k_idx, m] != np.inf:
                # Create next node in the chain
                next_node = self.get_new_node()
                
                # Add inductor and capacitor in parallel between current and next
                l_name = self.make_unique_name(f'LiLF1_{p-1}_{k}')
                l_symbol = self.create_symbolic_value(LiLF1_H[k_idx, m], 'L', l_name, 
                                                     Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
                self.components.append(JCComponent(l_name, str(current_node), str(next_node), l_symbol))
                
                self.add_capacitor(f'iLF1_{p-1}_{k}', current_node, next_node, CiLF1_F[k_idx, m], 
                                 Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
                
                current_node = next_node
                p += 1
        
        # Last parallel LC pair (connects to output)
        if n_poles > 0 and CiLF1_F[k_idx, n_poles-1] != np.inf:
            output_node = self.get_new_node()
            
            l_name = self.make_unique_name(f'LiLF1_{p-1}_{k}')
            l_symbol = self.create_symbolic_value(LiLF1_H[k_idx, n_poles-1], 'L', l_name, 
                                                 Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
            self.components.append(JCComponent(l_name, str(current_node), str(output_node), l_symbol))
            
            self.add_capacitor(f'iLF1_{p-1}_{k}', current_node, output_node, CiLF1_F[k_idx, n_poles-1], 
                             Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
            return output_node
        
        return current_node
    
    def add_foster2_L_stage(self, start_node, k, k_idx, nonlinearity, n_jj_struct, n_zeros,
                           L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                           Lg_H, epsilon_perA, xi_perA2,
                           Ncpersc_cell=None, ind_g_C_with_filters=None, dispersion_type=None,
                           Lj_numeric=None, Cj_numeric=None, Cjx_numeric=None):
        """Add Foster 2 L stage components inline and return output node"""

        current_node = start_node
        output_node = self.get_new_node()

        # Handle L0LF2 - main series inductance
        self.add_inductance(f'0LF2_{k}', current_node, output_node, nonlinearity,
                          check_flat(L0LF2_H, k_idx), check_flat(L0LF2_rem_H, k_idx),
                          Lg_H, epsilon_perA, xi_perA2, n_jj_struct,
                          Lj_numeric=Lj_numeric, Cj_numeric=Cj_numeric, Cjx_numeric=Cjx_numeric)
        
        # Handle CinfLF2 - parallel capacitance (across the series inductance)
        if check_flat(CinfLF2_F, k_idx) != 0:
            self.add_capacitor(f'infLF2_{k}', current_node, output_node, check_flat(CinfLF2_F, k_idx), 
                             Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
        
        # Handle series LC branches (each branch goes from input to output)
        for m in range(n_zeros):
            # Create branch node
            branch_node = self.get_new_node()
            
            # Add inductor from input to branch node
            l_name = self.make_unique_name(f'LiLF2_{m+1}_{k}')
            l_symbol = self.create_symbolic_value(LiLF2_H[k_idx, m], 'L', l_name, 
                                                 Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
            self.components.append(JCComponent(l_name, str(current_node), str(branch_node), l_symbol))
            
            # Add capacitor from branch node to output
            self.add_capacitor(f'iLF2_{m+1}_{k}', branch_node, output_node, CiLF2_F[k_idx, m], 
                             Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
        
        return output_node
    
    def add_foster1_C_stage(self, node, k, k_idx, n_zeros,
                           LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                           Ncpersc_cell=None, ind_g_C_with_filters=None, dispersion_type=None):
        """Add Foster 1 C stage components (shunt to ground)"""
        
        p = 1
        current_shunt_node = node
        
        # Handle LinfCF1
        if check_flat(LinfCF1_H, k_idx) != 0:
            if check_flat(C0CF1_F, k_idx) != np.inf or (n_zeros > 0 and CiCF1_F[k_idx, n_zeros-1] != np.inf):
                shunt_node = self.get_new_node()
                l_name = self.make_unique_name(f'LinfCF1_{k}')
                l_symbol = self.create_symbolic_value(check_flat(LinfCF1_H, k_idx), 'L', l_name, 
                                                     Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
                self.components.append(JCComponent(l_name, str(node), str(shunt_node), l_symbol))
                current_shunt_node = shunt_node
            else:
                # Connects directly to ground
                l_name = self.make_unique_name(f'LinfCF1_{k}')
                l_symbol = self.create_symbolic_value(check_flat(LinfCF1_H, k_idx), 'L', l_name, 
                                                     Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
                self.components.append(JCComponent(l_name, str(node), "0", l_symbol))
                return
            p += 1
        
        # Handle C0CF1
        if check_flat(C0CF1_F, k_idx) != np.inf:
            if n_zeros > 0 and CiCF1_F[k_idx, n_zeros-1] != np.inf:
                next_shunt_node = self.get_new_node()
                self.add_capacitor(f'0CF1_{k}', current_shunt_node, next_shunt_node, check_flat(C0CF1_F, k_idx), 
                                 Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
                current_shunt_node = next_shunt_node
            else:
                # Connects to ground
                self.add_capacitor(f'0CF1_{k}', current_shunt_node, "0", check_flat(C0CF1_F, k_idx), 
                                 Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
                return
            p += 1
        
        # Handle shunt parallel LC pairs
        for m in range(n_zeros - 1):
            if CiCF1_F[k_idx, m] != np.inf:
                next_shunt_node = self.get_new_node()
                
                # Add inductor
                l_name = self.make_unique_name(f'LiCF1_{p-1}_{k}')
                l_symbol = self.create_symbolic_value(LiCF1_H[k_idx, m], 'L', l_name, 
                                                     Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
                self.components.append(JCComponent(l_name, str(current_shunt_node), str(next_shunt_node), l_symbol))
                
                # Add capacitor
                self.add_capacitor(f'iCF1_{p-1}_{k}', current_shunt_node, next_shunt_node, CiCF1_F[k_idx, m], 
                                 Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
                
                current_shunt_node = next_shunt_node
                p += 1
        
        # Last shunt LC pair to ground
        if n_zeros > 0 and CiCF1_F[k_idx, n_zeros-1] != np.inf:
            l_name = self.make_unique_name(f'LiCF1_{p-1}_{k}')
            l_symbol = self.create_symbolic_value(LiCF1_H[k_idx, n_zeros-1], 'L', l_name, 
                                                 Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
            self.components.append(JCComponent(l_name, str(current_shunt_node), "0", l_symbol))
            
            self.add_capacitor(f'iCF1_{p-1}_{k}', current_shunt_node, "0", CiCF1_F[k_idx, n_zeros-1], 
                             Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
    
    def add_foster2_C_stage(self, node, k, k_idx, n_poles,
                           L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                           Ncpersc_cell=None, ind_g_C_with_filters=None, dispersion_type=None):
        """Add Foster 2 C stage components (shunt to ground)"""
        
        # Handle L0CF2 - shunt inductor
        if check_flat(L0CF2_H, k_idx) != np.inf:
            l_name = self.make_unique_name(f'L0CF2_{k}')
            l_symbol = self.create_symbolic_value(check_flat(L0CF2_H, k_idx), 'L', l_name, 
                                                 Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
            self.components.append(JCComponent(l_name, str(node), "0", l_symbol))
        
        # Handle CinfCF2 - shunt capacitor
        if check_flat(CinfCF2_F, k_idx) != 0:
            self.add_capacitor(f'infCF2_{k}', node, "0", check_flat(CinfCF2_F, k_idx), 
                             Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
        
        # Handle shunt LC pairs
        for m in range(n_poles):
            # Create branch node
            branch_node = self.get_new_node()
            
            # Add inductor from main node to branch
            l_name = self.make_unique_name(f'LiCF2_{m+1}_{k}')
            l_symbol = self.create_symbolic_value(LiCF2_H[k_idx, m], 'L', l_name, 
                                                 Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
            self.components.append(JCComponent(l_name, str(node), str(branch_node), l_symbol))
            
            # Add capacitor from branch to ground
            self.add_capacitor(f'iCF2_{m+1}_{k}', branch_node, "0", CiCF2_F[k_idx, m], 
                             Ncpersc_cell, False, ind_g_C_with_filters, dispersion_type)
        
    
    def add_filtered_stages(self, start_node, k, k_idx, Foster_form_L, Foster_form_C, nonlinearity, n_jj_struct, n_poles, n_zeros,
                           LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                           L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                           LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                           L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                           Lg_H, epsilon_perA, xi_perA2,
                           Ncpersc_cell=None, ind_g_C_with_filters=None, dispersion_type=None,
                           Lj_numeric=None, Cj_numeric=None, Cjx_numeric=None):
        """Add filter stages inline and return output node"""

        current_node = start_node

        # Series filter (changes the node)
        if Foster_form_L == 1:
            series_k_idx = k_idx
            if len(LinfLF1_H) == 1:
                series_k_idx = 0
            current_node = self.add_foster1_L_stage(current_node, k, series_k_idx, nonlinearity, n_jj_struct, n_poles,
                                                  LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                                                  Lg_H, epsilon_perA, xi_perA2,
                                                  Ncpersc_cell, ind_g_C_with_filters, dispersion_type,
                                                  Lj_numeric=Lj_numeric, Cj_numeric=Cj_numeric, Cjx_numeric=Cjx_numeric)
        else:
            series_k_idx = k_idx
            if len(L0LF2_H) == 1:
                series_k_idx = 0
            current_node = self.add_foster2_L_stage(current_node, k, series_k_idx, nonlinearity, n_jj_struct, n_zeros,
                                                  L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                                                  Lg_H, epsilon_perA, xi_perA2,
                                                  Ncpersc_cell, ind_g_C_with_filters, dispersion_type,
                                                  Lj_numeric=Lj_numeric, Cj_numeric=Cj_numeric, Cjx_numeric=Cjx_numeric)
        
        # Shunt filter (doesn't change the node) - use original k_idx
        if Foster_form_C == 1:
            shunt_k_idx = k_idx
            if len(C0CF1_F) == 1:
                shunt_k_idx = 0
            self.add_foster1_C_stage(current_node, k, shunt_k_idx, n_zeros,
                                   LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                                   Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
        else:
            shunt_k_idx = k_idx
            if len(CinfCF2_F) == 1:
                shunt_k_idx = 0
            self.add_foster2_C_stage(current_node, k, shunt_k_idx, n_poles,
                                   L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                                   Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
        
        return current_node
    

    
    def expand_supercell_inline(self, cell_index, start_node, Ncpersc_cell, width, ind_g_C_with_filters,
                              nonlinearity, LTLsec_rem_H, Lg_H, L0_H, epsilon_perA, xi_perA2, 
                              CJ_F, LTLsec_H, CTLsec_F, Ic_JJ_uA, ngL, ngC, Foster_form_L, 
                              Foster_form_C, n_jj_struct, n_poles, n_zeros,
                              LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                              L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                              LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                              L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                              n_filters_per_sc=0, dispersion_type=None):
        """Expand a supercell inline instead of using subcircuit call"""
        
        current_node = start_node
        
        # Handle pure filter case (dispersion_type == 'filter')
        if dispersion_type == 'filter' and 'nTLsec' in self.workspace_data:
            
            # Get nTLsec value
            n_tl_sections = self.workspace_data.get('nTLsec', 0)
            
            if n_tl_sections == 0:
                # Supercell is just one filter
                k_idx = 0  # All filters are identical in filter case
                current_node = self.add_filtered_stages(current_node, cell_index, k_idx, 
                                                      Foster_form_L, Foster_form_C, nonlinearity, 
                                                      n_jj_struct, n_poles, n_zeros,
                                                      LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                                                      L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                                                      LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                                                      L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                                                      Lg_H, epsilon_perA, xi_perA2,
                                                      Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
            else:
                # Supercell has nTLsec/2 TL sections, then filter, then nTLsec/2 TL sections
                # First half of TL sections
                for j in range(n_tl_sections // 2):
                    next_node = self.get_new_node()
                    cell_name = f'{cell_index}_{j}'
                    self.add_inductance(f'TLsec_{cell_name}', current_node, next_node, nonlinearity, 
                                      LTLsec_H, LTLsec_rem_H, Lg_H, epsilon_perA, xi_perA2, n_jj_struct)
                    # Handle CTLsec_F which might be array or scalar
                    cap_value = CTLsec_F if np.isscalar(CTLsec_F) else check_flat(CTLsec_F, 0)
                    self.add_capacitor(f'TLsec_{cell_name}', next_node, "0", cap_value, Ncpersc_cell)
                    current_node = next_node
                
                # Add the filter
                k_idx = 0  # All filters are identical
                current_node = self.add_filtered_stages(current_node, cell_index, k_idx, 
                                                      Foster_form_L, Foster_form_C, nonlinearity, 
                                                      n_jj_struct, n_poles, n_zeros,
                                                      LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                                                      L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                                                      LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                                                      L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                                                      Lg_H, epsilon_perA, xi_perA2,
                                                      Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
                
                # Second half of TL sections
                for j in range(n_tl_sections // 2, n_tl_sections):
                    next_node = self.get_new_node()
                    cell_name = f'{cell_index}_{j}'
                    self.add_inductance(f'TLsec_{cell_name}', current_node, next_node, nonlinearity, 
                                      LTLsec_H, LTLsec_rem_H, Lg_H, epsilon_perA, xi_perA2, n_jj_struct)
                    cap_value = CTLsec_F if np.isscalar(CTLsec_F) else check_flat(CTLsec_F, 0)
                    self.add_capacitor(f'TLsec_{cell_name}', next_node, "0", cap_value, Ncpersc_cell)
                    current_node = next_node
                    
            return current_node
        
        # Handle periodic/both cases - use ind_g_C_with_filters
        cell_idx_base = width + (cell_index - 1) * Ncpersc_cell  # Starting cell index for this supercell
        
        # Calculate filter index offset for this supercell
        p_offset = int(n_filters_per_sc * width / Ncpersc_cell) + (cell_index - 1) * n_filters_per_sc
        p = 0  # Local filter counter within this supercell
        
        # Build each cell in the supercell
        for j in range(Ncpersc_cell):
            cell_idx = cell_idx_base + j
        
            # For periodic/both cases, check if LOCAL position j is a filter position
            # by checking if j is in the pattern [8, 17] (modulo Ncpersc_cell)
            local_filter_positions = [pos % Ncpersc_cell for pos in ind_g_C_with_filters]
            
            if j in local_filter_positions:
                # This is a filter cell
                # Map to position within supercell pattern (0 or 1 for 2 filters per supercell)
                k_idx = local_filter_positions.index(j)  # This will be 0 or 1
                
                current_node = self.add_filtered_stages(current_node, cell_idx, k_idx, 
                                                      Foster_form_L, Foster_form_C, nonlinearity, 
                                                      n_jj_struct, n_poles, n_zeros,
                                                      LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                                                      L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                                                      LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                                                      L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                                                      Lg_H, epsilon_perA, xi_perA2,
                                                      Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
                p += 1
            else:
                # Regular TL cell
                next_node = self.get_new_node()
                
                self.add_inductance(f'TLsec_{cell_idx}', current_node, next_node, nonlinearity, 
                                  LTLsec_H, LTLsec_rem_H, Lg_H, epsilon_perA, xi_perA2, n_jj_struct)
                
                self.add_capacitor(f'TLsec_{cell_idx}', next_node, "0", check_flat(CTLsec_F, cell_idx), Ncpersc_cell)
                
                current_node = next_node
        
        return current_node
    
    
    def expand_periodicfiltered_TWPA(self, start_node, Nsc_cell, Ncpersc_cell, width, ind_g_C_with_filters,
                                    nonlinearity, LTLsec_rem_H, Lg_H, L0_H, epsilon_perA, xi_perA2, 
                                    CJ_F, LTLsec_H, CTLsec_F, Ic_JJ_uA, ngL, ngC, Foster_form_L, 
                                    Foster_form_C, n_jj_struct, n_poles, n_zeros,
                                    LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                                    L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                                    LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                                    L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                                    n_periodic_sc, n_filters_per_sc, n_periodic_sc_init,
                                    dispersion_type=None):
        """Expand periodicfiltered TWPA inline"""
        
        # Get the linear window flag
        use_linear_in_window = self.workspace_data.get('use_linear_in_window', False)
        # Choose nonlinearity type for window
        window_nonlinearity = 'lin' if use_linear_in_window else nonlinearity        

        current_node = start_node
        p = 0  # Filter index tracker
        cell_idx = 0  # Cell index tracker
        
        # Extract one supercell pattern from CTLsec_F if it's periodic
        if hasattr(CTLsec_F, '__len__') and width > 0:
            # Get one supercell pattern from the original periodic section
            sc_start = width  # Start of first periodic supercell
            sc_end = sc_start + Ncpersc_cell
            if sc_end <= len(CTLsec_F):
                CTLsec_pattern = CTLsec_F[sc_start:sc_end]
            else:
                # Fallback if original is too small
                CTLsec_pattern = CTLsec_F[:Ncpersc_cell] if len(CTLsec_F) >= Ncpersc_cell else CTLsec_F
        else:
            CTLsec_pattern = None
        
        # 1. First window part - use exact values from CTLsec_F
        for _ in range(width):
            if cell_idx in ind_g_C_with_filters:
                # This is a filter cell
                # Map to position within supercell pattern
                filter_positions_in_sc = [pos % Ncpersc_cell for pos in ind_g_C_with_filters]
                cell_pos_in_sc = cell_idx % Ncpersc_cell
                k_idx = filter_positions_in_sc.index(cell_pos_in_sc)  # This will be 0 or 1
                
                current_node = self.add_filtered_stages(current_node, cell_idx, k_idx, 
                                                      Foster_form_L, Foster_form_C, nonlinearity, 
                                                      n_jj_struct, n_poles, n_zeros,
                                                      LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                                                      L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                                                      LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                                                      L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                                                      Lg_H, epsilon_perA, xi_perA2,
                                                      Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
                p += 1
            else:
                # Regular TL section in window
                next_node = self.get_new_node()                

                self.add_inductance(f'TLsec_{cell_idx}', current_node, next_node, window_nonlinearity, 
                                  LTLsec_H, LTLsec_rem_H, Lg_H, epsilon_perA, xi_perA2, n_jj_struct)
                
                # Pass is_windowed=True for window cells
                if hasattr(CTLsec_F, '__len__'):
                    cap_value = CTLsec_F[cell_idx] if cell_idx < len(CTLsec_F) else CTLsec_F[0]
                else:
                    cap_value = CTLsec_F                
                self.add_capacitor(f'TLsec_{cell_idx}', next_node, "0", cap_value, Ncpersc_cell, is_windowed=True)
                current_node = next_node
            
            cell_idx += 1
        
        # 2. Periodic part - repeat the supercell pattern
        for sc in range(n_periodic_sc):
            # Each supercell has Ncpersc_cell cells
            for j in range(Ncpersc_cell):
                cell_idx = width + sc * Ncpersc_cell + j
                
                if cell_idx in ind_g_C_with_filters:
                    # Filter cell
                    # Map to position within supercell pattern
                    filter_positions_in_sc = [pos % Ncpersc_cell for pos in ind_g_C_with_filters]
                    cell_pos_in_sc = cell_idx % Ncpersc_cell
                    k_idx = filter_positions_in_sc.index(cell_pos_in_sc)  # This will be 0 or 1

                    
                    current_node = self.add_filtered_stages(current_node, cell_idx, k_idx, 
                                                          Foster_form_L, Foster_form_C, nonlinearity, 
                                                          n_jj_struct, n_poles, n_zeros,
                                                          LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                                                          L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                                                          LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                                                          L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                                                          Lg_H, epsilon_perA, xi_perA2,
                                                          Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
                    p += 1
                else:
                    # TL section - use pattern
                    next_node = self.get_new_node()
                    self.add_inductance(f'TLsec_{cell_idx}', current_node, next_node, nonlinearity, 
                                      LTLsec_H, LTLsec_rem_H, Lg_H, epsilon_perA, xi_perA2, n_jj_struct)
                    
                    # Use the supercell pattern for capacitance
                    if CTLsec_pattern is not None and hasattr(CTLsec_pattern, '__len__'):
                        cap_value = CTLsec_pattern[j % len(CTLsec_pattern)]
                    elif hasattr(CTLsec_F, '__len__'):
                        cap_value = CTLsec_F[0]  # Fallback
                    else:
                        cap_value = CTLsec_F
                    
                    self.add_capacitor(f'TLsec_{cell_idx}', next_node, "0", cap_value, Ncpersc_cell)
                    current_node = next_node
        
        # Update p to account for skipped supercells (if we reduced n_periodic_sc)
        if n_periodic_sc < n_periodic_sc_init:
            p += (n_periodic_sc_init - n_periodic_sc) * n_filters_per_sc
        
        # 3. Last window part - use exact values from the end of original CTLsec_F
        original_last_window_start = width + n_periodic_sc_init * Ncpersc_cell
        
        for i in range(width):
            cell_idx = width + n_periodic_sc * Ncpersc_cell + i
            original_idx = original_last_window_start + i
            
            if cell_idx in ind_g_C_with_filters:
                # Filter cell
                # Map to position within supercell pattern
                filter_positions_in_sc = [pos % Ncpersc_cell for pos in ind_g_C_with_filters]
                cell_pos_in_sc = cell_idx % Ncpersc_cell
                k_idx = filter_positions_in_sc.index(cell_pos_in_sc)  # This will be 0 or 1
                
                current_node = self.add_filtered_stages(current_node, cell_idx, k_idx, 
                                                      Foster_form_L, Foster_form_C, nonlinearity, 
                                                      n_jj_struct, n_poles, n_zeros,
                                                      LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                                                      L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                                                      LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                                                      L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                                                      Lg_H, epsilon_perA, xi_perA2,
                                                      Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
                p += 1
            else:
                # Regular TL section in window
                next_node = self.get_new_node()
                self.add_inductance(f'TLsec_{cell_idx}', current_node, next_node, window_nonlinearity, 
                                  LTLsec_H, LTLsec_rem_H, Lg_H, epsilon_perA, xi_perA2, n_jj_struct)
                
                # Pass is_windowed=True for window cells
                if hasattr(CTLsec_F, '__len__') and original_idx < len(CTLsec_F):
                    cap_value = CTLsec_F[original_idx]
                elif hasattr(CTLsec_F, '__len__'):
                    cap_value = CTLsec_F[-1]
                else:
                    cap_value = CTLsec_F
                    
                self.add_capacitor(f'TLsec_{cell_idx}', next_node, "0", cap_value, Ncpersc_cell, is_windowed=True)
                current_node = next_node
        
        return current_node

    def expand_floquet_TWPA(self, start_node, Ntot_cell, Ncpersc_cell, width,
                           ind_g_C_with_filters, nonlinearity,
                           LTLsec_rem_H, Lg_H, L0_H, epsilon_perA, xi_perA2,
                           CJ_F, LTLsec_H, CTLsec_F, Ic_JJ_uA, ngL, ngC,
                           Foster_form_L, Foster_form_C, n_jj_struct, n_poles, n_zeros,
                           LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                           L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                           LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                           L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                           n_periodic_sc, n_filters_per_sc, n_periodic_sc_init,
                           dispersion_type=None):
        """Expand Floquet-tapered TWPA: taper cells use per-cell numeric values,
        center cells use symbolic parameters (same as non-Floquet)."""

        current_node = start_node
        cell_idx = 0
        p = 0  # Filter index tracker
        nTLsec = self.workspace_data.get('nTLsec', 0)

        def is_filter_cell(idx):
            """Determine if cell is a filter position.

            For both 'filter' and 'both' dispersion the filter cells recur every
            Ncpersc_cell cells. `ind_g_C_with_filters` lists positions *within
            one supercell* (not absolute indices along the whole line), so we
            modulo by Ncpersc_cell before checking — otherwise only the first
            supercell would actually get filters.
            """
            if dispersion_type == 'filter':
                return (idx % Ncpersc_cell) == (nTLsec // 2)
            elif dispersion_type == 'both':
                if not ind_g_C_with_filters:
                    return False
                cell_in_sc = idx % Ncpersc_cell
                return any(cell_in_sc == (pos % Ncpersc_cell) for pos in ind_g_C_with_filters)
            return False

        # Extract CTLsec pattern for center supercells
        if hasattr(CTLsec_F, '__len__') and width > 0 and len(CTLsec_F) > 2 * width:
            CTLsec_pattern = CTLsec_F[width:width + Ncpersc_cell]
        elif hasattr(CTLsec_F, '__len__') and len(CTLsec_F) >= Ncpersc_cell:
            CTLsec_pattern = CTLsec_F[:Ncpersc_cell]
        else:
            CTLsec_pattern = None

        # Per-cell arrays for taper filter cells
        floquet_params = self.workspace_data.get('floquet_cell_params', {})
        L0_arr = floquet_params.get('L0_H')
        Lj_dyn_arr = floquet_params.get('Lj_H')  # JJ kinetic inductance per cell
        CJ_arr = floquet_params.get('CJ_F')
        CJx_arr = floquet_params.get('Cj_extra_F')
        ffc = self.workspace_data.get('floquet_filter_components', {})

        def _taper_filter(idx):
            """Add a filter cell in the taper using per-cell component values."""
            # For JJ component: use Lj_dyn (the actual JJ kinetic inductance).
            # For bare JJ: Lj_dyn == L0. For rf_squid: Lj_dyn != L0 (Lg adds in parallel).
            # Falls back to L0 if Lj_dyn not present (e.g., KI nonlinearity).
            if Lj_dyn_arr is not None:
                lj_n = float(Lj_dyn_arr[idx])
            elif L0_arr is not None:
                lj_n = float(L0_arr[idx])
            else:
                lj_n = None
            cj_n = float(CJ_arr[idx]) if CJ_arr is not None else None
            cjx_n = float(CJx_arr[idx]) if CJx_arr is not None else None
            f = ffc.get(idx, {})
            self.force_numeric = True
            self._taper_cell_idx = idx
            result = self.add_filtered_stages(
                current_node, idx, p, Foster_form_L, Foster_form_C, nonlinearity,
                n_jj_struct, n_poles, n_zeros,
                f.get('LinfLF1_H', LinfLF1_H), f.get('LinfLF1_rem_H', LinfLF1_rem_H),
                f.get('C0LF1_F', C0LF1_F), f.get('LiLF1_H', LiLF1_H), f.get('CiLF1_F', CiLF1_F),
                f.get('L0LF2_H', L0LF2_H), f.get('L0LF2_rem_H', L0LF2_rem_H),
                f.get('CinfLF2_F', CinfLF2_F), f.get('LiLF2_H', LiLF2_H), f.get('CiLF2_F', CiLF2_F),
                f.get('LinfCF1_H', LinfCF1_H), f.get('C0CF1_F', C0CF1_F),
                f.get('LiCF1_H', LiCF1_H), f.get('CiCF1_F', CiCF1_F),
                f.get('L0CF2_H', L0CF2_H), f.get('CinfCF2_F', CinfCF2_F),
                f.get('LiCF2_H', LiCF2_H), f.get('CiCF2_F', CiCF2_F),
                Lg_H, epsilon_perA, xi_perA2,
                Ncpersc_cell, ind_g_C_with_filters, dispersion_type,
                Lj_numeric=lj_n, Cj_numeric=cj_n, Cjx_numeric=cjx_n)
            self.force_numeric = False
            self._taper_cell_idx = None
            return result

        # 1. Left taper (cell-by-cell with per-cell numeric values)
        for _ in range(width):
            if is_filter_cell(cell_idx):
                current_node = _taper_filter(cell_idx)
                p += 1
            else:
                next_node = self.get_new_node()
                self.add_inductance_floquet(f'TLsec_{cell_idx}', current_node, next_node,
                                          cell_idx, nonlinearity, n_jj_struct)
                cap_value = CTLsec_F[cell_idx] if hasattr(CTLsec_F, '__len__') else CTLsec_F
                self.add_capacitor(f'TLsec_{cell_idx}', next_node, "0", cap_value,
                                 Ncpersc_cell, is_windowed=True)
                current_node = next_node
            cell_idx += 1

        # 2. Center (symbolic parameters, supercell pattern)
        for sc in range(n_periodic_sc):
            for j in range(Ncpersc_cell):
                center_cell_idx = width + j
                if is_filter_cell(center_cell_idx):
                    current_node = self.add_filtered_stages(
                        current_node, cell_idx, p, Foster_form_L, Foster_form_C, nonlinearity,
                        n_jj_struct, n_poles, n_zeros,
                        LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                        L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                        LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                        L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                        Lg_H, epsilon_perA, xi_perA2,
                        Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
                    p += 1
                else:
                    next_node = self.get_new_node()
                    self.add_inductance(f'TLsec_{cell_idx}', current_node, next_node,
                                      nonlinearity, LTLsec_H, LTLsec_rem_H,
                                      Lg_H, epsilon_perA, xi_perA2, n_jj_struct)
                    cap_value = CTLsec_pattern[j] if CTLsec_pattern is not None else CTLsec_F
                    self.add_capacitor(f'TLsec_{cell_idx}', next_node, "0", cap_value,
                                     Ncpersc_cell)
                    current_node = next_node
                cell_idx += 1

        # 3. Right taper (cell-by-cell with per-cell numeric values)
        for i in range(width):
            cell_idx_mapped = width + n_periodic_sc * Ncpersc_cell + i

            if is_filter_cell(cell_idx_mapped):
                current_node = _taper_filter(cell_idx_mapped)
                p += 1
            else:
                next_node = self.get_new_node()
                self.add_inductance_floquet(f'TLsec_{cell_idx_mapped}', current_node, next_node,
                                          cell_idx_mapped, nonlinearity, n_jj_struct)
                cap_value = CTLsec_F[cell_idx_mapped] if hasattr(CTLsec_F, '__len__') and cell_idx_mapped < len(CTLsec_F) else (CTLsec_F[-1] if hasattr(CTLsec_F, '__len__') else CTLsec_F)
                self.add_capacitor(f'TLsec_{cell_idx_mapped}', next_node, "0", cap_value,
                                 Ncpersc_cell, is_windowed=True)
                current_node = next_node

        return current_node

#####################################################################################

def load_design_parameters(design_file: str) -> Dict[str, Any]:
    """Load design parameters from file (Cell 3 from notebook)."""
    print(f"Loading design parameters from {design_file}...")
    
    # Load the design module
    spec = importlib.util.spec_from_file_location("design", design_file)
    if spec is None:
        raise ImportError(f"Could not load module spec from {design_file}")
    
    design = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Module spec has no loader for {design_file}")
        
    spec.loader.exec_module(design)
    
    # Extract all parameters from the three dictionaries
    all_params = {}
    
    # Load circuit parameters
    if hasattr(design, 'circuit'):
        all_params.update(design.circuit)
    
    # Load config parameters  
    if hasattr(design, 'config'):
        all_params.update(design.config)
        
    # Load characteristics (if needed)
    if hasattr(design, 'characteristics'):
        # Store characteristics separately if you need them
        characteristics = design.characteristics
    else:
        characteristics = {}
    
    # Set defaults for optional parameters (from Cell 3)
    if 'ind_g_C_with_filters' not in all_params:
        all_params['ind_g_C_with_filters'] = []
    if 'n_filters_per_sc' not in all_params:
        all_params['n_filters_per_sc'] = 0
    if 'width' not in all_params:
        all_params['width'] = 0
    if 'n_periodic_sc' not in all_params:
        all_params['n_periodic_sc'] = all_params.get('Nsc_cell', 0)
    if 'ngL' not in all_params:
        all_params['ngL'] = 1
    if 'ngC' not in all_params:
        all_params['ngC'] = 1
        
    print(f"Loaded design from {design_file}")
    print(f"  Nonlinearity: {all_params.get('nonlinearity', 'unknown')}")
    print(f"  Dispersion type: {all_params.get('dispersion_type', 'unknown')}")
    
    return {
        'params': all_params,
        'characteristics': characteristics,
        'source_file': design_file
    }

#####################################################################################

def prepare_workspace_variables(design_data: Dict, netlist_config: NetlistConfig) -> Dict[str, Any]:
    """Prepare workspace variables for building (Cell 4 from notebook)."""
    # Extract all parameters from design_data
    params = design_data['params'].copy()  # Make a copy to avoid modifying original
    
    # Apply Ntot_cell override if specified
    if netlist_config.Ntot_cell_override is not None:
        params['Ntot_cell'] = netlist_config.Ntot_cell_override
    
    # Extract commonly used parameters
    Ntot_cell = params['Ntot_cell']
    Ncpersc_cell = params['Ncpersc_cell']
    dispersion_type = params.get('dispersion_type', 'filter')
    window_type = params.get('window_type', 'boxcar')
    nonlinearity = params.get('nonlinearity', 'JJ')
    
    # Record initial values
    Ntot_cell_init = design_data['params']['Ntot_cell']  # Original value
    
    # Recalculate supercell counts
    Nsc_cell = int(np.round(Ntot_cell/Ncpersc_cell))
    Ntot_cell = int(Nsc_cell*Ncpersc_cell)  # Ensure it's a multiple
    
    # Update params with recalculated values
    params['Nsc_cell'] = Nsc_cell
    params['Ntot_cell'] = Ntot_cell
    
    # Calculate periodic supercells based on window/taper type. Either a Floquet
    # nonlinearity taper or a standalone impedance taper triggers per-cell numeric
    # expansion of the line (see helper_functions.compute_taper_arrays).
    is_tapered = params.get('floquet_taper', False) or params.get('Z_taper', False)
    is_floquet = is_tapered  # legacy alias used downstream
    if is_floquet:
        width = params.get('width', 0)
        n_periodic_sc = params.get('n_periodic_sc', int((Ntot_cell - 2*width)/Ncpersc_cell))
        n_periodic_sc_init = n_periodic_sc
    elif dispersion_type == 'both' and window_type.lower() == 'tukey':
        width = params.get('width', 0)
        n_periodic_sc = int((Ntot_cell - 2*width)/Ncpersc_cell)
        n_periodic_sc_init = params.get('window_params', {}).get('n_periodic_sc', params.get('n_periodic_sc', Nsc_cell))
    else:
        width = 0
        n_periodic_sc = Nsc_cell
        n_periodic_sc_init = Nsc_cell
    
    params['width'] = width
    params['n_periodic_sc'] = n_periodic_sc
    params['n_periodic_sc_init'] = n_periodic_sc_init
    
    # ========== RECONSTRUCT ARRAYS FOR NEW SIZE ==========
    # Copy the array reconstruction code from Cell 4 here
    
    if dispersion_type in ['periodic', 'both']:
        # Reconstruct periodic capacitance array
        if 'CTLsec_pattern' in params:
            CTLsec_pattern = params.get('CTLsec_pattern')
            if window_type.lower() == 'boxcar':
                CTLsec_F = np.tile(CTLsec_pattern, Nsc_cell)
            else:
                # Windowed case - properly reconstruct from parts
                if 'CTLsec_window_start' in params and 'CTLsec_window_end' in params:
                    window_start = params.get('CTLsec_window_start', np.array([]))
                    window_end = params.get('CTLsec_window_end', np.array([]))
                    
                    # For the new size, we need to adjust the middle part
                    if Nsc_cell != params.get('Nsc_cell'):
                        # Different size - reconstruct with new n_periodic_sc
                        middle_part = np.tile(CTLsec_pattern, n_periodic_sc)
                        CTLsec_F = np.concatenate([window_start, middle_part, window_end])
                    else:
                        # Same size - use original reconstruction
                        window_params = params.get('window_params', {})
                        n_periodic_sc_saved = window_params.get('n_periodic_sc', n_periodic_sc)
                        middle_part = np.tile(CTLsec_pattern, n_periodic_sc_saved)
                        CTLsec_F = np.concatenate([window_start, middle_part, window_end])
                else:
                    # Fallback - use what's saved
                    CTLsec_F = params.get('CTLsec_F', CTLsec_pattern)
        else:
            CTLsec_F = params.get('CTLsec_F', 0)
        
        # Reconstruct g_C_mod if needed (same logic)
        if 'g_C_pattern' in params:
            g_C_pattern = params.get('g_C_pattern')
            if window_type.lower() == 'boxcar':
                g_C_mod = np.tile(g_C_pattern, Nsc_cell)
            else:
                # Windowed case - properly reconstruct from parts
                if 'g_C_window_start' in params and 'g_C_window_end' in params:
                    window_start = params.get('g_C_window_start', np.array([]))
                    window_end = params.get('g_C_window_end', np.array([]))
                    
                    if Nsc_cell != params.get('Nsc_cell'):
                        # Different size - reconstruct with new n_periodic_sc
                        middle_part = np.tile(g_C_pattern, n_periodic_sc)
                        g_C_mod = np.concatenate([window_start, middle_part, window_end])
                    else:
                        # Same size - use original reconstruction
                        window_params = params.get('window_params', {})
                        n_periodic_sc_saved = window_params.get('n_periodic_sc', n_periodic_sc)
                        middle_part = np.tile(g_C_pattern, n_periodic_sc_saved)
                        g_C_mod = np.concatenate([window_start, middle_part, window_end])
                else:
                    g_C_mod = params.get('g_C_mod', g_C_pattern)
        else:
            g_C_mod = params.get('g_C_mod', None)
        
        # Get filter positions
        ind_g_C_with_filters = params.get('ind_g_C_with_filters', [])
        n_filters_per_sc = params.get('n_filters_per_sc', 0)
    else:
        # Non-periodic case
        CTLsec_F = params.get('CTLsec_F', 0)
        g_C_mod = None
        ind_g_C_with_filters = []
        n_filters_per_sc = 0
    
    # Update params with reconstructed arrays
    params['CTLsec_F'] = CTLsec_F
    params['g_C_mod'] = g_C_mod
    params['ind_g_C_with_filters'] = ind_g_C_with_filters
    params['n_filters_per_sc'] = n_filters_per_sc
    
    # ========== INITIALIZE COMPONENT PARAMETERS ==========
    # Filter parameters (if applicable)
    if dispersion_type in ['filter', 'both']:
        n_zeros = params.get('n_zeros', 0)
        n_poles = params.get('n_poles', 0)
        Foster_form_L = params.get('Foster_form_L', 1)
        Foster_form_C = params.get('Foster_form_C', 1)
    else:
        n_zeros = n_poles = 0
        Foster_form_L = Foster_form_C = 1
    
    # Nonlinearity-specific parameters
    if nonlinearity == 'JJ':
        Ic_JJ_uA = params.get('Ic_JJ_uA', None)
        CJ_F = params.get('CJ_F', np.inf) if params.get('CJ_F', np.inf) != np.inf else np.inf
        Lg_H = params.get('Lg_H', np.inf) if params.get('Lg_H', np.inf) != np.inf else np.inf
        L0_H = params.get('L0_H', 0)  # Single JJ structure inductance
        LJ0_H = params.get('LJ0_H', 0)
        jj_structure_type = params.get('jj_structure_type', 'rf_squid')
    else:
        # KI case
        Ic_JJ_uA = None
        CJ_F = Lg_H = np.inf
        L0_H = params.get('L0_H', None)
        if L0_H is None or L0_H == 0:
            raise ValueError("L0_H must be defined and non-zero for KI (kinetic inductance) nonlinearity. "
                           "Check that 'L0_H' is present in the 'circuit' dict of your design file.")
        LJ0_H = 0  # No JJ inductance in KI case
        jj_structure_type = None
    
    # ========== LOAD FILTER ARRAYS ==========
    # Initialize all filter arrays efficiently
    # Using Dict[str, Any] to allow numpy arrays
    filter_arrays = {}
    filter_keys = [
        'LinfLF1_H', 'LinfLF1_rem_H', 'C0LF1_F', 'LiLF1_H', 'CiLF1_F',
        'L0LF2_H', 'L0LF2_rem_H', 'CinfLF2_F', 'LiLF2_H', 'CiLF2_F',
        'LinfCF1_H', 'C0CF1_F', 'LiCF1_H', 'CiCF1_F',
        'L0CF2_H', 'CinfCF2_F', 'LiCF2_H', 'CiCF2_F'
    ]
    
    if dispersion_type in ['filter', 'both']:
        for key in filter_keys:
            val = params.get(key, np.array([]))
            # Ensure it's a numpy array (handles lists from new export format)
            filter_arrays[key] = np.array(val) if not isinstance(val, np.ndarray) else val
    else:
        # Initialize empty arrays for non-filter case
        for key in filter_keys:
            filter_arrays[key] = np.array([])
    
    # Update params with all these values
    params.update({
        'n_zeros': n_zeros,
        'n_poles': n_poles,
        'Foster_form_L': Foster_form_L,
        'Foster_form_C': Foster_form_C,
        'Ic_JJ_uA': Ic_JJ_uA,
        'CJ_F': CJ_F,
        'Lg_H': Lg_H,
        'L0_H': L0_H,
        'LJ0_H': LJ0_H,
        'jj_structure_type': jj_structure_type,
        **filter_arrays  # Unpack all filter arrays into params
    })
    
    # Prepare workspace variables for the builder
    workspace_vars = {
        'c1_taylor': params.get('c1_taylor', None),
        'c2_taylor': params.get('c2_taylor', None),
        'c3_taylor': params.get('c3_taylor', None),
        'c4_taylor': params.get('c4_taylor', None),
        'L0_H': params.get('L0_H', None),
        'nTLsec': params.get('nTLsec', 0),
        'dispersion_type': dispersion_type,
        'jj_structure_type': params.get('jj_structure_type', ''),
        'use_taylor_insteadof_JJ': netlist_config.use_taylor_insteadof_JJ,
        'enable_dielectric_loss': netlist_config.enable_dielectric_loss,
        'loss_tangent': netlist_config.loss_tangent,
        'use_linear_in_window': netlist_config.use_linear_in_window,
    }

    # Tapered TWPA: reconstruct per-cell parameters from saved config params.
    # Calls the same shared helper as the designer to guarantee identical arrays.
    if is_tapered:
        from .helper_functions import compute_taper_arrays
        workspace_vars['floquet_taper'] = True  # downstream expand_floquet_TWPA dispatch
        workspace_vars['use_linear_in_window'] = False

        # Resolve <X>_A from explicit <X>_A or <X>_uA, defaulting Id_A to 0.0
        # so KI Taylor coefficients can be computed even when no bias is stored.
        def _resolve_A(key_A, key_uA, default_A=None):
            v = params.get(key_A)
            if v is not None:
                return v
            v_uA = params.get(key_uA)
            if v_uA is not None:
                return v_uA * 1e-6
            return default_A

        Z0_env = 50  # environment impedance
        Z0_TWPA = params.get('Z0_TWPA_ohm', 50)
        LTLsec_H_center = params.get('LTLsec_H', L0_H)
        g_L = np.sqrt(2)
        g_C = np.sqrt(2)
        fc_TLsec_center = params.get('fc_TLsec_GHz',
                                     g_L * Z0_TWPA / (2 * np.pi * LTLsec_H_center) * 1e-9)

        taper = compute_taper_arrays(
            Ntot_cell=Ntot_cell, Ncpersc_cell=Ncpersc_cell,
            Z_taper=params.get('Z_taper', False),
            Z_taper_width=params.get('Z_taper_width', 0.3),
            Z_profile=params.get('Z_profile', 'linear'),
            klopfenstein_A=params.get('klopfenstein_A', None),
            floquet_taper=params.get('floquet_taper', False),
            floquet_taper_width=params.get('floquet_taper_width', 0.3),
            floquet_profile=params.get('floquet_profile', 'gaussian'),
            taper_cutoff=params.get('taper_cutoff', False),
            Z0_ohm=Z0_env, Z0_TWPA_ohm=Z0_TWPA,
            fc_TLsec_GHz=fc_TLsec_center,
            LTLsec_H_center=LTLsec_H_center, L0_H_center=L0_H,
            g_L=g_L, g_C=g_C,
            nonlinearity=nonlinearity,
            jj_structure_type=params.get('jj_structure_type', 'jj'),
            phi_dc=params.get('phi_dc', 0),
            beta_L=params.get('beta_L'),
            LJ0_H=params.get('LJ0_H'),
            Lg_H=params.get('Lg_H'),
            CJ_F=params.get('CJ_F'),
            Ic_JJ_A=_resolve_A('Ic_JJ_A', 'Ic_JJ_uA'),
            Istar_A=_resolve_A('Istar_A', 'Istar_uA'),
            Id_A=_resolve_A('Id_A', 'Id_uA', default_A=0.0),
            L0_pH=params.get('L0_pH'),
            n_jj_struct=params.get('n_jj_struct', 1),
            LinfLF1_H=params.get('LinfLF1_H'),
            L0LF2_H=params.get('L0LF2_H'),
            Foster_form_L=Foster_form_L,
            Foster_form_C=Foster_form_C,
            zero_at_zero=params.get('zero_at_zero', True),
            select_one_form=params.get('select_one_form', 'C'),
            f_zeros_GHz=params.get('f_zeros_GHz', np.array([])),
            f_poles_GHz=params.get('f_poles_GHz', np.array([])),
            dispersion_type=dispersion_type,
            g_C_mod=params.get('g_C_mod'),
            rf_squid_constant_plasma=params.get('rf_squid_constant_plasma', False),
        )

        workspace_vars['floquet_weights'] = taper['w_percell']
        workspace_vars['floquet_center_start'] = taper['center_start']
        workspace_vars['floquet_center_end'] = taper['center_end']
        workspace_vars['floquet_cell_params'] = taper['floquet_cell_params']
        if nonlinearity == 'JJ' and jj_structure_type == 'rf_squid':
            workspace_vars['Lg_H_value'] = params.get('Lg_H')

        if taper['linear_varies']:
            params['CTLsec_F'] = taper['CTLsec_F']
            params['LTLsec_H_percell'] = taper['LTLsec_H_percell']
            workspace_vars['floquet_filter_components'] = taper['floquet_filter_components']

    # Create and configure the builder
    builder = JCNetlistBuilder()
    builder.load_workspace_data(workspace_vars)
    
    # Set JJ parameters if applicable
    if nonlinearity == 'JJ':
        Ic_JJ_uA = params.get('Ic_JJ_uA', None)
        CJ_F = params.get('CJ_F', np.inf)
        cj_value = CJ_F if CJ_F != np.inf else None
        builder.set_jj_parameters(Ic_JJ_uA, cj_value)
    
    # Return everything needed for the next steps
    return {
        'params': params,  # All parameters with updates
        'workspace_vars': workspace_vars,
        'builder': builder,
        'Ntot_cell_init': Ntot_cell_init,
        'netlist_has_loss': netlist_config.enable_dielectric_loss,
        'netlist_loss_tangent': netlist_config.loss_tangent,
    }

#####################################################################################

def create_output_filename(device_name: str, ntot_cells: int,
                          folder: Optional[str] = None) -> str:
    """Generate output filename with auto-increment (Cell 5 from notebook).
    
    Parameters
    ----------
    device_name : str
        Device name (e.g., 'b_jtwpa')
    ntot_cells : int
        Total number of cells
    folder : str
        Output folder (default: 'netlists')
        
    Returns
    -------
    str
        Full path to output file with incremented number
    """
    # Import filecounter here instead of nested import
    from .helper_functions import filecounter
    
    # Use package default if not specified
    if folder is None:
        folder = str(NETLISTS_DIR)
    
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    print(f"✓ Netlists folder: {os.path.abspath(folder)}")
    
    # Create filename pattern for this specific configuration
    base_pattern = f'{device_name}_{ntot_cells}cells_*.py'
    full_pattern = os.path.join(folder, base_pattern)
    
    # Get the next available number
    filename, file_number = filecounter(full_pattern)
    
    print(f"  Output file: {os.path.basename(filename)}")
    
    return filename


#####################################################################################

def build_netlist(prepared_data: Dict) -> Tuple[JCNetlistBuilder, Dict[str, Any]]:
    """Build netlist (Cell 6 from notebook).
    
    Parameters
    ----------
    prepared_data : dict
        Output from prepare_workspace_variables()
        'forw' or 'back'
        
    Returns
    -------
    tuple
        (builder, stats) - builder with completed netlist and statistics
    """
    print(f"\nBuilding netlist...")
    
    # Extract what we need
    params = prepared_data['params']
    workspace_vars = prepared_data['workspace_vars']
    builder = prepared_data['builder']
    
    # Reset builder
    builder.reset()
    
    # Re-load workspace data and JJ parameters after reset
    builder.load_workspace_data(workspace_vars)
    
    # Set JJ parameters if applicable
    if params['nonlinearity'] == 'JJ':
        ic_value = params.get('Ic_JJ_uA', None)
        cj_value = params.get('CJ_F', None)
        builder.set_jj_parameters(ic_value, cj_value)
    
    # Use builder's node counter for input node
    input_node = str(builder.node_counter)
    builder.node_counter += 1
    
    # Add input port and resistance
    builder.add_port(input_node, "1")
    builder.add_resistor(f"R{input_node}_0", input_node, "0", 50.0, "R_port")
    
    current_node = input_node  # Start from input node
    
    # Extract parameters needed for building
    dispersion_type = params['dispersion_type']
    window_type = params.get('window_type', 'boxcar')
    Nsc_cell = params['Nsc_cell']
    Ncpersc_cell = params['Ncpersc_cell']
    width = params.get('width', 0)
    ind_g_C_with_filters = params.get('ind_g_C_with_filters', [])
    nonlinearity = params['nonlinearity']

    # All the other parameters needed by expand_supercell_inline
    LTLsec_rem_H = params.get('LTLsec_rem_H', 0)
    Lg_H = params.get('Lg_H', np.inf)
    L0_H = params.get('L0_H', None)

    # Validate L0_H for KI devices
    if nonlinearity == 'KI' and (L0_H is None or L0_H == 0):
        raise ValueError("L0_H must be defined and non-zero for KI (kinetic inductance) nonlinearity. "
                       "Check that 'L0_H' is present in the 'circuit' dict of your design file.")
    epsilon_perA = params.get('epsilon_perA', 0)
    xi_perA2 = params.get('xi_perA2', 0)
    LTLsec_H = params.get('LTLsec_H', None)

    # For KI devices, if LTLsec_H is not specified, use L0_H as the transmission line inductance
    if nonlinearity == 'KI' and (LTLsec_H is None or LTLsec_H == 0):
        LTLsec_H = L0_H
    CTLsec_F = params.get('CTLsec_F', 0)
    ngL = params.get('ngL', 1)
    ngC = params.get('ngC', 1)
    Foster_form_L = params.get('Foster_form_L', 1)
    Foster_form_C = params.get('Foster_form_C', 1)
    n_jj_struct = params.get('n_jj_struct', 1)
    n_poles = params.get('n_poles', 0)
    n_zeros = params.get('n_zeros', 0)
    n_filters_per_sc = params.get('n_filters_per_sc', 0)
    n_periodic_sc = params.get('n_periodic_sc', Nsc_cell)
    n_periodic_sc_init = params.get('n_periodic_sc_init', Nsc_cell)
    
    # Get filter arrays
    LinfLF1_H = params.get('LinfLF1_H', [])
    LinfLF1_rem_H = params.get('LinfLF1_rem_H', [])
    C0LF1_F = params.get('C0LF1_F', [])
    LiLF1_H = params.get('LiLF1_H', [])
    CiLF1_F = params.get('CiLF1_F', [])
    L0LF2_H = params.get('L0LF2_H', [])
    L0LF2_rem_H = params.get('L0LF2_rem_H', [])
    CinfLF2_F = params.get('CinfLF2_F', [])
    LiLF2_H = params.get('LiLF2_H', [])
    CiLF2_F = params.get('CiLF2_F', [])
    LinfCF1_H = params.get('LinfCF1_H', [])
    C0CF1_F = params.get('C0CF1_F', [])
    LiCF1_H = params.get('LiCF1_H', [])
    CiCF1_F = params.get('CiCF1_F', [])
    L0CF2_H = params.get('L0CF2_H', [])
    CinfCF2_F = params.get('CinfCF2_F', [])
    LiCF2_H = params.get('LiCF2_H', [])
    CiCF2_F = params.get('CiCF2_F', [])
    
    # Build the main structure. Either taper triggers per-cell numeric expansion.
    is_floquet = params.get('floquet_taper', False) or params.get('Z_taper', False)
    Ntot_cell = params.get('Ntot_cell', Nsc_cell * Ncpersc_cell)

    if is_floquet:
        current_node = builder.expand_floquet_TWPA(
            current_node, Ntot_cell, Ncpersc_cell, width,
            ind_g_C_with_filters, nonlinearity,
            LTLsec_rem_H, Lg_H, L0_H, epsilon_perA, xi_perA2,
            params.get('CJ_F'), LTLsec_H, CTLsec_F, params.get('Ic_JJ_uA'), ngL, ngC,
            Foster_form_L, Foster_form_C, n_jj_struct, n_poles, n_zeros,
            LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
            L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
            LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
            L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
            n_periodic_sc, n_filters_per_sc, n_periodic_sc_init, dispersion_type)
    elif dispersion_type == 'filter' or ((dispersion_type == 'periodic' or dispersion_type == 'both') and window_type.lower() == 'boxcar'):
        # Expand supercells inline
        for i in range(1, Nsc_cell + 1):
            current_node = builder.expand_supercell_inline(
                i, current_node, Ncpersc_cell, width,
                ind_g_C_with_filters, nonlinearity,
                LTLsec_rem_H, Lg_H, L0_H, epsilon_perA,
                xi_perA2, params.get('CJ_F'), LTLsec_H, CTLsec_F,
                params.get('Ic_JJ_uA'), ngL, ngC, Foster_form_L,
                Foster_form_C, n_jj_struct, n_poles, n_zeros,
                LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
                L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
                n_filters_per_sc, dispersion_type)
    else:
        # For periodicfiltered
        current_node = builder.expand_periodicfiltered_TWPA(
            current_node, Nsc_cell, Ncpersc_cell, width,
            ind_g_C_with_filters, nonlinearity,
            LTLsec_rem_H, Lg_H, L0_H, epsilon_perA, xi_perA2,
            params.get('CJ_F'), LTLsec_H, CTLsec_F, params.get('Ic_JJ_uA'), ngL, ngC,
            Foster_form_L, Foster_form_C, n_jj_struct, n_poles, n_zeros,
            LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
            L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
            LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,
            L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,
            n_periodic_sc, n_filters_per_sc, n_periodic_sc_init, dispersion_type)
    
    # Add output port and resistance
    output_node = current_node
    builder.add_resistor(f"R{output_node}_0", output_node, "0", 50.0, "R_port")
    builder.add_port(output_node, "2")
    
    print(f"  Built netlist with {len(builder.components)} components")
    print(f"  Parameters: {len(builder.circuit_parameters)}")
    
    # Get statistics
    stats = builder.get_statistics()
    
    return builder, stats


def save_raw_netlist_to_file(jc_components: list, circuit_parameters: dict,
                            metadata: Dict[str, Any], output_file: str):
    """Save raw netlist data to a Python file readable by julia_wrapper.

    This is the low-level save function that writes jc_components,
    circuit_parameters, and metadata to a .py file. It can be called
    directly with raw data (e.g. from filter_builder.compose_chain)
    or indirectly via save_netlist_to_file (which extracts data from
    a JCNetlistBuilder first).

    Parameters
    ----------
    jc_components : list
        List of (name, node1, node2, value) tuples.
    circuit_parameters : dict
        Parameter name -> value mapping.
    metadata : dict
        Metadata to include in file (device_name, etc.).
    output_file : str
        Output filename (full path).
    """
    from datetime import datetime

    metadata['component_count'] = len(jc_components)
    metadata['parameter_count'] = len(circuit_parameters)
    metadata['generated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(output_file, 'w') as f:
        # Write header comment
        f.write(f"# JC netlist for {metadata.get('device_name', 'TWPA')}\n")
        f.write(f"# Generated: {metadata['generated']}\n")
        f.write(f"# Components: {len(jc_components)}\n")
        f.write(f"# Parameters: {len(circuit_parameters)}\n\n")

        # Write components
        f.write("jc_components = [\n")
        for comp in jc_components:
            f.write(f'    {comp},\n')
        f.write("]\n\n")

        # Write parameters
        f.write("circuit_parameters = {\n")
        for param, value in sorted(circuit_parameters.items()):
            # Check if this is a capacitor parameter and loss is enabled
            # Exclude Cj (junction capacitance) since it's not a dielectric capacitor
            if (param.startswith('C') and param not in ['c1', 'c2', 'c3', 'c4', 'Cj'] and
                metadata.get('dielectric_loss_enabled', False)):
                # Write as complex value in Julia-compatible format
                loss_tan = metadata.get('loss_tangent', 0.0)
                if isinstance(value, (int, float)):
                    f.write(f'    "{param}": "{value:.6e}/(1+{loss_tan:.6e}im)",\n')
                else:
                    f.write(f'    "{param}": "{value}/(1+{loss_tan:.6e}im)",\n')
            else:
                if isinstance(value, (int, float)):
                    f.write(f'    "{param}": {value:.6e},\n')
                elif isinstance(value, np.ndarray):
                    if value.size == 1:
                        f.write(f'    "{param}": {float(value):.6e},\n')
                    else:
                        f.write(f'    "{param}": {value.tolist()},\n')
                else:
                    f.write(f'    "{param}": {repr(value)},\n')
        f.write("}\n\n")

        # Write metadata
        f.write("metadata = {\n")
        for key, value in metadata.items():
            if isinstance(value, str):
                f.write(f'    "{key}": "{value}",\n')
            else:
                f.write(f'    "{key}": {value},\n')
        f.write("}\n")

    print(f"✓ Saved netlist: {output_file}")
    print(f"  Components: {len(jc_components)}")
    print(f"  Parameters: {len(circuit_parameters)}")
    print(f"  Metadata keys: {list(metadata.keys())}")


def save_netlist_to_file(builder: JCNetlistBuilder, output_file: str, metadata: Dict[str, Any]):
    """Save netlist from a JCNetlistBuilder to Python file.

    Parameters
    ----------
    builder : JCNetlistBuilder
        Builder with completed netlist
    output_file : str
        Output filename (full path)
    metadata : dict
        Metadata to include in file (device_name, etc.)
    """
    jc_components = builder.get_netlist_tuples()
    circuit_parameters = builder.get_used_parameters()
    save_raw_netlist_to_file(jc_components, circuit_parameters, metadata, output_file)


def build_netlist_from_config(netlist_config: NetlistConfig) -> str:
    """High-level function to build netlist from configuration.
    
    This runs the complete netlist building workflow from notebook 2.
    
    Parameters
    ----------
    netlist_config : NetlistConfig
        Configuration object with all settings
        
    Returns
    -------
    str
        Path to generated netlist file
    """
    print(f"=== Building Netlist from {netlist_config.design_file} ===\n")
    
    # Step 1: Check input file exists
    netlist_config.check_input_file()
    
    # Step 2: Load design parameters
    design_data = load_design_parameters(netlist_config.design_path)
    
    # Step 3: Prepare workspace variables
    prepared_data = prepare_workspace_variables(design_data, netlist_config)
    
    # Extract key info for output file
    device_name = prepared_data['params']['device_name']
    ntot_cells = prepared_data['params']['Ntot_cell']
    
    # Step 4: Create output filename
    output_file = create_output_filename(device_name, ntot_cells, folder=netlist_config.output_dir)
    
    # Step 5: Build netlist
    builder, stats = build_netlist(prepared_data)
    
    # Step 6: Prepare metadata
    metadata = {
        'device_name': device_name,
        'total_cells': ntot_cells,
        'cells_per_supercell': prepared_data['params']['Ncpersc_cell'],
        'num_supercells': prepared_data['params']['Nsc_cell'],
        'dispersion_type': prepared_data['params']['dispersion_type'],
        'nonlinearity': prepared_data['params']['nonlinearity'],
        'dielectric_loss_enabled': netlist_config.enable_dielectric_loss,
        'use_taylor_insteadof_JJ': netlist_config.use_taylor_insteadof_JJ,
    }
    
    # Add optional metadata
    dispersion_type = prepared_data['params'].get('dispersion_type', 'filter')
    
    # Filter-specific metadata  
    if dispersion_type in ['filter', 'both']:
        if 'f_zeros_GHz' in prepared_data['params']:
            zeros = prepared_data['params']['f_zeros_GHz']
            # Convert numpy arrays to lists for JSON compatibility
            if hasattr(zeros, 'tolist'):
                metadata['filter_zeros_GHz'] = zeros.tolist()
            else:
                metadata['filter_zeros_GHz'] = zeros
                
        if 'f_poles_GHz' in prepared_data['params']:
            poles = prepared_data['params']['f_poles_GHz']
            # Convert numpy arrays to lists for JSON compatibility
            if hasattr(poles, 'tolist'):
                metadata['filter_poles_GHz'] = poles.tolist()
            else:
                metadata['filter_poles_GHz'] = poles
                
        if 'nTLsec' in prepared_data['params']:
            metadata['TL_sections_per_supercell'] = prepared_data['params']['nTLsec']

    # Periodic modulation metadata
    if dispersion_type in ['periodic', 'both']:
        metadata['window_type'] = prepared_data['params'].get('window_type', 'boxcar')
        if 'width' in prepared_data['params']:
            metadata['window_width'] = prepared_data['params']['width']
        if 'stopbands_config_GHz' in prepared_data['params']:
            metadata['stopbands_config_GHz'] = prepared_data['params']['stopbands_config_GHz']

    # Loss parameters
    if netlist_config.enable_dielectric_loss:
        metadata['loss_tangent'] = netlist_config.loss_tangent
    
    # Nonlinearity-specific metadata
    if prepared_data['params']['nonlinearity'] == 'JJ':
        metadata['critical_current_uA'] = prepared_data['params'].get('Ic_JJ_uA')
        metadata['jj_structure_type'] = prepared_data['params'].get('jj_structure_type', 'rf_squid')
    elif prepared_data['params']['nonlinearity'] == 'KI':
        metadata['scaling_current_uA'] = prepared_data['params'].get('Istar_uA')
        metadata['inductance_per_cell_pH'] = prepared_data['params'].get('L0_pH')
    
    # Step 7: Save netlist
    save_netlist_to_file(builder, output_file, metadata)
    
    print(f"\n✓ Netlist building complete!")
    
    return output_file