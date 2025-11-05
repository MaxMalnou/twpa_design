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
        """Create symbolic parameter and store value"""
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
    
    def add_bare_jj(self, component_name, node1, node2, n_jj_struct, use_taylor=False):
        """
        Add bare Josephson junction(s) or Taylor equivalent (no geometric inductance).
        Each structure includes its CJ capacitor.
        
        Parameters:
        - component_name: Base name for components
        - node1, node2: Circuit nodes
        - n_jj_struct: Number of JJs in series
        - use_taylor: Use Taylor expansion if True
        """
        if use_taylor:
            # Use Taylor expansion for bare JJ(s)
            L0_H = self.workspace_data.get('L0_H')
            if L0_H is None:
                raise ValueError("L0_H must be defined for Taylor expansion")
            
            # Use the same shared L0 parameter
            if 'L0' not in self.circuit_parameters:
                self.circuit_parameters['L0'] = L0_H
            
            # Create the NL element with Taylor expansion
            taylor_str = self.get_taylor_poly_string()
            
            if n_jj_struct == 1:
                # Single NL element with CJ
                nl_name = self.make_unique_name(f"NL{component_name}")
                self.components.append(JCComponent(nl_name, str(node1), str(node2), taylor_str))
                
                # Add CJ in parallel with the NL element
                if self.Cj_value is not None and self.Cj_value > 0:
                    cj_name = self.make_unique_name(f"Cj{component_name}")
                    self.components.append(JCComponent(cj_name, str(node1), str(node2), 'Cj'))
            else:
                # Multiple NL elements in series, each with CJ
                current_node = node1
                for i in range(n_jj_struct):
                    next_node = self.get_new_node() if i < n_jj_struct - 1 else node2
                    
                    # Add NL element
                    nl_name = self.make_unique_name(f"NL{component_name}_{i+1}")
                    self.components.append(JCComponent(nl_name, str(current_node), str(next_node), taylor_str))
                    
                    # Add CJ in parallel with this NL element
                    if self.Cj_value is not None and self.Cj_value > 0:
                        cj_name = self.make_unique_name(f"Cj{component_name}_{i+1}")
                        self.components.append(JCComponent(cj_name, str(current_node), str(next_node), 'Cj'))
                    
                    current_node = next_node
        else:
            # Traditional bare JJ(s)
            if n_jj_struct == 1:
                # Single JJ with its capacitor
                lj_name = self.make_unique_name(f"Lj{component_name}")
                self.components.append(JCComponent(lj_name, str(node1), str(node2), 'Lj'))
                
                # Add shunt capacitance if it exists
                if self.Cj_value is not None and self.Cj_value > 0:
                    cj_name = self.make_unique_name(f"Cj{component_name}")
                    self.components.append(JCComponent(cj_name, str(node1), str(node2), 'Cj'))
            else:
                # Multiple JJs in series, each with its own capacitor
                current_node = node1
                for i in range(n_jj_struct):
                    next_node = self.get_new_node() if i < n_jj_struct - 1 else node2
                    
                    # Add the JJ
                    lj_name = self.make_unique_name(f"Lj{component_name}_{i+1}")
                    self.components.append(JCComponent(lj_name, str(current_node), str(next_node), 'Lj'))
                    
                    # Add shunt capacitance for this JJ
                    if self.Cj_value is not None and self.Cj_value > 0:
                        cj_name = self.make_unique_name(f"Cj{component_name}_{i+1}")
                        self.components.append(JCComponent(cj_name, str(current_node), str(next_node), 'Cj'))
                    
                    current_node = next_node

    
    def add_rf_squid(self, component_name, node1, node2, n_jj_struct, Lg_H, use_taylor=False):
        """
        Add rf-SQUID(s) (JJ in parallel with geometric inductance) or Taylor equivalent.
        Each rf-SQUID unit includes its own CJ capacitor.
        
        Parameters:
        - component_name: Base name for the components
        - node1, node2: Circuit nodes
        - n_jj_struct: Number of rf-SQUIDs in series
        - Lg_H: Geometric inductance in Henries (None or np.inf means no Lg)
        - use_taylor: If True, use Taylor expansion instead of JJ
        
        Returns: None (components are added to self.components)
        """
        # Check if we actually have a geometric inductance
        has_Lg = Lg_H is not None and not np.isinf(Lg_H) and Lg_H > 0
        
        if use_taylor:
            # Use Taylor expansion for rf-SQUID(s)
            L0_H = self.workspace_data.get('L0_H')
            if L0_H is None:
                raise ValueError("L0_H must be defined to use Taylor expansion for rf-SQUID")
            
            # Set up Taylor coefficients
            if 'L0' not in self.circuit_parameters:
                self.circuit_parameters['L0'] = L0_H
            
            # Create the NL element with Taylor expansion
            taylor_str = self.get_taylor_poly_string()
            
            if n_jj_struct == 1:
                # Single NL element (represents Lj||Lg) with CJ
                nl_name = self.make_unique_name(f"NL{component_name}")
                self.components.append(JCComponent(nl_name, str(node1), str(node2), taylor_str))
                
                # Add CJ in parallel with the NL element
                if self.Cj_value is not None and self.Cj_value > 0:
                    cj_name = self.make_unique_name(f"Cj{component_name}")
                    self.components.append(JCComponent(cj_name, str(node1), str(node2), 'Cj'))
            else:
                # Multiple NL elements in series, each with CJ
                current_node = node1
                for i in range(n_jj_struct):
                    next_node = self.get_new_node() if i < n_jj_struct - 1 else node2
                    
                    # Add NL element (represents one rf-SQUID)
                    nl_name = self.make_unique_name(f"NL{component_name}_{i+1}")
                    self.components.append(JCComponent(nl_name, str(current_node), str(next_node), taylor_str))
                    
                    # Add CJ in parallel with this NL element
                    if self.Cj_value is not None and self.Cj_value > 0:
                        cj_name = self.make_unique_name(f"Cj{component_name}_{i+1}")
                        self.components.append(JCComponent(cj_name, str(current_node), str(next_node), 'Cj'))
                    
                    current_node = next_node
        else:
            # Traditional rf-SQUID: JJ(s) in parallel with Lg
            if n_jj_struct == 1:
                # Single rf-SQUID unit
                # Add JJ
                lj_name = self.make_unique_name(f"Lj{component_name}")
                self.components.append(JCComponent(lj_name, str(node1), str(node2), 'Lj'))
                
                # Add geometric inductance in parallel if it exists
                if has_Lg:
                    lg_name = self.make_unique_name(f"Lg{component_name}")
                    lg_symbol = self.create_symbolic_value(Lg_H, 'L', f"Lg{component_name}")
                    self.components.append(JCComponent(lg_name, str(node1), str(node2), lg_symbol))
                
                # Add shunt capacitance for this rf-SQUID
                if self.Cj_value is not None and self.Cj_value > 0:
                    cj_name = self.make_unique_name(f"Cj{component_name}")
                    self.components.append(JCComponent(cj_name, str(node1), str(node2), 'Cj'))
            else:
                # Multiple rf-SQUIDs in series
                current_node = node1
                for i in range(n_jj_struct):
                    next_node = self.get_new_node() if i < n_jj_struct - 1 else node2
                    
                    # Add JJ for this rf-SQUID
                    lj_name = self.make_unique_name(f"Lj{component_name}_{i+1}")
                    self.components.append(JCComponent(lj_name, str(current_node), str(next_node), 'Lj'))
                    
                    # Add geometric inductance in parallel if it exists
                    if has_Lg:
                        lg_name = self.make_unique_name(f"Lg{component_name}_{i+1}")
                        lg_symbol = self.create_symbolic_value(Lg_H, 'L', f"Lg{component_name}_{i+1}")
                        self.components.append(JCComponent(lg_name, str(current_node), str(next_node), lg_symbol))
                    
                    # Add shunt capacitance for this rf-SQUID
                    if self.Cj_value is not None and self.Cj_value > 0:
                        cj_name = self.make_unique_name(f"Cj{component_name}_{i+1}")
                        self.components.append(JCComponent(cj_name, str(current_node), str(next_node), 'Cj'))
                    
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
        
    def add_jj_structure(self, component_name, node1, node2, structure_params):
        """
        Dispatcher method for different JJ structures.
        
        Parameters:
        - component_name: Base name for components
        - node1, node2: Circuit nodes
        - structure_params: Dict with structure-specific parameters including 'type'
        """
        structure_type = structure_params.get('type', 'rf_squid')
        use_taylor = self.workspace_data.get('use_taylor_insteadof_JJ', False)
        
        if structure_type == 'jj':
            # Bare JJ - no geometric inductance
            self.add_bare_jj(
                component_name, node1, node2,
                structure_params.get('n_jj_struct', 1),
                use_taylor
            )
        elif structure_type == 'rf_squid':
            self.add_rf_squid(
                component_name, node1, node2,
                structure_params.get('n_jj_struct', 1),
                structure_params.get('Lg_H'),
                use_taylor
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
                      Lg_H, epsilon_perA, xi_perA2, n_jj_struct):
        """Add inductance to JC netlist (handles JJ, KI, and linear inductances)"""
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
            self.add_jj_structure(component_name, node1, main_output_node, structure_params)
            
            # CJ capacitors are now added within add_bare_jj and add_rf_squid
                    
        elif nonlinearity == 'KI':
            # Create NL (nonlinear) inductor with Taylor coefficients
            nl_name = self.make_unique_name(f"NL{component_name}")
            
            # Use shared parameters for all NL inductors
            if 'L0' not in self.circuit_parameters:
                self.circuit_parameters['L0'] = L_H
            
            # Add all available Taylor coefficients
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
            
            # Check for c3 and c4
            if 'c3' not in self.circuit_parameters and 'c3_taylor' in self.workspace_data:
                self.circuit_parameters['c3'] = self.workspace_data['c3_taylor']
            
            if 'c4' not in self.circuit_parameters and 'c4_taylor' in self.workspace_data:
                self.circuit_parameters['c4'] = self.workspace_data['c4_taylor']
            
            # Use the same method as JJ structures to build the poly string
            taylor_str = self.get_taylor_poly_string()
            
            # Add the NL component
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
    
    
    def add_capacitor(self, component_name, node1, node2, cap_value, Ncpersc_cell=None, 
                     is_windowed=False, ind_g_C_with_filters=None, dispersion_type=None):
        """Add capacitor to JC netlist"""
        c_name = self.make_unique_name(f"C{component_name}")        
        
        # For windowed cells, write value directly inline
        if is_windowed and 'TLsec' in component_name:
            # Write numeric value directly
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
                            Ncpersc_cell=None, ind_g_C_with_filters=None, dispersion_type=None):
        """Add Foster 1 L stage components inline and return output node"""
        
        current_node = start_node
        p = 1
        
        # Handle LinfLF1
        if check_flat(LinfLF1_H, k_idx) != 0:
            if check_flat(C0LF1_F, k_idx) != np.inf or (n_poles > 0 and CiLF1_F[k_idx, n_poles-1] != np.inf):
                # Create intermediate node LF1_1_k
                next_node = self.get_new_node()
                self.add_inductance(f'infLF1_{k}', current_node, next_node, nonlinearity,
                                  check_flat(LinfLF1_H, k_idx), check_flat(LinfLF1_rem_H, k_idx),
                                  Lg_H, epsilon_perA, xi_perA2, n_jj_struct)
                current_node = next_node
            else:
                # Connects directly to output
                output_node = self.get_new_node()
                self.add_inductance(f'infLF1_{k}', current_node, output_node, nonlinearity,
                                  check_flat(LinfLF1_H, k_idx), check_flat(LinfLF1_rem_H, k_idx),
                                  Lg_H, epsilon_perA, xi_perA2, n_jj_struct)
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
                           Ncpersc_cell=None, ind_g_C_with_filters=None, dispersion_type=None):
        """Add Foster 2 L stage components inline and return output node"""
        
        current_node = start_node
        output_node = self.get_new_node()
        
        # Handle L0LF2 - main series inductance
        self.add_inductance(f'0LF2_{k}', current_node, output_node, nonlinearity,
                          check_flat(L0LF2_H, k_idx), check_flat(L0LF2_rem_H, k_idx),
                          Lg_H, epsilon_perA, xi_perA2, n_jj_struct)
        
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
            if check_flat(C0CF1_F, k_idx) != np.inf or CiCF1_F[k_idx, n_zeros-1] != np.inf:
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
            if CiCF1_F[k_idx, n_zeros-1] != np.inf:
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
        if CiCF1_F[k_idx, n_zeros-1] != np.inf:
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
                           Ncpersc_cell=None, ind_g_C_with_filters=None, dispersion_type=None):
        """Add filter stages inline and return output node"""
        
        current_node = start_node
        
        # Series filter (changes the node)
        if Foster_form_L == 1:
            # Use local k_idx for series filter
            series_k_idx = k_idx
            if len(LinfLF1_H) == 1:
                series_k_idx = 0
            current_node = self.add_foster1_L_stage(current_node, k, series_k_idx, nonlinearity, n_jj_struct, n_poles,
                                                  LinfLF1_H, LinfLF1_rem_H, C0LF1_F, LiLF1_H, CiLF1_F,
                                                  Lg_H, epsilon_perA, xi_perA2,
                                                  Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
        else:
            series_k_idx = k_idx
            if len(L0LF2_H) == 1:
                series_k_idx = 0
            current_node = self.add_foster2_L_stage(current_node, k, series_k_idx, nonlinearity, n_jj_struct, n_zeros,
                                                  L0LF2_H, L0LF2_rem_H, CinfLF2_F, LiLF2_H, CiLF2_F,
                                                  Lg_H, epsilon_perA, xi_perA2,
                                                  Ncpersc_cell, ind_g_C_with_filters, dispersion_type)
        
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
    
    # Calculate periodic supercells based on window type
    if dispersion_type == 'both' and window_type.lower() == 'tukey':
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
    
    # Build the main structure
    if dispersion_type == 'filter' or ((dispersion_type == 'periodic' or dispersion_type == 'both') and window_type.lower() == 'boxcar'):
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


def save_netlist_to_file(builder: JCNetlistBuilder, output_file: str, metadata: Dict[str, Any]):
    """Save netlist to Python file (Cell 7 from notebook).
    
    Parameters
    ----------
    builder : JCNetlistBuilder
        Builder with completed netlist
    output_file : str
        Output filename (full path)
    metadata : dict
        Metadata to include in file (device_name, etc.)
    """
    from datetime import datetime
    
    # Get netlist data from builder
    jc_components = builder.get_netlist_tuples()
    circuit_parameters = builder.get_used_parameters()  # Only get parameters actually used
    
    # Add component and parameter counts to metadata
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
            if (param.startswith('C') and param not in ['c1', 'c2', 'c3', 'c4'] and 
                metadata.get('dielectric_loss_enabled', False)):
                # Write as complex value in Julia-compatible format
                loss_tan = metadata.get('loss_tangent', 0.0)
                if isinstance(value, (int, float)):
                    # Format as Julia complex number: value/(1+im*tandelta)
                    f.write(f'    "{param}": "{value:.6e}/(1+{loss_tan:.6e}im)",\n')
                else:
                    # Handle array values if needed
                    f.write(f'    "{param}": "{value}/(1+{loss_tan:.6e}im)",\n')
            else:
                # Non-capacitor parameters or loss disabled
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