# Netlist JC Builder Module - Complete Reference

This module converts TWPA designs into netlists compatible with JosephsonCircuits.jl for harmonic balance simulations.

## Quick Start

```python
from twpa_design.netlist_JC_builder import NetlistConfig, build_netlist_from_config

# Simple usage - build netlist from design file
config = NetlistConfig(design_file='b_jtwpa_01.py')
output_file = build_netlist_from_config(config)
print(f"Netlist saved to: {output_file}")
```

## Classes

### NetlistConfig

Configuration dataclass for netlist building process.

#### Constructor with All Parameters
```python
config = NetlistConfig(
    # Required
    design_file="b_jtwpa_01.py",      # str: Design file from designs/ folder
    
    # Optional - all have defaults
    use_taylor_insteadof_JJ=False,    # bool: Use Taylor expansion (default: False)
    enable_dielectric_loss=False,     # bool: Add dielectric loss (default: False)
    loss_tangent=2e-4,                # float: tan(δ) value (default: 2e-4)
    use_linear_in_window=True,        # bool: Linear elements in apodization (default: True)
    Ntot_cell_override=None           # int: Override total cells (default: None)
)
```

#### Parameters Explained

- **design_file**: Name of the design file in the designs/ folder (required)
- **use_taylor_insteadof_JJ**: 
  - `False`: Use JosephsonCircuits.jl hardcoded JJ potential (more accurate)
  - `True`: Use Taylor expansion with c1, c2, c3, c4 coefficients
- **enable_dielectric_loss**: Add loss to all capacitors as complex admittance
- **loss_tangent**: Dielectric loss tangent tan(δ), typical values 1e-4 to 1e-3
- **use_linear_in_window**: In windowed periodic structures, use linear L in apodization regions
- **Ntot_cell_override**: Force different number of cells than design specifies

---

### JCNetlistBuilder

Main class that performs the netlist conversion.

#### Constructor
```python
builder = JCNetlistBuilder(design_params, build_config=None)
```
- `design_params` (dict): Parameters from the design file
- `build_config` (NetlistConfig): Optional configuration object

#### Primary Methods

##### build_flattened_netlist
```python
components, parameters, metadata = builder.build_flattened_netlist()
```
Main method that builds the complete netlist.

Returns tuple of:
- `components`: List of (name, node1, node2, value) tuples
- `parameters`: Dict of circuit parameters
- `metadata`: Dict with build information

##### save_flattened_netlist
```python
filename = builder.save_flattened_netlist(
    components,          # List of component tuples
    parameters,          # Dict of parameters
    metadata=None,       # Optional metadata dict
    output_dir=None,     # Output directory (default: netlists/)
    filename=None        # Explicit filename (default: auto-generate)
)
```
Saves the netlist to a Python file. Returns the output filename.

#### Component Building Methods

##### build_unit_cell
```python
cell_components = builder.build_unit_cell(
    cell_num,           # int: Cell number (1-indexed)
    modulation_factor=1.0  # float: Modulation for periodic structures
)
```
Builds components for a single unit cell.

##### build_nonlinear_element
```python
nl_components = builder.build_nonlinear_element(
    cell_num,           # int: Cell number
    node_left,          # int: Left node number
    node_right,         # int: Right node number
    element_num=1,      # int: Element number within cell
    modulation_factor=1.0  # float: Modulation factor
)
```
Creates nonlinear element(s) based on the nonlinearity type.

##### build_filter_section
```python
filter_components = builder.build_filter_section(
    cell_num,           # int: Cell number
    node_left,          # int: Left node
    node_right,         # int: Right node
    filter_type='LF1'   # str: Filter type (LF1, LF2, CF1, CF2)
)
```
Builds filter components for dispersion engineering.

##### build_transmission_line_section
```python
tl_components = builder.build_transmission_line_section(
    start_cell,         # int: Starting cell number
    end_cell,           # int: Ending cell number
    node_left,          # int: Left node
    node_right          # int: Right node
)
```
Builds regular transmission line sections between filters.

#### Helper Methods

##### get_modulation_profile
```python
profile = builder.get_modulation_profile()  # Returns numpy array
```
Calculates the spatial modulation profile for periodic structures.

##### apply_dielectric_loss
```python
lossy_value = builder.apply_dielectric_loss(
    capacitance,        # float: Capacitance value
    loss_tangent=None   # float: Override default loss tangent
)
```
Adds dielectric loss to a capacitance value. Returns complex admittance string.

##### validate_design
```python
is_valid = builder.validate_design()  # Returns bool
```
Validates that the design parameters are complete and consistent.

---

## Standalone Functions

### build_netlist_from_config
```python
output_file = build_netlist_from_config(
    config,             # NetlistConfig object
    output_dir=None,    # Override output directory
    verbose=True        # Print progress messages
)
```
High-level function that handles the complete workflow. Returns output filename.

### load_design_from_file
```python
design_params = load_design_from_file(
    filename,           # str: Design filename
    designs_dir=None    # str: Override designs directory
)
```
Loads design parameters from a Python file. Returns parameter dictionary.

---

## Design File Format

Design files must be Python files in the designs/ folder containing specific variables:

```python
# Required parameters based on device type

# Basic parameters
device_name = '4wm_jtwpa'
dispersion_type = 'filter'  # 'filter', 'periodic', or 'both'
nonlinearity = 'JJ'  # 'JJ' or 'KI'

# JJ-specific parameters
jj_structure_type = 'jj'  # 'jj' or 'rf_squid'
Ic_JJ_uA = 5.0
CJ_F = 1e-15

# RF-SQUID additional parameters
beta_L = 0.4
phi_dc = 1.57  # π/2

# KI-specific parameters
Istar_uA = 100.0
L0_pH = 100.0

# Filter parameters (if dispersion_type includes 'filter')
f_zeros_GHz = [9.0]
f_poles_GHz = [8.85]
# ... filter component arrays ...

# Periodic parameters (if dispersion_type includes 'periodic')
ind_Lsh_start = 10
ind_Lsh_stop = 20
# ... modulation arrays ...

# TWPA line parameters
Ntot_cell = 2000
Z0_TWPA_ohm = 50
```

---

## Complete Usage Examples

### Example 1: Basic Netlist Generation
```python
from twpa_design.netlist_JC_builder import NetlistConfig, build_netlist_from_config

# Generate netlist with default settings
config = NetlistConfig(design_file='4wm_jtwpa_01.py')
output_file = build_netlist_from_config(config)
print(f"Created: {output_file}")
```

### Example 2: With Dielectric Loss
```python
# Add realistic dielectric loss
config = NetlistConfig(
    design_file='b_jtwpa_01.py',
    enable_dielectric_loss=True,
    loss_tangent=1e-4  # tan(δ) = 0.0001
)
output_file = build_netlist_from_config(config)
```

### Example 3: Using Taylor Expansion
```python
# Use Taylor expansion for nonlinearity
config = NetlistConfig(
    design_file='4wm_ktwpa_01.py',
    use_taylor_insteadof_JJ=True,  # Uses c1-c4 coefficients
    use_linear_in_window=True
)
output_file = build_netlist_from_config(config)
```

### Example 4: Override Cell Count
```python
# Test with fewer cells for faster simulation
config = NetlistConfig(
    design_file='4wm_jtwpa_01.py',
    Ntot_cell_override=500  # Use 500 cells instead of design value
)
output_file = build_netlist_from_config(config)
```

### Example 5: Manual Builder Usage
```python
from twpa_design.netlist_JC_builder import JCNetlistBuilder, load_design_from_file

# Load design manually
design = load_design_from_file('b_jtwpa_01.py')

# Create builder with custom config
config = NetlistConfig(
    design_file='b_jtwpa_01.py',
    enable_dielectric_loss=True
)
builder = JCNetlistBuilder(design, config)

# Build netlist
components, parameters, metadata = builder.build_flattened_netlist()

# Inspect before saving
print(f"Total components: {len(components)}")
print(f"Parameters: {list(parameters.keys())}")

# Save with custom filename
output = builder.save_flattened_netlist(
    components, 
    parameters, 
    metadata,
    filename="custom_netlist.py"
)
```

### Example 6: Batch Processing Multiple Designs
```python
designs = ['4wm_jtwpa_01.py', 'b_jtwpa_01.py', '4wm_ktwpa_01.py']

for design_file in designs:
    # Create versions with and without loss
    for with_loss in [False, True]:
        config = NetlistConfig(
            design_file=design_file,
            enable_dielectric_loss=with_loss,
            loss_tangent=1e-4 if with_loss else 0
        )
        
        output = build_netlist_from_config(config, verbose=False)
        loss_str = "with_loss" if with_loss else "no_loss"
        print(f"{design_file} ({loss_str}): {output}")
```

### Example 7: Analyzing Netlist Structure
```python
# Build netlist and analyze structure
config = NetlistConfig(design_file='b_jtwpa_01.py')
builder = JCNetlistBuilder(load_design_from_file('b_jtwpa_01.py'), config)

components, parameters, metadata = builder.build_flattened_netlist()

# Count component types
component_types = {}
for name, _, _, _ in components:
    comp_type = name[0]  # First letter indicates type
    component_types[comp_type] = component_types.get(comp_type, 0) + 1

print("Component breakdown:")
for comp_type, count in sorted(component_types.items()):
    print(f"  {comp_type}: {count}")

# Check for nonlinear elements
nl_count = sum(1 for name, _, _, _ in components if name.startswith('NL'))
print(f"\nNonlinear elements: {nl_count}")

# Check for loss
has_loss = any('im' in str(v) for v in parameters.values())
print(f"Has dielectric loss: {has_loss}")
```

### Example 8: Custom Output Directory
```python
import os

# Save to custom directory
output_dir = os.path.join(os.getcwd(), "my_netlists")
os.makedirs(output_dir, exist_ok=True)

config = NetlistConfig(design_file='4wm_jtwpa_01.py')
output = build_netlist_from_config(config, output_dir=output_dir)
print(f"Saved to: {output}")
```

---

## Netlist Output Format

The generated Python file contains:

```python
# Component list: (name, node1, node2, value)
jc_components = [
    ("P1_0", "1", "0", "1"),
    ("R1_0", "1", "0", "R_port"),
    ("L1_2", "1", "2", "L0"),
    ("C2_0", "2", "0", "C0"),
    # ... more components
]

# Circuit parameters
circuit_parameters = {
    'L0': 1e-11,
    'C0': 1e-15,
    'R_port': 50,
    # ... more parameters
}

# Metadata
metadata = {
    'design_file': 'source.py',
    'total_cells': 2000,
    'dielectric_loss_enabled': False,
    # ... more metadata
}
```

---

## Component Naming Convention

- **P**: Port (P1_0 = port 1 to ground)
- **R**: Resistor (R1_2 = resistor from node 1 to 2)
- **L**: Inductor
- **C**: Capacitor
- **NL**: Nonlinear inductor
- **B**: Josephson junction (backward compatible)
- **K**: Mutual inductance

Node naming:
- Regular cells: 1, 2, 3, ...
- Filter nodes: Can have intermediate nodes like 2.1, 2.2

---

## Supported Nonlinear Elements

### Currently Implemented
- **JJ**: Josephson junctions
  - Standard JJ with Ic and shunt capacitance
  - Uses "B" component type in netlist
  - When n_jj_struct > 1, creates multiple JJs in series
  - Each JJ gets its own shunt capacitor (CJ_F)
- **RF-SQUID**: RF-biased SQUID
  - Includes flux bias and beta_L
  - Uses modified JJ with geometric inductance
  - When n_jj_struct > 1, creates multiple RF-SQUIDs in series
  - Each RF-SQUID gets its own shunt capacitor
- **KI**: Kinetic inductance
  - Uses Taylor expansion with Istar
  - Implemented as "NL" component

### Not Yet Implemented
- **SNAIL**: Raises NotImplementedError
- **DC-SQUID**: Raises NotImplementedError

---

## Filter Types

When `dispersion_type` includes 'filter':

- **LF1**: Series L filter, Foster form 1
- **LF2**: Series L filter, Foster form 2  
- **CF1**: Shunt C filter, Foster form 1
- **CF2**: Shunt C filter, Foster form 2

Selection controlled by:
- `Foster_form_L`: 1 or 2 for series filters
- `Foster_form_C`: 1 or 2 for shunt filters
- `select_one_form`: 'L', 'C', or 'both'

---

## Common Issues and Solutions

### Design File Not Found
```python
# Check available designs
import os
from twpa_design import DESIGNS_DIR
print("Available designs:")
for file in os.listdir(DESIGNS_DIR):
    if file.endswith('.py'):
        print(f"  {file}")
```

### Missing Design Parameters
```python
# Validate design before building
builder = JCNetlistBuilder(design_params)
if not builder.validate_design():
    print("Design validation failed - check parameters")
```

### Large Netlists
```python
# Reduce cells for testing
config = NetlistConfig(
    design_file='design.py',
    Ntot_cell_override=100  # Test with 100 cells first
)
```

### Complex Loss Values
```python
# Dielectric loss creates complex capacitance values
# Format: "C/(1+loss_tangent*im)"
# Example: "1e-15/(1+0.001im)" for tan(δ)=0.001
```

---

## Design Parameter Requirements

### All Designs Must Have
- `device_name`: String identifier
- `dispersion_type`: 'filter', 'periodic', or 'both'
- `nonlinearity`: 'JJ' or 'KI'
- `Ntot_cell`: Total number of cells
- `Z0_TWPA_ohm`: Characteristic impedance

### JJ Designs Must Have
- `jj_structure_type`: 'jj' or 'rf_squid'
- `Ic_JJ_uA`: Critical current
- `CJ_F`: Junction capacitance

### KI Designs Must Have
- `Istar_uA`: Nonlinearity scale current
- `L0_pH`: Inductance per cell

### Filter Designs Must Have
- Component arrays (e.g., `LinfLF1_H`, `C0CF1_F`)
- Filter parameters (`n_zeros`, `n_poles`)

### Periodic Designs Must Have
- Modulation indices (`ind_Lsh_start`, `ind_Lsh_stop`)
- Component modulation arrays

---

*Note: Generated netlists are compatible with JosephsonCircuits.jl for harmonic balance analysis.*