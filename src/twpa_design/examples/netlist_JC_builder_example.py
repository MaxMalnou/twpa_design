"""
Example: Netlist Builder for JosephsonCircuits.jl

This example demonstrates how to use the netlist_JC_builder module
to generate netlists compatible with Kevin O'Brien's JosephsonCircuits.jl
"""

from twpa_design.netlist_JC_builder import NetlistConfig, build_netlist_from_config
import os

# Configure LaTeX rendering for plots (set to False to disable LaTeX)
USE_LATEX = True

# Apply the LaTeX setting
import twpa_design.plots_params as plots_params
plots_params.USE_LATEX = USE_LATEX
plots_params.configure_matplotlib(USE_LATEX)

"""
Parameters that can be customized for netlist building.
The unspecified parameters will be set to their default values.

NetlistConfig parameters
------------------------
'design_file': Name of design file from designs/ folder (required)
'use_taylor_insteadof_JJ': Use Taylor expansion instead of JJ hardcoded potential. default: False
'enable_dielectric_loss': Enable dielectric loss in the netlist. default: False  
'loss_tangent': Dielectric loss tangent value. default: 2e-4
'use_linear_in_window': Use linear elements in apodization windows. default: True
'Ntot_cell_override': Override total number of cells if needed. default: None

Available design files:
- '4wm_jtwpa_01.py': Standard 4-wave mixing JTWPA  
- 'b_jtwpa_01.py': Broadband JTWPA with rf-SQUID nonlinearity
- '4wm_ktwpa_01.py': 4WM KTWPA with kinetic inductance
"""

NETLIST_CONFIGS = {
    'jtwpa': {
        'design_file': '4wm_jtwpa_01.py',        
        'enable_dielectric_loss': True,        
    },
    'b_jtwpa': {
        'design_file': 'b_jtwpa_01.py',
        'use_taylor_insteadof_JJ': True,           
    },
    'ktwpa': {
        'design_file': '4wm_ktwpa_01.py',
        'use_taylor_insteadof_JJ': True,
        'enable_dielectric_loss': False,
        'use_linear_in_window': True,   
        'Ntot_cell_override': 5000,  # Override total number of cells
    }
}

simulation_type = "ktwpa"  # Choose: 'jtwpa', 'b_jtwpa', 'ktwpa'

# Get the config for the chosen netlist
if simulation_type in NETLIST_CONFIGS:
    config_dict = NETLIST_CONFIGS[simulation_type]
else:
    raise ValueError(f"Unknown netlist type: {simulation_type}")

# Create NetlistConfig object
netlist_config = NetlistConfig(**config_dict)

print(f"Building netlist with configuration: {simulation_type}")
print(f"Design file: {netlist_config.design_file}")
print(f"Taylor expansion: {netlist_config.use_taylor_insteadof_JJ}")
print(f"Dielectric loss: {netlist_config.enable_dielectric_loss}")

if netlist_config.enable_dielectric_loss:
    print(f"Loss tangent: {netlist_config.loss_tangent}")

print("\n" + "="*50)

# Build the netlist
try:
    output_file = build_netlist_from_config(netlist_config)
    print(f"\n✅ Netlist successfully generated!")
    print(f"📁 Output file: {output_file}")
    
    if os.path.exists(output_file):
        print(f"📏 File size: {os.path.getsize(output_file)} bytes")
    
except Exception as e:
    print(f"\n❌ Error building netlist: {e}")
    raise