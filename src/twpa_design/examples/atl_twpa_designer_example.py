"""
Example: JTWPA Design with Custom Configuration
"""

from twpa_design.atl_twpa_designer import ATLTWPADesigner
import numpy as np

# Configure LaTeX rendering for plots (set to False to disable LaTeX)
USE_LATEX = True

# Apply the LaTeX setting
import twpa_design.plots_params as plots_params
plots_params.USE_LATEX = USE_LATEX
plots_params.configure_matplotlib(USE_LATEX)

"""
Parameters that can be customized for different devices.
The unspecified parameters will be set to their default values.

Basic parameters
----------------
'device_name': device name used for saving. default: '4wm_jtwpa'.
'f_step_MHz': frequency steps to plot and look for phase-matching. default: 5
'fmax_GHz': maximum frequency to plot and look for phase-matching. default: 30

Filter parameters
-----------------
'f_zeros_GHz': position of the ATL filter zeros. default: []. Can be a single number, list, or empty list.
              
'f_poles_GHz': position of the ATL filter poles. default: []. Can be a single number, list, or empty list.


'fc_filter_GHz': normalizing frequency of the filer. default: 500. If fc_filter_GHz > fcmax_GHz
                 defined by fcmax_GHz = sqrt(2)/sqrt(L0_H*C0_F), where L0 is the static inductor of the series nonlinear elements,
                 and C0_F is the shunt capacitor defined by C0_F = L0_H / (Z0_TWPA_ohm**2), then fc_filter_GHz is set to
                 fc_filter_GHz = fcmax_GHz
'fc_TLsec_GHz':  normalizing frequency of the filer. default: 500. If fc_TLsec_GHz > fcmax_GHz
                 defined by fcmax_GHz = sqrt(2)/sqrt(L0_H*C0_F), where L0 is the static inductor of the series nonlinear elements,
                 and C0_F is the shunt capacitor defined by C0_F = L0_H / (Z0_TWPA_ohm**2), then fc_TLsec_GHz is set to
                 fc_TLsec_GHz = fcmax_GHz

both fc_filter_GHz and fc_TLsec_GHz are handled such that one can lower the cutoff frequency of the line by adding more inductance,
and the lowest cutoff frequencies are set by the nonlinear element used in the series branch.

'Foster_form_L': type of Foster form for the filter in the series branch. default: 1. Can be 1 or 2
'Foster_form_C': type of Foster form for the filter in the shunt branch. default: 1. Can be 1 or 2
'select_one_form': select if we choose to keep only the filters in the series or shunt branch. default: 'C'. Can be 'L', 'C', or 'both'

In regular rpm twpas, one can choose to have only one filter form in either the series or shunt branch. In that case, for the branch
where the filter is discarded, it is replaced by the right (left) handed element, depending on whether zero_at_zero = 0 (1).
The zero_at_zero value is determined from whether we put a pole or a zero first along the frequency line.
In more complicated twpas (for example the b_twpa) one needs the filters in both branches.

Periodic modulation parameters
-----------------
'stopbands_config_GHz': dictionary defining stopband frequencies and their edges. default: {}.
                        Format: {freq_GHz: {'min': width} or {'max': width}}
                        Example: {27: {'max': 4}} creates a stopband at 27 GHz with upper edge 4 GHz above center.
                        For each stopband, specify EITHER 'min' (lower edge width) or 'max' (upper edge width), not both.
'window_type': apodization window at the beginning and end of the periodic modulation. default 'boxcar' (i.e. no window). Can be 'tukey' Or 'hann', 'boxcar'
'alpha': apodization length (as a proportion of the line's total length) of the window. default: 0.0
'n_filters_per_sc': number of filters per supercell. default: 1

Nonlinearity parameters
-----------------
'nonlinearity': type of nonlinearity. default 'JJ'. Can be 'JJ' (Josephson junction) or 'KI' (kinetic inductance)
'Id_uA': dc-current bias. default 0

Nonlinearity parameters - JJ specific
-----------------
'jj_structure_type': IF we choose the JJ nonlinearity, specifies which jj structure we want to choose. default 'jj'. Can be 'jj' or 'rf_squid'
'Ic_JJ_uA': JJ critical current. default 5
'fJ_GHz': JJ plasma frequency. default 40
'beta_L': rf-squid participation ratio. default np.inf,
'phi_dc': rf-squid dc-bias (in normalized flux quantum units). default 0

Nonlinearity parameters - KI specific
-----------------
'Istar_uA': KI scaling current. default 100
'L0_pH': KI inductance per cell. default 100

Phase-matching parameters
-----------------
'WM': wave mixing process. default '4WM'. Can be '3WM' or '4WM'
'dir_prop_PA': parametric amplification pump injection. default 'forw'. Can be 'forw' or 'back'
'Ia0_uA': pump amplitude. default 1
'detsigidlGHz': detuning between signal and idler for which we look at phase-matching. default 2
'fa_min_GHz': min frequency at which we look for the phase-matched pump. default 0
'fa_max_GHz': max frequency at which we look for the phase-matched pump. default 30

TWPA line parameters
-----------------
'Ntot_cell': total number of cells within the TWPA. default 2000
'nTLsec': number of regular transmission line section between filter sections. default 0. 
          When non-zero, we add half of the cells on both sides of the filter sections. So nTLsec is always even.
'n_jj_struct': number of jj structures (jj or rf_squids) per unit cell. default 1
'Z0_TWPA_ohm': TWPA base impedance. default 50
"""


DEVICE_CONFIGS = {
    'jtwpa': {
        # Basic parameters
        'device_name': '4wm_jtwpa',
        'fmax_GHz':25,
        # Filter parameters
        'f_zeros_GHz': 9,
        'f_poles_GHz': 8.85,
        'fc_filter_GHz': 200,
        'fc_TLsec_GHz': 200,
        'Foster_form_C': 1,
        'Foster_form_L': 1,
        'select_one_form': 'C',
        # Nonlinearity parameters
        'nonlinearity': 'JJ',
        'jj_structure_type': 'jj',
        'Ic_JJ_uA': 5,
        'fJ_GHz':40,
        # Phase-matching parameters
        'WM': '4WM',
        'dir_prop_PA': 'forw',
        'Ia0_uA': 3,
        'detsigidlGHz': 3,        
        'fa_min_GHz': 7.75,
        'fa_max_GHz': 8.75,
        # TWPA line parameters
        'Ntot_cell': 2000,
        'nTLsec': 10,
    },
    'b_jtwpa': {
        # Basic parameters
        'device_name': 'b_jtwpa',
        'fmax_GHz': 50,
        # Filter parameters
        'f_zeros_GHz': 9.5,
        'f_poles_GHz': [],  # Empty list
        'fc_filter_GHz': 48,        
        'Foster_form_L': 1,
        'Foster_form_C': 1,
        'select_one_form': 'both',
        # Nonlinearity parameters
        'nonlinearity':'JJ',
        'jj_structure_type': 'rf_squid',
        'Ic_JJ_uA': 2,
        'fJ_GHz': 40,
        'beta_L': 0.4,
        'phi_dc': np.pi/2, # Kerr free point for rf-SQUID        
        # Phase-matching parameters
        'WM': '3WM',
        'dir_prop_PA': 'back',
        'Ia0_uA': 2,
        'detsigidlGHz': 0.5,
        'fa_min_GHz': 10,
        'fa_max_GHz': 20,
        # TWPA line parameters
        'Ntot_cell': 2000,
        'nTLsec': 0,
        'n_jj_struct': 1,
    },
    'ktwpa': {
        # Basic parameters
        'device_name': '4wm_ktwpa',
        'fmax_GHz': 40, # 40
        # Filter parameters
        'f_zeros_GHz': 9.8,
        'f_poles_GHz': 9.7,
        'fc_filter_GHz': 500,
        'fc_TLsec_GHz': 500,
        'Foster_form_L': 2,
        'Foster_form_C': 1,        
        'select_one_form': 'L',
        'stopbands_config_GHz': {27: {'max': 4}},  # Stopband at 27 GHz with upper edge at +4 GHz        
        'n_filters_per_sc': 1,  
        # Nonlinearity parameters      
        'nonlinearity': 'KI',
        'Istar_uA': 100,
        'L0_pH': 100,
        # Phase-matching parameters
        'WM': '4WM',
        'dir_prop_PA': 'forw',
        'Ia0_uA': 30,
        'detsigidlGHz': 3,
        'fa_min_GHz': 8.6,
        'fa_max_GHz': 9.6,   
        # TWPA line parameters
        'Ntot_cell': 5000, # 5598
    }
}


simulation_type = "jtwpa"  # Choose: 'jtwpa', 'b_jtwpa', 'ktwpa'

# Get the config for the chosen device
if simulation_type in DEVICE_CONFIGS:
    device_config = DEVICE_CONFIGS[simulation_type]
else:
    raise ValueError(f"Unknown device type: {simulation_type}")

# Create designer with the selected config
designer = ATLTWPADesigner(
    custom_params=device_config,  # Pass the config dict, not the string
    verbose=True
)

# Run interactive design
# save_results=True will export parameters to designs/ folder
# save_plots=True will save the phase matching plot as SVG to designs/ folder
results = designer.run_design(interactive=True, save_results=False, save_plots=False)