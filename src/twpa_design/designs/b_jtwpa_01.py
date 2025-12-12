# Auto-generated TWPA design parameters
# Generated on: 2025-11-15 13:50:04
# Device: b_jtwpa
# Nonlinearity: JJ
# Dispersion type: filter

import numpy as np

# ========== CONFIGURATION ==========
# Parameters defining the TWPA design choices

config = {
    'device_name': 'b_jtwpa',
    'f_step_MHz': 5,
    'fmax_GHz': 50,
    'nonlinearity': 'JJ',
    'window_type': 'boxcar',
    'f_zeros_GHz': 9.5,
    'f_poles_GHz': [],
    'fc_filter_GHz': 48,
    'fc_TLsec_GHz': 170.97765899143656,
    'Foster_form_L': 1,
    'Foster_form_C': 1,
    'select_one_form': 'both',
    'jj_structure_type': 'rf_squid',
    'Ic_JJ_uA': 2,
    'fJ_GHz': 40,
    'beta_L': 0.4,
    'phi_dc': 1.5707963267948966,
    'WM': '3WM',
    'dir_prop_PA': 'back',
    'Ia0_uA': 2,
    'detsigidlGHz': 0.5,
    'fa_min_GHz': 10,
    'fa_max_GHz': 20,
    'Ntot_cell': 2000,
    'nTLsec': 0,
    'n_jj_struct': 1,
    'Z0_TWPA_ohm': 50,
}

# ========== CIRCUIT COMPONENTS ==========
# Parameters needed to build the netlist

circuit = {
    'dispersion_type': 'filter',
    'Nsc_cell': 2000,
    'Ncpersc_cell': 1,
    'ngL': 1,
    'ngC': 1,
    'width': 0,
    'n_periodic_sc': 2000,
    'L0_H': 6.582119569509068e-11,
    'C0_F': 2.6328478278036272e-14,
    'LJ0_H': 1.645529892377267e-10,
    'CJ_F': 9.620873505521005e-14,
    'Lg_H': 6.582119569509068e-11,
    'epsilon_perA': 80000.0,
    'xi_perA2': 9600000000.0,
    'c1_taylor': 0.4,
    'c2_taylor': 0.16000000000000003,
    'c3_taylor': -0.002666666666666647,
    'c4_taylor': -0.027733333333333332,
    'n_zeros': 1,
    'n_poles': 0,
    'LinfLF1_H': [2.344573739992464e-10],
    'LinfLF1_rem_H': [1.6863617830415574e-10],
    'C0LF1_F': [1.197096486857094e-12],
    'LiLF1_H': [[]],
    'CiLF1_F': [[]],
    'L0LF2_H': [np.nan],
    'L0LF2_rem_H': [np.nan],
    'CinfLF2_F': [np.nan],
    'LiLF2_H': [[]],
    'CiLF2_F': [[]],
    'LinfCF1_H': [0.0],
    'C0CF1_F': [np.inf],
    'LiCF1_H': [[2.9927412171427356e-09]],
    'CiCF1_F': [[9.378294959969857e-14]],
    'L0CF2_H': [np.nan],
    'CinfCF2_F': [np.nan],
    'LiCF2_H': [[np.nan]],
    'CiCF2_F': [[np.nan]],
}

# ========== DEVICE CHARACTERISTICS ==========
# Calculated performance parameters

characteristics = {
    'fa_GHz': 15.015,
    'fs_GHz': 8.0075,
    'fi_GHz': 7.0075,
    'jj_structure_type': 'rf_squid',
    'beta_L': 0.4,
    'Ic_JJ_uA': 2,
    'Ia_uA': 2,
    'v_cellpernsec': 213.25838103160154,
    'lambda_PA_cell': 14.203022379727042,
    'l_device_lambda_PA': 140.8150988239474,
    'Pa_dBm': -66.98970004336019,
    'fcmax_GHz': 170.97765899143656,
    'maxL_ind_H': 2.344573739992464e-10,
    'maxL_cap_F': 1.197096486857094e-12,
    'maxC_ind_H': 2.9927412171427356e-09,
    'maxC_cap_F': 9.378294959969857e-14,
    'ind2jjstruct_ratio': 3.562034562321595,
    'phi_dc': 1.5707963267948966,
    'phi_ext': 1.9707963267948965,
    'J_uA': 2.0,
}
