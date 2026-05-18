# Auto-generated TWPA design parameters
# Generated on: 2026-05-18 11:20:45
# Device: rfsq_twpa
# Nonlinearity: JJ
# Dispersion type: filter

import numpy as np

# ========== CONFIGURATION ==========
# Parameters defining the TWPA design choices

config = {
    'device_name': 'rfsq_twpa',
    'f_step_MHz': 5,
    'fmax_GHz': 25,
    'nonlinearity': 'JJ',
    'window_type': 'boxcar',
    'f_zeros_GHz': 9.0,
    'f_poles_GHz': 8.85,
    'fc_filter_GHz': 384.69973273073236,
    'fc_TLsec_GHz': 384.69973273073236,
    'Foster_form_L': 1,
    'Foster_form_C': 1,
    'select_one_form': 'C',
    'jj_structure_type': 'rf_squid',
    'Ic_JJ_uA': 5,
    'fJ_GHz': 40,
    'beta_L': 0.8,
    'phi_dc': 0,
    'WM': '4WM',
    'dir_prop_PA': 'forw',
    'Ia0_uA': 8,
    'detsigidlGHz': 3,
    'fa_min_GHz': 7.75,
    'fa_max_GHz': 8.75,
    'Ntot_cell': 2497,
    'nTLsec': 10,
    'n_jj_struct': 1,
    'Z0_TWPA_ohm': 50,
}

# ========== CIRCUIT COMPONENTS ==========
# Parameters needed to build the netlist

circuit = {
    'dispersion_type': 'filter',
    'Nsc_cell': 227,
    'Ncpersc_cell': 11,
    'ngL': 1,
    'ngC': 1,
    'width': 0,
    'n_periodic_sc': 227,
    'L0_H': 2.925386475337363e-11,
    'C0_F': 1.1701545901349452e-14,
    'LTLsec_H': 2.925386475337363e-11,
    'LTLsec_rem_H': 0,
    'CTLsec_F': 1.1701545901349452e-14,
    'LJ0_H': 6.582119569509067e-11,
    'CJ_F': 2.405218376380252e-13,
    'Lg_H': 5.265695655607254e-11,
    'epsilon_perA': 0.0,
    'xi_perA2': 1755829903.9780524,
    'c1_taylor': 0.0,
    'c2_taylor': 0.4,
    'c3_taylor': 0.0,
    'c4_taylor': 0.030864197530864206,
    'n_zeros': 1,
    'n_poles': 1,
    'LinfLF1_H': [2.925386475337363e-11],
    'LinfLF1_rem_H': [0.0],
    'C0LF1_F': [np.inf],
    'LiLF1_H': [[0.0]],
    'CiLF1_F': [[np.inf]],
    'L0LF2_H': [np.nan],
    'L0LF2_rem_H': [np.nan],
    'CinfLF2_F': [np.nan],
    'LiLF2_H': [[np.nan]],
    'CiLF2_F': [[np.nan]],
    'LinfCF1_H': [0.0],
    'C0CF1_F': [1.2101570021504746e-14],
    'LiCF1_H': [[8.83398107374225e-10]],
    'CiCF1_F': [[3.53996346595447e-13]],
    'L0CF2_H': [np.nan],
    'CinfCF2_F': [np.nan],
    'LiCF2_H': [[np.nan]],
    'CiCF2_F': [[np.nan]],
}

# ========== DEVICE CHARACTERISTICS ==========
# Calculated performance parameters

characteristics = {
    'fa_GHz': 8.440000000000001,
    'fs_GHz': 11.440000000000001,
    'fi_GHz': 5.440000000000001,
    'jj_structure_type': 'rf_squid',
    'beta_L': 0.8,
    'Ic_JJ_uA': 5,
    'Ia_uA': 8,
    'v_cellpernsec': 1706.5862184409339,
    'lambda_PA_cell': 202.202158583049,
    'l_device_lambda_PA': 12.349027416413191,
    'Pa_dBm': -54.94850021680094,
    'fcmax_GHz': 384.69973273073236,
    'maxL_ind_H': 2.925386475337363e-11,
    'maxL_cap_F': 0,
    'maxC_ind_H': 8.83398107374225e-10,
    'maxC_cap_F': 3.53996346595447e-13,
    'ind2jjstruct_ratio': 1.0,
    'phi_dc': 0,
    'phi_ext': 0.0,
    'J_uA': 0.0,
}
