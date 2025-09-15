# Auto-generated TWPA design parameters
# Generated on: 2025-09-10 09:33:38
# Device: 4wm_jtwpa
# Nonlinearity: JJ
# Dispersion type: filter

import numpy as np

# ========== CONFIGURATION ==========
# Parameters defining the TWPA design choices

config = {
    'device_name': '4wm_jtwpa',
    'f_step_MHz': 5,
    'fmax_GHz': 25,
    'nonlinearity': 'JJ',
    'window_type': 'boxcar',
    'f_zeros_GHz': 9.0,
    'f_poles_GHz': 8.85,
    'fc_filter_GHz': 170.97765899143658,
    'fc_TLsec_GHz': 170.97765899143658,
    'Foster_form_L': 1,
    'Foster_form_C': 1,
    'select_one_form': 'C',
    'jj_structure_type': 'jj',
    'Ic_JJ_uA': 5,
    'fJ_GHz': 40,
    'beta_L': np.inf,
    'phi_dc': 0,
    'WM': '4WM',
    'dir_prop_PA': 'forw',
    'Ia0_uA': 3,
    'detsigidlGHz': 3,
    'fa_min_GHz': 7.75,
    'fa_max_GHz': 8.75,
    'Ntot_cell': 2002,
    'nTLsec': 10,
    'n_jj_struct': 1,
    'Z0_TWPA_ohm': 50,
}

# ========== CIRCUIT COMPONENTS ==========
# Parameters needed to build the netlist

circuit = {
    'dispersion_type': 'filter',
    'Nsc_cell': 182,
    'Ncpersc_cell': 11,
    'ngL': 1,
    'ngC': 1,
    'width': 0,
    'n_periodic_sc': 182,
    'L0_H': 6.582119569509067e-11,
    'C0_F': 2.632847827803627e-14,
    'LTLsec_H': 6.582119569509068e-11,
    'LTLsec_rem_H': 0,
    'CTLsec_F': 2.632847827803627e-14,
    'LJ0_H': 6.582119569509067e-11,
    'CJ_F': 2.405218376380252e-13,
    'Lg_H': np.inf,
    'epsilon_perA': 0.0,
    'xi_perA2': 20000000000.000004,
    'c1_taylor': 0.0,
    'c2_taylor': 0.5,
    'c3_taylor': 0.0,
    'c4_taylor': 0.20833333333333334,
    'n_zeros': 1,
    'n_poles': 1,
    'LinfLF1_H': [6.582119569509068e-11],
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
    'C0CF1_F': [2.7228532548385686e-14],
    'LiCF1_H': [[3.9262138105521366e-10]],
    'CiCF1_F': [[7.964917798397503e-13]],
    'L0CF2_H': [np.nan],
    'CinfCF2_F': [np.nan],
    'LiCF2_H': [[np.nan]],
    'CiCF2_F': [[np.nan]],
}

# ========== DEVICE CHARACTERISTICS ==========
# Calculated performance parameters

characteristics = {
    'fa_GHz': 8.63,
    'fs_GHz': 11.63,
    'fi_GHz': 5.630000000000001,
    'jj_structure_type': 'jj',
    'Ic_JJ_uA': 5,
    'Ia_uA': 3,
    'v_cellpernsec': 758.482763751526,
    'lambda_PA_cell': 87.88908038835758,
    'l_device_lambda_PA': 22.778711429835365,
    'Pa_dBm': -63.46787486224656,
    'fcmax_GHz': 170.97765899143658,
    'maxL_ind_H': 6.582119569509068e-11,
    'maxL_cap_F': 0,
    'maxC_ind_H': 3.9262138105521366e-10,
    'maxC_cap_F': 7.964917798397503e-13,
    'ind2jjstruct_ratio': 1.0000000000000002,
    'phi_dc': 0,
    'phi_ext': 0,
    'J_uA': 0,
}
