# Auto-generated TWPA design parameters
# Generated on: 2025-08-26 11:20:28
# Device: 4wm_ktwpa
# Nonlinearity: KI
# Dispersion type: both

import numpy as np

# ========== CONFIGURATION ==========
# Parameters defining the TWPA design choices

config = {
    'device_name': '4wm_ktwpa',
    'f_step_MHz': 5,
    'fmax_GHz': 40,
    'nonlinearity': 'KI',
    'window_type': 'boxcar',
    'f_zeros_GHz': 9.8,
    'f_poles_GHz': 9.7,
    'fc_filter_GHz': 112.53953951963828,
    'fc_TLsec_GHz': 112.53953951963828,
    'Foster_form_L': 2,
    'Foster_form_C': 1,
    'select_one_form': 'L',
    'stopbands_config_GHz': {
        27: {'max': 4},
    },
    'alpha': 0.0,
    'n_filters_per_sc': 1,
    'Istar_uA': 100,
    'Id_uA': 0,
    'L0_pH': 100,
    'WM': '4WM',
    'dir_prop_PA': 'forw',
    'Ia0_uA': 30,
    'detsigidlGHz': 3,
    'fa_min_GHz': 8.6,
    'fa_max_GHz': 9.6,
    'Ntot_cell': 5004,
    'nTLsec': 0,
    'n_jj_struct': 1,
    'Z0_TWPA_ohm': 50,
}

# ========== CIRCUIT COMPONENTS ==========
# Parameters needed to build the netlist

circuit = {
    'dispersion_type': 'both',
    'Nsc_cell': 556,
    'Ncpersc_cell': 9,
    'ngL': 9,
    'ngC': 9,
    'width': 0,
    'n_periodic_sc': 556,
    'L0_H': 1e-10,
    'C0_F': 4e-14,
    'LTLsec_H': 9.999999999999999e-11,
    'LTLsec_rem_H': 0,
    'epsilon_perA': 0.0,
    'xi_perA2': 100000000.00000001,
    'c1_taylor': 0.0,
    'c2_taylor': 0.0010831074506828559,
    'n_zeros': 1,
    'n_poles': 1,
    'LinfLF1_H': [np.nan],
    'LinfLF1_rem_H': [np.nan],
    'C0LF1_F': [np.nan],
    'LiLF1_H': [[np.nan]],
    'CiLF1_F': [[np.nan]],
    'L0LF2_H': [1.0207248379211393e-10],
    'L0LF2_rem_H': [2.0724837921139232e-12],
    'CinfLF2_F': [0.0],
    'LiLF2_H': [[4.925128205128162e-09]],
    'CiLF2_F': [[5.355136827971176e-14]],
    'LinfCF1_H': [0.0],
    'C0CF1_F': [3.1321806135047795e-14],
    'LiCF1_H': [[0.0]],
    'CiCF1_F': [[np.inf]],
    'L0CF2_H': [np.nan],
    'CinfCF2_F': [np.nan],
    'LiCF2_H': [[np.nan]],
    'CiCF2_F': [[np.nan]],
    'ind_g_C_with_filters': [8],
    'n_filters_per_sc': 1,
    'g_C_pattern': [1.0136878627560741, 1.1073930758551351, 1.344663004525827, 1.6144764121816055, 1.790584606738323, 1.790584606738323, 1.6144764121816058, 1.3446630045258272, 1.1073930758551354],
    'CTLsec_pattern': [2.867142247045272e-14, 3.132180613504778e-14, 3.803281315643556e-14, 4.5664288764773614e-14, 5.0645380708516605e-14, 5.0645380708516605e-14, 4.566428876477362e-14, 3.803281315643557e-14, 3.132180613504779e-14],
}

# ========== DEVICE CHARACTERISTICS ==========
# Calculated performance parameters

characteristics = {
    'fa_GHz': 9.165000000000001,
    'fs_GHz': 12.165000000000001,
    'fi_GHz': 6.165000000000001,
    'Istar_uA': 100,
    'Ia_uA': 30,
    'v_cellpernsec': 500.0,
    'lambda_PA_cell': 54.55537370430987,
    'l_device_lambda_PA': 91.72332,
    'Pa_dBm': -43.46787486224656,
    'fcmax_GHz': 112.53953951963828,
    'maxL_ind_H': 4.925128205128162e-09,
    'maxL_cap_F': 5.355136827971176e-14,
    'maxC_ind_H': 0.0,
    'maxC_cap_F': 3.1321806135047795e-14,
}
