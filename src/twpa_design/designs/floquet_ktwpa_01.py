# Auto-generated TWPA design parameters
# Generated on: 2026-05-18 11:35:52
# Device: floquet_ktwpa
# Nonlinearity: KI
# Dispersion type: both

import numpy as np

# ========== CONFIGURATION ==========
# Parameters defining the TWPA design choices

config = {
    'device_name': 'floquet_ktwpa',
    'f_step_MHz': 5,
    'fmax_GHz': 30,
    'nonlinearity': 'KI',
    'window_type': 'boxcar',
    'f_zeros_GHz': 9.8,
    'f_poles_GHz': 9.75,
    'fc_filter_GHz': 150.05271935951774,
    'fc_TLsec_GHz': 150.05271935951774,
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
    'L0_pH': 150,
    'Z_taper': True,
    'Z_taper_width': 0.3,
    'Z_profile': 'klopfenstein',
    'floquet_taper': True,
    'floquet_profile': 'gaussian',
    'floquet_taper_width': 0.3,
    'rf_squid_constant_plasma': True,
    'taper_cutoff': False,
    'WM': '4WM',
    'dir_prop_PA': 'forw',
    'Ia0_uA': 30,
    'detsigidlGHz': 3,
    'fa_min_GHz': 8.7,
    'fa_max_GHz': 9.7,
    'Ntot_cell': 9996,
    'nTLsec': 0,
    'n_jj_struct': 1,
    'Z0_TWPA_ohm': 100,
}

# ========== CIRCUIT COMPONENTS ==========
# Parameters needed to build the netlist

circuit = {
    'dispersion_type': 'both',
    'Nsc_cell': 833,
    'Ncpersc_cell': 12,
    'ngL': 12,
    'ngC': 12,
    'width': 1500,
    'n_periodic_sc': 583,
    'L0_H': 1.5e-10,
    'C0_F': 1.5e-14,
    'epsilon_perA': 0.0,
    'xi_perA2': 100000000.00000001,
    'c1_taylor': 0.0,
    'c2_taylor': 0.0004813810891923803,
    'n_zeros': 1,
    'n_poles': 1,
    'LinfLF1_H': [np.nan],
    'LinfLF1_rem_H': [np.nan],
    'C0LF1_F': [np.nan],
    'LiLF1_H': [[np.nan]],
    'CiLF1_F': [[np.nan]],
    'L0LF2_H': [1.5154240631163702e-10],
    'L0LF2_rem_H': [1.5424063116370208e-12],
    'CinfLF2_F': [0.0],
    'LiLF2_H': [[1.4737595907928532e-08]],
    'CiLF2_F': [[1.7896226493475993e-14]],
    'LinfCF1_H': [0.0],
    'C0CF1_F': [1.867910350886471e-14],
    'LiCF1_H': [[0.0]],
    'CiCF1_F': [[np.inf]],
    'L0CF2_H': [np.nan],
    'CinfCF2_F': [np.nan],
    'LiCF2_H': [[np.nan]],
    'CiCF2_F': [[np.nan]],
    'ind_g_C_with_filters': [11],
    'n_filters_per_sc': 1,
    'g_C_pattern': [1.8147436204887275, 1.7610827676804903, 1.6144785914309114, 1.4142135623730951, 1.213948533315279, 1.0673443570657002, 1.0136835042574628, 1.0673443570657, 1.2139485333152789, 1.4142135623730951, 1.614478591430911, 1.76108276768049],
    'CTLsec_pattern': [3.8496525604878146e-14, 3.734816393734766e-14, 3.4229830091295935e-14, 2.9975756110020653e-14, 2.5723961511209215e-14, 2.261123554435335e-14, 2.1468613927558113e-14, 2.259892641173775e-14, 2.569596184459492e-14, 2.992682801147227e-14, 3.4155354551076626e-14, 3.724661620143345e-14],
}

# ========== DEVICE CHARACTERISTICS ==========
# Calculated performance parameters

characteristics = {
    'fa_GHz': 9.565000000000001,
    'fs_GHz': 12.565000000000001,
    'fi_GHz': 6.565000000000001,
    'Istar_uA': 100,
    'Ia_uA': 30,
    'v_cellpernsec': 666.6666666666669,
    'lambda_PA_cell': 69.69855375500958,
    'l_device_lambda_PA': 143.41761,
    'Pa_dBm': -43.46787486224656,
    'fcmax_GHz': 150.05271935951774,
    'maxL_ind_H': 1.4737595907928532e-08,
    'maxL_cap_F': 1.7896226493475993e-14,
    'maxC_ind_H': 0.0,
    'maxC_cap_F': 1.867910350886471e-14,
}
