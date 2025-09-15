# src/twpa_design/atl_twpa_designer.py
"""
ATL TWPA Designer Module

This module implements the Artificial Transmission Line (ATL) TWPA design methodology
for creating traveling-wave parametric amplifiers with engineered dispersion.

NOTE: Incomplete/Future implementations
----------------------------------------
The following JJ structures are not yet implemented:
1. SNAIL (Superconducting Nonlinear Asymmetric Inductive eLement)
2. DC-SQUID (DC Superconducting Quantum Interference Device)

Currently supported nonlinear elements:
- Josephson Junctions (JJ): 'jj_structure_type' = 'jj'
- RF-SQUIDs: 'jj_structure_type' = 'rf_squid'
- Kinetic Inductance (KI): 'nonlinearity' = 'KI'
"""

import numpy as np
from scipy import constants
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import warnings
from typing import Optional, Dict, Any, Union, List
import numpy.typing as npt

from twpa_design.helper_functions import *
from twpa_design.plots_params import *

# Parameter documentation
PARAMETER_DOCS = """
Parameters that can be customized for different devices:

Basic parameters
----------------
'device_name': device name used for saving. default: '4wm_jtwpa'.
'f_step_MHz': frequency steps to plot and look for phase-matching. default: 5
'fmax_GHz': maximum frequency to plot and look for phase-matching. default: 30

Filter parameters
-----------------
'f_zeros_GHz': position of the ATL filter zeros. default: []. Can be a number (9), list ([9, 8.5]), or empty list ([]).
'f_poles_GHz': position of the ATL filter poles. default: []. Can be a number (8.85), list ([8.85, 9.2]), or empty list ([]).


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
'stopbands_config_GHz': Dict of {freq_GHz: {'min': val, 'max': val}} for defining stopbands.
                        For each stopband frequency, specify either 'min' or 'max' (not both).
                        Example: {27: {'max': 4}, 16: {'min': 1}}. default: {}
'window_type': apodization window at the beginning and end of the periodic modulation. default 'boxcar' (i.e. no window). Can be 'tukey' Or 'hann', 'boxcar'
'alpha': apodization length (as a proportion of the line's total length) of the window. default: 0.0
'n_filters_per_sc': number of filters per supercell. default: 1

Nonlinearity parameters
-----------------
'nonlinearity': type of nonlinearity. default 'JJ'. Cen be 'JJ' (Josephson junction) or 'KI' (kinetic inductance)
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

# Import the configuration dictionaries from cell 2 of your notebook
DEFAULT_CONFIG = {
    # Basic parameters
    'device_name': '4wm_jtwpa',
    'f_step_MHz': 5,
    'fmax_GHz': 30,
    
    # Filter parameters
    'f_zeros_GHz': [],
    'f_poles_GHz': [],
    'fc_filter_GHz': 500,
    'fc_TLsec_GHz': 500,
    'Foster_form_L': 1,
    'Foster_form_C': 1,
    'select_one_form': 'C', # 'L', 'C', or 'both'
    
    # Periodic modulation parameters
    'stopbands_config_GHz': {},  # Dict of {freq_GHz: {'min': val, 'max': val}} 
    'window_type': 'boxcar', # 'tukey' Or 'hann', 'boxcar'
    'alpha': 0.0,
    'n_filters_per_sc': 1,
    
    # Nonlinearity parameters
    'nonlinearity': 'JJ',  # 'JJ' or 'KI'
    'Id_uA': 0,
    # Nonlinearity parameters - JJ specific
    'jj_structure_type': 'jj', # 'jj' or 'rf_squid' 
    'Ic_JJ_uA': 5,
    'fJ_GHz': 40,
    'beta_L': np.inf,
    'phi_dc': 0,    
    # Nonlinearity parameters - KI specific
    'Istar_uA': 100,
    'L0_pH': 100,
    
    # Phase-matching parameters
    'WM': '4WM', # '3WM' or '4WM'
    'dir_prop_PA': 'forw', # 'forw' or 'back'
    'Ia0_uA': 1,
    'detsigidlGHz': 2,
    'fa_min_GHz': 0,
    'fa_max_GHz': 30,
    
    # TWPA line parameters
    'Ntot_cell': 2000,
    'nTLsec': 0,
    'n_jj_struct': 1,
    'Z0_TWPA_ohm': 50,
}



########################################################################################           

class ATLTWPADesigner:
    """Interactive TWPA design tool with filter synthesis and phase matching

    See PARAMETER_DOCS for detailed parameter documentation.

    """
    
    # Type hints for configuration parameters
    device_name: str
    f_step_MHz: float
    fmax_GHz: float
    f_zeros_GHz: npt.NDArray[np.float64]
    f_poles_GHz: npt.NDArray[np.float64]
    fc_filter_GHz: float
    fc_TLsec_GHz: float
    Foster_form_L: int
    Foster_form_C: int
    select_one_form: str
    stopbands_config_GHz: Dict[float, Dict[str, Optional[float]]]
    window_type: str
    alpha: float
    n_filters_per_sc: int
    nonlinearity: str
    jj_structure_type: str
    Ic_JJ_uA: float
    fJ_GHz: float
    beta_L: float
    phi_dc: float    
    Istar_uA: float
    Id_uA: float
    L0_pH: float
    WM: str
    dir_prop_PA: str
    Ia0_uA: float
    detsigidlGHz: float
    fa_min_GHz: float
    fa_max_GHz: float
    Ntot_cell: int
    nTLsec: int
    n_jj_struct: int
    Z0_TWPA_ohm: float
    
    # Type hints for calculated parameters
    sig_above_idl: int
    g_L: float
    g_C: float
    Z0_ohm: float
    Ia0_A: float
    Pa_dBm: float
    f_GHz: npt.NDArray[np.float64]
    w: npt.NDArray[np.float64]
    n_f: int
    C0_F: float
    L0_H: float
    fcmax_GHz: float
    
    # JJ-specific parameters
    Ic_JJ_A: float
    LJ0_H: float
    wJ: float
    CJ_F: float
    Lg_H: float
    phi_ext: float
    phi_dc: float
    J_uA: float
    c1_currentphase: float
    c2_currentphase: float
    c3_currentphase: float
    c4_currentphase: float
    epsilon_perA: float
    xi_perA2: float
    c1_taylor: float
    c2_taylor: float
    c3_taylor: float
    c4_taylor: float
    
    # KI-specific parameters
    Istar_A: float
    Id_A: float
    
    # Filter components (will be arrays or scalars)
    LinfLF1_H: Union[float, npt.NDArray[np.float64]]
    LinfLF1_rem_H: Union[float, npt.NDArray[np.float64]]
    C0LF1_F: Union[float, npt.NDArray[np.float64]]
    LiLF1_H: Union[float, npt.NDArray[np.float64]]
    CiLF1_F: Union[float, npt.NDArray[np.float64]]
    L0LF2_H: Union[float, npt.NDArray[np.float64]]
    L0LF2_rem_H: Union[float, npt.NDArray[np.float64]]
    CinfLF2_F: Union[float, npt.NDArray[np.float64]]
    LiLF2_H: Union[float, npt.NDArray[np.float64]]
    CiLF2_F: Union[float, npt.NDArray[np.float64]]
    LinfCF1_H: Union[float, npt.NDArray[np.float64]]
    C0CF1_F: Union[float, npt.NDArray[np.float64]]
    LiCF1_H: Union[float, npt.NDArray[np.float64]]
    CiCF1_F: Union[float, npt.NDArray[np.float64]]
    L0CF2_H: Union[float, npt.NDArray[np.float64]]
    CinfCF2_F: Union[float, npt.NDArray[np.float64]]
    LiCF2_H: Union[float, npt.NDArray[np.float64]]
    CiCF2_F: Union[float, npt.NDArray[np.float64]]
    
    # Transmission line and filter parameters
    LTLsec_H: float
    LTLsec_rem_H: float
    CTLsec_F: Union[float, npt.NDArray[np.float64]]
    Ncpersc_cell: int
    Nsc_cell: int
    ngL: int
    ngC: int
    width: int
    n_periodic_sc: int
    g_C_mod: Optional[npt.NDArray[np.float64]]
    ind_g_C_with_filters: Optional[List[int]]
    k_radpercell: npt.NDArray[np.float64]
    k_radpersc: npt.NDArray[np.float64]
    
    # Results storage
    results: Dict[str, Any]
    verbose: bool
    simulation_type: str
    
    
    def __init__(self, custom_params=None, verbose=True):
        """
        Initialize designer with configuration.
        
        Parameters
        ----------
        custom_params : dict, optional
            Custom parameter overrides for DEFAULT_CONFIG
        verbose : bool
            Whether to print initialization info
        """
        self.verbose = verbose
        
        # Load configuration from DEFAULT_CONFIG
        self._clear_variables()
        
        # Start with default configuration
        config = DEFAULT_CONFIG.copy()
        
        # Apply custom overrides
        if custom_params:
            config.update(custom_params)
            
        # Set all parameters as attributes
        for key, value in config.items():
            setattr(self, key, value)
        
        # Convert stopbands_config_GHz to the old format for internal use
        if hasattr(self, 'stopbands_config_GHz') and self.stopbands_config_GHz:
            self.f_stopbands_GHz = list(self.stopbands_config_GHz.keys())
            self.deltaf_min_GHz = [config.get('min') for config in self.stopbands_config_GHz.values()]
            self.deltaf_max_GHz = [config.get('max') for config in self.stopbands_config_GHz.values()]
        else:
            self.f_stopbands_GHz = []
            self.deltaf_min_GHz = []
            self.deltaf_max_GHz = []
        
        # Convert f_zeros_GHz and f_poles_GHz to numpy arrays for internal use
        self.f_zeros_GHz = ensure_numpy_array(self.f_zeros_GHz)
        self.f_poles_GHz = ensure_numpy_array(self.f_poles_GHz)
        
        # Derive dispersion_type from the parameters
        self.dispersion_type = derive_dispersion_type(self.f_zeros_GHz, self.f_poles_GHz, self.f_stopbands_GHz)
        
        # Normalize string parameters to lowercase
        string_params_to_normalize = [
            'window_type',        
            'dir_prop_PA',     
        ]
        
        for param in string_params_to_normalize:
            if hasattr(self, param) and isinstance(getattr(self, param), str):
                setattr(self, param, getattr(self, param).lower())
        
        # Configuration summary
        if self.verbose:
            print(f"\n=== Configuration Loaded: {self.device_name} ===")
            print(f"Nonlinearity: {self.nonlinearity} ", end='')
            if self.nonlinearity == 'JJ':
                print(f"({self.jj_structure_type}, Ic={self.Ic_JJ_uA}μA)")
            else:
                print(f"(Istar={self.Istar_uA}μA)")
            print(f"Wave mixing: {self.WM}")
            print(f"Propagation direction: {self.dir_prop_PA}")
            print(f"Dispersion type: {self.dispersion_type}")
            if custom_params:
                print(f"\nCustom overrides applied: {list(custom_params.keys())}")
        
        # Initialize storage for results
        self.results = {}
        
    def _clear_variables(self):
        """Clear potentially conflicting variables from previous runs."""
        variables_to_clear = [
            'LJ0_H', 'CJ_F', 'Ic_JJ_uA',  # JJ-specific
            'c1_taylor', 'c2_taylor', 'c3_taylor', 'c4_taylor', 'epsilon_perA', 'xi_perA2',
            'L0_H',  # Effective inductance for Taylor expansion
            'LinfLF1_H', 'LinfLF1_rem_H', 'C0LF1_F', 'LiLF1_H', 'CiLF1_F',  # Filter arrays
            'L0LF2_H', 'L0LF2_rem_H', 'CinfLF2_F', 'LiLF2_H', 'CiLF2_F',
            'LinfCF1_H', 'C0CF1_F', 'LiCF1_H', 'CiCF1_F',
            'L0CF2_H', 'CinfCF2_F', 'LiCF2_H', 'CiCF2_F',
            'dispersion_type',
            'f_stopbands_GHz', 'deltaf_min_GHz', 'deltaf_max_GHz',  # Derived from stopbands_config_GHz
            'ind_g_C_with_filters', 'n_filters_per_sc', 'width',
            'CTLsec_F', 'LTLsec_H', 'Ncpersc_cell',                        
        ]
        
        for var in variables_to_clear:
            if hasattr(self, var):
                delattr(self, var)
                if self.verbose:
                    print(f"  Cleared {var}")                        


    ########################################################################################           

    def run_initial_calculations(self):
        """Run initial calculations for general derived quantities."""
        if self.verbose:
            print("\n=== Running Initial Calculations ===")
        
        # Suppress warnings as in the notebook
        warnings.filterwarnings('ignore', 'Casting complex values to real discards the imaginary part')
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in matmul")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in multiply")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in scalar multiply")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in scalar subtract")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in scalar power")
        np.seterr(divide='ignore', invalid='ignore')
        
        # GENERAL DERIVED QUANTITIES
        self.sig_above_idl = 1
        
        # Butterworth LPF prototype for UC
        self.g_L = np.sqrt(2)
        self.g_C = np.sqrt(2)
        
        self.Z0_ohm = 50  # Impedance
        
        self.Ia0_A = self.Ia0_uA * 1E-6  # Convert to Amps
        
        # Power calculations
        self.Pa_dBm = 10 * np.log10((self.Ia0_A)**2 * self.Z0_ohm * 1E3)
        
        # Vector of frequency
        self.f_GHz = np.arange(self.f_step_MHz * 1E-3, self.fmax_GHz + self.f_step_MHz * 1E-3, self.f_step_MHz * 1E-3).astype(np.float64)
        self.w = 2 * np.pi * self.f_GHz * 1E9
        self.n_f = len(self.f_GHz)
        
        if self.nonlinearity == 'JJ':
            self._calculate_jj_parameters()
        elif self.nonlinearity == 'KI':
            self._calculate_ki_parameters()
            
        self.C0_F = self.L0_H / (self.Z0_TWPA_ohm**2)
        
        # Cutoff without filter
        self.fcmax_GHz = np.sqrt(self.g_L * self.g_C) / (2 * np.pi * np.sqrt(self.L0_H * self.C0_F)) * 1E-9
        
        # Store key results
        self.results['initial_calculations'] = {
            'Pa_dBm': self.Pa_dBm,
            'f_GHz': self.f_GHz,
            'w': self.w,
            'fcmax_GHz': self.fcmax_GHz,
            'L0_H': self.L0_H,
            'C0_F': self.C0_F
        }
        
    def _calculate_jj_parameters(self):
        """Calculate Josephson junction specific parameters."""
        # Circuit parameters
        self.Ic_JJ_A = self.Ic_JJ_uA * 1E-6
        self.LJ0_H = phi0 / self.Ic_JJ_A
        self.wJ = 2 * np.pi * self.fJ_GHz * 1E9
        self.CJ_F = 1 / (self.LJ0_H * self.wJ**2)
        
        self.Lg_H = self.beta_L * self.LJ0_H
        
        if self.jj_structure_type == 'jj':
            self.phi_ext = 0
            self.phi_dc = np.arcsin(self.Id_uA/self.Ic_JJ_uA) if self.Id_uA != 0 else 0
            self.J_uA = 0
            
        elif self.jj_structure_type == 'rf_squid':
            # Calculate bias points            
            self.phi_ext = self.phi_dc + self.beta_L * np.sin(self.phi_dc)
            self.Id_uA = self.phi_ext * phi0 / self.Lg_H * 1E6
            
            # Solve for circulating current
            def circ_current_eq(J):
                return J + self.Ic_JJ_A * np.sin(self.phi_ext + self.beta_L * J / self.Ic_JJ_A)
            self.J_uA = np.abs(fsolve(circ_current_eq, 0)[0]) * 1E6
            
        # Coefficients of the current-phase relation
        self.c1_currentphase = (1/self.beta_L + np.cos(self.phi_dc))
        self.c2_currentphase = - np.sin(self.phi_dc)
        self.c3_currentphase = - np.cos(self.phi_dc)
        self.c4_currentphase = + np.sin(self.phi_dc)
        
        if self.verbose:
            print("Taylor expansion of the current-phase relation at the dc operating point:")
            print(f"    c1 = {self.c1_currentphase}")
            print(f"    c2 = {self.c2_currentphase/math.factorial(2)}")
            print(f"    c3 = {self.c3_currentphase/math.factorial(3)}")
            print(f"    c4 = {self.c4_currentphase/math.factorial(4)}")
            
        self.epsilon_perA = - self.c2_currentphase / (self.c1_currentphase**2) * 1/self.Ic_JJ_A
        self.xi_perA2 = (3*self.c2_currentphase**2 - self.c1_currentphase*self.c3_currentphase) / (2*self.c1_currentphase**4) * 1/(self.Ic_JJ_A**2)
        
        # Calculate Taylor coefficients of L(φ)
        if self.jj_structure_type == 'jj':
            self.L0_H = self.LJ0_H
            self.c1_taylor = 0.0
            self.c2_taylor = 1/2
            self.c3_taylor = 0.0
            self.c4_taylor = 5/24
        elif self.jj_structure_type == 'rf_squid':
            self.L0_H = self.Lg_H / (1 + self.beta_L * np.cos(self.phi_dc))
            # Taylor coefficients for rf-squid
            self.c1_taylor = self.beta_L*np.sin(self.phi_dc)/(1+self.beta_L*np.cos(self.phi_dc))
            self.c2_taylor = self.beta_L*(np.cos(self.phi_dc) + self.beta_L*(1+np.sin(self.phi_dc)**2))/(2*(1+self.beta_L*np.cos(self.phi_dc)))
            # Complete Taylor coefficients for rf-squid
            denominator_cubed = (1 + self.beta_L * np.cos(self.phi_dc))**3
            self.c3_taylor = (self.beta_L * np.sin(self.phi_dc) * 
                            (3*self.beta_L*np.cos(self.phi_dc) - 2*self.beta_L**2*np.sin(self.phi_dc)**2 + 1)) / (6 * denominator_cubed)
            self.c4_taylor = (self.beta_L * (np.cos(self.phi_dc) + 
                            self.beta_L*(1 + np.sin(self.phi_dc)**2 + self.beta_L*np.cos(self.phi_dc)*(2 - 3*np.sin(self.phi_dc)**2)) + 
                            self.beta_L**3*np.sin(self.phi_dc)**4)) / (24 * (1 + self.beta_L * np.cos(self.phi_dc))**4)
            
        if self.verbose:
            print("Taylor expansion of L(φ) at the dc operating point:")
            print(f"    c1_taylor = {self.c1_taylor}")
            print(f"    c2_taylor = {self.c2_taylor}")
            print(f"    c3_taylor = {getattr(self, 'c3_taylor', 'N/A')}")
            print(f"    c4_taylor = {getattr(self, 'c4_taylor', 'N/A')}")
            
    def _calculate_ki_parameters(self):
        """Calculate kinetic inductance specific parameters."""
        # Set unused JJ parameters to appropriate values
        self.n_jj_struct = 1
        self.CJ_F = np.inf
        
        self.Istar_A = self.Istar_uA * 1E-6
        self.Id_A = self.Id_uA * 1E-6
        
        self.epsilon_perA = 2*self.Id_A/(self.Istar_A**2 + self.Id_A**2)
        self.xi_perA2 = 1/(self.Istar_A**2 + self.Id_A**2)
        
        self.L0_H = self.L0_pH * 1E-12
        
        self.c1_taylor = phi0/self.L0_H*self.epsilon_perA
        self.c2_taylor = 1/2*(phi0/self.L0_H)**2*(2*self.xi_perA2-3*self.epsilon_perA**2)
        
        if self.verbose:
            print("Taylor expansion of L(φ) at the dc operating point:")
            print(f"    c1_taylor = {self.c1_taylor}")
            print(f"    c2_taylor = {self.c2_taylor}")

    ########################################################################################           

    def calculate_derived_quantities(self):
        """Calculate derived quantities depending on the type of dispersion."""
        if self.verbose:
            print("\n=== Calculating Derived Quantities ===")
            
        if self.dispersion_type == 'filter' or self.dispersion_type == 'both':
            self.zero_at_zero = should_have_zero_at_zero(self.f_zeros_GHz, self.f_poles_GHz)
            
            # Handle fc_TLsec_GHz default
            if not hasattr(self, 'fc_TLsec_GHz'):
                self.fc_TLsec_GHz = self.fc_filter_GHz
                
            # Limit frequencies to cutoff
            if self.fc_filter_GHz > self.fcmax_GHz:
                self.fc_filter_GHz = self.fcmax_GHz
            if self.fc_TLsec_GHz > self.fcmax_GHz:
                self.fc_TLsec_GHz = self.fcmax_GHz
                
            self.wc_filter = 2 * np.pi * self.fc_filter_GHz * 1E9
            self.s = 1j * self.w / self.wc_filter
            
            self.w_zeros = self.f_zeros_GHz / self.fc_filter_GHz
            self.w_poles = self.f_poles_GHz / self.fc_filter_GHz
            
            self.n_zeros = len(self.w_zeros)
            self.n_poles = len(self.w_poles)
            
            # Frequency transformation
            self.lambda_val = frequency_transform(self.s, self.w_zeros, self.w_poles, self.zero_at_zero)
            
        elif self.dispersion_type == 'periodic':
            self.Foster_form_L = 1  # Arbitrary for periodic modulation
            self.Foster_form_C = 2
            
            if self.fc_TLsec_GHz > self.fcmax_GHz:
                self.fc_TLsec_GHz = self.fcmax_GHz
                
        # Transmission line section design
        self.LTLsec_H = self.g_L * self.Z0_TWPA_ohm / (2 * np.pi * self.fc_TLsec_GHz * 1E9)
        # Total inductance used by n_jj_struct nonlinear structures in series
        L0_total_H = self.n_jj_struct * self.L0_H
        self.LTLsec_rem_H = self.LTLsec_H - L0_total_H
        self.LTLsec_rem_H = max(0, self.LTLsec_rem_H)
        # Numerical precision
        if self.LTLsec_rem_H < 1e-20:
            self.LTLsec_rem_H = 0
            
        if self.dispersion_type == 'periodic' or self.dispersion_type == 'both':
            self._calculate_periodic_modulation()
            
        if self.dispersion_type == 'filter':
            self._calculate_filter_only()
            
        elif self.dispersion_type == 'both':
            self._calculate_filter_and_periodic()
            
        if self.dispersion_type == 'filter' or self.dispersion_type == 'both':
            self._finalize_filter_calculations()
            
        # Store results
        self.results['derived_quantities'] = {
            'LTLsec_H': self.LTLsec_H,
            'LTLsec_rem_H': self.LTLsec_rem_H,
            'Ncpersc_cell': self.Ncpersc_cell,
            'Nsc_cell': self.Nsc_cell,
            'Ntot_cell': self.Ntot_cell
        }
        
    def _calculate_periodic_modulation(self):
        """Calculate periodic modulation parameters."""
        # Get derived quantities from helper function
        (self.n_stopbands, self.v_cellpernsec, self.Ncpersc_cell, 
         self.w_tilde_stopband_edges, self.ind_stopband, self.is_default_value, 
         self.w_tilde_param, self.ind_param, self.max_ind_stopband, 
         self.n_param, self.skipped_indices) = pl_derived_quantities(
            self.f_stopbands_GHz, self.deltaf_min_GHz, self.deltaf_max_GHz, 
            self.fc_TLsec_GHz/np.sqrt(self.g_L*self.g_C)
        )
        
        # Calculate delta values
        self.delta_map, self.selected_delta = calculate_delta_values(
            self.ind_stopband, self.ind_param, self.w_tilde_stopband_edges, 
            self.is_default_value, self.max_ind_stopband, self.n_param, 
            self.skipped_indices
        )
        
        self.nTLsec = 0
        self.Nsc_cell = int(np.round(self.Ntot_cell/self.Ncpersc_cell))
        self.Ntot_cell = int(self.Nsc_cell*self.Ncpersc_cell)
        
        # Window-specific settings
        if self.window_type == 'boxcar':
            self.ngL = self.Ncpersc_cell
            self.ngC = self.Ncpersc_cell
        else:
            self.ngL = self.Ntot_cell
            self.ngC = self.Ntot_cell
            
        # Calculate cap modulation
        x_tilde = np.linspace(0, 1 - 1/self.Ncpersc_cell, self.Ncpersc_cell) * 2 * np.pi
        g_C_mod = self.g_C * np.ones(self.Ncpersc_cell) # Local variable first
        
        if self.verbose:
            print("Delta mapping for shunt capacitance modulation:")
            
        # Apply delta values
        for idx in range(1, self.max_ind_stopband + 1):
            if idx in self.delta_map:
                delta_val = self.delta_map[idx]
                g_C_mod = g_C_mod + 2 * delta_val * np.cos(idx * x_tilde)  # Work with local
                if self.verbose:
                    print(f"Applied delta_{idx} = {delta_val:.6f} to mode {idx}")
            else:
                if self.verbose:
                    print(f"Skipped mode {idx} (no free parameter)")
                    
        # Apply window
        if self.window_type == 'boxcar':
            self.g_C_mod = np.tile(g_C_mod, self.Nsc_cell)  # Now numpy knows it's an array
        else:
            self.g_C_mod = create_windowed_transmission_line(
                g_C_mod, self.Nsc_cell, window_type=self.window_type, alpha=self.alpha
            )
            
        if self.window_type == 'tukey':
            self.width = int(self.alpha * (self.Ntot_cell - 1) / 2)
            self.width = round(self.width / self.Ncpersc_cell) * self.Ncpersc_cell
            self.n_periodic_sc = int((self.ngC - 2*self.width)/self.Ncpersc_cell)
        else:
            self.width = 0
            self.n_periodic_sc = self.Nsc_cell
            
        # Calculate dimensioned shunt capacitance
        self.CTLsec_F = self.g_C_mod / (self.Z0_TWPA_ohm * 2 * np.pi * self.fc_TLsec_GHz * 1E9)
        
    def _calculate_filter_only(self):
        """Calculate parameters for filter-only dispersion."""        
        self.nTLsec = self.nTLsec & ~1  # Clear LSB to make even
        self.ngL = 1
        self.ngC = 1
        self.width = 0
        
        # Calculate filter components
        # Pass total inductance of all n_jj_struct structures for filter design
        L0_total_H = self.n_jj_struct * self.L0_H
        results = calculate_filter_components(
            self.Foster_form_L, self.Foster_form_C, self.g_L, self.g_C,
            self.w_zeros, self.w_poles, self.Z0_TWPA_ohm, self.fc_filter_GHz,
            self.zero_at_zero, L0_total_H, self.select_one_form
        )
        
        (self.LinfLF1_H, self.C0LF1_F, self.LiLF1_H, self.CiLF1_F, self.LinfLF1_rem_H,
         self.L0LF2_H, self.CinfLF2_F, self.LiLF2_H, self.CiLF2_F, self.L0LF2_rem_H,
         self.LinfCF1_H, self.C0CF1_F, self.LiCF1_H, self.CiCF1_F,
         self.L0CF2_H, self.CinfCF2_F, self.LiCF2_H, self.CiCF2_F,
         self.maxL_ind_H, self.maxL_cap_F, self.maxC_ind_H, self.maxC_cap_F) = results
         
        if self.nonlinearity == 'JJ':
            if self.Foster_form_L == 1:
                self.ind2jjstruct_ratio = self.LinfLF1_H / self.L0_H
            elif self.Foster_form_L == 2:
                self.ind2jjstruct_ratio = self.L0LF2_H / self.L0_H
                
        self.Ncpersc_cell = int(self.nTLsec + 1)
        self.Nsc_cell = int(np.round(self.Ntot_cell/self.Ncpersc_cell))
        self.Ntot_cell = int(self.Nsc_cell*self.Ncpersc_cell)
        self.n_periodic_sc = self.Nsc_cell
        self.window_type = 'boxcar'
        self.CTLsec_F = self.g_C / (self.Z0_TWPA_ohm * 2 * np.pi * self.fc_TLsec_GHz * 1E9)
        
    def _calculate_filter_and_periodic(self):
        """Calculate parameters for combined filter and periodic dispersion."""
        if self.g_C_mod is None:
            raise ValueError("g_C_mod must be calculated before filter+periodic calculations")
        
        if self.window_type == 'boxcar':
            self.ind_g_C_with_filters = list(range(
                self.Ncpersc_cell // self.n_filters_per_sc - 1, 
                self.Ncpersc_cell, 
                self.Ncpersc_cell // self.n_filters_per_sc
            ))
        else:
            if self.window_type == 'tukey' and self.width > 0:
                all_filter_positions = list(range(
                    self.Ntot_cell // (self.Nsc_cell*self.n_filters_per_sc) - 1,
                    self.Ntot_cell,
                    self.Ntot_cell // (self.Nsc_cell*self.n_filters_per_sc)
                ))
                
                self.ind_g_C_with_filters = []
                for pos in all_filter_positions:
                    if self.width <= pos < (self.Ntot_cell - self.width):
                        self.ind_g_C_with_filters.append(pos)
                        
                if self.verbose:
                    print(f"Total filter positions: {len(all_filter_positions)}")
                    print(f"Filters after excluding windows: {len(self.ind_g_C_with_filters)}")
                    print(f"Excluded {len(all_filter_positions) - len(self.ind_g_C_with_filters)} filters from window regions")
            else:
                self.ind_g_C_with_filters = list(range(
                    self.Ntot_cell // (self.Nsc_cell*self.n_filters_per_sc) - 1,
                    self.Ntot_cell,
                    self.Ntot_cell // (self.Nsc_cell*self.n_filters_per_sc)
                ))
                
        # Calculate filter components with modulated g_C
        # Pass total inductance of all n_jj_struct structures for filter design
        L0_total_H = self.n_jj_struct * self.L0_H
        results = calculate_filter_components(
            self.Foster_form_L, self.Foster_form_C, self.g_L, 
            self.g_C_mod[self.ind_g_C_with_filters],
            self.w_zeros, self.w_poles, self.Z0_TWPA_ohm, self.fc_filter_GHz,
            self.zero_at_zero, L0_total_H, self.select_one_form
        )
        
        (self.LinfLF1_H, self.C0LF1_F, self.LiLF1_H, self.CiLF1_F, self.LinfLF1_rem_H,
         self.L0LF2_H, self.CinfLF2_F, self.LiLF2_H, self.CiLF2_F, self.L0LF2_rem_H,
         self.LinfCF1_H, self.C0CF1_F, self.LiCF1_H, self.CiCF1_F,
         self.L0CF2_H, self.CinfCF2_F, self.LiCF2_H, self.CiCF2_F,
         self.maxL_ind_H, self.maxL_cap_F, self.maxC_ind_H, self.maxC_cap_F) = results
         
        if self.nonlinearity == 'JJ':
            if self.Foster_form_L == 1:
                self.ind2jjstruct_ratio = self.LinfLF1_H / self.L0_H
            elif self.Foster_form_L == 2:
                self.ind2jjstruct_ratio = self.L0LF2_H / self.L0_H
                
    def _finalize_filter_calculations(self):
        """Finalize filter calculations."""
        self.max_ind_H = np.max([self.maxL_ind_H, self.maxC_ind_H])
        self.max_cap_F = np.max([self.maxL_cap_F, self.maxC_cap_F])
        
        # Find indices for poles and zeros
        self.ind_f_zeros = np.zeros(self.n_zeros, dtype=int)
        self.ind_f_poles = np.zeros(self.n_poles, dtype=int)
        
        for i in range(self.n_zeros):
            self.ind_f_zeros[i] = np.argmin(np.abs(self.f_GHz - self.f_zeros_GHz[i]))
            
        for i in range(self.n_poles):
            self.ind_f_poles[i] = np.argmin(np.abs(self.f_GHz - self.f_poles_GHz[i]))

    ########################################################################################    

    def plot_modulation_profile(self):
        """Plot the capacitance modulation profile for periodic/both dispersion types."""
        if self.dispersion_type not in ['periodic', 'both']:
            if self.verbose:
                print("Modulation profile plot only available for periodic or both dispersion types")
            return
            
        if not hasattr(self, 'g_C_mod') or self.g_C_mod is None:
            raise ValueError("Must run calculate_derived_quantities() before plotting modulation profile")
            
        Nmax2plot_cell = 1000
        if Nmax2plot_cell > self.Ntot_cell:
            Nmax2plot_cell = self.Ntot_cell
            
        fig = plt.figure(figsize=(8.6/2.54, 2))
        subp = fig.add_subplot(111)
        subp.plot(range(1, Nmax2plot_cell), self.g_C_mod[1:Nmax2plot_cell]/self.g_C, 
                  linewidth=0.5, color=blue)
        plt.grid(True)
        plt.xlim((0, Nmax2plot_cell))
        plt.ylim((0, 2))
        plt.xlabel('Cell number')
        plt.ylabel('$C$')
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        
        fig.set_size_inches(4, 2)

        plt.tight_layout()
        plt.show(block=False)
        
        # Store figure reference
        self.results['modulation_profile_fig'] = fig

    ########################################################################################       

    def calculate_linear_response(self):
        """Calculate linear response of the ATL."""
        if self.verbose:
            print("\n=== Calculating Linear Response ===")

        # Start timer for windowed cases
        import time
        if self.dispersion_type in ['periodic', 'both'] and self.window_type != 'boxcar':
            start_time = time.time()
            print(f"Note: Windowed calculation may take longer...")
            print(f"Started at {time.strftime('%H:%M:%S')}")
            
        # Initialize ABCD matrices
        self.ABCD_sc = np.zeros((2, 2, self.n_f), dtype=complex)
        self.ABCD = np.zeros((2, 2, self.n_f), dtype=complex)
        
        # Calculate based on dispersion type
        if self.dispersion_type == 'filter':
            self._calculate_filter_response()
        elif self.dispersion_type == 'periodic' and self.window_type == 'boxcar':
            self._calculate_periodic_boxcar_response()
        elif self.dispersion_type == 'periodic' and self.window_type != 'boxcar':
            self._calculate_periodic_windowed_response()
        elif self.dispersion_type == 'both' and self.window_type == 'boxcar':
            self._calculate_both_boxcar_response()
        elif self.dispersion_type == 'both' and self.window_type != 'boxcar':
            self._calculate_both_windowed_response()
            
        # Calculate S-parameters and Bloch impedance
        with np.errstate(over='ignore'):
            self.S21 = 2 / (self.ABCD[0,0,:] + self.ABCD[0,1,:]/self.Z0_ohm + 
                            self.ABCD[1,0,:]*self.Z0_ohm + self.ABCD[1,1,:])
            
        # Calculate Bloch impedance
        self.Zbloch_ohm = np.abs(np.real(
            (self.ABCD_sc[0,0,:] - self.ABCD_sc[1,1,:] + 
            np.sqrt((self.ABCD_sc[1,1,:] + self.ABCD_sc[0,0,:] - 2) * 
                    (self.ABCD_sc[1,1,:] + self.ABCD_sc[0,0,:] + 2))) / 
            (2 * self.ABCD_sc[1,0,:])
        ))
        
        # Calculate k (dispersion relation)
        self._calculate_dispersion()

        # Print elapsed time for windowed cases
        if self.dispersion_type in ['periodic', 'both'] and self.window_type != 'boxcar':
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"✓ Linear response calculated in {mins:02d}:{secs:02d}")
        else:
            print("✓ Linear response calculated")
        
        # Store results
        self.results['linear_response'] = {
            'S21': self.S21,
            'Zbloch_ohm': self.Zbloch_ohm,
            'k_radpercell': self.k_radpercell,
            'ABCD': self.ABCD,
            'ABCD_sc': self.ABCD_sc
        }

    

    def _calculate_filter_response(self):
        """Calculate linear response for filter-only dispersion."""
        self.ABCD_TLsec = np.zeros((2, 2, self.n_f), dtype=complex)
        
        for i in range(self.n_f):
            # TL section
            if self.nTLsec != 0:
                self.ABCD_TLsec[:,:,i] = calculate_ABCD_TLsec(
                    self.L0_H, self.n_jj_struct, self.CJ_F, self.LTLsec_rem_H, 
                    self.CTLsec_F, self.w[i], self.LTLsec_H, self.nonlinearity
                )
                
            ABCD_filter = get_ABCD_filter(
                self.Foster_form_L, self.Foster_form_C, self.L0_H, self.n_jj_struct, 
                self.CJ_F, self.C0LF1_F, self.LiLF1_H, self.CiLF1_F, self.LinfLF1_rem_H,
                self.L0LF2_H, self.CinfLF2_F, self.LiLF2_H, self.CiLF2_F,
                self.LinfCF1_H, self.C0CF1_F, self.LiCF1_H, self.CiCF1_F,
                self.L0CF2_H, self.CinfCF2_F, self.LiCF2_H, self.CiCF2_F,
                self.w[i], self.n_poles, self.n_zeros, self.LinfLF1_H, self.nonlinearity
            )
            
            # Calculate supercell
            self.ABCD_sc[:,:,i] = (
                np.linalg.matrix_power(self.ABCD_TLsec[:,:,i], self.nTLsec//2) @ 
                ABCD_filter @ 
                np.linalg.matrix_power(self.ABCD_TLsec[:,:,i], self.nTLsec//2)
            )
            
            self.ABCD[:,:,i] = np.linalg.matrix_power(self.ABCD_sc[:,:,i], self.Nsc_cell)
            
    def _calculate_periodic_boxcar_response(self):
        """Calculate linear response for periodic dispersion with boxcar window."""
        # Type guard for CTLsec_F
        if isinstance(self.CTLsec_F, (int, float)):
            raise ValueError("CTLsec_F should be an array for periodic dispersion")

        for i in range(self.n_f):
            self.ABCD_sc[:,:,i] = np.eye(2)
            for j in range(self.ngC):
                ABCD_filter = calculate_ABCD_TLsec(
                    self.L0_H, self.n_jj_struct, self.CJ_F, self.LTLsec_rem_H, 
                    self.CTLsec_F[j], self.w[i], self.LTLsec_H, self.nonlinearity
                )
                self.ABCD_sc[:,:,i] = self.ABCD_sc[:,:,i] @ ABCD_filter
                
            self.ABCD[:,:,i] = np.linalg.matrix_power(self.ABCD_sc[:,:,i], self.Nsc_cell)
            
    def _calculate_periodic_windowed_response(self):
        """Calculate linear response for periodic dispersion with windowing."""
        # Type guard for CTLsec_F
        if isinstance(self.CTLsec_F, (int, float)):
            raise ValueError("CTLsec_F should be an array for periodic dispersion")
        
        for i in range(self.n_f):
            self.ABCD[:,:,i] = np.eye(2)
            self.ABCD_sc[:,:,i] = np.eye(2)
            idx_cell = 0
            
            # First window part
            for j in range(self.width):
                ABCD_filter = calculate_ABCD_TLsec(
                    self.L0_H, self.n_jj_struct, self.CJ_F, self.LTLsec_rem_H,
                    self.CTLsec_F[idx_cell], self.w[i], self.LTLsec_H, self.nonlinearity
                )
                self.ABCD[:,:,i] = self.ABCD[:,:,i] @ ABCD_filter
                idx_cell += 1
                
            # Periodic parts
            ABCD_filter = np.eye(2)
            for j in range(self.Ncpersc_cell):
                ABCD_periodic_sec = calculate_ABCD_TLsec(
                    self.L0_H, self.n_jj_struct, self.CJ_F, self.LTLsec_rem_H,
                    self.CTLsec_F[idx_cell], self.w[i], self.LTLsec_H, self.nonlinearity
                )
                ABCD_filter = ABCD_filter @ ABCD_periodic_sec
                idx_cell += 1
                
            self.ABCD_sc[:,:,i] = self.ABCD_sc[:,:,i] @ ABCD_filter
            self.ABCD[:,:,i] = self.ABCD[:,:,i] @ np.linalg.matrix_power(ABCD_filter, self.n_periodic_sc)
            
            idx_cell += self.Ncpersc_cell * (self.n_periodic_sc - 1)
            
            # Last window part
            for j in range(self.width):
                ABCD_filter = calculate_ABCD_TLsec(
                    self.L0_H, self.n_jj_struct, self.CJ_F, self.LTLsec_rem_H,
                    self.CTLsec_F[idx_cell], self.w[i], self.LTLsec_H, self.nonlinearity
                )
                self.ABCD[:,:,i] = self.ABCD[:,:,i] @ ABCD_filter
                idx_cell += 1
                
    def _calculate_both_boxcar_response(self):
        """Calculate linear response for combined filter+periodic with boxcar."""
        # Type guards
        if isinstance(self.CTLsec_F, (int, float)):
            raise ValueError("CTLsec_F should be an array for combined dispersion")
        if self.ind_g_C_with_filters is None:
            raise ValueError("ind_g_C_with_filters must be defined for combined dispersion")
        if isinstance(self.LinfCF1_H, (int, float)):
            raise ValueError("LinfCF1_H should be an array for combined dispersion")
        if isinstance(self.C0CF1_F, (int, float)):
            raise ValueError("C0CF1_F should be an array for combined dispersion")
        if isinstance(self.LiCF1_H, (int, float)):
            raise ValueError("LiCF1_H should be an array for combined dispersion")
        if isinstance(self.CiCF1_F, (int, float)):
            raise ValueError("CiCF1_F should be an array for combined dispersion")
        if isinstance(self.L0CF2_H, (int, float)):
            raise ValueError("L0CF2_H should be an array for combined dispersion")
        if isinstance(self.CinfCF2_F, (int, float)):
            raise ValueError("CinfCF2_F should be an array for combined dispersion")
        if isinstance(self.LiCF2_H, (int, float)):
            raise ValueError("LiCF2_H should be an array for combined dispersion")
        if isinstance(self.CiCF2_F, (int, float)):
            raise ValueError("CiCF2_F should be an array for combined dispersion")


        for i in range(self.n_f):
            self.ABCD_sc[:,:,i] = np.eye(2)
            p = 0
            
            for j in range(self.ngC):
                if j in self.ind_g_C_with_filters:
                    # Filter sections
                    ABCD_filter = get_ABCD_filter(
                        self.Foster_form_L, self.Foster_form_C, self.L0_H, self.n_jj_struct,
                        self.CJ_F, self.C0LF1_F, self.LiLF1_H, self.CiLF1_F, self.LinfLF1_rem_H,
                        self.L0LF2_H, self.CinfLF2_F, self.LiLF2_H, self.CiLF2_F,
                        self.LinfCF1_H[p], self.C0CF1_F[p], self.LiCF1_H[p], self.CiCF1_F[p],
                        self.L0CF2_H[p], self.CinfCF2_F[p], self.LiCF2_H[p], self.CiCF2_F[p],
                        self.w[i], self.n_poles, self.n_zeros, self.LinfLF1_H, self.nonlinearity
                    )
                    p += 1
                else:
                    ABCD_filter = calculate_ABCD_TLsec(
                        self.L0_H, self.n_jj_struct, self.CJ_F, self.LTLsec_rem_H,
                        self.CTLsec_F[j], self.w[i], self.LTLsec_H, self.nonlinearity
                    )
                self.ABCD_sc[:,:,i] = self.ABCD_sc[:,:,i] @ ABCD_filter
                
            self.ABCD[:,:,i] = np.linalg.matrix_power(self.ABCD_sc[:,:,i], self.Nsc_cell)

    def _calculate_both_windowed_response(self):
        """Calculate linear response for combined filter+periodic with windowing."""
        # Type guards
        if isinstance(self.CTLsec_F, (int, float)):
            raise ValueError("CTLsec_F should be an array for combined dispersion")
        if self.ind_g_C_with_filters is None:
            raise ValueError("ind_g_C_with_filters must be defined for combined dispersion")
        
        if isinstance(self.LinfCF1_H, (int, float)):
            raise ValueError("LinfCF1_H should be an array for combined dispersion")
        if isinstance(self.C0CF1_F, (int, float)):
            raise ValueError("C0CF1_F should be an array for combined dispersion")
        if isinstance(self.LiCF1_H, (int, float)):
            raise ValueError("LiCF1_H should be an array for combined dispersion")
        if isinstance(self.CiCF1_F, (int, float)):
            raise ValueError("CiCF1_F should be an array for combined dispersion")
        if isinstance(self.L0CF2_H, (int, float)):
            raise ValueError("L0CF2_H should be an array for combined dispersion")
        if isinstance(self.CinfCF2_F, (int, float)):
            raise ValueError("CinfCF2_F should be an array for combined dispersion")
        if isinstance(self.LiCF2_H, (int, float)):
            raise ValueError("LiCF2_H should be an array for combined dispersion")
        if isinstance(self.CiCF2_F, (int, float)):
            raise ValueError("CiCF2_F should be an array for combined dispersion")
            
        for i in range(self.n_f):
            self.ABCD[:,:,i] = np.eye(2)
            self.ABCD_sc[:,:,i] = np.eye(2)
            p = 0
            idx_cell = 0
            
            # First window part
            for j in range(self.width):
                if idx_cell in self.ind_g_C_with_filters:
                    # Filter sections
                    ABCD_filter = get_ABCD_filter(
                        self.Foster_form_L, self.Foster_form_C, self.L0_H, self.n_jj_struct,
                        self.CJ_F, self.C0LF1_F, self.LiLF1_H, self.CiLF1_F, self.LinfLF1_rem_H,
                        self.L0LF2_H, self.CinfLF2_F, self.LiLF2_H, self.CiLF2_F,
                        self.LinfCF1_H[p], self.C0CF1_F[p], self.LiCF1_H[p], self.CiCF1_F[p],
                        self.L0CF2_H[p], self.CinfCF2_F[p], self.LiCF2_H[p], self.CiCF2_F[p],
                        self.w[i], self.n_poles, self.n_zeros, self.LinfLF1_H, self.nonlinearity
                    )
                    p += 1
                else:
                    ABCD_filter = calculate_ABCD_TLsec(
                        self.L0_H, self.n_jj_struct, self.CJ_F, self.LTLsec_rem_H,
                        self.CTLsec_F[idx_cell], self.w[i], self.LTLsec_H, self.nonlinearity
                    )
                self.ABCD[:,:,i] = self.ABCD[:,:,i] @ ABCD_filter
                idx_cell += 1
                
            # Periodic parts
            ABCD_filter = np.eye(2)
            for j in range(self.Ncpersc_cell):
                if idx_cell in self.ind_g_C_with_filters:
                    ABCD_periodic_sec = get_ABCD_filter(
                        self.Foster_form_L, self.Foster_form_C, self.L0_H, self.n_jj_struct,
                        self.CJ_F, self.C0LF1_F, self.LiLF1_H, self.CiLF1_F, self.LinfLF1_rem_H,
                        self.L0LF2_H, self.CinfLF2_F, self.LiLF2_H, self.CiLF2_F,
                        self.LinfCF1_H[p], self.C0CF1_F[p], self.LiCF1_H[p], self.CiCF1_F[p],
                        self.L0CF2_H[p], self.CinfCF2_F[p], self.LiCF2_H[p], self.CiCF2_F[p],
                        self.w[i], self.n_poles, self.n_zeros, self.LinfLF1_H, self.nonlinearity
                    )
                    p += 1
                else:
                    ABCD_periodic_sec = calculate_ABCD_TLsec(
                        self.L0_H, self.n_jj_struct, self.CJ_F, self.LTLsec_rem_H,
                        self.CTLsec_F[idx_cell], self.w[i], self.LTLsec_H, self.nonlinearity
                    )
                    
                ABCD_filter = ABCD_filter @ ABCD_periodic_sec
                idx_cell += 1
                
            self.ABCD_sc[:,:,i] = self.ABCD_sc[:,:,i] @ ABCD_filter
            self.ABCD[:,:,i] = self.ABCD[:,:,i] @ np.linalg.matrix_power(ABCD_filter, self.n_periodic_sc)
            
            idx_cell += self.Ncpersc_cell * (self.n_periodic_sc - 1)
            p += self.n_filters_per_sc * (self.n_periodic_sc - 1)
            
            # Last window part
            for j in range(self.width):
                if idx_cell in self.ind_g_C_with_filters:
                    ABCD_filter = get_ABCD_filter(
                        self.Foster_form_L, self.Foster_form_C, self.L0_H, self.n_jj_struct,
                        self.CJ_F, self.C0LF1_F, self.LiLF1_H, self.CiLF1_F, self.LinfLF1_rem_H,
                        self.L0LF2_H, self.CinfLF2_F, self.LiLF2_H, self.CiLF2_F,
                        self.LinfCF1_H[p], self.C0CF1_F[p], self.LiCF1_H[p], self.CiCF1_F[p],
                        self.L0CF2_H[p], self.CinfCF2_F[p], self.LiCF2_H[p], self.CiCF2_F[p],
                        self.w[i], self.n_poles, self.n_zeros, self.LinfLF1_H, self.nonlinearity
                    )
                    p += 1
                else:
                    ABCD_filter = calculate_ABCD_TLsec(
                        self.L0_H, self.n_jj_struct, self.CJ_F, self.LTLsec_rem_H,
                        self.CTLsec_F[idx_cell], self.w[i], self.LTLsec_H, self.nonlinearity
                    )
                self.ABCD[:,:,i] = self.ABCD[:,:,i] @ ABCD_filter
                idx_cell += 1

    ########################################################################################       

    def _calculate_dispersion(self):
        """Calculate k (dispersion relation) from ABCD matrices."""
        # Calculate k_radpersc
        self.k_radpersc = np.zeros(self.n_f)
        for i in range(self.n_f):
            # The sign of C tells us which spatial harmonic we're on
            sign_val = np.sign(np.imag(self.ABCD_sc[1,0,i])) if np.imag(self.ABCD_sc[1,0,i]) != 0 else 1
            self.k_radpersc[i] = sign_val * np.imag(np.arccosh((self.ABCD_sc[0,0,i]+self.ABCD_sc[1,1,i])/2))
            
        # Process based on dispersion type
        if self.dispersion_type == 'filter':
            self._process_filter_dispersion()
        elif self.dispersion_type == 'periodic':
            self._process_periodic_dispersion()
        elif self.dispersion_type == 'both':
            self._process_both_dispersion()
            
    def _process_filter_dispersion(self):
        """Process dispersion for filter-only case."""
        # No unwrapping needed for filter case
        self.k_radpercell = self.k_radpersc / self.Ncpersc_cell
        
        # Find true zeros
        self.inds_true_zeros = np.ones(self.n_zeros, dtype=int)
        if self.nTLsec != 0:
            if self.zero_at_zero:
                for i in range(self.n_zeros):
                    if i+1 <= self.n_poles-1:
                        search_range = self.k_radpercell[self.ind_f_poles[i]:self.ind_f_poles[i+1]]
                        self.inds_true_zeros[i] = np.argmin(np.abs(search_range)) + self.ind_f_poles[i]
                    else:
                        search_range = self.k_radpercell[self.ind_f_poles[i]:]
                        self.inds_true_zeros[i] = np.argmin(np.abs(search_range)) + self.ind_f_poles[i]
        else:
            for i in range(self.n_zeros):
                self.inds_true_zeros[i] = np.argmin(np.abs(self.f_GHz - self.f_zeros_GHz[i]))
                
    def _process_periodic_dispersion(self):
        """Process dispersion for periodic case."""
        # Need to unwrap for periodic case
        self.k_radpercell = np.unwrap(self.k_radpersc) / self.Ncpersc_cell
        
    def _process_both_dispersion(self):
        """Process dispersion for combined filter+periodic case."""
        # Filter out None values and convert to numpy array
        valid_stopbands = [f for f in self.f_stopbands_GHz if f is not None]
        if not valid_stopbands:
            # No valid stopbands, just use simple division
            self.k_radpercell = self.k_radpersc / self.Ncpersc_cell
            return
            
        f_stopbands_array = np.array(valid_stopbands)
        
        # Find indices of stopband frequencies
        idx_stopband_freqs = np.argmin(np.abs(self.f_GHz[:, None] - f_stopbands_array), axis=0)
        idx_stopband_freqs = np.sort(idx_stopband_freqs)
        
        # Define thresholds
        re_gamma_threshold = 0.001
        jump_threshold = np.pi/2
        
        # Process each stopband
        for i in range(len(idx_stopband_freqs)):
            idx_center = idx_stopband_freqs[i]
            
            # The phase at the stopband center should be (i+1)*pi
            target_k_at_stopband = (i+1) * np.pi
            
            # Scan rightward from stopband center
            idx = idx_center
            while idx < self.n_f - 1:
                re_gamma = np.abs(np.real(np.arccosh((self.ABCD_sc[0,0,idx]+self.ABCD_sc[1,1,idx])/2)))
                
                if re_gamma < re_gamma_threshold:
                    break
                    
                current_k = self.k_radpersc[idx]
                
                if np.abs(current_k - target_k_at_stopband) > jump_threshold:
                    shift = target_k_at_stopband - current_k
                    self.k_radpersc[idx:] += shift
                    
                idx += 1
                
            # Scan leftward from stopband center
            idx = idx_center - 1
            while idx >= 0:
                re_gamma = np.abs(np.real(np.arccosh((self.ABCD_sc[0,0,idx]+self.ABCD_sc[1,1,idx])/2)))
                
                if re_gamma < re_gamma_threshold:
                    break
                    
                current_k = self.k_radpersc[idx]
                
                if np.abs(current_k - target_k_at_stopband) > jump_threshold:
                    shift = target_k_at_stopband - current_k
                    self.k_radpersc[idx] += shift
                    
                idx -= 1
                
        self.k_radpercell = self.k_radpersc / self.Ncpersc_cell

    ########################################################################################     

    def plot_linear_response(self, block=False):
        """Plot S21 magnitude and dispersion relation."""
        if not hasattr(self, 'S21') or not hasattr(self, 'k_radpercell'):
            raise ValueError("Must run calculate_linear_response() before plotting")
            
        fig = plt.figure(figsize=(8.6/2.54, 3))
        fig.subplots_adjust(hspace=0.3)
        
        # First subplot - S21
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(self.f_GHz, 20*np.log10(np.abs(self.S21)), color='k', linewidth=linewidth)
        ax1.set_xlim((min(self.f_GHz), max(self.f_GHz)))
        ax1.axhline(y=0, color='k', linestyle='-')
        ax1.set_ylim((-10, 2))
        ax1.set_ylabel(r'$|S_{21}|^2$ [dB]')
        ax1.grid(True)
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        
        # Second subplot - k
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(self.f_GHz, self.k_radpercell, color=black, linewidth=linewidth)
        ax2.set_xlim((min(self.f_GHz), max(self.f_GHz)))
        
        # Set y-axis limits and ticks based on dispersion type
        if self.dispersion_type != 'filter':
            ymin = (np.min(self.ind_param) - 2) * np.pi/self.Ncpersc_cell
            ymax = (np.max(self.ind_param) + 1) * np.pi/self.Ncpersc_cell
        else:
            ymin = -np.pi/self.Ncpersc_cell
            ymax = np.pi/self.Ncpersc_cell
            
        ax2.set_ylim(ymin, ymax)
        
        # Calculate tick positions and labels
        min_n = int(np.round(ymin / (np.pi/self.Ncpersc_cell)))
        max_n = int(np.round(ymax / (np.pi/self.Ncpersc_cell)))
        
        if self.dispersion_type != 'filter':
            n_values = np.arange(min_n, max_n + 1)
            tick_positions = n_values * np.pi/self.Ncpersc_cell
            tick_labels = ['0' if n == 0 else f'${n}\\pi/d$' if n not in [-1, 1] else 
                          '$\\pi/d$' if n == 1 else '$-\\pi/d$' for n in n_values]
        else:
            # For filter case, use half-integer steps
            n_values = np.arange(min_n * 2, max_n * 2 + 1) / 2
            tick_positions = n_values * np.pi/self.Ncpersc_cell
            tick_labels = []
            for n in n_values:
                if n == 0:
                    tick_labels.append('0')
                elif n == 1:
                    tick_labels.append('$\\pi$')
                elif n == -1:
                    tick_labels.append('$-\\pi$')
                elif n == 0.5:
                    tick_labels.append('$\\pi/2$')
                elif n == -0.5:
                    tick_labels.append('$-\\pi/2$')
                else:
                    tick_labels.append(f'${int(n*2)}\\pi/2$')
                    
        ax2.set_yticks(tick_positions)
        ax2.set_yticklabels(tick_labels)
        ax2.set_xlabel('frequency [GHz]')
        ax2.set_ylabel(r'$k$ [rad/cell]')
        ax2.grid(True)
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        
        plt.tight_layout()
        plt.show(block=block)
        
        # Store figure reference
        self.results['linear_response_fig'] = fig


    ########################################################################################     

    def calculate_phase_matching(self):
        """Calculate phase matching conditions."""
        if self.verbose:
            print("\n=== Calculating Phase Matching ===")
            
        if not hasattr(self, 'k_radpercell'):
            raise ValueError("Must run calculate_linear_response() before phase matching")
            
        # Find frequency indices
        self.ind_fa_min = np.argmin(np.abs(self.f_GHz - self.fa_min_GHz))
        self.ind_fa_max = np.argmin(np.abs(self.f_GHz - self.fa_max_GHz))
        
        # Initialize arrays
        self.delta_kPA = np.zeros(self.ind_fa_max - self.ind_fa_min)
        self.delta_betaPA = np.zeros(self.ind_fa_max - self.ind_fa_min)
        
        # Calculate phase matching for each pump frequency
        for i in range(self.ind_fa_min, self.ind_fa_max):
            if self.WM == '3WM':
                fs_GHz_val = self.f_GHz[i] / 2 + self.sig_above_idl * self.detsigidlGHz
                fi_GHz_val = self.f_GHz[i] / 2 - self.sig_above_idl * self.detsigidlGHz
            elif self.WM == '4WM':
                fs_GHz_val = self.f_GHz[i] + self.sig_above_idl * self.detsigidlGHz
                fi_GHz_val = self.f_GHz[i] - self.sig_above_idl * self.detsigidlGHz
                
            ind_sig_val = np.argmin(np.abs(self.f_GHz - fs_GHz_val))
            ind_idl_val = np.argmin(np.abs(self.f_GHz - fi_GHz_val))
            
            
            # Calculate phase mismatch
            idx = i - self.ind_fa_min
            if self.WM == '3WM':
                if self.dir_prop_PA == 'forw':
                    self.delta_kPA[idx] = (self.k_radpercell[i] - self.k_radpercell[ind_sig_val] - 
                                          self.k_radpercell[ind_idl_val])
                    self.delta_betaPA[idx] = (self.delta_kPA[idx] + np.abs(self.Ia0_A)**2 / 8 *
                        (self.xi_perA2 * self.k_radpercell[i] - 2 * self.xi_perA2 * self.k_radpercell[ind_sig_val] - 
                         2 * self.xi_perA2 * self.k_radpercell[ind_idl_val]))
                elif self.dir_prop_PA == 'back':
                    self.delta_kPA[idx] = (- self.k_radpercell[i] - self.k_radpercell[ind_sig_val] - 
                                          self.k_radpercell[ind_idl_val])
                    self.delta_betaPA[idx] = (self.delta_kPA[idx] + np.abs(self.Ia0_A)**2 / 8 *
                        (- self.xi_perA2 * self.k_radpercell[i] - 2 * self.xi_perA2 * self.k_radpercell[ind_sig_val] - 
                         2 * self.xi_perA2 * self.k_radpercell[ind_idl_val]))
            elif self.WM == '4WM':
                if self.dir_prop_PA == 'forw':
                    self.delta_kPA[idx] = (2 * self.k_radpercell[i] - self.k_radpercell[ind_sig_val] - 
                                          self.k_radpercell[ind_idl_val])
                    self.delta_betaPA[idx] = (self.delta_kPA[idx] + self.xi_perA2 * self.Ia0_A**2 / 4 *
                        (self.k_radpercell[i] - self.k_radpercell[ind_sig_val] - self.k_radpercell[ind_idl_val]))
                elif self.dir_prop_PA == 'back':
                    self.delta_kPA[idx] = (- 2 * self.k_radpercell[i] - self.k_radpercell[ind_sig_val] - 
                                          self.k_radpercell[ind_idl_val])
                    self.delta_betaPA[idx] = (self.delta_kPA[idx] + self.xi_perA2 * self.Ia0_A**2 / 4 *
                        (- self.k_radpercell[i] - self.k_radpercell[ind_sig_val] - self.k_radpercell[ind_idl_val]))
                        
        # Find optimum pump frequency
        ind_min_delta_betaPA = np.argsort(np.abs(self.delta_betaPA))
        ind_min_delta_betaPA = ind_min_delta_betaPA + self.ind_fa_min
        
        self.ind_PA = ind_min_delta_betaPA[0]
        self.fa_GHz = self.f_GHz[self.ind_PA]
        
        # Check that pump frequency is not in stop band
        k = 0
        while 20 * np.log10(np.abs(self.S21[self.ind_PA])) < -20:
            k += 1
            if k >= len(ind_min_delta_betaPA):
                k -= 1
                break
            self.fa_GHz = self.f_GHz[ind_min_delta_betaPA[k]]
            self.ind_PA = np.argmin(np.abs(self.f_GHz - self.fa_GHz))
            
        # Calculate signal and idler frequencies
        if self.WM == '3WM':
            self.fs_GHz = self.fa_GHz / 2 + self.sig_above_idl * self.detsigidlGHz
            self.fi_GHz = self.fa_GHz / 2 - self.sig_above_idl * self.detsigidlGHz
        elif self.WM == '4WM':
            self.fs_GHz = self.fa_GHz + self.sig_above_idl * self.detsigidlGHz
            self.fi_GHz = self.fa_GHz - self.sig_above_idl * self.detsigidlGHz
            
        self.ind_sig = np.argmin(np.abs(self.f_GHz - self.fs_GHz))
        self.ind_idl = np.argmin(np.abs(self.f_GHz - self.fi_GHz))
        
        # Calculate phase velocity
        self._calculate_phase_velocity()
        
        # Store results
        self.results['phase_matching'] = {
            'delta_kPA': self.delta_kPA,
            'delta_betaPA': self.delta_betaPA,
            'fa_GHz': self.fa_GHz,
            'fs_GHz': self.fs_GHz,
            'fi_GHz': self.fi_GHz,
            'ind_PA': self.ind_PA,
            'ind_sig': self.ind_sig,
            'ind_idl': self.ind_idl,
            'lambda_PA_cell': self.lambda_PA_cell,
            'l_device_lambda_PA': self.l_device_lambda_PA
        }
        
    def _calculate_phase_velocity(self):
        """Calculate phase velocity and wavelength parameters."""
        # Calculate phase velocity
        if self.dispersion_type == 'filter':
            if self.Foster_form_L == 1 and self.Foster_form_C == 1:
                if self.C0CF1_F != np.inf and self.LinfLF1_H != 0:
                    v_cellpersec = np.mean(1 / np.sqrt(self.LinfLF1_H * self.C0CF1_F))
                elif self.C0CF1_F == np.inf and self.LinfLF1_H != 0:
                    v_cellpersec = np.mean(1 / np.sqrt(self.LinfLF1_H * check_flat(self.CiCF1_F, 0)))
                elif self.C0CF1_F != np.inf and self.LinfLF1_H == 0:
                    v_cellpersec = np.mean(1 / np.sqrt(check_flat(self.LiLF1_H, 0) * self.C0CF1_F))
                elif self.C0CF1_F == np.inf and self.LinfLF1_H == 0:
                    v_cellpersec = np.mean(1 / np.sqrt(check_flat(self.LiLF1_H, 0) * check_flat(self.CiCF1_F, 0)))
            elif self.Foster_form_L == 1 and self.Foster_form_C == 2:
                if self.LinfLF1_H != 0 and self.CinfCF2_F != 0:
                    v_cellpersec = np.mean(1 / np.sqrt(self.LinfLF1_H * self.CinfCF2_F))
                elif self.LinfLF1_H != 0 and self.CinfCF2_F == 0:
                    v_cellpersec = np.mean(1 / np.sqrt(self.LinfLF1_H * check_flat(self.CiCF2_F, 0)))
                elif self.LinfLF1_H == 0 and self.CinfCF2_F != 0:
                    v_cellpersec = np.mean(1 / np.sqrt(check_flat(self.LiLF1_H, 0) * self.CinfCF2_F))
                elif self.LinfLF1_H == 0 and self.CinfCF2_F == 0:
                    v_cellpersec = np.mean(1 / np.sqrt(check_flat(self.LiLF1_H, 0) * check_flat(self.CiCF2_F, 0)))
            elif self.Foster_form_L == 2 and self.Foster_form_C == 1:
                if self.L0LF2_H != np.inf and self.C0CF1_F != np.inf:
                    v_cellpersec = np.mean(1 / np.sqrt(self.L0LF2_H * self.C0CF1_F))
                elif self.L0LF2_H == np.inf and self.C0CF1_F != np.inf:
                    v_cellpersec = np.mean(1 / np.sqrt(check_flat(self.LiLF2_H, 0) * self.C0CF1_F))
                elif self.L0LF2_H != np.inf and self.C0CF1_F == np.inf:
                    v_cellpersec = np.mean(1 / np.sqrt(self.L0LF2_H * check_flat(self.CiCF1_F, 0)))
                elif self.L0LF2_H == np.inf and self.C0CF1_F == np.inf:
                    v_cellpersec = np.mean(1 / np.sqrt(check_flat(self.LiLF2_H, 0) * check_flat(self.CiCF1_F, 0)))
            elif self.Foster_form_L == 2 and self.Foster_form_C == 2:
                if self.L0LF2_H != np.inf and self.CinfCF2_F != 0:
                    v_cellpersec = np.mean(1 / np.sqrt(self.L0LF2_H * self.CinfCF2_F))
                elif self.L0LF2_H == np.inf and self.CinfCF2_F != 0:
                    v_cellpersec = np.mean(1 / np.sqrt(check_flat(self.LiLF2_H, 0) * self.CinfCF2_F))
                elif self.L0LF2_H != np.inf and self.CinfCF2_F == 0:
                    v_cellpersec = np.mean(1 / np.sqrt(self.L0LF2_H * check_flat(self.CiCF2_F, 0)))
                elif self.L0LF2_H == np.inf and self.CinfCF2_F == 0:
                    v_cellpersec = np.mean(1 / np.sqrt(check_flat(self.LiLF2_H, 0) * check_flat(self.CiCF2_F, 0)))
            
            # Include transmission line sections
            v_cellpersec = (v_cellpersec / self.Ncpersc_cell + 
                           self.nTLsec / np.sqrt(self.LTLsec_H * self.CTLsec_F) / self.Ncpersc_cell)
            self.v_cellpernsec = v_cellpersec * 1E-9
        elif self.dispersion_type == 'periodic' or self.dispersion_type == 'both':
            v_cellpersec = self.v_cellpernsec * 1E9
            
        self.lambda_PA_cell = v_cellpersec / (self.fa_GHz * 1E9)
        self.l_device_lambda_PA = self.Ntot_cell / self.lambda_PA_cell


    ########################################################################################     

    def plot_phase_matching(self, save_plot=False, filename=None, block=False):
        """Plot phase matching results in a 2x2 grid.
        
        Args:
            save_plot (bool): If True, saves the plot to designs/ folder
            filename (str): Optional custom filename (without extension)
        """
        if not hasattr(self, 'delta_betaPA') or not hasattr(self, 'fa_GHz'):
            raise ValueError("Must run calculate_phase_matching() before plotting")
            
        linewidth_vert_lines = 1
        
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(17.8/2.54, 4))
        axs = axs.flatten()
        
        # Subplot 1: S21 magnitude
        axs[0].set_facecolor('none')
        axs[0].plot(self.f_GHz, 20*np.log10(np.abs(self.S21)), color=black, linewidth=linewidth)
        axs[0].axvline(x=self.fa_GHz, color=purple, linewidth=linewidth_vert_lines)
        axs[0].axvline(x=self.fs_GHz, color=blue, linewidth=linewidth_vert_lines)
        axs[0].axvline(x=self.fi_GHz, color=darkblue, linewidth=linewidth_vert_lines)
        if self.WM == '3WM':
            axs[0].axvline(x=2*self.fa_GHz, color=purple, linewidth=linewidth_vert_lines)
        elif self.WM == '4WM':
            axs[0].axvline(x=3*self.fa_GHz, color=purple, linewidth=linewidth_vert_lines)
            
        axs[0].legend(['$S_{21}$', 'a', 's', 'i'], loc='best')
        axs[0].set_xlim((0, self.fmax_GHz))
        axs[0].set_ylim((-10, 2))
        axs[0].set_xlabel('frequency [GHz]')
        axs[0].set_ylabel('$|S_{21}|^2$ [dB]')
        axs[0].grid(True)
        axs[0].tick_params(labelsize=fontsize)
        axs[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        
        # Subplot 2: k_radpercell
        axs[1].set_facecolor('none')
        axs[1].plot(self.f_GHz, self.k_radpercell, color=black, linewidth=linewidth)
        axs[1].axvline(x=self.fa_GHz, color=purple, linewidth=linewidth_vert_lines)
        axs[1].axvline(x=self.fs_GHz, color=blue, linewidth=linewidth_vert_lines)
        axs[1].axvline(x=self.fi_GHz, color=darkblue, linewidth=linewidth_vert_lines)
        
        # Set y-axis limits and ticks
        if self.dispersion_type != 'filter':
            ymin = (np.min(self.ind_param) - 2) * np.pi/self.Ncpersc_cell
            ymax = (np.max(self.ind_param) + 1) * np.pi/self.Ncpersc_cell
        else:
            ymin = -np.pi/self.Ncpersc_cell
            ymax = np.pi/self.Ncpersc_cell
            
        axs[1].set_ylim((ymin, ymax))
        
        # Calculate tick positions and labels
        min_n = int(np.round(ymin / (np.pi/self.Ncpersc_cell)))
        max_n = int(np.round(ymax / (np.pi/self.Ncpersc_cell)))
        
        if self.dispersion_type != 'filter':
            n_values = np.arange(min_n, max_n + 1)
            tick_positions = n_values * np.pi/self.Ncpersc_cell
            tick_labels = ['0' if n == 0 else f'${n}\\pi/d$' if n not in [-1, 1] else 
                        '$\\pi/d$' if n == 1 else '$-\\pi/d$' for n in n_values]
        else:
            n_values = np.arange(min_n * 2, max_n * 2 + 1) / 2
            tick_positions = n_values * np.pi/self.Ncpersc_cell
            tick_labels = []
            for n in n_values:
                if n == 0:
                    tick_labels.append('0')
                elif n == 1:
                    tick_labels.append('$\\pi$')
                elif n == -1:
                    tick_labels.append('$-\\pi$')
                elif n == 0.5:
                    tick_labels.append('$\\pi/2$')
                elif n == -0.5:
                    tick_labels.append('$-\\pi/2$')
                else:
                    tick_labels.append(f'${int(n*2)}\\pi/2$')
                    
        axs[1].set_yticks(tick_positions)
        axs[1].set_yticklabels(tick_labels)
        axs[1].set_xlabel('frequency [GHz]')
        axs[1].set_ylabel('k [rad/cell]')
        axs[1].grid(True)
        axs[1].axhline(y=0, color='gray', linestyle='-')
        
        if self.WM == '3WM':
            axs[1].set_xlim((min(self.f_GHz), np.floor(self.fa_GHz/10)*10+10))
        elif self.WM == '4WM':
            axs[1].set_xlim((min(self.f_GHz), np.floor(2*self.fa_GHz/10)*10+10))
        axs[1].tick_params(labelsize=fontsize)
        axs[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        
        # Subplot 3: Zbloch
        axs[2].set_facecolor('none')
        axs[2].plot(self.f_GHz, self.Zbloch_ohm, color=black, linewidth=linewidth)
        axs[2].axvline(x=self.fa_GHz, color=purple, linewidth=linewidth_vert_lines)
        axs[2].axvline(x=self.fs_GHz, color=blue, linewidth=linewidth_vert_lines)
        axs[2].axvline(x=self.fi_GHz, color=darkblue, linewidth=linewidth_vert_lines)
        axs[2].set_xlabel('frequency [GHz]')
        axs[2].set_ylabel('$Z_\\mathrm{Bloch}$ [Ohm]')
        axs[2].axhline(y=50, color=black)
        axs[2].grid(True)
        
        if self.WM == '3WM':
            axs[2].set_xlim((min(self.f_GHz), np.floor(self.fa_GHz/10)*10+10))
        elif self.WM == '4WM':
            axs[2].set_xlim((min(self.f_GHz), np.floor(2*self.fa_GHz/10)*10+10))
        axs[2].set_ylim((0, 100))
        axs[2].tick_params(labelsize=fontsize)
        axs[2].xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        axs[2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        
        # Subplot 4: delta_betaPA
        axs[3].set_facecolor('none')
        if self.WM == '3WM':
            axs[3].plot(self.f_GHz[self.ind_fa_min:self.ind_fa_max]/2, 
                    np.real(self.delta_kPA), '--', color=black, linewidth=linewidth)
            axs[3].plot(self.f_GHz[self.ind_fa_min:self.ind_fa_max]/2, 
                    np.real(self.delta_betaPA), color=black, linewidth=linewidth)
            axs[3].set_xlim((self.f_GHz[self.ind_fa_min]/2, self.f_GHz[self.ind_fa_max]/2))
            axs[3].set_xlabel('half pump frequency [GHz]')
        elif self.WM == '4WM':
            axs[3].plot(self.f_GHz[self.ind_fa_min:self.ind_fa_max], 
                    np.real(self.delta_kPA), '--', color=black, linewidth=linewidth)
            axs[3].plot(self.f_GHz[self.ind_fa_min:self.ind_fa_max], 
                    np.real(self.delta_betaPA), color=black, linewidth=linewidth)
            axs[3].set_xlim((self.f_GHz[self.ind_fa_min], self.f_GHz[self.ind_fa_max]))
            axs[3].set_xlabel('pump frequency [GHz]')
            
        if max(np.abs(np.real(self.delta_betaPA))) > 0.2:
            axs[3].set_ylim((-0.2, 0.2))
            
        axs[3].axhline(y=0, color='gray', linestyle='-', zorder=1)
        axs[3].axvline(x=self.fa_GHz, color=purple, linewidth=linewidth_vert_lines)
        axs[3].set_ylabel('$\\Delta_\\beta^\\mathrm{PA}$ [rad/cell]')
        axs[3].grid(True)
        axs[3].tick_params(labelsize=fontsize)
        axs[3].legend(['pump off', f'$I_a$ = {self.Ia0_uA:.1f} µA'], loc='best')
        axs[3].xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        axs[3].yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            from pathlib import Path
            import os
            from twpa_design.helper_functions import filecounter
            from twpa_design import DESIGNS_DIR
            
            # Create designs folder path
            designs_dir = DESIGNS_DIR
            designs_dir.mkdir(exist_ok=True)
            
            if filename is None:
                # Auto-generate filename with file counter, matching design export convention
                base_pattern = f'{self.device_name}_*.svg'
                full_pattern = os.path.join(designs_dir, base_pattern)
                save_path, _ = filecounter(full_pattern)
            else:
                # Ensure filename doesn't have extension
                if filename.endswith('.svg'):
                    filename = filename[:-4]
                save_path = os.path.join(designs_dir, f"{filename}.svg")
            fig.savefig(save_path, format='svg', bbox_inches='tight')
            if self.verbose:
                print(f"✓ Phase matching plot saved to: {save_path}")
        
        plt.show(block=block)
        
        # Store figure reference
        self.results['phase_matching_fig'] = fig

    ########################################################################################   

    def print_parameters(self):
        """Print output parameters summary."""
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        if self.nonlinearity == 'JJ':
            print(f'plasma frequency = {self.fJ_GHz:.2f} GHz')
            print(f'JJ critical current = {self.Ic_JJ_uA} uA')
            print(f'shunt capacitance = {self.CJ_F*1E15:.2f} fF')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(f'phi_dc/(2*pi) that minimizes gamma: phi_dc/(2*pi) = {self.phi_dc/(2*np.pi):.2f}')
            print(f'it corresponds to phi_ext/(2*pi) = {self.phi_ext/(2*np.pi):.2f}')
            print(f'it also corresponds to a dc current Id = {self.Id_uA:.2f} uA')
            print(f'circulating current through the rf-SQUIDs at this bias point J = {self.J_uA:.2f} uA')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(f'n_jj_struct = {self.n_jj_struct}')
            print(f'ind2jjstruct ratio = {check_flat(self.ind2jjstruct_ratio, 0):.2f}')
        elif self.nonlinearity == 'KI':
            print(f'line cutoff frequency = {self.fc_TLsec_GHz:.2f} GHz')
            print(f'KI Istar = {self.Istar_uA:.2f} uA')
            print(f'dc bias Id = {self.Id_uA:.2f} uA')
            
        print(f'static inductance at the bias point LTLsec_H = {self.LTLsec_H*1E12:.2f} pH')
        
        if self.dispersion_type == 'filter' or self.dispersion_type == 'both':
            print(f'mean capacitance at the bias point CTLsec_F = {np.mean(self.CTLsec_F)*1E15:.2f} fF')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(f'pump amplitude: Ia = {self.Ia0_uA:.2f} uA')
            print(f'phase velocity under bias (at the first transmission zero): v = {np.real(self.v_cellpernsec):.2f} cell/ns')
            print(f'PA pump wavelength lambda_a= {self.lambda_PA_cell:.2f} cells')
            print(f'length of the device: l= {self.l_device_lambda_PA:.2f} lambda_PA')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            
            if self.dispersion_type == 'filter' and hasattr(self, 'inds_true_zeros'):
                print(f'zeros at f = {[f"{freq:.2f}" for freq in self.f_GHz[self.inds_true_zeros]]} GHz')
            print(f'poles at f = {self.f_poles_GHz} GHz')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(f'number of cells per supercell: Ncpersc_cell = {self.Ncpersc_cell}')
            print(f'total number of cells: Ntot = {self.Ntot_cell}')
            print(f'number of supercells: Ntot = {self.Nsc_cell}')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(f'for signal and idler separated by {2*self.detsigidlGHz} GHz')
            print(f'phase matched amplification pump frequency: fp = {self.fa_GHz:.2f} GHz')
            print(f'phase matched signal frequency: fs = {self.fs_GHz:.2f} GHz')
            print(f'phase matched idler frequency: fi = {self.fi_GHz:.2f} GHz')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(f'initial filter cutoff frequency {self.fc_filter_GHz:.2f} GHz')
            print(f'maximum inductance in the filter {self.max_ind_H*1E9:.2f} nH')
            print(f'maximum capacitance in the filter {self.max_cap_F*1E12:.2f} pF')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')    

        
    ########################################################################################  

    def get_results(self):
        """
        Get comprehensive results in the same format as export_parameters.
        
        Returns
        -------
        dict
            Dictionary with 'config', 'circuit', and 'characteristics' sections
        """        
        
        # 1. CONFIGURATION PARAMETERS
        config = {}
        
        # Basic parameters (always needed)
        config['device_name'] = self.device_name
        config['f_step_MHz'] = self.f_step_MHz
        config['fmax_GHz'] = self.fmax_GHz
        config['nonlinearity'] = self.nonlinearity
        config['window_type'] = self.window_type
        
        # Filter parameters (only for 'filter' or 'both')
        if self.dispersion_type in ['filter', 'both']:
            config['f_zeros_GHz'] = save_array_intelligently(self.f_zeros_GHz)
            config['f_poles_GHz'] = save_array_intelligently(self.f_poles_GHz)
            config['fc_filter_GHz'] = save_parameter_intelligently(self.fc_filter_GHz)
            config['fc_TLsec_GHz'] = save_parameter_intelligently(self.fc_TLsec_GHz)
            config['Foster_form_L'] = self.Foster_form_L
            config['Foster_form_C'] = self.Foster_form_C
            config['select_one_form'] = self.select_one_form
            
        # Periodic modulation parameters (only for 'periodic' or 'both')
        if self.dispersion_type in ['periodic', 'both']:
            config['stopbands_config_GHz'] = self.stopbands_config_GHz
            config['alpha'] = save_parameter_intelligently(self.alpha)
            config['n_filters_per_sc'] = self.n_filters_per_sc
            
        # JJ-specific parameters (only for JJ nonlinearity)
        if self.nonlinearity == 'JJ':
            config['jj_structure_type'] = self.jj_structure_type
            config['Ic_JJ_uA'] = save_parameter_intelligently(self.Ic_JJ_uA)
            config['fJ_GHz'] = save_parameter_intelligently(self.fJ_GHz)
            config['beta_L'] = save_parameter_intelligently(self.beta_L)
            config['phi_dc'] = save_parameter_intelligently(self.phi_dc)            
            
        # KI-specific parameters (only for KI nonlinearity)
        elif self.nonlinearity == 'KI':
            config['Istar_uA'] = save_parameter_intelligently(self.Istar_uA)
            config['Id_uA'] = save_parameter_intelligently(self.Id_uA)
            config['L0_pH'] = save_parameter_intelligently(self.L0_pH)
            
        # Phase-matching parameters (always needed)
        config['WM'] = self.WM
        config['dir_prop_PA'] = self.dir_prop_PA
        config['Ia0_uA'] = save_parameter_intelligently(self.Ia0_uA)
        config['detsigidlGHz'] = save_parameter_intelligently(self.detsigidlGHz)
        config['fa_min_GHz'] = save_parameter_intelligently(self.fa_min_GHz)
        config['fa_max_GHz'] = save_parameter_intelligently(self.fa_max_GHz)
        
        # TWPA line parameters (always needed)
        config['Ntot_cell'] = self.Ntot_cell
        config['nTLsec'] = self.nTLsec
        config['n_jj_struct'] = self.n_jj_struct
        config['Z0_TWPA_ohm'] = save_parameter_intelligently(self.Z0_TWPA_ohm)
        
        # 2. CIRCUIT COMPONENTS
        circuit = {}
        
        # Derived dispersion type
        circuit['dispersion_type'] = self.dispersion_type
        
        # Structure parameters (always needed - check if they exist)
        if hasattr(self, 'Nsc_cell'):
            circuit['Nsc_cell'] = self.Nsc_cell
            circuit['Ncpersc_cell'] = self.Ncpersc_cell
            circuit['ngL'] = getattr(self, 'ngL', 1)
            circuit['ngC'] = getattr(self, 'ngC', 1)
            
        # Window parameters (only for 'both' with tukey)
        if self.dispersion_type == 'both' and self.window_type == 'tukey':
            circuit['width'] = getattr(self, 'width', 0)
            circuit['n_periodic_sc'] = getattr(self, 'n_periodic_sc', self.Nsc_cell)
        else:
            circuit['width'] = getattr(self, 'width', 0)
            circuit['n_periodic_sc'] = getattr(self, 'n_periodic_sc', self.Nsc_cell)
            
        # Basic electrical components (always needed if calculated)
        if hasattr(self, 'L0_H'):
            circuit['L0_H'] = save_parameter_intelligently(self.L0_H)
            circuit['C0_F'] = save_parameter_intelligently(self.C0_F)
            
        # TL section parameters (only when nTLsec > 0)
        if hasattr(self, 'LTLsec_H') and getattr(self, 'nTLsec', 0) > 0:
            circuit['LTLsec_H'] = save_parameter_intelligently(self.LTLsec_H)
            circuit['LTLsec_rem_H'] = save_parameter_intelligently(self.LTLsec_rem_H)
            
        # CTLsec_F - only when nTLsec > 0 and handle differently for periodic vs non-periodic
        if hasattr(self, 'CTLsec_F') and getattr(self, 'nTLsec', 0) > 0:
            if self.dispersion_type not in ['periodic', 'both']:
                circuit['CTLsec_F'] = self.CTLsec_F
                
        # JJ-specific components
        if self.nonlinearity == 'JJ' and hasattr(self, 'LJ0_H'):
            circuit['LJ0_H'] = self.LJ0_H
            circuit['CJ_F'] = self.CJ_F
            circuit['Lg_H'] = self.Lg_H
            circuit['L0_H'] = self.L0_H
            
        # Nonlinearity coefficients (always needed for phase-matching calculations)
        if hasattr(self, 'epsilon_perA'):
            circuit['epsilon_perA'] = save_parameter_intelligently(self.epsilon_perA)
            circuit['xi_perA2'] = save_parameter_intelligently(self.xi_perA2)
            circuit['c1_taylor'] = save_parameter_intelligently(self.c1_taylor)
            circuit['c2_taylor'] = save_parameter_intelligently(self.c2_taylor)
            circuit['c3_taylor'] = getattr(self, 'c3_taylor', None)
            circuit['c4_taylor'] = getattr(self, 'c4_taylor', None)
            
        # Filter components (only for 'filter' or 'both')
        if self.dispersion_type in ['filter', 'both'] and hasattr(self, 'n_zeros'):
            circuit['n_zeros'] = self.n_zeros
            circuit['n_poles'] = self.n_poles
            
            # Helper function to ensure array format
            def ensure_array(value):
                if value is None:
                    return None
                elif isinstance(value, np.ndarray):
                    return value
                else:
                    return np.array([value])
                    
            # Add all filter component arrays
            filter_components = [
                'LinfLF1_H', 'LinfLF1_rem_H', 'C0LF1_F', 'LiLF1_H', 'CiLF1_F',
                'L0LF2_H', 'L0LF2_rem_H', 'CinfLF2_F', 'LiLF2_H', 'CiLF2_F',
                'LinfCF1_H', 'C0CF1_F', 'LiCF1_H', 'CiCF1_F',
                'L0CF2_H', 'CinfCF2_F', 'LiCF2_H', 'CiCF2_F'
            ]
            
            for comp in filter_components:
                if hasattr(self, comp):
                    value = getattr(self, comp)
                    # Use save_filter_array_intelligently for all filter components
                    circuit[comp] = save_filter_array_intelligently(value if isinstance(value, np.ndarray) else ensure_array(value))
                        
        # Periodic modulation arrays (only for 'periodic' or 'both')
        if self.dispersion_type in ['periodic', 'both']:
            if hasattr(self, 'ind_g_C_with_filters'):
                circuit['ind_g_C_with_filters'] = self.ind_g_C_with_filters
            circuit['n_filters_per_sc'] = getattr(self, 'n_filters_per_sc', 0)
            
            # Smart saving for periodic arrays
            if hasattr(self, 'CTLsec_F'):
                CTLsec_F_full = self.CTLsec_F
                
            if hasattr(self, 'g_C_mod'):
                g_C_mod_full = self.g_C_mod
                
                if self.window_type == 'boxcar':
                    # For boxcar, save only one supercell pattern
                    if g_C_mod_full is not None and len(g_C_mod_full) >= self.Ncpersc_cell:
                        circuit['g_C_pattern'] = save_filter_array_intelligently(g_C_mod_full[:self.Ncpersc_cell])
                    else:
                        circuit['g_C_pattern'] = save_filter_array_intelligently(g_C_mod_full)
                        
                    if hasattr(self, 'CTLsec_F') and isinstance(CTLsec_F_full, np.ndarray):
                        if len(CTLsec_F_full) >= self.Ncpersc_cell:
                            circuit['CTLsec_pattern'] = save_filter_array_intelligently(CTLsec_F_full[:self.Ncpersc_cell])
                    elif hasattr(self, 'CTLsec_F'):
                        circuit['CTLsec_F'] = CTLsec_F_full
                else:
                    # For windowed cases, save pattern + window data
                    width = getattr(self, 'width', 0)
                    
                    if g_C_mod_full is not None and width > 0:
                        if len(g_C_mod_full) > 2*width + self.Ncpersc_cell:
                            start_idx = width
                            circuit['g_C_pattern'] = save_filter_array_intelligently(g_C_mod_full[start_idx:start_idx + self.Ncpersc_cell])
                            circuit['g_C_window_start'] = save_filter_array_intelligently(g_C_mod_full[:width])
                            circuit['g_C_window_end'] = save_filter_array_intelligently(g_C_mod_full[-width:])
                        else:
                            circuit['g_C_pattern'] = save_filter_array_intelligently(g_C_mod_full[:self.Ncpersc_cell] if len(g_C_mod_full) >= self.Ncpersc_cell else g_C_mod_full)
                    else:
                        if g_C_mod_full is not None and len(g_C_mod_full) >= self.Ncpersc_cell:
                            circuit['g_C_pattern'] = save_filter_array_intelligently(g_C_mod_full[:self.Ncpersc_cell])
                        else:
                            circuit['g_C_pattern'] = save_filter_array_intelligently(g_C_mod_full)
                            
                    if hasattr(self, 'CTLsec_F') and isinstance(CTLsec_F_full, np.ndarray):
                        if width > 0 and len(CTLsec_F_full) > 2*width + self.Ncpersc_cell:
                            start_idx = width
                            circuit['CTLsec_pattern'] = save_filter_array_intelligently(CTLsec_F_full[start_idx:start_idx + self.Ncpersc_cell])
                            circuit['CTLsec_window_start'] = save_filter_array_intelligently(CTLsec_F_full[:width])
                            circuit['CTLsec_window_end'] = save_filter_array_intelligently(CTLsec_F_full[-width:])
                        else:
                            circuit['CTLsec_pattern'] = save_filter_array_intelligently(CTLsec_F_full[:self.Ncpersc_cell] if len(CTLsec_F_full) >= self.Ncpersc_cell else CTLsec_F_full)
                    elif hasattr(self, 'CTLsec_F'):
                        circuit['CTLsec_F'] = CTLsec_F_full
                        
                    if width > 0:
                        circuit['window_params'] = {
                            'type': self.window_type,
                            'alpha': getattr(self, 'alpha', 0),
                            'width': width,
                            'n_periodic_sc': getattr(self, 'n_periodic_sc', 0),
                        }
                        
        # 3. DEVICE CHARACTERISTICS
        characteristics = {}
        
        # Frequencies (if calculated)
        if hasattr(self, 'fa_GHz'):
            characteristics['fa_GHz'] = save_parameter_intelligently(self.fa_GHz)
            characteristics['fs_GHz'] = save_parameter_intelligently(self.fs_GHz)
            characteristics['fi_GHz'] = save_parameter_intelligently(self.fi_GHz)
            
        # Device-specific characteristics
        if self.nonlinearity == 'JJ':
            characteristics['jj_structure_type'] = self.jj_structure_type
            if self.jj_structure_type == 'rf_squid':
                characteristics['beta_L'] = save_parameter_intelligently(self.beta_L)
            characteristics['Ic_JJ_uA'] = save_parameter_intelligently(self.Ic_JJ_uA)
        elif self.nonlinearity == 'KI':
            characteristics['Istar_uA'] = save_parameter_intelligently(self.Istar_uA)
            
        characteristics['Ia_uA'] = save_parameter_intelligently(self.Ia0_uA)
        
        # Phase velocity and wavelength (if calculated)
        if hasattr(self, 'v_cellpernsec'):
            characteristics['v_cellpernsec'] = save_parameter_intelligently(self.v_cellpernsec)
            characteristics['lambda_PA_cell'] = save_parameter_intelligently(self.lambda_PA_cell)
            characteristics['l_device_lambda_PA'] = save_parameter_intelligently(self.l_device_lambda_PA)
            
        # Electrical characteristics (if calculated)
        if hasattr(self, 'Pa_dBm'):
            characteristics['Pa_dBm'] = save_parameter_intelligently(self.Pa_dBm)
            characteristics['fcmax_GHz'] = save_parameter_intelligently(self.fcmax_GHz)
            
        # Filter characteristics (if calculated)
        if self.dispersion_type in ['filter', 'both'] and hasattr(self, 'maxL_ind_H'):
            characteristics['maxL_ind_H'] = save_parameter_intelligently(self.maxL_ind_H)
            characteristics['maxL_cap_F'] = save_parameter_intelligently(self.maxL_cap_F)
            characteristics['maxC_ind_H'] = save_parameter_intelligently(self.maxC_ind_H)
            characteristics['maxC_cap_F'] = save_parameter_intelligently(self.maxC_cap_F)
            if hasattr(self, 'ind2jjstruct_ratio'):
                characteristics['ind2jjstruct_ratio'] = self.ind2jjstruct_ratio
                
        # JJ-specific characteristics (if calculated)
        if self.nonlinearity == 'JJ' and hasattr(self, 'phi_dc'):
            characteristics['phi_dc'] = self.phi_dc
            characteristics['phi_ext'] = self.phi_ext
            characteristics['J_uA'] = self.J_uA
            
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        circuit = {k: v for k, v in circuit.items() if v is not None}
        characteristics = {k: v for k, v in characteristics.items() if v is not None}
        
        return {
            'config': config,
            'circuit': circuit,
            'characteristics': characteristics
        }

    def export_parameters(self, filename=None):
        """Export parameters to Python file."""
        import os
        from datetime import datetime
        from pathlib import Path
        from twpa_design import DESIGNS_DIR
        
        # Get all results using the get_results method
        results = self.get_results()
        
        # Create designs folder path
        designs_folder = DESIGNS_DIR
        designs_folder.mkdir(exist_ok=True)
        
        # Auto-generate filename if not provided
        if filename is None:
            base_pattern = f'{self.device_name}_*.py'
            full_pattern = os.path.join(designs_folder, base_pattern)
            filename, _ = filecounter(full_pattern)
        else:
            filename = os.path.join(designs_folder, filename)
            
        # Write to file
        with open(filename, 'w') as f:
            f.write("# Auto-generated TWPA design parameters\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Device: {self.device_name}\n")
            f.write(f"# Nonlinearity: {self.nonlinearity}\n")
            f.write(f"# Dispersion type: {self.dispersion_type}\n\n")
            
            f.write("import numpy as np\n\n")
            
            # Write configuration
            f.write("# ========== CONFIGURATION ==========\n")
            f.write("# Parameters defining the TWPA design choices\n\n")
            f.write("config = {\n")
            for key, value in results['config'].items():
                self._write_parameter(f, key, value)
            f.write("}\n\n")
            
            # Write circuit components
            f.write("# ========== CIRCUIT COMPONENTS ==========\n")
            f.write("# Parameters needed to build the netlist\n\n")
            f.write("circuit = {\n")
            for key, value in results['circuit'].items():
                self._write_parameter(f, key, value)
            f.write("}\n\n")
            
            # Write characteristics
            f.write("# ========== DEVICE CHARACTERISTICS ==========\n")
            f.write("# Calculated performance parameters\n\n")
            f.write("characteristics = {\n")
            for key, value in results['characteristics'].items():
                self._write_parameter(f, key, value)
            f.write("}\n")
            
        if self.verbose:
            print(f"✓ Parameters exported to: {filename}")
            print(f"  Configuration parameters: {len(results['config'])}")
            print(f"  Circuit parameters: {len(results['circuit'])}")
            print(f"  Characteristics: {len(results['characteristics'])}")
            
        return filename
        
    def _write_parameter(self, f, key, value):
        """Helper method to write a parameter with proper formatting."""
        # Copy the write_parameter function from the notebook
        # This is the same function, just as a method
        
        # List of keys that should always be saved as arrays
        always_array_keys = [
            'LinfLF1_H', 'LinfLF1_rem_H', 'C0LF1_F', 'LiLF1_H', 'CiLF1_F',
            'L0LF2_H', 'L0LF2_rem_H', 'CinfLF2_F', 'LiLF2_H', 'CiLF2_F',
            'LinfCF1_H', 'C0CF1_F', 'LiCF1_H', 'CiCF1_F',
            'L0CF2_H', 'CinfCF2_F', 'LiCF2_H', 'CiCF2_F',
            'f_zeros_GHz', 'f_poles_GHz', 'CTLsec_F', 'g_C_mod',
            'ind_g_C_with_filters'
        ]
        
        if isinstance(value, np.ndarray):
            if value.size == 1 and key not in always_array_keys:
                single_val = value.item()
                if np.isinf(single_val):
                    f.write(f"    '{key}': np.inf,\n")
                elif np.isnan(single_val):
                    f.write(f"    '{key}': np.nan,\n")
                else:
                    f.write(f"    '{key}': {single_val},\n")
            else:
                # Keep as array
                if value.ndim == 1:
                    list_repr = []
                    for val in value:
                        if np.isinf(val):
                            list_repr.append('np.inf')
                        elif np.isnan(val):
                            list_repr.append('np.nan')
                        else:
                            list_repr.append(repr(float(val)))
                    f.write(f"    '{key}': np.array([{', '.join(list_repr)}]),\n")
                else:
                    # Multi-dimensional array
                    def format_nested_list(lst):
                        if isinstance(lst, list):
                            formatted = []
                            for item in lst:
                                formatted.append(format_nested_list(item))
                            return '[' + ', '.join(formatted) + ']'
                        else:
                            if np.isnan(lst):
                                return 'np.nan'
                            elif np.isinf(lst):
                                return 'np.inf'
                            else:
                                return repr(float(lst))
                    
                    formatted_array = format_nested_list(value.tolist())
                    f.write(f"    '{key}': np.array({formatted_array}),\n")
        elif isinstance(value, (float, np.floating)):
            if np.isinf(value):
                f.write(f"    '{key}': np.inf,\n")
            elif np.isnan(value):
                f.write(f"    '{key}': np.nan,\n")
            else:
                f.write(f"    '{key}': {float(value)},\n")
        elif isinstance(value, str):
            f.write(f"    '{key}': '{value}',\n")
        elif isinstance(value, dict):
            # Handle dictionary parameters (like stopbands_config_GHz)
            f.write(f"    '{key}': {{\n")
            for k, v in value.items():
                f.write(f"        {k}: {repr(v)},\n")
            f.write("    },\n")
        elif isinstance(value, (list, tuple)):
            list_repr = []
            for val in value:
                if isinstance(val, float) and np.isinf(val):
                    list_repr.append('np.inf')
                elif isinstance(val, float) and np.isnan(val):
                    list_repr.append('np.nan')
                else:
                    list_repr.append(repr(val))
            f.write(f"    '{key}': [{', '.join(list_repr)}],\n")
        else:
            f.write(f"    '{key}': {repr(value)},\n")

    ######################################################################################## 

    def run_design(self, interactive=True, save_results=False, save_plots=False):
        """
        Run the complete TWPA design workflow.
    
        Parameters
        ----------
        interactive : bool
            If True, pause at each plot and ask to continue
            If False, run everything without stopping
        save_results : bool
            If True, export parameters at the end
        save_plots : bool
            If True, save phase matching plot to designs/ folder as SVG
            
        Returns
        -------
        dict
            Results from get_results() method
        """
        print(f"\n{'='*60}")
        print(f"    Running TWPA Design: {self.device_name}")
        print(f"{'='*60}")
        
        # Step 1: Initial calculations
        print("\n[1/6] Initial calculations...")
        self.run_initial_calculations()
        
        # Step 2: Derived quantities
        print("\n[2/6] Derived quantities...")
        self.calculate_derived_quantities()
        
        # Step 3: Modulation profile (if applicable)
        if self.dispersion_type in ['periodic', 'both']:
            print("\n[3/6] Modulation profile")
            self.plot_modulation_profile()
            if interactive and not self._continue_prompt():
                return self._get_results_summary()
        
        # Step 4: Linear response
        print("\n[4/6] Linear response...")
        self.calculate_linear_response()
        self.plot_linear_response()
        if interactive and not self._continue_prompt():
            return self._get_results_summary()
        
        # Step 5: Phase matching
        print("\n[5/6] Phase matching...")
        self.calculate_phase_matching()
        self.plot_phase_matching(save_plot=save_plots)
        if interactive and not self._continue_prompt():
            return self._get_results_summary()
        
        # Step 6: Summary
        print("\n[6/6] Summary")
        self.print_parameters()
        
        # Save if requested
        if save_results:
            # Save design parameters (.py file)
            design_filename = self.export_parameters()
            print(f"\n✓ Design parameters saved to: {design_filename}")
            
            # Save data for plot regeneration (.npz file)
            data_filename = self.save_data()
            print(f"✓ Plot data saved to: {data_filename}")
        
        # Return the comprehensive results
        return self.get_results()

    def _continue_prompt(self):
        """Ask user if they want to continue while keeping plots visible."""
        response = input("\nContinue? [Y/n]: ").strip().lower()
        
        # Close all figures before continuing
        import matplotlib.pyplot as plt
        plt.close('all')
        
        return response != 'n'

    def _get_results_summary(self):
        """Get summary of key results."""
        return {
            'fa_GHz': getattr(self, 'fa_GHz', None),
            'fs_GHz': getattr(self, 'fs_GHz', None), 
            'fi_GHz': getattr(self, 'fi_GHz', None),
            'device_length': getattr(self, 'l_device_lambda_PA', None),
            'completed': hasattr(self, 'fa_GHz')
        }
    
    def save_data(self, filename=None):
        """
        Save all data needed to regenerate linear response and phase matching plots.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, auto-generates based on device_name
            
        Returns
        -------
        str
            Path to saved file
        """
        import os
        from datetime import datetime
        from pathlib import Path
        from twpa_design import DESIGNS_DIR
        from twpa_design.helper_functions import filecounter
        
        # Create designs folder path
        designs_folder = DESIGNS_DIR
        designs_folder.mkdir(exist_ok=True)
        
        # Auto-generate filename if not provided
        if filename is None:
            base_pattern = f'{self.device_name}_*.npz'
            full_pattern = os.path.join(designs_folder, base_pattern)
            filename, file_counter = filecounter(full_pattern)
        else:
            if not filename.endswith('.npz'):
                filename = filename + '.npz'
            filename = os.path.join(designs_folder, filename)
            
        # Collect all data needed for plots
        save_dict = {}
        
        # Configuration and metadata
        save_dict['device_name'] = self.device_name
        save_dict['dispersion_type'] = self.dispersion_type
        save_dict['nonlinearity'] = self.nonlinearity
        save_dict['WM'] = self.WM
        save_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Frequency array (common to both plots)
        if hasattr(self, 'f_GHz'):
            save_dict['f_GHz'] = self.f_GHz
            save_dict['fmax_GHz'] = self.fmax_GHz
            
        # Linear response data
        if hasattr(self, 'S21'):
            save_dict['S21'] = self.S21
            save_dict['k_radpercell'] = self.k_radpercell
            save_dict['Zbloch_ohm'] = self.Zbloch_ohm
            save_dict['ABCD'] = self.ABCD
            save_dict['ABCD_sc'] = self.ABCD_sc
            
        # Phase matching data
        if hasattr(self, 'delta_betaPA'):
            save_dict['delta_kPA'] = self.delta_kPA
            save_dict['delta_betaPA'] = self.delta_betaPA
            save_dict['fa_GHz'] = self.fa_GHz
            save_dict['fs_GHz'] = self.fs_GHz
            save_dict['fi_GHz'] = self.fi_GHz
            save_dict['ind_PA'] = self.ind_PA
            save_dict['ind_sig'] = self.ind_sig
            save_dict['ind_idl'] = self.ind_idl
            save_dict['lambda_PA_cell'] = self.lambda_PA_cell
            save_dict['l_device_lambda_PA'] = self.l_device_lambda_PA
            
        # Dispersion-specific parameters for plotting
        save_dict['Ncpersc_cell'] = getattr(self, 'Ncpersc_cell', 1)
        if self.dispersion_type != 'filter' and hasattr(self, 'ind_param'):
            save_dict['ind_param'] = self.ind_param
            
        # Get all configuration parameters for completeness
        config_results = self.get_results()
        save_dict['config'] = config_results['config']
        save_dict['circuit'] = config_results['circuit']
        save_dict['characteristics'] = config_results['characteristics']
        
        # Save to compressed npz file
        np.savez_compressed(filename, **save_dict)
        
        if self.verbose:
            print(f"✓ Data saved to: {filename}")
            print(f"  Linear response data: {'Yes' if 'S21' in save_dict else 'No'}")
            print(f"  Phase matching data: {'Yes' if 'delta_betaPA' in save_dict else 'No'}")
            
        return filename
    
    @classmethod
    def load_data(cls, filename):
        """
        Load saved data from npz file.
        
        Parameters
        ----------
        filename : str
            Path to npz file
            
        Returns
        -------
        dict
            Dictionary containing all saved data
        """
        import os
        from pathlib import Path
        from twpa_design import DESIGNS_DIR
        
        # Handle filename
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        # If no path specified, assume designs directory
        if os.path.sep not in filename and '/' not in filename:
            filename = os.path.join(str(DESIGNS_DIR), filename)
            
        # Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
            
        # Load data
        data = np.load(filename, allow_pickle=True)
        
        # Convert to regular dict for easier access
        result = {}
        for key in data.files:
            value = data[key]
            # Handle scalar arrays
            if value.ndim == 0:
                result[key] = value.item()
            else:
                result[key] = value
                
        print(f"✓ Data loaded from: {filename}")
        print(f"  Device: {result.get('device_name', 'Unknown')}")
        print(f"  Timestamp: {result.get('timestamp', 'Unknown')}")
        print(f"  Has linear response data: {'f_GHz' in result and 'S21' in result}")
        print(f"  Has phase matching data: {'delta_betaPA' in result}")
        
        return result
    
    @classmethod
    def plot_from_data(cls, data=None, filename=None, plot_linear=True, plot_phase=True):
        """
        Generate plots from saved data using the existing plot methods.
        
        Parameters
        ----------
        data : dict, optional
            Data dictionary from load_data(). If None, must provide filename
        filename : str, optional
            Path to npz file to load. Ignored if data is provided
        plot_linear : bool
            Whether to plot linear response
        plot_phase : bool
            Whether to plot phase matching
            
        Returns
        -------
        tuple
            (fig_linear, fig_phase) - matplotlib figure objects or None if not plotted
        """
        # Load data if not provided
        if data is None:
            if filename is None:
                raise ValueError("Must provide either data or filename")
            data = cls.load_data(filename)
        
        # Create a minimal instance with required attributes for plotting
        instance = cls.__new__(cls)
        
        # Initialize the results dictionary that plotting methods may access
        instance.results = {}
        
        # Set basic attributes from saved data
        instance.device_name = data.get('device_name', 'Unknown')
        instance.dispersion_type = data.get('dispersion_type', 'filter')
        instance.nonlinearity = data.get('nonlinearity', 'JJ')
        instance.WM = data.get('WM', '4WM')
        instance.verbose = False
        
        # Set frequency and response data
        if 'f_GHz' in data:
            instance.f_GHz = data['f_GHz']
            instance.fmax_GHz = data.get('fmax_GHz', 30)
        
        # Set linear response data
        if 'S21' in data:
            instance.S21 = data['S21']
            instance.k_radpercell = data['k_radpercell']
            instance.Zbloch_ohm = data.get('Zbloch_ohm', np.zeros_like(data['S21']))
            instance.ABCD = data.get('ABCD', np.zeros((2, 2, len(data['S21']))))
            instance.ABCD_sc = data.get('ABCD_sc', np.zeros((2, 2, len(data['S21']))))
        
        # Set phase matching data
        if 'delta_betaPA' in data:
            instance.delta_kPA = data['delta_kPA']
            instance.delta_betaPA = data['delta_betaPA']
            instance.fa_GHz = data['fa_GHz']
            instance.fs_GHz = data['fs_GHz']
            instance.fi_GHz = data['fi_GHz']
            instance.ind_PA = data.get('ind_PA', 0)
            instance.ind_sig = data.get('ind_sig', 0)
            instance.ind_idl = data.get('ind_idl', 0)
            instance.lambda_PA_cell = data.get('lambda_PA_cell', 0)
            instance.l_device_lambda_PA = data.get('l_device_lambda_PA', 0)
            
            # Calculate frequency indices needed for plotting (from configuration)
            config = data.get('config', {})
            fa_min_GHz = config.get('fa_min_GHz', 0)
            fa_max_GHz = config.get('fa_max_GHz', 30)
            instance.fa_min_GHz = fa_min_GHz
            instance.fa_max_GHz = fa_max_GHz
            instance.Ia0_uA = config.get('Ia0_uA', 1)
            instance.ind_fa_min = np.argmin(np.abs(instance.f_GHz - fa_min_GHz))
            instance.ind_fa_max = np.argmin(np.abs(instance.f_GHz - fa_max_GHz))
        
        # Set dispersion-specific parameters for plotting
        instance.Ncpersc_cell = data.get('Ncpersc_cell', 1)
        if instance.dispersion_type != 'filter' and 'ind_param' in data:
            instance.ind_param = data['ind_param']
        
        fig_linear = None
        fig_phase = None
        
        # Plot linear response using original method with block=True
        if plot_linear and hasattr(instance, 'S21'):
            instance.plot_linear_response(block=True)
            fig_linear = instance.results.get('linear_response_fig')
            
        # Plot phase matching using original method with block=True
        if plot_phase and hasattr(instance, 'delta_betaPA'):
            instance.plot_phase_matching(save_plot=False, block=True)
            fig_phase = instance.results.get('phase_matching_fig')
            
        return fig_linear, fig_phase