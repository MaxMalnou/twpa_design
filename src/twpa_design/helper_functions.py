# helper functions and other dependencies

from scipy import constants
import pickle
import os
import glob
import datetime
import numpy as np
import mpmath as mp
import IPython
import re
import math

import warnings
warnings.filterwarnings('ignore', 'Casting complex values to real discards the imaginary part')
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in multiply")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in scalar multiply")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in scalar subtract")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in scalar power")
np.seterr(divide='ignore', invalid='ignore')  # Suppress divide by zero and invalid value warnings

from scipy.optimize import basinhopping, minimize


################################################################################################

# Define global constants
phi0 = constants.hbar / (2 * constants.e)  # Flux quantum
Phi0 = constants.h / (2 * constants.e)  # Flux quantum
mu0 = constants.mu_0
c0 = constants.c

################################################################################################
# Define determinant function

def det_D(delta, ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband):
    """
    Determinant function that works with delta parameters directly.
    
    Parameters:
    delta (array): Array of delta values (free parameters)
    ind_stopband (array): Array of stopband indices
    ind_param (array): Array of stopband indices for parameters (non-default values)
    w_tilde_stopband_edges (array): Array of normalized frequencies at stopband edges
    is_default_value (array): Boolean array indicating which edges are default values
    max_ind_stopband (int): Maximum stopband index

    Returns:
    array: Determinant values at each stopband edge
    """
    n_tot_stopband_edges = len(w_tilde_stopband_edges)
    n_modes = 2 * max_ind_stopband + 1
    
    # Define k at the band edge
    k_norm = 0.5 * ind_stopband
    
    # Create the mode indices
    v_modes = np.arange(-np.floor(n_modes/2), np.floor(n_modes/2) + 1, dtype=int)
    
    detD_delta = []
    
    # loop over all stopband edges. 
    # Each stopband edge corresponds to a different frequency (and k_norm),
    # so we need a different matrix for each one.
    for j in range(n_tot_stopband_edges):
        # Create a numerical matrix using numpy
        D = np.zeros((n_modes, n_modes))
        
        # Fill diagonal terms
        for i in range(n_modes):
            D[i, i] = (k_norm[j] + v_modes[i])**2 - w_tilde_stopband_edges[j]**2
        
        # Skip default values
        if not is_default_value[j]:            
            # Fill off-diagonal terms exactly as in original function
            # All delta values are applied to each non-default stopband edge,
            # with each delta affecting a different off-diagonal
            for i in range(len(delta)):
                # Get the stopband index for this parameter
                sb_idx = ind_param[i]

                off_diag_val = delta[i] * w_tilde_stopband_edges[j]**2
                for k in range(n_modes - sb_idx):
                    D[k, k+sb_idx] -= off_diag_val
                    D[k+sb_idx, k] -= off_diag_val
        
        # Calculate determinant using numpy
        detD_delta.append(np.linalg.det(D))
    
    return np.array(detD_delta)



################################################################################################
# Solution-finding functions with precision control

def find_all_solutions(ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param, 
                      bounds=(-0.5, 0.5), n_attempts=30, tolerance=1e-8, always_return_best=True, verbose=True):
    """
    Find multiple solutions to the determinant system using basin-hopping with multiple starting points.
    
    Parameters:
    ind_stopband (array): Array of stopband indices
    ind_param (array): Array of stopband indices for parameters (non-default values)
    w_tilde_stopband_edges (array): Array of normalized frequencies at stopband edges
    is_default_value (array): Boolean array indicating which edges are default values
    max_ind_stopband (int): Maximum stopband index
    n_param (int): Number of free parameters
    bounds (tuple): Bounds for the solutions
    n_attempts (int): Number of random starting points to try
    tolerance (float): Tolerance for accepting solutions (lower = more precise)
    always_return_best (bool): If True, always return the best solution found, even if it doesn't meet the tolerance
    verbose (bool): Whether to print progress messages

    Returns:
    list: List of all unique solutions found
    """
    # Define the objective function to minimize (sum of squared determinants)
    def objective(delta_values):
        det_values = det_D(delta_values, ind_stopband, ind_param, w_tilde_stopband_edges, 
                          is_default_value, max_ind_stopband)
        return sum(d*d for d in det_values)
    
    # List to store unique solutions
    unique_solutions = []
    # Using the provided tolerance for solution validation
    solution_tolerance = tolerance
    bounds_list = [(bounds[0], bounds[1]) for _ in range(n_param)]
    
    # Track the best solution found even if it doesn't meet the tolerance
    best_solution = None
    best_objective_value = float('inf')
    
    # Try multiple basin-hopping runs with different starting points
    if verbose:
        print(f"Running {n_attempts} basin-hopping searches for {n_param} parameters...")
        print(f"Using solution tolerance: {solution_tolerance}")
    
    # Generate a diverse set of starting points
    starting_points = []
    
    # Include a set of evenly distributed points
    grid_points = np.linspace(bounds[0], bounds[1], int(np.ceil(np.power(n_attempts/2, 1/n_param))))
    
    if n_param == 1:
        for x in grid_points:
            starting_points.append(np.array([x]))
    elif n_param == 2:
        for x in grid_points:
            for y in grid_points:
                starting_points.append(np.array([x, y]))
    else:
        # For higher dimensions, just use random points
        for _ in range(n_attempts // 2):
            starting_points.append(np.random.uniform(bounds[0], bounds[1], n_param))
    
    # Add some completely random points to improve coverage
    for _ in range(n_attempts - len(starting_points)):
        starting_points.append(np.random.uniform(bounds[0], bounds[1], n_param))
    
    # Custom step-taking function to respect bounds
    def take_bounded_step(x):
        s = np.random.uniform(-0.1, 0.1, len(x))
        new_x = x + s
        # Ensure we stay within bounds
        new_x = np.clip(new_x, bounds[0], bounds[1])
        return new_x
    
    # Try each starting point
    for i, x0 in enumerate(starting_points):
        if verbose:
            print(f"Starting point {i+1}/{len(starting_points)}: {x0}")
        
        # Define minimizer parameters with tolerance-based options
        minimizer_kwargs = {
            "method": "L-BFGS-B", 
            "bounds": bounds_list,
            "options": {"ftol": tolerance/10, "gtol": tolerance/10, "maxiter": 1000}
        }
        
        # Run basin-hopping
        result = basinhopping(
            objective, 
            x0, 
            minimizer_kwargs=minimizer_kwargs, 
            niter=100,
            T=0.5,  # Temperature parameter (controls acceptance of uphill moves)
            stepsize=0.05,  # Initial step size
            take_step=take_bounded_step,
            interval=10  # Interval for step size adjustment
        )
        
        # Check if this is the best solution found so far
        current_objective = objective(result.x)
        if current_objective < best_objective_value:
            best_objective_value = current_objective
            best_solution = result.x
            if verbose:
                print(f"New best solution found (objective value: {best_objective_value:.2e})")
        
        # If solution is valid (objective function close to zero)
        if current_objective < solution_tolerance:
            # Check if this solution is unique
            is_unique = True
            for sol in unique_solutions:
                if np.allclose(result.x, sol, rtol=tolerance/10, atol=tolerance/10):
                    is_unique = False
                    break
            
            if is_unique:
                unique_solutions.append(result.x)
                if verbose:
                    print(f"Found valid solution (attempt {i+1}): {result.x}")
                
                # Verify solution
                det_values = det_D(result.x, ind_stopband, ind_param, w_tilde_stopband_edges, 
                                  is_default_value, max_ind_stopband)
                if verbose:
                    print("Verification:")
                    for j, val in enumerate(det_values):
                        print(f"Det{j+1} = {val:.2e}")
                    print()
        elif verbose:
            print(f"No valid solution found from starting point {i+1}, objective value: {current_objective:.2e}")
    
    # If no solutions meet the tolerance but we want the best effort solution
    if len(unique_solutions) == 0 and always_return_best and best_solution is not None:
        if verbose:
            print(f"\nNo solutions met the tolerance of {solution_tolerance}.")
            print(f"Returning best solution found (objective value: {best_objective_value:.2e})")
            
            # Verify best solution
            det_values = det_D(best_solution, ind_stopband, ind_param, w_tilde_stopband_edges, 
                              is_default_value, max_ind_stopband)
            print("Verification of best solution:")
            for j, val in enumerate(det_values):
                print(f"Det{j+1} = {val:.2e}")
            
        unique_solutions.append(best_solution)
    
    if verbose:
        print(f"\nFound {len(unique_solutions)} solutions (including best-effort solutions).")
    return unique_solutions


def multi_stage_solution_finding(ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param,
                                bounds=(-0.5, 0.5), n_attempts=20, tolerance=1e-8, always_return_best=True, verbose=True):
    """
    Multi-stage approach to find all solutions with increasing precision.
    
    Parameters:
    ind_stopband (array): Array of stopband indices
    ind_param (array): Array of stopband indices for parameters (non-default values)
    w_tilde_stopband_edges (array): Array of normalized frequencies at stopband edges
    is_default_value (array): Boolean array indicating which edges are default values
    max_ind_stopband (int): Maximum stopband index
    n_param (int): Number of free parameters
    bounds (tuple): Bounds for the solutions
    n_attempts (int): Number of starting points to try
    tolerance (float): Tolerance for accepting solutions (lower = more precise)
    always_return_best (bool): If True, always return the best solution found, even if it doesn't meet the tolerance
    verbose (bool): Whether to print progress messages
    
    Returns:
    list: List of refined solutions
    """
    # First stage: Find approximate solutions with relaxed tolerance
    first_stage_tolerance = tolerance * 100  # More relaxed in first stage
    if verbose:
        print(f"Stage 1: Finding approximate solutions (tolerance: {first_stage_tolerance})...")
    initial_solutions = find_all_solutions(
        ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param,
        bounds=bounds, 
        n_attempts=n_attempts,
        tolerance=first_stage_tolerance,
        always_return_best=always_return_best,
        verbose=verbose
    )
    
    # If no solutions found in Stage 1, return early
    if len(initial_solutions) == 0:
        if verbose:
            print("No solutions found in Stage 1, not even best-effort solutions. Skipping refinement.")
        return []
    
    # Second stage: Refine each solution with higher precision
    if verbose:
        print(f"\nStage 2: Refining solutions (tolerance: {tolerance})...")
    refined_solutions = []
    
    # Track the best refined solution
    best_refined_solution = None
    best_refined_objective_value = float('inf')
    
    # Define the objective function
    def objective(delta_values):
        det_values = det_D(delta_values, ind_stopband, ind_param, w_tilde_stopband_edges,
                          is_default_value, max_ind_stopband)
        return sum(d*d for d in det_values)
    
    for i, sol in enumerate(initial_solutions):
        if verbose:
            print(f"Refining solution {i+1}/{len(initial_solutions)}")
        
        # Use tight bounds around the initial solution
        bounds_list = []
        for val in sol:
            lower = max(bounds[0], val - 0.05)
            upper = min(bounds[1], val + 0.05)
            bounds_list.append((lower, upper))
        
        try:
            # High-precision minimization with the specified tolerance
            result = minimize(
                objective,
                sol,
                method='L-BFGS-B',
                bounds=bounds_list,
                options={'ftol': tolerance, 'gtol': tolerance, 'maxiter': 10000}
            )
            
            # Check if this is the best refined solution so far
            current_objective = objective(result.x)
            if current_objective < best_refined_objective_value:
                best_refined_objective_value = current_objective
                best_refined_solution = result.x
                if verbose:
                    print(f"New best refined solution (objective value: {best_refined_objective_value:.2e})")
            
            # Check if result is valid using the specified tolerance
            if current_objective < tolerance:
                refined_solutions.append(result.x)
                if verbose:
                    print(f"Refined solution meets tolerance: {result.x}")
                
                # Verify solution
                det_values = det_D(result.x, ind_stopband, ind_param, w_tilde_stopband_edges,
                                  is_default_value, max_ind_stopband)
                if verbose:
                    print("Verification:")
                    for j, val in enumerate(det_values):
                        print(f"Det{j+1} = {val:.2e}")
                    print()
            else:
                # Keep track of best sub-tolerance solution
                if verbose:
                    print(f"Refined solution does not meet tolerance. Objective value: {current_objective:.2e}")
        except Exception as e:
            # If refinement throws an error
            if verbose:
                print(f"Error during refinement: {e}")
                current_objective = objective(sol)
                print(f"Using original solution with objective value: {current_objective:.2e}")
            
            # Check if the original solution is better than our best so far
            current_objective = objective(sol)
            if current_objective < best_refined_objective_value:
                best_refined_objective_value = current_objective
                best_refined_solution = sol
    
    # If no solutions meet the tolerance but we want the best effort solution
    if len(refined_solutions) == 0 and always_return_best and best_refined_solution is not None:
        if verbose:
            print(f"\nNo refined solutions met the tolerance of {tolerance}.")
            print(f"Returning best refined solution found (objective value: {best_refined_objective_value:.2e})")
            
            # Verify best solution
            det_values = det_D(best_refined_solution, ind_stopband, ind_param, w_tilde_stopband_edges, 
                              is_default_value, max_ind_stopband)
            print("Verification of best refined solution:")
            for j, val in enumerate(det_values):
                print(f"Det{j+1} = {val:.2e}")
            
        refined_solutions.append(best_refined_solution)
    
    if verbose:
        print(f"\nFinal results: Found {len(refined_solutions)} refined solutions.")
    
    # Sort solutions for easier comparison
    if len(refined_solutions) > 0:
        for i in range(n_param):
            if verbose:
                print(f"\nSolutions sorted by delta{i+1}:")
            sorted_sols = sorted(refined_solutions, key=lambda x: x[i])
            if verbose:
                for j, sol in enumerate(sorted_sols):
                    print(f"Solution {j+1}: {sol} (delta{i+1} = {sol[i]:.6f})")
    
    return refined_solutions

################################################################################################
################################################################################################
################################################################################################

# Derive dispersion_type from the parameters
def derive_dispersion_type(f_zeros_GHz, f_poles_GHz, f_stopbands_GHz):
    """Automatically determine dispersion type from provided parameters."""
    
    def has_content(param):
        """Check if parameter has content (handles various input formats)."""
        if param is None:
            return False
        elif isinstance(param, (int, float)):
            return True  # Single number means content
        elif isinstance(param, (list, dict)):
            return len(param) != 0
        elif isinstance(param, np.ndarray):
            return len(param) != 0
        else:
            return False
    
    has_filter = has_content(f_zeros_GHz) or has_content(f_poles_GHz)
    has_periodic = has_content(f_stopbands_GHz)

    if has_filter and has_periodic:
        return 'both'
    elif has_periodic:
        return 'periodic'
    else:
        return 'filter'  # Default to filter if nothing specified
        
def ensure_numpy_array(param):
      """Convert parameter to numpy array regardless of input format."""
      if param is None:
          return np.array([])
      elif isinstance(param, (int, float)):
          return np.array([param])  # Single value -> array
      elif isinstance(param, list):
          return np.array(param)    # List -> array
      elif isinstance(param, np.ndarray):
          return param              # Already array -> keep as is
      else:
          return np.array([])       # Fallback for unexpected types

def save_array_intelligently(arr):
    """Convert numpy array back to user-friendly format for saving."""
    if not isinstance(arr, np.ndarray):
        return arr  # Not an array, return as is
    
    if len(arr) == 0:
        return []           # Empty array -> empty list
    elif len(arr) == 1:
        return float(arr[0]) if np.isreal(arr[0]) else complex(arr[0])  # Single element -> just the number
    else:
        return arr.tolist() # Multiple elements -> Python list

def save_parameter_intelligently(param):
    """Convert any parameter to a clean, readable format for saving."""
    if isinstance(param, np.ndarray):
        return save_array_intelligently(param)
    elif isinstance(param, (np.floating, np.integer)):
        return float(param)  # Convert numpy scalars to Python types
    elif isinstance(param, np.complexfloating):
        return complex(param)
    elif param is None:
        return None
    elif isinstance(param, (list, dict, str, bool)):
        return param  # Already clean
    elif isinstance(param, (int, float, complex)):
        return param  # Already clean
    else:
        return param  # Unknown type, return as-is

class FilterArray:
    """Custom array wrapper that formats correctly when written to file."""
    def __init__(self, array):
        self.array = array
        
    def __repr__(self):
        def format_value(val):
            if np.isnan(val):
                return 'np.nan'
            elif np.isinf(val):
                return 'np.inf' if val > 0 else '-np.inf'
            else:
                return repr(float(val) if np.isreal(val) else complex(val))
        
        if self.array.ndim == 1:
            values = [format_value(val) for val in self.array]
            return '[' + ', '.join(values) + ']'
        elif self.array.ndim == 2:
            rows = []
            for row in self.array:
                row_values = [format_value(val) for val in row]
                rows.append('[' + ', '.join(row_values) + ']')
            return '[' + ', '.join(rows) + ']'
        else:
            return str(self.array.tolist())
    
    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, key):
        return self.array[key]
    
    def __iter__(self):
        return iter(self.array)


def save_filter_array_intelligently(arr):
    """Convert filter arrays to clean format while preserving numpy special values.
    
    Returns a custom wrapper that formats nicely in exported files while keeping
    np.inf and np.nan syntax for compatibility with netlist_JC_builder.
    """
    if not isinstance(arr, np.ndarray):
        return arr  # Not an array, return as is
    
    if len(arr) == 0:
        return []  # Empty array -> empty list
    
    return FilterArray(arr)

def should_have_zero_at_zero(w_zeros, w_poles, zero_at_zero=True):
    """Determine if the filter should have a zero at DC (low-pass) or a pole at DC (high-pass).

    When finite poles/zeros exist, the value is auto-determined by the alternation rule.
    When both w_zeros and w_poles are empty, the user's choice (zero_at_zero) is respected.

    Args:
        w_zeros: Array of zero frequencies
        w_poles: Array of pole frequencies
        zero_at_zero: User preference for pure LP/HP filters (True=low-pass, False=high-pass)

    Returns:
        bool: True for zero at DC (low-pass), False for pole at DC (high-pass)
    """
    if w_zeros.size > 0 and w_poles.size > 0:
        if np.min(w_poles) < np.min(w_zeros): # means that there must be a zero at zero since the lowest pole is lower than the lowest zero
            zero_at_zero = True
        else:
            zero_at_zero = False
    elif w_zeros.size > 0 and w_poles.size == 0: # then there must be a pole at zero        
        zero_at_zero = False
    elif w_poles.size > 0 and w_zeros.size == 0: # then there must be a zero at zero
        zero_at_zero = True
    return zero_at_zero

def frequency_transform(s, w_zeros, w_poles, zero_at_zero):
    """Transform frequency for filter design."""
    n_s = len(s)
    n_zeros = len(w_zeros)
    n_poles = len(w_poles)
    
    N = np.ones(n_s, dtype=complex)
    for j in range(n_zeros):
        N = N * (s**2 + w_zeros[j]**2)
        
    D = np.ones(n_s, dtype=complex)
    for j in range(n_poles):
        D = D * (s**2 + w_poles[j]**2)

    zero_at_zero = should_have_zero_at_zero(w_zeros, w_poles, zero_at_zero)
    if zero_at_zero:
        lambda_val = s * N / D
    else:
        lambda_val = N / (s * D)

    return lambda_val

def filter_transfo_Foster1(g_k, ind_or_cap, zero_at_zero, w_poles, w_zeros, Z0_ohm, fc_filter_GHz, verbose=True):
    """Filter transformation using Foster form 1."""
    wc_filter = 2 * np.pi * fc_filter_GHz * 1E9
    
    n_zeros = len(w_zeros)
    n_poles = len(w_poles)
    
    zero_at_zero = should_have_zero_at_zero(w_zeros, w_poles, zero_at_zero)

    if ind_or_cap == 'ind':
        if zero_at_zero:
            k0 = 0  # Zero at zero frequency means no series cap
        else:
            k0 = np.prod(w_zeros**2) / np.prod(w_poles**2)
        
        if zero_at_zero:
            if n_zeros == n_poles:
                kinf = 1
            else:
                kinf = 0
        else:
            if n_zeros == n_poles:
                kinf = 0
            else:
                kinf = 1
        
        ki = np.zeros(n_poles)
        
        if zero_at_zero:
            for i in range(n_poles):
                w_poles_tmp = np.delete(w_poles, i)
                ki[i] = np.prod((1j * w_poles[i])**2 + w_zeros**2) / (2 * np.prod((1j * w_poles[i])**2 + w_poles_tmp**2))
        else:
            for i in range(n_poles):
                w_poles_tmp = np.delete(w_poles, i)
                ki[i] = np.prod((1j * w_poles[i])**2 + w_zeros**2) / (-2 * w_poles[i]**2 * np.prod((1j * w_poles[i])**2 + w_poles_tmp**2))
        
        LiFoster1_H = np.zeros(n_poles)
        CiFoster1_F = np.zeros(n_poles)
        
        C0Foster1_F = 1 / (k0 * g_k * Z0_ohm * wc_filter)
        LinfFoster1_H = kinf * g_k * Z0_ohm / wc_filter
        
        for j in range(n_poles):
            LiFoster1_H[j] = 2 * ki[j] * g_k * Z0_ohm / (wc_filter * w_poles[j]**2)
            CiFoster1_F[j] = 1 / (2 * ki[j] * g_k * Z0_ohm * wc_filter)
    
    else:  # 'cap'
        if zero_at_zero:
            k0 = np.prod(w_poles**2) / np.prod(w_zeros**2)
        else:
            k0 = 0  # Pole at zero means no shunt inductance
        
        if zero_at_zero:
            if n_zeros == n_poles:
                kinf = 0
            else:
                kinf = 1
        else:
            if n_zeros == n_poles:
                kinf = 1
            else:
                kinf = 0
        
        ki = np.zeros(n_zeros)
        
        if zero_at_zero:
            for i in range(n_zeros):
                w_zeros_tmp = np.delete(w_zeros, i)
                ki[i] = np.prod((1j * w_zeros[i])**2 + w_poles**2) / (-2 * w_zeros[i]**2 * np.prod((1j * w_zeros[i])**2 + w_zeros_tmp**2))
        else:
            for i in range(n_zeros):
                w_zeros_tmp = np.delete(w_zeros, i)
                ki[i] = np.prod((1j * w_zeros[i])**2 + w_poles**2) / (2 * np.prod((1j * w_zeros[i])**2 + w_zeros_tmp**2))
        
        CiFoster1_F = np.zeros(n_zeros)
        LiFoster1_H = np.zeros(n_zeros)
        
        LinfFoster1_H = kinf * Z0_ohm / (g_k * wc_filter)
        C0Foster1_F = g_k / (k0 * Z0_ohm * wc_filter)
        
        for j in range(n_zeros):
            CiFoster1_F[j] = g_k / (Z0_ohm * wc_filter * 2 * ki[j])
            LiFoster1_H[j] = 2 * ki[j] * Z0_ohm / (g_k * wc_filter * w_zeros[j]**2)
    
    if verbose:
        print(f"{ind_or_cap}")
        if np.sum(LinfFoster1_H) != 0 and np.prod(np.array([True if x is not None else False for x in [LinfFoster1_H]])) == 1:
            print(f"LinfFoster1 [nH] =")
            print(LinfFoster1_H * 1E9)
        
        if np.sum(C0Foster1_F) != np.inf and np.prod(np.array([True if x is not None else False for x in [C0Foster1_F]])) == 1:
            print(f"C0Foster1 [pF] =")
            print(C0Foster1_F * 1E12)
        
        if np.sum(LiFoster1_H) != 0 and np.prod(np.array([True if x is not None else False for x in [LiFoster1_H]])) == 1:
            print(f"LiFoster1 [nH] =")
            print(LiFoster1_H * 1E9)
        
        if np.sum(CiFoster1_F) != np.inf and np.prod(np.array([True if x is not None else False for x in [CiFoster1_F]])) == 1:
            print(f"CiFoster1 [pF] =")
            print(CiFoster1_F * 1E12)
    
    return LinfFoster1_H, C0Foster1_F, LiFoster1_H, CiFoster1_F

def filter_transfo_Foster2(g_k, ind_or_cap, zero_at_zero, w_poles, w_zeros, Z0_ohm, fc_filter_GHz,verbose=True):
    """Filter transformation using Foster form 2."""
    wc_filter = 2 * np.pi * fc_filter_GHz * 1E9
    
    n_zeros = len(w_zeros)
    n_poles = len(w_poles)

    zero_at_zero = should_have_zero_at_zero(w_zeros, w_poles, zero_at_zero)
    
    if ind_or_cap == 'ind':
        if zero_at_zero:
            k0 = np.prod(w_poles**2) / np.prod(w_zeros**2)
        else:
            k0 = 0  # Pole at zero means no shunt inductance
        
        if zero_at_zero:
            if n_zeros == n_poles:
                kinf = 0
            else:
                kinf = 1
        else:
            if n_zeros == n_poles:
                kinf = 1
            else:
                kinf = 0
        
        ki = np.zeros(n_zeros)
        
        if zero_at_zero:
            for i in range(n_zeros):
                w_zeros_tmp = np.delete(w_zeros, i)
                ki[i] = np.prod((1j * w_zeros[i])**2 + w_poles**2) / (-2 * w_zeros[i]**2 * np.prod((1j * w_zeros[i])**2 + w_zeros_tmp**2))
        else:
            for i in range(n_zeros):
                w_zeros_tmp = np.delete(w_zeros, i)
                ki[i] = np.prod((1j * w_zeros[i])**2 + w_poles**2) / (2 * np.prod((1j * w_zeros[i])**2 + w_zeros_tmp**2))
        
        LiFoster2_H = np.zeros(n_zeros)
        CiFoster2_F = np.zeros(n_zeros)
        
        L0Foster2_H = g_k * Z0_ohm / (k0 * wc_filter)
        CinfFoster2_F = kinf / (g_k * Z0_ohm * wc_filter)
        
        for j in range(len(ki)):
            LiFoster2_H[j] = g_k * Z0_ohm / wc_filter / (2 * ki[j])
            CiFoster2_F[j] = 2 * ki[j] / (g_k * Z0_ohm * wc_filter * w_zeros[j]**2)
    
    else:  # 'cap'
        if zero_at_zero:
            k0 = 0  # Zero at zero frequency means no series cap
        else:
            k0 = np.prod(w_zeros**2) / np.prod(w_poles**2)
        
        if zero_at_zero:
            if n_zeros == n_poles:
                kinf = 1
            else:
                kinf = 0
        else:
            if n_zeros == n_poles:
                kinf = 0
            else:
                kinf = 1
        
        ki = np.zeros(n_poles)
        
        if zero_at_zero:
            for i in range(n_poles):
                w_poles_tmp = np.delete(w_poles, i)
                ki[i] = np.prod((1j * w_poles[i])**2 + w_zeros**2) / (2 * np.prod((1j * w_poles[i])**2 + w_poles_tmp**2))
        else:
            for i in range(n_poles):
                w_poles_tmp = np.delete(w_poles, i)
                ki[i] = np.prod((1j * w_poles[i])**2 + w_zeros**2) / (-2 * w_poles[i]**2 * np.prod((1j * w_poles[i])**2 + w_poles_tmp**2))
        
        CiFoster2_F = np.zeros(n_poles)
        LiFoster2_H = np.zeros(n_poles)
        
        L0Foster2_H = Z0_ohm / (k0 * g_k * wc_filter)
        CinfFoster2_F = kinf * g_k / (Z0_ohm * wc_filter)
        
        for j in range(n_poles):
            CiFoster2_F[j] = 2 * ki[j] * g_k / (Z0_ohm * wc_filter * w_poles[j]**2)
            LiFoster2_H[j] = Z0_ohm / (2 * ki[j] * g_k * wc_filter)
    
    if verbose:
        print(f"{ind_or_cap}")
        if np.sum(L0Foster2_H) != np.inf and np.prod(np.array([True if x is not None else False for x in [L0Foster2_H]])) == 1:
            print(f"L0Foster2 [nH] =")
            print(L0Foster2_H * 1E9)
        
        if np.sum(CinfFoster2_F) != 0 and np.prod(np.array([True if x is not None else False for x in [CinfFoster2_F]])) == 1:
            print(f"CinfFoster2 [pF] =")
            print(CinfFoster2_F * 1E12)
        
        if np.sum(LiFoster2_H) != 0 and np.prod(np.array([True if x is not None else False for x in [LiFoster2_H]])) == 1:
            print(f"LiFoster2 [nH] =")
            print(LiFoster2_H * 1E9)
        
        if np.sum(CiFoster2_F) != np.inf and np.prod(np.array([True if x is not None else False for x in [CiFoster2_F]])) == 1:
            print(f"CiFoster2 [pF] =")
            print(CiFoster2_F * 1E12)
    
    return L0Foster2_H, CinfFoster2_F, LiFoster2_H, CiFoster2_F


def calculate_filter_components(Foster_form_L,Foster_form_C,g_L,g_C,w_zeros,w_poles,Z0_TWPA_ohm,fc_filter_GHz,zero_at_zero,L0_H,select_one_form,verbose=True):
    
    try:
        ngC = len(g_C)
    except TypeError:
        ngC = 1

    try:
        ngL = len(g_L)
    except:
        ngL = 1

    n_zeros = len(w_zeros)
    n_poles = len(w_poles)

    if Foster_form_L == 1:
        
        if select_one_form != 'C':
            LinfLF1_H, C0LF1_F, LiLF1_H, CiLF1_F = \
            filter_design_L(g_L, w_poles, w_zeros, Z0_TWPA_ohm, fc_filter_GHz, zero_at_zero,  Foster_form_L)   
        else:
            LinfLF1_H, C0LF1_F, LiLF1_H, CiLF1_F = \
            filter_design_L(g_L, np.array([]), np.array([]), Z0_TWPA_ohm, fc_filter_GHz, zero_at_zero,  Foster_form_L)                
            
            LiLF1_H = np.zeros((ngL, n_poles))
            CiLF1_F = np.ones((ngL, n_poles))*np.inf                

        maxL_ind_H = [np.max(LinfLF1_H) if LinfLF1_H.size > 0 else 0, np.max(LiLF1_H) if LiLF1_H.size > 0 else 0]
        maxL_cap_F = [np.max(C0LF1_F) if C0LF1_F.size > 0 else 0, np.max(CiLF1_F) if CiLF1_F.size > 0 else 0]

        LinfLF1_rem_H = LinfLF1_H - L0_H
        if LinfLF1_rem_H < 0 or LinfLF1_rem_H < 1e-20: # numerical precision here
                LinfLF1_rem_H = 0

        L0LF2_H, CinfLF2_F, LiLF2_H, CiLF2_F, L0LF2_rem_H = LinfLF1_H*np.nan, C0LF1_F*np.nan, LiLF1_H*np.nan, CiLF1_F*np.nan, LinfLF1_rem_H*np.nan

    else:

        if select_one_form != 'C':
            L0LF2_H, CinfLF2_F, LiLF2_H, CiLF2_F = \
            filter_design_L(g_L, w_poles, w_zeros, Z0_TWPA_ohm, fc_filter_GHz, zero_at_zero, Foster_form_L)
        else:
            L0LF2_H, CinfLF2_F, LiLF2_H, CiLF2_F = \
            filter_design_L(g_L, np.array([]), np.array([]), Z0_TWPA_ohm, fc_filter_GHz, zero_at_zero, Foster_form_L)
        
            LiLF2_H = np.ones((ngL, n_zeros))*np.inf
            CiLF2_F = np.zeros((ngL, n_zeros))

        maxL_ind_H = [np.max(L0LF2_H) if L0LF2_H.size > 0 else 0, np.max(LiLF2_H) if LiLF2_H.size > 0 else 0]
        maxL_cap_F = [np.max(CinfLF2_F) if CinfLF2_F.size > 0 else 0, np.max(CiLF2_F) if CiLF2_F.size > 0 else 0]           

        L0LF2_rem_H = L0LF2_H - L0_H
        if L0LF2_rem_H < 0 or L0LF2_rem_H < 1e-20: # numerical precision here
            L0LF2_rem_H = 0

        LinfLF1_H, C0LF1_F, LiLF1_H, CiLF1_F, LinfLF1_rem_H = L0LF2_H*np.nan, CinfLF2_F*np.nan, LiLF2_H*np.nan, CiLF2_F*np.nan, L0LF2_rem_H*np.nan

    maxL_ind_H = max([x for x in maxL_ind_H if not np.isinf(x)]) if any(not np.isinf(x) for x in maxL_ind_H) else 0
    maxL_cap_F = max([x for x in maxL_cap_F if not np.isinf(x)]) if any(not np.isinf(x) for x in maxL_cap_F) else 0

    if Foster_form_C == 1:

        if select_one_form != 'L':
            LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F = \
            filter_design_C(g_C, w_poles, w_zeros, Z0_TWPA_ohm, fc_filter_GHz, zero_at_zero, Foster_form_C)    
        else:
            LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F = \
            filter_design_C(g_C, np.array([]), np.array([]), Z0_TWPA_ohm, fc_filter_GHz, zero_at_zero, Foster_form_C)    
                                 
            LiCF1_H = np.zeros((ngC, n_zeros))
            CiCF1_F = np.ones((ngC, n_zeros))*np.inf

        maxC_ind_H = [np.max(LinfCF1_H) if LinfCF1_H.size > 0 else 0, np.max(LiCF1_H) if LiCF1_H.size > 0 else 0]
        maxC_cap_F = [np.max(C0CF1_F) if C0CF1_F.size > 0 else 0, np.max(CiCF1_F) if CiCF1_F.size > 0 else 0]

        L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F = LinfCF1_H*np.nan, C0CF1_F*np.nan, LiCF1_H*np.nan, CiCF1_F*np.nan
    else:

        if select_one_form != 'L':
            L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F = \
            filter_design_C(g_C, w_poles, w_zeros, Z0_TWPA_ohm, fc_filter_GHz, zero_at_zero, Foster_form_C)
        else:
            L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F = \
            filter_design_C(g_C, np.array([]), np.array([]), Z0_TWPA_ohm, fc_filter_GHz, zero_at_zero, Foster_form_C)
                    
            LiCF2_H = np.ones((ngC, n_poles))*np.inf
            CiCF2_F = np.zeros((ngC, n_poles))

        maxC_ind_H = [np.max(L0CF2_H) if L0CF2_H.size > 0 else 0, np.max(LiCF2_H) if LiCF2_H.size > 0 else 0]
        maxC_cap_F = [np.max(CinfCF2_F) if CinfCF2_F.size > 0 else 0, np.max(CiCF2_F) if CiCF2_F.size > 0 else 0]

        LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F = L0CF2_H*np.nan, CinfCF2_F*np.nan, LiCF2_H*np.nan, CiCF2_F*np.nan

    maxC_ind_H = max([x for x in maxC_ind_H if not np.isinf(x)]) if any(not np.isinf(x) for x in maxC_ind_H) else 0
    maxC_cap_F = max([x for x in maxC_cap_F if not np.isinf(x)]) if any(not np.isinf(x) for x in maxC_cap_F) else 0

    if verbose:
        print(f"Max inductance in the series branch (pH): {maxL_ind_H*1E12:.4f}\n"
              f"Max capacitance in the series branch (fF): {maxL_cap_F*1E15:.4f}\n"
              f"Max inductance in the shunt branch (pH): {maxC_ind_H*1E12:.4f}\n"
              f"Max capacitance in the shunt branch (fF): {maxC_cap_F*1E15:.4f}")

    return LinfLF1_H, C0LF1_F, LiLF1_H, CiLF1_F, LinfLF1_rem_H,\
            L0LF2_H, CinfLF2_F, LiLF2_H, CiLF2_F, L0LF2_rem_H,\
            LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,\
            L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,\
            maxL_ind_H, maxL_cap_F, maxC_ind_H, maxC_cap_F



################################################################################################
################################################################################################
# adding apodization to the TL synthesis


def create_windowed_transmission_line(g_C_supercell, Nsc_cell, window_type='tukey', alpha=0.5):
    """
    Create a windowed transmission line by repeating the supercell pattern and
    applying a window function to the entire line.
    
    Parameters:
    -----------
    g_C_supercell : numpy.ndarray
        The normalized capacitance profile for a single supercell
    Nsc_cell : int
        Number of supercells to cascade
    window_type : str
        The type of window to apply to the entire transmission line
        Options: 'boxcar', 'hann', 'tukey', 'cosine', 'lanczos'
    alpha : float
        Parameter for Tukey window controlling the width of the cosine-tapered region
        
    Returns:
    --------
    numpy.ndarray
        The windowed capacitance profile for the entire transmission line
    """
    import numpy as np
    
    # Number of cells in a supercell
    Ncpersc_cell = len(g_C_supercell)
    
    # Total number of cells in the transmission line
    Ntot_cell = Ncpersc_cell * Nsc_cell
    
    # Create the full capacitance profile by repeating the supercell
    g_C_full = np.tile(g_C_supercell, Nsc_cell)
    
    # Extract the baseline and modulation depth
    baseline = np.mean(g_C_supercell)
    
    # Create the window function for the entire transmission line
    if window_type.lower() == 'boxcar':
        # No windowing (rectangular window)
        window = np.ones(Ntot_cell)
    
    elif window_type.lower() == 'hann':
        # Hann window (raised cosine)
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(Ntot_cell) / (Ntot_cell - 1)))
    
    elif window_type.lower() == 'tukey':
        # Tukey window (tapered cosine)
        window = np.ones(Ntot_cell)
        # Width of the cosine-tapered region on each side
        width = int(alpha * (Ntot_cell - 1) / 2)

        # Adjust width to be a multiple of Ncpersc_cell
        width = round(width / Ncpersc_cell) * Ncpersc_cell
        
        if width > 0:
            # Apply cosine taper at the beginning
            window[:width] = 0.5 * (1 + np.cos(np.pi * (np.arange(width) / width - 1)))
            # Apply cosine taper at the end
            window[-width:] = 0.5 * (1 + np.cos(np.pi * (np.arange(width) / width)))
    
    elif window_type.lower() == 'cosine':
        # Simple cosine window
        window = np.sin(np.pi * np.arange(Ntot_cell) / (Ntot_cell - 1))
    
    elif window_type.lower() == 'lanczos':
        # Lanczos window
        x = 2 * np.arange(Ntot_cell) / (Ntot_cell - 1) - 1  # Scale to [-1, 1]
        window = np.sinc(x)
    
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    # Apply the window to each supercell's modulation
    g_C_windowed_full = np.ones(Ntot_cell)
    
    for i in range(Nsc_cell):
        start_idx = i * Ncpersc_cell
        end_idx = (i + 1) * Ncpersc_cell
        
        # Get the window coefficients for this section
        section_window = window[start_idx:end_idx]
        
        # Apply the window amplitude to the modulation of this supercell
        section_modulation = g_C_full[start_idx:end_idx] - baseline
        g_C_windowed_full[start_idx:end_idx] = baseline + section_modulation * section_window
    
    return g_C_windowed_full

################################################################################################
################################################################################################

def filter_design_L(g_L, w_poles, w_zeros, Z0_TWPA_ohm, fc_filter_GHz, zero_at_zero, Foster_form_L):
    
    g_L_array = np.atleast_1d(g_L)
    ngL = len(g_L_array)

    n_zeros = len(w_zeros)
    n_poles = len(w_poles)
    
    # Filter design
    if Foster_form_L == 1:

        LinfLF1_H = np.zeros(ngL)
        C0LF1_F = np.zeros(ngL)
        LiLF1_H = np.zeros((ngL, n_poles))
        CiLF1_F = np.zeros((ngL, n_poles))

        for i in range(ngL):
            LinfLF1_H[i], C0LF1_F[i], LiLF1_H[i,:], CiLF1_F[i,:] = filter_transfo_Foster1(
                g_L_array[i], 'ind', zero_at_zero, w_poles, w_zeros, Z0_TWPA_ohm, fc_filter_GHz, verbose=False
            )
                        
    else:
        L0LF2_H = np.zeros(ngL)
        CinfLF2_F = np.zeros(ngL)
        LiLF2_H = np.zeros((ngL, n_zeros))
        CiLF2_F = np.zeros((ngL, n_zeros))

        for i in range(ngL):
            L0LF2_H[i], CinfLF2_F[i], LiLF2_H[i,:], CiLF2_F[i,:] = filter_transfo_Foster2(
                g_L_array[i], 'ind', zero_at_zero, w_poles, w_zeros, Z0_TWPA_ohm, fc_filter_GHz, verbose=False
            )
            

    if Foster_form_L == 1:
        return LinfLF1_H, C0LF1_F, LiLF1_H, CiLF1_F
    else:
        return L0LF2_H, CinfLF2_F, LiLF2_H, CiLF2_F


def filter_design_C(g_C, w_poles, w_zeros, Z0_TWPA_ohm, fc_filter_GHz, zero_at_zero, Foster_form_C):
    
    g_C_array = np.atleast_1d(g_C)
    ngC = len(g_C_array)

    n_zeros = len(w_zeros)
    n_poles = len(w_poles)
   
    if Foster_form_C == 1:

        LinfCF1_H = np.zeros(ngC)
        C0CF1_F = np.zeros(ngC)
        LiCF1_H = np.zeros((ngC, n_zeros))
        CiCF1_F = np.zeros((ngC, n_zeros))
        
        for i in range(ngC):
            LinfCF1_H[i], C0CF1_F[i], LiCF1_H[i,:], CiCF1_F[i,:] = filter_transfo_Foster1(
                g_C_array[i], 'cap', zero_at_zero, w_poles, w_zeros, Z0_TWPA_ohm, fc_filter_GHz, verbose=False
            )            
                
    else:
        L0CF2_H = np.zeros(ngC)
        CinfCF2_F = np.zeros(ngC)
        LiCF2_H = np.zeros((ngC, n_poles))
        CiCF2_F = np.zeros((ngC, n_poles))

        for i in range(ngC):
            L0CF2_H[i], CinfCF2_F[i], LiCF2_H[i,:], CiCF2_F[i,:] = filter_transfo_Foster2(
                g_C_array[i], 'cap', zero_at_zero, w_poles, w_zeros, Z0_TWPA_ohm, fc_filter_GHz, verbose=False
            )


    if Foster_form_C == 1:
        return LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F
    else:
        return L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F
    

def Z_Foster_form_L(L0_H,n_jj_struct,CJ_F, C0LF1_F, LiLF1_H, CiLF1_F, LinfLF1_rem_H,w,n_poles,LinfLF1_H,nonlinearity='JJ'):  

    if C0LF1_F != np.inf: # python does not handle 1/(a+inf) well
        if nonlinearity == 'JJ':
            if 1-L0_H*CJ_F*w**2 != 0:
                Z_filter_L = 1/(1j*w*C0LF1_F) + 1j*w*LinfLF1_rem_H + n_jj_struct*1j*w*L0_H/(1-L0_H*CJ_F*w**2)
            else:
                Z_filter_L = 1j*np.inf     
        elif nonlinearity == 'KI':
            Z_filter_L = 1/(1j*w*C0LF1_F) + 1j*w*LinfLF1_rem_H + 1j*w*LinfLF1_H                    
    else:
        if nonlinearity == 'JJ':
            if 1-L0_H*CJ_F*w**2 != 0:                
                Z_filter_L = 1j*w*LinfLF1_rem_H + n_jj_struct*1j*w*L0_H/(1-L0_H*CJ_F*w**2)
            else:
                Z_filter_L = 1j*np.inf
        elif nonlinearity == 'KI':
            Z_filter_L = 1j*w*LinfLF1_rem_H + 1j*w*LinfLF1_H

    for j in range(n_poles):
        if LiLF1_H[j] != 0 and CiLF1_F[j] != np.inf:
            if 1/(1j*w*LiLF1_H[j]) != -1j*w*CiLF1_F[j]: # avoid division by zero
                Z_filter_L += 1/(1/(1j*w*LiLF1_H[j]) + 1j*w*CiLF1_F[j])
            else:
                Z_filter_L += 1j*np.inf 

    Z_filter_L = np.asarray(Z_filter_L).item()

    return  Z_filter_L

def Y_Foster_form_L(L0LF2_H, CinfLF2_F, LiLF2_H, CiLF2_F,w,n_zeros):
   
    if L0LF2_H != np.inf:
        Y_filter_L = 1j*w*CinfLF2_F + 1/(1j*w*L0LF2_H)                
    else:
        Y_filter_L = 1j*w*CinfLF2_F                
    Y_trunkfilter_L = 1j*w*CinfLF2_F

    for j in range(n_zeros):
        if LiLF2_H[j] != np.inf and CiLF2_F[j] != 0:
            Y_filter_L += 1/(1j*w*LiLF2_H[j] + 1/(1j*w*CiLF2_F[j]))
            Y_trunkfilter_L += 1/(1j*w*LiLF2_H[j] + 1/(1j*w*CiLF2_F[j]))

    Y_filter_L = np.asarray(Y_filter_L).item()
    Y_trunkfilter_L = np.asarray(Y_trunkfilter_L).item()

    return Y_filter_L, Y_trunkfilter_L

def Z_Foster_form_C(LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,w,n_zeros):    

    if C0CF1_F != np.inf:
        Z_filter_C = 1j*w*LinfCF1_H + 1/(1j*w*C0CF1_F)
    else:
        Z_filter_C = 1j*w*LinfCF1_H

    for j in range(n_zeros):
        if LiCF1_H[j] != 0 and CiCF1_F[j] != np.inf:
            if 1/(1j*w*LiCF1_H[j]) != -1j*w*CiCF1_F[j]: # avoid division by zero
                Z_filter_C += 1/(1/(1j*w*LiCF1_H[j]) + 1j*w*CiCF1_F[j])
            else:
                Z_filter_C += 1j*np.inf

    Z_filter_C = np.asarray(Z_filter_C).item()
    
    return Z_filter_C

def Y_Foster_form_C(L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,w,n_poles):

    if L0CF2_H != np.inf:
        Y_filter_C = 1/(1j*w*L0CF2_H) + 1j*w*CinfCF2_F
    else:
        Y_filter_C = 1j*w*CinfCF2_F

    for j in range(n_poles):
        if LiCF2_H[j] != np.inf and CiCF2_F[j] != 0:
            Y_filter_C += 1/(1j*w*LiCF2_H[j] + 1/(1j*w*CiCF2_F[j]))

    Y_filter_C = np.asarray(Y_filter_C).item()

    return Y_filter_C

def get_ABCD_filter(Foster_form_L,Foster_form_C,L0_H,n_jj_struct,CJ_F,C0LF1_F,LiLF1_H,CiLF1_F,LinfLF1_rem_H,
                              L0LF2_H,CinfLF2_F,LiLF2_H,CiLF2_F,
                              LinfCF1_H,C0CF1_F,LiCF1_H,CiCF1_F,
                              L0CF2_H,CinfCF2_F,LiCF2_H,CiCF2_F,
                              w,n_poles,n_zeros,LinfLF1_H,nonlinearity):

    if Foster_form_L == 1:            
        Z_filter_L = Z_Foster_form_L(L0_H,n_jj_struct,CJ_F, C0LF1_F, LiLF1_H, CiLF1_F, LinfLF1_rem_H,w,n_poles,LinfLF1_H,nonlinearity)
        Y_filter_L = np.eye(2)
    else:
        Z_filter_L = np.eye(2)
        Y_filter_L, _ = Y_Foster_form_L(L0LF2_H, CinfLF2_F, LiLF2_H, CiLF2_F,w,n_zeros)                
    if Foster_form_C == 1:
        Z_filter_C = Z_Foster_form_C(LinfCF1_H, C0CF1_F, LiCF1_H, CiCF1_F,w,n_zeros)
        Y_filter_C = np.eye(2)
    else:
        Z_filter_C = np.eye(2)
        Y_filter_C = Y_Foster_form_C(L0CF2_H, CinfCF2_F, LiCF2_H, CiCF2_F,w,n_poles)

    ABCD_filter = calculate_ABCD_filter(Foster_form_L,Foster_form_C,Z_filter_L,Z_filter_C,Y_filter_L,Y_filter_C)   

    return ABCD_filter 


def calculate_ABCD_filter(Foster_form_L,Foster_form_C,Z_filter_L,Z_filter_C,Y_filter_L,Y_filter_C):
    
    if Foster_form_L == 1 and Foster_form_C == 1:
        ABCD_filter = np.array([
            [1, Z_filter_L],
            [0, 1]
        ]) @ np.array([
            [1, 0],
            [1/Z_filter_C if Z_filter_C != 0 else 1j*np.inf, 1]
        ])
                
    elif Foster_form_L == 1 and Foster_form_C == 2:
        ABCD_filter = np.array([
            [1, Z_filter_L],
            [0, 1]
        ]) @ np.array([
            [1, 0],
            [Y_filter_C, 1]
        ])
                
    elif Foster_form_L == 2 and Foster_form_C == 1:
        ABCD_filter = np.array([
            [1, 1/Y_filter_L if Y_filter_L != 0 else 1j*np.inf],
            [0, 1]
        ]) @ np.array([
            [1, 0],
            [1/Z_filter_C if Z_filter_C != 0 else 1j*np.inf, 1]
        ])
                
    elif Foster_form_L == 2 and Foster_form_C == 2:
        ABCD_filter = np.array([
            [1, 1/Y_filter_L if Y_filter_L != 0 else 1j*np.inf],
            [0, 1]
        ]) @ np.array([
            [1, 0],
            [Y_filter_C, 1]
        ])
                
    return ABCD_filter


def calculate_ABCD_TLsec(L0_H,n_jj_struct,CJ_F,LinfTLsec_rem_H,CinfTLsec_F,w,LinfTLsec_H,nonlinearity='JJ'):

    if nonlinearity == 'JJ':
        if 1-L0_H*CJ_F*w**2 != 0:
            ABCD_TLsec = np.array([
                [1, n_jj_struct*1j*w*L0_H/(1-L0_H*CJ_F*w**2)],
                [0, 1]
            ])
        else:
            ABCD_TLsec = np.array([
                [1, 1j*np.inf],
                [0, 1]
            ])
    elif nonlinearity == 'KI':
        ABCD_TLsec = np.array([
            [1, 1j*w*LinfTLsec_H],
            [0, 1]
        ])

    ABCD_remTLsec =  np.array([
        [1, 1j*w*LinfTLsec_rem_H],
        [0, 1]
        ]) @ np.array([
        [1, 0],
        [1j*w*CinfTLsec_F, 1]
        ])


    ABCD_TLsec =  ABCD_TLsec @ ABCD_remTLsec  

    return ABCD_TLsec


################################# pl helper #################################

def pl_derived_quantities(f_stopbands_GHz, deltaf_min_GHz, deltaf_max_GHz, f0_GHz):
    n_stopbands = len(f_stopbands_GHz) # number of stopbands
    print(f"Number of designed stopbands: {n_stopbands}")

    f1_GHz = math.gcd(*f_stopbands_GHz)
    print(f"f1_GHz: {f1_GHz}")

    ind_stopband = [int(np.round(f/f1_GHz)) for f in f_stopbands_GHz] # stopband indices
    print(f"stopband indices: {ind_stopband}")

    # Find the maximum stopband index
    max_ind_stopband = max(ind_stopband)

    # Find all "skipped" indices (indices that are missing)
    all_indices = set(range(1, max_ind_stopband + 1))
    existing_indices = set(ind_stopband)
    skipped_indices = list(all_indices - existing_indices)
    skipped_indices.sort()  # Sort them for consistent ordering
    print(f"Skipped indices: {skipped_indices}")

    fc_GHz = f0_GHz * 2 # cutoff frequency of the TL = 2/(2*pi*sqrt(LC))
    v_cellpernsec = f0_GHz * 2*np.pi
    Ncpersc_cell = int(np.round(np.pi/2 * fc_GHz/f1_GHz))  # Length of one supercell, in cells units
    # there will always be a rounding error associated with this calculation.

    print(f"phase velocity: {v_cellpernsec} cells/ns")
    print(f"Length of one supercell: {Ncpersc_cell} cells")

    w_tilde_stopband_edges_list = []
    ind_stopband_list = []

    # Keep track of non-default parameters
    n_param = 0
    is_default_value = []  # Will track whether each edge is a default value or not

    # Process each stopband
    for i in range(n_stopbands):
        # Process min delta if it exists and is not None
        if i < len(deltaf_min_GHz) and deltaf_min_GHz[i] is not None:
            f_stopband_min_GHz = f_stopbands_GHz[i] - deltaf_min_GHz[i]
            w_tilde_stopband_min = f_stopband_min_GHz/f0_GHz * Ncpersc_cell / (2*np.pi)
            w_tilde_stopband_edges_list.append(w_tilde_stopband_min)
            ind_stopband_list.append(ind_stopband[i])
            is_default_value.append(False)  # This is not a default value
            n_param += 1  # Increment parameter count
            print(f"Stopband {ind_stopband[i]} - Normalized stopband min: {w_tilde_stopband_min}")

        # Process max delta if it exists and is not None
        if i < len(deltaf_max_GHz) and deltaf_max_GHz[i] is not None:
            f_stopband_max_GHz = f_stopbands_GHz[i] + deltaf_max_GHz[i]
            w_tilde_stopband_max = f_stopband_max_GHz/f0_GHz * Ncpersc_cell / (2*np.pi)
            w_tilde_stopband_edges_list.append(w_tilde_stopband_max)
            ind_stopband_list.append(ind_stopband[i])
            is_default_value.append(False)  # This is not a default value
            n_param += 1  # Increment parameter count
            print(f"Stopband {ind_stopband[i]} - Normalized stopband max: {w_tilde_stopband_max}")


    # Handle the skipped indices - add them to the list with a normalized value of 0.5 * index
    for idx in skipped_indices:
        normalized_value = 0.5 * idx  # As per your requirement (normalized to 1 in your unit system)
        w_tilde_stopband_edges_list.append(normalized_value)
        ind_stopband_list.append(idx)
        is_default_value.append(True)  # This is a default value - not a parameter
        print(f"Skipped Stopband {idx} - Added with normalized value: {normalized_value}")

    # Combine the lists for sorting (including the is_default_value information)
    combined = list(zip(ind_stopband_list, w_tilde_stopband_edges_list, is_default_value))
    # Sort by the stopband indices
    combined.sort(key=lambda x: x[0])
    # Unzip the sorted lists
    ind_stopband_list, w_tilde_stopband_edges_list, is_default_value = zip(*combined) if combined else ([], [], [])

    # Convert lists to numpy arrays
    w_tilde_stopband_edges = np.array(w_tilde_stopband_edges_list)
    ind_stopband = np.array(ind_stopband_list)
    is_default_value = np.array(is_default_value)

    print(f"w_tilde_stopband_edges: {w_tilde_stopband_edges}")
    print(f"ind_stopband: {ind_stopband}")
    print(f"Number of parameters (non-default values): {n_param}")
    print(f"Total number of stopband edges: {len(w_tilde_stopband_edges)}")
    print(f"Default values mask: {is_default_value}")

    # Create arrays for just the parameters (non-default values)
    non_default_indices = np.where(~is_default_value)[0]
    w_tilde_param = w_tilde_stopband_edges[non_default_indices]
    ind_param = ind_stopband[non_default_indices]

    print(f"Parameters only (w_tilde_param): {w_tilde_param}")
    print(f"Parameter indices (ind_param): {ind_param}")

    return n_stopbands, v_cellpernsec, Ncpersc_cell, w_tilde_stopband_edges, ind_stopband, is_default_value, w_tilde_param, ind_param, max_ind_stopband, n_param, skipped_indices


def calculate_delta_values(ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param, skipped_indices,
                           solution_tolerance=1e-8, always_return_best=True, n_attempts=10, verbose=True):

    ################################################################################################
    # Calculate the delta values for the actual parameters (not the skipped ones)

    # Define the precision tolerance - can be adjusted for more/less precise solutions
    # Lower values (e.g., 1e-10) are more precise but harder to find
    # Higher values (e.g., 1e-6) are less precise but easier to find
    solution_tolerance = 1e-8  # Default value - adjust as needed

    # Set to True to always return the best solution found, even if it doesn't meet the tolerance
    always_return_best = True

    # Number of random starting points to try
    n_attempts = 10  # Default value: 10 - adjust as needed


    # Run the multi-stage solution finding approach
    delta_solutions = multi_stage_solution_finding(
        ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param,
        bounds=(-0.5, 0.5), n_attempts=n_attempts, tolerance=solution_tolerance, 
        always_return_best=always_return_best, verbose=True)

    # Print all solutions
    print("\nAll solutions found:")
    if len(delta_solutions) > 0:
        for i, sol in enumerate(delta_solutions):
            solution_str = f"Solution {i+1}: "
            for j, val in enumerate(sol):
                # Map parameter index to the actual stopband index
                sb_idx = ind_param[j]
                solution_str += f"delta_{sb_idx} = {val:.6f}  "
            print(solution_str)

        print("Selecting solution with the lowest magnitude deltas...")
        
        # Calculate the total magnitude (sum of absolute values) for each solution
        magnitudes = [(sol, np.sum(np.abs(sol))) for sol in delta_solutions]
        
        # Sort by the total magnitude (ascending)
        sorted_solutions = sorted(magnitudes, key=lambda x: x[1])
        
        # Select the solution with the lowest total magnitude
        selected_delta = sorted_solutions[0][0]
        
        # Print information about the selected solution
        total_magnitude = np.sum(np.abs(selected_delta))
        print(f"Selected solution has total magnitude: {total_magnitude:.6f}")
        solution_str = ""
        for j, val in enumerate(selected_delta):
            # Map parameter index to the actual stopband index
            sb_idx = ind_param[j]
            solution_str += f"delta_{sb_idx} = {val:.6f}  "
        print(solution_str)
    else:
        print("No solutions found, even best-effort solutions. Using default small magnitude values.")
        print(f"Try adjusting the solution_tolerance (current: {solution_tolerance}) to a higher value.")
        # Create a default solution with small magnitude values
        selected_delta = np.ones(n_param) * 0.01  # Small positive values

    print(f"\nFinal selected solution: {selected_delta}")

    # Create a mapping from stopband indices to delta values
    delta_map = {}
    for j, val in enumerate(selected_delta):
        sb_idx = ind_param[j]
        delta_map[sb_idx] = val

    print("\nDelta values mapped to stopband indices:")
    for idx in sorted(delta_map.keys()):
        print(f"delta_{idx} = {delta_map[idx]:.6f}")

    print(f"\nNote: Stopbands with indices {', '.join(str(idx) for idx in skipped_indices)} were skipped (no free parameters).")
    print(f"Solution parameters:")
    print(f"  Tolerance: {solution_tolerance}")
    print(f"  Always return best: {always_return_best}")

    return delta_map, selected_delta
    

################################# misc helper #################################


def filecounter(pathname):
    """Find and increment file counter for unique file naming"""
    import glob
    import re
    import os
    
    # Find existing files matching pattern
    files = glob.glob(pathname)
    if not files:
        return pathname.replace('*', '01'), 1
    
    # Extract numbers from file names before the extension
    numbers = []
    for file in files:
        # Get the filename without extension
        base_name = os.path.splitext(file)[0]
        # Look for number at the end of the base name
        match = re.search(r'(\d+)$', base_name)
        if match:
            numbers.append(int(match.group(1)))
    
    if numbers:
        n_files = max(numbers) + 1
    else:
        n_files = 1
    
    new_pathname = pathname.replace('*', f'{n_files:02d}')
    return new_pathname, n_files


def check_flat(element,idx):

    try: 
        element_flat = element.flat[idx]
    except:
        element_flat = element 

    return element_flat  