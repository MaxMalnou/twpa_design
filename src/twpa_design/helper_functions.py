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
from scipy.stats import qmc  # For Latin Hypercube Sampling in Hermitian solver
from scipy.special import iv as _bessel_iv  # Modified Bessel I_n, for Klopfenstein φ
from scipy.integrate import quad as _quad_integrate  # for Klopfenstein φ integral


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


def det_D_hermitian(delta_cs, ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband):
    """
    Determinant function with Hermitian coupling matrix (allows phase in modulation).

    The modulation is: g_C(x) = 1 + 2*sum_n [delta_c[n]*cos(n*x) + delta_s[n]*sin(n*x)]

    This corresponds to complex Fourier coefficients:
        epsilon_n = delta_c[n] - 1j*delta_s[n]
        epsilon_{-n} = conj(epsilon_n) = delta_c[n] + 1j*delta_s[n]

    The coupling matrix D is Hermitian: D[k, k+n] = conj(D[k+n, k])

    Parameters:
    delta_cs (array): Array of length 2*n_param, structured as [delta_c_1, delta_c_2, ..., delta_s_1, delta_s_2, ...]
                      First half are cosine coefficients, second half are sine coefficients.
    ind_stopband (array): Array of stopband indices
    ind_param (array): Array of stopband indices for parameters (non-default values)
    w_tilde_stopband_edges (array): Array of normalized frequencies at stopband edges
    is_default_value (array): Boolean array indicating which edges are default values
    max_ind_stopband (int): Maximum stopband index

    Returns:
    array: Determinant values at each stopband edge (real, since D is Hermitian)
    """
    n_param = len(ind_param)
    delta_c = delta_cs[:n_param]  # cosine components
    delta_s = delta_cs[n_param:]  # sine components

    n_tot_stopband_edges = len(w_tilde_stopband_edges)
    n_modes = 2 * max_ind_stopband + 1

    # Define k at the band edge
    k_norm = 0.5 * ind_stopband

    # Create the mode indices
    v_modes = np.arange(-np.floor(n_modes/2), np.floor(n_modes/2) + 1, dtype=int)

    detD_delta = []

    # loop over all stopband edges
    for j in range(n_tot_stopband_edges):
        # Create a complex matrix for Hermitian structure
        D = np.zeros((n_modes, n_modes), dtype=complex)

        # Fill diagonal terms (real)
        for i in range(n_modes):
            D[i, i] = (k_norm[j] + v_modes[i])**2 - w_tilde_stopband_edges[j]**2

        # Skip default values
        if not is_default_value[j]:
            # Fill off-diagonal terms with Hermitian structure
            for i in range(n_param):
                sb_idx = ind_param[i]

                # Complex coupling coefficient: epsilon_n = delta_c - 1j*delta_s
                epsilon_n = (delta_c[i] - 1j*delta_s[i]) * w_tilde_stopband_edges[j]**2
                epsilon_n_conj = (delta_c[i] + 1j*delta_s[i]) * w_tilde_stopband_edges[j]**2

                for k in range(n_modes - sb_idx):
                    D[k, k+sb_idx] -= epsilon_n        # upper off-diagonal
                    D[k+sb_idx, k] -= epsilon_n_conj  # lower off-diagonal (conjugate)

        # Calculate determinant - should be real for Hermitian matrix
        det_val = np.linalg.det(D)
        detD_delta.append(np.real(det_val))

    return np.array(detD_delta)


def evaluate_modulation(delta_c, delta_s, ind_param, n_points=1000):
    """
    Evaluate the modulation profile and compute peak deviation.

    The modulation is: g_C(x) = 1 + 2*sum_n [delta_c[n]*cos(n*x) + delta_s[n]*sin(n*x)]

    Parameters:
    delta_c (array): Cosine coefficients for each harmonic
    delta_s (array): Sine coefficients for each harmonic
    ind_param (array): Harmonic indices corresponding to delta_c and delta_s
    n_points (int): Number of points to evaluate

    Returns:
    tuple: (peak_deviation, g_min, g_max)
           peak_deviation is max(g_C) - 1, or np.inf if g_C goes negative
    """
    x = np.linspace(0, 2*np.pi, n_points)
    g_C = np.ones(n_points)

    for i, idx in enumerate(ind_param):
        g_C += 2 * (delta_c[i] * np.cos(idx * x) + delta_s[i] * np.sin(idx * x))

    g_min = np.min(g_C)
    g_max = np.max(g_C)

    if g_min <= 0:
        return np.inf, g_min, g_max  # infeasible - negative capacitance
    else:
        return g_max - 1, g_min, g_max  # peak positive deviation


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


def find_all_solutions_hermitian(ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param,
                                  bounds=(-0.5, 0.5), n_attempts=30, tolerance=1e-8, always_return_best=True, verbose=True):
    """
    Find multiple solutions to the determinant system using Hermitian coupling matrix.

    This version optimizes 2*n_param parameters (delta_c and delta_s for each harmonic),
    allowing phase freedom in the modulation.

    Parameters:
    ind_stopband (array): Array of stopband indices
    ind_param (array): Array of stopband indices for parameters (non-default values)
    w_tilde_stopband_edges (array): Array of normalized frequencies at stopband edges
    is_default_value (array): Boolean array indicating which edges are default values
    max_ind_stopband (int): Maximum stopband index
    n_param (int): Number of harmonics (actual optimization has 2*n_param parameters)
    bounds (tuple): Bounds for the solutions
    n_attempts (int): Number of random starting points to try
    tolerance (float): Tolerance for accepting solutions (lower = more precise)
    always_return_best (bool): If True, always return the best solution found
    verbose (bool): Whether to print progress messages

    Returns:
    list: List of all unique solutions found, each of shape (2*n_param,) as [delta_c..., delta_s...]
    """
    n_opt_param = 2 * n_param  # Optimize both cosine and sine components

    # Define the objective function to minimize (sum of squared determinants)
    def objective(delta_cs):
        det_values = det_D_hermitian(delta_cs, ind_stopband, ind_param, w_tilde_stopband_edges,
                                     is_default_value, max_ind_stopband)
        return sum(d*d for d in det_values)

    # List to store unique solutions
    unique_solutions = []
    solution_tolerance = tolerance
    bounds_list = [(bounds[0], bounds[1]) for _ in range(n_opt_param)]

    # Track the best solution found
    best_solution = None
    best_objective_value = float('inf')

    if verbose:
        print(f"Running {n_attempts} basin-hopping searches for {n_opt_param} parameters (Hermitian)...")
        print(f"  ({n_param} cosine + {n_param} sine components)")
        print(f"Using solution tolerance: {solution_tolerance}")

    # Generate starting points using Latin Hypercube Sampling for better coverage
    # LHS ensures each "slice" of each dimension is sampled exactly once
    sampler = qmc.LatinHypercube(d=n_opt_param)
    lhs_samples = sampler.random(n=n_attempts)
    # Scale from [0,1] to [bounds[0], bounds[1]]
    starting_points = qmc.scale(lhs_samples, bounds[0], bounds[1])

    # Custom step-taking function
    def take_bounded_step(x):
        s = np.random.uniform(-0.1, 0.1, len(x))
        new_x = x + s
        new_x = np.clip(new_x, bounds[0], bounds[1])
        return new_x

    # Try each starting point
    for i, x0 in enumerate(starting_points):
        if verbose:
            print(f"Starting point {i+1}/{len(starting_points)}")

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds_list,
            "options": {"ftol": tolerance/10, "gtol": tolerance/10, "maxiter": 1000}
        }

        result = basinhopping(
            objective,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=100,
            T=0.5,
            stepsize=0.05,
            take_step=take_bounded_step,
            interval=10
        )

        current_objective = objective(result.x)
        if current_objective < best_objective_value:
            best_objective_value = current_objective
            best_solution = result.x
            if verbose:
                print(f"New best solution found (objective value: {best_objective_value:.2e})")

        if current_objective < solution_tolerance:
            is_unique = True
            for sol in unique_solutions:
                if np.allclose(result.x, sol, rtol=tolerance/10, atol=tolerance/10):
                    is_unique = False
                    break

            if is_unique:
                unique_solutions.append(result.x)
                if verbose:
                    delta_c = result.x[:n_param]
                    delta_s = result.x[n_param:]
                    print(f"Found valid solution (attempt {i+1}):")
                    print(f"  delta_c = {delta_c}")
                    print(f"  delta_s = {delta_s}")

                    # Verify solution
                    det_values = det_D_hermitian(result.x, ind_stopband, ind_param, w_tilde_stopband_edges,
                                                 is_default_value, max_ind_stopband)
                    print("Verification:")
                    for j, val in enumerate(det_values):
                        print(f"  Det{j+1} = {val:.2e}")
                    print()
        elif verbose:
            print(f"No valid solution from starting point {i+1}, objective: {current_objective:.2e}")

    # Return best-effort solution if no valid ones found
    if len(unique_solutions) == 0 and always_return_best and best_solution is not None:
        if verbose:
            print(f"\nNo solutions met tolerance {solution_tolerance}.")
            print(f"Returning best solution (objective: {best_objective_value:.2e})")
        unique_solutions.append(best_solution)

    if verbose:
        print(f"\nFound {len(unique_solutions)} solutions.")
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


def multi_stage_solution_finding_hermitian(ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param,
                                            bounds=(-0.5, 0.5), n_attempts=20, tolerance=1e-8, always_return_best=True, verbose=True):
    """
    Multi-stage approach to find solutions with Hermitian coupling matrix.

    Parameters:
    ind_stopband (array): Array of stopband indices
    ind_param (array): Array of stopband indices for parameters (non-default values)
    w_tilde_stopband_edges (array): Array of normalized frequencies at stopband edges
    is_default_value (array): Boolean array indicating which edges are default values
    max_ind_stopband (int): Maximum stopband index
    n_param (int): Number of harmonics (actual optimization has 2*n_param parameters)
    bounds (tuple): Bounds for the solutions
    n_attempts (int): Number of starting points to try
    tolerance (float): Tolerance for accepting solutions
    always_return_best (bool): If True, always return the best solution found
    verbose (bool): Whether to print progress messages

    Returns:
    list: List of refined solutions, each of shape (2*n_param,)
    """
    n_opt_param = 2 * n_param

    # First stage: Find approximate solutions
    first_stage_tolerance = tolerance * 100
    if verbose:
        print(f"Stage 1: Finding approximate solutions (tolerance: {first_stage_tolerance})...")
    initial_solutions = find_all_solutions_hermitian(
        ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param,
        bounds=bounds,
        n_attempts=n_attempts,
        tolerance=first_stage_tolerance,
        always_return_best=always_return_best,
        verbose=verbose
    )

    if len(initial_solutions) == 0:
        if verbose:
            print("No solutions found in Stage 1. Skipping refinement.")
        return []

    # Second stage: Refine solutions
    if verbose:
        print(f"\nStage 2: Refining solutions (tolerance: {tolerance})...")
    refined_solutions = []

    best_refined_solution = None
    best_refined_objective_value = float('inf')

    def objective(delta_cs):
        det_values = det_D_hermitian(delta_cs, ind_stopband, ind_param, w_tilde_stopband_edges,
                                     is_default_value, max_ind_stopband)
        return sum(d*d for d in det_values)

    for i, sol in enumerate(initial_solutions):
        if verbose:
            print(f"Refining solution {i+1}/{len(initial_solutions)}")

        bounds_list = []
        for val in sol:
            lower = max(bounds[0], val - 0.05)
            upper = min(bounds[1], val + 0.05)
            bounds_list.append((lower, upper))

        try:
            result = minimize(
                objective,
                sol,
                method='L-BFGS-B',
                bounds=bounds_list,
                options={'ftol': tolerance, 'gtol': tolerance, 'maxiter': 10000}
            )

            current_objective = objective(result.x)
            if current_objective < best_refined_objective_value:
                best_refined_objective_value = current_objective
                best_refined_solution = result.x
                if verbose:
                    print(f"New best refined solution (objective: {best_refined_objective_value:.2e})")

            if current_objective < tolerance:
                refined_solutions.append(result.x)
                if verbose:
                    delta_c = result.x[:n_param]
                    delta_s = result.x[n_param:]
                    print(f"Refined solution meets tolerance:")
                    print(f"  delta_c = {delta_c}")
                    print(f"  delta_s = {delta_s}")
            else:
                if verbose:
                    print(f"Solution does not meet tolerance. Objective: {current_objective:.2e}")
        except Exception as e:
            if verbose:
                print(f"Error during refinement: {e}")
            current_objective = objective(sol)
            if current_objective < best_refined_objective_value:
                best_refined_objective_value = current_objective
                best_refined_solution = sol

    if len(refined_solutions) == 0 and always_return_best and best_refined_solution is not None:
        if verbose:
            print(f"\nNo refined solutions met tolerance {tolerance}.")
            print(f"Returning best refined solution (objective: {best_refined_objective_value:.2e})")
        refined_solutions.append(best_refined_solution)

    if verbose:
        print(f"\nFinal results: Found {len(refined_solutions)} refined Hermitian solutions.")

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
# Floquet taper utilities

def _klopfenstein_phi(z, A):
    """Klopfenstein φ function used by the equiripple impedance taper.

    φ(z, A) = ∫₀^z I₁(A·√(1−y²)) / (A·√(1−y²)) dy

    Odd in z; defined on |z| ≤ 1. For |z| > 1 the integrand is imaginary and we
    clamp to ±1 (Klopfenstein's profile is only defined inside the taper).

    Parameters
    ----------
    z : float
        Position parameter, normally in [-1, 1].
    A : float
        Klopfenstein design parameter.

    Returns
    -------
    float
        Value of φ(z, A).
    """
    if abs(z) < 1e-15:
        return 0.0
    sign = 1.0 if z > 0 else -1.0
    z_abs = min(abs(z), 1.0)

    def integrand(y):
        u = 1.0 - y * y
        if u <= 0.0:
            return 0.0
        s = A * np.sqrt(u)
        if s < 1e-12:
            # I_1(x)/x → 1/2 as x → 0
            return 0.5
        return _bessel_iv(1, s) / s

    val, _ = _quad_integrate(integrand, 0.0, z_abs)
    return sign * val


def compute_linear_Z_profile(Ntot_cell, taper_cells, Z_env, Z_TWPA):
    """Linear-in-cell-index impedance profile across the device.

    The taper runs linearly from Z_env at the edge cell to Z_TWPA at the last
    taper cell; the center cells are clamped at Z_TWPA. Mirrored on the right.

    Parameters
    ----------
    Ntot_cell : int
        Total number of cells.
    taper_cells : int
        Number of cells per taper region.
    Z_env : float
        Environment impedance (at the device edges).
    Z_TWPA : float
        Center / design impedance.

    Returns
    -------
    Z_profile : np.ndarray, shape (Ntot_cell,)
        Per-cell impedance.
    """
    Z_profile = np.full(Ntot_cell, float(Z_TWPA))
    if Z_env == Z_TWPA or taper_cells <= 0:
        return Z_profile
    if taper_cells == 1:
        Z_profile[0] = Z_env
        Z_profile[-1] = Z_env
        return Z_profile

    # Left taper: cell index n in [0, taper_cells-1] maps linearly to [Z_env, Z_TWPA]
    n = np.arange(taper_cells)
    taper_vals = Z_env + (Z_TWPA - Z_env) * n / (taper_cells - 1)
    Z_profile[:taper_cells] = taper_vals
    # Right taper is the mirror
    Z_profile[Ntot_cell - taper_cells:] = taper_vals[::-1]
    return Z_profile


def compute_klopfenstein_Z_profile(Ntot_cell, taper_cells, Z_env, Z_TWPA,
                                    A=None, max_ripple=0.05):
    """Klopfenstein (equiripple-optimal) impedance profile across the device.

    Within each taper region, the profile is the rescaled Klopfenstein taper

        ln(Z(n)/sqrt(Z_env · Z_TWPA)) = Γ_0 · φ(2n/(taper_cells-1) − 1, A) / φ(1, A)

    where Γ_0 = ½·ln(Z_TWPA/Z_env) is the DC reflection magnitude and φ(z, A) is
    the Klopfenstein function (integral involving I₁). The rescaling by 1/φ(1, A)
    forces Z(0) = Z_env and Z(taper_cells−1) = Z_TWPA exactly, sacrificing a
    small amount of canonical-Klopfenstein equiripple optimality for clean
    port/center matching. Center cells stay at Z_TWPA. Right taper is mirrored.

    Parameters
    ----------
    Ntot_cell : int
        Total number of cells.
    taper_cells : int
        Number of cells per taper region.
    Z_env : float
        Environment impedance (at the device edges).
    Z_TWPA : float
        Center / design impedance.
    A : float, optional
        Klopfenstein design parameter. Larger A ⇒ sharper cutoff and larger
        in-band ripple. If None (default), computed from `max_ripple` via the
        standard relation A = cosh⁻¹(Γ_0 / Γ_m).
    max_ripple : float, optional
        Target maximum in-band reflection amplitude (Γ_m), used only when
        A is None. Default 0.05.

    Returns
    -------
    Z_profile : np.ndarray, shape (Ntot_cell,)
        Per-cell impedance.
    """
    Z_profile = np.full(Ntot_cell, float(Z_TWPA))
    if Z_env == Z_TWPA or taper_cells <= 0:
        return Z_profile
    if taper_cells == 1:
        Z_profile[0] = Z_env
        Z_profile[-1] = Z_env
        return Z_profile

    Gamma_0 = 0.5 * np.log(Z_TWPA / Z_env)
    if A is None:
        ratio = abs(Gamma_0) / max(max_ripple, 1e-12)
        # cosh⁻¹ requires argument ≥ 1; clamp gracefully.
        A = float(np.arccosh(max(ratio, 1.0)))
        if A < 1e-6:
            A = 1e-6  # numerically safe lower bound
    A = float(A)

    phi_max = _klopfenstein_phi(1.0, A)
    if abs(phi_max) < 1e-12:
        # Degenerate case (A → 0): fall back to linear.
        return compute_linear_Z_profile(Ntot_cell, taper_cells, Z_env, Z_TWPA)

    # Build the left-taper profile, then mirror.
    taper_vals = np.empty(taper_cells)
    log_geo_mean = 0.5 * np.log(Z_env * Z_TWPA)
    for n in range(taper_cells):
        z = 2.0 * n / (taper_cells - 1) - 1.0
        phi_val = _klopfenstein_phi(z, A)
        taper_vals[n] = np.exp(log_geo_mean + Gamma_0 * phi_val / phi_max)

    Z_profile[:taper_cells] = taper_vals
    Z_profile[Ntot_cell - taper_cells:] = taper_vals[::-1]
    return Z_profile


def compute_floquet_profile(Ntot_cell, profile_type='gaussian', floquet_taper_width=0.3,
                            w_min=0.05, w_target=0.95):
    """
    Compute the Floquet nonlinearity weight profile w(n) over the full device length.

    The returned weight profile drives the **nonlinearity** taper only:
    Istar(n) for KI, Lj(n) for bare JJ, beta_L_eff(n) for rf_squid. The impedance
    ramp is independent (see `compute_linear_Z_profile` /
    `compute_klopfenstein_Z_profile`). See "Floquet nonlinearity taper" in
    docs/engineering_notes.md.

    Parameters
    ----------
    Ntot_cell : int
        Total number of cells.
    profile_type : str
        'gaussian' or 'tukey'.
    floquet_taper_width : float
        Total fraction of the line used for the nonlinearity ramp. Each side is
        floquet_taper_width/2 of the line, the center is (1 - floquet_taper_width).
        E.g., floquet_taper_width=0.5 means 25% taper per side, 50% center.
    w_min : float
        Minimum weight at the device edges. Default 0.05. Kept above 0 to avoid
        singularities in the nonlinearity branch (rf_squid `c1_cp = 1/beta_L_eff
        + cos(phi_dc)` and bare-JJ `Ic = φ₀/Lj` both blow up at w → 0).
    w_target : float
        Target weight at the boundary between taper and center.
        Default 0.95: the first center cell has w = 0.95 (Gaussian). The
        Gaussian then drifts up to ~`w_min + (w_target - w_min)/w_target`
        (≈ 0.997 for defaults) as you go deeper into the center. Tukey hits
        exactly 1.0 at the boundary and stays at 1.0 in the center.

    Returns
    -------
    weights : np.ndarray, shape (Ntot_cell,)
        Taper weights in [w_min, ~1], w_min at edges, ~1 in center.
    center_start : int
        First cell index in center region.
    center_end : int
        Last cell index + 1 in center region.
    """
    n = np.arange(Ntot_cell)
    taper_cells = int(floquet_taper_width * Ntot_cell / 2)

    if profile_type == 'gaussian':
        if taper_cells <= 0:
            weights = np.ones(Ntot_cell)
        else:
            sigma = taper_cells / np.sqrt(-2 * np.log(1 - w_target))
            g_left = 1 - np.exp(-n ** 2 / (2 * sigma ** 2))
            g_right = 1 - np.exp(-(Ntot_cell - 1 - n) ** 2 / (2 * sigma ** 2))
            g = np.minimum(g_left, g_right)
            weights = w_min + (w_target - w_min) * g / w_target
    elif profile_type == 'tukey':
        weights = np.ones(Ntot_cell)
        if taper_cells > 0:
            raw_taper = 0.5 * (1 - np.cos(np.pi * np.arange(taper_cells) / taper_cells))
            taper = w_min + (1 - w_min) * raw_taper
            weights[:taper_cells] = taper
            weights[-taper_cells:] = taper[::-1]
    else:
        raise ValueError(f"Unknown Floquet profile type: {profile_type}")

    center_start = taper_cells
    center_end = Ntot_cell - taper_cells
    if center_start >= center_end:
        center_start = Ntot_cell // 2
        center_end = center_start + 1

    return weights, center_start, center_end


def compute_floquet_cell_parameters(weights, nonlinearity, jj_structure_type,
                                     LJ0_H, Lg_H, beta_L, phi_dc, CJ_F, Ic_JJ_A,
                                     Istar_A, Id_A, L0_pH,
                                     LTLsec_H, n_jj_struct,
                                     LinfLF1_H=None, L0LF2_H=None, Foster_form_L=1,
                                     rf_squid_constant_plasma=False,
                                     L0_H_KI_percell=None):
    """
    Compute per-cell nonlinearity parameters from Floquet weights.

    The Floquet weight w(n) scales the nonlinear inductance at each cell.
    The remainder inductance adjusts so the total series inductance is preserved.

    Parameters
    ----------
    weights : np.ndarray, shape (Ntot_cell,)
        Floquet taper weights in (0, 1].
    nonlinearity : str
        'JJ' or 'KI'.
    jj_structure_type : str
        'jj' or 'rf_squid' (only used when nonlinearity='JJ').
    LJ0_H : float or None
        Nominal JJ inductance (center value).
    Lg_H : float or None
        Geometric inductance for rf_squid (stays constant in taper).
    beta_L : float or None
        RF-SQUID participation ratio (center value).
    phi_dc : float
        DC flux bias.
    CJ_F : float or None
        Junction capacitance (center value).
    Ic_JJ_A : float or None
        Critical current in Amps.
    Istar_A : float or None
        KI scaling current.
    Id_A : float or None
        KI DC bias current.
    L0_pH : float or None
        KI inductance per cell in pH.
    LTLsec_H : float
        Total TL section inductance.
    n_jj_struct : int
        Number of JJ structures per cell.
    LinfLF1_H : float or None
        Total series inductor from Foster form 1 filter design.
    L0LF2_H : float or None
        Total series inductor from Foster form 2 filter design.
    Foster_form_L : int
        1 or 2.
    rf_squid_constant_plasma : bool
        rf_squid only. If True, populate 'Cj_extra_F' with the per-cell extra
        shunt capacitance needed to keep the plasma frequency (Lj||Lg)*Cj_total
        constant along the taper:
        Cj_total(n) = CJ_F * (1 + beta_L*w(n)*cos(phi_dc)) / (1 + beta_L*cos(phi_dc))
        Cjx(n)      = max(0, Cj_total(n) - CJ_F * w(n))
        If False, 'Cj_extra_F' is filled with zeros. Ignored for bare JJ and KI.
    L0_H_KI_percell : np.ndarray, optional
        KI only. Per-cell kinetic inductance array. Used when the impedance and/or
        cutoff is tapered along the line and L0(n) needs to follow the local Z(n)
        and fc(n) (see "Floquet taper for KI nonlinearity" in engineering_notes.md).
        If None, L0(n) = L0_pH * 1e-12 (constant) as a backward-compat default.
        Ignored for JJ and rf_squid.

    Returns
    -------
    dict with per-cell arrays:
        'L0_H' : np.ndarray -- per-cell effective NL inductance (Lj||Lg for rf_squid)
        'Lj_H' : np.ndarray -- per-cell JJ kinetic inductance (= L0_H for bare JJ;
                 = Lg/(beta_L*w(n)) for rf_squid). Used by the netlist builder for
                 the JJ component of the rf-SQUID, distinct from the parallel L0.
        'CJ_F' : np.ndarray or None -- per-cell junction capacitance
        'Cj_extra_F' : np.ndarray -- per-cell extra shunt cap for rf_squid_constant_plasma
                       (only present for jj_structure_type='rf_squid'; zeros when
                       rf_squid_constant_plasma=False)
        'LTLsec_rem_H' : np.ndarray -- per-cell TL section remainder
        'filter_rem_H' : np.ndarray or None -- per-cell filter section remainder
        'c1_taylor' through 'c4_taylor' : np.ndarray -- per-cell Taylor coefficients
        'epsilon_perA' : np.ndarray -- per-cell epsilon
        'xi_perA2' : np.ndarray -- per-cell xi
    """
    Ntot = len(weights)
    result = {}

    if nonlinearity == 'JJ':
        if jj_structure_type == 'jj':
            # Bare JJ: Lj scales by weight, CJ inversely (keep fJ constant)
            L0_H_arr = LJ0_H * weights
            CJ_F_arr = CJ_F / np.maximum(weights, 1e-30)
            # For bare JJ, Lj_dyn = L0_H (no Lg branch). Expose for consistency.
            result['Lj_H'] = L0_H_arr
            # Taylor coefficients are fixed for bare JJ
            result['c1_taylor'] = np.zeros(Ntot)
            result['c2_taylor'] = np.full(Ntot, 0.5)
            result['c3_taylor'] = np.zeros(Ntot)
            result['c4_taylor'] = np.full(Ntot, 5.0 / 24.0)
            # Nonlinearity coefficients (per A², matching the designer's center-cell
            # formula (3*c2² - c1*c3)/(2*c1⁴)/Ic² with bare-JJ c1=1, c2=0, c3=-1).
            Ic_arr = phi0 / L0_H_arr
            c1_cp = 1.0  # c1_currentphase for bare JJ at phi_dc=0
            c2_cp = 0.0
            result['epsilon_perA'] = np.zeros(Ntot)
            result['xi_perA2'] = 0.5 / (Ic_arr ** 2)

        elif jj_structure_type == 'rf_squid':
            # RF-SQUID: scale beta_L by w(n). Small beta_L at edges = linear (L0 ≈ Lg).
            # The effective inductance L0 = Lg/(1 + beta_L*w(n)*cos(phi_dc)) is NOT
            # proportional to w(n) — it's a nonlinear function.
            beta_L_eff = beta_L * weights
            cos_phi = np.cos(phi_dc)
            sin_phi = np.sin(phi_dc)
            denom1 = 1 + beta_L_eff * cos_phi

            L0_H_arr = Lg_H / denom1
            # CJ scales inversely with Lj = Lg/beta_L_eff to keep plasma freq constant
            Lj_arr = Lg_H / np.maximum(beta_L_eff, 1e-30)
            CJ_F_arr = CJ_F * (LJ0_H / Lj_arr)
            # Expose Lj_dyn (the JJ kinetic inductance per cell) so the netlist
            # builder can use it for the JJ component (instead of L0_H, which
            # is the effective parallel-combined inductance).
            result['Lj_H'] = Lj_arr

            # Optional: extra shunt cap to keep the rf_squid plasma frequency
            # ((Lj||Lg) * Cj_total) constant along the line. The intrinsic JJ
            # cap scales as CJ_F * w(n), but the parallel L (= Lg/(1+beta_L*w*cos))
            # varies differently. Adding an extra cap brings the total to
            # Cj_total(n) = CJ_F * (1 + beta_L*w*cos)/(1 + beta_L*cos) so that
            # (Lj||Lg)(n) * Cj_total(n) = (Lj||Lg)_center * CJ_F.
            if rf_squid_constant_plasma:
                denom_center = 1 + beta_L * cos_phi
                Cj_total = CJ_F * denom1 / denom_center  # denom1 = 1 + beta_L_eff*cos_phi
                result['Cj_extra_F'] = np.maximum(0, Cj_total - CJ_F_arr)
            else:
                result['Cj_extra_F'] = np.zeros(Ntot)

            result['c1_taylor'] = beta_L_eff * sin_phi / denom1
            result['c2_taylor'] = beta_L_eff * (cos_phi + beta_L_eff * (1 + sin_phi ** 2)) / (2 * denom1)

            denom3 = denom1 ** 3
            result['c3_taylor'] = beta_L_eff * sin_phi * (
                6 * beta_L_eff ** 2 * sin_phi ** 2 + 5 * beta_L_eff ** 2 * cos_phi ** 2
                + 4 * beta_L_eff * cos_phi - 1) / (6 * denom3)

            denom4 = denom1 ** 4
            result['c4_taylor'] = beta_L_eff * (
                -cos_phi + 3 * beta_L_eff * cos_phi ** 2
                + 9 * beta_L_eff ** 2 * cos_phi ** 3
                + 5 * beta_L_eff ** 3 * cos_phi ** 4
                - 8 * beta_L_eff * sin_phi ** 2
                + 20 * beta_L_eff ** 2 * cos_phi * sin_phi ** 2
                + 28 * beta_L_eff ** 3 * cos_phi ** 2 * sin_phi ** 2
                + 24 * beta_L_eff ** 3 * sin_phi ** 4
            ) / (24 * denom4)

            c1_cp = (1 / beta_L_eff + cos_phi)
            c2_cp = -sin_phi
            Ic_JJ_arr = phi0 / (Lg_H / beta_L_eff)
            result['epsilon_perA'] = -c2_cp / (c1_cp ** 2) / Ic_JJ_arr
            result['xi_perA2'] = (3 * c2_cp ** 2 - c1_cp * (-cos_phi)) / (2 * c1_cp ** 4) / (Ic_JJ_arr ** 2)

        result['CJ_F'] = CJ_F_arr

    elif nonlinearity == 'KI':
        if L0_H_KI_percell is not None:
            L0_H_arr = np.asarray(L0_H_KI_percell, dtype=float)
            if L0_H_arr.shape != (Ntot,):
                raise ValueError(
                    f"L0_H_KI_percell shape {L0_H_arr.shape} does not match Ntot={Ntot}")
        else:
            L0_val = L0_pH * 1e-12
            L0_H_arr = np.full(Ntot, L0_val)
        Istar_eff = Istar_A / np.maximum(weights, 1e-30)

        result['epsilon_perA'] = 2 * Id_A / (Istar_eff ** 2 + Id_A ** 2)
        result['xi_perA2'] = 1.0 / (Istar_eff ** 2 + Id_A ** 2)

        result['c1_taylor'] = phi0 / L0_H_arr * result['epsilon_perA']
        result['c2_taylor'] = 0.5 * (phi0 / L0_H_arr) ** 2 * (2 * result['xi_perA2'] - 3 * result['epsilon_perA'] ** 2)
        result['c3_taylor'] = None
        result['c4_taylor'] = None
        result['CJ_F'] = None

    result['L0_H'] = L0_H_arr

    # TL section remainder
    result['LTLsec_rem_H'] = np.maximum(0, LTLsec_H - n_jj_struct * L0_H_arr)

    # Filter section remainder (for the active Foster form)
    if Foster_form_L == 1 and LinfLF1_H is not None and not np.any(np.isinf(LinfLF1_H)):
        L_total = np.atleast_1d(LinfLF1_H)
        if L_total.size == 1:
            result['filter_rem_H'] = np.maximum(0, float(L_total.item()) - n_jj_struct * L0_H_arr)
        else:
            result['filter_rem_H'] = np.maximum(0, L_total[:, np.newaxis] - n_jj_struct * L0_H_arr)
    elif Foster_form_L == 2 and L0LF2_H is not None and not np.any(np.isinf(L0LF2_H)):
        L_total = np.atleast_1d(L0LF2_H)
        if L_total.size == 1:
            result['filter_rem_H'] = np.maximum(0, float(L_total.item()) - n_jj_struct * L0_H_arr)
        else:
            result['filter_rem_H'] = np.maximum(0, L_total[:, np.newaxis] - n_jj_struct * L0_H_arr)
    else:
        result['filter_rem_H'] = None

    return result


def compute_taper_arrays(
        Ntot_cell, Ncpersc_cell,
        Z_taper, Z_taper_width, Z_profile, klopfenstein_A,
        floquet_taper, floquet_taper_width, floquet_profile,
        taper_cutoff,
        Z0_ohm, Z0_TWPA_ohm, fc_TLsec_GHz,
        LTLsec_H_center, L0_H_center, g_L, g_C,
        nonlinearity, jj_structure_type='jj',
        phi_dc=0.0, beta_L=None,
        LJ0_H=None, Lg_H=None, CJ_F=None,
        Ic_JJ_A=None, Istar_A=None, Id_A=None,
        L0_pH=None, n_jj_struct=1,
        LinfLF1_H=None, L0LF2_H=None,
        Foster_form_L=1, Foster_form_C=1,
        zero_at_zero=True, select_one_form='C',
        f_zeros_GHz=None, f_poles_GHz=None, dispersion_type='periodic',
        g_C_mod=None, rf_squid_constant_plasma=False):
    """Compute all per-cell taper arrays — impedance, nonlinearity weight, and derived linear/NL params.

    This is the single source of truth for the tapered-TWPA per-cell design and
    is called by both `atl_twpa_designer._compute_taper_arrays` and the netlist
    builder's workspace preparation. The two tapers (impedance and Floquet
    nonlinearity) are computed independently with possibly different widths;
    the per-cell numeric region is the union of the two taper regions, the
    symbolic-supercell center region is the intersection.

    Parameters
    ----------
    Ntot_cell, Ncpersc_cell : int
        Total cell count, and cells per supercell (for center-region alignment).
    Z_taper : bool
        If True, ramp the cell impedance from Z0_ohm at the edges to Z0_TWPA_ohm
        in the center over `Z_taper_width` of the line. If False, Z(n)=Z0_TWPA_ohm.
    Z_taper_width : float
        Total fraction of the line used for the impedance ramp.
    Z_profile : str
        'linear' or 'klopfenstein'.
    klopfenstein_A : float or None
        Klopfenstein design parameter; None ⇒ auto from 5% max ripple.
    floquet_taper : bool
        If True, ramp the nonlinearity from weak at edges to full strength in
        the center over `floquet_taper_width` of the line. If False, w(n)=1.
    floquet_taper_width : float
        Total fraction of the line used for the nonlinearity ramp.
    floquet_profile : str
        'gaussian' or 'tukey'.
    taper_cutoff : bool
        False: fc(n)=fc_center, C(n) absorbs variation. True: C(n)=C_center,
        fc(n) absorbs variation.
    Z0_ohm, Z0_TWPA_ohm, fc_TLsec_GHz : float
        Environment impedance, center TWPA impedance, center cutoff frequency.
    LTLsec_H_center, L0_H_center : float
        Center per-cell total series inductance and nonlinear-element inductance.
    g_L, g_C : float
        Filter prototype g-values.
    nonlinearity : str
        'JJ' or 'KI'.
    jj_structure_type : str
        'jj' or 'rf_squid'. Ignored for KI.
    phi_dc, beta_L, LJ0_H, Lg_H, CJ_F, Ic_JJ_A, Istar_A, Id_A, L0_pH, n_jj_struct
        Forwarded to `compute_floquet_cell_parameters`.
    LinfLF1_H, L0LF2_H, Foster_form_L, Foster_form_C, zero_at_zero, select_one_form
        Filter design parameters. Used both for `compute_floquet_cell_parameters`
        and (when filters exist and the linear design varies) for per-cell filter
        recomputation via `calculate_filter_components`.
    f_zeros_GHz, f_poles_GHz, dispersion_type
        Filter geometry. When `dispersion_type` is 'filter' or 'both' and
        `f_zeros_GHz`/`f_poles_GHz` are non-empty, the function recomputes per-cell
        filter components in the union taper region.
    g_C_mod : np.ndarray or None
        Per-cell shunt-cap modulation (length Ntot_cell). Used when present to
        overlay periodic modulation on the per-cell shunt cap.
    rf_squid_constant_plasma : bool
        Forwarded to `compute_floquet_cell_parameters`.

    Returns
    -------
    dict with keys:
        'w_percell'              -- nonlinearity weight w(n) (shape Ntot_cell)
        'Z_percell'              -- cell impedance Z(n) (shape Ntot_cell)
        'fc_percell'             -- cell cutoff fc(n), GHz (shape Ntot_cell)
        'LTLsec_H_percell'       -- per-cell total series inductance (None if not varying)
        'CTLsec_F'               -- per-cell shunt cap (None if not varying)
        'w_eff'                  -- effective NL weight for cell-component scaling
                                    (= w for bare JJ, = (1+βcos)/(1+wβcos) for rf_squid,
                                    = 1 for KI)
        'center_start', 'center_end' -- symbolic-supercell center region (cells where
                                        both w=1 AND Z=Z_TWPA, supercell-aligned)
        'width'                  -- number of taper cells on each side (= center_start)
        'n_periodic_sc'          -- number of symbolic supercells in the center
        'linear_varies'          -- True if Z(n), fc(n), or C(n) varies along the line
        'floquet_cell_params'    -- full output of compute_floquet_cell_parameters
        'L0_H_percell'           -- per-cell NL inductance (from floquet_cell_params)
        'CJ_F_percell'           -- per-cell JJ cap (from floquet_cell_params)
        'floquet_filter_components' -- {cell_idx: filter_components_dict} for cells in the
                                       union taper region when filters exist and the linear
                                       design varies (empty dict otherwise)
    """
    Ntot = int(Ntot_cell)
    Ncpersc = int(Ncpersc_cell) if Ncpersc_cell else 1

    # --- Profile 1: nonlinearity weight w(n) ---
    if floquet_taper:
        weights, _, _ = compute_floquet_profile(
            Ntot, floquet_profile, floquet_taper_width)
        floquet_taper_cells = int(floquet_taper_width * Ntot / 2)
    else:
        weights = np.ones(Ntot)
        floquet_taper_cells = 0

    # --- Profile 2: cell impedance Z(n) ---
    if Z_taper and Z0_TWPA_ohm != Z0_ohm:
        Z_taper_cells = int(Z_taper_width * Ntot / 2)
        if Z_profile == 'klopfenstein':
            Z_percell = compute_klopfenstein_Z_profile(
                Ntot, Z_taper_cells, Z0_ohm, Z0_TWPA_ohm,
                A=klopfenstein_A, max_ripple=0.05)
        else:
            Z_percell = compute_linear_Z_profile(
                Ntot, Z_taper_cells, Z0_ohm, Z0_TWPA_ohm)
    else:
        Z_percell = np.full(Ntot, float(Z0_TWPA_ohm))
        Z_taper_cells = 0

    # --- Union taper region, supercell-aligned for the symbolic center power ---
    taper_cells_raw = max(floquet_taper_cells, Z_taper_cells)
    width = int(round(taper_cells_raw / Ncpersc) * Ncpersc) if Ncpersc > 0 else taper_cells_raw
    width = min(width, Ntot // 2)
    center_start = width
    center_end = Ntot - width
    if Ncpersc > 0:
        n_periodic_sc = int((center_end - center_start) / Ncpersc)
    else:
        n_periodic_sc = max(0, center_end - center_start)

    # --- Effective NL weight for cell-component scaling ---
    # For KI: L0 is decoupled from w (the impedance taper drives L0(n) directly),
    # so w_eff = 1 — the nonlinearity ramp doesn't disturb the linear cell design.
    # For bare JJ: w_eff = w (Lj scales with w, cell is "thinner" at edges).
    # For rf_squid: w_eff = (1+β·cos)/(1+w·β·cos) > 1 at edges (cell is "thicker").
    if nonlinearity == 'JJ' and jj_structure_type == 'rf_squid':
        cos_phi = np.cos(phi_dc)
        beta_val = beta_L if beta_L is not None else 0.0
        w_eff = (1 + beta_val * cos_phi) / (1 + weights * beta_val * cos_phi)
    elif nonlinearity == 'JJ':
        w_eff = weights
    else:  # KI
        w_eff = np.ones(Ntot)

    # --- fc(n) per taper_cutoff semantics ---
    if taper_cutoff:
        safe_Z = np.maximum(Z_percell, 1e-30)
        safe_w_eff = np.maximum(w_eff, 1e-30)
        # General formula: L_total(n) = g_L·Z(n)/(2π·fc(n)) = L_total_center · (Z(n)/Z_TWPA) / w_eff(n) · ?
        # Equivalent: fc(n) = g_L·Z(n)/(2π · L_total_center · w_eff(n))
        fc_percell = (g_L * safe_Z
                      / (2 * np.pi * LTLsec_H_center * safe_w_eff)) * 1e-9
    else:
        fc_percell = np.full(Ntot, float(fc_TLsec_GHz))

    # --- Does the linear design actually vary along the line? ---
    # Z taper always makes it vary (Z, C, fc all depend on it).
    # Floquet taper + taper_cutoff makes fc vary only for JJ (w_eff != 1).
    # KI without Z taper has w_eff = 1 ⇒ uniform line even if floquet_taper is on.
    linear_varies = bool(Z_taper and Z0_TWPA_ohm != Z0_ohm) \
                    or bool(floquet_taper and taper_cutoff and nonlinearity == 'JJ')

    # --- Per-cell L_total and C arrays (only when varying) ---
    LTLsec_H_percell = None
    CTLsec_F_percell = None
    if linear_varies:
        LTLsec_H_percell = g_L * Z_percell / (2 * np.pi * fc_percell * 1e9)
        if g_C_mod is not None and hasattr(g_C_mod, '__len__') and len(g_C_mod) == Ntot:
            g_C_eff = np.asarray(g_C_mod)
        else:
            g_C_eff = g_C
        CTLsec_F_percell = g_C_eff / (Z_percell * 2 * np.pi * fc_percell * 1e9)

    # --- Per-cell filter recomputation in the union taper region ---
    floquet_filter_components = {}
    if (linear_varies and dispersion_type in ['filter', 'both']
            and f_zeros_GHz is not None and f_poles_GHz is not None
            and (len(np.atleast_1d(f_zeros_GHz)) + len(np.atleast_1d(f_poles_GHz)) > 0)):
        taper_indices = list(range(width)) + list(range(Ntot - width, Ntot))
        is_rf_squid = (nonlinearity == 'JJ' and jj_structure_type == 'rf_squid')
        cos_phi = np.cos(phi_dc)
        L0_ratio_KI = ((L0_H_center / LTLsec_H_center)
                       if (LTLsec_H_center > 0 and nonlinearity == 'KI') else 1.0)
        f_zeros_arr = np.atleast_1d(f_zeros_GHz)
        f_poles_arr = np.atleast_1d(f_poles_GHz)

        for idx in taper_indices:
            Z_cell = float(Z_percell[idx])
            fc_cell = float(fc_percell[idx])
            w_zeros_cell = f_zeros_arr / fc_cell
            w_poles_cell = f_poles_arr / fc_cell

            w_n = float(weights[idx])
            if is_rf_squid:
                L0_cell = Lg_H / (1 + beta_L * w_n * cos_phi)
            elif nonlinearity == 'JJ':
                L0_cell = LJ0_H * w_n
            else:  # KI: L0(n) tracks L_total(n) proportionally
                L0_cell = L0_ratio_KI * float(LTLsec_H_percell[idx])
            L0_total_H_cell = n_jj_struct * L0_cell

            (LinfLF1, C0LF1, LiLF1, CiLF1, LinfLF1_rem,
             L0LF2_v, CinfLF2, LiLF2, CiLF2, L0LF2_rem,
             LinfCF1, C0CF1, LiCF1, CiCF1,
             L0CF2_v, CinfCF2, LiCF2, CiCF2,
             _, _, _, _) = calculate_filter_components(
                Foster_form_L, Foster_form_C, g_L, g_C,
                w_zeros_cell, w_poles_cell, Z_cell, fc_cell,
                zero_at_zero, L0_total_H_cell, select_one_form, verbose=False)

            floquet_filter_components[idx] = {
                'LinfLF1_H': LinfLF1, 'LinfLF1_rem_H': LinfLF1_rem,
                'C0LF1_F': C0LF1, 'LiLF1_H': LiLF1, 'CiLF1_F': CiLF1,
                'L0LF2_H': L0LF2_v, 'L0LF2_rem_H': L0LF2_rem,
                'CinfLF2_F': CinfLF2, 'LiLF2_H': LiLF2, 'CiLF2_F': CiLF2,
                'LinfCF1_H': LinfCF1, 'C0CF1_F': C0CF1, 'LiCF1_H': LiCF1, 'CiCF1_F': CiCF1,
                'L0CF2_H': L0CF2_v, 'CinfCF2_F': CinfCF2, 'LiCF2_H': LiCF2, 'CiCF2_F': CiCF2,
            }

    # --- KI: L0(n) tracks L_total(n) only when the linear design varies ---
    L0_H_KI_percell = None
    if nonlinearity == 'KI' and linear_varies and LTLsec_H_percell is not None:
        L0_ratio = (L0_H_center / LTLsec_H_center) if LTLsec_H_center > 0 else 1.0
        L0_H_KI_percell = L0_ratio * LTLsec_H_percell

    # --- Per-cell nonlinearity split ---
    floquet_cell_params = compute_floquet_cell_parameters(
        weights, nonlinearity, jj_structure_type,
        LJ0_H, Lg_H, beta_L, phi_dc, CJ_F,
        Ic_JJ_A, Istar_A, Id_A,
        L0_pH, LTLsec_H_center, n_jj_struct,
        LinfLF1_H=LinfLF1_H, L0LF2_H=L0LF2_H,
        Foster_form_L=Foster_form_L,
        rf_squid_constant_plasma=rf_squid_constant_plasma,
        L0_H_KI_percell=L0_H_KI_percell,
    )
    L0_H_percell = floquet_cell_params['L0_H']
    CJ_F_percell = floquet_cell_params.get('CJ_F')

    # Refresh TL section remainder against the per-cell L_total when it varies.
    if linear_varies and LTLsec_H_percell is not None:
        floquet_cell_params['LTLsec_rem_H'] = np.maximum(
            0, LTLsec_H_percell - n_jj_struct * L0_H_percell)

    return {
        'w_percell': weights,
        'Z_percell': Z_percell,
        'fc_percell': fc_percell,
        'LTLsec_H_percell': LTLsec_H_percell,
        'CTLsec_F': CTLsec_F_percell,
        'w_eff': w_eff,
        'center_start': center_start,
        'center_end': center_end,
        'width': width,
        'n_periodic_sc': n_periodic_sc,
        'linear_varies': linear_varies,
        'floquet_cell_params': floquet_cell_params,
        'L0_H_percell': L0_H_percell,
        'CJ_F_percell': CJ_F_percell,
        'floquet_filter_components': floquet_filter_components,
    }


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

    # Handle w=0 (DC): capacitors are open circuit (Z=inf), inductors are short circuit (Z=0)
    if w == 0:
        # At DC, if there's a capacitor C0 in series, impedance is infinite
        if C0LF1_F != np.inf and C0LF1_F != 0:
            return np.inf
        # Otherwise, inductors are short circuits, so Z=0
        return 0.0

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

    # Handle w=0 (DC): inductor admittance 1/(jwL) diverges, capacitor admittance jwC=0
    if w == 0:
        # At DC, if there's an inductor L0 in shunt, admittance is infinite
        if L0LF2_H != np.inf and L0LF2_H != 0:
            return np.inf, np.inf
        # Otherwise, capacitors have zero admittance at DC
        return 0.0, 0.0

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

    # Handle w=0 (DC): capacitors are open circuit (Z=inf), inductors are short circuit (Z=0)
    if w == 0:
        # At DC, if there's a capacitor C0 in series, impedance is infinite
        if C0CF1_F != np.inf and C0CF1_F != 0:
            return np.inf
        # Otherwise, inductors are short circuits, so Z=0
        return 0.0

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

    # Handle w=0 (DC): inductor admittance 1/(jwL) diverges, capacitor admittance jwC=0
    if w == 0:
        # At DC, if there's an inductor L0 in shunt, admittance is infinite
        if L0CF2_H != np.inf and L0CF2_H != 0:
            return np.inf
        # Otherwise, capacitors have zero admittance at DC
        return 0.0

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

    # Generalized gcd over rationals: scale to integer milli-GHz (1 MHz resolution),
    # take integer gcd, then scale back. Handles fractional GHz stopbands such as 8.5
    # while staying exact for the integer-GHz case used elsewhere.
    GHZ_TO_INT = 1000  # 1 MHz resolution
    scaled = [int(round(f * GHZ_TO_INT)) for f in f_stopbands_GHz]
    f1_GHz = math.gcd(*scaled) / GHZ_TO_INT
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
                           solution_tolerance=1e-8, always_return_best=True, n_attempts=10, verbose=True,
                           force_zero_phase=True, g_C_base=None, Ncpersc_cell=None):
    """
    Calculate delta values for periodic modulation.

    This is the unified entry point that dispatches to either:
    - Real-symmetric solver (force_zero_phase=True): cosine-only modulation
    - Hermitian solver (force_zero_phase=False): cosine + sine modulation

    Parameters:
    ind_stopband (array): Array of stopband indices
    ind_param (array): Array of stopband indices for parameters (non-default values)
    w_tilde_stopband_edges (array): Array of normalized frequencies at stopband edges
    is_default_value (array): Boolean array indicating which edges are default values
    max_ind_stopband (int): Maximum stopband index
    n_param (int): Number of free parameters
    skipped_indices (list): List of skipped stopband indices
    solution_tolerance (float): Tolerance for solution precision
    always_return_best (bool): If True, always return best solution found
    n_attempts (int): Number of random starting points
    verbose (bool): Whether to print progress
    force_zero_phase (bool): If True, use real-symmetric (cosine-only) modulation.
                             If False, use Hermitian (cosine + sine) modulation.
    g_C_base (float): Base capacitance value. Used by Hermitian solver when force_zero_phase=False.
    Ncpersc_cell (int): Number of cells per supercell. Used by Hermitian solver when force_zero_phase=False.

    Returns:
    tuple: (delta_c_map, delta_s_map, selected_delta_c, selected_delta_s, modulation_info)
        - For force_zero_phase=True: delta_s_map is empty, selected_delta_s is zeros
        - For force_zero_phase=False: All values populated from Hermitian solver
    """
    if force_zero_phase:
        # Use original real-symmetric solver
        delta_map, selected_delta = _calculate_delta_values_real(
            ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value,
            max_ind_stopband, n_param, skipped_indices,
            solution_tolerance=solution_tolerance, always_return_best=always_return_best,
            n_attempts=n_attempts, verbose=verbose
        )

        # Convert to unified return format
        delta_c_map = delta_map
        delta_s_map = {}
        selected_delta_c = selected_delta
        selected_delta_s = np.zeros_like(selected_delta)
        modulation_info = {
            'mode': 'real_symmetric',
            'peak_deviation': None,
            'g_min': None,
            'g_max': None,
            'feasible': True  # Assumed feasible in old mode
        }

        return delta_c_map, delta_s_map, selected_delta_c, selected_delta_s, modulation_info
    else:
        # Use Hermitian solver
        return calculate_delta_values_hermitian(
            ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value,
            max_ind_stopband, n_param, skipped_indices,
            solution_tolerance=solution_tolerance, always_return_best=always_return_best,
            n_attempts=n_attempts, verbose=verbose
        )


def _calculate_delta_values_real(ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param, skipped_indices,
                                  solution_tolerance=1e-8, always_return_best=True, n_attempts=10, verbose=True):
    """
    Original real-symmetric delta value calculation (cosine-only modulation).
    Internal function - use calculate_delta_values as the main entry point.
    """
    # Run the multi-stage solution finding approach
    delta_solutions = multi_stage_solution_finding(
        ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param,
        bounds=(-0.5, 0.5), n_attempts=n_attempts, tolerance=solution_tolerance,
        always_return_best=always_return_best, verbose=verbose)

    # Print all solutions
    if verbose:
        print("\nAll solutions found:")
    if len(delta_solutions) > 0:
        for i, sol in enumerate(delta_solutions):
            solution_str = f"Solution {i+1}: "
            for j, val in enumerate(sol):
                # Map parameter index to the actual stopband index
                sb_idx = ind_param[j]
                solution_str += f"delta_{sb_idx} = {val:.6f}  "
            if verbose:
                print(solution_str)

        if verbose:
            print("Selecting solution with the lowest magnitude deltas...")

        # Calculate the total magnitude (sum of absolute values) for each solution
        magnitudes = [(sol, np.sum(np.abs(sol))) for sol in delta_solutions]

        # Sort by the total magnitude (ascending)
        sorted_solutions = sorted(magnitudes, key=lambda x: x[1])

        # Select the solution with the lowest total magnitude
        selected_delta = sorted_solutions[0][0]

        # Print information about the selected solution
        if verbose:
            total_magnitude = np.sum(np.abs(selected_delta))
            print(f"Selected solution has total magnitude: {total_magnitude:.6f}")
            solution_str = ""
            for j, val in enumerate(selected_delta):
                # Map parameter index to the actual stopband index
                sb_idx = ind_param[j]
                solution_str += f"delta_{sb_idx} = {val:.6f}  "
            print(solution_str)
    else:
        if verbose:
            print("No solutions found, even best-effort solutions. Using default small magnitude values.")
            print(f"Try adjusting the solution_tolerance (current: {solution_tolerance}) to a higher value.")
        # Create a default solution with small magnitude values
        selected_delta = np.ones(n_param) * 0.01  # Small positive values

    if verbose:
        print(f"\nFinal selected solution: {selected_delta}")

    # Create a mapping from stopband indices to delta values
    delta_map = {}
    for j, val in enumerate(selected_delta):
        sb_idx = ind_param[j]
        delta_map[sb_idx] = val

    if verbose:
        print("\nDelta values mapped to stopband indices:")
        for idx in sorted(delta_map.keys()):
            print(f"delta_{idx} = {delta_map[idx]:.6f}")

        print(f"\nNote: Stopbands with indices {', '.join(str(idx) for idx in skipped_indices)} were skipped (no free parameters).")
        print(f"Solution parameters:")
        print(f"  Tolerance: {solution_tolerance}")
        print(f"  Always return best: {always_return_best}")

    return delta_map, selected_delta


def calculate_delta_values_hermitian(ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param, skipped_indices,
                                      solution_tolerance=1e-8, always_return_best=True, n_attempts=20, verbose=True):
    """
    Calculate delta values using Hermitian coupling matrix (with phase freedom).

    This version finds both cosine (delta_c) and sine (delta_s) components for each harmonic,
    allowing phase optimization to minimize peak modulation amplitude.

    The modulation is: g_C(x) = 1 + 2*sum_n [delta_c[n]*cos(n*x) + delta_s[n]*sin(n*x)]

    Selection criterion: minimize peak positive deviation (max(g_C) - 1),
    rejecting solutions where g_C goes negative (infeasible for capacitance).

    Parameters:
    ind_stopband (array): Array of stopband indices
    ind_param (array): Array of stopband indices for parameters (non-default values)
    w_tilde_stopband_edges (array): Array of normalized frequencies at stopband edges
    is_default_value (array): Boolean array indicating which edges are default values
    max_ind_stopband (int): Maximum stopband index
    n_param (int): Number of harmonics
    skipped_indices (list): List of skipped stopband indices
    solution_tolerance (float): Tolerance for solution precision
    always_return_best (bool): If True, always return best solution found
    n_attempts (int): Number of random starting points
    verbose (bool): Whether to print progress

    Returns:
    tuple: (delta_c_map, delta_s_map, selected_delta_c, selected_delta_s, modulation_info)
           - delta_c_map: dict mapping stopband index to cosine coefficient
           - delta_s_map: dict mapping stopband index to sine coefficient
           - selected_delta_c: array of cosine coefficients
           - selected_delta_s: array of sine coefficients
           - modulation_info: dict with peak_deviation, g_min, g_max, mode
    """
    if verbose:
        print("\n=== Hermitian Modulation Solver ===")

    # Run the multi-stage solution finding with Hermitian matrix
    delta_solutions = multi_stage_solution_finding_hermitian(
        ind_stopband, ind_param, w_tilde_stopband_edges, is_default_value, max_ind_stopband, n_param,
        bounds=(-0.5, 0.5), n_attempts=n_attempts, tolerance=solution_tolerance,
        always_return_best=always_return_best, verbose=verbose)

    # Print all solutions
    if verbose:
        print(f"\nAll Hermitian solutions found:")
    if len(delta_solutions) > 0:
        for i, sol in enumerate(delta_solutions):
            delta_c = sol[:n_param]
            delta_s = sol[n_param:]

            # Compute amplitude and phase for each harmonic
            amplitudes = np.sqrt(delta_c**2 + delta_s**2)
            phases = np.arctan2(delta_s, delta_c) * 180 / np.pi  # in degrees

            if verbose:
                print(f"\nSolution {i+1}:")
                for j in range(n_param):
                    sb_idx = ind_param[j]
                    print(f"  Harmonic {sb_idx}: delta_c={delta_c[j]:.6f}, delta_s={delta_s[j]:.6f} "
                          f"(amplitude={amplitudes[j]:.6f}, phase={phases[j]:.1f}deg)")

            # Evaluate modulation for this solution
            peak_dev, g_min, g_max = evaluate_modulation(delta_c, delta_s, ind_param)
            if verbose:
                if peak_dev == np.inf:
                    print(f"  ** INFEASIBLE: g_C goes negative (min={g_min:.4f}) **")
                else:
                    print(f"  Peak deviation: {peak_dev:.6f}, g_min={g_min:.4f}, g_max={g_max:.4f}")

        # Select solution with smallest peak deviation (feasible solutions only)
        if verbose:
            print("\nSelecting solution with smallest peak modulation amplitude...")

        feasible_solutions = []
        for sol in delta_solutions:
            delta_c = sol[:n_param]
            delta_s = sol[n_param:]
            peak_dev, g_min, g_max = evaluate_modulation(delta_c, delta_s, ind_param)
            if peak_dev != np.inf:
                feasible_solutions.append((sol, peak_dev, g_min, g_max))

        if len(feasible_solutions) > 0:
            # Sort by peak deviation
            feasible_solutions.sort(key=lambda x: x[1])
            selected_sol, best_peak_dev, best_g_min, best_g_max = feasible_solutions[0]

            selected_delta_c = selected_sol[:n_param]
            selected_delta_s = selected_sol[n_param:]

            if verbose:
                print(f"\nSelected solution (peak deviation: {best_peak_dev:.6f}):")
                print(f"  g_min = {best_g_min:.4f}, g_max = {best_g_max:.4f}")
        else:
            if verbose:
                print("\nNo feasible solutions found (all have negative g_C)!")
                print("Falling back to solution with least negative g_min...")

            # Fall back to least-bad solution
            all_evaluated = []
            for sol in delta_solutions:
                delta_c = sol[:n_param]
                delta_s = sol[n_param:]
                _, g_min, g_max = evaluate_modulation(delta_c, delta_s, ind_param)
                all_evaluated.append((sol, g_min, g_max))

            # Sort by g_min (descending, so least negative first)
            all_evaluated.sort(key=lambda x: x[1], reverse=True)
            selected_sol, best_g_min, best_g_max = all_evaluated[0]

            selected_delta_c = selected_sol[:n_param]
            selected_delta_s = selected_sol[n_param:]
            best_peak_dev = best_g_max - 1

            if verbose:
                print(f"\nSelected (infeasible) solution:")
                print(f"  g_min = {best_g_min:.4f}, g_max = {best_g_max:.4f}")
                print(f"  WARNING: This solution has negative capacitance!")
    else:
        if verbose:
            print("No solutions found, even best-effort solutions. Using default small values.")
        selected_delta_c = np.ones(n_param) * 0.01
        selected_delta_s = np.zeros(n_param)
        best_peak_dev, best_g_min, best_g_max = evaluate_modulation(selected_delta_c, selected_delta_s, ind_param)

    # Compute final amplitudes and phases
    amplitudes = np.sqrt(selected_delta_c**2 + selected_delta_s**2)
    phases = np.arctan2(selected_delta_s, selected_delta_c) * 180 / np.pi

    if verbose:
        print(f"\nFinal selected solution:")
        for j in range(n_param):
            sb_idx = ind_param[j]
            print(f"  Harmonic {sb_idx}: delta_c={selected_delta_c[j]:.6f}, delta_s={selected_delta_s[j]:.6f}")
            print(f"              amplitude={amplitudes[j]:.6f}, phase={phases[j]:.1f}deg")

    # Create mappings
    delta_c_map = {}
    delta_s_map = {}
    for j in range(n_param):
        sb_idx = ind_param[j]
        delta_c_map[sb_idx] = selected_delta_c[j]
        delta_s_map[sb_idx] = selected_delta_s[j]

    if verbose:
        print("\nDelta values mapped to stopband indices:")
        for idx in sorted(delta_c_map.keys()):
            print(f"  delta_c_{idx} = {delta_c_map[idx]:.6f}, delta_s_{idx} = {delta_s_map[idx]:.6f}")

        print(f"\nNote: Stopbands with indices {', '.join(str(idx) for idx in skipped_indices)} were skipped.")

    # Modulation info for downstream use
    modulation_info = {
        'mode': 'hermitian',
        'peak_deviation': best_peak_dev,
        'g_min': best_g_min,
        'g_max': best_g_max,
        'amplitudes': amplitudes,
        'phases': phases,
        'ind_param': ind_param,
        'feasible': best_g_min > 0
    }

    return delta_c_map, delta_s_map, selected_delta_c, selected_delta_s, modulation_info
    

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