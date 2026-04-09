"""
Filter Builder Module for TWPA Design Package

This module provides tools for designing peripheral filter circuits (standalone
filters, diplexers, and N-way multiplexers) and integrating them into TWPA
netlists for simulation with JosephsonCircuits.jl.

The module contains three main sections:
  1. Low-pass prototype g-value computation (Butterworth, Chebyshev Type I)
  2. Frequency transformations, denormalization, and response analysis
  3. Design API, netlist generation, and topology composition

Example usage:
    from twpa_design.filter_builder import (
        FilterSpec, design_filter, design_multiplexer,
        filter_to_netlist, multiplexer_to_netlist, compose_chain
    )

    # Standalone lowpass filter
    spec = FilterSpec(response='lp', order=5, fc=8.5e9)
    design = design_filter(spec)
    netlist = filter_to_netlist(design)

    # LP/HP diplexer
    mux = design_multiplexer([
        FilterSpec('lp', 7, 8.5e9),
        FilterSpec('hp', 7, 8.5e9),
    ])
    mux_netlist = multiplexer_to_netlist(mux, prefix='m1')
"""

# ============================================================================
# Section 1: Low-pass prototype g-value computation
#   Copied from filter_coefficients.py
# ============================================================================

"""
Low-Pass Filter Prototype Coefficient Calculator

This module provides a class for calculating g-values (normalized element values)
for low-pass prototype filters: Butterworth and Chebyshev Type I.

Supports both doubly terminated (standard) and singly terminated (for diplexer design)
filter synthesis.

Example usage:
    # Butterworth filter (doubly terminated, default)
    filt = LowPassPrototypeFilter(order=5, filter_type='butterworth')
    g = filt.g_values()

    # Chebyshev Type I filter with 0.5 dB ripple
    filt = LowPassPrototypeFilter(order=5, filter_type='chebyshev1', ripple_dB=0.5)
    g = filt.g_values()

    # Singly terminated filter (for diplexer/multiplexer design)
    filt = LowPassPrototypeFilter(order=5, filter_type='butterworth', termination='single')
    g = filt.g_values()
"""

import numpy as np
from collections import defaultdict

# High precision arithmetic using mpmath
from mpmath import mp
mp.dps = 50  # 50 decimal places for high precision calculations


# =============================================================================
# MPPoly class and helper functions for high-precision polynomial arithmetic
# =============================================================================

class MPPoly:
    """Polynomial with mpmath precision coefficients.

    Coefficients are stored in ascending order: [c0, c1, c2, ...]
    representing c0 + c1*s + c2*s^2 + ...
    """

    def __init__(self, coeffs):
        # Store coefficients in ascending order as mpmath types
        if isinstance(coeffs, list):
            self.coeffs = [mp.mpc(c) if not isinstance(c, (mp.mpf, mp.mpc)) else c for c in coeffs]
        else:
            self.coeffs = [mp.mpc(coeffs)]

        # Remove trailing zeros
        while len(self.coeffs) > 1 and abs(self.coeffs[-1]) < mp.mpf(1e-40):
            self.coeffs.pop()

        # For compatibility with code that uses .coef (numpy style)
        self.coef = np.array([float(mp.re(c)) for c in self.coeffs])

    def degree(self):
        """Return degree of polynomial."""
        for i in range(len(self.coeffs) - 1, -1, -1):
            if abs(self.coeffs[i]) > mp.mpf(1e-40):
                return i
        return 0

    def roots(self):
        """Find roots of the polynomial using mpmath."""
        coeffs = self.coeffs.copy()
        while len(coeffs) > 1 and abs(coeffs[-1]) < mp.mpf(1e-30):
            coeffs.pop()

        if len(coeffs) <= 1:
            return np.array([])

        # Convert to descending order for polyroots
        coeffs_desc = coeffs[::-1]

        # Clean up very small coefficients
        max_coeff = max(abs(c) for c in coeffs_desc)
        coeffs_desc_cleaned = [c if abs(c) > max_coeff * mp.mpf(1e-20) else mp.mpf(0)
                               for c in coeffs_desc]

        try:
            with mp.extradps(20):
                roots_mp = mp.polyroots(coeffs_desc_cleaned, maxsteps=200, error=False)
        except mp.NoConvergence:
            # Fallback to numpy
            import numpy.polynomial.polynomial as npp
            float_coeffs = [float(mp.re(c)) for c in coeffs]
            return npp.polyroots(float_coeffs)

        return np.array([complex(float(r.real), float(r.imag)) for r in roots_mp])

    def __call__(self, x):
        """Evaluate polynomial at x."""
        result = mp.mpc(0)
        for i, c in enumerate(self.coeffs):
            result += c * (x ** i)
        return result

    def __add__(self, other):
        if isinstance(other, MPPoly):
            return MPPoly(_mp_poly_add(self.coeffs, other.coeffs))
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] += mp.mpc(other)
            return MPPoly(new_coeffs)

    def __sub__(self, other):
        if isinstance(other, MPPoly):
            return MPPoly(_mp_poly_sub(self.coeffs, other.coeffs))
        else:
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] -= mp.mpc(other)
            return MPPoly(new_coeffs)

    def __mul__(self, other):
        if isinstance(other, (int, float, mp.mpf, mp.mpc)):
            other_mp = mp.mpc(other) if not isinstance(other, (mp.mpf, mp.mpc)) else other
            return MPPoly([c * other_mp for c in self.coeffs])
        elif isinstance(other, MPPoly):
            result = [mp.mpc(0)] * (len(self.coeffs) + len(other.coeffs) - 1)
            for i, c1 in enumerate(self.coeffs):
                for j, c2 in enumerate(other.coeffs):
                    result[i + j] += c1 * c2
            return MPPoly(result)
        else:
            other_mp = mp.mpc(other) if not isinstance(other, (mp.mpf, mp.mpc)) else other
            return MPPoly([c * other_mp for c in self.coeffs])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float, mp.mpf, mp.mpc)):
            return MPPoly([c / other for c in self.coeffs])
        else:
            raise NotImplementedError("Polynomial division - use divmod or _mp_polydiv")

    def __divmod__(self, other):
        return _mp_polydiv(self, other)

    def __repr__(self):
        terms = []
        for i, c in enumerate(self.coeffs):
            c_val = float(mp.re(c))
            if abs(c_val) > 1e-10:
                if i == 0:
                    terms.append(f"{c_val:.6g}")
                elif i == 1:
                    terms.append(f"{c_val:.6g}*s")
                else:
                    terms.append(f"{c_val:.6g}*s^{i}")
        return " + ".join(terms) if terms else "0"


def _mp_poly_mul_root(coeffs, root):
    """Multiply polynomial by (s - root) in mpmath precision."""
    new_coeffs = [mp.mpc(0)] * (len(coeffs) + 1)
    for i in range(len(coeffs)):
        new_coeffs[i] -= coeffs[i] * root
        new_coeffs[i + 1] += coeffs[i]
    return new_coeffs


def _mp_poly_add(coeffs1, coeffs2):
    """Add two polynomials in mpmath precision."""
    max_len = max(len(coeffs1), len(coeffs2))
    result = [mp.mpc(0)] * max_len
    for i in range(len(coeffs1)):
        result[i] += coeffs1[i]
    for i in range(len(coeffs2)):
        result[i] += coeffs2[i]
    return result


def _mp_poly_sub(coeffs1, coeffs2):
    """Subtract two polynomials in mpmath precision."""
    max_len = max(len(coeffs1), len(coeffs2))
    result = [mp.mpc(0)] * max_len
    for i in range(len(coeffs1)):
        result[i] += coeffs1[i]
    for i in range(len(coeffs2)):
        result[i] -= coeffs2[i]
    return result


def _mp_polydiv(num, den):
    """Divide two polynomials in mpmath precision.

    Returns (quotient, remainder).
    """
    if isinstance(num, MPPoly):
        num = num.coeffs
    if isinstance(den, MPPoly):
        den = den.coeffs

    # Remove leading zeros
    num = num.copy()
    den = den.copy()
    while len(num) > 1 and abs(num[-1]) < mp.mpf(1e-40):
        num.pop()
    while len(den) > 1 and abs(den[-1]) < mp.mpf(1e-40):
        den.pop()

    if len(den) == 0 or all(abs(c) < mp.mpf(1e-40) for c in den):
        raise ValueError("Division by zero polynomial")

    remainder = [mp.mpc(c) for c in num]
    quotient = []

    deg_num = len(remainder) - 1
    deg_den = len(den) - 1

    if deg_num < deg_den:
        return MPPoly([mp.mpf(0)]), MPPoly(remainder)

    for i in range(deg_num - deg_den + 1):
        if len(remainder) < len(den):
            break

        coeff = remainder[-1] / den[-1]
        quotient.append(coeff)

        for j in range(len(den)):
            remainder[-(j + 1)] -= coeff * den[-(j + 1)]

        remainder.pop()

    quotient = quotient[::-1]

    if len(quotient) == 0:
        quotient = [mp.mpf(0)]
    if len(remainder) == 0:
        remainder = [mp.mpf(0)]

    return MPPoly(quotient), MPPoly(remainder)


def _clean_imag_parts(poly, tol=1e-40):
    """Remove tiny imaginary parts from polynomial coefficients."""
    cleaned_coeffs = []
    for c in poly.coeffs:
        if abs(mp.im(c)) < tol:
            cleaned_coeffs.append(mp.mpf(mp.re(c)))
        else:
            cleaned_coeffs.append(c)
    return MPPoly(cleaned_coeffs)


def _symmetrize_roots(roots, tol=1e-10):
    """Force exact complex conjugate pairing for roots.

    This ensures the resulting polynomial has real coefficients.
    """
    upper_half = []
    on_real = []

    for p in roots:
        if isinstance(p, mp.mpc):
            imag_val = float(p.imag)
        else:
            imag_val = p.imag

        if imag_val > tol:
            upper_half.append(p)
        elif imag_val >= -tol:
            on_real.append(p)

    symmetrized = []
    for p in upper_half:
        symmetrized.append(p)
        if isinstance(p, mp.mpc):
            symmetrized.append(mp.conj(p))
        else:
            symmetrized.append(np.conj(p))

    for p in on_real:
        symmetrized.append(p)

    return symmetrized


# =============================================================================
# Main Filter Class
# =============================================================================

class LowPassPrototypeFilter:
    """Low-pass prototype filter coefficient calculator.

    Calculates g-values (normalized element values) for ladder network synthesis
    of Butterworth and Chebyshev Type I filters.

    Parameters
    ----------
    order : int
        Filter order (number of reactive elements, must be >= 1).
    filter_type : str, optional
        Filter type: 'butterworth' or 'chebyshev1'. Default is 'butterworth'.
    ripple_dB : float, optional
        Passband ripple in dB. Required for 'chebyshev1', ignored for 'butterworth'.
    termination : str, optional
        Termination type: 'double' for doubly terminated (standard),
        'single' for singly terminated (diplexer design). Default is 'double'.
    even_order_mod : bool, optional
        Apply even-order modification for Chebyshev Type I. Default is False.
        Only relevant for even-order Chebyshev filters.
    verbose : bool, optional
        Print debug information during calculation. Default is False.

    Examples
    --------
    >>> # Butterworth filter (doubly terminated)
    >>> filt = LowPassPrototypeFilter(order=5, filter_type='butterworth')
    >>> g = filt.g_values()

    >>> # Chebyshev Type I filter
    >>> filt = LowPassPrototypeFilter(order=5, filter_type='chebyshev1', ripple_dB=0.5)
    >>> g = filt.g_values()

    >>> # Singly terminated filter (for diplexer design)
    >>> filt = LowPassPrototypeFilter(order=5, filter_type='butterworth', termination='single')
    >>> g = filt.g_values()
    """

    SUPPORTED_FILTER_TYPES = ('butterworth', 'chebyshev1')
    SUPPORTED_TERMINATIONS = ('double', 'single')

    def __init__(self, order, filter_type='butterworth', ripple_dB=None,
                 termination='double', even_order_mod=False, verbose=False):

        # Validate order
        if not isinstance(order, int) or order < 1:
            raise ValueError("Filter order must be an integer >= 1")
        self._order = order

        # Validate and store filter type
        filter_type = filter_type.lower()
        if filter_type not in self.SUPPORTED_FILTER_TYPES:
            raise ValueError(f"Unsupported filter type: '{filter_type}'. "
                           f"Supported types: {self.SUPPORTED_FILTER_TYPES}")
        self._filter_type = filter_type

        # Validate and store termination type
        termination = termination.lower()
        if termination not in self.SUPPORTED_TERMINATIONS:
            raise ValueError(f"Unsupported termination type: '{termination}'. "
                           f"Supported types: {self.SUPPORTED_TERMINATIONS}")
        self._termination = termination

        # Handle filter-specific parameters
        if filter_type == 'butterworth':
            self._epsilon = 1.0
            self._ripple_dB = None
            self._even_order_mod = False  # Not applicable for Butterworth

        elif filter_type == 'chebyshev1':
            if ripple_dB is None:
                raise ValueError("ripple_dB is required for Chebyshev Type I filter")
            if ripple_dB <= 0:
                raise ValueError("ripple_dB must be positive")
            self._ripple_dB = ripple_dB
            self._epsilon = np.sqrt(10 ** (ripple_dB / 10) - 1)
            self._even_order_mod = even_order_mod

        self._verbose = verbose

        # Cache for computed values
        self._g_values_cache = None
        self._gamma_zeros = None
        self._gamma_poles = None
        self._gamma2_poles = None  # Needed for singly terminated synthesis

    @property
    def order(self):
        """Filter order."""
        return self._order

    @property
    def filter_type(self):
        """Filter type ('butterworth' or 'chebyshev1')."""
        return self._filter_type

    @property
    def epsilon(self):
        """Ripple factor."""
        return self._epsilon

    @property
    def ripple_dB(self):
        """Passband ripple in dB (None for Butterworth)."""
        return self._ripple_dB

    @property
    def even_order_mod(self):
        """Whether even-order modification is applied."""
        return self._even_order_mod

    @property
    def termination(self):
        """Termination type ('double' or 'single')."""
        return self._termination

    def __repr__(self):
        if self._filter_type == 'butterworth':
            return (f"LowPassPrototypeFilter(order={self._order}, filter_type='butterworth', "
                   f"termination='{self._termination}')")
        else:
            return (f"LowPassPrototypeFilter(order={self._order}, filter_type='chebyshev1', "
                   f"ripple_dB={self._ripple_dB}, termination='{self._termination}', "
                   f"even_order_mod={self._even_order_mod})")

    # =========================================================================
    # Private methods for the synthesis pipeline
    # =========================================================================

    def _find_gamma2_zeros_poles(self):
        """Find zeros and poles of |Gamma|^2 using analytical formulas.

        Returns
        -------
        tuple
            (gamma2_zeros, gamma2_poles) - lists of complex zeros and poles.
        """
        n = self._order
        epsilon = self._epsilon

        if self._verbose:
            print(f"Finding zeros and poles of |Γ|² for {self._filter_type}, order {n}")

        if self._filter_type == 'butterworth':
            # Butterworth: All 2n zeros at origin
            gamma2_zeros = [0] * (2 * n)

            # Poles on unit circle
            gamma2_poles = []
            for k in range(2 * n):
                angle = mp.mpf(2 * k + 1) * mp.pi / mp.mpf(2 * n) + mp.pi / 2
                radius = mp.power(mp.mpf(1) / mp.mpf(epsilon),
                                 mp.mpf(1) / (mp.mpf(2) * mp.mpf(n)))
                pole_real = radius * mp.cos(angle)
                pole_imag = radius * mp.sin(angle)
                pole = mp.mpc(pole_real, pole_imag)
                gamma2_poles.append(pole)

        elif self._filter_type == 'chebyshev1':
            # Chebyshev Type I: Zeros on imaginary axis
            gamma2_zeros = []
            for k in range(n):
                zero = 1j * np.cos((2 * k + 1) * np.pi / (2 * n))
                gamma2_zeros.extend([zero, -zero])

            # Poles on ellipse
            gamma2_poles = []
            beta = np.arcsinh(1 / epsilon) / n
            for k in range(2 * n):
                alpha = (2 * k + 1) * np.pi / (2 * n)
                sinh_val = np.sinh(beta)
                cosh_val = np.cosh(beta)
                pole = np.sin(alpha) * sinh_val + 1j * np.cos(alpha) * cosh_val
                gamma2_poles.append(pole)

        # Symmetrize to ensure exact complex conjugate pairs
        gamma2_zeros = _symmetrize_roots(gamma2_zeros)
        gamma2_poles = _symmetrize_roots(gamma2_poles)

        if self._verbose:
            print(f"  Found {len(gamma2_zeros)} zeros and {len(gamma2_poles)} poles of |Γ|²")

        return gamma2_zeros, gamma2_poles

    def _apply_even_order_modification(self, gamma2_zeros, gamma2_poles):
        """Apply even-order modification for Chebyshev Type I filters.

        This modification adjusts the zeros for even-order Chebyshev filters
        to achieve specific response characteristics.

        Parameters
        ----------
        gamma2_zeros : list
            Zeros of |Gamma|^2.
        gamma2_poles : list
            Poles of |Gamma|^2.

        Returns
        -------
        tuple
            (modified_zeros, modified_poles)
        """
        n = self._order

        # Only apply for even order Chebyshev with modification enabled
        if not (n % 2 == 0 and n > 2 and self._even_order_mod and
                self._filter_type == 'chebyshev1'):
            return gamma2_zeros, gamma2_poles

        if self._verbose:
            print(f"Applying even-order modification for n={n}")

        cos_val = np.cos(np.pi * (n + 1) / (2 * n))

        # Remap zeros
        new_zeros = []
        for z in gamma2_zeros:
            if abs(z.imag) > 1e-10:
                new_z_imag = np.sign(z.imag) * np.sqrt((z.imag**2 - cos_val**2) / (1 - cos_val**2))
                new_z = 1j * new_z_imag
                new_zeros.append(new_z)
            else:
                new_zeros.append(z)

        # Symmetrize the modified zeros
        new_zeros = _symmetrize_roots(new_zeros)

        if self._verbose:
            print(f"  Modified {len(new_zeros)} zeros")

        return new_zeros, gamma2_poles

    def _convert_gamma2_to_gamma(self, gamma2_zeros, gamma2_poles):
        """Convert from |Gamma|^2 to Gamma.

        Selects left-half plane poles and halves zero multiplicities.

        Parameters
        ----------
        gamma2_zeros : list
            Zeros of |Gamma|^2.
        gamma2_poles : list
            Poles of |Gamma|^2.

        Returns
        -------
        tuple
            (gamma_zeros, gamma_poles) for Gamma.
        """
        if self._verbose:
            print("Converting from |Γ|² to Γ...")

        # Select left-half plane poles (negative real part)
        gamma_poles = [p for p in gamma2_poles if p.real < 0]

        # Group zeros and take half from each group
        tolerance = 1e-8

        def round_to_tolerance(value, tol):
            return round(value / tol) * tol

        zero_groups = defaultdict(list)
        for z in gamma2_zeros:
            abs_val = abs(z)
            if abs_val < tolerance:
                key = 0.0
            else:
                abs_imag = abs(z.imag)
                key = round_to_tolerance(abs_imag, tolerance)
            zero_groups[key].append(z)

        gamma_zeros = []
        for abs_imag, zeros in sorted(zero_groups.items()):
            half_count = len(zeros) // 2
            zeros_sorted = sorted(zeros, key=lambda z: abs(z))
            selected = zeros_sorted[:half_count]
            gamma_zeros.extend(selected)

        if self._verbose:
            print(f"  Γ has {len(gamma_zeros)} zeros and {len(gamma_poles)} poles")

        return gamma_zeros, gamma_poles

    def _build_input_impedance(self, gamma_zeros, gamma_poles):
        """Build input impedance Z(s) from zeros and poles of Gamma.

        Computes Z(s) = (gamma_d + gamma_n) / (gamma_d - gamma_n)
        where gamma_n is built from zeros and gamma_d from poles.

        Parameters
        ----------
        gamma_zeros : list
            Zeros of Gamma.
        gamma_poles : list
            Poles of Gamma.

        Returns
        -------
        tuple
            (num_poly, den_poly) - MPPoly polynomials for Z(s) = num(s)/den(s).
        """
        if self._verbose:
            print("Building input impedance Z(s)...")

        finite_zeros = [z for z in gamma_zeros if (isinstance(z, mp.mpc) or np.isfinite(z))]
        finite_poles = [p for p in gamma_poles if (isinstance(p, mp.mpc) or np.isfinite(p))]

        # Build gamma_n from zeros
        if len(finite_zeros) > 0:
            mp_zeros = [mp.mpc(z) if not isinstance(z, mp.mpc) else z for z in finite_zeros]
            gamma_n_coeffs = [mp.mpf(1)]
            for z in mp_zeros:
                gamma_n_coeffs = _mp_poly_mul_root(gamma_n_coeffs, z)
            gamma_n = MPPoly(gamma_n_coeffs)
        else:
            gamma_n = MPPoly([1])

        # Build gamma_d from poles
        if len(finite_poles) == 0:
            gamma_d = MPPoly([1])
        else:
            mp_poles = [mp.mpc(p) if not isinstance(p, mp.mpc) else p for p in finite_poles]
            gamma_d_coeffs = [mp.mpf(1)]
            for p in mp_poles:
                gamma_d_coeffs = _mp_poly_mul_root(gamma_d_coeffs, p)
            gamma_d = MPPoly(gamma_d_coeffs)

        if self._verbose:
            print(f"  gamma_n degree: {gamma_n.degree()}, gamma_d degree: {gamma_d.degree()}")

        # Compute Z = (gamma_d + gamma_n) / (gamma_d - gamma_n)
        num_poly = gamma_d + gamma_n
        den_poly = gamma_d - gamma_n

        # Normalize by lowest coefficient of numerator
        normalization_factor = float(mp.re(num_poly.coeffs[0]))
        if abs(normalization_factor) > 1e-40:
            num_poly = num_poly / normalization_factor
            den_poly = den_poly / normalization_factor

        # Clean tiny imaginary parts
        num_poly = _clean_imag_parts(num_poly)
        den_poly = _clean_imag_parts(den_poly)

        if self._verbose:
            print(f"  Z(s) numerator degree: {num_poly.degree()}")
            print(f"  Z(s) denominator degree: {den_poly.degree()}")

        return num_poly, den_poly

    def _extract_z11(self, num_poly, den_poly):
        """Extract z11 polynomials based on network topology.

        For even order: antimetrical network (topology A)
        For odd order: symmetrical network (topology B)

        Parameters
        ----------
        num_poly : MPPoly
            Numerator of Z(s).
        den_poly : MPPoly
            Denominator of Z(s).

        Returns
        -------
        tuple
            (z11_num, z11_den) - polynomials for z11.
        """
        n = self._order

        # Split numerator into even and odd parts
        coeffs = num_poly.coeffs
        even_coeffs = [mp.mpc(0)] * len(coeffs)
        odd_coeffs = [mp.mpc(0)] * len(coeffs)

        for i in range(len(coeffs)):
            if i % 2 == 0:
                even_coeffs[i] = coeffs[i]
            else:
                odd_coeffs[i] = coeffs[i]

        even_num_poly = MPPoly(even_coeffs)
        odd_num_poly = MPPoly(odd_coeffs)

        # Split denominator into even and odd parts
        coeffs = den_poly.coeffs
        even_coeffs = [mp.mpc(0)] * len(coeffs)
        odd_coeffs = [mp.mpc(0)] * len(coeffs)

        for i in range(len(coeffs)):
            if i % 2 == 0:
                even_coeffs[i] = coeffs[i]
            else:
                odd_coeffs[i] = coeffs[i]

        even_den_poly = MPPoly(even_coeffs)
        odd_den_poly = MPPoly(odd_coeffs)

        # Select topology based on order
        if n % 2 == 0:
            # Antimetrical network (even order) - topology A
            z11_num = even_num_poly
            z11_den = odd_den_poly
            if self._verbose:
                print(f"  Network topology: A (antimetrical, n={n} is even)")
        else:
            # Symmetrical network (odd order) - topology B
            z11_num = odd_num_poly
            z11_den = even_den_poly
            if self._verbose:
                print(f"  Network topology: B (symmetrical, n={n} is odd)")

        return z11_num, z11_den

    def _build_input_impedance_single_termination(self, gamma_poles, gamma2_poles):
        """Build input impedance Z₁(s) for singly terminated filter.

        For singly terminated filters, |H|² = |Z₁₂|² = Re(Y₁).
        We perform partial fraction expansion to get Y₁(s) from Re(Y₁),
        then return Z₁ = 1/Y₁.

        Parameters
        ----------
        gamma_poles : list
            Left half-plane poles of Gamma (same as poles of H).
        gamma2_poles : list
            All poles of |Gamma|² (both LHP and RHP).

        Returns
        -------
        tuple
            (num_poly, den_poly) - MPPoly polynomials for Z₁(s) = num(s)/den(s).
        """
        if self._verbose:
            print("Building input impedance Z₁(s) for singly terminated filter...")

        # Get finite LHP poles (these become denominator of Y₁)
        finite_poles = [p for p in gamma_poles if (isinstance(p, mp.mpc) or np.isfinite(p))]

        # Build Y₁ denominator from LHP poles: D_Y1(s) = prod(s - p_i)
        mp_poles = [mp.mpc(p) if not isinstance(p, mp.mpc) else p for p in finite_poles]
        Y1_den_coeffs = [mp.mpf(1)]
        for p in mp_poles:
            Y1_den_coeffs = _mp_poly_mul_root(Y1_den_coeffs, p)
        Y1_den = MPPoly(Y1_den_coeffs)

        if self._verbose:
            print(f"  Y₁ denominator built from {len(finite_poles)} LHP poles")

        # Build D(s) from ALL gamma2_poles (denominator of |H|²)
        all_poles = [mp.mpc(p) if not isinstance(p, mp.mpc) else p for p in gamma2_poles]
        D_coeffs = [mp.mpf(1)]
        for p in all_poles:
            D_coeffs = _mp_poly_mul_root(D_coeffs, p)
        D_poly = MPPoly(D_coeffs)

        # Normalize so D(0) = 1 (ensures correct sign for residues)
        D_at_0 = D_poly(mp.mpf(0))
        if abs(D_at_0) > mp.mpf(1e-40):
            D_coeffs = [c / D_at_0 for c in D_coeffs]
            D_poly = MPPoly(D_coeffs)

        if self._verbose:
            print(f"  D(0) normalization factor: {float(mp.re(D_at_0)):.6g}")

        # Derivative of D(s): D'(s)
        D_deriv_coeffs = [mp.mpf(i+1) * D_coeffs[i+1] for i in range(len(D_coeffs)-1)]
        D_deriv = MPPoly(D_deriv_coeffs)

        # Compute residues: k_i = 1/D'(p_i) for simple poles
        # Then r_i = 2*k_i (since Re(Y₁) = ½[Y₁(s) + Y₁(-s)])
        # Y₁(s) = Σ r_i/(s - p_i) = N(s)/D_Y1(s)
        # N(s) = Σ r_i * prod_{j≠i}(s - p_j)
        Y1_num_coeffs = [mp.mpf(0)] * len(finite_poles)

        for i, p_i in enumerate(finite_poles):
            # Residue calculation
            D_deriv_at_pi = D_deriv(mp_poles[i])
            k_i = mp.mpf(1) / D_deriv_at_pi
            r_i = 2 * k_i  # Factor of 2 from Re(Y₁) = ½[Y₁ + Y₁*]

            # Build product term: prod_{j≠i}(s - p_j)
            term_coeffs = [mp.mpf(1)]
            for j, p_j in enumerate(mp_poles):
                if j != i:
                    term_coeffs = _mp_poly_mul_root(term_coeffs, p_j)

            # Add r_i * term to numerator
            for k in range(len(term_coeffs)):
                if k < len(Y1_num_coeffs):
                    Y1_num_coeffs[k] += r_i * term_coeffs[k]
                else:
                    Y1_num_coeffs.append(r_i * term_coeffs[k])

        Y1_num = MPPoly(Y1_num_coeffs)

        # Clean tiny imaginary parts
        Y1_num = _clean_imag_parts(Y1_num)
        Y1_den = _clean_imag_parts(Y1_den)

        if self._verbose:
            print(f"  Y₁ numerator degree: {Y1_num.degree()}")
            print(f"  Y₁ denominator degree: {Y1_den.degree()}")

        # Return Z₁ = 1/Y₁ (swap num and den for continued fraction expansion)
        return Y1_den, Y1_num

    def _continued_fraction_expansion(self, z11_num, z11_den):
        """Extract g-values using continued fraction (Cauer) expansion.

        Parameters
        ----------
        z11_num : MPPoly
            Numerator of z11.
        z11_den : MPPoly
            Denominator of z11.

        Returns
        -------
        list
            g-values from the lossless network (excluding g0 and g_load).
        """
        if self._verbose:
            print("Performing continued fraction expansion...")

        g = []
        num = z11_num
        den = z11_den
        rem = z11_num

        max_iterations = 100
        iteration = 0

        while rem.degree() > 0 and iteration < max_iterations:
            iteration += 1

            quo, rem = _mp_polydiv(num, den)

            # Extract the coefficient of s (degree 1 term)
            if len(quo.coef) > 1:
                g.append(float(quo.coef[1]))

            if rem.degree() == 0:
                # Last element
                if len(den.coef) > 1:
                    g.append(float(den.coef[1]))

            num = den
            den = rem

        if self._verbose:
            print(f"  Extracted {len(g)} elements from continued fraction")

        return g

    # =========================================================================
    # Public methods
    # =========================================================================

    def g_values(self, plot=False):
        """Calculate and return the g-values for this filter.

        For both termination types, returns n+2 elements:
        - g[0]: Source resistance (normalized to 1.0)
        - g[1] to g[n]: Reactive element values (inductors/capacitors)
        - g[n+1]: Load resistance/conductance

        For doubly terminated filters:
        - g[0] = 1.0 (or calculated value for some Chebyshev cases)
        - g[n+1] = finite load resistance

        For singly terminated filters:
        - g[0] = 1.0 (source resistance)
        - g[n+1] = inf (open circuit load)

        Parameters
        ----------
        plot : bool, optional
            If True, plot the poles and zeros of Gamma. Default is False.

        Returns
        -------
        list
            g-values [g0, g1, g2, ..., gn, g_{n+1}] with n+2 elements.
            For single termination, g[n+1] = float('inf').
        """
        # Return cached values if available
        if self._g_values_cache is not None:
            if plot:
                self.plot_poles_zeros()
            return self._g_values_cache.copy()

        if self._verbose:
            print(f"\n{'='*60}")
            print(f"Calculating g-values for {self}")
            print(f"{'='*60}")

        # Step 1: Find zeros and poles of |Gamma|^2
        gamma2_zeros, gamma2_poles = self._find_gamma2_zeros_poles()

        # Step 2: Apply even-order modification if needed (only for doubly terminated)
        if self._termination == 'double':
            gamma2_zeros, gamma2_poles = self._apply_even_order_modification(
                gamma2_zeros, gamma2_poles)

        # Step 3: Convert from |Gamma|^2 to Gamma
        gamma_zeros, gamma_poles = self._convert_gamma2_to_gamma(
            gamma2_zeros, gamma2_poles)

        # Store for plotting
        self._gamma_zeros = gamma_zeros
        self._gamma_poles = gamma_poles
        self._gamma2_poles = gamma2_poles  # Needed for single termination

        # Branch based on termination type
        if self._termination == 'single':
            # Singly terminated: use partial fraction expansion approach
            # Build Z₁ = 1/Y₁ directly and apply continued fraction
            num_poly, den_poly = self._build_input_impedance_single_termination(
                gamma_poles, gamma2_poles)

            # Build g-values list with g0 = 1 (source) and g_{n+1} = inf (open load)
            g_values = [1.0]  # g0 = 1 (source resistance)

            # Apply continued fraction directly to Z₁ (no z11 extraction)
            g_reactive = self._continued_fraction_expansion(num_poly, den_poly)
            g_reactive.reverse()  # Reverse order to match standard table convention (source to load)
            g_values.extend(g_reactive)

            # g_{n+1} = infinity (open circuit load)
            g_values.append(float('inf'))

            if self._verbose:
                print(f"\nResult (singly terminated): {len(g_values)} g-values")
                for i, g in enumerate(g_values):
                    if g == float('inf'):
                        print(f"  g[{i}] = inf")
                    else:
                        print(f"  g[{i}] = {g:.6f}")

        else:
            # Doubly terminated: standard approach
            # Step 4: Build input impedance Z(s)
            num_poly, den_poly = self._build_input_impedance(gamma_zeros, gamma_poles)

            # Step 5: Extract z11
            z11_num, z11_den = self._extract_z11(num_poly, den_poly)

            # Step 6: Calculate g-values
            g_values = []

            # Source impedance (g0)
            if num_poly.degree() == den_poly.degree():
                g0 = num_poly.coef[-1].real / den_poly.coef[-1].real
            else:
                g0 = 1.0
            g_values.append(g0)

            # Lossless network elements via continued fraction
            g_z11 = self._continued_fraction_expansion(z11_num, z11_den)
            g_values.extend(g_z11)

            # Load impedance (g_{n+1})
            g_load = num_poly.coef[0].real / den_poly.coef[0].real
            g_values.append(g_load)

            if self._verbose:
                print(f"\nResult (doubly terminated): {len(g_values)} g-values")
                for i, g in enumerate(g_values):
                    print(f"  g[{i}] = {g:.6f}")

        # Cache and return
        self._g_values_cache = g_values

        if plot:
            self.plot_poles_zeros()

        return g_values.copy()

    def plot_poles_zeros(self):
        """Plot the poles and zeros of Gamma in the complex plane.

        Requires matplotlib to be installed.
        Uses plotting style from plots_params if available.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return

        # Try to import plotting parameters, use defaults if not available
        try:
            from plots_params import (
                fontsize_title, fontsize_legend,
                blue, orange, black
            )
            # Note: configure_matplotlib() should be called by the user script
            # to set USE_LATEX before plotting
        except ImportError:
            # Fallback defaults if plots_params not available
            fontsize_title = 8
            fontsize_legend = 8
            blue = np.array([0.22, 0.49, 0.72])  # #387EB9
            orange = np.array([0.96, 0.49, 0.13])  # #F57E20
            black = np.array([0.0, 0.0, 0.0])

        # Ensure we have computed the poles and zeros
        if self._gamma_zeros is None or self._gamma_poles is None:
            self.g_values(plot=False)

        fig, ax = plt.subplots(figsize=(8.6/2.54, 8.6/2.54))

        # Plot zeros
        if self._gamma_zeros:
            zeros_for_plot = [complex(float(z.real), float(z.imag))
                             if isinstance(z, mp.mpc) else z for z in self._gamma_zeros]
            zeros_array = np.array(zeros_for_plot)
            ax.scatter(zeros_array.real, zeros_array.imag, s=30, c=[blue],
                      marker='o', label=f'Zeros ({len(zeros_array)})', zorder=3)

        # Plot poles
        if self._gamma_poles:
            poles_for_plot = [complex(float(p.real), float(p.imag))
                             if isinstance(p, mp.mpc) else p for p in self._gamma_poles]
            poles_array = np.array(poles_for_plot)
            ax.scatter(poles_array.real, poles_array.imag, s=30, c=[orange],
                      marker='x', label=f'Poles ({len(poles_array)})', zorder=3)

        # Formatting
        ax.axhline(y=0, color=black, linewidth=0.5)
        ax.axvline(x=0, color=black, linewidth=0.5)
        ax.set_xlabel(r'$\sigma$')
        ax.set_ylabel(r'$j\omega$')
        ax.set_title(f'Poles and Zeros of $\\Gamma(s)$', fontsize=fontsize_title)
        ax.legend(loc='upper right', fontsize=fontsize_legend)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Set axis limits
        all_points = list(self._gamma_poles or []) + list(self._gamma_zeros or [])
        if all_points:
            max_real = max(abs(float(p.real) if isinstance(p, mp.mpc) else p.real)
                          for p in all_points if p != 0)
            max_imag = max(abs(float(p.imag) if isinstance(p, mp.mpc) else p.imag)
                          for p in all_points if p != 0)
            max_val = max(max_real, max_imag) * 1.2
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)

        plt.tight_layout()
        plt.show()

        return fig, ax


# =============================================================================
# Convenience functions
# =============================================================================

def calculate_g_values(filter_type, order, ripple_dB=None, termination='double',
                       even_order_mod=False):
    """Convenience function to calculate g-values for a filter.

    Parameters
    ----------
    filter_type : str
        Either 'butterworth' or 'chebyshev1'.
    order : int
        Filter order.
    ripple_dB : float, optional
        Passband ripple in dB (required for Chebyshev).
    termination : str, optional
        'double' for doubly terminated (default), 'single' for singly terminated.
    even_order_mod : bool, optional
        Apply even-order modification for Chebyshev. Default is False.

    Returns
    -------
    list
        g-values [g0, g1, ..., g_{n+1}] with n+2 elements.
        For single termination, g[0] = 1.0 and g[n+1] = inf.

    Examples
    --------
    >>> g = calculate_g_values('butterworth', 3)
    >>> g = calculate_g_values('chebyshev1', 5, ripple_dB=0.5)
    >>> g = calculate_g_values('butterworth', 3, termination='single')
    """
    filt = LowPassPrototypeFilter(
        order=order,
        filter_type=filter_type,
        ripple_dB=ripple_dB,
        termination=termination,
        even_order_mod=even_order_mod
    )
    return filt.g_values()


def calculate_g_values_range(filter_type, max_order, ripple_dB=None, termination='double',
                             even_order_mod=False):
    """Calculate g-values for filter orders 1 through max_order.

    Parameters
    ----------
    filter_type : str
        Either 'butterworth' or 'chebyshev1'.
    max_order : int
        Maximum filter order to calculate.
    ripple_dB : float, optional
        Passband ripple in dB (required for Chebyshev).
    termination : str, optional
        'double' for doubly terminated (default), 'single' for singly terminated.
    even_order_mod : bool, optional
        Apply even-order modification for Chebyshev. Default is False.

    Returns
    -------
    dict
        Dictionary mapping order to g-values list.

    Examples
    --------
    >>> all_g = calculate_g_values_range('butterworth', 10)
    >>> g_5th_order = all_g[5]
    >>> all_g_single = calculate_g_values_range('butterworth', 10, termination='single')
    """
    import warnings

    results = {}
    for order in range(1, max_order + 1):
        try:
            results[order] = calculate_g_values(
                filter_type, order, ripple_dB, termination, even_order_mod)
        except Exception as e:
            warnings.warn(f"Failed to calculate g-values for order {order}: {e}")
            results[order] = None
    return results


# =============================================================================
# Main - example usage and testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Low-Pass Prototype Filter Coefficients")
    print("=" * 60)

    # Butterworth examples
    print("\nButterworth Filters:")
    print("-" * 40)
    for n in range(1, 6):
        filt = LowPassPrototypeFilter(order=n, filter_type='butterworth')
        g = filt.g_values()
        g_str = ", ".join(f"{v:.4f}" for v in g)
        print(f"  Order {n}: [{g_str}]")

    # Chebyshev Type I examples
    print("\nChebyshev Type I Filters (0.5 dB ripple):")
    print("-" * 40)
    for n in range(1, 6):
        filt = LowPassPrototypeFilter(order=n, filter_type='chebyshev1', ripple_dB=0.5)
        g = filt.g_values()
        g_str = ", ".join(f"{v:.4f}" for v in g)
        print(f"  Order {n}: [{g_str}]")

    # Helper function to format g-values (handles inf)
    def format_g(val):
        if val == float('inf'):
            return "inf"
        return f"{val:.4f}"

    # Singly terminated examples
    print("\nSingly Terminated Butterworth Filters:")
    print("-" * 40)
    for n in range(1, 6):
        filt = LowPassPrototypeFilter(order=n, filter_type='butterworth', termination='single')
        g = filt.g_values()
        g_str = ", ".join(format_g(v) for v in g)
        print(f"  Order {n}: [{g_str}]")

    print("\nSingly Terminated Chebyshev Type I Filters (0.5 dB ripple):")
    print("-" * 40)
    for n in range(1, 6):
        filt = LowPassPrototypeFilter(order=n, filter_type='chebyshev1', ripple_dB=0.5, termination='single')
        g = filt.g_values()
        g_str = ", ".join(format_g(v) for v in g)
        print(f"  Order {n}: [{g_str}]")

    # Using convenience function
    print("\nUsing convenience function:")
    print("-" * 40)
    g = calculate_g_values('butterworth', 3)
    print(f"  Butterworth order 3 (double): [{', '.join(format_g(v) for v in g)}]")

    g = calculate_g_values('butterworth', 3, termination='single')
    print(f"  Butterworth order 3 (single): [{', '.join(format_g(v) for v in g)}]")

    g = calculate_g_values('chebyshev1', 3, ripple_dB=0.1)
    print(f"  Chebyshev1 order 3 (0.1 dB, double): [{', '.join(format_g(v) for v in g)}]")

    g = calculate_g_values('chebyshev1', 3, ripple_dB=0.1, termination='single')
    print(f"  Chebyshev1 order 3 (0.1 dB, single): [{', '.join(format_g(v) for v in g)}]")


# ============================================================================
# Section 2: Frequency transformations and filter analysis
#   Copied from filter_transformations.py (imports re-wired)
# ============================================================================

"""
Filter transformations module.

Thin wrapper around twpa_design.helper_functions for filter transformations.
Transforms a complete LP prototype filter (g-values) into component networks.

Main functions:
    frequency_transfo_normalized(g_values, w_zeros, w_poles, foster_form=1, zero_at_zero=None)
    frequency_transfo(g_values, w_zeros, w_poles, Z0_ohm, fc_Hz, foster_form=1, zero_at_zero=None)
    denormalize(L_norm, C_norm, Z0, fc)
    normalize(L_H, C_F, Z0, fc)
    calculate_S21(result, f, Z0=None, units='GHz')
    calculate_ABCD(result, f, units='GHz')
    calculate_input_impedance(result, f, units='GHz')
    calculate_multiplexer_response(filter_results, f, Z0=None, units='GHz')
    plot_multiplexer_response(filter_results, f=None, Z0=None, units='GHz', ...)
"""

import numpy as np

# Import from package helper_functions
from .helper_functions import (
    filter_transfo_Foster1,
    filter_transfo_Foster2,
    should_have_zero_at_zero,
)

# Normalized frequency: wc = 1 rad/s corresponds to fc = 1/(2*pi) Hz = 1/(2*pi*1e9) GHz
_FC_NORMALIZED_GHZ = 1 / (2 * np.pi * 1e9)


def frequency_transfo_normalized(g_values, w_zeros, w_poles,
                                  foster_form=1, zero_at_zero=None, verbose=False):
    """
    Transform a complete LP prototype filter to component networks (normalized).

    Takes the full g-values list and returns the Foster expansion for each
    reactive element, organized by branch type.

    Parameters
    ----------
    g_values : list
        LP prototype g-values [g0, g1, g2, ..., gn, g_{n+1}].
    w_zeros : array_like
        Normalized frequencies of transmission zeros.
    w_poles : array_like
        Normalized frequencies of transmission poles.
    foster_form : int, optional
        Foster form to use (1 or 2). Default is 1.
    zero_at_zero : bool, optional
        Whether frequency transform has zero at w=0. If None, determined automatically.
    verbose : bool, optional
        Print component values. Default is False.

    Returns
    -------
    dict
        Complete filter transformation with keys:
        - 'series': list of dicts for series (inductor) branches (odd indices: g1, g3, ...)
        - 'shunt': list of dicts for shunt (capacitor) branches (even indices: g2, g4, ...)
        - 'g0': source impedance
        - 'g_load': load impedance
        - 'foster_form': which Foster form was used
        - 'zero_at_zero': whether there's a zero at w=0
        - 'w_zeros': the transmission zeros used
        - 'w_poles': the transmission poles used

        Each branch dict contains:
        - Foster 1: {'g_k', 'Linf', 'C0', 'Li', 'Ci', 'w_res'}
        - Foster 2: {'g_k', 'L0', 'Cinf', 'Li', 'Ci', 'w_res'}

        All component values are normalized (Z0=1 Ohm, wc=1 rad/s).
    """
    w_zeros = np.atleast_1d(w_zeros)
    w_poles = np.atleast_1d(w_poles)

    # Determine zero_at_zero: pass user preference to should_have_zero_at_zero
    # When both w_zeros and w_poles are empty, the user's choice is respected
    # When finite poles/zeros exist, the value is auto-determined by the alternation rule
    default_zero_at_zero = True if zero_at_zero is None else zero_at_zero
    zero_at_zero = should_have_zero_at_zero(w_zeros, w_poles, default_zero_at_zero)

    n = len(g_values) - 2  # filter order (excluding g0 and g_{n+1})

    series_branches = []  # odd indices (g1, g3, g5, ...) - series inductors
    shunt_branches = []   # even indices (g2, g4, g6, ...) - shunt capacitors

    for k in range(1, n + 1):
        g_k = np.float64(g_values[k])  # Convert to numpy float for proper inf handling

        if k % 2 == 1:
            # Odd index: series inductor branch
            ind_or_cap = 'ind'
            branch_list = series_branches
        else:
            # Even index: shunt capacitor branch
            ind_or_cap = 'cap'
            branch_list = shunt_branches

        # Call twpa_design functions with normalized parameters (Z0=1, wc=1)
        # Note: filter_transfo_Foster1 signature is (g_k, ind_or_cap, zero_at_zero, w_poles, w_zeros, ...)
        if foster_form == 1:
            Linf, C0, Li, Ci = filter_transfo_Foster1(
                g_k, ind_or_cap, zero_at_zero, w_poles, w_zeros,
                Z0_ohm=1.0, fc_filter_GHz=_FC_NORMALIZED_GHZ, verbose=verbose
            )
            w_res = w_poles if ind_or_cap == 'ind' else w_zeros
            branch = {
                'g_k': g_k, 'index': k,
                'Linf': Linf, 'C0': C0, 'Li': Li, 'Ci': Ci,
                'w_res': w_res
            }
        else:
            L0, Cinf, Li, Ci = filter_transfo_Foster2(
                g_k, ind_or_cap, zero_at_zero, w_poles, w_zeros,
                Z0_ohm=1.0, fc_filter_GHz=_FC_NORMALIZED_GHZ, verbose=verbose
            )
            w_res = w_zeros if ind_or_cap == 'ind' else w_poles
            branch = {
                'g_k': g_k, 'index': k,
                'L0': L0, 'Cinf': Cinf, 'Li': Li, 'Ci': Ci,
                'w_res': w_res
            }

        branch_list.append(branch)

    return {
        'series': series_branches,
        'shunt': shunt_branches,
        'g0': g_values[0],
        'g_load': g_values[-1],
        'foster_form': foster_form,
        'zero_at_zero': zero_at_zero,
        'w_zeros': w_zeros,
        'w_poles': w_poles,
        'order': n
    }


def frequency_transfo(g_values, f_zeros, f_poles, Z0_ohm, fc,
                      foster_form=1, zero_at_zero=None, units='GHz', verbose=False):
    """
    Transform a complete LP prototype filter to component networks (denormalized).

    Takes the full g-values list and returns the Foster expansion for each
    reactive element, with actual component values in H and F.

    Parameters
    ----------
    g_values : list
        LP prototype g-values [g0, g1, g2, ..., gn, g_{n+1}].
    f_zeros : array_like
        Transmission zero frequencies (in units specified by `units`).
    f_poles : array_like
        Transmission pole frequencies (in units specified by `units`).
    Z0_ohm : float
        Reference impedance in Ohms.
    fc : float
        Center/cutoff frequency (in units specified by `units`).
    foster_form : int, optional
        Foster form to use (1 or 2). Default is 1.
    zero_at_zero : bool, optional
        Whether frequency transform has zero at w=0. If None, determined automatically.
    units : str, optional
        Frequency units: 'GHz', 'MHz', 'kHz', or 'Hz'. Default is 'GHz'.
    verbose : bool, optional
        Print component values. Default is False.

    Returns
    -------
    dict
        Complete filter transformation with keys:
        - 'series': list of dicts for series branches (component values in H and F)
        - 'shunt': list of dicts for shunt branches (component values in H and F)
        - 'g0': source impedance
        - 'g_load': load impedance
        - 'Z0': reference impedance used
        - 'fc_Hz': center frequency in Hz
        - 'foster_form': which Foster form was used
        - 'zero_at_zero': whether there's a zero at w=0
        - 'f_zeros': transmission zeros (in specified units)
        - 'f_poles': transmission poles (in specified units)
        - 'w_zeros': normalized transmission zeros (f_zeros / fc)
        - 'w_poles': normalized transmission poles (f_poles / fc)
        - 'units': frequency units used
    """
    # Unit conversion to Hz
    unit_multipliers = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}
    if units not in unit_multipliers:
        raise ValueError(f"Unknown units '{units}'. Use 'Hz', 'kHz', 'MHz', or 'GHz'.")
    unit_mult = unit_multipliers[units]

    f_zeros = np.atleast_1d(f_zeros) if f_zeros is not None else np.array([])
    f_poles = np.atleast_1d(f_poles) if f_poles is not None else np.array([])

    # Convert to Hz
    fc_Hz = fc * unit_mult

    # Normalize frequencies (ratio is unit-independent)
    w_zeros = f_zeros / fc if len(f_zeros) > 0 else np.array([])
    w_poles = f_poles / fc if len(f_poles) > 0 else np.array([])
    wc = 2 * np.pi * fc_Hz

    # Determine zero_at_zero: pass user preference to should_have_zero_at_zero
    # When both w_zeros and w_poles are empty, the user's choice is respected
    # When finite poles/zeros exist, the value is auto-determined by the alternation rule
    default_zero_at_zero = True if zero_at_zero is None else zero_at_zero
    zero_at_zero = should_have_zero_at_zero(w_zeros, w_poles, default_zero_at_zero)

    n = len(g_values) - 2  # filter order

    series_branches = []
    shunt_branches = []

    for k in range(1, n + 1):
        g_k = np.float64(g_values[k])  # Convert to numpy float for proper inf handling

        if k % 2 == 1:
            ind_or_cap = 'ind'
            branch_list = series_branches
        else:
            ind_or_cap = 'cap'
            branch_list = shunt_branches

        fc_GHz = fc_Hz / 1e9  # For twpa_design functions

        if foster_form == 1:
            Linf, C0, Li, Ci = filter_transfo_Foster1(
                g_k, ind_or_cap, zero_at_zero, w_poles, w_zeros,
                Z0_ohm=Z0_ohm, fc_filter_GHz=fc_GHz, verbose=verbose
            )
            w_res = w_poles if ind_or_cap == 'ind' else w_zeros
            branch = {
                'g_k': g_k, 'index': k,
                'Linf': Linf, 'C0': C0, 'Li': Li, 'Ci': Ci,
                'w_res': w_res * wc  # denormalized resonance frequencies
            }
        else:
            L0, Cinf, Li, Ci = filter_transfo_Foster2(
                g_k, ind_or_cap, zero_at_zero, w_poles, w_zeros,
                Z0_ohm=Z0_ohm, fc_filter_GHz=fc_GHz, verbose=verbose
            )
            w_res = w_zeros if ind_or_cap == 'ind' else w_poles
            branch = {
                'g_k': g_k, 'index': k,
                'L0': L0, 'Cinf': Cinf, 'Li': Li, 'Ci': Ci,
                'w_res': w_res * wc
            }

        branch_list.append(branch)

    return {
        'series': series_branches,
        'shunt': shunt_branches,
        'g0': g_values[0],
        'g_load': g_values[-1],
        'Z0': Z0_ohm,
        'fc_Hz': fc_Hz,
        'fc': fc,
        'foster_form': foster_form,
        'zero_at_zero': zero_at_zero,
        'f_zeros': f_zeros,
        'f_poles': f_poles,
        'w_zeros': w_zeros,
        'w_poles': w_poles,
        'units': units,
        'order': n
    }


def denormalize(L_norm, C_norm, Z0, fc):
    """
    Denormalize inductance and capacitance values.

    Parameters
    ----------
    L_norm : float or array_like
        Normalized inductance (dimensionless).
    C_norm : float or array_like
        Normalized capacitance (dimensionless).
    Z0 : float
        Reference impedance in Ohms.
    fc : float
        Center/cutoff frequency in Hz.

    Returns
    -------
    tuple
        (L_H, C_F) - Inductance in Henries, capacitance in Farads.
    """
    wc = 2 * np.pi * fc
    L_H = L_norm * Z0 / wc
    C_F = C_norm / (Z0 * wc)
    return L_H, C_F


def normalize(L_H, C_F, Z0, fc):
    """
    Normalize inductance and capacitance values.

    Parameters
    ----------
    L_H : float or array_like
        Inductance in Henries.
    C_F : float or array_like
        Capacitance in Farads.
    Z0 : float
        Reference impedance in Ohms.
    fc : float
        Center/cutoff frequency in Hz.

    Returns
    -------
    tuple
        (L_norm, C_norm) - Normalized (dimensionless) values.
    """
    wc = 2 * np.pi * fc
    L_norm = L_H * wc / Z0
    C_norm = C_F * Z0 * wc
    return L_norm, C_norm


def print_filter(result, units='auto', units_L='auto', units_C='auto'):
    """
    Pretty-print the transformed filter components.

    Parameters
    ----------
    result : dict
        Output from frequency_transfo_normalized() or frequency_transfo().
    units : str
        'normalized' for dimensionless, 'auto' for SI with prefixes.
    units_L : str
        Inductance units: 'auto', 'pH', 'nH', 'µH', 'uH', or 'H'.
        Default is 'auto' (choose based on magnitude).
    units_C : str
        Capacitance units: 'auto', 'fF', 'pF', 'nF', or 'F'.
        Default is 'auto' (choose based on magnitude).
    """
    is_normalized = 'Z0' not in result
    foster_form = result['foster_form']

    print(f"Filter Transformation Results")
    print(f"=" * 50)
    print(f"Order: {result['order']}")
    print(f"Foster Form: {foster_form}")
    print(f"Zero at ω=0: {result['zero_at_zero']}")
    if not is_normalized:
        print(f"Z0 = {result['Z0']} Ω, fc = {result['fc']:.4g} {result['units']}")
    print()

    # Unit multipliers and labels
    L_units = {'pH': (1e12, 'pH'), 'nH': (1e9, 'nH'), 'µH': (1e6, 'µH'), 'uH': (1e6, 'µH'), 'H': (1, 'H')}
    C_units = {'fF': (1e15, 'fF'), 'pF': (1e12, 'pF'), 'nF': (1e9, 'nF'), 'F': (1, 'F')}

    def format_val(val, unit_type):
        if val is None or (isinstance(val, (int, float)) and (np.isinf(val) or val == 0)):
            return f"{val}"
        if is_normalized or units == 'normalized':
            return f"{val:.6g}"
        else:
            if unit_type == 'L':
                if units_L != 'auto' and units_L in L_units:
                    mult, label = L_units[units_L]
                    return f"{val*mult:.4g} {label}"
                else:
                    # Auto-select based on magnitude
                    if abs(val) < 1e-9:
                        return f"{val*1e12:.4g} pH"
                    elif abs(val) < 1e-6:
                        return f"{val*1e9:.4g} nH"
                    else:
                        return f"{val*1e6:.4g} µH"
            else:  # C
                if units_C != 'auto' and units_C in C_units:
                    mult, label = C_units[units_C]
                    return f"{val*mult:.4g} {label}"
                else:
                    # Auto-select based on magnitude
                    if abs(val) < 1e-12:
                        return f"{val*1e15:.4g} fF"
                    elif abs(val) < 1e-9:
                        return f"{val*1e12:.4g} pF"
                    else:
                        return f"{val*1e9:.4g} nF"

    # Combine all branches and sort by index for sequential output
    all_branches = result['series'] + result['shunt']
    all_branches.sort(key=lambda b: b['index'])

    def format_g(val):
        """Format normalized g-value."""
        if val is None:
            return "None"
        if isinstance(val, (int, float)) and np.isinf(val):
            return "inf"
        if isinstance(val, (int, float)) and val == 0:
            return "0"
        return f"{val:.6g}"

    # Determine if LP or HP transformation
    zero_at_zero = result['zero_at_zero']

    print(f"Components:")
    print("-" * 50)
    for branch in all_branches:
        branch_type = "series" if branch['index'] % 2 == 1 else "shunt"
        g_k = branch['g_k']
        print(f"  g[{branch['index']}] = {g_k:.6g} ({branch_type}):")

        if foster_form == 1:
            Linf = branch['Linf']
            C0 = branch['C0']

            # Calculate effective normalized g-values for this transformation
            # LP (zero_at_zero=True): g_inf = g_k, g_0 = inf
            # HP (zero_at_zero=False): g_inf = inf, g_0 = 1/g_k
            if zero_at_zero:
                g_inf = g_k
                g_0 = np.inf
            else:
                g_inf = np.inf
                g_0 = 1.0 / g_k if g_k != 0 else np.inf

            print(f"    g_inf = {format_g(g_inf):>10s} -> Linf = {format_val(Linf, 'L')}")
            print(f"    g_0   = {format_g(g_0):>10s} -> C0   = {format_val(C0, 'C')}")
        else:
            L0 = branch['L0']
            Cinf = branch['Cinf']

            # For Foster Form 2
            if zero_at_zero:
                g_0 = g_k
                g_inf = np.inf
            else:
                g_0 = np.inf
                g_inf = 1.0 / g_k if g_k != 0 else np.inf

            print(f"    g_0   = {format_g(g_0):>10s} -> L0   = {format_val(L0, 'L')}")
            print(f"    g_inf = {format_g(g_inf):>10s} -> Cinf = {format_val(Cinf, 'C')}")

        if len(branch['Li']) > 0:
            print(f"    Li   = [{', '.join(format_val(v, 'L') for v in branch['Li'])}]")
            print(f"    Ci   = [{', '.join(format_val(v, 'C') for v in branch['Ci'])}]")


def get_max_LC(result):
    """Extract maximum inductance and capacitance from frequency_transfo result.

    Parameters
    ----------
    result : dict
        Output from frequency_transfo() or frequency_transfo_normalized().

    Returns
    -------
    tuple
        (max_L, max_C) - Maximum inductance and capacitance values.
        Units match the input result (H and F for frequency_transfo,
        normalized for frequency_transfo_normalized).
    """
    all_L = []
    all_C = []

    foster_form = result['foster_form']

    for branch in result['series'] + result['shunt']:
        if foster_form == 1:
            # Foster Form 1: Linf, C0, Li, Ci
            if branch['Linf'] != 0 and not np.isinf(branch['Linf']):
                all_L.append(branch['Linf'])
            if branch['C0'] != np.inf and branch['C0'] != 0:
                all_C.append(branch['C0'])
        else:
            # Foster Form 2: L0, Cinf, Li, Ci
            if branch['L0'] != np.inf and branch['L0'] != 0:
                all_L.append(branch['L0'])
            if branch['Cinf'] != 0 and not np.isinf(branch['Cinf']):
                all_C.append(branch['Cinf'])

        # LC tank components (both forms)
        all_L.extend([L for L in branch['Li'] if L != 0 and not np.isinf(L)])
        all_C.extend([C for C in branch['Ci'] if C != np.inf and C != 0])

    max_L = max(all_L) if all_L else 0
    max_C = max(all_C) if all_C else 0

    return max_L, max_C


# =============================================================================
# ABCD Matrix and S-Parameter Calculations
# =============================================================================

# Import ABCD helper functions from package
from .helper_functions import (
    Z_Foster_form_C,
    Y_Foster_form_C,
)


def _Z_series_foster1(Linf, C0, Li, Ci, w, n_poles):
    """
    Calculate series impedance for Foster Form 1 series (inductor) branch.

    This is a simplified version for pure filters (no JJ nonlinearity).
    Z = j*w*Linf + 1/(j*w*C0) + sum_i(parallel LC tanks)
    """
    # Handle w=0 (DC): capacitors are open circuit (Z=inf), inductors are short circuit (Z=0)
    if w == 0:
        # At DC, if there's a capacitor C0 in series, impedance is infinite
        if C0 != np.inf and C0 != 0:
            return np.inf
        # Otherwise, inductors are short circuits, so Z=0
        return 0.0

    Z = 0j

    # Linf contribution
    if Linf != 0 and not np.isinf(Linf):
        Z += 1j * w * Linf

    # C0 contribution
    if C0 != np.inf and C0 != 0:
        Z += 1 / (1j * w * C0)

    # LC tank contributions
    for j in range(n_poles):
        if Li[j] != 0 and Ci[j] != np.inf:
            denom = 1 / (1j * w * Li[j]) + 1j * w * Ci[j]
            if np.abs(denom) > 1e-30:
                Z += 1 / denom

    return np.asarray(Z).item()


def _Y_series_foster2(L0, Cinf, Li, Ci, w, n_zeros):
    """
    Calculate series admittance for Foster Form 2 series (inductor) branch.

    Y = j*w*Cinf + 1/(j*w*L0) + sum_i(series LC tanks as admittance)
    """
    # Handle w=0 (DC): inductor admittance 1/(jwL) diverges, capacitor admittance jwC=0
    if w == 0:
        # At DC, if there's an inductor L0 in shunt, admittance is infinite
        if L0 != np.inf and L0 != 0:
            return np.inf
        # Otherwise, capacitors have zero admittance at DC
        return 0.0

    Y = 0j

    # Cinf contribution
    if Cinf != 0 and not np.isinf(Cinf):
        Y += 1j * w * Cinf

    # L0 contribution
    if L0 != np.inf and L0 != 0:
        Y += 1 / (1j * w * L0)

    # Series LC tank contributions (as admittances in parallel)
    for j in range(n_zeros):
        if Li[j] != np.inf and Ci[j] != 0:
            Z_tank = 1j * w * Li[j] + 1 / (1j * w * Ci[j])
            if np.abs(Z_tank) > 1e-30:
                Y += 1 / Z_tank

    return np.asarray(Y).item()


def calculate_ABCD(result, f, units='GHz'):
    """
    Calculate the ABCD matrix of the transformed filter at given frequencies.

    Parameters
    ----------
    result : dict
        Output from frequency_transfo_normalized() or frequency_transfo().
    f : array_like
        Frequencies at which to evaluate (in units specified by `units`).
    units : str, optional
        Frequency units: 'GHz', 'MHz', 'kHz', 'Hz', or 'normalized'.
        Default is 'GHz'. Use 'normalized' for normalized frequencies (w).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'ABCD': array of shape (2, 2, n_f) containing ABCD matrices
        - 'f': frequency array used
        - 'w': angular frequency array (rad/s or normalized)
        - 'units': frequency units
    """
    f = np.atleast_1d(f)
    n_f = len(f)

    is_normalized = 'Z0' not in result
    foster_form = result['foster_form']
    n_zeros = len(result['w_zeros'])
    n_poles = len(result['w_poles'])

    # Convert frequency to angular frequency
    if units == 'normalized' or is_normalized:
        w = f  # f is actually w (normalized angular frequency)
    else:
        unit_multipliers = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}
        if units not in unit_multipliers:
            raise ValueError(f"Unknown units '{units}'. Use 'Hz', 'kHz', 'MHz', 'GHz', or 'normalized'.")
        w = 2 * np.pi * f * unit_multipliers[units]

    # Sort branches by index for proper cascading
    all_branches = result['series'] + result['shunt']
    all_branches.sort(key=lambda b: b['index'])

    # Initialize ABCD matrices
    ABCD = np.zeros((2, 2, n_f), dtype=complex)

    for i_f in range(n_f):
        # Start with identity matrix
        ABCD_total = np.eye(2, dtype=complex)

        for branch in all_branches:
            is_series = (branch['index'] % 2 == 1)

            if is_series:
                # Series branch (inductor type)
                if foster_form == 1:
                    Z = _Z_series_foster1(
                        branch['Linf'], branch['C0'],
                        branch['Li'], branch['Ci'],
                        w[i_f], n_poles
                    )
                    ABCD_branch = np.array([[1, Z], [0, 1]], dtype=complex)
                else:
                    Y = _Y_series_foster2(
                        branch['L0'], branch['Cinf'],
                        branch['Li'], branch['Ci'],
                        w[i_f], n_zeros
                    )
                    Z = 1/Y if np.abs(Y) > 1e-30 else np.inf
                    ABCD_branch = np.array([[1, Z], [0, 1]], dtype=complex)
            else:
                # Shunt branch (capacitor type) - use twpa_design functions
                if foster_form == 1:
                    # Z_Foster_form_C returns impedance
                    Z = Z_Foster_form_C(
                        branch['Linf'], branch['C0'],
                        branch['Li'], branch['Ci'],
                        w[i_f], n_zeros
                    )
                    Y = 1/Z if np.abs(Z) > 1e-30 else np.inf
                    ABCD_branch = np.array([[1, 0], [Y, 1]], dtype=complex)
                else:
                    # Y_Foster_form_C returns admittance
                    Y = Y_Foster_form_C(
                        branch['L0'], branch['Cinf'],
                        branch['Li'], branch['Ci'],
                        w[i_f], n_poles
                    )
                    ABCD_branch = np.array([[1, 0], [Y, 1]], dtype=complex)

            # Cascade
            ABCD_total = ABCD_total @ ABCD_branch

        ABCD[:, :, i_f] = ABCD_total

    return {
        'ABCD': ABCD,
        'f': f,
        'w': w,
        'units': units
    }


def calculate_S21(result, f, Z0=None, units='GHz'):
    """
    Calculate S21 (forward transmission) of the transformed filter.

    Parameters
    ----------
    result : dict
        Output from frequency_transfo_normalized() or frequency_transfo().
    f : array_like
        Frequencies at which to evaluate.
    Z0 : float, optional
        Reference impedance for S-parameters. If None, uses result['Z0']
        or 1.0 for normalized results.
    units : str, optional
        Frequency units. Default is 'GHz'. Use 'normalized' for normalized.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'S21': complex S21 values
        - 'S21_dB': |S21| in dB
        - 'S21_phase_deg': phase of S21 in degrees
        - 'f': frequency array
        - 'w': angular frequency array
        - 'units': frequency units
        - 'Z0': reference impedance used
        - 'ABCD': the ABCD matrices used
    """
    # Get reference impedance
    if Z0 is None:
        Z0 = result.get('Z0', 1.0)

    # Calculate ABCD matrices
    abcd_result = calculate_ABCD(result, f, units=units)
    ABCD = abcd_result['ABCD']

    n_f = ABCD.shape[2]

    # S21 = 2 / (A + B/Z0 + C*Z0 + D)
    A = ABCD[0, 0, :]
    B = ABCD[0, 1, :]
    C = ABCD[1, 0, :]
    D = ABCD[1, 1, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        S21 = 2 / (A + B/Z0 + C*Z0 + D)

    return {
        'S21': S21,
        'S21_dB': 20 * np.log10(np.abs(S21) + 1e-30),
        'S21_phase_deg': np.angle(S21, deg=True),
        'f': abcd_result['f'],
        'w': abcd_result['w'],
        'units': units,
        'Z0': Z0,
        'ABCD': ABCD
    }


def plot_response(result, f=None, Z0=None, units='GHz', n_points=1001,
                  f_min=None, f_max=None, show=True):
    """
    Plot the amplitude and phase response (S21) of the transformed filter.

    Parameters
    ----------
    result : dict
        Output from frequency_transfo_normalized() or frequency_transfo().
    f : array_like, optional
        Frequencies at which to evaluate. If None, auto-generated.
    Z0 : float, optional
        Reference impedance. If None, uses result['Z0'] or 1.0.
    units : str, optional
        Frequency units: 'GHz', 'MHz', 'kHz', 'Hz', or 'normalized'.
    n_points : int, optional
        Number of frequency points if f is auto-generated.
    f_min : float, optional
        Minimum frequency for auto-generation.
    f_max : float, optional
        Maximum frequency for auto-generation.
    show : bool, optional
        Whether to call plt.show(). Default is True.

    Returns
    -------
    tuple
        (fig, axes, s21_result) - Figure, axes array, and S21 calculation result.
    """
    import matplotlib.pyplot as plt

    # Try to import plots_params for consistent styling
    try:
        from plots_params import fontsize_title, blue, black, gray
    except ImportError:
        fontsize_title = 8
        blue = np.array([0.22, 0.49, 0.72])
        black = np.array([0.0, 0.0, 0.0])
        gray = np.array([0.6, 0.6, 0.6])

    is_normalized = 'Z0' not in result

    # Auto-generate frequency array if not provided
    if f is None:
        if is_normalized:
            # For normalized, use w around the zeros/poles
            w_zeros = result.get('w_zeros', np.array([1.0]))
            w_poles = result.get('w_poles', np.array([]))
            all_w = np.concatenate([np.atleast_1d(w_zeros), np.atleast_1d(w_poles)])
            if len(all_w) > 0 and np.any(all_w > 0):
                w_center = np.mean(all_w[all_w > 0])
            else:
                w_center = 1.0
            f_min = f_min if f_min is not None else 0.01 * w_center
            f_max = f_max if f_max is not None else 3.0 * w_center
            units = 'normalized'
        else:
            # For denormalized, use frequency around fc
            fc = result['fc']
            f_zeros = result.get('f_zeros', np.array([]))
            f_poles = result.get('f_poles', np.array([]))
            all_f = np.concatenate([np.atleast_1d(f_zeros), np.atleast_1d(f_poles)])
            if len(all_f) > 0 and np.any(all_f > 0):
                f_center = np.mean(all_f[all_f > 0])
            else:
                f_center = fc
            f_min = f_min if f_min is not None else 0.01 * f_center
            f_max = f_max if f_max is not None else 3.0 * f_center
            units = result.get('units', 'GHz')

        f = np.linspace(f_min, f_max, n_points)

    # Calculate S21
    s21_result = calculate_S21(result, f, Z0=Z0, units=units)

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(8.6/2.54 * 1.5, 8.6/2.54 * 1.2), sharex=True)
    fig.subplots_adjust(hspace=0.1)

    # Determine x-axis label
    if units == 'normalized':
        xlabel = r'$\omega$ (normalized)'
    else:
        xlabel = f'Frequency [{units}]'

    # Plot amplitude (|S21| in dB)
    ax1 = axes[0]
    ax1.plot(s21_result['f'], s21_result['S21_dB'], color=blue)
    ax1.set_ylabel(r'$|S_{21}|$ [dB]')
    ax1.axhline(y=0, color=black, linestyle='-', linewidth=0.5)
    ax1.axhline(y=-3, color=gray, linestyle=':', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=max(-60, np.min(s21_result['S21_dB']) - 5))

    # Plot phase
    ax2 = axes[1]
    ax2.plot(s21_result['f'], s21_result['S21_phase_deg'], color=blue)
    ax2.set_ylabel(r'$\angle S_{21}$ [deg]')
    ax2.set_xlabel(xlabel)
    ax2.axhline(y=0, color=black, linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    # Add title with filter info
    order = result['order']
    foster_form = result['foster_form']
    title = f"Order {order} Filter Response (Foster Form {foster_form})"
    if not is_normalized:
        title += f"\n$Z_0$ = {result['Z0']} $\\Omega$, $f_c$ = {result['fc']:.4g} {result['units']}"
    ax1.set_title(title, fontsize=fontsize_title)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes, s21_result


def calculate_input_impedance(result, f, Z_load=None, port='input', units='GHz'):
    """
    Calculate the input impedance of a filter.

    Parameters
    ----------
    result : dict
        Output from frequency_transfo_normalized() or frequency_transfo().
    f : array_like
        Frequencies at which to evaluate.
    Z_load : float, optional
        Load impedance at the opposite port. If None, uses result['Z0'] or 50 ohms.
    port : str, optional
        Which port to look into: 'input' or 'output'.
        - 'input': Look into pass port (g0 side) with load at common port (g_{n+1} side).
                   Uses Z_in = (A*Z_load + B) / (C*Z_load + D)
        - 'output': Look into common port (g_{n+1} side) with load at pass port (g0 side).
                    Uses reversed ABCD: Z_in = (D*Z_load + B) / (C*Z_load + A)
        Default is 'input' for backward compatibility.
    units : str, optional
        Frequency units. Default is 'GHz'.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'Z_in': complex input impedance array
        - 'Y_in': complex input admittance array (1/Z_in)
        - 'f': frequency array
        - 'w': angular frequency array
        - 'units': frequency units
        - 'port': which port was used
        - 'Z_load': load impedance used
    """
    # Calculate ABCD matrices
    abcd_result = calculate_ABCD(result, f, units=units)
    ABCD = abcd_result['ABCD']

    A = ABCD[0, 0, :]
    B = ABCD[0, 1, :]
    C = ABCD[1, 0, :]
    D = ABCD[1, 1, :]

    # Get load impedance
    if Z_load is None:
        Z_load = result.get('Z0', 50.0)

    if port == 'input':
        # Looking into input (pass port, g0 side) with load at output (common port)
        # Z_in = (A*Z_load + B) / (C*Z_load + D)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z_in = (A * Z_load + B) / (C * Z_load + D)
    elif port == 'output':
        # Looking into output (common port, g_{n+1} side) with load at input (pass port)
        # Use reversed ABCD: Z_in = (D*Z_load + B) / (C*Z_load + A)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z_in = (D * Z_load + B) / (C * Z_load + A)
    else:
        raise ValueError(f"Unknown port '{port}'. Use 'input' or 'output'.")

    # Calculate admittance
    with np.errstate(divide='ignore', invalid='ignore'):
        Y_in = 1.0 / Z_in
        Y_in = np.where(np.isinf(Z_in), 0.0, Y_in)

    return {
        'Z_in': Z_in,
        'Y_in': Y_in,
        'f': abcd_result['f'],
        'w': abcd_result['w'],
        'units': units,
        'port': port,
        'Z_load': Z_load,
        'ABCD': ABCD
    }


def calculate_multiplexer_response(filter_results, f, Z0=None, units='GHz'):
    """
    Calculate the S-parameters of a multiplexer (diplexer, triplexer, etc.).

    Topology:
        Port 2 (Z0) ──[Filter 1]──┐
                                  ├── Port 1 (Z0, common)
        Port 3 (Z0) ──[Filter 2]──┘
        ...

    Each filter is designed with singly-terminated g-values. The "open" end
    of each filter connects to the common port (Port 1). All ports are
    terminated in Z0.

    Parameters
    ----------
    filter_results : list of dict
        List of filter results from frequency_transfo(). Each filter should
        be designed with singly-terminated g-values (g_load = inf).
    f : array_like
        Frequencies at which to evaluate.
    Z0 : float, optional
        Reference impedance for S-parameters at ALL ports.
        If None, uses Z0 from the first filter result, or 50 ohms.
    units : str, optional
        Frequency units. Default is 'GHz'.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'S11': complex S11 (reflection at common port)
        - 'S11_dB': |S11| in dB
        - 'S_k1': list of complex transmission coefficients (S21, S31, ...)
        - 'S_k1_dB': list of |S_k1| in dB for each port
        - 'Z_total': total input impedance at common port
        - 'Y_total': total input admittance at common port
        - 'Z_branches': list of input impedances for each branch
        - 'f': frequency array
        - 'w': angular frequency array
        - 'units': frequency units
        - 'Z0': reference impedance

    Notes
    -----
    The ABCD matrix from calculate_ABCD is oriented from pass port (g0) to
    common port (g_{n+1}). Since we look INTO filters from the common port,
    we use the reversed ABCD: [D, B; C, A].

    For each filter looking from Port 1 (common) toward the pass port (Z0 load):
        Z_in_k = (D*Z0 + B) / (C*Z0 + A)

    The filters are in parallel at Port 1:
        Y_total = Y_1 + Y_2 + ... + Y_N
        Z_total = 1 / Y_total
        S11 = (Z_total - Z0) / (Z_total + Z0)

    For transmission S_k1 through filter k, we cascade:
        1. A shunt admittance Y_others representing all OTHER filters
           ABCD_shunt = [1, 0; Y_others, 1]
        2. The reversed ABCD matrix of filter k

        ABCD_eff = ABCD_shunt @ ABCD_k_reversed

    Then use standard S21 formula with Z0 at both ports:
        S21 = 2 / (A_eff + B_eff/Z0 + C_eff*Z0 + D_eff)
    """
    f = np.atleast_1d(f)
    n_f = len(f)
    n_filters = len(filter_results)

    # Get reference impedance
    if Z0 is None:
        Z0 = filter_results[0].get('Z0', 50.0)

    # Calculate ABCD and input impedance for each filter
    #
    # The ABCD matrix from calculate_ABCD is oriented from g0 (pass port) to g_{n+1} (common port).
    # In the diplexer, we look INTO the filter from the common port side, so we need
    # the reversed ABCD matrix: [D, B; C, A] (swap A <-> D for reciprocal network).
    #
    # Z_in from common port = (D*Z0 + B) / (C*Z0 + A)
    Z_branches = []
    Y_branches = []
    ABCD_branches = []

    for i, result in enumerate(filter_results):
        # Get ABCD matrices (oriented from pass port to common port)
        abcd_result = calculate_ABCD(result, f, units=units)
        ABCD = abcd_result['ABCD']

        A = ABCD[0, 0, :]
        B = ABCD[0, 1, :]
        C = ABCD[1, 0, :]
        D = ABCD[1, 1, :]

        # Reversed ABCD matrix for looking from common port: [D, B; C, A]
        ABCD_reversed = np.zeros_like(ABCD)
        ABCD_reversed[0, 0, :] = D
        ABCD_reversed[0, 1, :] = B
        ABCD_reversed[1, 0, :] = C
        ABCD_reversed[1, 1, :] = A
        ABCD_branches.append(ABCD_reversed)

        # Input impedance looking into filter FROM COMMON PORT with Z0 load at pass port
        # Z_in = (D*Z0 + B) / (C*Z0 + A)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z_in = (D * Z0 + B) / (C * Z0 + A)

        Z_branches.append(Z_in)
        Y_branches.append(1.0 / Z_in)

    # Total admittance (parallel combination at common port)
    Y_total = np.zeros(n_f, dtype=complex)
    for Y_branch in Y_branches:
        Y_total += Y_branch

    # Total impedance
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_total = 1.0 / Y_total
        Z_total = np.where(np.abs(Y_total) < 1e-30, np.inf, Z_total)

    # Calculate transmission S_k1 to each output port
    #
    # For S21 (through filter k), we cascade:
    # 1. A shunt admittance representing all OTHER filters in parallel
    #    ABCD_shunt = [1, 0; Y_others, 1]
    # 2. The ABCD matrix of filter k
    #
    # ABCD_eff = ABCD_shunt @ ABCD_k
    # Then use standard S21 formula with Z0 at both ports.

    S_k1 = []
    S_k1_dB = []

    for k in range(n_filters):
        ABCD_k = ABCD_branches[k]
        A_k = ABCD_k[0, 0, :]
        B_k = ABCD_k[0, 1, :]
        C_k = ABCD_k[1, 0, :]
        D_k = ABCD_k[1, 1, :]

        # Calculate Y_others = sum of input admittances of all OTHER filters
        Y_others = np.zeros(n_f, dtype=complex)
        for j in range(n_filters):
            if j != k:
                Y_others += Y_branches[j]

        # ABCD_shunt = [1, 0; Y_others, 1]
        # ABCD_eff = ABCD_shunt @ ABCD_k
        # [1    0  ] [A_k  B_k]   [A_k           B_k        ]
        # [Y    1  ] [C_k  D_k] = [Y*A_k + C_k   Y*B_k + D_k]
        A_eff = A_k
        B_eff = B_k
        C_eff = Y_others * A_k + C_k
        D_eff = Y_others * B_k + D_k

        # Standard S21 formula with matched ports (Z0 at input and output)
        # S21 = 2 / (A + B/Z0 + C*Z0 + D)
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = A_eff + B_eff / Z0 + C_eff * Z0 + D_eff
            S_k1_i = 2.0 / denom

        S_k1.append(S_k1_i)
        S_k1_dB.append(20 * np.log10(np.abs(S_k1_i) + 1e-30))

    # Calculate S11 at common port
    # Z_total is the parallel combination of all filter input impedances
    # S11 = (Z_total - Z0) / (Z_total + Z0)
    with np.errstate(divide='ignore', invalid='ignore'):
        S11 = (Z_total - Z0) / (Z_total + Z0)

    # Get frequency/angular frequency from ABCD calculation
    abcd_result = calculate_ABCD(filter_results[0], f, units=units)

    return {
        'S11': S11,
        'S11_dB': 20 * np.log10(np.abs(S11) + 1e-30),
        'S11_phase_deg': np.angle(S11, deg=True),
        'S_k1': S_k1,
        'S_k1_dB': S_k1_dB,
        'Z_total': Z_total,
        'Y_total': Y_total,
        'Z_branches': Z_branches,
        'f': abcd_result['f'],
        'w': abcd_result['w'],
        'units': units,
        'Z0': Z0,
        'n_filters': n_filters
    }


def plot_multiplexer_response(filter_results, f=None, Z0=None, units='GHz',
                               n_points=1001, f_min=None, f_max=None,
                               labels=None, show=True):
    """
    Plot the response of a multiplexer (diplexer, triplexer, etc.).

    Parameters
    ----------
    filter_results : list of dict
        List of filter results from frequency_transfo().
    f : array_like, optional
        Frequencies at which to evaluate. If None, auto-generated.
    Z0 : float, optional
        Reference impedance for S-parameters.
    units : str, optional
        Frequency units. Default is 'GHz'.
    n_points : int, optional
        Number of frequency points if f is auto-generated.
    f_min : float, optional
        Minimum frequency for auto-generation.
    f_max : float, optional
        Maximum frequency for auto-generation.
    labels : list of str, optional
        Labels for each filter branch (e.g., ['LPF', 'HPF']).
    show : bool, optional
        Whether to call plt.show(). Default is True.

    Returns
    -------
    tuple
        (fig, axes, result) - Figure, axes array, and multiplexer result dict.
    """
    import matplotlib.pyplot as plt

    # Try to import plots_params for consistent styling
    try:
        from plots_params import fontsize_title, blue, orange, black, gray
    except ImportError:
        fontsize_title = 8
        blue = np.array([0.22, 0.49, 0.72])
        orange = np.array([0.94, 0.50, 0.15])
        black = np.array([0.0, 0.0, 0.0])
        gray = np.array([0.6, 0.6, 0.6])

    # Default colors for multiple branches
    colors = [blue, orange, np.array([0.2, 0.7, 0.3]), np.array([0.8, 0.2, 0.2]),
              np.array([0.5, 0.2, 0.8]), np.array([0.2, 0.7, 0.7])]

    n_filters = len(filter_results)

    # Auto-generate frequency array if not provided
    if f is None:
        # Find frequency range from all filters
        fc_list = [r.get('fc', 1.0) for r in filter_results]
        fc_max = max(fc_list)
        fc_min = min(fc_list)

        if f_min is None:
            f_min = 0.0
        if f_max is None:
            f_max = 2.0 * fc_max

        f = np.linspace(f_min, f_max, n_points)

    # Calculate multiplexer response
    result = calculate_multiplexer_response(filter_results, f, Z0=Z0, units=units)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    # Default labels
    if labels is None:
        labels = [f'Filter {i+1}' for i in range(n_filters)]

    # Plot amplitude
    ax1 = axes[0]
    ax1.plot(result['f'], result['S11_dB'], color=black, linestyle='--',
             label=r'$S_{11}$ (reflection)')
    for i in range(n_filters):
        ax1.plot(result['f'], result['S_k1_dB'][i],
                 color=colors[i % len(colors)],
                 label=f'$S_{{{i+2}1}}$ ({labels[i]})')

    ax1.set_ylabel(r'$|S|$ [dB]')
    ax1.axhline(y=0, color=black, linestyle='-', linewidth=0.5)
    ax1.axhline(y=-3, color=gray, linestyle=':', linewidth=0.5)
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=max(-60, min([min(s) for s in result['S_k1_dB']]) - 5))

    # Plot phase
    ax2 = axes[1]
    ax2.plot(result['f'], result['S11_phase_deg'], color=black, linestyle='--',
             label=r'$S_{11}$')
    for i in range(n_filters):
        ax2.plot(result['f'], np.angle(result['S_k1'][i], deg=True),
                 color=colors[i % len(colors)],
                 label=f'$S_{{{i+2}1}}$ ({labels[i]})')

    ax2.set_ylabel(r'$\angle S$ [deg]')
    if units == 'normalized':
        ax2.set_xlabel(r'$\omega$ (normalized)')
    else:
        ax2.set_xlabel(f'Frequency [{units}]')
    ax2.axhline(y=0, color=black, linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3)

    # Title
    n_plex = {2: 'Diplexer', 3: 'Triplexer', 4: 'Quadplexer'}.get(n_filters, f'{n_filters}-plexer')
    ax1.set_title(f'{n_plex} Response ($Z_0$ = {result["Z0"]} $\\Omega$)', fontsize=fontsize_title)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes, result


# ============================================================================
# Section 3: Dataclasses
# ============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum


class FilterType(Enum):
    """Supported filter response types."""
    LOWPASS = "lp"
    HIGHPASS = "hp"
    BANDPASS = "bp"
    BANDSTOP = "bs"


@dataclass
class FilterSpec:
    """Specification for a single filter.

    Parameters
    ----------
    response : str or FilterType
        Filter response: 'lp', 'hp', 'bp', or 'bs'.
    order : int
        Filter order (number of reactive elements).
    fc : float
        Cutoff frequency (LP/HP) or center frequency (BP/BS) in Hz.
    approx : str
        Approximation type: 'butterworth' or 'chebyshev1'.
    ripple_dB : float, optional
        Passband ripple in dB. Required for 'chebyshev1'.
    Z0 : float
        Reference impedance in Ohms. Default 50.
    bw : float, optional
        Bandwidth in Hz. Required for 'bp' and 'bs'.
    termination : str
        'double' for standalone filter, 'single' for multiplexer arm.
    foster_form : int
        Foster form to use (1 or 2). Default 1.
    label : str, optional
        Human-readable label (e.g. 'lp', 'hp', 'bp1'). Auto-generated
        from response type if not provided.
    """
    response: Union[FilterType, str]
    order: int
    fc: float
    approx: str = "butterworth"
    ripple_dB: Optional[float] = None
    Z0: float = 50.0
    bw: Optional[float] = None
    termination: str = "double"
    foster_form: int = 1
    label: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.response, str):
            self.response = FilterType(self.response.lower())
        if self.response in (FilterType.BANDPASS, FilterType.BANDSTOP) and self.bw is None:
            raise ValueError("bw (bandwidth in Hz) is required for bandpass/bandstop filters")
        if self.label is None:
            self.label = self.response.value


@dataclass
class FilterDesign:
    """Result of designing a single filter.

    Attributes
    ----------
    spec : FilterSpec
        The specification that produced this design.
    g_values : list
        LP prototype g-values [g0, g1, ..., gn, g_{n+1}].
    transfo_result : dict
        Output of frequency_transfo(). Contains 'series' and 'shunt'
        branch dicts with L/C values in SI units (H, F).
    """
    spec: FilterSpec
    g_values: list
    transfo_result: dict


@dataclass
class MultiplexerDesign:
    """Result of designing an N-way multiplexer.

    Attributes
    ----------
    arms : list of FilterDesign
        One FilterDesign per arm (all singly terminated).
    Z0 : float
        Common reference impedance.
    label : str
        Identifier like 'diplexer_8.5GHz'.
    """
    arms: List[FilterDesign]
    Z0: float
    label: str = ""


ComponentTuple = Tuple[str, str, str, Union[str, float]]


@dataclass
class PeripheralNetlist:
    """A peripheral circuit (filter, diplexer, ...) in JC netlist form.

    Attributes
    ----------
    components : list of ComponentTuple
        JC-format component tuples (name, node1, node2, value).
    parameters : dict
        Symbolic parameter name -> numeric value mapping.
    port_map : dict
        Maps logical port role to (port_number, node_string).
        Single filter: {'input': (1, 'f1_in'), 'output': (2, 'f1_out')}
        Diplexer: {'common': (1, 'm1_c'), 'lp': (2, 'm1_lp_out'), 'hp': (3, 'm1_hp_out')}
    metadata : dict
        Free-form info: design type, component count, etc.
    """
    components: List[ComponentTuple]
    parameters: Dict[str, float]
    port_map: Dict[str, Tuple[int, str]]
    metadata: Dict = field(default_factory=dict)

    @property
    def port_numbers(self) -> List[int]:
        """Sorted list of port numbers."""
        return sorted(pnum for pnum, _ in self.port_map.values())

    @property
    def n_ports(self) -> int:
        return len(self.port_map)


# ============================================================================
# Section 4: Design functions
# ============================================================================

def _response_to_zeros_poles(spec: FilterSpec) -> Tuple[np.ndarray, np.ndarray, Optional[bool]]:
    """Map a FilterSpec to (f_zeros_Hz, f_poles_Hz, zero_at_zero) for frequency_transfo."""
    resp = spec.response
    fc = spec.fc

    if resp == FilterType.LOWPASS:
        return np.array([]), np.array([]), True
    elif resp == FilterType.HIGHPASS:
        return np.array([]), np.array([]), False
    elif resp == FilterType.BANDPASS:
        return np.array([fc]), np.array([]), True
    elif resp == FilterType.BANDSTOP:
        return np.array([fc]), np.array([fc]), None
    else:
        raise ValueError(f"Unknown filter response: {resp}")


def design_filter(spec: FilterSpec) -> FilterDesign:
    """Design a standalone doubly-terminated filter.

    Computes g-values and applies frequency transformation to obtain
    physical L/C values in SI units.

    Parameters
    ----------
    spec : FilterSpec
        Complete filter specification. Termination is forced to 'double'.

    Returns
    -------
    FilterDesign
    """
    # Force double termination for standalone filters
    spec = FilterSpec(
        response=spec.response,
        order=spec.order,
        fc=spec.fc,
        approx=spec.approx,
        ripple_dB=spec.ripple_dB,
        Z0=spec.Z0,
        bw=spec.bw,
        termination='double',
        foster_form=spec.foster_form,
        label=spec.label,
    )

    # Compute g-values
    g = calculate_g_values(
        filter_type=spec.approx,
        order=spec.order,
        ripple_dB=spec.ripple_dB,
        termination='double',
    )

    # Determine zeros/poles and zero_at_zero
    f_zeros, f_poles, zero_at_zero = _response_to_zeros_poles(spec)

    # Convert Hz to GHz for frequency_transfo
    fc_GHz = spec.fc / 1e9
    f_zeros_GHz = f_zeros / 1e9
    f_poles_GHz = f_poles / 1e9

    # Run frequency transformation
    result = frequency_transfo(
        g, f_zeros_GHz, f_poles_GHz,
        Z0_ohm=spec.Z0,
        fc=fc_GHz,
        foster_form=spec.foster_form,
        zero_at_zero=zero_at_zero,
        units='GHz',
    )

    return FilterDesign(spec=spec, g_values=g, transfo_result=result)


def design_multiplexer(arm_specs: List[FilterSpec],
                       Z0: float = 50.0,
                       label: str = "") -> MultiplexerDesign:
    """Design an N-way multiplexer from a list of arm specifications.

    Each arm is designed with single termination. Arms can be any mix
    of filter types (LP, HP, BP, BS).

    Parameters
    ----------
    arm_specs : list of FilterSpec
        One spec per multiplexer arm. Termination is forced to 'single'.
    Z0 : float
        Reference impedance for all ports.
    label : str, optional
        Descriptive label.

    Returns
    -------
    MultiplexerDesign

    Examples
    --------
    >>> # LP/HP diplexer
    >>> design_multiplexer([FilterSpec('lp', 7, 8.5e9), FilterSpec('hp', 7, 8.5e9)])

    >>> # Triplexer
    >>> design_multiplexer([
    ...     FilterSpec('lp', 5, 6e9),
    ...     FilterSpec('bp', 5, 8e9, bw=2e9, label='bp1'),
    ...     FilterSpec('hp', 5, 10e9),
    ... ])
    """
    arms = []
    for spec in arm_specs:
        # Force single termination and shared Z0
        arm_spec = FilterSpec(
            response=spec.response,
            order=spec.order,
            fc=spec.fc,
            approx=spec.approx,
            ripple_dB=spec.ripple_dB,
            Z0=Z0,
            bw=spec.bw,
            termination='single',
            foster_form=spec.foster_form,
            label=spec.label,
        )

        g = calculate_g_values(
            filter_type=arm_spec.approx,
            order=arm_spec.order,
            ripple_dB=arm_spec.ripple_dB,
            termination='single',
        )

        f_zeros, f_poles, zero_at_zero = _response_to_zeros_poles(arm_spec)
        fc_GHz = arm_spec.fc / 1e9
        f_zeros_GHz = f_zeros / 1e9
        f_poles_GHz = f_poles / 1e9

        # For singly terminated: g[-1] is inf, replace with 1.0 for transfo
        g_for_transfo = list(g)
        if len(g_for_transfo) > 0 and g_for_transfo[-1] == float('inf'):
            g_for_transfo[-1] = 1.0

        result = frequency_transfo(
            g_for_transfo, f_zeros_GHz, f_poles_GHz,
            Z0_ohm=Z0,
            fc=fc_GHz,
            foster_form=arm_spec.foster_form,
            zero_at_zero=zero_at_zero,
            units='GHz',
        )

        arms.append(FilterDesign(spec=arm_spec, g_values=g, transfo_result=result))

    return MultiplexerDesign(arms=arms, Z0=Z0, label=label)


# ============================================================================
# Section 5: Netlist generation
# ============================================================================

def _to_float(val):
    """Cast numpy scalar to plain Python float for Julia compatibility."""
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    return val


def _branch_to_components(branch: dict, branch_type: str, foster_form: int,
                          prefix: str, node_counter: list,
                          comp_counter: list) -> Tuple[List[ComponentTuple], str, str]:
    """Convert one Foster branch dict into component tuples.

    Parameters
    ----------
    branch : dict
        A branch dict from frequency_transfo result ('series' or 'shunt').
    branch_type : str
        'series' or 'shunt'.
    foster_form : int
        1 or 2.
    prefix : str
        Node name prefix.
    node_counter : list
        Mutable [int] for sequential node allocation.
    comp_counter : list
        Mutable [int] for sequential component naming.

    Returns
    -------
    (components, input_node, output_node)
    """
    components = []

    def next_node():
        node_counter[0] += 1
        return f"{prefix}_{node_counter[0]}"

    def next_comp(comp_type):
        comp_counter[0] += 1
        return f"{comp_type}_{prefix}_{comp_counter[0]}"

    input_node = next_node()
    current_node = input_node

    if branch_type == 'series':
        # Series branch: components in series between input and output
        if foster_form == 1:
            Linf = _to_float(branch.get('Linf', 0))
            C0 = _to_float(branch.get('C0', 0))
            Li_list = [_to_float(x) for x in branch.get('Li', [])]
            Ci_list = [_to_float(x) for x in branch.get('Ci', [])]

            if Linf and Linf > 0 and not np.isinf(Linf):
                next_n = next_node()
                components.append((next_comp('L'), current_node, next_n, Linf))
                current_node = next_n

            if C0 and C0 > 0 and not np.isinf(C0):
                next_n = next_node()
                components.append((next_comp('C'), current_node, next_n, C0))
                current_node = next_n

            for Li, Ci in zip(Li_list, Ci_list):
                if Li and Ci and Li > 0 and Ci > 0:
                    tap_node = next_node()
                    components.append((next_comp('C'), current_node, tap_node, Ci))
                    components.append((next_comp('L'), tap_node, '0', Li))
                    current_node = tap_node

        elif foster_form == 2:
            L0 = _to_float(branch.get('L0', 0))
            Cinf = _to_float(branch.get('Cinf', 0))
            Li_list = [_to_float(x) for x in branch.get('Li', [])]
            Ci_list = [_to_float(x) for x in branch.get('Ci', [])]

            if L0 and L0 > 0 and not np.isinf(L0):
                next_n = next_node()
                components.append((next_comp('L'), current_node, next_n, L0))
                current_node = next_n

            if Cinf and Cinf > 0 and not np.isinf(Cinf):
                next_n = next_node()
                components.append((next_comp('C'), current_node, next_n, Cinf))
                current_node = next_n

            for Li, Ci in zip(Li_list, Ci_list):
                if Li and Ci and Li > 0 and Ci > 0:
                    tap_node = next_node()
                    components.append((next_comp('L'), current_node, tap_node, Li))
                    components.append((next_comp('C'), tap_node, '0', Ci))
                    current_node = tap_node

    elif branch_type == 'shunt':
        # Shunt branch: components from current node to ground
        output_node = current_node  # shunt doesn't advance along the chain

        if foster_form == 1:
            Linf = _to_float(branch.get('Linf', 0))
            C0 = _to_float(branch.get('C0', 0))
            Li_list = [_to_float(x) for x in branch.get('Li', [])]
            Ci_list = [_to_float(x) for x in branch.get('Ci', [])]

            # Shunt inductor (HP case: Linf has value, C0=inf)
            if Linf and Linf > 0 and not np.isinf(Linf):
                components.append((next_comp('L'), current_node, '0', Linf))

            # Shunt capacitor (LP case: C0 has value, Linf=0)
            if C0 and C0 > 0 and not np.isinf(C0):
                components.append((next_comp('C'), current_node, '0', C0))

            for Li, Ci in zip(Li_list, Ci_list):
                if Li and Ci and Li > 0 and Ci > 0:
                    tap_node = next_node()
                    components.append((next_comp('L'), current_node, tap_node, Li))
                    components.append((next_comp('C'), tap_node, '0', Ci))

        elif foster_form == 2:
            L0 = _to_float(branch.get('L0', 0))
            Cinf = _to_float(branch.get('Cinf', 0))
            Li_list = [_to_float(x) for x in branch.get('Li', [])]
            Ci_list = [_to_float(x) for x in branch.get('Ci', [])]

            if L0 and L0 > 0 and not np.isinf(L0):
                components.append((next_comp('L'), current_node, '0', L0))

            if Cinf and Cinf > 0 and not np.isinf(Cinf):
                components.append((next_comp('C'), current_node, '0', Cinf))

            for Li, Ci in zip(Li_list, Ci_list):
                if Li and Ci and Li > 0 and Ci > 0:
                    tap_node = next_node()
                    components.append((next_comp('C'), current_node, tap_node, Ci))
                    components.append((next_comp('L'), tap_node, '0', Li))

        return components, current_node, current_node  # shunt: in == out

    output_node = current_node
    return components, input_node, output_node


def filter_to_netlist(design: FilterDesign, prefix: str = "f1") -> PeripheralNetlist:
    """Convert a FilterDesign into a JC-compatible netlist.

    Parameters
    ----------
    design : FilterDesign
        A designed filter with physical component values.
    prefix : str
        Node name prefix. Default "f1".

    Returns
    -------
    PeripheralNetlist
        With port_map {'input': (1, node), 'output': (2, node)}.
    """
    result = design.transfo_result
    foster_form = result.get('foster_form', design.spec.foster_form)
    series_branches = result.get('series', [])
    shunt_branches = result.get('shunt', [])

    components = []
    node_counter = [0]
    comp_counter = [0]

    # Input port node
    input_node = f"{prefix}_in"

    # Build alternating series/shunt ladder
    current_node = input_node
    s_idx = 0  # series branch index
    h_idx = 0  # shunt branch index

    # Ladder network: series-shunt-series-shunt-...
    total_branches = len(series_branches) + len(shunt_branches)
    for i in range(total_branches):
        if i % 2 == 0 and s_idx < len(series_branches):
            # Series branch
            comps, in_n, out_n = _branch_to_components(
                series_branches[s_idx], 'series', foster_form,
                prefix, node_counter, comp_counter
            )
            # Connect input to current chain position
            comps = _merge_nodes_list(comps, in_n, current_node)
            components.extend(comps)
            current_node = out_n
            s_idx += 1
        elif h_idx < len(shunt_branches):
            # Shunt branch
            comps, in_n, out_n = _branch_to_components(
                shunt_branches[h_idx], 'shunt', foster_form,
                prefix, node_counter, comp_counter
            )
            comps = _merge_nodes_list(comps, in_n, current_node)
            components.extend(comps)
            h_idx += 1

    output_node = current_node

    # Add ports and termination resistors
    port_components = [
        (f'P1_{prefix}', input_node, '0', '1'),
        (f'R1_{prefix}', input_node, '0', 'R_port'),
        (f'P2_{prefix}', output_node, '0', '2'),
        (f'R2_{prefix}', output_node, '0', 'R_port'),
    ]

    all_components = port_components + components

    return PeripheralNetlist(
        components=all_components,
        parameters={'R_port': design.spec.Z0},
        port_map={
            'input': (1, input_node),
            'output': (2, output_node),
        },
        metadata={
            'type': 'filter',
            'response': design.spec.response.value,
            'order': design.spec.order,
            'n_components': len(components),
        },
    )


def multiplexer_to_netlist(design: MultiplexerDesign,
                           prefix: str = "m1") -> PeripheralNetlist:
    """Convert a MultiplexerDesign into a JC-compatible netlist.

    The common port (where all arms join) is Port 1. Each arm's
    output is a subsequent port (2, 3, ...).

    Node naming: common node is '{prefix}_c', arm nodes are
    '{prefix}_{label}_1', '{prefix}_{label}_2', etc.

    Parameters
    ----------
    design : MultiplexerDesign
        A designed multiplexer.
    prefix : str
        Node name prefix. Default "m1".

    Returns
    -------
    PeripheralNetlist
    """
    common_node = f"{prefix}_c"
    components = []
    port_map = {'common': (1, common_node)}
    comp_counter = [0]

    # Common port
    components.append((f'P1_{prefix}', common_node, '0', '1'))
    components.append((f'R1_{prefix}', common_node, '0', 'R_port'))

    for arm_idx, arm_design in enumerate(design.arms):
        arm_label = arm_design.spec.label or f"arm{arm_idx}"
        arm_prefix = f"{prefix}_{arm_label}"
        node_counter = [0]

        result = arm_design.transfo_result
        foster_form = result.get('foster_form', arm_design.spec.foster_form)
        series_branches = result.get('series', [])
        shunt_branches = result.get('shunt', [])

        # Reverse branch order: g-values go g1 (at 50 Ohm port) to g_n (at
        # open common node). Since we build from common outward, place g_n
        # first (at common) and g1 last (at arm port).
        series_branches = list(reversed(series_branches))
        shunt_branches = list(reversed(shunt_branches))

        # Build arm ladder starting from common node toward arm port
        current_node = common_node
        s_idx = 0
        h_idx = 0
        total_branches = len(series_branches) + len(shunt_branches)

        for i in range(total_branches):
            if i % 2 == 0 and s_idx < len(series_branches):
                comps, in_n, out_n = _branch_to_components(
                    series_branches[s_idx], 'series', foster_form,
                    arm_prefix, node_counter, comp_counter
                )
                comps = _merge_nodes_list(comps, in_n, current_node)
                components.extend(comps)
                current_node = out_n
                s_idx += 1
            elif h_idx < len(shunt_branches):
                comps, in_n, out_n = _branch_to_components(
                    shunt_branches[h_idx], 'shunt', foster_form,
                    arm_prefix, node_counter, comp_counter
                )
                comps = _merge_nodes_list(comps, in_n, current_node)
                components.extend(comps)
                h_idx += 1

        # Arm output port
        arm_out_node = current_node
        port_num = arm_idx + 2
        components.append((f'P{port_num}_{prefix}', arm_out_node, '0', str(port_num)))
        components.append((f'R{port_num}_{prefix}', arm_out_node, '0', 'R_port'))
        port_map[arm_label] = (port_num, arm_out_node)

    return PeripheralNetlist(
        components=components,
        parameters={'R_port': design.Z0},
        port_map=port_map,
        metadata={
            'type': 'multiplexer',
            'n_arms': len(design.arms),
            'arm_labels': [a.spec.label for a in design.arms],
            'n_components': len(components),
        },
    )


# ============================================================================
# Section 6: Analysis (standalone peripheral response)
# ============================================================================

def peripheral_response(design_or_netlist, f: np.ndarray,
                        units: str = "GHz") -> dict:
    """Calculate S-parameter response of a peripheral circuit.

    Accepts a FilterDesign, MultiplexerDesign, or the underlying
    transfo_result dict.

    Parameters
    ----------
    design_or_netlist : FilterDesign, MultiplexerDesign, or dict
        The circuit to analyze.
    f : array_like
        Frequency array.
    units : str
        Frequency units ('GHz', 'MHz', 'kHz', 'Hz').

    Returns
    -------
    dict
        For single filter: output of calculate_S21()
        For multiplexer: output of calculate_multiplexer_response()
    """
    if isinstance(design_or_netlist, FilterDesign):
        return calculate_S21(design_or_netlist.transfo_result, f, units=units)
    elif isinstance(design_or_netlist, MultiplexerDesign):
        filter_results = [arm.transfo_result for arm in design_or_netlist.arms]
        return calculate_multiplexer_response(filter_results, f, Z0=design_or_netlist.Z0, units=units)
    elif isinstance(design_or_netlist, dict):
        return calculate_S21(design_or_netlist, f, units=units)
    else:
        raise TypeError(f"Expected FilterDesign, MultiplexerDesign, or dict, got {type(design_or_netlist)}")


def plot_peripheral_response(design_or_netlist,
                             f: Optional[np.ndarray] = None,
                             units: str = "GHz",
                             show: bool = True):
    """Plot S-parameter response of a peripheral circuit.

    Parameters
    ----------
    design_or_netlist : FilterDesign, MultiplexerDesign, or dict
        The circuit to analyze.
    f : array_like, optional
        Frequency array. Auto-generated if None.
    units : str
        Frequency units.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    tuple (fig, axes, response_dict)
    """
    if isinstance(design_or_netlist, FilterDesign):
        return plot_response(design_or_netlist.transfo_result, f=f, units=units, show=show)
    elif isinstance(design_or_netlist, MultiplexerDesign):
        filter_results = [arm.transfo_result for arm in design_or_netlist.arms]
        labels = [arm.spec.label for arm in design_or_netlist.arms]
        return plot_multiplexer_response(filter_results, f=f, Z0=design_or_netlist.Z0,
                                         units=units, labels=labels, show=show)
    elif isinstance(design_or_netlist, dict):
        return plot_response(design_or_netlist, f=f, units=units, show=show)
    else:
        raise TypeError(f"Expected FilterDesign, MultiplexerDesign, or dict, got {type(design_or_netlist)}")


def save_peripheral_netlist(jc_components: list, circuit_parameters: dict,
                           metadata: dict, output_file: str):
    """Save a composed or standalone peripheral netlist to a .py file.

    Uses the same file format as netlist_JC_builder, so the result
    can be loaded directly by julia_wrapper.load_netlist().

    Parameters
    ----------
    jc_components : list
        Component tuples from compose_chain(), filter_to_netlist(), etc.
    circuit_parameters : dict
        Parameter name -> value mapping.
    metadata : dict
        Metadata (device_name, etc.).
    output_file : str
        Output filename (full path, with or without .py extension).
    """
    from .netlist_JC_builder import save_raw_netlist_to_file

    if not output_file.endswith('.py'):
        output_file += '.py'
    save_raw_netlist_to_file(jc_components, circuit_parameters, metadata, output_file)


# ============================================================================
# Section 7: Topology composition
# ============================================================================

def _merge_nodes_list(components: List[ComponentTuple],
                      old_node: str, new_node: str) -> List[ComponentTuple]:
    """Replace all occurrences of old_node with new_node in component list."""
    if old_node == new_node:
        return components
    return [
        (name, new_node if n1 == old_node else n1,
               new_node if n2 == old_node else n2, val)
        for name, n1, n2, val in components
    ]


def _remove_port_and_termination(components: List[ComponentTuple],
                                 port_number: Union[int, str]) -> List[ComponentTuple]:
    """Remove a port component and its associated termination resistor."""
    port_str = str(port_number)
    # Find the port component to identify its node
    port_node = None
    for name, n1, n2, val in components:
        if name.startswith('P') and str(val) == port_str:
            port_node = n1
            break

    if port_node is None:
        return components

    # Remove port component and its resistor at the same node
    result = []
    for name, n1, n2, val in components:
        is_port = name.startswith('P') and str(val) == port_str
        is_resistor = name.startswith('R') and n1 == port_node and n2 == '0'
        if not is_port and not is_resistor:
            result.append((name, n1, n2, val))
    return result


def _renumber_ports(components: List[ComponentTuple],
                    port_order: Dict[int, int]) -> List[ComponentTuple]:
    """Renumber port components according to mapping {old_num: new_num}."""
    result = []
    for name, n1, n2, val in components:
        if name.startswith('P') and str(val) in [str(k) for k in port_order]:
            old_num = int(val) if isinstance(val, (int, float)) else int(val)
            new_num = port_order.get(old_num, old_num)
            new_name = name.replace(f'P{old_num}', f'P{new_num}')
            result.append((new_name, n1, n2, str(new_num)))
        elif name.startswith('R'):
            # Check if this R corresponds to a renumbered port
            for old_num, new_num in port_order.items():
                if f'R{old_num}' in name:
                    new_name = name.replace(f'R{old_num}', f'R{new_num}')
                    result.append((new_name, n1, n2, val))
                    break
            else:
                result.append((name, n1, n2, val))
        else:
            result.append((name, n1, n2, val))
    return result


def compose_chain(blocks: list,
                  connections: Optional[List[Tuple[str, str]]] = None,
                  Z0: float = 50.0) -> Tuple[List[ComponentTuple], Dict[str, float], Dict]:
    """Compose an arbitrary chain of peripherals and TWPAs.

    Each block is either:
    - A PeripheralNetlist (filter, diplexer, multiplexer)
    - A dict with 'jc_components', 'circuit_parameters', 'metadata'
      (TWPA netlist as loaded or built)

    Adjacent blocks are stitched by:
    1. Stripping ALL ports and their termination resistors from every block
    2. Merging nodes at junctions (so connected blocks share a node)
    3. Concatenating all stripped components
    4. Adding fresh ports and resistors at the surviving external nodes

    Parameters
    ----------
    blocks : list
        Alternating list of PeripheralNetlist and TWPA netlist dicts.
    connections : list of (output_role, input_role) tuples, optional
        Specifies which port of block[i] connects to which port of block[i+1].
        Default: ('output', 'input') for filters, inferred for multiplexers.
        Length must be len(blocks) - 1.
    Z0 : float
        Reference impedance for port terminations.

    Returns
    -------
    tuple of (jc_components, circuit_parameters, metadata)
    """
    if len(blocks) < 2:
        raise ValueError("Need at least 2 blocks to compose")

    # --- Normalize all blocks to: (components, parameters, port_map, metadata) ---
    def _normalize_block(block, idx):
        if isinstance(block, PeripheralNetlist):
            return (list(block.components), dict(block.parameters),
                    dict(block.port_map), dict(block.metadata))
        elif isinstance(block, dict) and 'jc_components' in block:
            comps = list(block['jc_components'])
            params = dict(block.get('circuit_parameters', {}))
            meta = dict(block.get('metadata', {}))
            # Discover ports from components
            port_map = {}
            for name, n1, n2, val in comps:
                if name.startswith('P'):
                    try:
                        pnum = int(val)
                        port_map[f'port{pnum}'] = (pnum, n1)
                    except (ValueError, TypeError):
                        pass
            # Alias first port as 'input', last as 'output'
            port_nums = sorted(port_map.keys(), key=lambda k: port_map[k][0])
            if len(port_nums) >= 1:
                port_map['input'] = port_map[port_nums[0]]
            if len(port_nums) >= 2:
                port_map['output'] = port_map[port_nums[-1]]
            return (comps, params, port_map, meta)
        else:
            raise TypeError(f"Block {idx}: expected PeripheralNetlist or dict, got {type(block)}")

    normalized = [_normalize_block(b, i) for i, b in enumerate(blocks)]

    # --- Default connections ---
    if connections is None:
        connections = []
        for i in range(len(blocks) - 1):
            _, _, pm_out, _ = normalized[i]
            _, _, pm_in, _ = normalized[i + 1]
            out_role = 'output' if 'output' in pm_out else 'lp'
            in_role = 'input' if 'input' in pm_in else 'common'
            connections.append((out_role, in_role))

    # --- Step 1: Identify all port nodes and which ports are consumed ---
    # Build a set of (block_idx, port_num) for consumed (junction) ports
    consumed_ports = set()
    junctions = []  # (out_node, in_node) pairs to merge

    for conn_idx, (out_role, in_role) in enumerate(connections):
        _, _, pm_out, _ = normalized[conn_idx]
        _, _, pm_in, _ = normalized[conn_idx + 1]
        out_port_num, out_node = pm_out[out_role]
        in_port_num, in_node = pm_in[in_role]
        consumed_ports.add((conn_idx, out_port_num))
        consumed_ports.add((conn_idx + 1, in_port_num))
        junctions.append((out_node, in_node))

    # Collect all port nodes per block (to identify which resistors are port terminations)
    block_port_nodes = []  # block_idx -> set of port nodes
    for block_idx, (_, _, port_map, _) in enumerate(normalized):
        port_nodes = set()
        for role, (pnum, node) in port_map.items():
            port_nodes.add(node)
        block_port_nodes.append(port_nodes)

    # --- Step 2: Strip ALL ports and port-termination resistors from every block ---
    stripped_blocks = []
    for block_idx, (comps, params, port_map, meta) in enumerate(normalized):
        port_nodes = block_port_nodes[block_idx]
        stripped = []
        for name, n1, n2, val in comps:
            # Skip port components
            if name.startswith('P') and n2 == '0':
                try:
                    int(val)
                    continue  # it's a port, skip
                except (ValueError, TypeError):
                    pass
            # Skip resistors that are port terminations (Rxx at port node to ground)
            if (name.startswith('R') and n1 in port_nodes and n2 == '0'
                    and (val == 'R_port' or (isinstance(val, (int, float)) and val == Z0))):
                continue
            stripped.append((name, n1, n2, val))
        stripped_blocks.append(stripped)

    # --- Step 3: Merge nodes at junctions ---
    node_rename = {}
    for out_node, in_node in junctions:
        if out_node != in_node:
            # Follow chains: if in_node was already renamed, follow it
            target = out_node
            while target in node_rename:
                target = node_rename[target]
            node_rename[in_node] = target

    def _apply_renames(comps):
        result = []
        for name, n1, n2, val in comps:
            n1 = node_rename.get(n1, n1)
            n2 = node_rename.get(n2, n2)
            result.append((name, n1, n2, val))
        return result

    # --- Step 4: Collect surviving external ports ---
    external_ports = []  # (block_idx, role, port_num, node)
    for block_idx, (_, _, port_map, _) in enumerate(normalized):
        for role, (pnum, node) in port_map.items():
            # Skip alias roles ('input'/'output') that duplicate a port{N} entry
            if role in ('input', 'output'):
                is_alias = any(
                    r != role and p == pnum for r, (p, _) in port_map.items()
                )
                if is_alias:
                    continue
            if (block_idx, pnum) not in consumed_ports:
                renamed_node = node_rename.get(node, node)
                external_ports.append((block_idx, role, pnum, renamed_node))

    # Sort by block index, then original port number
    external_ports.sort(key=lambda x: (x[0], x[2]))

    # Assign sequential port numbers
    port_roles = {}
    port_assignments = {}  # (block_idx, old_pnum) -> new_num
    for new_num, (bidx, role, old_pnum, node) in enumerate(external_ports, start=1):
        port_roles[role] = new_num
        port_assignments[(bidx, old_pnum)] = (new_num, node)

    # --- Step 5: Assemble output in logical order ---
    # The netlist should read from device edges inward, then back out:
    #   P1 R1 HP_arm(port→common) | P2 R2 LP_arm(port→common) | TWPA | common→arm R3 P3 | common→arm R4 P4
    #
    # A block on the LEFT side of a connection (provides out_role) connects
    # forward: its components are reversed so they read port → junction.
    # A block on the RIGHT side (provides in_role) connects backward:
    # its components stay in natural order junction → port.

    # Determine which blocks connect forward
    forward_blocks = set()
    for conn_idx in range(len(connections)):
        forward_blocks.add(conn_idx)  # block at conn_idx connects forward

    all_comps = []
    all_params = {}
    all_meta = {'blocks': []}

    for block_idx, stripped in enumerate(stripped_blocks):
        renamed = _apply_renames(stripped)

        # For input-side peripheral blocks, reverse each arm individually
        # so they read port → junction, while preserving arm order.
        if block_idx in forward_blocks and isinstance(blocks[block_idx], PeripheralNetlist):
            pnet = blocks[block_idx]
            if pnet.metadata.get('type') == 'multiplexer':
                # Split components into per-arm groups by node prefix,
                # reverse each group, reassemble in original arm order.
                arm_labels = pnet.metadata.get('arm_labels', [])
                prefix = list(pnet.port_map.values())[0][1].split('_')[0]  # e.g. 'm1'
                arm_comps = {label: [] for label in arm_labels}
                other_comps = []
                for comp in renamed:
                    name, n1, n2, val = comp
                    assigned = False
                    for label in arm_labels:
                        arm_prefix = f"{prefix}_{label}_"
                        if arm_prefix in n1 or arm_prefix in n2:
                            arm_comps[label].append(comp)
                            assigned = True
                            break
                    if not assigned:
                        other_comps.append(comp)
                # Reassemble: each arm reversed, in original arm order
                renamed = []
                for label in arm_labels:
                    renamed.extend(reversed(arm_comps[label]))
                renamed.extend(other_comps)  # any components not in an arm (shouldn't happen)
            else:
                renamed = list(reversed(renamed))

        # Collect this block's external ports: {node: new_port_num}
        block_ports = {}
        for (bidx, old_pnum), (new_num, node) in port_assignments.items():
            if bidx == block_idx:
                block_ports[node] = new_num

        if not block_ports:
            # No external ports for this block (e.g. TWPA in the middle)
            all_comps.extend(renamed)
        else:
            is_forward = block_idx in forward_blocks
            port_nodes = set(block_ports.keys())

            if is_forward:
                # Input side: port is the starting point of each arm.
                # Insert port+resistor BEFORE the first component touching the port node.
                inserted_ports = set()
                for name, n1, n2, val in renamed:
                    for pnode in port_nodes:
                        if pnode in (n1, n2) and pnode not in inserted_ports:
                            new_num = block_ports[pnode]
                            all_comps.append((f'P{new_num}', pnode, '0', str(new_num)))
                            all_comps.append((f'R{new_num}', pnode, '0', 'R_port'))
                            inserted_ports.add(pnode)
                    all_comps.append((name, n1, n2, val))
            else:
                # Output side: port is the endpoint of each arm.
                # Insert port+resistor AFTER the last component touching the port node.
                # First pass: find the last index where each port node appears.
                last_touch = {}  # pnode -> last index in renamed
                for idx, (name, n1, n2, val) in enumerate(renamed):
                    for pnode in port_nodes:
                        if pnode in (n1, n2):
                            last_touch[pnode] = idx

                # Second pass: emit components, inserting port after the last touch.
                inserted_ports = set()
                for idx, (name, n1, n2, val) in enumerate(renamed):
                    all_comps.append((name, n1, n2, val))
                    for pnode in port_nodes:
                        if pnode not in inserted_ports and last_touch.get(pnode) == idx:
                            new_num = block_ports[pnode]
                            all_comps.append((f'P{new_num}', pnode, '0', str(new_num)))
                            all_comps.append((f'R{new_num}', pnode, '0', 'R_port'))
                            inserted_ports.add(pnode)

            # Safety: add any ports whose nodes didn't appear in components
            inserted_ports = inserted_ports if 'inserted_ports' in dir() else set()
            for pnode, new_num in block_ports.items():
                if pnode not in inserted_ports:
                    all_comps.append((f'P{new_num}', pnode, '0', str(new_num)))
                    all_comps.append((f'R{new_num}', pnode, '0', 'R_port'))

        all_params.update(normalized[block_idx][1])
        all_meta['blocks'].append(normalized[block_idx][3])

    all_meta['n_ports'] = len(external_ports)
    all_meta['port_roles'] = port_roles

    return all_comps, all_params, all_meta


def stitch_filter_twpa_filter(input_filter: Optional[PeripheralNetlist],
                              twpa_netlist: dict,
                              output_filter: Optional[PeripheralNetlist],
                              Z0: float = 50.0) -> Tuple[List[ComponentTuple], Dict[str, float], Dict]:
    """Stitch: input_filter -> TWPA -> output_filter.

    Either filter can be None (no filter on that side).

    Parameters
    ----------
    input_filter : PeripheralNetlist or None
    twpa_netlist : dict
        TWPA netlist with 'jc_components', 'circuit_parameters', 'metadata'.
    output_filter : PeripheralNetlist or None
    Z0 : float

    Returns
    -------
    tuple of (jc_components, circuit_parameters, metadata)
    """
    blocks = []
    if input_filter is not None:
        blocks.append(input_filter)
    blocks.append(twpa_netlist)
    if output_filter is not None:
        blocks.append(output_filter)

    if len(blocks) == 1:
        # Just the TWPA, no filters
        return (twpa_netlist['jc_components'],
                twpa_netlist.get('circuit_parameters', {}),
                twpa_netlist.get('metadata', {}))

    return compose_chain(blocks, Z0=Z0)


def stitch_diplexer_twpa_chain(diplexers: List[PeripheralNetlist],
                               twpas: List[dict],
                               Z0: float = 50.0) -> Tuple[List[ComponentTuple], Dict[str, float], Dict]:
    """Stitch: diplexer -> TWPA -> diplexer -> TWPA -> ... -> diplexer.

    Expects len(diplexers) == len(twpas) + 1.

    Parameters
    ----------
    diplexers : list of PeripheralNetlist
        N+1 diplexer netlists.
    twpas : list of dict
        N TWPA netlists.
    Z0 : float

    Returns
    -------
    tuple of (jc_components, circuit_parameters, metadata)
    """
    if len(diplexers) != len(twpas) + 1:
        raise ValueError(f"Need len(diplexers) == len(twpas) + 1, "
                         f"got {len(diplexers)} diplexers and {len(twpas)} TWPAs")

    # Interleave: diplexer, twpa, diplexer, twpa, ..., diplexer
    blocks = []
    for i, twpa in enumerate(twpas):
        blocks.append(diplexers[i])
        blocks.append(twpa)
    blocks.append(diplexers[-1])

    return compose_chain(blocks, Z0=Z0)
