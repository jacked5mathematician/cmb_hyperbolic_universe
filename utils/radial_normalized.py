"""
Paper-faithful normalized radial functions X_k^ℓ(ρ).

This module implements the correctly normalized radial eigenfunctions as described in
Cornish & Spergel (1999), matching the mathematical requirements:

1. Radial ODE: X'' + 2coth(ρ)X' - ℓ(ℓ+1)/sinh²(ρ) X + (k²+1)X = 0
2. Asymptotic form: X_k^ℓ(ρ) ~ cos(kρ+φ)/sinh(ρ)  => X*sinh(ρ) has O(1) amplitude
3. ℓ=0 closed form: X_k^0(ρ) = sin(kρ)/sinh(ρ)  => X*sinh(ρ) = sin(kρ) (amplitude 1)

    CRITICAL FINDING:
    Phi_nu_l(k, ℓ, ρ) from special_functions.py is a valid solution to the radial ODE,
    but has WRONG NORMALIZATION by a factor of k√2.
    
    Verified via ℓ=0 closed form:
        Phi_nu_l(k, 0, ρ) * sinh(ρ) = sin(kρ) / (k√2)  (ratio to sin(kρ) is 1/(k√2))
        Target: X_k^0(ρ) = sin(kρ) / sinh(ρ)  => X*sinh(ρ) = sin(kρ)
    
    Analytic fix: X_paper(k, ℓ, ρ) = k√2 * Phi_nu_l(k, ℓ, ρ)
"""
from __future__ import annotations

import logging
from typing import Tuple
import numpy as np

from .special_functions import Phi_nu_l

LOGGER = logging.getLogger(__name__)

# Cache for normalization factors (now analytic, but kept for consistency with tests)
_normalization_cache: dict[Tuple[float, int], float] = {}

def _compute_normalization_factor(k: float, ell: int) -> float:
    """
    Compute normalization factor R(k) such that X_paper = R(k) * Phi_nu_l matches paper's X_k^ℓ.
    
    ANALYTIC DERIVATION (from ℓ=0 closed form):
        Target: X_k^0(ρ) = sin(kρ)/sinh(ρ)  => X*sinh(ρ) = sin(kρ)  [amplitude = k]
        Actual: Phi_nu_l(k, 0, ρ) * sinh(ρ) = sin(kρ) / (k√2)  [amplitude = k/(k√2) = 1/√2]
        
        Ratio: [Phi*sinh] / [sin(kρ)] = 1/(k√2)
        
        To get amplitude = k: X*sinh = k * [Phi*sinh * k√2] = sin(kρ)
        
        Therefore: R(k) = k√2
    
    This is INDEPENDENT of ℓ because the radial ODE normalization factor is k-dependent only.
    """
    return float(k * np.sqrt(2.0))


def get_normalization_factor(k: float, ell: int) -> float:
    """
    Get cached normalization factor R(k, ℓ), computing if necessary.
    
    Args:
        k: Wavenumber
        ell: Angular momentum quantum number
    
    Returns:
        R(k, ℓ): Normalization factor
    """
    # Quantize k to avoid cache explosion (round to 3 decimals)
    k_quantized = round(float(k), 3)
    ell_int = int(ell)
    
    key = (k_quantized, ell_int)
    
    if key not in _normalization_cache:
        R = _compute_normalization_factor(k_quantized, ell_int)
        _normalization_cache[key] = R
        LOGGER.debug("Computed normalization: R(k=%s, ell=%s) = %s", k_quantized, ell_int, R)
    
    return _normalization_cache[key]


def X_normalized(k: float, ell: int, rho: float) -> float:
    """
    Paper-faithful normalized radial eigenfunction X_k^ℓ(ρ) matching Cornish & Spergel (1999).
    
    Mathematical properties:
        1. Satisfies radial ODE: X'' + 2coth(ρ)X' - ℓ(ℓ+1)/sinh²(ρ) X + (k²+1)X = 0
        2. Asymptotic form: X_k^ℓ(ρ) ~ cos(kρ+φ)/sinh(ρ)  => X*sinh(ρ) has O(1) amplitude
        3. ℓ=0 closed form: X_k^0(ρ) = sin(kρ)/sinh(ρ)  => X*sinh(ρ) = sin(kρ)
    
    Normalization:
        X_k^ℓ(ρ) = k√2 * Phi_nu_l(k, ℓ, ρ)
    
    This analytic scaling ensures the ℓ=0 closed form is exactly satisfied.
    
    Args:
        k: Wavenumber
        ell: Angular momentum quantum number
        rho: Radial coordinate
    
    Returns:
        X_k^ℓ(ρ): Paper-faithful radial eigenfunction
    """
    # Get raw Phi value
    Phi = Phi_nu_l(k, ell, rho)
    
    # Get normalization factor (cached, but now analytic: R = k√2)
    R = get_normalization_factor(k, ell)
    
    # Return normalized value: X_paper = R * Phi = k√2 * Phi
    return float(R * Phi)


def clear_normalization_cache():
    """Clear the normalization factor cache."""
    global _normalization_cache
    _normalization_cache.clear()
    LOGGER.info("Cleared normalization cache.")


def get_cache_size() -> int:
    """Return the number of cached normalization factors."""
    return len(_normalization_cache)


def X_paper(k: float, ell: int, rho: float) -> float:
    """
    Explicit alias for X_normalized to emphasize paper-faithful normalization.
    
    Use this name when you want to be crystal-clear that you're using the
    Cornish & Spergel (1999) normalization convention, NOT the raw Phi_nu_l.
    """
    return X_normalized(k, ell, rho)


__all__ = [
    "X_normalized",
    "X_paper",
    "get_normalization_factor",
    "clear_normalization_cache",
    "get_cache_size",
]
