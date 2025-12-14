from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
from scipy.optimize import brentq

from .radial_normalized import X_paper

LOGGER = logging.getLogger(__name__)
ENVELOPE_L_SCALE = 0.5  # Heuristic factor to widen fallback rho_max with increasing L


def _rho_turning_point(k: float, ell: int) -> float:
    """Paper equation (2.8) from Cornish & Spergel (1999) turning point (transition to oscillatory regime): asinh(sqrt(ell(ell+1))/k)."""
    k = float(k)
    if k <= 0:
        raise ValueError("k must be positive")
    ell = int(ell)
    return float(np.arcsinh(np.sqrt(ell * (ell + 1)) / k))


def _abs_radial_envelope(k: float, ell: int, rho: float) -> float:
    """
    Paper-inspired envelope for |X_k^ell(rho) * sinh(rho)|.

    For rho >= rho_0 (turning point), equation (2.8) gives
        X_k^ell(rho) ~ cos(k * rho + phi_0) / sinh(rho)
    Substituting phi_0 = -k * rho_0 collapses cos(k * rho + phi_0) to cos(k * (rho - rho_0));
    multiplying by sinh(rho) follows that phase.

    We ignore the rho << rho_0 behavior and suppress crossings before rho_0 by
    returning +inf in that region.
    """
    rho0 = _rho_turning_point(k, ell)
    if rho < rho0:
        return float("inf")
    phase = float(k) * (float(rho) - rho0)  # phi_0 = -k * rho_0
    return float(abs(np.cos(phase)))


def rho_turning_point(k: float, ell: int) -> float:
    """Public wrapper for the paper's rho_0 turning point."""
    return _rho_turning_point(k, ell)


def abs_radial_envelope(k: float, ell: int, rho: float) -> float:
    """Public wrapper for the paper-inspired |X_k^ell(rho) * sinh(rho)| envelope."""
    return _abs_radial_envelope(k, ell, rho)


def _find_crossing(
    k: float,
    ell: int,
    threshold: float,
    rho_cap: float,
    step: float,
    rho_start: float,
) -> float | None:
    """
    Find the first rho >= rho_start where |X_k^ell(rho)*sinh(rho)| <= threshold.
    
    PAPER-FAITHFUL (Part 3 of cutoff fix):
    Uses actual X_normalized(k, ell, rho) with proper asymptotic normalization,
    not the broken envelope approximation.
    
    Method: Deterministic 'first crossing AFTER first maximum' scanning.
    1. Find the first maximum of |X*sinh| after turning point
    2. Then find where it drops below threshold
    
    This ensures we get the physically meaningful cutoff, not an early node.

    Important: we intentionally do NOT start at rho=0 because near-origin behavior
    can trivially satisfy the threshold and yield unusably small rho_max.
    """
    rho = float(rho_start)
    
    # Compute turning point to respect paper's physical constraint
    rho0 = _rho_turning_point(k, ell)
    
    # Start after turning point if rho_start is before it
    if rho < rho0:
        rho = rho0 + 0.01  # Small offset after turning point
    
    # Phase 1: Find first maximum
    # Scan until we see a local maximum (derivative changes from + to -)
    max_found = False
    prev_val = None
    prev_prev_val = None
    
    while rho <= rho_cap and not max_found:
        X_val = X_paper(k, ell, rho)
        val = abs(X_val * np.sinh(rho))
        
        # Detect local maximum: prev_prev < prev_val > val
        if prev_prev_val is not None and prev_val is not None:
            if prev_prev_val < prev_val and prev_val > val:
                # Found local maximum at previous rho
                max_found = True
                # Continue from current position
                break
        
        prev_prev_val = prev_val
        prev_val = val
        rho += step
    
    if not max_found:
        # No maximum found - just look for first crossing
        LOGGER.warning(
            "No local maximum found for (k=%.3f, ell=%d) up to rho_cap=%.1f. "
            "Falling back to first threshold crossing.",
            float(k), int(ell), rho_cap
        )
        rho = float(rho_start)
        if rho < rho0:
            rho = rho0 + 0.01
    
    # Phase 2: Find first crossing below threshold after maximum
    # Bracket the root first, then refine with Brent's method
    prev_rho = None
    prev_val = None
    
    while rho <= rho_cap:
        X_val = X_paper(k, ell, rho)
        val = abs(X_val * np.sinh(rho))
        
        # Check for bracket: val crosses threshold
        if prev_val is not None and prev_val > threshold >= val:
            # Found bracket [prev_rho, rho] where function crosses threshold
            # Refine with Brent's method
            def residual(r):
                X = X_paper(k, ell, r)
                return abs(X * np.sinh(r)) - threshold
            
            try:
                rho_refined = brentq(residual, prev_rho, rho, xtol=1e-6)
                return float(rho_refined)
            except ValueError:
                # Brentq failed, return coarse crossing
                return rho
        
        if val <= threshold:
            return rho
        
        prev_rho = rho
        prev_val = val
        rho += step
    
    return None


def compute_rho_cutoffs(
    k: float,
    L: int,
    l_min: int,
    threshold: float = 0.25,
    rho_cap: float = 120.0,
    step: float = 0.05,
    # New robustness parameters:
    rho_start: float = 0.75,
    rho_max_floor: float = 1.0,
    fallback_mode: str = "fixed_rho",
    verbose: bool = True,
) -> Tuple[float, float, bool]:
    """
    Compute rho cutoffs following the paper-inspired policy.

    Primary method tries to find the first rho where:
        |X_k^ell(rho) * sinh(rho)| <= threshold
    for ell=l_min (rho_min) and ell=L (rho_max).

    PAPER-FAITHFUL (Parts 3-4 of cutoff fix):
    - Uses actual X_normalized with correct asymptotic normalization
    - Deterministic 'first sign change after turning point' scanning
    - Fallback modes when no root exists (marked NON-PAPER in logs)

    Args:
        k: Wavenumber
        L: Maximum angular momentum (paper: L = floor(k) + 10)
        l_min: Minimum angular momentum for rho_min cutoff
        threshold: Paper value is 0.25
        rho_cap: Maximum search radius
        step: Grid spacing for root search
        rho_start: Start search at this rho (avoid near-origin artifacts)
        rho_max_floor: Minimum acceptable rho_max (fallback if below)
        fallback_mode: How to handle no-root cases (Part 4)
            - "fixed_rho": Use rho = arcsinh(4) + 0.5*L (NON-PAPER)
            - "relative_envelope": Scale threshold by fitted amplitude (NON-PAPER)

    Returns:
        (rho_min, rho_max, fallback_used)

    Robustness adjustments:
    - We start searching at rho_start (default 0.75) to avoid pathological near-zero crossings.
    - If rho_max is found but is < rho_max_floor (default 1.0), we treat that as unusable
      and fall back according to fallback_mode.
    """
    if L < l_min:
        raise ValueError("L must be >= l_min")
    if rho_start < 0:
        raise ValueError("rho_start must be >= 0")
    if rho_cap <= rho_start:
        raise ValueError("rho_cap must be > rho_start")
    if rho_max_floor <= 0:
        raise ValueError("rho_max_floor must be > 0")
    if fallback_mode not in ["fixed_rho", "relative_envelope"]:
        raise ValueError(f"Unknown fallback_mode: {fallback_mode}")

    rho_min = _find_crossing(k, l_min, threshold, rho_cap, step, rho_start=rho_start)
    rho_max = _find_crossing(k, L, threshold, rho_cap, step, rho_start=rho_start)
    fallback_used = False

    # Fallback strategies (Part 4 - NON-PAPER)
    def _fallback_rho_max_fixed() -> float:
        """Fixed heuristic: rho = arcsinh(4) + 0.5*L"""
        rho_guess = float(np.arcsinh(4.0) + ENVELOPE_L_SCALE * L)
        return float(max(rho_guess, rho_max_floor))
    
    def _fallback_rho_max_relative() -> float:
        """Relative envelope: use threshold as fraction of fitted amplitude"""
        # This would require computing max value and finding where it drops to threshold
        # For now, use a simple heuristic based on asymptotic decay
        rho_guess = float(np.arcsinh(1.0 / threshold))
        return float(max(rho_guess, rho_guess + ENVELOPE_L_SCALE * L, rho_max_floor))

    if rho_min is None:
        rho_min = 0.0
        fallback_used = True

    # If rho_max not found OR found but too small, fallback.
    if rho_max is None or float(rho_max) < rho_max_floor:
        if rho_max is not None:
            LOGGER.warning(
                "[NON-PAPER] rho_max found too small (rho_max=%.4f < floor=%.4f) for (k=%.3f, L=%d). "
                "Falling back to mode='%s'.",
                float(rho_max), rho_max_floor, float(k), int(L), fallback_mode,
            )
        else:
            LOGGER.warning(
                "[NON-PAPER] No rho_max root found for (k=%.3f, L=%d) up to rho_cap=%.1f. "
                "Falling back to mode='%s'.",
                float(k), int(L), rho_cap, fallback_mode,
            )
        
        fallback_used = True
        if fallback_mode == "fixed_rho":
            rho_max = _fallback_rho_max_fixed()
        else:  # relative_envelope
            rho_max = _fallback_rho_max_relative()

    # Ensure ordering and a non-degenerate window
    rho_min = float(max(0.0, rho_min))
    rho_max = float(rho_max)
    if rho_max <= rho_min:
        rho_max = rho_min + max(step, 1e-3)
        fallback_used = True

    if not (0.0 <= rho_min < rho_max):
        raise ValueError(
            f"Invalid rho window computed for (k={k}, L={L}, l_min={l_min}). "
            f"Got rho_min={rho_min}, rho_max={rho_max}."
        )
    
    # Log diagnostics (as requested in spec)
    if verbose:
        # Compute f(rho_max) to verify threshold
        X_at_max = X_paper(k, L, rho_max)
        f_rho_max = abs(X_at_max * np.sinh(rho_max))
        
        status = "PAPER-FAITHFUL" if not fallback_used else f"FALLBACK({fallback_mode})"
        
        LOGGER.info(
            "Cutoff for k=%.3f, L=%d: rho_max=%.4f, f(rho_max)=%.4f, threshold=%.2f, status=%s",
            float(k), int(L), rho_max, f_rho_max, threshold, status
        )

    return rho_min, rho_max, fallback_used


__all__ = ["compute_rho_cutoffs", "rho_turning_point", "abs_radial_envelope"]
