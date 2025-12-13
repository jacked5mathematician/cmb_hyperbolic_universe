from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from .special_functions import Phi_nu_l_cached
from .conventions import k_to_nu

LOGGER = logging.getLogger(__name__)
ENVELOPE_L_SCALE = 0.5  # Heuristic factor to widen fallback rho_max with increasing L


def _abs_radial_envelope(k: float, ell: int, rho: float) -> float:
    """Compute |X_k^ell(rho) * sinh(rho)| using the current radial implementation."""
    nu = k_to_nu(k)
    return float(abs(Phi_nu_l_cached(nu, ell, rho) * np.sinh(rho)))


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

    Important: we intentionally do NOT start at rho=0 because near-origin behavior
    can trivially satisfy the threshold and yield unusably small rho_max.
    """
    prev_val = None
    rho = float(rho_start)
    while rho <= rho_cap:
        val = _abs_radial_envelope(k, ell, rho)
        if prev_val is not None and prev_val > threshold >= val:
            return rho
        if val <= threshold:
            return rho
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
) -> Tuple[float, float, bool]:
    """
    Compute rho cutoffs following the paper-inspired policy.

    Primary method tries to find the first rho where:
        |X_k^ell(rho) * sinh(rho)| <= threshold
    for ell=l_min (rho_min) and ell=L (rho_max).

    Robustness adjustments:
    - We start searching at rho_start (default 0.75) to avoid pathological near-zero crossings.
    - If rho_max is found but is < rho_max_floor (default 1.0), we treat that as unusable
      and fall back to an envelope heuristic.
    """
    if L < l_min:
        raise ValueError("L must be >= l_min")
    if rho_start < 0:
        raise ValueError("rho_start must be >= 0")
    if rho_cap <= rho_start:
        raise ValueError("rho_cap must be > rho_start")
    if rho_max_floor <= 0:
        raise ValueError("rho_max_floor must be > 0")

    rho_min = _find_crossing(k, l_min, threshold, rho_cap, step, rho_start=rho_start)
    rho_max = _find_crossing(k, L, threshold, rho_cap, step, rho_start=rho_start)
    fallback_used = False

    # Envelope fallback (paper-inspired heuristic)
    def _fallback_rho_max() -> float:
        rho_guess = float(np.arcsinh(1.0 / threshold))  # envelope ~ 1/sinh(rho)
        return float(max(rho_guess, rho_guess + ENVELOPE_L_SCALE * L, rho_max_floor))

    if rho_min is None:
        rho_min = 0.0
        fallback_used = True

    # If rho_max not found OR found but too small, fallback.
    if rho_max is None or float(rho_max) < rho_max_floor:
        if rho_max is not None:
            LOGGER.warning(
                "rho_max found too small (rho_max=%.4f < floor=%.4f) for (k=%.3f, L=%d). "
                "Falling back to envelope heuristic.",
                float(rho_max), rho_max_floor, float(k), int(L),
            )
        fallback_used = True
        rho_max = _fallback_rho_max()

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

    return rho_min, rho_max, fallback_used


__all__ = ["compute_rho_cutoffs"]