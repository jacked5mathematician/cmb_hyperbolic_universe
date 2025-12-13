from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from .special_functions import Phi_nu_l_cached
from .conventions import k_to_nu

LOGGER = logging.getLogger(__name__)
ENVELOPE_L_SCALE = 0.5  # Heuristic factor to widen fallback rho_max with increasing L


def _find_crossing(k: float, ell: int, threshold: float, rho_cap: float, step: float) -> float | None:
    nu = k_to_nu(k)
    prev_val = None
    rho = 0.0
    while rho <= rho_cap:
        val = abs(Phi_nu_l_cached(nu, ell, rho) * np.sinh(rho))
        if prev_val is not None and prev_val > threshold >= val:
            return rho
        if val <= threshold:
            return rho
        prev_val = val
        rho += step
    return None


def compute_rho_cutoffs(k: float, L: int, l_min: int, threshold: float = 0.25,
                        rho_cap: float = 120.0, step: float = 0.05) -> Tuple[float, float, bool]:
    """
    Compute rho cutoffs following the paper-faithful policy.

    Primary method scans for the first crossing of |X_k^ell(rho) * sinh(rho)| <= threshold
    for ell = l_min and ell = L. If no crossing is found up to rho_cap, an envelope-based
    heuristic fallback is used.

    Returns
    -------
    rho_min : float
        First crossing for ell=l_min (Eq. 2.9 style).
    rho_max : float
        First crossing for ell=L.
    fallback_used : bool
        True if the envelope-based heuristic was applied.
    """
    if L < l_min:
        raise ValueError("L must be >= l_min")

    rho_min = _find_crossing(k, l_min, threshold, rho_cap, step)
    rho_max = _find_crossing(k, L, threshold, rho_cap, step)
    fallback_used = False

    if rho_min is None or rho_max is None:
        fallback_used = True
        rho_guess = float(np.arcsinh(1.0 / threshold))
        rho_env = max(rho_guess, rho_guess + ENVELOPE_L_SCALE * L)
        LOGGER.warning(
            "Falling back to envelope heuristic for rho cutoffs (k=%s, L=%s, l_min=%s): "
            "rho_guess=%s -> rho_env=%s",
            k, L, l_min, rho_guess, rho_env,
        )
        if rho_min is None:
            rho_min = 0.0
        if rho_max is None:
            rho_max = rho_env

    if rho_max <= rho_min:
        rho_max = rho_min + max(step, 1e-3)

    if rho_min < 0 or rho_max <= rho_min:
        raise ValueError(
            f"Invalid rho window computed for (k={k}, L={L}, l_min={l_min}). "
            f"Got rho_min={rho_min}, rho_max={rho_max}."
        )

    return float(rho_min), float(rho_max), fallback_used


__all__ = ["compute_rho_cutoffs"]
