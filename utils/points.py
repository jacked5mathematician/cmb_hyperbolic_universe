from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .ghosts import get_group_elements
from .transformations import (
    apply_so31_action,
    klein_to_poincare,
    poincare_distance,
    poincare_to_pseudo_spherical,
    project_to_klein,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_FALLBACK_RADIUS = 0.85  # Conservative radius to keep well inside the Poincaré ball
MAX_ATTEMPT_MULTIPLIER = 50  # Try up to this multiple of n_points before falling back
PointSamples = Tuple[np.ndarray, np.ndarray]


def _sample_in_ball(n_points: int, rng: np.random.Generator, radius: float = DEFAULT_FALLBACK_RADIUS) -> np.ndarray:
    points = []
    while len(points) < n_points:
        candidate = rng.uniform(-radius, radius, size=3)
        if np.linalg.norm(candidate) < radius:
            points.append(candidate)
    return np.array(points)


def sample_points_in_dirichlet_domain(
    manifold_name: str,
    n_points: int,
    seed: int | None = None,
    fallback_radius: float = DEFAULT_FALLBACK_RADIUS,
    word_depth: int = 3,
    tolerance: float = 1e-6,
    return_metadata: bool = False,
) -> PointSamples | Tuple[np.ndarray, np.ndarray, dict]:
    """
    Sample points that satisfy the Dirichlet-domain inequality d(x, p0) <= d(x, γ(p0))
    for a finite set of group elements γ (words up to `word_depth`). Falls back to
    Poincaré-ball sampling if generators are unavailable.
    """
    rng = np.random.default_rng(seed)
    metadata = {
        "dirichlet_checked": False,
        "fallback_used": False,
        "word_depth": word_depth,
        "group_elements": 0,
    }

    group_elements, generators_fallback = get_group_elements(manifold_name, word_depth)
    metadata["fallback_used"] = metadata["fallback_used"] or generators_fallback
    metadata["group_elements"] = len(group_elements)

    if not group_elements:
        LOGGER.warning("No group elements available; falling back to ball sampling.")
        points = _sample_in_ball(n_points, rng, fallback_radius)
        metadata["fallback_used"] = True
        pseudo = poincare_to_pseudo_spherical(points)
        return (points, pseudo, metadata) if return_metadata else (points, pseudo)

    base_point = np.zeros(3, dtype=float)
    gamma_p0 = []
    for mat in group_elements:
        transformed = apply_so31_action(mat, base_point)
        klein = project_to_klein(transformed)
        poincare = klein_to_poincare([klein])[0]
        if np.linalg.norm(poincare) >= 1.0:
            continue
        gamma_p0.append(poincare)

    def in_dirichlet_domain(candidate: np.ndarray) -> bool:
        d0 = poincare_distance(candidate, base_point)
        for other in gamma_p0:
            if d0 > poincare_distance(candidate, other) + tolerance:
                return False
        return True

    accepted: List[np.ndarray] = []
    attempts = 0
    radius = min(fallback_radius, 0.85)
    while len(accepted) < n_points and attempts < n_points * MAX_ATTEMPT_MULTIPLIER:
        attempts += 1
        candidate = rng.uniform(-radius, radius, size=3)
        if np.linalg.norm(candidate) >= radius:
            continue
        if not in_dirichlet_domain(candidate):
            continue
        accepted.append(candidate)

    if len(accepted) < n_points:
        LOGGER.warning(
            "Only accepted %s/%s points via Dirichlet sampling; filling with fallback.",
            len(accepted),
            n_points,
        )
        extra = _sample_in_ball(n_points - len(accepted), rng, fallback_radius)
        points = np.vstack([accepted, extra])
        metadata["fallback_used"] = True
    else:
        points = np.array(accepted)
    metadata["dirichlet_checked"] = True
    pseudo = poincare_to_pseudo_spherical(points)
    return (points, pseudo, metadata) if return_metadata else (points, pseudo)


__all__ = ["sample_points_in_dirichlet_domain"]
