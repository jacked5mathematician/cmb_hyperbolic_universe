from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .transformations import poincare_to_pseudo_spherical

LOGGER = logging.getLogger(__name__)


def _sample_in_ball(n_points: int, rng: np.random.Generator, radius: float = 0.85) -> np.ndarray:
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
    fallback_radius: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points inside the Dirichlet domain. If SnapPy is unavailable, fall back to
    rejection sampling in the Poincaré ball.
    """
    rng = np.random.default_rng(seed)
    try:
        import snappy  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("SnapPy not available (%s); using fallback sampling.", exc)
        points = _sample_in_ball(n_points, rng, fallback_radius)
        return points, poincare_to_pseudo_spherical(points)

    try:
        manifold = snappy.Manifold(manifold_name)
        domain = manifold.dirichlet_domain()
        verts = np.array([list(v["position"]) for v in domain.vertex_list(details=True)], dtype=float)
        faces = domain.face_list()
    except Exception as exc:  # pragma: no cover - SnapPy errors
        LOGGER.warning("Failed to construct Dirichlet domain for %s (%s); using fallback sampling.", manifold_name, exc)
        points = _sample_in_ball(n_points, rng, fallback_radius)
        return points, poincare_to_pseudo_spherical(points)

    min_corner = verts.min(axis=0)
    max_corner = verts.max(axis=0)
    accepted: List[np.ndarray] = []
    attempts = 0
    while len(accepted) < n_points and attempts < n_points * 50:
        attempts += 1
        candidate = rng.uniform(min_corner, max_corner)
        inside = True
        for face in faces:
            face_vertices = verts[face["vertex_indices"]]
            if len(face_vertices) < 3:
                continue
            v1 = face_vertices[1] - face_vertices[0]
            v2 = face_vertices[2] - face_vertices[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            if np.dot(candidate - face_vertices[0], normal) > 0:
                inside = False
                break
        if inside:
            accepted.append(candidate)

    if len(accepted) < n_points:
        LOGGER.warning(
            "Only accepted %s/%s points via domain sampling; filling with fallback.",
            len(accepted),
            n_points,
        )
        extra = _sample_in_ball(n_points - len(accepted), rng, fallback_radius)
        points = np.vstack([accepted, extra])
    else:
        points = np.array(accepted)

    return points, poincare_to_pseudo_spherical(points)


__all__ = ["sample_points_in_dirichlet_domain"]
