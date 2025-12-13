from __future__ import annotations

import logging
from collections import deque
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .transformations import apply_so31_action, project_to_klein, klein_to_pseudo_spherical

LOGGER = logging.getLogger(__name__)
MATRIX_ROUND_DECIMALS = 8  # Precision used for deduplicating group elements


def _load_generators(manifold_name: str) -> List[np.ndarray]:
    try:
        import snappy  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("SnapPy not available (%s); falling back to synthetic ghosts.", exc)
        return []

    try:
        manifold = snappy.Manifold(manifold_name)
        domain = manifold.dirichlet_domain()
        pairing_mats = domain.pairing_matrices()
    except Exception as exc:  # pragma: no cover - SnapPy errors
        LOGGER.warning("Failed to load pairing matrices for %s (%s); using synthetic ghosts.", manifold_name, exc)
        return []

    generators = []
    for mat in pairing_mats:
        arr = np.array(mat, dtype=float).reshape(4, 4)
        generators.append(arr)
        try:
            generators.append(np.linalg.inv(arr))
        except np.linalg.LinAlgError:
            pass
    return generators


def _synthetic_images(base_point: Sequence[float], rho_min: float, rho_max: float, count: int) -> List[Tuple[float, float, float]]:
    rng = np.random.default_rng(abs(hash(tuple(base_point))) % (2**32))
    rhos = rng.uniform(rho_min, rho_max, size=count)
    thetas = rng.uniform(0.0, np.pi, size=count)
    phis = rng.uniform(-np.pi, np.pi, size=count)
    return [(float(r), float(t), float(p)) for r, t, p in zip(rhos, thetas, phis)]


def enumerate_ghost_images(
    manifold_name: str,
    base_points: Iterable[Sequence[float]],
    rho_min: float,
    rho_max: float,
    rho_margin: float = 0.5,
    min_images: int = 10,
    max_word_length: int = 6,
    tolerance: float = 1e-8,
) -> List[List[Tuple[float, float, float]]]:
    """
    Enumerate ghost images for each base point using group words up to max_word_length.
    Falls back to a synthetic generator if SnapPy data are unavailable.
    """
    generators = _load_generators(manifold_name)
    points_images: List[List[Tuple[float, float, float]]] = []

    for point in base_points:
        if not generators:
            images = _synthetic_images(point, rho_min, rho_max, max(min_images, 12))
            points_images.append([img for img in images if rho_min <= img[0] <= rho_max])
            continue

        queue = deque([(np.eye(4), 0)])
        seen = {tuple(np.round(np.eye(4).flatten(), MATRIX_ROUND_DECIMALS))}
        images: List[Tuple[float, float, float]] = []

        while queue:
            mat, depth = queue.popleft()
            transformed = apply_so31_action(mat, np.asarray(point, dtype=float))
            klein = project_to_klein(transformed)
            rho, theta, phi = klein_to_pseudo_spherical([klein])[0]

            if rho_min <= rho <= rho_max + rho_margin:
                images.append((float(rho), float(theta), float(phi)))
            if rho > rho_max + rho_margin:
                continue

            if depth >= max_word_length:
                continue

            for gen in generators:
                new_mat = mat @ gen
                key = tuple(np.round(new_mat.flatten(), MATRIX_ROUND_DECIMALS))
                if key in seen:
                    continue
                seen.add(key)
                queue.append((new_mat, depth + 1))

            if len(images) >= min_images and depth > 0:
                # We have enough images; stop expanding further
                break

        if len(images) < min_images:
            LOGGER.warning(
                "Only %s images collected for point %s; dropping point.", len(images), point
            )
            continue

        # Keep images strictly inside requested window
        filtered = [img for img in images if rho_min <= img[0] <= rho_max]
        if len(filtered) < min_images:
            LOGGER.warning(
                "After filtering, only %s images remain for point %s (rho_min=%s, rho_max=%s).",
                len(filtered), point, rho_min, rho_max,
            )
            continue
        points_images.append(filtered)

    return points_images


__all__ = ["enumerate_ghost_images"]
