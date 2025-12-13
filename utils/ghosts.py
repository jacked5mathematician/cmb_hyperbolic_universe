from __future__ import annotations

import logging
from collections import deque
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .transformations import apply_so31_action, project_to_klein, klein_to_pseudo_spherical

LOGGER = logging.getLogger(__name__)
MATRIX_ROUND_DECIMALS = 8  # Precision used for deduplicating group elements
MAX_IMAGES_DEFAULT = 200
GhostImages = List[List[Tuple[float, float, float]]]


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


def enumerate_group_elements(generators: List[np.ndarray], max_depth: int) -> List[np.ndarray]:
    """
    Enumerate unique group elements up to the given word depth (BFS order).
    """
    if not generators:
        return []

    queue = deque([(np.eye(4), 0)])
    seen = set()
    elements: List[np.ndarray] = []

    while queue:
        mat, depth = queue.popleft()
        key = tuple(np.round(mat.flatten(), MATRIX_ROUND_DECIMALS))
        if key in seen:
            continue
        seen.add(key)
        elements.append(mat)
        if depth >= max_depth:
            continue
        for gen in generators:
            queue.append((mat @ gen, depth + 1))
    return elements


def get_group_elements(manifold_name: str, max_depth: int) -> tuple[List[np.ndarray], bool]:
    """
    Load generators for a manifold and enumerate words up to max_depth.

    Returns a tuple of (elements, fallback_used).
    """
    generators = _load_generators(manifold_name)
    if not generators:
        return [], True
    return enumerate_group_elements(generators, max_depth), False


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
    max_images: int = MAX_IMAGES_DEFAULT,
    group_elements: List[np.ndarray] | None = None,
    return_metadata: bool = False,
) -> GhostImages | tuple[GhostImages, dict]:
    """
    Enumerate ghost images for each base point using group words up to max_word_length.
    Falls back to a synthetic generator if SnapPy data are unavailable.
    """
    fallback_used = False
    if group_elements is None:
        group_elements, fallback_used = get_group_elements(manifold_name, max_word_length)
    else:
        group_elements = list(group_elements)
        fallback_used = fallback_used or (len(group_elements) == 0)
    points_images: List[List[Tuple[float, float, float]]] = []

    for point in base_points:
        if not group_elements:
            images = _synthetic_images(point, rho_min, rho_max, min(max_images, max(min_images, 12)))
            points_images.append([img for img in images if rho_min <= img[0] <= rho_max])
            continue

        images: List[Tuple[float, float, float]] = []
        seen_images = set()

        for mat in group_elements:
            transformed = apply_so31_action(mat, np.asarray(point, dtype=float))
            klein = project_to_klein(transformed)
            rho, theta, phi = klein_to_pseudo_spherical([klein])[0]

            if rho_min <= rho <= rho_max + rho_margin:
                key = tuple(np.round([rho, theta, phi], MATRIX_ROUND_DECIMALS))
                if key in seen_images:
                    continue
                seen_images.add(key)
                images.append((float(rho), float(theta), float(phi)))
            if len(images) >= max_images:
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

    if return_metadata:
        return points_images, {"fallback_used": fallback_used, "total_group_elements": len(group_elements)}
    return points_images


__all__ = ["enumerate_ghost_images", "get_group_elements", "enumerate_group_elements"]
