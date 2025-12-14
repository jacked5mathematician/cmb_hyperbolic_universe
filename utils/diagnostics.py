from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .points import compute_dirichlet_images
from .transformations import poincare_distance


def _ensure_array(values: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Expected Nx3 array of points")
    return arr


def diagnose_base_points(
    poincare_points: Sequence[Sequence[float]],
    pseudo_points: Sequence[Sequence[float]],
    manifold_name: str,
    group_elements: Sequence[np.ndarray],
    sampling_metadata: Dict[str, Any],
    tolerance: float = 1e-6,
    dirichlet_images: Sequence[np.ndarray] | None = None,
    max_dirichlet_images: int = 500,
) -> Dict[str, Any]:
    """
    Analyze sampled base points and measure how well they satisfy the Dirichlet inequality.

    If `dirichlet_images` is provided it should be a sequence of Poincaré points for γ·p0.
    Otherwise they will be computed from `group_elements`.
    """
    points = _ensure_array(poincare_points)
    pseudos = _ensure_array(pseudo_points)

    total_points = len(points)
    dirichlet_count = int(sampling_metadata.get("dirichlet_count", total_points))
    dirichlet_count = min(dirichlet_count, total_points)
    fallback_count = total_points - dirichlet_count

    if dirichlet_images is not None:
        gamma_points = [np.asarray(pt, dtype=float) for pt in dirichlet_images]
    else:
        gamma_points = compute_dirichlet_images(group_elements)
    if max_dirichlet_images is not None and len(gamma_points) > max_dirichlet_images:
        gamma_points = gamma_points[:max_dirichlet_images]

    base_point = np.zeros(3, dtype=float)
    diagnostics: List[Dict[str, Any]] = []
    invalid = 0
    margins: List[float] = []

    for idx, (pt, pseudo) in enumerate(zip(points, pseudos)):
        entry: Dict[str, Any] = {
            "index": idx,
            "source": "dirichlet" if idx < dirichlet_count else "fallback",
            "poincare": pt.tolist(),
            "rho": float(pseudo[0]),
            "theta": float(pseudo[1]),
            "phi": float(pseudo[2]),
            "norm": float(np.linalg.norm(pt)),
        }
        if gamma_points:
            d0 = poincare_distance(pt, base_point)
            min_margin = math.inf
            closest_idx = None
            valid = True
            for g_idx, other in enumerate(gamma_points):
                d_other = poincare_distance(pt, other)
                margin = d_other - d0
                if margin < min_margin:
                    min_margin = margin
                    closest_idx = g_idx
                if d0 > d_other + tolerance:
                    valid = False
            entry["dirichlet_margin"] = float(min_margin)
            entry["closest_gamma_index"] = closest_idx
            entry["valid"] = valid
            if math.isfinite(min_margin):
                margins.append(float(min_margin))
            if not valid:
                invalid += 1
        else:
            entry["dirichlet_margin"] = None
            entry["closest_gamma_index"] = None
            entry["valid"] = None
        diagnostics.append(entry)

    summary = {
        "total_points": total_points,
        "dirichlet_points": dirichlet_count,
        "fallback_points": fallback_count,
        "invalid_points": invalid if gamma_points else None,
        "min_margin": min(margins) if margins else None,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "has_dirichlet_reference": bool(gamma_points),
    }

    return {
        "manifold": manifold_name,
        "tolerance": tolerance,
        "summary": summary,
        "sampling_metadata": sampling_metadata,
        "dirichlet_images": [pt.tolist() for pt in gamma_points],
        "points": diagnostics,
    }


def write_point_diagnostics(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
