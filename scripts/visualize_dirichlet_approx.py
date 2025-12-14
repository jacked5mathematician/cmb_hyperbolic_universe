#!/usr/bin/env python
"""
Visualize the *approximate* Dirichlet domain used by this repo's sampler.

Important: This does NOT render the exact SnapPy Dirichlet polyhedron. Instead it
visualizes the same finite-word inequality used in `utils/points.sample_points_in_dirichlet_domain`:

  d(x, p0) <= d(x, γ(p0))   for all γ in a finite set (word_depth-limited)

We do this by computing the "margin field":

  margin(x) = min_γ ( d(x, γ(p0)) - d(x, p0) )

Points with margin(x) >= 0 satisfy the inequality; margin(x) ≈ 0 is an approximation
to the fundamental domain boundary (for the chosen word depth).

Usage:
  python scripts/visualize_dirichlet_approx.py \
    --manifold "m003(-2,3)" --word-depth 3 \
    --diagnostics output_values/point_diag.json \
    --output output_values/domain_approx.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure repo-root imports work when executed as a file
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ghosts import get_group_elements  # noqa: E402
from utils.points import compute_dirichlet_images  # noqa: E402


def _load_points_from_diagnostics(path: Path) -> Tuple[np.ndarray, list, list]:
    with open(path, "r") as fh:
        payload = json.load(fh)
    pts = np.array([p["poincare"] for p in payload["points"]], dtype=float)
    sources = [p.get("source") for p in payload["points"]]
    valids = [p.get("valid") for p in payload["points"]]
    return pts, sources, valids


def _poincare_distance_to_origin(points: np.ndarray) -> np.ndarray:
    """Vectorized hyperbolic distance d(x,0) in Poincaré ball."""
    r2 = np.sum(points * points, axis=1)
    r2 = np.clip(r2, 0.0, 1.0 - 1e-12)
    cosh_arg = 1.0 + (2.0 * r2) / (1.0 - r2)
    return np.arccosh(cosh_arg)


def _poincare_distance(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    """
    Vectorized distances between points (M,3) and others (B,3).

    Returns array (M,B).
    """
    p2 = np.sum(points * points, axis=1, keepdims=True)  # (M,1)
    o2 = np.sum(others * others, axis=1, keepdims=True).T  # (1,B)
    diff2 = np.sum((points[:, None, :] - others[None, :, :]) ** 2, axis=2)  # (M,B)
    denom = (1.0 - p2) * (1.0 - o2)
    denom = np.clip(denom, 1e-15, None)
    cosh_arg = 1.0 + (2.0 * diff2) / denom
    cosh_arg = np.clip(cosh_arg, 1.0, None)
    return np.arccosh(cosh_arg)


def compute_margin_field(
    grid: np.ndarray,
    gamma_p0: np.ndarray,
    block: int = 128,
) -> np.ndarray:
    """
    Compute margin(x) = min_γ ( d(x, γ(p0)) - d(x, 0) ) for each x in grid.
    """
    if len(grid) == 0:
        return np.array([], dtype=float)
    d0 = _poincare_distance_to_origin(grid)  # (M,)

    if len(gamma_p0) == 0:
        # If we have no γ points, the inequality is undefined; return NaNs.
        return np.full(len(grid), np.nan, dtype=float)

    min_margin = np.full(len(grid), np.inf, dtype=float)
    for start in range(0, len(gamma_p0), block):
        block_pts = gamma_p0[start : start + block]
        d_other = _poincare_distance(grid, block_pts)  # (M,B)
        margin_block = d_other - d0[:, None]
        min_margin = np.minimum(min_margin, np.min(margin_block, axis=1))
    return min_margin


def _nearest_seed_indices(points: np.ndarray, seeds: np.ndarray, block: int = 128) -> np.ndarray:
    """Return argmin_j d(points[i], seeds[j]) for each point."""
    if len(points) == 0 or len(seeds) == 0:
        return np.zeros(len(points), dtype=int)
    best_idx = np.zeros(len(points), dtype=int)
    best_d = np.full(len(points), np.inf, dtype=float)
    for start in range(0, len(seeds), block):
        block_seeds = seeds[start : start + block]
        d = _poincare_distance(points, block_seeds)  # (M,B)
        local_best = np.min(d, axis=1)
        local_arg = np.argmin(d, axis=1) + start
        improved = local_best < best_d
        best_d[improved] = local_best[improved]
        best_idx[improved] = local_arg[improved]
    return best_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the word-depth Dirichlet approximation.")
    parser.add_argument("--manifold", type=str, required=True, help='Manifold name, e.g. "m003(-2,3)"')
    parser.add_argument("--word-depth", type=int, default=3, help="Word depth used to build γ(p0) set")
    parser.add_argument("--diagnostics", type=Path, default=None, help="Optional point_diagnostics.json to overlay points")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--grid", type=int, default=31, help="Grid resolution per axis (odd recommended)")
    parser.add_argument("--radius", type=float, default=0.90, help="Grid cutoff radius inside the unit ball")
    parser.add_argument("--boundary-eps", type=float, default=0.02, help="|margin| threshold for boundary point cloud")
    parser.add_argument("--max-gamma", type=int, default=800, help="Max γ(p0) points to use (speed control)")
    parser.add_argument("--compare-depth", type=int, default=None, help="Optional second depth to compare boundaries")
    parser.add_argument(
        "--boundary-max-points",
        type=int,
        default=30000,
        help="Max boundary points to plot (subsamples for speed/readability).",
    )
    parser.add_argument(
        "--tiling-seeds",
        type=int,
        default=25,
        help="Number of orbit seeds to show in the tiling slice (includes p0 as seed 0).",
    )
    parser.add_argument(
        "--tiling",
        action="store_true",
        help="Add a z=0 tiling visualization (Voronoi cells of p0 orbit under chosen word depth).",
    )
    parser.add_argument(
        "--overlay-sampled",
        action="store_true",
        help="Overlay sampled points from --diagnostics (recommended).",
    )
    args = parser.parse_args()

    group_elements, fallback = get_group_elements(args.manifold, args.word_depth)
    if fallback or not group_elements:
        raise SystemExit(
            "No SnapPy group elements available; cannot visualize Dirichlet approximation. "
            "Install SnapPy / use --require-snappy runs."
        )

    gamma_p0 = np.array(compute_dirichlet_images(group_elements), dtype=float)
    if args.max_gamma and len(gamma_p0) > args.max_gamma:
        gamma_p0 = gamma_p0[: args.max_gamma]

    lin = np.linspace(-args.radius, args.radius, args.grid)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    inside = np.sum(grid * grid, axis=1) < (args.radius * args.radius)
    grid_in = grid[inside]

    margins = compute_margin_field(grid_in, gamma_p0)
    boundary_mask = np.isfinite(margins) & (np.abs(margins) <= args.boundary_eps)
    boundary = grid_in[boundary_mask]
    if args.boundary_max_points is not None and len(boundary) > args.boundary_max_points:
        rng = np.random.default_rng(0)
        boundary = boundary[rng.choice(len(boundary), size=args.boundary_max_points, replace=False)]

    boundary2 = None
    if args.compare_depth is not None:
        group2, fallback2 = get_group_elements(args.manifold, args.compare_depth)
        if not fallback2 and group2:
            gamma2 = np.array(compute_dirichlet_images(group2), dtype=float)
            if args.max_gamma and len(gamma2) > args.max_gamma:
                gamma2 = gamma2[: args.max_gamma]
            margins2 = compute_margin_field(grid_in, gamma2)
            boundary2 = grid_in[np.isfinite(margins2) & (np.abs(margins2) <= args.boundary_eps)]
            if args.boundary_max_points is not None and len(boundary2) > args.boundary_max_points:
                rng = np.random.default_rng(1)
                boundary2 = boundary2[rng.choice(len(boundary2), size=args.boundary_max_points, replace=False)]

    show_tiling = bool(args.tiling)
    fig = plt.figure(figsize=(13, 6))

    # 3D boundary + points overlay
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    if len(boundary) > 0:
        ax3d.scatter(boundary[:, 0], boundary[:, 1], boundary[:, 2], s=1, alpha=0.25, label=f"boundary d={args.word_depth}")
    if boundary2 is not None and len(boundary2) > 0:
        ax3d.scatter(boundary2[:, 0], boundary2[:, 1], boundary2[:, 2], s=1, alpha=0.20, label=f"boundary d={args.compare_depth}")

    if args.overlay_sampled and args.diagnostics is not None and args.diagnostics.exists():
        pts, sources, valids = _load_points_from_diagnostics(args.diagnostics)
        for (x, y, z), src, valid in zip(pts, sources, valids):
            if src == "fallback":
                c = "tab:orange"
                m = "o"
            else:
                c = "tab:blue" if (valid is True or valid is None) else "tab:red"
                m = "o" if (valid is True or valid is None) else "x"
            ax3d.scatter(x, y, z, color=c, marker=m, s=25, alpha=0.9)

    # Unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_wireframe(xs, ys, zs, color="gray", alpha=0.15, linewidth=0.5)

    ax3d.set_title("Approx Dirichlet boundary (Poincaré ball)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.legend(loc="upper left")

    ax2d = fig.add_subplot(1, 2, 2)
    xx2, yy2 = np.meshgrid(lin, lin, indexing="ij")
    slice_pts = np.column_stack([xx2.ravel(), yy2.ravel(), np.zeros(xx2.size)])
    inside2 = np.sum(slice_pts * slice_pts, axis=1) < (args.radius * args.radius)
    slice_in = slice_pts[inside2]

    if show_tiling:
        # Build seeds: p0 plus nearby orbit points (limit by Euclidean norm so plot remains legible).
        p0 = np.zeros((1, 3), dtype=float)
        gamma_sorted = gamma_p0[np.argsort(np.sum(gamma_p0 * gamma_p0, axis=1))]
        seeds = np.vstack([p0, gamma_sorted[: max(0, args.tiling_seeds - 1)]])

        labels = _nearest_seed_indices(slice_in, seeds)
        sc = ax2d.scatter(slice_in[:, 0], slice_in[:, 1], c=labels, s=8, cmap="tab20", alpha=0.9)
        ax2d.scatter(seeds[:, 0], seeds[:, 1], color="k", s=35, marker=".", alpha=0.9, label="orbit seeds")
        ax2d.set_title("z=0 slice: orbit Voronoi tiling (finite depth)")
        ax2d.legend(loc="upper left")
    else:
        slice_margin = compute_margin_field(slice_in, gamma_p0)
        sc = ax2d.scatter(slice_in[:, 0], slice_in[:, 1], c=slice_margin, s=8, cmap="coolwarm", alpha=0.9)
        plt.colorbar(sc, ax=ax2d, shrink=0.8, label="min_γ (d(x,γp0) - d(x,p0))")
        ax2d.set_title("z=0 slice: Dirichlet margin field")

    if args.overlay_sampled and args.diagnostics is not None and args.diagnostics.exists():
        pts, sources, valids = _load_points_from_diagnostics(args.diagnostics)
        close = np.abs(pts[:, 2]) < (2.0 * args.radius / args.grid)
        for (x, y, z), src, valid in zip(pts[close], np.array(sources)[close], np.array(valids)[close]):
            if src == "fallback":
                c = "tab:orange"
            else:
                c = "tab:blue" if (valid is True or valid is None) else "tab:red"
            ax2d.scatter(x, y, color=c, edgecolor="k", linewidth=0.5, s=55)

    t = np.linspace(0, 2 * np.pi, 200)
    ax2d.plot(np.cos(t), np.sin(t), color="gray", alpha=0.3)
    ax2d.set_aspect("equal", "box")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")

    fig.suptitle(
        f"{args.manifold} Dirichlet approximation (word_depth={args.word_depth}"
        + (f", compare={args.compare_depth}" if args.compare_depth is not None else "")
        + (", tiling slice" if show_tiling else ", margin slice")
        + ")"
    )
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
