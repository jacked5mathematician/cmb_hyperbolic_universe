#!/usr/bin/env python
"""
Visualize sampled base points (and optional Dirichlet references) inside the Poincaré ball.

Usage:
    python scripts/visualize_point_sampling.py --diagnostics path/to/point_diagnostics.json --output plot.png

The diagnostics file is produced by running main.py with --dump-point-diagnostics <path>.
"""
import argparse
import json
from pathlib import Path

import matplotlib

if "MPLBACKEND" not in matplotlib.rcParams:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_diagnostics(path: Path) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def plot_points(payload: dict, output: Path | None = None, show_gamma: bool = True):
    pts = np.array([p["poincare"] for p in payload["points"]], dtype=float)
    sources = [p["source"] for p in payload["points"]]
    valids = [p["valid"] for p in payload["points"]]

    colors = []
    markers = []
    for src, valid in zip(sources, valids):
        if src == "fallback":
            colors.append("tab:orange")
        else:
            colors.append("tab:blue" if valid or valid is None else "tab:red")
        markers.append("o" if valid or valid is None else "x")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for (x, y, z), c, m in zip(pts, colors, markers):
        ax.scatter(x, y, z, color=c, marker=m, alpha=0.8)

    # Unit sphere for reference (Poincaré ball boundary)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.15, linewidth=0.5)

    if show_gamma:
        gamma_points = payload.get("dirichlet_images")
        if gamma_points:
            gamma = np.array(gamma_points, dtype=float)
            ax.scatter(gamma[:, 0], gamma[:, 1], gamma[:, 2], color="k", marker=".", alpha=0.5, label="γ·p0")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Base points for {payload.get('manifold', '')}")
    ax.legend(loc="upper left")
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize sampled base points in the Poincaré ball.")
    parser.add_argument("--diagnostics", type=Path, required=True, help="Path to point_diagnostics.json")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (default: show window)")
    parser.add_argument("--no-gamma", action="store_true", help="Do not plot gamma·p0 reference points")
    args = parser.parse_args()

    payload = load_diagnostics(args.diagnostics)
    plot_points(payload, output=args.output, show_gamma=not args.no_gamma)


if __name__ == "__main__":
    main()
