"""
Plot the paper's cutoff envelope |X_k^ell(rho) * sinh(rho)| and mark crossings.

Example:
    python -m scripts.debug_cutoff --k 1.0 --L 10 --l-min 5 --output cutoff_debug.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.cutoffs import (
    ENVELOPE_L_SCALE,
    _abs_radial_envelope,
    _rho_turning_point,
    compute_rho_cutoffs,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug rho cutoff envelope and crossings.")
    parser.add_argument("--k", type=float, required=True, help="k value to analyze")
    parser.add_argument("--L", type=int, required=True, help="Maximum ell (paper: 10 + floor(k))")
    parser.add_argument("--l-min", type=int, default=5, help="Minimum ell for rho_min (paper: 5)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Crossing threshold")
    parser.add_argument("--rho-cap", type=float, default=10.0, help="Maximum rho to plot/search")
    parser.add_argument("--step", type=float, default=0.01, help="Rho sampling step for the plot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cutoff_debug.png"),
        help="Output plot path (default: cutoff_debug.png)",
    )
    return parser


def _evaluate_envelope(k: float, ell: int, rhos: np.ndarray) -> np.ndarray:
    vals = np.array([_abs_radial_envelope(k, ell, float(r)) for r in rhos], dtype=float)
    vals[~np.isfinite(vals)] = np.nan
    return vals


def main() -> None:
    args = _build_parser().parse_args()

    rho_min, rho_max, fallback = compute_rho_cutoffs(
        k=args.k,
        L=args.L,
        l_min=args.l_min,
        threshold=args.threshold,
        rho_cap=args.rho_cap,
    )

    rhos = np.arange(0.0, args.rho_cap + args.step, args.step)
    env_lmin = _evaluate_envelope(args.k, args.l_min, rhos)
    env_L = _evaluate_envelope(args.k, args.L, rhos)
    rho0_lmin = _rho_turning_point(args.k, args.l_min)
    rho0_L = _rho_turning_point(args.k, args.L)

    plt.figure(figsize=(9, 5))
    plt.plot(rhos, env_lmin, label=f"|X_k^{args.l_min}|sinh|", alpha=0.9)
    plt.plot(rhos, env_L, label=f"|X_k^{args.L}|sinh|", alpha=0.9)
    plt.axhline(args.threshold, color="red", linestyle="--", label="threshold")
    plt.axvline(rho_min, color="purple", linestyle=":", label="rho_min")
    plt.axvline(rho_max, color="green", linestyle=":", label="rho_max")
    plt.axvline(rho0_lmin, color="purple", linestyle="--", alpha=0.6, label=r"$\\rho_0(\\ell_{min})$")
    plt.axvline(rho0_L, color="green", linestyle="--", alpha=0.6, label=r"$\\rho_0(L)$")

    plt.title(
        rf"Cutoff debug: k={args.k:.3f}, L={args.L}, l_min={args.l_min}, "
        rf"fallback={fallback}, L scale={ENVELOPE_L_SCALE}"
    )
    plt.xlabel(r"$\\rho$")
    plt.ylabel(r"$|X_k^\\ell(\\rho)\\,\\sinh(\\rho)|$")
    finite_vals = np.concatenate(
        [env_lmin[np.isfinite(env_lmin)], env_L[np.isfinite(env_L)], np.array([args.threshold])]
    )
    ymax = finite_vals.max() * 1.1 if finite_vals.size else 1.0
    plt.ylim(0, ymax)
    plt.legend()
    plt.grid(True, alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"Saved cutoff debug plot to {args.output} (fallback_used={fallback})")
    print(f"rho_min={rho_min:.4f}, rho_max={rho_max:.4f}, rho0_lmin={rho0_lmin:.4f}, rho0_L={rho0_L:.4f}")


if __name__ == "__main__":
    main()
