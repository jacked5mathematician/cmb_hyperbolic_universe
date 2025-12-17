#!/usr/bin/env python3
"""Combine per-chunk outputs (produced by main.py --k-chunk-index/--k-num-chunks)
into a single rho-adaptive comparison report.

Intended workflow on HPC:
- For each mode in {paper, adaptive_images, adaptive_rank}:
    run main.py with the same k-min/k-max/num-k/n-points and with
    --k-num-chunks N and --k-chunk-index i for i=0..N-1.
  This writes outputs under:
      <base>/<mode>/chunk_<i>/spectrum.npz

- After all jobs finish, run this combiner locally or on the head node:
      python scripts/combine_rho_adaptive_chunks.py --base-dir <base>

It produces:
- REPORT_RHO_ADAPTIVE.md (in current working dir by default)

Notes:
- This script is intentionally file-based and does not rerun expensive computation.
- It assumes each chunk writes a non-overlapping subset of k_values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


K_BAND_EDGES = [1.0, 3.0, 5.0, 7.0, 9.0, 10.0]


@dataclass
class ModeResult:
    mode: str
    k_values: np.ndarray
    chi2_values: np.ndarray
    rho_max_values: np.ndarray
    num_images: np.ndarray
    sigma_min: np.ndarray
    tau_values: np.ndarray
    cond_values: np.ndarray
    numerical_rank: np.ndarray
    frac_below_5tau: np.ndarray

    @property
    def success_mask(self) -> np.ndarray:
        return self.chi2_values < 1.0

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.success_mask)) if len(self.chi2_values) else 0.0

    @property
    def mean_chi2(self) -> float:
        ok = self.success_mask
        if not np.any(ok):
            return float("inf")
        return float(np.mean(self.chi2_values[ok]))


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def _get(arrs: Dict[str, np.ndarray], key: str, like: np.ndarray) -> np.ndarray:
    if key in arrs:
        return arrs[key]
    return np.zeros_like(like)


def load_mode_chunks(mode_dir: Path) -> ModeResult:
    """Load and concatenate all chunk_<i>/spectrum.npz under a given mode dir."""
    chunk_dirs = sorted([p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("chunk_")])
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk_* directories under {mode_dir}")

    k_list: List[np.ndarray] = []
    chi2_list: List[np.ndarray] = []
    rho_list: List[np.ndarray] = []
    img_list: List[np.ndarray] = []
    smin_list: List[np.ndarray] = []
    tau_list: List[np.ndarray] = []
    cond_list: List[np.ndarray] = []
    rank_list: List[np.ndarray] = []
    frac5_list: List[np.ndarray] = []

    for cd in chunk_dirs:
        npz = cd / "spectrum.npz"
        if not npz.exists():
            raise FileNotFoundError(f"Missing {npz}")
        arrs = _load_npz(npz)

        k = arrs["k_values"]
        chi2 = arrs["chi2_rank_1"]

        k_list.append(k)
        chi2_list.append(chi2)
        rho_list.append(_get(arrs, "rho_max", k))
        img_list.append(_get(arrs, "images_per_point", k))
        smin_list.append(_get(arrs, "sigma_min", k))
        tau_list.append(_get(arrs, "tau", k))
        cond_list.append(_get(arrs, "condition_number", k))
        rank_list.append(_get(arrs, "numerical_rank", k))
        frac5_list.append(_get(arrs, "frac_below_5tau", k))

    k_all = np.concatenate(k_list)
    order = np.argsort(k_all)

    def cat(xs: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs)[order]

    return ModeResult(
        mode=mode_dir.name,
        k_values=k_all[order],
        chi2_values=cat(chi2_list),
        rho_max_values=cat(rho_list),
        num_images=cat(img_list),
        sigma_min=cat(smin_list),
        tau_values=cat(tau_list),
        cond_values=cat(cond_list),
        numerical_rank=cat(rank_list),
        frac_below_5tau=cat(frac5_list),
    )


def band_stats(result: ModeResult, k_lo: float, k_hi: float):
    mask = (result.k_values >= k_lo) & (result.k_values < k_hi)
    if k_hi == K_BAND_EDGES[-1]:
        mask = (result.k_values >= k_lo) & (result.k_values <= k_hi)
    if np.sum(mask) == 0:
        return None

    ok = result.success_mask[mask]

    return {
        "n": int(np.sum(mask)),
        "success": float(np.mean(ok)),
        "mean_chi2": float(np.mean(result.chi2_values[mask][ok])) if np.any(ok) else float("inf"),
        "mean_rho": float(np.mean(result.rho_max_values[mask])),
        "mean_images": float(np.mean(result.num_images[mask])),
        "mean_sigma_min": float(np.mean(result.sigma_min[mask])),
        "mean_tau": float(np.mean(result.tau_values[mask])),
        "floor_fail": float(np.mean(result.sigma_min[mask] <= result.tau_values[mask])),
        "mean_cond": float(np.mean(result.cond_values[mask])),
        "mean_rank": float(np.mean(result.numerical_rank[mask])),
        "mean_frac_below_5tau": float(np.mean(result.frac_below_5tau[mask])),
    }


def generate_report(results: Dict[str, ModeResult], k_min: float, k_max: float, num_k: int, n_points: int, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Adaptive ρ Window Comparison Report (chunk-combined)")
    lines.append("")
    lines.append(f"k in [{k_min}, {k_max}], N={num_k}, base_points={n_points}")
    lines.append("")

    lines.append("## Overall Summary")
    lines.append("| Mode | Success Rate | Mean χ² (successful) | Floor Fail (σ_min ≤ τ) |")
    lines.append("|------|-------------:|---------------------:|--------------------------:|")
    for mode in ["paper", "adaptive_images", "adaptive_rank"]:
        if mode not in results:
            continue
        r = results[mode]
        floor_fail = float(np.mean(r.sigma_min <= r.tau_values)) * 100 if np.any(r.tau_values > 0) else 0.0
        mean_chi2 = r.mean_chi2
        mean_chi2_str = f"{mean_chi2:.4f}" if mean_chi2 < 100 else "inf"
        lines.append(f"| {mode} | {r.success_rate*100:6.1f}% | {mean_chi2_str:>21} | {floor_fail:23.1f}% |")
    lines.append("")

    lines.append("## Band Statistics")
    for mode in ["paper", "adaptive_images", "adaptive_rank"]:
        if mode not in results:
            continue
        r = results[mode]
        lines.append(f"### `{mode}`")
        lines.append("| Band | N | Success | Mean χ² | Mean ρ_max | Mean images | σ_min/τ | Mean cond | Mean rank | Frac ≤5τ |")
        lines.append("|------|--:|--------:|-------:|----------:|-----------:|--------:|----------:|----------:|---------:|")
        for i in range(len(K_BAND_EDGES) - 1):
            k_lo, k_hi = K_BAND_EDGES[i], K_BAND_EDGES[i + 1]
            bs = band_stats(r, k_lo, k_hi)
            if not bs:
                continue
            chi2_str = f"{bs['mean_chi2']:.4f}" if bs['mean_chi2'] < 100 else "inf"
            ratio = (bs["mean_sigma_min"] / bs["mean_tau"]) if bs["mean_tau"] > 0 else float("inf")
            lines.append(
                "| "
                + f"[{k_lo:.0f},{k_hi:.0f}) | {bs['n']} | {bs['success']*100:6.1f}% | {chi2_str:>7} | "
                + f"{bs['mean_rho']:9.2f} | {bs['mean_images']:10.1f} | {ratio:7.1f} | "
                + f"{bs['mean_cond']:9.1e} | {bs['mean_rank']:9.1f} | {bs['mean_frac_below_5tau']*100:8.1f}% |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, required=True, help="Base directory containing per-mode dirs")
    p.add_argument("--k-min", type=float, default=1.0)
    p.add_argument("--k-max", type=float, default=10.0)
    p.add_argument("--num-k", type=int, default=400)
    p.add_argument("--n-points", type=int, default=60)
    p.add_argument("--out", type=str, default="REPORT_RHO_ADAPTIVE.md")
    args = p.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        raise SystemExit(f"Base dir not found: {base}")

    results: Dict[str, ModeResult] = {}
    for mode in ["paper", "adaptive_images", "adaptive_rank"]:
        mode_dir = base / mode
        if not mode_dir.exists():
            continue
        results[mode] = load_mode_chunks(mode_dir)

    if not results:
        raise SystemExit(f"No mode outputs found under {base}")

    out_path = Path(args.out)
    generate_report(results, args.k_min, args.k_max, args.num_k, args.n_points, out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
