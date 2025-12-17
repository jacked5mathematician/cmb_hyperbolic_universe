#!/usr/bin/env python3
"""Run all three rho window modes and generate a comparison report.

This script is intended for *local smoke tests* (small num-k) and for driving
HPC chunked runs via main.py's built-in k-chunking:

    main.py --k-chunk-index i --k-num-chunks N

For full 400-k runs on HPC, prefer:
    1) run per-mode/per-chunk jobs that write:
                <base>/<mode>/chunk_<i>/spectrum.npz
    2) combine them into a report with:
                python scripts/combine_rho_adaptive_chunks.py --base-dir <base>
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
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
    success_mask: np.ndarray

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.success_mask))

    @property
    def mean_chi2(self) -> float:
        valid = self.success_mask
        if np.sum(valid) == 0:
            return float('inf')
        return float(np.mean(self.chi2_values[valid]))


def run_mode(mode: str, k_min: float, k_max: float, num_k: int, n_points: int,
             output_dir: Path, chi2_threshold: float = 1.0) -> ModeResult:
    cmd = [
        sys.executable, "main.py",
        "--k-min", str(k_min),
        "--k-max", str(k_max),
        "--num-k", str(num_k),
        "--n-points", str(n_points),
        "--output-dir", str(output_dir),
        "--rho-window-mode", mode,
    ]
    # Mode-specific parameters (using actual CLI arg names)
    if mode == "adaptive_images":
        cmd.extend(["--adaptive-target-q10-images", "10", "--adaptive-max-drop-fraction", "0.3"])
    elif mode == "adaptive_rank":
        cmd.extend(["--adaptive-target-rank-frac", "0.95", "--adaptive-target-nullity", "2"])

    print(f"\n{'='*60}")
    print(f"Running mode: {mode}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Pipeline failed for mode {mode}")

    spectrum_file = output_dir / "spectrum.npz"
    data = np.load(spectrum_file)

    # NOTE: main.py writes these keys (see spectrum.npz):
    #   chi2_rank_1, rho_max, images_per_point, sigma_min, tau, condition_number, numerical_rank, frac_below_5tau, ...
    k_values = data["k_values"]
    chi2_values = data["chi2_rank_1"]
    rho_max_values = data.get("rho_max", np.zeros_like(k_values))
    num_images = data.get("images_per_point", np.zeros_like(k_values))
    sigma_min = data.get("sigma_min", np.zeros_like(k_values))
    tau_values = data.get("tau", np.zeros_like(k_values))
    cond_values = data.get("condition_number", np.zeros_like(k_values))
    numerical_rank = data.get("numerical_rank", np.zeros_like(k_values))
    frac_below_5tau = data.get("frac_below_5tau", np.zeros_like(k_values))

    return ModeResult(
        mode=mode,
        k_values=k_values,
        chi2_values=chi2_values,
        rho_max_values=rho_max_values,
        num_images=num_images,
        sigma_min=sigma_min,
        tau_values=tau_values,
        cond_values=cond_values,
        numerical_rank=numerical_rank,
        frac_below_5tau=frac_below_5tau,
        success_mask=chi2_values < chi2_threshold,
    )


def band_stats(result: ModeResult, k_lo: float, k_hi: float):
    mask = (result.k_values >= k_lo) & (result.k_values < k_hi)
    if k_hi == K_BAND_EDGES[-1]:
        mask = (result.k_values >= k_lo) & (result.k_values <= k_hi)
    if np.sum(mask) == 0:
        return None
    return {
        "n": int(np.sum(mask)),
        "success": float(np.mean(result.success_mask[mask])),
        "mean_chi2": float(np.mean(result.chi2_values[mask][result.success_mask[mask]])) if np.any(result.success_mask[mask]) else float('inf'),
        "mean_rho": float(np.mean(result.rho_max_values[mask])),
        "mean_images": float(np.mean(result.num_images[mask])),
        "mean_sigma_min": float(np.mean(result.sigma_min[mask])),
        "mean_tau": float(np.mean(result.tau_values[mask])),
        "floor_fail": float(np.mean(result.sigma_min[mask] <= result.tau_values[mask])),
    }


def generate_report(results: dict, args, output_path: Path):
    lines = ["# Adaptive rho Window Comparison Report", ""]
    lines.append(f"k in [{args.k_min}, {args.k_max}], N={args.num_k}, base_points={args.n_points}")
    lines.append("")

    lines.append("## Overall Summary")
    lines.append("| Mode | Success Rate | Mean chi2 | Floor Fail |")
    lines.append("|------|-------------|-----------|------------|")
    for mode_name in ["paper", "adaptive_images", "adaptive_rank"]:
        if mode_name not in results:
            continue
        r = results[mode_name]
        ff = float(np.mean(r.sigma_min <= r.tau_values)) * 100 if np.any(r.tau_values > 0) else 0
        lines.append(f"| {mode_name} | {r.success_rate*100:.1f}% | {r.mean_chi2:.4f} | {ff:.1f}% |")
    lines.append("")

    lines.append("## Band Statistics")
    for mode_name in ["paper", "adaptive_images", "adaptive_rank"]:
        if mode_name not in results:
            continue
        r = results[mode_name]
        lines.append(f"### {mode_name}")
        lines.append("| Band | N | Success | Mean chi2 | Mean rho | Images | sigma_min | tau | Floor Fail |")
        lines.append("|------|---|---------|-----------|----------|--------|-----------|-----|------------|")
        for i in range(len(K_BAND_EDGES) - 1):
            k_lo, k_hi = K_BAND_EDGES[i], K_BAND_EDGES[i+1]
            bs = band_stats(r, k_lo, k_hi)
            if bs:
                chi2_str = f"{bs['mean_chi2']:.4f}" if bs['mean_chi2'] < 100 else "inf"
                lines.append(f"| [{k_lo:.0f},{k_hi:.0f}) | {bs['n']} | {bs['success']*100:.0f}% | {chi2_str} | {bs['mean_rho']:.2f} | {bs['mean_images']:.1f} | {bs['mean_sigma_min']:.2e} | {bs['mean_tau']:.2e} | {bs['floor_fail']*100:.0f}% |")
        lines.append("")

    report_text = "\n".join(lines)
    output_path.write_text(report_text)
    return report_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-min", type=float, default=1.0)
    parser.add_argument("--k-max", type=float, default=10.0)
    parser.add_argument("--num-k", type=int, default=400)
    parser.add_argument("--n-points", type=int, default=60)
    parser.add_argument("--chi2-threshold", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--hpc-num-chunks",
        type=int,
        default=None,
        help="If set, print suggested per-chunk commands instead of running locally.",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="rho_adaptive_"))
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {base_dir}")

    modes = ["paper", "adaptive_images", "adaptive_rank"]

    if args.hpc_num_chunks is not None:
        n = int(args.hpc_num_chunks)
        print("\nHPC chunked run commands (one mode x one chunk per job):\n")
        for mode in modes:
            for i in range(n):
                # Use the correct main.py flag: --rho-window-mode
                parts = [
                    sys.executable,
                    "main.py",
                    "--k-min",
                    str(args.k_min),
                    "--k-max",
                    str(args.k_max),
                    "--num-k",
                    str(args.num_k),
                    "--n-points",
                    str(args.n_points),
                    "--no-plot",
                    "--no-eigenvalues",
                    "--rho-window-mode",
                    mode,
                    "--k-num-chunks",
                    str(n),
                    "--k-chunk-index",
                    str(i),
                    "--output-dir",
                    str(base_dir / mode),
                ]
                if mode == "adaptive_images":
                    parts += ["--adaptive-target-q10-images", "10", "--adaptive-max-drop-fraction", "0.3"]
                elif mode == "adaptive_rank":
                    parts += ["--adaptive-target-rank-frac", "0.95", "--adaptive-target-nullity", "2"]
                print(" ".join(parts))

        print("\nAfter all chunks finish:")
        print(
            f"{sys.executable} scripts/combine_rho_adaptive_chunks.py --base-dir {base_dir} "
            f"--k-min {args.k_min} --k-max {args.k_max} --num-k {args.num_k} --n-points {args.n_points}"
        )
        return
    results = {}

    for mode in modes:
        mode_dir = base_dir / f"output_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        try:
            r = run_mode(mode, args.k_min, args.k_max, args.num_k, args.n_points, mode_dir, args.chi2_threshold)
            results[mode] = r
            print(f"{mode}: {r.success_rate*100:.1f}% success")
        except Exception as e:
            print(f"ERROR {mode}: {e}")

    report_path = Path("REPORT_RHO_ADAPTIVE.md")
    generate_report(results, args, report_path)
    print(f"\nReport: {report_path}")

    print("\n## SUMMARY ##")
    print(f"k in [{args.k_min}, {args.k_max}], N={args.num_k}, pts={args.n_points}")
    print("| Mode | Success | Mean chi2 |")
    print("|------|---------|-----------|")
    for m in modes:
        if m in results:
            print(f"| {m} | {results[m].success_rate*100:.0f}% | {results[m].mean_chi2:.4f} |")


if __name__ == "__main__":
    main()
