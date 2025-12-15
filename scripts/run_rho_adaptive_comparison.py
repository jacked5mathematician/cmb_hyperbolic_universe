#!/usr/bin/env python3
"""
Run all three rho window modes and generate a comparison report.

Usage:
    python scripts/run_rho_adaptive_comparison.py [--k-min 1.0] [--k-max 10.0] [--num-k 400] [--n-points 40]

This script:
1. Runs the pipeline in paper mode (baseline, expected to fail at high k)
2. Runs adaptive_images mode (expand rho_max for more images)
3. Runs adaptive_rank mode (expand rho_max for better rank)
4. Generates REPORT_RHO_ADAPTIVE.md with comparison tables
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

LOGGER = logging.getLogger(__name__)

# Output directories
OUTPUT_BASE = Path("outputs")
PAPER_DIR = OUTPUT_BASE / "paper_baseline"
ADAPTIVE_IMAGES_DIR = OUTPUT_BASE / "adaptive_images"
ADAPTIVE_RANK_DIR = OUTPUT_BASE / "adaptive_rank"

REPORT_PATH = Path("REPORT_RHO_ADAPTIVE.md")


def run_mode(mode: str, output_dir: Path, k_min: float, k_max: float, num_k: int, n_points: int, extra_args: list[str] = None) -> bool:
    """Run the pipeline with a specific rho window mode."""
    cmd = [
        sys.executable, "main.py",
        "--k-min", str(k_min),
        "--k-max", str(k_max),
        "--num-k", str(num_k),
        "--n-points", str(n_points),
        "--output-dir", str(output_dir),
        "--rho-window-mode", mode,
        "--no-plot",
        "--no-eigenvalues",
        "--self-check",
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    LOGGER.info("Running mode=%s: %s", mode, " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            LOGGER.warning("Mode %s returned non-zero exit code %d", mode, result.returncode)
            LOGGER.warning("stderr: %s", result.stderr[-2000:] if result.stderr else "")
        return True
    except subprocess.TimeoutExpired:
        LOGGER.error("Mode %s timed out after 1 hour", mode)
        return False
    except Exception as e:
        LOGGER.error("Mode %s failed: %s", mode, e)
        return False


def load_spectrum(output_dir: Path) -> dict | None:
    """Load spectrum.npz from an output directory."""
    spectrum_path = output_dir / "spectrum.npz"
    if not spectrum_path.exists():
        LOGGER.warning("No spectrum.npz found in %s", output_dir)
        return None
    
    try:
        data = dict(np.load(spectrum_path, allow_pickle=True))
        return data
    except Exception as e:
        LOGGER.error("Failed to load %s: %s", spectrum_path, e)
        return None


def analyze_results(paper: dict, adaptive_images: dict, adaptive_rank: dict) -> dict:
    """Analyze and compare results from all three modes."""
    analysis = {
        "paper": {},
        "adaptive_images": {},
        "adaptive_rank": {},
        "comparison": {},
    }
    
    for name, data in [("paper", paper), ("adaptive_images", adaptive_images), ("adaptive_rank", adaptive_rank)]:
        if data is None:
            analysis[name] = {"error": "No data available"}
            continue
        
        k_values = data.get("k_values", np.array([]))
        chi2_rank_1 = data.get("chi2_rank_1", np.array([]))
        kept_points = data.get("kept_points", np.array([]))
        rho_max = data.get("rho_max", np.array([]))
        rho_max_original = data.get("rho_max_original", rho_max)
        adaptive_expansions = data.get("adaptive_expansions", np.zeros_like(k_values, dtype=int))
        images_per_point = data.get("images_per_point", np.array([]))
        
        # Count valid (finite) chi² values
        valid_chi2_mask = np.isfinite(chi2_rank_1)
        n_valid = int(np.sum(valid_chi2_mask))
        n_total = len(k_values)
        
        # Count points dropped (kept_points == 0)
        n_dropped = int(np.sum(kept_points == 0))
        
        # Summary statistics
        analysis[name] = {
            "n_k_values": n_total,
            "n_valid_chi2": n_valid,
            "n_dropped_points": n_dropped,
            "success_rate": n_valid / n_total if n_total > 0 else 0,
            "mean_kept_points": float(np.nanmean(kept_points)),
            "min_kept_points": int(np.nanmin(kept_points)) if len(kept_points) > 0 else 0,
            "max_rho_expansion": float(np.max(rho_max - rho_max_original)) if len(rho_max) > 0 else 0,
            "mean_expansions": float(np.mean(adaptive_expansions)),
            "mean_images_per_point": float(np.nanmean(images_per_point)) if len(images_per_point) > 0 else 0,
        }
        
        if len(k_values) > 0:
            # Find first k where chi² becomes NaN
            nan_indices = np.where(~valid_chi2_mask)[0]
            if len(nan_indices) > 0:
                first_nan_k = k_values[nan_indices[0]]
                analysis[name]["first_failure_k"] = float(first_nan_k)
            else:
                analysis[name]["first_failure_k"] = None  # No failures
    
    # Comparison metrics
    if paper and adaptive_images:
        paper_valid = np.sum(np.isfinite(paper.get("chi2_rank_1", [])))
        adaptive_valid = np.sum(np.isfinite(adaptive_images.get("chi2_rank_1", [])))
        analysis["comparison"]["images_improvement"] = int(adaptive_valid - paper_valid)
    
    if paper and adaptive_rank:
        paper_valid = np.sum(np.isfinite(paper.get("chi2_rank_1", [])))
        rank_valid = np.sum(np.isfinite(adaptive_rank.get("chi2_rank_1", [])))
        analysis["comparison"]["rank_improvement"] = int(rank_valid - paper_valid)
    
    return analysis


def generate_report(analysis: dict, k_min: float, k_max: float, num_k: int, n_points: int, paper: dict, adaptive_images: dict, adaptive_rank: dict) -> str:
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        "# Adaptive Rho Window Comparison Report",
        "",
        f"Generated: {timestamp}",
        "",
        "## Configuration",
        "",
        f"- k range: [{k_min}, {k_max}]",
        f"- Number of k samples: {num_k}",
        f"- Base points: {n_points}",
        "",
        "## Summary",
        "",
        "| Mode | Valid χ² | Success Rate | First Failure k | Mean Kept Points | Mean Images/Point |",
        "|------|----------|--------------|-----------------|------------------|-------------------|",
    ]
    
    for mode in ["paper", "adaptive_images", "adaptive_rank"]:
        data = analysis.get(mode, {})
        if "error" in data:
            lines.append(f"| {mode} | ERROR | - | - | - | - |")
            continue
        
        valid = data.get("n_valid_chi2", 0)
        rate = data.get("success_rate", 0) * 100
        first_fail = data.get("first_failure_k")
        first_fail_str = f"{first_fail:.2f}" if first_fail is not None else "None"
        kept = data.get("mean_kept_points", 0)
        images = data.get("mean_images_per_point", 0)
        
        lines.append(f"| {mode} | {valid} | {rate:.1f}% | {first_fail_str} | {kept:.1f} | {images:.1f} |")
    
    lines.extend([
        "",
        "## Improvement Over Paper Baseline",
        "",
    ])
    
    comparison = analysis.get("comparison", {})
    images_imp = comparison.get("images_improvement", "N/A")
    rank_imp = comparison.get("rank_improvement", "N/A")
    
    lines.extend([
        f"- **adaptive_images mode**: +{images_imp} additional valid k-values",
        f"- **adaptive_rank mode**: +{rank_imp} additional valid k-values",
        "",
    ])
    
    # Detailed per-k table (first 20 + last 5)
    if paper is not None and len(paper.get("k_values", [])) > 0:
        lines.extend([
            "## Detailed Results (Sample)",
            "",
            "### First 20 k-values",
            "",
            "| k | Paper χ² | Images χ² | Rank χ² | Paper rho_max | Images rho_max | Rank rho_max |",
            "|---|----------|-----------|---------|---------------|----------------|--------------|",
        ])
        
        k_vals = paper.get("k_values", [])
        for i in range(min(20, len(k_vals))):
            k = k_vals[i]
            p_chi2 = paper.get("chi2_rank_1", [np.nan])[i] if paper else np.nan
            i_chi2 = adaptive_images.get("chi2_rank_1", [np.nan])[i] if adaptive_images and i < len(adaptive_images.get("chi2_rank_1", [])) else np.nan
            r_chi2 = adaptive_rank.get("chi2_rank_1", [np.nan])[i] if adaptive_rank and i < len(adaptive_rank.get("chi2_rank_1", [])) else np.nan
            
            p_rho = paper.get("rho_max", [np.nan])[i] if paper else np.nan
            i_rho = adaptive_images.get("rho_max", [np.nan])[i] if adaptive_images and i < len(adaptive_images.get("rho_max", [])) else np.nan
            r_rho = adaptive_rank.get("rho_max", [np.nan])[i] if adaptive_rank and i < len(adaptive_rank.get("rho_max", [])) else np.nan
            
            def fmt(v):
                if np.isnan(v):
                    return "NaN"
                elif v < 0.01:
                    return f"{v:.2e}"
                else:
                    return f"{v:.4f}"
            
            lines.append(f"| {k:.3f} | {fmt(p_chi2)} | {fmt(i_chi2)} | {fmt(r_chi2)} | {p_rho:.3f} | {i_rho:.3f} | {r_rho:.3f} |")
        
        if len(k_vals) > 25:
            lines.extend([
                "",
                "### Last 5 k-values",
                "",
                "| k | Paper χ² | Images χ² | Rank χ² | Paper rho_max | Images rho_max | Rank rho_max |",
                "|---|----------|-----------|---------|---------------|----------------|--------------|",
            ])
            
            for i in range(max(0, len(k_vals) - 5), len(k_vals)):
                k = k_vals[i]
                p_chi2 = paper.get("chi2_rank_1", [np.nan])[i] if paper else np.nan
                i_chi2 = adaptive_images.get("chi2_rank_1", [np.nan])[i] if adaptive_images and i < len(adaptive_images.get("chi2_rank_1", [])) else np.nan
                r_chi2 = adaptive_rank.get("chi2_rank_1", [np.nan])[i] if adaptive_rank and i < len(adaptive_rank.get("chi2_rank_1", [])) else np.nan
                
                p_rho = paper.get("rho_max", [np.nan])[i] if paper else np.nan
                i_rho = adaptive_images.get("rho_max", [np.nan])[i] if adaptive_images and i < len(adaptive_images.get("rho_max", [])) else np.nan
                r_rho = adaptive_rank.get("rho_max", [np.nan])[i] if adaptive_rank and i < len(adaptive_rank.get("rho_max", [])) else np.nan
                
                def fmt(v):
                    if np.isnan(v):
                        return "NaN"
                    elif v < 0.01:
                        return f"{v:.2e}"
                    else:
                        return f"{v:.4f}"
                
                lines.append(f"| {k:.3f} | {fmt(p_chi2)} | {fmt(i_chi2)} | {fmt(r_chi2)} | {p_rho:.3f} | {i_rho:.3f} | {r_rho:.3f} |")
    
    lines.extend([
        "",
        "## Adaptive Mode Statistics",
        "",
    ])
    
    for mode in ["adaptive_images", "adaptive_rank"]:
        data = analysis.get(mode, {})
        if "error" not in data:
            lines.extend([
                f"### {mode}",
                "",
                f"- Max rho expansion: {data.get('max_rho_expansion', 0):.3f}",
                f"- Mean expansion steps: {data.get('mean_expansions', 0):.2f}",
                "",
            ])
    
    lines.extend([
        "## Conclusions",
        "",
        "1. **Paper mode failure pattern**: The paper-faithful cutoffs lead to insufficient ghost images at high k, causing point dropout and NaN chi² values.",
        "",
        "2. **Adaptive strategies effectiveness**: Compare success rates to determine if adaptive window expansion recovers valid eigenvalue estimates.",
        "",
        "3. **Trade-offs**: Expanding rho_max increases images but may include contributions from beyond the intended radial cutoff. Investigate if chi² values remain physically meaningful.",
        "",
    ])
    
    return "\n".join(lines)


def print_pasteable_summary(analysis: dict):
    """Print a copy-pasteable summary for chat/PR."""
    print("\n" + "="*60)
    print("PASTEABLE SUMMARY")
    print("="*60)
    
    lines = [
        "## Rho Adaptive Window Experiment Results",
        "",
    ]
    
    for mode in ["paper", "adaptive_images", "adaptive_rank"]:
        data = analysis.get(mode, {})
        if "error" in data:
            lines.append(f"- **{mode}**: ERROR - {data['error']}")
        else:
            valid = data.get("n_valid_chi2", 0)
            total = data.get("n_k_values", 0)
            rate = data.get("success_rate", 0) * 100
            first_fail = data.get("first_failure_k")
            fail_str = f"k={first_fail:.2f}" if first_fail else "none"
            lines.append(f"- **{mode}**: {valid}/{total} valid ({rate:.1f}%), first failure: {fail_str}")
    
    comparison = analysis.get("comparison", {})
    if comparison:
        lines.extend([
            "",
            "**Improvements over paper baseline:**",
            f"- adaptive_images: +{comparison.get('images_improvement', '?')} k-values",
            f"- adaptive_rank: +{comparison.get('rank_improvement', '?')} k-values",
        ])
    
    print("\n".join(lines))
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run adaptive rho window comparison")
    parser.add_argument("--k-min", type=float, default=1.0)
    parser.add_argument("--k-max", type=float, default=10.0)
    parser.add_argument("--num-k", type=int, default=400)
    parser.add_argument("--n-points", type=int, default=40)
    parser.add_argument("--skip-paper", action="store_true", help="Skip paper mode (reuse existing)")
    parser.add_argument("--skip-images", action="store_true", help="Skip adaptive_images mode")
    parser.add_argument("--skip-rank", action="store_true", help="Skip adaptive_rank mode")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # Run each mode
    if not args.skip_paper:
        run_mode("paper", PAPER_DIR, args.k_min, args.k_max, args.num_k, args.n_points)
    
    if not args.skip_images:
        run_mode("adaptive_images", ADAPTIVE_IMAGES_DIR, args.k_min, args.k_max, args.num_k, args.n_points)
    
    if not args.skip_rank:
        run_mode("adaptive_rank", ADAPTIVE_RANK_DIR, args.k_min, args.k_max, args.num_k, args.n_points)
    
    # Load results
    paper = load_spectrum(PAPER_DIR)
    adaptive_images = load_spectrum(ADAPTIVE_IMAGES_DIR)
    adaptive_rank = load_spectrum(ADAPTIVE_RANK_DIR)
    
    # Analyze
    analysis = analyze_results(paper, adaptive_images, adaptive_rank)
    
    # Generate report
    report = generate_report(analysis, args.k_min, args.k_max, args.num_k, args.n_points, paper, adaptive_images, adaptive_rank)
    
    REPORT_PATH.write_text(report)
    LOGGER.info("Wrote report to %s", REPORT_PATH)
    
    # Save analysis as JSON
    analysis_path = OUTPUT_BASE / "analysis.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=convert)
    LOGGER.info("Wrote analysis to %s", analysis_path)
    
    # Print pasteable summary
    print_pasteable_summary(analysis)


if __name__ == "__main__":
    main()
