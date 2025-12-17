#!/usr/bin/env python3
"""
Local diagnostics suite for investigating high-k SVD floor / rank deficiency.

This script runs a controlled set of experiments to diagnose whether the issue is:
(A) paper rho cutoff starving constraints (insufficient images)
(B) insufficient diversity of base points in function space
(C) numerical scaling / redundancy that persists even with many images

Run with:
    python scripts/run_local_diagnostics_suite.py [--output-dir OUTPUT_DIR]

Output:
    - REPORT_LOCAL_DIAGNOSTICS.md (in current directory)
    - <output_dir>/run_<X>/ directories with per-run artifacts
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import numpy as np

# Constants
SEED = 0
DEFAULT_N_POINTS = 30
PYTHON = sys.executable


@dataclass
class RunConfig:
    """Configuration for a single diagnostic run."""
    name: str
    k_values: List[float]
    n_points: int = DEFAULT_N_POINTS
    rho_window_mode: str = "paper"
    diversity_fps: Optional[int] = None
    description: str = ""
    

@dataclass
class RunResult:
    """Results from a single diagnostic run."""
    config: RunConfig
    csv_path: Optional[Path] = None
    success: bool = False
    error: Optional[str] = None
    # Parsed from CSV
    rows: List[Dict[str, Any]] = field(default_factory=list)


def run_single(config: RunConfig, output_dir: Path, seed: int = SEED) -> RunResult:
    """Execute a single diagnostic run."""
    result = RunResult(config=config)
    
    run_dir = output_dir / f"run_{config.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    k_min = min(config.k_values)
    k_max = max(config.k_values)
    num_k = len(config.k_values)
    
    cmd = [
        PYTHON, "main.py",
        "--k-min", str(k_min),
        "--k-max", str(k_max),
        "--num-k", str(num_k),
        "--n-points", str(config.n_points),
        "--seed", str(seed),
        "--output-dir", str(run_dir),
        "--rho-window-mode", config.rho_window_mode,
        "--extended-diagnostics",
        "--no-plot",
        "--no-eigenvalues",
    ]
    
    if config.diversity_fps is not None:
        cmd.extend(["--diversity-fps", str(config.diversity_fps)])
    
    print(f"\n{'='*60}")
    print(f"Run: {config.name}")
    print(f"Description: {config.description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            result.error = f"Exit code {proc.returncode}: {proc.stderr[-500:]}"
            print(f"FAILED: {result.error}")
            return result
    except subprocess.TimeoutExpired:
        result.error = "Timeout (600s)"
        print(f"FAILED: {result.error}")
        return result
    except Exception as e:
        result.error = str(e)
        print(f"FAILED: {result.error}")
        return result
    
    # Parse CSV
    csv_path = run_dir / "svd_diag.csv"
    if not csv_path.exists():
        result.error = "No svd_diag.csv produced"
        print(f"FAILED: {result.error}")
        return result
    
    result.csv_path = csv_path
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        result.rows = list(reader)
    
    result.success = True
    print(f"SUCCESS: {len(result.rows)} k-values processed")
    return result


def compute_gram_similarity(output_dir: Path, k1: float, k2: float) -> Optional[float]:
    """Compute relative Frobenius difference between Gram matrices at k1 and k2."""
    eig_dir = output_dir / "gram_eigvals"
    if not eig_dir.exists():
        return None
    
    # Find matching files
    eig1_file = None
    eig2_file = None
    for f in eig_dir.glob("eig_k_*.npy"):
        k_str = f.stem.replace("eig_k_", "")
        try:
            k_val = float(k_str)
            if abs(k_val - k1) < 0.001:
                eig1_file = f
            if abs(k_val - k2) < 0.001:
                eig2_file = f
        except ValueError:
            continue
    
    if eig1_file is None or eig2_file is None:
        return None
    
    eig1 = np.load(eig1_file)
    eig2 = np.load(eig2_file)
    
    # Pad to same length if needed
    max_len = max(len(eig1), len(eig2))
    eig1_padded = np.zeros(max_len)
    eig2_padded = np.zeros(max_len)
    eig1_padded[:len(eig1)] = eig1
    eig2_padded[:len(eig2)] = eig2
    
    # Relative Frobenius difference
    diff = np.linalg.norm(eig1_padded - eig2_padded)
    norm1 = np.linalg.norm(eig1_padded)
    if norm1 == 0:
        return float('inf')
    return diff / norm1


def generate_report(results: Dict[str, RunResult], output_dir: Path) -> str:
    """Generate markdown report from run results."""
    lines = []
    lines.append("# Local Diagnostics Report: High-k SVD Floor Investigation")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This report investigates whether high-k failures are caused by:")
    lines.append("- **(A)** Paper rho cutoff starving constraints (insufficient images)")
    lines.append("- **(B)** Insufficient diversity of base points in function space")
    lines.append("- **(C)** Numerical scaling / redundancy that persists even with many images")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Seed: {SEED}")
    lines.append(f"- Default n_points: {DEFAULT_N_POINTS}")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append("")
    
    # Run summaries
    lines.append("## Run Summaries")
    lines.append("")
    
    for name, result in results.items():
        cfg = result.config
        lines.append(f"### Run {name}: {cfg.description}")
        lines.append("")
        
        if not result.success:
            lines.append(f"**FAILED**: {result.error}")
            lines.append("")
            continue
        
        # Summary table
        lines.append("| k | χ² | M | N | kept_pts | total_img | σ_min | σ_2 | σ_max | τ | below_τ | rank | nullity | cond | failure |")
        lines.append("|---|---:|---:|---:|--------:|--------:|------:|----:|-----:|---:|-------:|-----:|-------:|-----:|--------|")
        
        for row in result.rows:
            k = float(row['k'])
            failure = row.get('failure_reason', '')
            
            # Handle NaN rows (failures)
            if failure == 'no_viable_points':
                lines.append(f"| {k:.2f} | - | 0 | {row['N']} | 0 | 0 | - | - | - | - | - | - | - | - | **no_viable_points** |")
                continue
            
            chi2_raw = row['chi2_rank1']
            chi2 = float(chi2_raw) if chi2_raw not in ('nan', '') else float('nan')
            M = int(row['M'])
            N = int(row['N'])
            kept = int(row['kept_points'])
            total_img = int(row['total_images'])
            sigma_min_raw = row['sigma_min']
            sigma_min = float(sigma_min_raw) if sigma_min_raw not in ('nan', '') else float('nan')
            sigma_2_raw = row['sigma_2']
            sigma_2 = float(sigma_2_raw) if sigma_2_raw not in ('nan', '') else float('nan')
            sigma_max_raw = row['sigma_max']
            sigma_max = float(sigma_max_raw) if sigma_max_raw not in ('nan', '') else float('nan')
            tau_raw = row['tau']
            tau = float(tau_raw) if tau_raw not in ('nan', '') else float('nan')
            below_tau = int(row['below_tau'])
            rank = int(row['numerical_rank'])
            nullity = int(row['nullity'])
            cond_raw = row['condition_number']
            cond = float(cond_raw) if cond_raw not in ('nan', '') else float('nan')
            
            chi2_str = f"{chi2:.4f}" if np.isfinite(chi2) and chi2 < 100 else (f"{chi2:.2e}" if np.isfinite(chi2) else "-")
            sigma_min_str = f"{sigma_min:.2e}" if np.isfinite(sigma_min) else "-"
            sigma_2_str = f"{sigma_2:.2e}" if np.isfinite(sigma_2) else "-"
            sigma_max_str = f"{sigma_max:.2e}" if np.isfinite(sigma_max) else "-"
            tau_str = f"{tau:.2e}" if np.isfinite(tau) else "-"
            cond_str = f"{cond:.2e}" if np.isfinite(cond) and cond > 1e6 else (f"{cond:.1f}" if np.isfinite(cond) else "-")
            
            lines.append(f"| {k:.2f} | {chi2_str} | {M} | {N} | {kept} | {total_img} | {sigma_min_str} | {sigma_2_str} | {sigma_max_str} | {tau_str} | {below_tau} | {rank} | {nullity} | {cond_str} | {failure} |")
        
        lines.append("")
    
    # Analysis section
    lines.append("## Analysis")
    lines.append("")
    
    # Check Run A (low-k sanity)
    if 'A' in results and results['A'].success:
        row = results['A'].rows[0]
        failure = row.get('failure_reason', '')
        if failure != 'no_viable_points':
            sigma_min = float(row['sigma_min'])
            tau = float(row['tau'])
            below = int(row['below_tau'])
            lines.append(f"### Low-k sanity (Run A, k=1.5)")
            lines.append(f"- σ_min = {sigma_min:.2e}, τ = {tau:.2e}")
            lines.append(f"- below_τ = {below} → {'✅ NOT at floor' if below == 0 else '❌ AT floor (unexpected)'}")
            lines.append("")
    
    # Check Run B (mid-k)
    if 'B' in results and results['B'].success:
        row = results['B'].rows[0]
        failure = row.get('failure_reason', '')
        if failure != 'no_viable_points':
            sigma_min = float(row['sigma_min'])
            tau = float(row['tau'])
            below = int(row['below_tau'])
            lines.append(f"### Mid-k transition (Run B, k=3.5)")
            lines.append(f"- σ_min = {sigma_min:.2e}, τ = {tau:.2e}")
            lines.append(f"- below_τ = {below} → {'at floor' if below == 1 else 'not at floor'}")
            lines.append("")
    
    # Check Run C vs D (high-k paper vs adaptive)
    run_c_is_failure = False
    if 'C' in results and results['C'].success:
        row_c = results['C'].rows[0]
        failure_c = row_c.get('failure_reason', '')
        lines.append(f"### High-k paper mode (Run C, k=9.8)")
        if failure_c == 'no_viable_points':
            run_c_is_failure = True
            lines.append(f"- **FAILURE**: no_viable_points - paper rho window is too narrow")
            lines.append(f"- This confirms hypothesis (A): paper rho cutoff starves constraints")
            lines.append("")
        else:
            lines.append(f"- σ_min = {float(row_c['sigma_min']):.2e}, τ = {float(row_c['tau']):.2e}")
            lines.append(f"- below_τ = {row_c['below_tau']}, total_images = {row_c['total_images']}")
            lines.append(f"- rank = {row_c['numerical_rank']}, nullity = {row_c['nullity']}")
            lines.append("")
    
    if 'D' in results and results['D'].success:
        row_d = results['D'].rows[0]
        failure_d = row_d.get('failure_reason', '')
        lines.append(f"### High-k adaptive mode (Run D, k=9.8)")
        if failure_d == 'no_viable_points':
            lines.append(f"- **FAILURE**: no_viable_points - even adaptive mode couldn't help")
            lines.append("")
        else:
            lines.append(f"- σ_min = {float(row_d['sigma_min']):.2e}, τ = {float(row_d['tau']):.2e}")
            lines.append(f"- below_τ = {row_d['below_tau']}, total_images = {row_d['total_images']}")
            lines.append(f"- rank = {row_d['numerical_rank']}, nullity = {row_d['nullity']}")
            lines.append(f"- rho_max_original = {row_d['rho_max_original']}, rho_max = {row_d['rho_max']}")
            lines.append(f"- adaptive_expansions = {row_d['adaptive_expansions']}")
            lines.append("")
        
        # Compare C and D
        if 'C' in results and results['C'].success:
            row_c = results['C'].rows[0]
            failure_c = row_c.get('failure_reason', '')
            
            lines.append("**Comparison C vs D:**")
            if run_c_is_failure and failure_d != 'no_viable_points':
                lines.append(f"- Paper mode FAILED (no points), adaptive mode SUCCEEDED")
                lines.append(f"- Adaptive yielded {row_d['total_images']} images, rank={row_d['numerical_rank']}")
                lines.append(f"- This confirms adaptive rho expansion is necessary for high-k ✅")
            elif run_c_is_failure and failure_d == 'no_viable_points':
                lines.append(f"- Both modes failed - severe constraint starvation")
            elif not run_c_is_failure and failure_d != 'no_viable_points':
                img_c = int(row_c['total_images'])
                img_d = int(row_d['total_images'])
                below_c = int(row_c['below_tau'])
                below_d = int(row_d['below_tau'])
                
                if img_d > img_c:
                    lines.append(f"- Adaptive mode increased images: {img_c} → {img_d} (+{img_d - img_c})")
                else:
                    lines.append(f"- Adaptive mode did NOT increase images: {img_c} → {img_d}")
                
                if below_c == 1 and below_d == 0:
                    lines.append(f"- Adaptive mode lifted σ_min above τ ✅")
                elif below_c == 0 and below_d == 0:
                    lines.append(f"- Both modes have σ_min > τ")
                elif below_c == 1 and below_d == 1:
                    lines.append(f"- Both modes at floor (adaptive did NOT help) ❌")
            lines.append("")
    
    # Check Run F (sensitivity to n_points)
    if 'F_15' in results and 'F_30' in results and 'F_60' in results:
        lines.append("### Sensitivity to base-point count (Run F)")
        lines.append("")
        lines.append("| n_points | result | σ_min | τ | below_τ | rank | nullity | total_images |")
        lines.append("|----------|--------|------:|---:|-------:|-----:|-------:|------------:|")
        for suffix in ['F_15', 'F_30', 'F_60']:
            if results[suffix].success and results[suffix].rows:
                row = results[suffix].rows[0]
                n = suffix.split('_')[1]
                failure = row.get('failure_reason', '')
                if failure == 'no_viable_points':
                    lines.append(f"| {n} | **FAIL** | - | - | - | - | - | 0 |")
                else:
                    sigma_min_str = f"{float(row['sigma_min']):.2e}" if row['sigma_min'] not in ('nan', '') else "-"
                    tau_str = f"{float(row['tau']):.2e}" if row['tau'] not in ('nan', '') else "-"
                    lines.append(f"| {n} | OK | {sigma_min_str} | {tau_str} | {row['below_tau']} | {row['numerical_rank']} | {row['nullity']} | {row['total_images']} |")
        lines.append("")
    
    # Check Run G (diversity)
    run_g_success = False
    if 'G' in results and results['G'].success and results['G'].rows:
        row_g = results['G'].rows[0]
        failure_g = row_g.get('failure_reason', '')
        if failure_g != 'no_viable_points':
            run_g_success = True
    
    run_c_data_ok = False
    if 'C' in results and results['C'].success and results['C'].rows:
        row_c = results['C'].rows[0]
        failure_c = row_c.get('failure_reason', '')
        if failure_c != 'no_viable_points':
            run_c_data_ok = True
    
    if run_g_success:
        lines.append("### Diversity FPS test (Run G)")
        lines.append("")
        row_g = results['G'].rows[0]
        lines.append(f"- σ_min = {float(row_g['sigma_min']):.2e}, τ = {float(row_g['tau']):.2e}")
        lines.append(f"- below_τ = {row_g['below_tau']}, total_images = {row_g['total_images']}")
        lines.append(f"- rank = {row_g['numerical_rank']}, nullity = {row_g['nullity']}")
        
        if run_c_data_ok:
            row_c = results['C'].rows[0]
            lines.append("")
            lines.append("**Comparison G vs C:**")
            lines.append("| Metric | Run C (no FPS) | Run G (FPS) |")
            lines.append("|--------|---------------:|------------:|")
            lines.append(f"| σ_min | {float(row_c['sigma_min']):.2e} | {float(row_g['sigma_min']):.2e} |")
            lines.append(f"| rank | {row_c['numerical_rank']} | {row_g['numerical_rank']} |")
            lines.append(f"| nullity | {row_c['nullity']} | {row_g['nullity']} |")
            lines.append(f"| condition | {float(row_c['condition_number']):.2e} | {float(row_g['condition_number']):.2e} |")
        lines.append("")
    elif 'G' in results and results['G'].success and results['G'].rows:
        row_g = results['G'].rows[0]
        if row_g.get('failure_reason', '') == 'no_viable_points':
            lines.append("### Diversity FPS test (Run G)")
            lines.append("")
            lines.append("- **FAILURE**: no_viable_points - paper mode fails even with FPS")
            lines.append("")
    
    # Conclusions
    lines.append("## Conclusions")
    lines.append("")
    
    # Determine answers to the key questions
    answers = []
    
    # Helper to check if a row is a failure
    def is_failure(row):
        return row.get('failure_reason', '') == 'no_viable_points'
    
    # Q1: Does paper rho fail because of insufficient images or well-populated matrix at floor?
    if 'C' in results and results['C'].success and results['C'].rows:
        row_c = results['C'].rows[0]
        if is_failure(row_c):
            answers.append("**Q1 (Paper rho at high-k)**: Paper rho FAILS completely at k=9.8 - **no viable points** retained. The paper rho window is too narrow to provide any usable images. This confirms hypothesis **(A)**: paper rho cutoff starves constraints.")
        else:
            total_img = int(row_c['total_images'])
            M = int(row_c['M'])
            below = int(row_c['below_tau'])
            
            if total_img == 0 or below == -1:
                answers.append("**Q1**: Paper rho produces no usable data at k=9.8.")
            elif below == 1:
                answers.append(f"**Q1**: Paper rho produces a well-populated matrix ({total_img} images, M={M}) that is **still at numerical floor**. This is NOT a dropout issue but hypothesis **(C)**: numerical redundancy.")
            else:
                answers.append("**Q1**: Paper rho does NOT hit the floor at k=9.8 in this test configuration.")
    
    # Q2: Does adaptive rho eliminate NaNs and change spectrum meaningfully?
    if 'C' in results and 'D' in results and results['C'].success and results['D'].success:
        row_c = results['C'].rows[0]
        row_d = results['D'].rows[0]
        c_failed = is_failure(row_c)
        d_failed = is_failure(row_d)
        
        if c_failed and not d_failed:
            img_d = int(row_d['total_images'])
            below_d = int(row_d['below_tau'])
            rank_d = int(row_d['numerical_rank'])
            nullity_d = int(row_d['nullity'])
            answers.append(f"**Q2 (Adaptive vs Paper)**: Adaptive rho **rescues** the computation - paper mode fails completely while adaptive produces {img_d} images with rank={rank_d}, nullity={nullity_d}.")
            if below_d == 1:
                answers.append(f"   → However, adaptive mode still hits numerical floor (σ_min < τ). Hypothesis **(C)** (redundancy) is also present.")
            else:
                answers.append(f"   → Adaptive mode achieves healthy numerical rank. Hypothesis **(A)** is the primary issue.")
        elif c_failed and d_failed:
            answers.append("**Q2**: Both paper and adaptive modes fail at k=9.8 - severe constraint starvation.")
        elif not c_failed and not d_failed:
            img_c = int(row_c['total_images'])
            img_d = int(row_d['total_images'])
            below_c = int(row_c['below_tau'])
            below_d = int(row_d['below_tau'])
            
            if img_d > img_c * 1.5:
                img_change = "significantly increases images"
            elif img_d > img_c:
                img_change = "slightly increases images"
            else:
                img_change = "does NOT increase images"
            
            if below_c == 1 and below_d == 0:
                spectrum_change = "lifts σ_min above τ (meaningful improvement)"
            elif below_c == 1 and below_d == 1:
                spectrum_change = "does NOT lift σ_min above τ (no improvement)"
            else:
                spectrum_change = "no floor issue to fix"
            
            answers.append(f"**Q2**: Adaptive rho {img_change} and {spectrum_change}.")
    
    # Q3: Does increasing base-point count help?
    f_results_ok = all(f'F_{n}' in results and results[f'F_{n}'].success and results[f'F_{n}'].rows for n in [15, 30, 60])
    if f_results_ok:
        row_15 = results['F_15'].rows[0]
        row_30 = results['F_30'].rows[0]
        row_60 = results['F_60'].rows[0]
        
        fail_15 = is_failure(row_15)
        fail_30 = is_failure(row_30)
        fail_60 = is_failure(row_60)
        
        if fail_15 and fail_30 and fail_60:
            answers.append("**Q3 (Base-point count)**: All n_points values (15, 30, 60) fail at k=9.8 with paper rho. The issue is rho window, not point count.")
        elif fail_15 and not fail_60:
            answers.append("**Q3**: Increasing base-point count helps - lower counts fail while higher counts succeed.")
        elif not fail_15 and not fail_60:
            below_15 = int(row_15['below_tau']) if not fail_15 else -1
            below_30 = int(row_30['below_tau']) if not fail_30 else -1
            below_60 = int(row_60['below_tau']) if not fail_60 else -1
            
            if below_15 == 1 and below_30 == 1 and below_60 == 1:
                answers.append("**Q3**: Increasing base-point count (15→30→60) does **NOT** lift σ_min above τ. Floor persists - hypothesis **(C)**.")
            elif below_60 == 0 and below_15 == 1:
                answers.append("**Q3**: Increasing base-point count **helps** lift σ_min above τ at 60 points.")
            else:
                answers.append(f"**Q3**: Mixed results with base-point count (below_τ: 15→{below_15}, 30→{below_30}, 60→{below_60}).")
    
    # Q4: Does diversity selection help?
    g_ok = 'G' in results and results['G'].success and results['G'].rows and not is_failure(results['G'].rows[0])
    c_ok = 'C' in results and results['C'].success and results['C'].rows and not is_failure(results['C'].rows[0])
    
    if g_ok:
        row_g = results['G'].rows[0]
        rank_g = int(row_g['numerical_rank'])
        cond_g = float(row_g['condition_number'])
        below_g = int(row_g['below_tau'])
        
        if c_ok:
            row_c = results['C'].rows[0]
            rank_c = int(row_c['numerical_rank'])
            cond_c = float(row_c['condition_number'])
            
            if rank_g > rank_c:
                answers.append(f"**Q4 (Diversity FPS)**: FPS **improves** numerical rank ({rank_c} → {rank_g}).")
            elif cond_g < cond_c * 0.9:
                answers.append(f"**Q4**: Diversity FPS improves conditioning ({cond_c:.1e} → {cond_g:.1e}) but not rank.")
            else:
                answers.append(f"**Q4**: Diversity FPS does NOT significantly improve rank or conditioning.")
        else:
            answers.append(f"**Q4 (Diversity FPS)**: FPS run succeeded with rank={rank_g}, nullity={row_g['nullity']}, below_τ={below_g}.")
    elif 'G' in results and results['G'].success and results['G'].rows and is_failure(results['G'].rows[0]):
        answers.append("**Q4 (Diversity FPS)**: FPS run also failed (no_viable_points). Diversity selection cannot help when paper rho is too narrow.")
    
    for ans in answers:
        lines.append(ans)
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Local diagnostics suite for high-k SVD floor investigation")
    parser.add_argument("--output-dir", type=str, default="output_diagnostics_suite",
                        help="Output directory for run artifacts")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define run configurations
    configs = {
        'A': RunConfig(
            name='A',
            k_values=[1.5],
            description='Low-k sanity (k=1.5, paper mode) - expected NOT at floor',
        ),
        'B': RunConfig(
            name='B',
            k_values=[3.5],
            description='Mid-k transition (k=3.5, paper mode)',
        ),
        'C': RunConfig(
            name='C',
            k_values=[9.8],
            description='High-k paper mode (k=9.8) - expected floor behavior',
        ),
        'D': RunConfig(
            name='D',
            k_values=[9.8],
            rho_window_mode='adaptive_images',
            description='High-k adaptive_images mode (k=9.8)',
        ),
        'E': RunConfig(
            name='E',
            k_values=[9.80, 9.82],
            description='High-k nearby pair for stability (k=9.80, 9.82)',
        ),
        'F_15': RunConfig(
            name='F_15',
            k_values=[9.8],
            n_points=15,
            description='Sensitivity test: n_points=15 at k=9.8',
        ),
        'F_30': RunConfig(
            name='F_30',
            k_values=[9.8],
            n_points=30,
            description='Sensitivity test: n_points=30 at k=9.8',
        ),
        'F_60': RunConfig(
            name='F_60',
            k_values=[9.8],
            n_points=60,
            description='Sensitivity test: n_points=60 at k=9.8',
        ),
        'G': RunConfig(
            name='G',
            k_values=[9.8],
            n_points=60,
            diversity_fps=30,
            description='Diversity FPS test: sample 60 points, select 30 via FPS at k=9.8',
        ),
    }
    
    # Run all experiments
    results: Dict[str, RunResult] = {}
    for name, config in configs.items():
        results[name] = run_single(config, output_dir, seed=SEED)
    
    # Generate report
    report_text = generate_report(results, output_dir)
    
    report_path = Path("REPORT_LOCAL_DIAGNOSTICS.md")
    report_path.write_text(report_text)
    print(f"\n{'='*60}")
    print(f"Report written to: {report_path}")
    print('='*60)
    
    # Also print report to stdout
    print("\n" + report_text)


if __name__ == "__main__":
    main()
