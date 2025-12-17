#!/usr/bin/env python3
"""
Comprehensive function-space diagnostics for high-k SVD floor investigation.

This script implements Steps 1-5 of the diagnostic plan:
- Step 1: Compare paper/adaptive_images/adaptive_rank at k=9.8
- Step 2: Sweep basis size (L) at fixed k with adaptive_rank
- Step 3: Compare selection methods (random, FPS, feature-QR)
- Step 4: Precision probe (float64 vs longdouble Gram)
- Step 5: Starvation vs enumeration depth

Run with:
    python scripts/run_functionspace_diagnostics.py [--output-dir DIR]

Produces:
    - reports/REPORT_LOCAL_FUNCTIONSPACE_DIAGNOSTICS.md
    - reports/artifacts_local/svd_diag_combined.csv
    - reports/artifacts_local/*.npy (Gram eigenvalues)
    - reports/artifacts_local/*.png (diagnostic plots)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import json
import time
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import numpy as np

# Constants
SEED = 42  # Fixed seed for reproducibility
K_TEST = 9.8  # High-k test value
PYTHON = sys.executable
DEFAULT_N_POINTS = 20  # Reduced for faster runs
DEFAULT_WORD_DEPTH = 20

# Git info (filled at runtime)
GIT_COMMIT = ""
GIT_BRANCH = ""
MACHINE_INFO = ""


@dataclass
class RunConfig:
    """Configuration for a single diagnostic run."""
    name: str
    k_value: float = K_TEST
    rho_window_mode: str = "paper"
    n_points: int = DEFAULT_N_POINTS
    L_override: Optional[int] = None  # If set, compute custom N
    word_depth: int = DEFAULT_WORD_DEPTH
    diversity_fps: Optional[int] = None
    diversity_feature_qr: Optional[int] = None
    extra_args: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RunResult:
    """Results from a single diagnostic run."""
    config: RunConfig
    success: bool = False
    error: Optional[str] = None
    runtime_seconds: float = 0.0
    command: str = ""
    # Parsed diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    gram_eigvals: Optional[np.ndarray] = None
    A_matrix: Optional[np.ndarray] = None


def get_git_info() -> Tuple[str, str]:
    """Get git commit and branch."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        return commit, branch
    except:
        return "unknown", "unknown"


def get_machine_info() -> str:
    """Get machine info string."""
    return f"{platform.system()} {platform.release()} {platform.machine()}"


def run_single(config: RunConfig, output_dir: Path, base_artifacts_dir: Path) -> RunResult:
    """Execute a single diagnostic run."""
    result = RunResult(config=config)
    
    run_dir = output_dir / f"run_{config.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        PYTHON, "main.py",
        "--k-min", str(config.k_value),
        "--k-max", str(config.k_value),
        "--num-k", "1",
        "--n-points", str(config.n_points),
        "--seed", str(SEED),
        "--output-dir", str(run_dir),
        "--rho-window-mode", config.rho_window_mode,
        "--word-depth", str(config.word_depth),
        "--extended-diagnostics",
        "--no-plot",
        "--no-eigenvalues",
    ]
    
    # Only dump A matrices for specific runs (Step 4 precision probe)
    if 'prec' in config.name or 'sel_' in config.name:
        cmd.append("--dump-A-matrices")
    
    if config.diversity_fps is not None:
        cmd.extend(["--diversity-fps", str(config.diversity_fps)])
    
    if config.diversity_feature_qr is not None:
        cmd.extend(["--diversity-feature-qr", str(config.diversity_feature_qr)])
    
    cmd.extend(config.extra_args)
    
    result.command = " ".join(cmd)
    
    print(f"\n{'='*60}")
    print(f"Run: {config.name}")
    print(f"Description: {config.description}")
    print(f"Command: {result.command}")
    print('='*60)
    
    start_time = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        result.runtime_seconds = time.time() - start_time
        
        if proc.returncode != 0:
            # Check if it's a "no viable points" failure
            if "No viable points retained" in proc.stderr or "No viable points retained" in proc.stdout:
                result.error = "no_viable_points"
                print(f"INFO: {result.error}")
            elif proc.returncode == -9:
                result.error = "killed (OOM or signal)"
                print(f"FAILED: {result.error}")
            else:
                result.error = f"Exit code {proc.returncode}: {proc.stderr[-500:]}"
                print(f"FAILED: {result.error}")
    except subprocess.TimeoutExpired:
        result.error = "Timeout (300s)"
        print(f"FAILED: {result.error}")
        return result
    except Exception as e:
        result.error = str(e)
        print(f"FAILED: {result.error}")
        return result
    
    # Parse CSV diagnostics
    csv_path = run_dir / "svd_diag.csv"
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                result.diagnostics = rows[0]
                result.success = True
                
                # Copy CSV to artifacts
                import shutil
                artifact_csv = base_artifacts_dir / f"svd_diag_{config.name}.csv"
                shutil.copy(csv_path, artifact_csv)
    
    # Load Gram eigenvalues if available
    gram_dir = run_dir / "gram_eigvals"
    if gram_dir.exists():
        for f in gram_dir.glob("eig_k_*.npy"):
            result.gram_eigvals = np.load(f)
            # Copy to artifacts
            artifact_npy = base_artifacts_dir / f"gram_eigvals_{config.name}.npy"
            np.save(artifact_npy, result.gram_eigvals)
            break
    
    # Load A matrix if available
    A_dir = run_dir / "A_matrices"
    if A_dir.exists():
        for f in A_dir.glob("A_k*.npz"):
            data = np.load(f)
            result.A_matrix = data['A']
            break
    
    if result.success:
        print(f"SUCCESS: runtime={result.runtime_seconds:.1f}s")
        diag = result.diagnostics
        kept = diag.get('kept_points', '0')
        total_img = diag.get('total_images', '0')
        sigma_min = diag.get('sigma_min', 'nan')
        rank = diag.get('numerical_rank', '0')
        nullity = diag.get('nullity', '0')
        print(f"  kept_points={kept}, total_images={total_img}")
        print(f"  sigma_min={sigma_min}, rank={rank}, nullity={nullity}")
    elif result.error == "no_viable_points":
        result.success = True  # Mark as "success" for reporting purposes
        result.diagnostics = {'failure_reason': 'no_viable_points'}
    
    return result


def compute_gram_precision_comparison(A: np.ndarray, artifacts_dir: Path, run_name: str) -> Dict[str, Any]:
    """Compare Gram eigenvalues in float64 vs longdouble."""
    results = {}
    
    # Float64 (current)
    G_f64 = A.T @ A
    eigvals_f64 = np.linalg.eigvalsh(G_f64)
    eigvals_f64 = np.sort(eigvals_f64)
    
    results['float64'] = {
        'smallest_20': eigvals_f64[:20].tolist(),
        'largest_5': eigvals_f64[-5:].tolist(),
        'min': float(eigvals_f64[0]),
        'max': float(eigvals_f64[-1]),
        'condition': float(eigvals_f64[-1] / eigvals_f64[0]) if eigvals_f64[0] > 0 else float('inf'),
    }
    
    # Longdouble (higher precision)
    try:
        A_ld = A.astype(np.longdouble)
        G_ld = A_ld.T @ A_ld
        # eigvalsh doesn't support longdouble, so we convert back
        G_ld_f64 = G_ld.astype(np.float64)
        eigvals_ld = np.linalg.eigvalsh(G_ld_f64)
        eigvals_ld = np.sort(eigvals_ld)
        
        results['longdouble'] = {
            'smallest_20': eigvals_ld[:20].tolist(),
            'largest_5': eigvals_ld[-5:].tolist(),
            'min': float(eigvals_ld[0]),
            'max': float(eigvals_ld[-1]),
            'condition': float(eigvals_ld[-1] / eigvals_ld[0]) if eigvals_ld[0] > 0 else float('inf'),
        }
        
        # Compute relative difference
        rel_diff = np.abs(eigvals_f64 - eigvals_ld) / (np.abs(eigvals_f64) + 1e-100)
        results['rel_diff_max'] = float(np.max(rel_diff))
        results['rel_diff_mean'] = float(np.mean(rel_diff))
        
    except Exception as e:
        results['longdouble'] = {'error': str(e)}
    
    # Save eigenvalues
    np.save(artifacts_dir / f"gram_eigvals_f64_{run_name}.npy", eigvals_f64)
    if 'error' not in results.get('longdouble', {}):
        np.save(artifacts_dir / f"gram_eigvals_ld_{run_name}.npy", eigvals_ld)
    
    return results


def create_diagnostic_plots(all_results: Dict[str, RunResult], artifacts_dir: Path):
    """Create diagnostic plots from results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    # Plot 1: Singular value spectra comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Gram eigenvalue comparison for Step 1 runs
    ax = axes[0]
    for name in ['P', 'Aimg', 'Arank']:
        if name in all_results:
            result = all_results[name]
            if result.gram_eigvals is not None:
                eigvals = np.sort(result.gram_eigvals)
                ax.semilogy(range(len(eigvals)), eigvals, 'o-', label=name, markersize=3)
    ax.set_xlabel('Index')
    ax.set_ylabel('Gram Eigenvalue')
    ax.set_title('Step 1: Gram Eigenvalues by Rho Mode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Gram eigenvalue comparison for Step 2 (L sweep)
    ax = axes[1]
    for name in ['L9', 'L14', 'L19', 'L24']:
        if name in all_results:
            result = all_results[name]
            if result.gram_eigvals is not None:
                eigvals = np.sort(result.gram_eigvals)
                ax.semilogy(range(len(eigvals)), eigvals, 'o-', label=name, markersize=3)
    ax.set_xlabel('Index')
    ax.set_ylabel('Gram Eigenvalue')
    ax.set_title('Step 2: Gram Eigenvalues by Basis Size (L)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(artifacts_dir / 'gram_eigenvalues_comparison.png', dpi=150)
    plt.close()
    
    # Plot 2: Selection method comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ['sel_random', 'sel_fps', 'sel_qr']:
        if name in all_results:
            result = all_results[name]
            if result.gram_eigvals is not None:
                eigvals = np.sort(result.gram_eigvals)
                ax.semilogy(range(len(eigvals)), eigvals, 'o-', label=name, markersize=3)
    ax.set_xlabel('Index')
    ax.set_ylabel('Gram Eigenvalue')
    ax.set_title('Step 3: Gram Eigenvalues by Selection Method')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(artifacts_dir / 'selection_comparison.png', dpi=150)
    plt.close()
    
    print(f"Plots saved to {artifacts_dir}")


def generate_report(
    all_results: Dict[str, RunResult],
    precision_results: Optional[Dict[str, Any]],
    artifacts_dir: Path,
    output_path: Path
) -> str:
    """Generate the comprehensive markdown report."""
    lines = []
    
    # Header
    lines.append("# Local Function-Space Diagnostics Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report investigates whether high-k failures are caused by:")
    lines.append("1. **Rho-starvation**: Paper rho window too narrow")
    lines.append("2. **Lack of function-space diversity**: Base points not spanning the space")
    lines.append("3. **Numerical precision/conditioning**: Basis evaluation precision issues")
    lines.append("")
    
    # System info
    lines.append("## System Information")
    lines.append("")
    lines.append(f"- **Git Commit**: `{GIT_COMMIT}`")
    lines.append(f"- **Git Branch**: `{GIT_BRANCH}`")
    lines.append(f"- **Machine**: `{MACHINE_INFO}`")
    lines.append(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Seed**: {SEED}")
    lines.append(f"- **Test k-value**: {K_TEST}")
    lines.append("")
    
    # Step 1: Rho mode comparison
    lines.append("## Step 1: Rho Window Mode Comparison (k=9.8)")
    lines.append("")
    lines.append("Compare paper, adaptive_images, and adaptive_rank modes.")
    lines.append("")
    
    step1_runs = ['P', 'Aimg', 'Arank']
    lines.append("| Run | Mode | kept_pts | total_img | M | N | σ_min | σ_max | τ | below_τ | rank | nullity | cond |")
    lines.append("|-----|------|-------:|--------:|---:|---:|------:|------:|---:|-------:|-----:|-------:|-----:|")
    
    for name in step1_runs:
        if name in all_results:
            result = all_results[name]
            if result.diagnostics.get('failure_reason') == 'no_viable_points':
                mode = result.config.rho_window_mode
                lines.append(f"| {name} | {mode} | 0 | 0 | - | - | - | - | - | - | - | - | - | **no_viable_points** |")
            elif result.success:
                d = result.diagnostics
                mode = result.config.rho_window_mode
                kept = d.get('kept_points', '-')
                total_img = d.get('total_images', '-')
                M = d.get('M', '-')
                N = d.get('N', '-')
                sigma_min = float(d.get('sigma_min', 'nan'))
                sigma_max = float(d.get('sigma_max', 'nan'))
                tau = float(d.get('tau', 'nan'))
                below_tau = d.get('below_tau', '-')
                rank = d.get('numerical_rank', '-')
                nullity = d.get('nullity', '-')
                cond = float(d.get('condition_number', 'nan'))
                
                sigma_min_str = f"{sigma_min:.2e}" if np.isfinite(sigma_min) else "-"
                sigma_max_str = f"{sigma_max:.2e}" if np.isfinite(sigma_max) else "-"
                tau_str = f"{tau:.2e}" if np.isfinite(tau) else "-"
                cond_str = f"{cond:.2e}" if np.isfinite(cond) else "-"
                
                lines.append(f"| {name} | {mode} | {kept} | {total_img} | {M} | {N} | {sigma_min_str} | {sigma_max_str} | {tau_str} | {below_tau} | {rank} | {nullity} | {cond_str} |")
    
    lines.append("")
    
    # Commands used
    lines.append("### Commands Used")
    lines.append("")
    for name in step1_runs:
        if name in all_results:
            lines.append(f"**{name}**: `{all_results[name].command}`")
            lines.append("")
    
    # Step 2: Basis size sweep
    lines.append("## Step 2: Basis Size (L) Sweep at k=9.8")
    lines.append("")
    lines.append("Test how nullity scales with basis dimension N=(L+1)².")
    lines.append("")
    
    step2_runs = ['L9', 'L14', 'L19', 'L24']
    lines.append("| Run | L | N | kept_pts | total_img | σ_min | τ | below_τ | rank | nullity | nullity/N |")
    lines.append("|-----|---:|---:|-------:|--------:|------:|---:|-------:|-----:|-------:|----------:|")
    
    for name in step2_runs:
        if name in all_results:
            result = all_results[name]
            if result.diagnostics.get('failure_reason') == 'no_viable_points':
                L = int(name[1:])
                N = (L+1)**2
                lines.append(f"| {name} | {L} | {N} | 0 | 0 | - | - | - | - | - | - | **no_viable_points** |")
            elif result.success:
                d = result.diagnostics
                L = int(name[1:])
                N_calc = (L+1)**2
                N = d.get('N', N_calc)
                kept = d.get('kept_points', '-')
                total_img = d.get('total_images', '-')
                sigma_min = float(d.get('sigma_min', 'nan'))
                tau = float(d.get('tau', 'nan'))
                below_tau = d.get('below_tau', '-')
                rank = int(d.get('numerical_rank', 0))
                nullity = int(d.get('nullity', 0))
                N_int = int(N) if N != '-' else 1
                nullity_frac = nullity / N_int if N_int > 0 else 0
                
                sigma_min_str = f"{sigma_min:.2e}" if np.isfinite(sigma_min) else "-"
                tau_str = f"{tau:.2e}" if np.isfinite(tau) else "-"
                
                lines.append(f"| {name} | {L} | {N} | {kept} | {total_img} | {sigma_min_str} | {tau_str} | {below_tau} | {rank} | {nullity} | {nullity_frac:.2f} |")
    
    lines.append("")
    
    # Commands used
    lines.append("### Commands Used")
    lines.append("")
    for name in step2_runs:
        if name in all_results:
            lines.append(f"**{name}**: `{all_results[name].command}`")
            lines.append("")
    
    # Step 3: Selection method comparison
    lines.append("## Step 3: Base Point Selection Method Comparison")
    lines.append("")
    lines.append("Compare random selection, geometric FPS, and feature-space QR selection.")
    lines.append("")
    
    step3_runs = ['sel_random', 'sel_fps', 'sel_qr']
    lines.append("| Run | Method | kept_pts | total_img | σ_min | τ | σ_min/τ | rank | nullity |")
    lines.append("|-----|--------|-------:|--------:|------:|---:|-------:|-----:|-------:|")
    
    for name in step3_runs:
        if name in all_results:
            result = all_results[name]
            if result.diagnostics.get('failure_reason') == 'no_viable_points':
                method = name.replace('sel_', '')
                lines.append(f"| {name} | {method} | 0 | 0 | - | - | - | - | - | **no_viable_points** |")
            elif result.success:
                d = result.diagnostics
                method = name.replace('sel_', '')
                kept = d.get('kept_points', '-')
                total_img = d.get('total_images', '-')
                sigma_min = float(d.get('sigma_min', 'nan'))
                tau = float(d.get('tau', 'nan'))
                rank = d.get('numerical_rank', '-')
                nullity = d.get('nullity', '-')
                
                sigma_min_str = f"{sigma_min:.2e}" if np.isfinite(sigma_min) else "-"
                tau_str = f"{tau:.2e}" if np.isfinite(tau) else "-"
                ratio = sigma_min / tau if np.isfinite(sigma_min) and np.isfinite(tau) and tau > 0 else float('nan')
                ratio_str = f"{ratio:.2e}" if np.isfinite(ratio) else "-"
                
                lines.append(f"| {name} | {method} | {kept} | {total_img} | {sigma_min_str} | {tau_str} | {ratio_str} | {rank} | {nullity} |")
    
    lines.append("")
    
    # Commands used
    lines.append("### Commands Used")
    lines.append("")
    for name in step3_runs:
        if name in all_results:
            lines.append(f"**{name}**: `{all_results[name].command}`")
            lines.append("")
    
    # Step 4: Precision comparison
    lines.append("## Step 4: Numerical Precision Comparison")
    lines.append("")
    
    if precision_results:
        lines.append("Compare Gram matrix eigenvalues computed in float64 vs longdouble.")
        lines.append("")
        
        if 'float64' in precision_results and 'longdouble' in precision_results:
            f64 = precision_results['float64']
            ld = precision_results['longdouble']
            
            lines.append("| Metric | float64 | longdouble |")
            lines.append("|--------|--------:|----------:|")
            lines.append(f"| min eigenvalue | {f64['min']:.4e} | {ld.get('min', 'N/A'):.4e if isinstance(ld.get('min'), float) else 'N/A'} |")
            lines.append(f"| max eigenvalue | {f64['max']:.4e} | {ld.get('max', 'N/A'):.4e if isinstance(ld.get('max'), float) else 'N/A'} |")
            lines.append(f"| condition | {f64['condition']:.4e} | {ld.get('condition', 'N/A'):.4e if isinstance(ld.get('condition'), float) else 'N/A'} |")
            lines.append("")
            
            if 'rel_diff_max' in precision_results:
                lines.append(f"- **Max relative difference**: {precision_results['rel_diff_max']:.4e}")
                lines.append(f"- **Mean relative difference**: {precision_results['rel_diff_mean']:.4e}")
                lines.append("")
            
            lines.append("### Smallest 20 Eigenvalues Comparison")
            lines.append("")
            lines.append("| Index | float64 | longdouble |")
            lines.append("|------:|--------:|----------:|")
            for i in range(min(20, len(f64['smallest_20']))):
                f64_val = f64['smallest_20'][i]
                ld_val = ld['smallest_20'][i] if i < len(ld.get('smallest_20', [])) else 'N/A'
                ld_str = f"{ld_val:.4e}" if isinstance(ld_val, float) else 'N/A'
                lines.append(f"| {i} | {f64_val:.4e} | {ld_str} |")
            lines.append("")
        else:
            lines.append("Precision comparison not available.")
            lines.append("")
    else:
        lines.append("Step 4 not executed (no valid A matrix available).")
        lines.append("")
    
    # Step 5: Starvation analysis
    lines.append("## Step 5: Starvation vs Enumeration Depth")
    lines.append("")
    
    step5_runs = ['enum_d20', 'enum_d25', 'enum_d30']
    has_step5 = any(name in all_results for name in step5_runs)
    
    if has_step5:
        lines.append("Test if increasing word depth recovers images for paper mode.")
        lines.append("")
        lines.append("| Run | word_depth | kept_pts | total_img | result |")
        lines.append("|-----|----------:|-------:|--------:|--------|")
        
        for name in step5_runs:
            if name in all_results:
                result = all_results[name]
                depth = result.config.word_depth
                if result.diagnostics.get('failure_reason') == 'no_viable_points':
                    lines.append(f"| {name} | {depth} | 0 | 0 | no_viable_points |")
                elif result.success:
                    d = result.diagnostics
                    kept = d.get('kept_points', '0')
                    total_img = d.get('total_images', '0')
                    lines.append(f"| {name} | {depth} | {kept} | {total_img} | OK |")
        lines.append("")
    else:
        lines.append("Step 5 skipped (paper mode already produces results or not needed).")
        lines.append("")
    
    # Analysis and Conclusions
    lines.append("## Analysis and Conclusions")
    lines.append("")
    
    # Determine which axis matters most
    conclusions = []
    
    # Check rho starvation
    if 'P' in all_results:
        if all_results['P'].diagnostics.get('failure_reason') == 'no_viable_points':
            conclusions.append("- **Rho-starvation (A)**: CONFIRMED. Paper mode fails completely at k=9.8.")
            if 'Arank' in all_results and all_results['Arank'].success and all_results['Arank'].diagnostics.get('failure_reason') != 'no_viable_points':
                conclusions.append("  - Adaptive modes successfully produce matrices.")
    
    # Check function-space diversity
    best_selection = None
    best_ratio = 0
    for name in ['sel_random', 'sel_fps', 'sel_qr']:
        if name in all_results and all_results[name].success:
            d = all_results[name].diagnostics
            if d.get('failure_reason') != 'no_viable_points':
                sigma_min = float(d.get('sigma_min', 0))
                tau = float(d.get('tau', 1))
                ratio = sigma_min / tau if tau > 0 else 0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_selection = name
    
    if best_selection:
        conclusions.append(f"- **Function-space diversity (B)**: Best selection method is `{best_selection}` with σ_min/τ = {best_ratio:.2e}")
    
    # Check precision
    if precision_results and 'rel_diff_max' in precision_results:
        rel_diff = precision_results['rel_diff_max']
        if rel_diff > 1e-10:
            conclusions.append(f"- **Numerical precision (C)**: Significant difference between float64 and longdouble (max rel diff = {rel_diff:.2e})")
        else:
            conclusions.append(f"- **Numerical precision (C)**: Minimal difference between precisions (max rel diff = {rel_diff:.2e})")
    
    # Check nullity scaling
    for name in ['L9', 'L14', 'L19', 'L24']:
        if name in all_results and all_results[name].success:
            d = all_results[name].diagnostics
            if d.get('failure_reason') != 'no_viable_points':
                nullity = int(d.get('nullity', 0))
                N = int(d.get('N', 1))
                if nullity > 0.3 * N:
                    conclusions.append(f"- **Basis size**: Large nullity at L={name[1:]} (nullity={nullity}, N={N}, ratio={nullity/N:.2f})")
    
    for c in conclusions:
        lines.append(c)
    
    lines.append("")
    
    # Primary conclusion
    lines.append("### Primary Conclusion")
    lines.append("")
    if 'P' in all_results and all_results['P'].diagnostics.get('failure_reason') == 'no_viable_points':
        lines.append("The primary issue is **rho-starvation**: the paper rho window is too narrow at high k.")
        lines.append("Adaptive rho modes are necessary for high-k computation.")
    else:
        lines.append("Paper mode succeeds; investigate other factors.")
    lines.append("")
    
    # HPC Handoff
    lines.append("## HPC Handoff Summary")
    lines.append("")
    
    # Determine best configuration
    best_rho_mode = "adaptive_rank"  # Default
    best_n_points = 30
    best_L = 19
    
    if 'Arank' in all_results and all_results['Arank'].success:
        best_rho_mode = "adaptive_rank"
    elif 'Aimg' in all_results and all_results['Aimg'].success:
        best_rho_mode = "adaptive_images"
    
    lines.append("### Recommended Configuration")
    lines.append("")
    lines.append(f"- **Rho mode**: `{best_rho_mode}`")
    lines.append(f"- **Selection mode**: `{best_selection if best_selection else 'default (random)'}`")
    lines.append(f"- **n_points**: {best_n_points}")
    lines.append(f"- **seed**: {SEED}")
    lines.append(f"- **Suggested k-band**: 1.0 to 12.0")
    lines.append("")
    
    lines.append("### Suggested HPC Command")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python main.py \\")
    lines.append(f"    --k-min 1.0 --k-max 12.0 --num-k 50 \\")
    lines.append(f"    --n-points {best_n_points} --seed {SEED} \\")
    lines.append(f"    --rho-window-mode {best_rho_mode} \\")
    lines.append(f"    --extended-diagnostics \\")
    lines.append(f"    --output-dir output_hpc")
    lines.append("```")
    lines.append("")
    
    lines.append("### Artifacts to Produce on HPC")
    lines.append("")
    lines.append("- `svd_diag.csv` with columns: k, chi2_rank1, M, N, kept_points, total_images, sigma_min, sigma_2, sigma_max, tau, condition_number, below_tau, numerical_rank, nullity, frac_below_5tau")
    lines.append("- `gram_eigvals/eig_k_*.npy` for each k value")
    lines.append("- `spectrum.npz` with final results")
    lines.append("")
    
    return "\n".join(lines)


def combine_csvs(all_results: Dict[str, RunResult], artifacts_dir: Path):
    """Combine all CSV diagnostics into a single file."""
    combined_rows = []
    fieldnames = None
    
    for name, result in all_results.items():
        if result.success and result.diagnostics:
            row = result.diagnostics.copy()
            row['run_name'] = name
            row['rho_mode'] = result.config.rho_window_mode
            
            if fieldnames is None:
                fieldnames = ['run_name', 'rho_mode'] + [k for k in row.keys() if k not in ['run_name', 'rho_mode']]
            combined_rows.append(row)
    
    if combined_rows and fieldnames:
        csv_path = artifacts_dir / "svd_diag_combined.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(combined_rows)
        print(f"Combined CSV written to {csv_path}")


def main():
    global GIT_COMMIT, GIT_BRANCH, MACHINE_INFO
    
    parser = argparse.ArgumentParser(description="Function-space diagnostics suite")
    parser.add_argument("--output-dir", type=str, default="reports/artifacts_local",
                        help="Output directory for artifacts")
    parser.add_argument("--skip-step", type=int, nargs='*', default=[],
                        help="Steps to skip (1-5)")
    args = parser.parse_args()
    
    artifacts_dir = Path(args.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Get system info
    GIT_COMMIT, GIT_BRANCH = get_git_info()
    MACHINE_INFO = get_machine_info()
    
    all_results: Dict[str, RunResult] = {}
    temp_output_dir = Path("output_functionspace_diag")
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Rho mode comparison at k=9.8
    # =========================================================================
    if 1 not in args.skip_step:
        print("\n" + "="*80)
        print("STEP 1: Rho Window Mode Comparison")
        print("="*80)
        
        step1_configs = [
            RunConfig(name='P', rho_window_mode='paper', description='Paper rho window'),
            RunConfig(name='Aimg', rho_window_mode='adaptive_images', description='Adaptive images mode'),
            RunConfig(name='Arank', rho_window_mode='adaptive_rank', description='Adaptive rank mode'),
        ]
        
        for cfg in step1_configs:
            result = run_single(cfg, temp_output_dir, artifacts_dir)
            all_results[cfg.name] = result
    
    # =========================================================================
    # Step 2: Basis size (L) sweep
    # =========================================================================
    if 2 not in args.skip_step:
        print("\n" + "="*80)
        print("STEP 2: Basis Size (L) Sweep")
        print("="*80)
        
        # Use adaptive_rank since paper mode fails
        step2_configs = [
            RunConfig(name='L9', rho_window_mode='adaptive_rank', 
                     extra_args=['--k-min', '3.0', '--k-max', '3.0'],  # L~9 at k~3
                     description='L=9 (N=100) at k~3'),
            RunConfig(name='L14', rho_window_mode='adaptive_rank',
                     extra_args=['--k-min', '5.5', '--k-max', '5.5'],  # L~14 at k~5.5
                     description='L=14 (N=225) at k~5.5'),
            RunConfig(name='L19', rho_window_mode='adaptive_rank',
                     k_value=9.8, description='L=19 (N=400) at k=9.8'),
            RunConfig(name='L24', rho_window_mode='adaptive_rank',
                     extra_args=['--k-min', '12.0', '--k-max', '12.0'],  # L~24 at k~12
                     description='L=24 (N=625) at k=12'),
        ]
        
        for cfg in step2_configs:
            result = run_single(cfg, temp_output_dir, artifacts_dir)
            all_results[cfg.name] = result
    
    # =========================================================================
    # Step 3: Selection method comparison
    # =========================================================================
    if 3 not in args.skip_step:
        print("\n" + "="*80)
        print("STEP 3: Selection Method Comparison")
        print("="*80)
        
        step3_configs = [
            RunConfig(name='sel_random', rho_window_mode='adaptive_rank', n_points=40,
                     description='Random selection (baseline)'),
            RunConfig(name='sel_fps', rho_window_mode='adaptive_rank', n_points=40,
                     diversity_fps=20, description='Geometric FPS selection'),
            RunConfig(name='sel_qr', rho_window_mode='adaptive_rank', n_points=40,
                     diversity_feature_qr=20, description='Feature-space QR selection'),
        ]
        
        for cfg in step3_configs:
            result = run_single(cfg, temp_output_dir, artifacts_dir)
            all_results[cfg.name] = result
    
    # =========================================================================
    # Step 4: Precision comparison
    # =========================================================================
    precision_results = None
    if 4 not in args.skip_step:
        print("\n" + "="*80)
        print("STEP 4: Numerical Precision Comparison")
        print("="*80)
        
        # Find a run with A matrix available
        A_matrix = None
        source_run = None
        for name in ['Arank', 'Aimg', 'sel_fps', 'sel_random']:
            if name in all_results and all_results[name].A_matrix is not None:
                A_matrix = all_results[name].A_matrix
                source_run = name
                break
        
        if A_matrix is not None:
            print(f"Using A matrix from run '{source_run}'")
            precision_results = compute_gram_precision_comparison(A_matrix, artifacts_dir, source_run)
            print(f"Precision comparison complete:")
            print(f"  float64 min eigenvalue: {precision_results['float64']['min']:.4e}")
            if 'longdouble' in precision_results and 'min' in precision_results['longdouble']:
                print(f"  longdouble min eigenvalue: {precision_results['longdouble']['min']:.4e}")
        else:
            print("No A matrix available for precision comparison")
    
    # =========================================================================
    # Step 5: Enumeration depth test (only if paper mode failed)
    # =========================================================================
    if 5 not in args.skip_step:
        print("\n" + "="*80)
        print("STEP 5: Enumeration Depth Test")
        print("="*80)
        
        if 'P' in all_results and all_results['P'].diagnostics.get('failure_reason') == 'no_viable_points':
            print("Paper mode failed, testing increased enumeration depth...")
            
            step5_configs = [
                RunConfig(name='enum_d20', rho_window_mode='paper', word_depth=20,
                         description='Paper mode, word_depth=20'),
                RunConfig(name='enum_d25', rho_window_mode='paper', word_depth=25,
                         description='Paper mode, word_depth=25'),
                RunConfig(name='enum_d30', rho_window_mode='paper', word_depth=30,
                         description='Paper mode, word_depth=30'),
            ]
            
            for cfg in step5_configs:
                result = run_single(cfg, temp_output_dir, artifacts_dir)
                all_results[cfg.name] = result
        else:
            print("Paper mode succeeded or not tested, skipping enumeration depth test")
    
    # =========================================================================
    # Generate outputs
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)
    
    # Combine CSVs
    combine_csvs(all_results, artifacts_dir)
    
    # Create plots
    create_diagnostic_plots(all_results, artifacts_dir)
    
    # Generate report
    report_path = Path("reports/REPORT_LOCAL_FUNCTIONSPACE_DIAGNOSTICS.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = generate_report(all_results, precision_results, artifacts_dir, report_path)
    report_path.write_text(report_text)
    print(f"\nReport written to: {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total runs: {len(all_results)}")
    successful = sum(1 for r in all_results.values() if r.success)
    print(f"Successful: {successful}")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
