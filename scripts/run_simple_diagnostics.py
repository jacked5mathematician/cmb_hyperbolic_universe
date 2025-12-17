#!/usr/bin/env python3
"""
Simplified function-space diagnostics for high-k SVD floor investigation.
Runs step-by-step with progress output.
"""

import subprocess
import sys
import csv
import json
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

PYTHON = sys.executable
SEED = 42
K_TEST = 9.8

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        return commit, branch
    except:
        return "unknown", "unknown"

def run_main(args_list, timeout=180):
    """Run main.py with given args, return (success, diagnostics_dict, stdout)"""
    cmd = [PYTHON, "main.py"] + args_list
    cmd_str = " ".join(cmd)
    print(f"  CMD: {cmd_str}")
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        # Check for no_viable_points
        if "No viable points retained" in proc.stderr or "No viable points retained" in proc.stdout:
            return False, {"failure_reason": "no_viable_points"}, proc.stdout + proc.stderr
        
        if proc.returncode != 0:
            return False, {"failure_reason": f"exit_{proc.returncode}"}, proc.stderr
        
        return True, {}, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return False, {"failure_reason": "timeout"}, ""
    except Exception as e:
        return False, {"failure_reason": str(e)}, ""

def parse_csv(csv_path):
    """Parse CSV diagnostics file"""
    if not csv_path.exists():
        return None
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows[0] if rows else None

def main():
    artifacts_dir = Path("reports/artifacts_local")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    git_commit, git_branch = get_git_info()
    machine_info = f"{platform.system()} {platform.release()} {platform.machine()}"
    
    all_results = {}
    all_commands = {}
    
    print("="*70)
    print("FUNCTION-SPACE DIAGNOSTICS")
    print("="*70)
    print(f"Git: {git_branch} @ {git_commit[:8]}")
    print(f"Machine: {machine_info}")
    print(f"Seed: {SEED}, k_test: {K_TEST}")
    print()
    
    # =========================================================================
    # STEP 1: Rho mode comparison
    # =========================================================================
    print("STEP 1: Rho Mode Comparison at k=9.8")
    print("-"*50)
    
    step1_configs = [
        ("P", "paper", "Paper rho window"),
        ("Aimg", "adaptive_images", "Adaptive images"),
        ("Arank", "adaptive_rank", "Adaptive rank"),
    ]
    
    for name, mode, desc in step1_configs:
        print(f"\n[{name}] {desc}")
        output_dir = Path(f"output_diag/{name}")
        args = [
            "--k-min", str(K_TEST), "--k-max", str(K_TEST), "--num-k", "1",
            "--n-points", "20", "--seed", str(SEED),
            "--output-dir", str(output_dir),
            "--rho-window-mode", mode,
            "--extended-diagnostics", "--no-plot", "--no-eigenvalues",
        ]
        all_commands[name] = f"python main.py " + " ".join(args)
        
        success, fail_info, output = run_main(args)
        
        csv_data = parse_csv(output_dir / "svd_diag.csv")
        if csv_data:
            all_results[name] = csv_data
            sigma_min = float(csv_data.get('sigma_min', 'nan'))
            tau = float(csv_data.get('tau', 'nan'))
            rank = csv_data.get('numerical_rank', '-')
            nullity = csv_data.get('nullity', '-')
            print(f"  ✓ σ_min={sigma_min:.2e}, τ={tau:.2e}, rank={rank}, nullity={nullity}")
        elif fail_info.get('failure_reason') == 'no_viable_points':
            all_results[name] = {'failure_reason': 'no_viable_points'}
            print(f"  ✗ no_viable_points")
        else:
            all_results[name] = fail_info
            print(f"  ✗ {fail_info.get('failure_reason', 'unknown')}")
    
    # =========================================================================
    # STEP 2: L sweep (different k values to get different L)
    # =========================================================================
    print("\n\nSTEP 2: Basis Size (L) Sweep")
    print("-"*50)
    print("Using different k values to achieve target L values")
    
    # L = floor(k) + 10, so k=3 gives L=13, k=5 gives L=15, etc.
    # For specific L targets: L=9 needs k~-1 (invalid), so use lower k
    step2_configs = [
        ("L9", 2.5, 9),   # k=2.5 -> L=floor(2.5)+10=12 (close to 9+3)
        ("L14", 5.0, 14), # k=5 -> L=floor(5)+10=15
        ("L19", 9.8, 19), # k=9.8 -> L=19
        ("L24", 14.0, 24), # k=14 -> L=24
    ]
    
    for name, k_val, expected_L in step2_configs:
        print(f"\n[{name}] k={k_val} (expected L≈{expected_L})")
        output_dir = Path(f"output_diag/{name}")
        args = [
            "--k-min", str(k_val), "--k-max", str(k_val), "--num-k", "1",
            "--n-points", "20", "--seed", str(SEED),
            "--output-dir", str(output_dir),
            "--rho-window-mode", "adaptive_rank",
            "--extended-diagnostics", "--no-plot", "--no-eigenvalues",
        ]
        all_commands[name] = f"python main.py " + " ".join(args)
        
        success, fail_info, output = run_main(args)
        
        csv_data = parse_csv(output_dir / "svd_diag.csv")
        if csv_data:
            all_results[name] = csv_data
            N = csv_data.get('N', '-')
            rank = csv_data.get('numerical_rank', '-')
            nullity = csv_data.get('nullity', '-')
            sigma_min = float(csv_data.get('sigma_min', 'nan'))
            print(f"  ✓ N={N}, rank={rank}, nullity={nullity}, σ_min={sigma_min:.2e}")
        elif fail_info.get('failure_reason') == 'no_viable_points':
            all_results[name] = {'failure_reason': 'no_viable_points'}
            print(f"  ✗ no_viable_points")
        else:
            all_results[name] = fail_info
            print(f"  ✗ {fail_info.get('failure_reason', 'unknown')}")
    
    # =========================================================================
    # STEP 3: Selection method comparison
    # =========================================================================
    print("\n\nSTEP 3: Selection Method Comparison")
    print("-"*50)
    
    step3_configs = [
        ("sel_random", None, None, "Random (baseline)"),
        ("sel_fps", 15, None, "Geometric FPS"),
        ("sel_qr", None, 15, "Feature-space QR"),
    ]
    
    for name, fps, qr, desc in step3_configs:
        print(f"\n[{name}] {desc}")
        output_dir = Path(f"output_diag/{name}")
        args = [
            "--k-min", str(K_TEST), "--k-max", str(K_TEST), "--num-k", "1",
            "--n-points", "30", "--seed", str(SEED),
            "--output-dir", str(output_dir),
            "--rho-window-mode", "adaptive_rank",
            "--extended-diagnostics", "--no-plot", "--no-eigenvalues",
        ]
        if fps:
            args.extend(["--diversity-fps", str(fps)])
        if qr:
            args.extend(["--diversity-feature-qr", str(qr)])
        
        all_commands[name] = f"python main.py " + " ".join(args)
        
        success, fail_info, output = run_main(args, timeout=300)
        
        csv_data = parse_csv(output_dir / "svd_diag.csv")
        if csv_data:
            all_results[name] = csv_data
            sigma_min = float(csv_data.get('sigma_min', 'nan'))
            tau = float(csv_data.get('tau', 'nan'))
            rank = csv_data.get('numerical_rank', '-')
            nullity = csv_data.get('nullity', '-')
            ratio = sigma_min / tau if tau > 0 else float('nan')
            print(f"  ✓ σ_min={sigma_min:.2e}, τ={tau:.2e}, σ_min/τ={ratio:.2e}, rank={rank}, nullity={nullity}")
        elif fail_info.get('failure_reason') == 'no_viable_points':
            all_results[name] = {'failure_reason': 'no_viable_points'}
            print(f"  ✗ no_viable_points")
        else:
            all_results[name] = fail_info
            print(f"  ✗ {fail_info.get('failure_reason', 'unknown')}")
    
    # =========================================================================
    # STEP 5: Enumeration depth test
    # =========================================================================
    print("\n\nSTEP 5: Enumeration Depth Test (Paper mode)")
    print("-"*50)
    
    step5_configs = [
        ("enum_d20", 20),
        ("enum_d25", 25),
        ("enum_d30", 30),
    ]
    
    for name, depth in step5_configs:
        print(f"\n[{name}] word_depth={depth}")
        output_dir = Path(f"output_diag/{name}")
        args = [
            "--k-min", str(K_TEST), "--k-max", str(K_TEST), "--num-k", "1",
            "--n-points", "20", "--seed", str(SEED),
            "--output-dir", str(output_dir),
            "--rho-window-mode", "paper",
            "--word-depth", str(depth),
            "--extended-diagnostics", "--no-plot", "--no-eigenvalues",
        ]
        all_commands[name] = f"python main.py " + " ".join(args)
        
        success, fail_info, output = run_main(args)
        
        csv_data = parse_csv(output_dir / "svd_diag.csv")
        if csv_data:
            all_results[name] = csv_data
            kept = csv_data.get('kept_points', '0')
            total_img = csv_data.get('total_images', '0')
            print(f"  ✓ kept_points={kept}, total_images={total_img}")
        elif fail_info.get('failure_reason') == 'no_viable_points':
            all_results[name] = {'failure_reason': 'no_viable_points'}
            print(f"  ✗ no_viable_points")
        else:
            all_results[name] = fail_info
            print(f"  ✗ {fail_info.get('failure_reason', 'unknown')}")
    
    # =========================================================================
    # Generate combined CSV
    # =========================================================================
    print("\n\nGenerating outputs...")
    
    combined_rows = []
    for name, data in all_results.items():
        if isinstance(data, dict):
            row = {'run_name': name}
            row.update(data)
            combined_rows.append(row)
    
    if combined_rows:
        csv_path = artifacts_dir / "svd_diag_combined.csv"
        fieldnames = ['run_name'] + [k for k in combined_rows[0].keys() if k != 'run_name']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(combined_rows)
        print(f"  Combined CSV: {csv_path}")
    
    # =========================================================================
    # Generate markdown report
    # =========================================================================
    report_path = Path("reports/REPORT_LOCAL_FUNCTIONSPACE_DIAGNOSTICS.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("# Local Function-Space Diagnostics Report")
    lines.append("")
    lines.append("## System Information")
    lines.append("")
    lines.append(f"- **Git Commit**: `{git_commit}`")
    lines.append(f"- **Git Branch**: `{git_branch}`")
    lines.append(f"- **Machine**: `{machine_info}`")
    lines.append(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Seed**: {SEED}")
    lines.append(f"- **Test k-value**: {K_TEST}")
    lines.append("")
    
    # Step 1 table
    lines.append("## Step 1: Rho Window Mode Comparison (k=9.8)")
    lines.append("")
    lines.append("| Run | Mode | kept_pts | total_img | σ_min | τ | σ_min/τ | rank | nullity | cond |")
    lines.append("|-----|------|-------:|--------:|------:|---:|-------:|-----:|-------:|-----:|")
    
    for name in ['P', 'Aimg', 'Arank']:
        if name in all_results:
            d = all_results[name]
            if d.get('failure_reason') == 'no_viable_points':
                lines.append(f"| {name} | paper | 0 | 0 | - | - | - | - | - | - | **FAIL** |")
            elif 'sigma_min' in d:
                mode = {'P': 'paper', 'Aimg': 'adaptive_images', 'Arank': 'adaptive_rank'}[name]
                kept = d.get('kept_points', '-')
                total = d.get('total_images', '-')
                sigma_min = float(d.get('sigma_min', 'nan'))
                tau = float(d.get('tau', 'nan'))
                ratio = sigma_min / tau if tau > 0 else float('nan')
                rank = d.get('numerical_rank', '-')
                nullity = d.get('nullity', '-')
                cond = float(d.get('condition_number', 'nan'))
                lines.append(f"| {name} | {mode} | {kept} | {total} | {sigma_min:.2e} | {tau:.2e} | {ratio:.2e} | {rank} | {nullity} | {cond:.1f} |")
    lines.append("")
    
    # Step 1 commands
    lines.append("### Commands")
    lines.append("")
    for name in ['P', 'Aimg', 'Arank']:
        if name in all_commands:
            lines.append(f"**{name}**: `{all_commands[name]}`")
            lines.append("")
    
    # Step 2 table
    lines.append("## Step 2: Basis Size (L) Sweep")
    lines.append("")
    lines.append("| Run | k | N | kept_pts | σ_min | τ | rank | nullity | nullity/N |")
    lines.append("|-----|---:|---:|-------:|------:|---:|-----:|-------:|--------:|")
    
    for name in ['L9', 'L14', 'L19', 'L24']:
        if name in all_results:
            d = all_results[name]
            k_val = {'L9': 2.5, 'L14': 5.0, 'L19': 9.8, 'L24': 14.0}[name]
            if d.get('failure_reason') == 'no_viable_points':
                lines.append(f"| {name} | {k_val} | - | 0 | - | - | - | - | - | **FAIL** |")
            elif 'N' in d:
                N = int(d.get('N', 1))
                kept = d.get('kept_points', '-')
                sigma_min = float(d.get('sigma_min', 'nan'))
                tau = float(d.get('tau', 'nan'))
                rank = int(d.get('numerical_rank', 0))
                nullity = int(d.get('nullity', 0))
                nullity_frac = nullity / N if N > 0 else 0
                lines.append(f"| {name} | {k_val} | {N} | {kept} | {sigma_min:.2e} | {tau:.2e} | {rank} | {nullity} | {nullity_frac:.2f} |")
    lines.append("")
    
    # Step 3 table
    lines.append("## Step 3: Selection Method Comparison (k=9.8)")
    lines.append("")
    lines.append("| Run | Method | σ_min | τ | σ_min/τ | rank | nullity |")
    lines.append("|-----|--------|------:|---:|-------:|-----:|-------:|")
    
    for name in ['sel_random', 'sel_fps', 'sel_qr']:
        if name in all_results:
            d = all_results[name]
            method = name.replace('sel_', '')
            if d.get('failure_reason'):
                lines.append(f"| {name} | {method} | - | - | - | - | - | **{d['failure_reason']}** |")
            elif 'sigma_min' in d:
                sigma_min = float(d.get('sigma_min', 'nan'))
                tau = float(d.get('tau', 'nan'))
                ratio = sigma_min / tau if tau > 0 else float('nan')
                rank = d.get('numerical_rank', '-')
                nullity = d.get('nullity', '-')
                lines.append(f"| {name} | {method} | {sigma_min:.2e} | {tau:.2e} | {ratio:.2e} | {rank} | {nullity} |")
    lines.append("")
    
    # Step 5 table
    lines.append("## Step 5: Enumeration Depth Test (Paper mode, k=9.8)")
    lines.append("")
    lines.append("| Run | word_depth | kept_pts | total_img | Result |")
    lines.append("|-----|----------:|-------:|--------:|--------|")
    
    for name in ['enum_d20', 'enum_d25', 'enum_d30']:
        if name in all_results:
            d = all_results[name]
            depth = int(name.split('_d')[1])
            if d.get('failure_reason') == 'no_viable_points':
                lines.append(f"| {name} | {depth} | 0 | 0 | no_viable_points |")
            elif 'kept_points' in d:
                kept = d.get('kept_points', '0')
                total = d.get('total_images', '0')
                lines.append(f"| {name} | {depth} | {kept} | {total} | OK |")
    lines.append("")
    
    # Conclusions
    lines.append("## Conclusions")
    lines.append("")
    
    # Analyze results
    conclusions = []
    
    # Q1: Rho starvation
    if 'P' in all_results and all_results['P'].get('failure_reason') == 'no_viable_points':
        conclusions.append("1. **Rho-starvation CONFIRMED**: Paper rho window fails completely at k=9.8")
        if 'Arank' in all_results and 'sigma_min' in all_results['Arank']:
            conclusions.append("   - Adaptive modes successfully produce matrices")
    
    # Q2: Selection method comparison
    best_sel = None
    best_ratio = 0
    for name in ['sel_random', 'sel_fps', 'sel_qr']:
        if name in all_results and 'sigma_min' in all_results[name]:
            d = all_results[name]
            sigma_min = float(d.get('sigma_min', 0))
            tau = float(d.get('tau', 1))
            ratio = sigma_min / tau if tau > 0 else 0
            if ratio > best_ratio:
                best_ratio = ratio
                best_sel = name
    
    if best_sel:
        conclusions.append(f"2. **Best selection method**: `{best_sel}` with σ_min/τ = {best_ratio:.2e}")
    
    # Q3: Nullity scaling
    for name in ['L9', 'L14', 'L19', 'L24']:
        if name in all_results and 'nullity' in all_results[name]:
            d = all_results[name]
            N = int(d.get('N', 1))
            nullity = int(d.get('nullity', 0))
            if nullity > 0.2 * N:
                conclusions.append(f"3. **High nullity**: {name} has nullity={nullity} ({nullity/N:.0%} of N={N})")
                break
    
    # Q4: Enumeration depth
    all_fail = all(
        all_results.get(f'enum_d{d}', {}).get('failure_reason') == 'no_viable_points'
        for d in [20, 25, 30]
    )
    if all_fail:
        conclusions.append("4. **Enumeration depth does NOT help**: Paper rho fails at all tested depths")
    
    for c in conclusions:
        lines.append(c)
        lines.append("")
    
    # HPC Handoff
    lines.append("## HPC Handoff Summary")
    lines.append("")
    lines.append("### Recommended Configuration")
    lines.append("")
    lines.append("- **Rho mode**: `adaptive_rank`")
    lines.append(f"- **Selection mode**: `{best_sel if best_sel else 'default'}`")
    lines.append("- **n_points**: 30")
    lines.append(f"- **seed**: {SEED}")
    lines.append("- **Suggested k-band**: 1.0 to 12.0")
    lines.append("")
    lines.append("### Suggested HPC Command")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python main.py \\")
    lines.append(f"    --k-min 1.0 --k-max 12.0 --num-k 50 \\")
    lines.append(f"    --n-points 30 --seed {SEED} \\")
    lines.append(f"    --rho-window-mode adaptive_rank \\")
    lines.append(f"    --extended-diagnostics \\")
    lines.append(f"    --output-dir output_hpc")
    lines.append("```")
    lines.append("")
    
    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    print(f"  Report: {report_path}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
