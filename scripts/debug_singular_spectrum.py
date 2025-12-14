#!/usr/bin/env python3
"""
Debug script to analyze singular value spectrum and diagnose eigenvalue sensitivity.

This script compares chi-squared behavior at expected eigenvalues vs off-eigenvalues
to verify that the constraint matrix exhibits eigenvalue-selective singularity.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_spectrum(spectrum_path: Path) -> dict:
    """Load spectrum data from .npz file."""
    data = np.load(spectrum_path)
    result = {key: data[key] for key in data.files}
    return result


def analyze_singular_values(data: dict, paper_eigenvalues: list[float] = None) -> None:
    """
    Analyze singular value behavior and eigenvalue sensitivity.
    
    Args:
        data: Dictionary containing spectrum data
        paper_eigenvalues: List of expected k values from paper Table I
    """
    k_values = data['k_values']
    sigma_min = data.get('sigma_min', None)
    sigma_max = data.get('sigma_max', None)
    chi2_rank_1 = data['chi2_rank_1']
    
    if sigma_min is None or sigma_max is None:
        print("ERROR: Spectrum does not contain sigma_min/sigma_max diagnostics.")
        print("Re-run with main.py to generate enhanced diagnostics.")
        return
    
    print("="*70)
    print("SINGULAR VALUE ANALYSIS")
    print("="*70)
    
    # Compute condition numbers
    condition = sigma_max / (sigma_min + 1e-30)
    
    print(f"\nK-value range: [{k_values.min():.3f}, {k_values.max():.3f}]")
    print(f"Number of k samples: {len(k_values)}")
    print(f"\nCondition number statistics:")
    print(f"  Min:    {condition.min():.2e}")
    print(f"  Max:    {condition.max():.2e}")
    print(f"  Median: {np.median(condition):.2e}")
    
    # Find chi-squared minima
    from scipy.signal import find_peaks
    
    # Invert to find minima as peaks
    peaks, properties = find_peaks(-chi2_rank_1, prominence=0.1*np.ptp(chi2_rank_1))
    
    print(f"\nDetected {len(peaks)} potential eigenvalue minima:")
    for i, idx in enumerate(peaks):
        k_at_min = k_values[idx]
        chi2_at_min = chi2_rank_1[idx]
        cond_at_min = condition[idx]
        print(f"  {i+1}. k={k_at_min:.3f}, χ²={chi2_at_min:.3e}, cond={cond_at_min:.3e}")
    
    # Compare to paper eigenvalues if provided
    if paper_eigenvalues:
        print(f"\n{'='*70}")
        print("COMPARISON TO PAPER EIGENVALUES")
        print("="*70)
        
        for k_paper in paper_eigenvalues:
            if k_paper < k_values.min() or k_paper > k_values.max():
                print(f"k={k_paper:.3f}: Outside sampled range")
                continue
            
            # Find nearest k value
            idx = np.argmin(np.abs(k_values - k_paper))
            k_actual = k_values[idx]
            chi2 = chi2_rank_1[idx]
            
            # Find local minimum within window
            window = 5
            start = max(0, idx - window)
            end = min(len(k_values), idx + window)
            local_min_idx = start + np.argmin(chi2_rank_1[start:end])
            k_local_min = k_values[local_min_idx]
            chi2_local_min = chi2_rank_1[local_min_idx]
            
            print(f"k_paper={k_paper:.3f} -> k_nearest={k_actual:.3f} (χ²={chi2:.3e}), "
                  f"k_local_min={k_local_min:.3f} (χ²={chi2_local_min:.3e})")
    
    # Analyze chi-squared dynamic range
    print(f"\n{'='*70}")
    print("CHI-SQUARED DYNAMIC RANGE")
    print("="*70)
    
    chi2_valid = chi2_rank_1[np.isfinite(chi2_rank_1)]
    if len(chi2_valid) > 0:
        print(f"Min chi²:  {chi2_valid.min():.3e}")
        print(f"Max chi²:  {chi2_valid.max():.3e}")
        print(f"Range:     {chi2_valid.max() / (chi2_valid.min() + 1e-30):.3e}x")
        print(f"Std dev:   {chi2_valid.std():.3e}")
        
        # Check for pathological flatness
        relative_variation = chi2_valid.std() / (chi2_valid.mean() + 1e-30)
        print(f"Relative variation (std/mean): {relative_variation:.3e}")
        
        if relative_variation < 0.01:
            print("\n⚠️  WARNING: Chi-squared shows very little variation!")
            print("    This suggests the matrix may not be sensitive to eigenvalues.")
            print("    Possible causes:")
            print("    - Insufficient ghost images per point")
            print("    - All base points rejected (kept_points ~0)")
            print("    - Fallback rho cutoffs too restrictive")
            print("    - Numerical conditioning issues")


def plot_diagnostics(data: dict, output_path: Path, paper_eigenvalues: list[float] = None) -> None:
    """Create diagnostic plots for singular value analysis."""
    k_values = data['k_values']
    sigma_min = data.get('sigma_min')
    sigma_max = data.get('sigma_max')
    chi2_rank_1 = data['chi2_rank_1']
    
    if sigma_min is None or sigma_max is None:
        print("Cannot create diagnostic plots without sigma_min/sigma_max.")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot 1: Chi-squared spectrum
    axes[0].plot(k_values, chi2_rank_1, 'b-', linewidth=1, label='Rank 1')
    if paper_eigenvalues:
        for k_ev in paper_eigenvalues:
            if k_values.min() <= k_ev <= k_values.max():
                axes[0].axvline(k_ev, color='r', alpha=0.3, linestyle='--', linewidth=0.5)
    axes[0].set_xlabel('k')
    axes[0].set_ylabel(r'$\chi^2$')
    axes[0].set_title(r'$\chi^2$ Spectrum (red lines: paper eigenvalues)')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Condition number
    condition = sigma_max / (sigma_min + 1e-30)
    axes[1].plot(k_values, condition, 'g-', linewidth=1)
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Condition Number (σ_max/σ_min)')
    axes[1].set_title('Matrix Condition Number')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Smallest singular value
    axes[2].plot(k_values, sigma_min, 'm-', linewidth=1)
    if paper_eigenvalues:
        for k_ev in paper_eigenvalues:
            if k_values.min() <= k_ev <= k_values.max():
                axes[2].axvline(k_ev, color='r', alpha=0.3, linestyle='--', linewidth=0.5)
    axes[2].set_xlabel('k')
    axes[2].set_ylabel(r'$\sigma_{min}$')
    axes[2].set_title('Smallest Singular Value (should show dips at eigenvalues)')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nDiagnostic plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug singular value spectrum and eigenvalue sensitivity"
    )
    parser.add_argument(
        '--spectrum',
        type=Path,
        required=True,
        help='Path to spectrum.npz file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Path for diagnostic plot (default: same dir as spectrum)'
    )
    parser.add_argument(
        '--paper-eigenvalues',
        type=str,
        default=None,
        help='Comma-separated list of expected k values from paper (e.g., "4.41,4.64,5.12")'
    )
    parser.add_argument(
        '--manifold',
        type=str,
        default=None,
        choices=['m188', 'm003_thurston', 'm003_weeks'],
        help='Use known eigenvalues for specified manifold'
    )
    
    args = parser.parse_args()
    
    if not args.spectrum.exists():
        print(f"ERROR: Spectrum file not found: {args.spectrum}")
        return 1
    
    # Load data
    data = load_spectrum(args.spectrum)
    
    # Get paper eigenvalues
    paper_eigenvalues = None
    if args.paper_eigenvalues:
        paper_eigenvalues = [float(x.strip()) for x in args.paper_eigenvalues.split(',')]
    elif args.manifold == 'm188':
        # From Table I in paper for m188(-1,1)
        # q² values: convert to k = sqrt(q² - 1)
        q2_values = [20.4, 22.6, 27.2, 30.2, 39.6, 46.2, 51.8, 55.3, 60.1, 
                     70.6, 75.5, 78.8, 80.9, 83.1, 86.0, 96.8, 98.0, 99.4]
        paper_eigenvalues = [np.sqrt(q2 - 1) for q2 in q2_values]
    elif args.manifold == 'm003_thurston':
        # From Table II for m003(-2,3)
        q2_values = [29.3, 33.5, 46.2, 47.8, 50.8, 59.1, 68.9, 73.8, 76.2, 
                     85.8, 95.1, 98.0, 100.1]
        paper_eigenvalues = [np.sqrt(q2 - 1) for q2 in q2_values]
    elif args.manifold == 'm003_weeks':
        # From Table III for m003(-3,1)
        q2_values = [27.8, 32.9, 43.0, 59.7, 66.3, 67.6, 69.7, 84.4, 90.5, 
                     93.9, 97.8]
        paper_eigenvalues = [np.sqrt(q2 - 1) for q2 in q2_values]
    
    # Analyze
    analyze_singular_values(data, paper_eigenvalues)
    
    # Plot
    if args.output:
        output_path = args.output
    else:
        output_path = args.spectrum.parent / 'singular_value_diagnostics.png'
    
    plot_diagnostics(data, output_path, paper_eigenvalues)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
