#!/usr/bin/env python
"""
Analyze chi-squared scale and identify which definition matches the paper.

This script loads a spectrum and analyzes the chi-squared values to determine
which definition produces O(1) values as shown in the paper's Figure 1.

Usage:
    python scripts/analyze_chi2_scale.py --spectrum path/to/spectrum.npz
"""
import argparse
import logging
from pathlib import Path
import sys

import numpy as np

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_chi2_definitions import recompute_chi2_with_definitions

LOGGER = logging.getLogger(__name__)


def analyze_chi2_scale(spectrum_path: Path, rank: int = 1):
    """
    Analyze chi-squared scale for different definitions.
    
    Args:
        spectrum_path: Path to spectrum.npz file
        rank: Which rank to analyze
    """
    # Load spectrum
    data = np.load(spectrum_path)
    spectrum_data = {key: data[key] for key in data.files}
    
    if "k_values" not in spectrum_data:
        raise ValueError("No k_values in spectrum file")
    
    k_values = spectrum_data["k_values"]
    
    # Recompute chi2 with different definitions
    chi2_dict = recompute_chi2_with_definitions(spectrum_data, rank=rank)
    
    print("=" * 70)
    print(f"Chi-squared Scale Analysis (Rank {rank})")
    print("=" * 70)
    print(f"Spectrum file: {spectrum_path}")
    print(f"k-value range: [{k_values[0]:.2f}, {k_values[-1]:.2f}]")
    print(f"Number of k-values: {len(k_values)}")
    print()
    
    # Analyze each definition
    for def_name, chi2_values in sorted(chi2_dict.items()):
        print(f"\n{def_name.upper()}:")
        print("-" * 70)
        
        # Filter finite values
        finite_vals = chi2_values[np.isfinite(chi2_values)]
        
        if len(finite_vals) == 0:
            print("  No finite values")
            continue
        
        # Statistics
        min_val = finite_vals.min()
        max_val = finite_vals.max()
        mean_val = finite_vals.mean()
        median_val = np.median(finite_vals)
        
        print(f"  Min:    {min_val:.6e}")
        print(f"  Max:    {max_val:.6e}")
        print(f"  Mean:   {mean_val:.6e}")
        print(f"  Median: {median_val:.6e}")
        
        # Order of magnitude
        if mean_val > 0:
            order = np.log10(mean_val)
            print(f"  Typical order of magnitude: ~10^{order:.1f}")
            
            # Check if O(1)
            if -1 <= order <= 1:
                print("  ✓ ORDER OF MAGNITUDE IS O(1) - LIKELY PAPER DEFINITION")
            elif order < -5:
                print("  ✗ Very small values (<<1) - unlikely paper definition")
            else:
                print(f"  ? Order ~10^{order:.0f} - may need further investigation")
        
        # Value distribution
        if len(finite_vals) > 1:
            q25, q75 = np.percentile(finite_vals, [25, 75])
            print(f"  25th percentile: {q25:.6e}")
            print(f"  75th percentile: {q75:.6e}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    
    # Find which definition produces O(1) values
    recommendations = []
    for def_name, chi2_values in chi2_dict.items():
        finite_vals = chi2_values[np.isfinite(chi2_values)]
        if len(finite_vals) > 0:
            mean_val = finite_vals.mean()
            if mean_val > 0:
                order = np.log10(mean_val)
                if -1 <= order <= 1:
                    recommendations.append((def_name, order, mean_val))
    
    if recommendations:
        print("\nThe following definitions produce O(1) scale values:")
        for def_name, order, mean_val in recommendations:
            print(f"  - {def_name}: mean={mean_val:.6e}, order=10^{order:.1f}")
        
        # Ratio is most likely since it's dimensionless and bounded
        if 'ratio' in [r[0] for r in recommendations]:
            print("\nBased on dimensionless nature and bounded range [0,1],")
            print("'ratio' definition is MOST LIKELY the paper's intended definition.")
        print("\nRun with: --chi2-definition ratio")
    else:
        print("\nNo definition produces O(1) scale values.")
        print("This may indicate:")
        print("  1. Synthetic ghost data (without SnapPy) produces unrealistic matrices")
        print("  2. Need more base points or different manifold")
        print("  3. The paper uses a different normalization not implemented here")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze chi-squared scale to identify paper definition"
    )
    parser.add_argument(
        "--spectrum",
        type=Path,
        required=True,
        help="Path to spectrum.npz file",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Which rank to analyze (default: 1)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    if not args.spectrum.exists():
        LOGGER.error(f"Spectrum file not found: {args.spectrum}")
        raise SystemExit(1)
    
    analyze_chi2_scale(args.spectrum, rank=args.rank)


if __name__ == "__main__":
    main()
