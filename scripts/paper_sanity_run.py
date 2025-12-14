#!/usr/bin/env python3
"""
Paper sanity run script for m188(-1,1) to verify eigenvalue-selective behavior.

This script runs the pipeline with paper-faithful parameters and validates
that the chi-squared spectrum exhibits narrow minima at expected eigenvalues.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_pipeline(
    manifold: str,
    k_min: float,
    k_max: float,
    num_k: int,
    n_points: int,
    output_dir: Path,
    word_depth: int = 4,
    chi2_mode: str = "paper",
    chi2_definition: str = "raw_residual",
    require_snappy: bool = True,
) -> int:
    """Run the main pipeline with paper-faithful parameters."""
    
    cmd = [
        sys.executable, "main.py",
        "--manifold", manifold,
        "--k-min", str(k_min),
        "--k-max", str(k_max),
        "--num-k", str(num_k),
        "--n-points", str(n_points),
        "--word-depth", str(word_depth),
        "--chi2-mode", chi2_mode,
        "--chi2-definition", chi2_definition,
        "--output-dir", str(output_dir),
        "--benchmark",
    ]
    
    if require_snappy:
        cmd.append("--require-snappy")
    
    print("Running pipeline with command:")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def validate_spectrum(spectrum_path: Path, expected_eigenvalues: list[float]) -> dict:
    """
    Validate that spectrum shows narrow minima at expected eigenvalues.
    
    Returns:
        Dictionary with validation results
    """
    if not spectrum_path.exists():
        return {"error": f"Spectrum file not found: {spectrum_path}"}
    
    data = np.load(spectrum_path)
    k_values = data['k_values']
    chi2_rank_1 = data['chi2_rank_1']
    
    results = {
        "total_eigenvalues": len(expected_eigenvalues),
        "eigenvalues_in_range": 0,
        "eigenvalues_with_minima": 0,
        "false_minima": 0,
        "details": []
    }
    
    # Find local minima in chi-squared
    from scipy.signal import find_peaks
    
    # Detect minima (invert for peak finding)
    prominence_threshold = 0.05 * np.ptp(chi2_rank_1)
    minima_indices, _ = find_peaks(-chi2_rank_1, prominence=prominence_threshold)
    minima_k = k_values[minima_indices]
    
    print(f"\nVALIDATION RESULTS")
    print("="*70)
    print(f"K-range sampled: [{k_values.min():.3f}, {k_values.max():.3f}]")
    print(f"Total k samples: {len(k_values)}")
    print(f"Detected minima: {len(minima_k)}")
    print()
    
    # Check each expected eigenvalue
    tolerance = 2.0 * (k_values[1] - k_values[0]) if len(k_values) > 1 else 0.1
    
    for k_expected in expected_eigenvalues:
        if k_expected < k_values.min() or k_expected > k_values.max():
            continue
        
        results["eigenvalues_in_range"] += 1
        
        # Check if there's a minimum nearby
        distances = np.abs(minima_k - k_expected)
        if len(distances) > 0 and distances.min() < tolerance:
            nearest_idx = np.argmin(distances)
            k_minimum = minima_k[nearest_idx]
            results["eigenvalues_with_minima"] += 1
            status = "✓ FOUND"
        else:
            k_minimum = None
            status = "✗ MISSING"
        
        detail = {
            "k_expected": k_expected,
            "k_minimum": k_minimum,
            "distance": distances.min() if len(distances) > 0 else float('inf'),
            "status": status
        }
        results["details"].append(detail)
        
        print(f"  {status} k={k_expected:.3f} -> minimum at {k_minimum:.3f if k_minimum else 'N/A':>6}")
    
    # Count false positives (minima not near any eigenvalue)
    for k_min in minima_k:
        distances_to_expected = np.abs(np.array(expected_eigenvalues) - k_min)
        if distances_to_expected.min() > tolerance:
            results["false_minima"] += 1
    
    print()
    print("SUMMARY:")
    print(f"  Expected eigenvalues in range: {results['eigenvalues_in_range']}")
    print(f"  Eigenvalues with minima:       {results['eigenvalues_with_minima']}")
    print(f"  False minima (not near expected): {results['false_minima']}")
    
    # Overall assessment
    if results["eigenvalues_in_range"] > 0:
        success_rate = results["eigenvalues_with_minima"] / results["eigenvalues_in_range"]
        print(f"  Success rate: {100*success_rate:.1f}%")
        
        if success_rate > 0.8 and results["false_minima"] < results["eigenvalues_with_minima"]:
            print("\n✓ PASS: Spectrum shows eigenvalue-selective behavior")
            results["overall"] = "PASS"
        elif success_rate > 0.5:
            print("\n⚠ PARTIAL: Some eigenvalues detected but spectrum is noisy")
            results["overall"] = "PARTIAL"
        else:
            print("\n✗ FAIL: Spectrum does not show clear eigenvalue minima")
            results["overall"] = "FAIL"
    else:
        print("\n⚠ No expected eigenvalues in sampled k-range")
        results["overall"] = "INCOMPLETE"
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run paper sanity check for eigenvalue detection"
    )
    parser.add_argument(
        '--manifold',
        type=str,
        default='m188(-1,1)',
        help='Manifold name (default: m188(-1,1))'
    )
    parser.add_argument(
        '--k-min',
        type=float,
        default=1.0,
        help='Minimum k value (default: 1.0)'
    )
    parser.add_argument(
        '--k-max',
        type=float,
        default=10.0,
        help='Maximum k value (default: 10.0)'
    )
    parser.add_argument(
        '--num-k',
        type=int,
        default=200,
        help='Number of k samples (default: 200)'
    )
    parser.add_argument(
        '--n-points',
        type=int,
        default=50,
        help='Number of base points (default: 50)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output_values_local/paper_sanity'),
        help='Output directory'
    )
    parser.add_argument(
        '--skip-run',
        action='store_true',
        help='Skip pipeline run, only validate existing spectrum'
    )
    parser.add_argument(
        '--no-snappy',
        action='store_true',
        help='Do not require SnapPy (allows synthetic fallback)'
    )
    
    args = parser.parse_args()
    
    # Known eigenvalues for m188(-1,1) from paper Table I
    # Convert q² to k = sqrt(q² - 1)
    if args.manifold == 'm188(-1,1)':
        q2_values = [20.4, 22.6, 27.2, 30.2, 39.6, 46.2, 51.8, 55.3, 60.1, 
                     70.6, 75.5, 78.8, 80.9, 83.1, 86.0, 96.8, 98.0, 99.4]
        expected_eigenvalues = [np.sqrt(q2 - 1) for q2 in q2_values]
    else:
        print(f"Warning: No known eigenvalues for manifold {args.manifold}")
        expected_eigenvalues = []
    
    # Run pipeline
    if not args.skip_run:
        print(f"Running paper sanity check for {args.manifold}")
        print(f"k ∈ [{args.k_min}, {args.k_max}] with {args.num_k} samples")
        print(f"Using {args.n_points} base points")
        print()
        
        returncode = run_pipeline(
            manifold=args.manifold,
            k_min=args.k_min,
            k_max=args.k_max,
            num_k=args.num_k,
            n_points=args.n_points,
            output_dir=args.output_dir,
            require_snappy=not args.no_snappy,
        )
        
        if returncode != 0:
            print(f"\nERROR: Pipeline failed with exit code {returncode}")
            return returncode
    
    # Validate spectrum
    spectrum_path = args.output_dir / 'spectrum.npz'
    results = validate_spectrum(spectrum_path, expected_eigenvalues)
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return 1
    
    # Run debug analysis
    print("\n" + "="*70)
    print("RUNNING DEBUG ANALYSIS")
    print("="*70)
    
    debug_cmd = [
        sys.executable,
        str(Path(__file__).parent / "debug_singular_spectrum.py"),
        "--spectrum", str(spectrum_path),
        "--manifold", "m188" if "m188" in args.manifold else None,
    ]
    
    if "m188" in args.manifold:
        subprocess.run([cmd for cmd in debug_cmd if cmd is not None])
    
    # Return success/failure
    overall = results.get("overall", "UNKNOWN")
    if overall == "PASS":
        return 0
    elif overall == "PARTIAL":
        return 0  # Still considered success
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
