"""
Diagnostic: Measure asymptotic amplitude of Phi_nu_l * sinh(rho) to confirm normalization mismatch.

This implements Part 1 of the cutoff fix:
- Fit A cos(kρ) + B sin(kρ) to Phi_nu_l(k, L, ρ) * sinh(ρ) on ρ ∈ [10, 20]
- Report amplitude vs k and L
- Confirm that amplitude ≠ 1 (normalization mismatch)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import curve_fit
from utils.special_functions import Phi_nu_l
from utils.cutoffs import rho_turning_point


def measure_asymptotic_amplitude(k: float, ell: int, rho_min: float = 10.0, rho_max: float = 20.0, n_points: int = 100):
    """
    Fit A cos(kρ) + B sin(kρ) to Phi_nu_l(k, ell, ρ) * sinh(ρ) in asymptotic regime.
    
    Returns:
        amplitude: sqrt(A^2 + B^2)
        fit_params: (A, B)
        rho_values: evaluation points
        actual_values: actual function values
        fitted_values: fitted function values
    """
    # Generate evaluation points
    rho_values = np.linspace(rho_min, rho_max, n_points)
    
    # Evaluate actual function
    actual_values = []
    for rho in rho_values:
        X = Phi_nu_l(k, ell, rho)
        val = float(X * np.sinh(rho))
        actual_values.append(val)
    
    actual_values = np.array(actual_values)
    
    # Define fitting function
    def fit_func(rho, A, B):
        return A * np.cos(k * rho) + B * np.sin(k * rho)
    
    # Fit
    try:
        params, _ = curve_fit(fit_func, rho_values, actual_values, p0=[1.0, 0.0])
        A, B = params
        amplitude = np.sqrt(A**2 + B**2)
        fitted_values = fit_func(rho_values, A, B)
    except Exception as e:
        print(f"  WARNING: Fit failed for k={k}, ell={ell}: {e}")
        amplitude = np.nan
        params = (np.nan, np.nan)
        fitted_values = np.zeros_like(actual_values)
    
    return amplitude, params, rho_values, actual_values, fitted_values


def main():
    """Run amplitude measurement for grid of k values."""
    print("=" * 80)
    print("DIAGNOSTIC: Asymptotic Amplitude of Phi_nu_l * sinh(rho)")
    print("=" * 80)
    print("\nPaper expectation: X_k^L(rho) * sinh(rho) should have amplitude ~1 for large rho")
    print("If amplitude ≠ 1, there is a normalization mismatch.\n")
    
    # Test grid
    k_values = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    
    print(f"{'k':>6s} {'L':>4s} {'rho0':>8s} {'Amplitude':>12s} {'A':>12s} {'B':>12s} {'Status':>10s}")
    print("-" * 80)
    
    results = []
    
    for k in k_values:
        L = int(np.floor(k)) + 10  # Paper rule
        
        # Compute turning point
        rho0 = rho_turning_point(k, L)
        
        # Measure amplitude on [10, 20]
        amplitude, (A, B), _, _, _ = measure_asymptotic_amplitude(k, L, rho_min=10.0, rho_max=20.0)
        
        # Check if close to 1
        if np.isnan(amplitude):
            status = "FIT FAIL"
        elif abs(amplitude - 1.0) < 0.1:
            status = "OK (~1)"
        else:
            status = f"WRONG ({amplitude:.2f})"
        
        print(f"{k:6.1f} {L:4d} {rho0:8.3f} {amplitude:12.6f} {A:12.6f} {B:12.6f} {status:>10s}")
        
        results.append({
            'k': k,
            'L': L,
            'rho0': rho0,
            'amplitude': amplitude,
            'A': A,
            'B': B,
        })
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    valid_amps = [r['amplitude'] for r in results if not np.isnan(r['amplitude'])]
    
    if valid_amps:
        print(f"\nAmplitude statistics:")
        print(f"  Min:    {np.min(valid_amps):.6f}")
        print(f"  Max:    {np.max(valid_amps):.6f}")
        print(f"  Mean:   {np.mean(valid_amps):.6f}")
        print(f"  Median: {np.median(valid_amps):.6f}")
        
        # Check if all are close to 1
        all_close_to_one = all(abs(amp - 1.0) < 0.1 for amp in valid_amps)
        
        if all_close_to_one:
            print("\n✅ All amplitudes are ~1. Normalization is correct (paper-faithful).")
        else:
            print(f"\n❌ Amplitudes vary from {np.min(valid_amps):.3f} to {np.max(valid_amps):.3f}.")
            print("   Normalization MISMATCH detected!")
            print("   Need to rescale Phi_nu_l to obtain paper-like X_k^L.")
    else:
        print("\n❌ All fits failed. Cannot determine normalization.")
    
    # Test a specific case in detail
    print("\n" + "=" * 80)
    print("DETAILED TEST: k=5.0, L=15")
    print("=" * 80)
    
    k_test, L_test = 5.0, 15
    amplitude, (A, B), rho_vals, actual_vals, fitted_vals = measure_asymptotic_amplitude(
        k_test, L_test, rho_min=10.0, rho_max=20.0, n_points=50
    )
    
    print(f"\nFitted amplitude: R(k={k_test}, L={L_test}) = {amplitude:.6f}")
    print(f"Fit parameters: A = {A:.6f}, B = {B:.6f}")
    
    # Compute residuals
    residuals = actual_vals - fitted_vals
    rms_residual = np.sqrt(np.mean(residuals**2))
    max_residual = np.max(np.abs(residuals))
    
    print(f"\nFit quality:")
    print(f"  RMS residual:  {rms_residual:.6e}")
    print(f"  Max residual:  {max_residual:.6e}")
    print(f"  Relative RMS:  {rms_residual / np.mean(np.abs(actual_vals)):.2%}")
    
    # Show sample values
    print(f"\nSample values at ρ=10, 15, 20:")
    for rho in [10.0, 15.0, 20.0]:
        X = Phi_nu_l(k_test, L_test, rho)
        val = float(X * np.sinh(rho))
        fit_val = A * np.cos(k_test * rho) + B * np.sin(k_test * rho)
        print(f"  ρ={rho:5.1f}: actual={val:10.6f}, fitted={fit_val:10.6f}, diff={val-fit_val:+10.6f}")
    
    return results


if __name__ == "__main__":
    main()
