"""
Debug: Why does the normalization work for k=1 but not k>1?
Check if there's a k-dependent factor missing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.special_functions import Phi_nu_l


def analyze_k_dependence():
    """
    For ℓ=0, check how Phi_nu_l(k, 0, ρ) relates to sin(kρ)/sinh(ρ) for various k.
    """
    print("=" * 80)
    print("Analyzing k-dependence of Phi_nu_l normalization")
    print("=" * 80)
    print()
    
    k_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    rho = 10.0  # Use large ρ where asymptotic form applies
    
    print(f"Testing at ρ = {rho} (asymptotic regime)")
    print()
    print("-" * 80)
    print(f"{'k':<8} {'Phi*sinh':<15} {'sin(kρ)':<15} {'Ratio':<15} {'Expected √(2k)':<15}")
    print("-" * 80)
    
    for k in k_values:
        # Compute Phi_nu_l
        phi_val = float(Phi_nu_l(k, 0, rho))
        phi_sinh = phi_val * np.sinh(rho)
        
        # Target amplitude (from ℓ=0 closed form)
        target = np.sin(k * rho)
        
        # Ratio
        ratio = phi_sinh / target if abs(target) > 1e-10 else np.nan
        
        # Expected from √(2k) scaling
        expected_ratio = 1.0 / np.sqrt(2 * k)
        
        print(f"{k:<8.2f} {phi_sinh:<15.8f} {target:<15.8f} {ratio:<15.8f} {expected_ratio:<15.8f}")
    
    print("-" * 80)
    print()
    print("Conclusion:")
    print("  If 'Ratio' matches 'Expected √(2k)', then X_paper = √(2k) * Phi_nu_l is correct.")
    print("  If 'Ratio' has a different pattern, we need a different scaling.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    analyze_k_dependence()
