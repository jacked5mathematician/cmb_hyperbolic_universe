"""
Direct test: compute Phi_nu_l(k, 0, rho) and compare to sin(kρ)/sinh(ρ).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.special_functions import Phi_nu_l


def test_phi_ell0_direct():
    """
    For ℓ=0, the closed form should be X_k^0(ρ) = sin(kρ)/sinh(ρ).
    Test if Phi_nu_l(k, 0, ρ) matches this (up to normalization).
    """
    print("=" * 80)
    print("Direct test: Phi_nu_l(k, 0, ρ) vs sin(kρ)/sinh(ρ)")
    print("=" * 80)
    print()
    
    k_values = [1.0, 2.0, 5.0]
    rho_values = [3.0, 5.0, 10.0]
    
    for k in k_values:
        print(f"\nk = {k:.1f}:")
        
        for rho in rho_values:
            # Compute Phi_nu_l directly
            phi_val = float(Phi_nu_l(k, 0, rho))
            
            # Target (up to normalization): sin(kρ)/sinh(ρ)
            target_unnorm = np.sin(k * rho) / np.sinh(rho)
            
            # Ratio
            ratio = phi_val / target_unnorm if abs(target_unnorm) > 1e-10 else np.nan
            
            print(f"  ρ={rho:5.1f}: Phi={phi_val:12.8f}, sin(kρ)/sinh(ρ)={target_unnorm:12.8f}, "
                  f"ratio={ratio:8.6f}")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    test_phi_ell0_direct()
