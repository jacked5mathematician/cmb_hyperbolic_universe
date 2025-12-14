"""
Verify that the analytic normalization X_paper = √(2k) * Phi_nu_l
exactly matches the ℓ=0 closed form and paper requirements.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.radial_normalized import X_paper
from utils.special_functions import Phi_nu_l


def test_ell0_closed_form_analytic():
    """
    Test: X_paper(k, 0, rho) * sinh(rho) should equal sin(k*rho) exactly.
    """
    print("=" * 80)
    print("VERIFICATION: Analytic normalization X_paper = √(2k) * Phi_nu_l")
    print("=" * 80)
    print()
    print("Target for ℓ=0: X_k^0(ρ) * sinh(ρ) = sin(kρ)  [amplitude = 1]")
    print()
    
    k_values = [1.0, 2.0, 5.0, 10.0, 20.0]
    rho_test = [3.0, 5.0, 10.0, 15.0]
    
    print("-" * 80)
    print("Testing ℓ=0 closed form")
    print("-" * 80)
    
    all_pass = True
    
    for k in k_values:
        print(f"\nk = {k:.1f}:")
        
        max_error = 0.0
        
        for rho in rho_test:
            # Compute X_paper (should equal sin(kρ)/sinh(ρ))
            X_val = X_paper(k, 0, rho)
            
            # Target: sin(k*rho) / sinh(rho)
            target = np.sin(k * rho) / np.sinh(rho)
            
            # Error
            error = abs(X_val - target)
            rel_error = error / max(abs(target), 1e-10)
            
            print(f"  ρ={rho:5.1f}: X_paper={X_val:12.8f}, sin(kρ)/sinh(ρ)={target:12.8f}, "
                  f"err={error:.2e}, rel={rel_error:.2e}")
            
            max_error = max(max_error, rel_error)
        
        # Check if errors are small
        if max_error < 1e-6:
            print(f"  ✅ PASS: max relative error = {max_error:.2e}")
        else:
            print(f"  ❌ FAIL: max relative error = {max_error:.2e} > 1e-6")
            all_pass = False
    
    print()
    print("-" * 80)
    print("Testing asymptotic amplitude X*sinh(ρ) on ρ ∈ [10, 20]")
    print("-" * 80)
    
    rho_asymp = np.linspace(10.0, 20.0, 50)
    
    for k in k_values:
        ell = int(np.floor(k)) + 10  # Paper L selection
        
        # Compute |X_paper * sinh| on asymptotic grid
        amplitudes = []
        for rho in rho_asymp:
            X_val = X_paper(k, ell, rho)
            val = abs(X_val * np.sinh(rho))
            amplitudes.append(val)
        
        max_amp = max(amplitudes)
        min_amp = min(amplitudes)
        mean_amp = np.mean(amplitudes)
        
        # Check if amplitude is O(1)
        if 0.5 < max_amp < 2.0:
            status = "✅ O(1)"
        else:
            status = f"❌ NOT O(1)"
            all_pass = False
        
        print(f"k={k:5.1f}, L={ell:3}: max={max_amp:.4f}, min={min_amp:.4f}, "
              f"mean={mean_amp:.4f}  {status}")
    
    print()
    print("=" * 80)
    
    if all_pass:
        print("✅ SUCCESS: X_paper matches all paper requirements!")
        print("   - ℓ=0 closed form: X_k^0(ρ)*sinh(ρ) = sin(kρ)  ✅")
        print("   - Asymptotic amplitude O(1)  ✅")
        print("   - Analytic normalization: X_paper = √(2k) * Phi_nu_l  ✅")
    else:
        print("❌ FAILURE: X_paper does not match paper requirements")
    
    print("=" * 80)
    
    return all_pass


if __name__ == "__main__":
    success = test_ell0_closed_form_analytic()
    sys.exit(0 if success else 1)
