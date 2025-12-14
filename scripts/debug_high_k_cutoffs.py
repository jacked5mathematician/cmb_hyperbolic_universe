"""
Debug script to understand what's happening with high-k cutoffs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.radial_normalized import X_normalized
from utils.cutoffs import _rho_turning_point


def analyze_radial_function(k: float, ell: int, threshold: float = 0.25):
    """Plot X_norm * sinh vs rho to understand the cutoff behavior."""
    
    rho0 = _rho_turning_point(k, ell)
    print(f"\nk={k:.1f}, L={ell}: ρ₀={rho0:.4f}")
    
    # Sample from turning point to reasonable distance
    rho_vals = np.linspace(rho0 + 0.01, min(10.0, rho0 + 8.0), 200)
    vals = []
    
    for rho in rho_vals:
        X = X_normalized(k, ell, rho)
        val = X * np.sinh(rho)
        vals.append(val)
    
    vals = np.array(vals)
    abs_vals = np.abs(vals)
    
    # Find max
    max_idx = np.argmax(abs_vals)
    max_val = abs_vals[max_idx]
    max_rho = rho_vals[max_idx]
    
    print(f"  Max |X*sinh|: {max_val:.4f} at ρ={max_rho:.4f}")
    print(f"  Threshold: {threshold:.2f}")
    
    if max_val < threshold:
        print(f"  ⚠️  MAX < THRESHOLD! No crossing possible.")
        return
    
    # Find first crossing after max
    for i in range(max_idx, len(rho_vals)):
        if abs_vals[i] <= threshold:
            print(f"  First crossing after max: ρ={rho_vals[i]:.4f}, |X*sinh|={abs_vals[i]:.4f}")
            return
    
    print(f"  No crossing found after max (checked up to ρ={rho_vals[-1]:.2f})")


if __name__ == "__main__":
    print("=" * 70)
    print("Debug: Radial function behavior for high-k cases")
    print("=" * 70)
    
    test_cases = [
        (1.0, 11),
        (2.0, 12),
        (5.0, 15),
        (10.0, 20),
        (15.0, 25),
        (20.0, 30),
    ]
    
    for k, L in test_cases:
        analyze_radial_function(k, L, threshold=0.25)
