"""
Detailed debug: Check what _find_crossing is actually returning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.radial_normalized import X_normalized
from utils.cutoffs import _rho_turning_point


def manual_find_crossing_debug(k: float, ell: int, threshold: float = 0.25):
    """Manually replicate the _find_crossing logic with debug output."""
    
    rho0 = _rho_turning_point(k, ell)
    print(f"\nk={k:.1f}, L={ell}")
    print(f"  ρ₀ (turning point): {rho0:.4f}")
    
    # Phase 1: Find first maximum
    rho = rho0 + 0.01
    step = 0.02
    rho_cap = 30.0
    
    max_found = False
    prev_val = None
    prev_prev_val = None
    max_rho = None
    
    while rho <= rho_cap and not max_found:
        X_val = X_normalized(k, ell, rho)
        val = abs(X_val * np.sinh(rho))
        
        # Detect local maximum
        if prev_prev_val is not None and prev_val is not None:
            if prev_prev_val < prev_val and prev_val > val:
                max_found = True
                max_rho = rho - step  # Previous rho
                print(f"  First maximum at ρ={max_rho:.4f}, |X*sinh|={prev_val:.4f}")
                break
        
        prev_prev_val = prev_val
        prev_val = val
        rho += step
    
    if not max_found:
        print(f"  ⚠️  No maximum found!")
        return
    
    # Phase 2: Find first crossing below threshold
    print(f"  Searching for crossing below {threshold:.2f} after maximum...")
    
    crossing_found = False
    for i in range(10):  # Check next 10 points
        X_val = X_normalized(k, ell, rho)
        val = abs(X_val * np.sinh(rho))
        
        print(f"    ρ={rho:.4f}: |X*sinh|={val:.4f}", end="")
        
        if val <= threshold:
            print(f" ← CROSSING!")
            crossing_found = True
            break
        else:
            print()
        
        rho += step
    
    if not crossing_found:
        print(f"  ⚠️  No crossing found within 10 steps after maximum")


if __name__ == "__main__":
    print("=" * 70)
    print("Manual _find_crossing Debug")
    print("=" * 70)
    
    for k in [10.0, 15.0, 20.0]:
        L = int(np.floor(k)) + 10
        manual_find_crossing_debug(k, L)
