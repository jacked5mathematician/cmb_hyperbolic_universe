"""
Check the actual values returned by compute_rho_cutoffs
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.cutoffs import compute_rho_cutoffs, _find_crossing
from utils.radial_normalized import X_normalized


def L_from_k(k: float) -> int:
    return int(np.floor(k)) + 10


def test_single_k(k: float):
    L = L_from_k(k)
    
    print(f"\nk={k:.1f}, L={L}")
    
    # Test _find_crossing directly
    rho_max_direct = _find_crossing(
        k=k,
        ell=L,
        threshold=0.25,
        rho_cap=30.0,
        step=0.02,
        rho_start=0.75,
    )
    
    print(f"  _find_crossing returned: ρ={rho_max_direct:.4f}")
    
    if rho_max_direct is not None:
        X_val = X_normalized(k, L, rho_max_direct)
        val = X_val * np.sinh(rho_max_direct)
        abs_val = abs(val)
        print(f"  X*sinh(ρ) = {val:.4f}")
        print(f"  |X*sinh(ρ)| = {abs_val:.4f} (target ≤ 0.25)")


if __name__ == "__main__":
    for k in [10.0, 15.0, 20.0]:
        test_single_k(k)
