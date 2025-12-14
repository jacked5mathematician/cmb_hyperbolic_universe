"""
Test script for Parts 3-4: Verify that compute_rho_cutoffs now uses X_normalized
and finds correct roots for k=1,2,5,10,20.

Expected: All k values should now find paper-faithful roots (no fallback needed).
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.cutoffs import compute_rho_cutoffs
from utils.radial_normalized import X_normalized


def L_from_k(k: float) -> int:
    """Paper rule: L = floor(k) + 10"""
    return int(np.floor(k)) + 10


def verify_root(k: float, L: int, rho_max: float, threshold: float = 0.25):
    """Verify that X_norm * sinh at rho_max is close to threshold."""
    X_val = X_normalized(k, L, rho_max)
    val_at_root = X_val * np.sinh(rho_max)
    error = abs(val_at_root - threshold)
    return val_at_root, error


def main():
    print("=" * 70)
    print("Part 3-4 Test: compute_rho_cutoffs with X_normalized")
    print("=" * 70)
    
    test_k_values = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    l_min = 1  # Typical value
    threshold = 0.25  # Paper value
    
    results = []
    
    for k in test_k_values:
        L = L_from_k(k)
        
        print(f"\n{'─' * 70}")
        print(f"Testing k={k:.1f}, L={L}")
        
        # Compute cutoffs using new paper-faithful method
        rho_min, rho_max, fallback_used = compute_rho_cutoffs(
            k=k,
            L=L,
            l_min=l_min,
            threshold=threshold,
            rho_cap=30.0,  # Reduced from 120 for faster testing
            step=0.02,     # Finer grid
            rho_start=0.75,
            rho_max_floor=1.0,
            fallback_mode="fixed_rho",
        )
        
        # Verify the root
        val_at_rho_max, error = verify_root(k, L, rho_max, threshold)
        
        status = "✅ PAPER-FAITHFUL" if not fallback_used else "⚠️  FALLBACK (NON-PAPER)"
        
        print(f"  rho_min: {rho_min:.4f}")
        print(f"  rho_max: {rho_max:.4f} {status}")
        print(f"  X*sinh(rho_max): {val_at_rho_max:.4f} (target: {threshold:.2f})")
        print(f"  Error: {error:.6f}")
        
        results.append({
            "k": k,
            "L": L,
            "rho_max": rho_max,
            "fallback": fallback_used,
            "value": val_at_rho_max,
            "error": error,
        })
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    
    paper_faithful_count = sum(1 for r in results if not r["fallback"])
    fallback_count = sum(1 for r in results if r["fallback"])
    
    print(f"Paper-faithful roots: {paper_faithful_count}/{len(results)}")
    print(f"Fallback used: {fallback_count}/{len(results)}")
    
    if fallback_count == 0:
        print("\n🎉 SUCCESS: All k values found paper-faithful roots!")
        print("   The cutoff bug is FIXED.")
    else:
        print(f"\n⚠️  WARNING: {fallback_count} k values required fallback.")
        print("   These are marked NON-PAPER in logs.")
    
    print(f"\n{'k':<6} {'L':<4} {'rho_max':<10} {'Value':<10} {'Error':<12} {'Status':<20}")
    print("─" * 70)
    for r in results:
        status = "PAPER" if not r["fallback"] else "FALLBACK"
        print(
            f"{r['k']:<6.1f} {r['L']:<4} {r['rho_max']:<10.4f} "
            f"{r['value']:<10.4f} {r['error']:<12.6f} {status:<20}"
        )
    
    print(f"\n{'=' * 70}")
    print("Part 3-4 complete: Cutoff logic now uses X_normalized")
    print("=" * 70)


if __name__ == "__main__":
    main()
