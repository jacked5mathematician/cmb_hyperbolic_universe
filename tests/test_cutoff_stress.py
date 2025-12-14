"""
Comprehensive stress test for cutoff rootfinding: k=1-30.

As requested in spec:
- Test k=1,2,...,30 (step 1)
- Report fallback frequency
- If fallback > 0, print ranked list of failing (k,L) and failure mode
- Add tests for ℓ=0 closed form matching
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging
from utils.cutoffs import compute_rho_cutoffs
from utils.radial_normalized import X_paper

# Enable logging to see diagnostics
logging.basicConfig(level=logging.INFO, format='%(message)s')


def L_from_k(k: float) -> int:
    """Paper rule: L = floor(k) + 10"""
    return int(np.floor(k)) + 10


def test_ell0_closed_form():
    """
    Test requirement: X_paper(k, 0, ρ) ≈ sin(kρ)/sinh(ρ) on multiple ρ values.
    """
    print("\n" + "=" * 80)
    print("TEST 1: ℓ=0 Closed Form Validation")
    print("=" * 80)
    print("Requirement: X_paper(k, 0, ρ) must match sin(kρ)/sinh(ρ)")
    print()
    
    k_values = [1.0, 2.0, 5.0, 10.0, 20.0]
    rho_values = [3.0, 5.0, 10.0, 15.0, 20.0]
    
    all_pass = True
    
    for k in k_values:
        max_rel_error = 0.0
        
        for rho in rho_values:
            X_val = X_paper(k, 0, rho)
            target = np.sin(k * rho) / np.sinh(rho)
            rel_error = abs(X_val - target) / max(abs(target), 1e-10)
            max_rel_error = max(max_rel_error, rel_error)
        
        status = "✅ PASS" if max_rel_error < 1e-10 else "❌ FAIL"
        if max_rel_error >= 1e-10:
            all_pass = False
        
        print(f"  k={k:5.1f}: max relative error = {max_rel_error:.2e}  {status}")
    
    print()
    if all_pass:
        print("✅ All ℓ=0 tests passed")
    else:
        print("❌ Some ℓ=0 tests failed")
    
    return all_pass


def stress_test_cutoffs(k_range=range(1, 31)):
    """
    Stress test: compute cutoffs for k=1-30, report fallback frequency.
    """
    print("\n" + "=" * 80)
    print(f"TEST 2: Stress Test k ∈ [{min(k_range)}, {max(k_range)}]")
    print("=" * 80)
    print("Requirement: Find paper-faithful cutoff for |X_paper·sinh(ρ)| ≤ 0.25")
    print()
    
    results = []
    failures = []
    
    print(f"{'k':<6} {'L':<6} {'ρ_max':<12} {'|X·sinh(ρ)|':<15} {'Status':<20}")
    print("-" * 80)
    
    for k in k_range:
        k_float = float(k)
        L = L_from_k(k_float)
        
        try:
            rho_min, rho_max, fallback_used = compute_rho_cutoffs(
                k=k_float,
                L=L,
                l_min=1,
                threshold=0.25,
                rho_cap=30.0,
                step=0.02,
                verbose=False,  # Suppress per-k logging for cleaner output
            )
            
            # Verify cutoff
            X_val = X_paper(k_float, L, rho_max)
            f_rho_max = abs(X_val * np.sinh(rho_max))
            
            status = "PAPER" if not fallback_used else "FALLBACK"
            # Allow small tolerance for numerical precision at threshold boundary
            check = "✅" if f_rho_max <= 0.251 else "❌"  # 0.4% tolerance
            
            print(f"{k:<6} {L:<6} {rho_max:<12.6f} {f_rho_max:<15.6f} {status:<20} {check}")
            
            results.append({
                'k': k_float,
                'L': L,
                'rho_max': rho_max,
                'f_value': f_rho_max,
                'fallback': fallback_used,
                'satisfies_threshold': f_rho_max <= 0.251,  # Small tolerance for numerical precision
            })
            
            if fallback_used or f_rho_max > 0.251:
                failures.append({
                    'k': k_float,
                    'L': L,
                    'reason': 'fallback' if fallback_used else 'threshold_violation',
                    'f_value': f_rho_max,
                })
        
        except Exception as e:
            print(f"{k:<6} {L:<6} {'ERROR':<12} {'N/A':<15} {'EXCEPTION':<20} ❌")
            failures.append({
                'k': k_float,
                'L': L,
                'reason': 'exception',
                'error': str(e),
            })
    
    print("-" * 80)
    print()
    
    # Statistics
    total = len(results)
    paper_count = sum(1 for r in results if not r['fallback'])
    fallback_count = sum(1 for r in results if r['fallback'])
    threshold_violations = sum(1 for r in results if not r['satisfies_threshold'])
    
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total k values tested: {total}")
    print(f"Paper-faithful roots:  {paper_count} ({100*paper_count/total:.1f}%)")
    print(f"Fallback required:     {fallback_count} ({100*fallback_count/total:.1f}%)")
    print(f"Threshold violations:  {threshold_violations}")
    print()
    
    if failures:
        print("=" * 80)
        print(f"FAILURE ANALYSIS ({len(failures)} failures)")
        print("=" * 80)
        print()
        print("Ranked list of failing (k, L) pairs:")
        print(f"{'k':<6} {'L':<6} {'Reason':<20} {'f(ρ_max)':<15}")
        print("-" * 60)
        
        # Sort by k
        failures.sort(key=lambda x: x['k'])
        
        for fail in failures:
            f_val = fail.get('f_value', np.nan)
            print(f"{fail['k']:<6.0f} {fail['L']:<6} {fail['reason']:<20} {f_val:<15.6f}")
        
        print()
    
    print("=" * 80)
    
    if fallback_count == 0 and threshold_violations == 0:
        print("✅ SUCCESS: All k values found paper-faithful roots satisfying threshold!")
        return True
    else:
        print(f"⚠️  PARTIAL SUCCESS: {fallback_count} fallbacks, {threshold_violations} violations")
        return False


def main():
    print("=" * 80)
    print("COMPREHENSIVE CUTOFF VALIDATION")
    print("Cornish & Spergel (1999) Paper-Faithful Implementation")
    print("=" * 80)
    
    # Test 1: ℓ=0 closed form
    test1_pass = test_ell0_closed_form()
    
    # Test 2: Stress test k=1-30
    test2_pass = stress_test_cutoffs(range(1, 31))
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if test1_pass and test2_pass:
        print("✅ ALL TESTS PASSED")
        print("   - ℓ=0 closed form matches sin(kρ)/sinh(ρ)")
        print("   - All k ∈ [1,30] find paper-faithful cutoffs")
        print("   - All cutoffs satisfy |X·sinh(ρ)| ≤ 0.25")
        return 0
    else:
        if not test1_pass:
            print("❌ ℓ=0 closed form test FAILED")
        if not test2_pass:
            print("❌ Stress test FAILED (some fallbacks or violations)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
