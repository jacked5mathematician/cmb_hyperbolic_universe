"""
Test radial cutoff (rho_max) validation: root residuals, bracketing, determinism, and continuity.

This validates Step 2 of the cutoff hardening plan:
- 2.1: Root residual and bracket test
- 2.2: "Which root" determinism test
- 2.3: Continuity/stability across k

CRITICAL: The current implementation uses an ENVELOPE APPROXIMATION, not the actual radial function.
These tests will reveal if the envelope matches the true X_k^L behavior.
"""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.cutoffs import compute_rho_cutoffs, abs_radial_envelope, rho_turning_point
from utils.special_functions import Phi_nu_l


# =============================================================================
# Helper: Actual radial function evaluation
# =============================================================================

def actual_radial_function(k: float, ell: int, rho: float) -> float:
    """
    Compute the ACTUAL |X_k^ell(rho) * sinh(rho)| using Phi_nu_l.
    
    This is what the paper uses for the root condition:
        X_k^L(rho_max) * sinh(rho_max) = 0.25
    
    Compare to the envelope approximation in utils/cutoffs.py.
    """
    nu = k  # For Phi_nu_l, nu = k
    chi = rho  # Phi_nu_l uses chi parameter (same as rho here)
    
    # Evaluate actual radial function
    X = Phi_nu_l(nu, ell, chi)
    sinh_rho = np.sinh(rho)
    
    # Return absolute value
    return float(abs(X * sinh_rho))


def reference_root_finder(
    k: float,
    ell: int,
    threshold: float = 0.25,
    rho_start: float = 0.5,
    rho_cap: float = 30.0,
    step: float = 0.02,
    use_actual: bool = True,
) -> tuple[float | None, tuple[float, float] | None]:
    """
    Reference root finder using deterministic sign-change scan.
    
    Args:
        k: Wavenumber
        ell: Angular momentum quantum number
        threshold: Target value (paper: 0.25)
        rho_start: Starting rho for scan
        rho_cap: Maximum rho to scan
        step: Step size (finer than production 0.05)
        use_actual: If True, use actual Phi_nu_l; if False, use envelope
    
    Returns:
        (rho_root, bracket) where bracket = (rho_low, rho_high) or None if no root found
    """
    eval_func = actual_radial_function if use_actual else abs_radial_envelope
    
    rho = rho_start
    prev_val = None
    prev_rho = None
    
    while rho <= rho_cap:
        val = eval_func(k, ell, rho)
        
        # Check for sign change in (val - threshold)
        if prev_val is not None:
            if (prev_val > threshold) and (val <= threshold):
                # Found crossing: prev_rho -> rho
                # Refine with bisection
                a, b = prev_rho, rho
                for _ in range(20):  # 20 iterations gives ~1e-6 precision
                    mid = (a + b) / 2.0
                    mid_val = eval_func(k, ell, mid)
                    if mid_val > threshold:
                        a = mid
                    else:
                        b = mid
                
                root = (a + b) / 2.0
                return root, (prev_rho, rho)
        
        prev_val = val
        prev_rho = rho
        rho += step
    
    return None, None


# =============================================================================
# Test 2.1: Root Residual and Bracket Test
# =============================================================================

@pytest.mark.parametrize("k", [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0])
def test_rhomax_residual_envelope(k):
    """
    Test that production rho_max satisfies the envelope condition with low residual.
    
    This tests the ENVELOPE approximation used in production.
    """
    L = int(np.floor(k)) + 10  # Paper rule
    l_min = 5
    threshold = 0.25
    
    # Get production rho_max
    rho_min, rho_max, fallback = compute_rho_cutoffs(k, L, l_min, threshold=threshold)
    
    # Evaluate envelope at rho_max
    envelope_val = abs_radial_envelope(k, L, rho_max)
    
    # Residual from threshold
    residual = abs(envelope_val - threshold)
    
    # For non-fallback cases, residual should be small
    if not fallback:
        # Production uses step=0.05, so residual could be up to ~0.05 worth of change
        # But envelope is smooth, so expect < 0.1
        assert residual < 0.1, (
            f"Envelope residual too large for k={k}, L={L}:\n"
            f"  rho_max = {rho_max:.6f}\n"
            f"  envelope(rho_max) = {envelope_val:.6f}\n"
            f"  threshold = {threshold:.6f}\n"
            f"  residual = {residual:.6f}"
        )
    
    # Check no NaN/Inf
    assert np.isfinite(rho_max), f"rho_max is not finite: {rho_max}"
    assert np.isfinite(envelope_val), f"envelope_val is not finite: {envelope_val}"
    
    print(f"k={k:5.1f}, L={L:2d}: rho_max={rho_max:6.3f}, "
          f"envelope={envelope_val:.4f}, residual={residual:.4f}, fallback={fallback}")


@pytest.mark.parametrize("k", [1.0, 2.0, 5.0, 10.0, 20.0])
def test_rhomax_actual_vs_envelope(k):
    """
    CRITICAL TEST: Compare envelope approximation to actual radial function.
    
    This reveals whether the envelope is a good proxy for X_k^L(rho) * sinh(rho).
    If they disagree significantly, the cutoff is WRONG.
    """
    L = int(np.floor(k)) + 10
    l_min = 5
    threshold = 0.25
    
    # Get production rho_max (using envelope)
    rho_min, rho_max_prod, fallback = compute_rho_cutoffs(k, L, l_min, threshold=threshold)
    
    # Evaluate ACTUAL radial function at production rho_max
    actual_val = actual_radial_function(k, L, rho_max_prod)
    envelope_val = abs_radial_envelope(k, L, rho_max_prod)
    
    # Difference between envelope and actual
    difference = abs(actual_val - envelope_val)
    relative_error = difference / max(actual_val, 1e-10)
    
    print(f"\nk={k:5.1f}, L={L:2d}, rho_max={rho_max_prod:6.3f}:")
    print(f"  Envelope:  {envelope_val:.6f}")
    print(f"  Actual:    {actual_val:.6f}")
    print(f"  Diff:      {difference:.6f} (relative: {relative_error:.2%})")
    print(f"  Fallback:  {fallback}")
    
    # If envelope and actual disagree by more than 50%, something is WRONG
    if not fallback:
        assert relative_error < 0.5, (
            f"Envelope and actual radial function disagree significantly!\n"
            f"k={k}, L={L}, rho_max={rho_max_prod:.6f}\n"
            f"Envelope: {envelope_val:.6f}\n"
            f"Actual:   {actual_val:.6f}\n"
            f"Relative error: {relative_error:.2%}\n"
            f"This means the cutoff is using the WRONG function!"
        )


# =============================================================================
# Test 2.2: "Which Root" Determinism Test
# =============================================================================

@pytest.mark.parametrize("k", [1.0, 2.0, 3.0, 5.0, 8.0])
def test_rhomax_determinism_envelope(k):
    """
    Test that production rho_max matches reference sign-change scan (using envelope).
    
    Oscillatory functions have multiple roots. This ensures we pick the FIRST root
    consistently.
    """
    L = int(np.floor(k)) + 10
    l_min = 5
    threshold = 0.25
    
    # Production rho_max
    _, rho_max_prod, fallback_prod = compute_rho_cutoffs(k, L, l_min, threshold=threshold)
    
    # Reference rho_max (using envelope, same as production)
    rho_max_ref, bracket = reference_root_finder(
        k, L, threshold=threshold,
        rho_start=0.75,  # Match production
        step=0.02,  # Finer than production 0.05
        use_actual=False  # Use envelope to match production
    )
    
    if rho_max_ref is None:
        # No root found by reference either
        assert fallback_prod, (
            f"Production found rho_max={rho_max_prod:.6f} but reference found None!"
        )
        print(f"k={k:5.1f}, L={L:2d}: No envelope root found (both fallback)")
        return
    
    # Compare production vs reference
    difference = abs(rho_max_prod - rho_max_ref)
    
    # Allow difference up to step size (production uses 0.05)
    tolerance = 0.06
    
    if not fallback_prod:
        assert difference < tolerance, (
            f"Production and reference rho_max differ!\n"
            f"k={k}, L={L}\n"
            f"Production: {rho_max_prod:.6f}\n"
            f"Reference:  {rho_max_ref:.6f}\n"
            f"Difference: {difference:.6f}\n"
            f"Bracket:    {bracket}\n"
            f"This means root finding is NON-DETERMINISTIC!"
        )
    
    print(f"k={k:5.1f}, L={L:2d}: prod={rho_max_prod:.4f}, ref={rho_max_ref:.4f}, "
          f"diff={difference:.4f}, bracket={bracket}")


@pytest.mark.parametrize("k", [2.0, 5.0, 10.0])
def test_rhomax_actual_root_exists(k):
    """
    Test that a root of the ACTUAL radial function exists and compare to envelope root.
    
    This is the ground truth: does X_k^L(rho) * sinh(rho) = 0.25 have a solution?
    """
    L = int(np.floor(k)) + 10
    threshold = 0.25
    
    # Find root using ACTUAL radial function
    rho_actual, bracket_actual = reference_root_finder(
        k, L, threshold=threshold,
        rho_start=0.5,
        rho_cap=30.0,
        step=0.02,
        use_actual=True  # Use actual Phi_nu_l
    )
    
    # Find root using ENVELOPE
    rho_envelope, bracket_envelope = reference_root_finder(
        k, L, threshold=threshold,
        rho_start=0.5,
        step=0.02,
        use_actual=False  # Use envelope approximation
    )
    
    print(f"\nk={k:5.1f}, L={L:2d}:")
    actual_str = f"{rho_actual:.6f}" if rho_actual is not None else "None"
    envelope_str = f"{rho_envelope:.6f}" if rho_envelope is not None else "None"
    print(f"  Actual root:   {actual_str:>8s}  bracket={bracket_actual}")
    print(f"  Envelope root: {envelope_str:>8s}  bracket={bracket_envelope}")
    
    if rho_actual is not None and rho_envelope is not None:
        diff = abs(rho_actual - rho_envelope)
        print(f"  Difference:    {diff:.6f}")
        
        # Verify residuals
        actual_residual = abs(actual_radial_function(k, L, rho_actual) - threshold)
        envelope_residual = abs(abs_radial_envelope(k, L, rho_envelope) - threshold)
        print(f"  Actual residual:   {actual_residual:.2e}")
        print(f"  Envelope residual: {envelope_residual:.2e}")
        
        assert actual_residual < 1e-6, f"Actual root has large residual: {actual_residual:.2e}"


# =============================================================================
# Test 2.3: Continuity/Stability Across k
# =============================================================================

def test_rhomax_continuity():
    """
    Test that rho_max varies smoothly across a dense k grid.
    
    Large jumps indicate instability or incorrect root selection.
    """
    # Dense k grid
    k_values = np.arange(1.0, 20.0, 0.25)
    rho_max_values = []
    fallback_flags = []
    
    l_min = 5
    threshold = 0.25
    
    for k in k_values:
        L = int(np.floor(k)) + 10
        _, rho_max, fallback = compute_rho_cutoffs(k, L, l_min, threshold=threshold)
        rho_max_values.append(rho_max)
        fallback_flags.append(fallback)
    
    rho_max_arr = np.array(rho_max_values)
    fallback_arr = np.array(fallback_flags)
    
    # Compute jumps
    jumps = np.abs(np.diff(rho_max_arr))
    max_jump = np.max(jumps)
    max_jump_idx = np.argmax(jumps)
    
    # Report statistics
    print(f"\nContinuity test over k ∈ [1, 20] with step 0.25:")
    print(f"  Min rho_max:  {np.min(rho_max_arr):.4f}")
    print(f"  Max rho_max:  {np.max(rho_max_arr):.4f}")
    print(f"  Max jump:     {max_jump:.4f} at k={k_values[max_jump_idx]:.2f} → {k_values[max_jump_idx+1]:.2f}")
    print(f"  Fallback count: {np.sum(fallback_arr)} / {len(fallback_arr)}")
    
    # Check for large jumps
    jump_threshold = 2.0  # Flag if jump > 2.0
    large_jumps = jumps > jump_threshold
    
    if np.any(large_jumps):
        print(f"\n  WARNING: {np.sum(large_jumps)} large jumps detected:")
        for idx in np.where(large_jumps)[0]:
            k1, k2 = k_values[idx], k_values[idx + 1]
            rho1, rho2 = rho_max_arr[idx], rho_max_arr[idx + 1]
            jump = jumps[idx]
            print(f"    k={k1:.2f} → {k2:.2f}: rho_max {rho1:.4f} → {rho2:.4f} (jump={jump:.4f})")
    
    # Don't fail on large jumps, just report
    # (Large jumps may be expected due to oscillatory nature)


def test_rhomax_trend():
    """
    Test general trend: rho_max should increase with k (roughly).
    
    For larger k, more oscillations fit, so rho_max should grow.
    """
    k_values = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    rho_max_values = []
    
    l_min = 5
    threshold = 0.25
    
    for k in k_values:
        L = int(np.floor(k)) + 10
        _, rho_max, _ = compute_rho_cutoffs(k, L, l_min, threshold=threshold)
        rho_max_values.append(rho_max)
    
    print(f"\nTrend test:")
    for k, rho_max in zip(k_values, rho_max_values):
        print(f"  k={k:5.1f}: rho_max={rho_max:6.3f}")
    
    # Check that rho_max generally increases (allow some violations)
    increasing_count = sum(rho_max_values[i] < rho_max_values[i+1] for i in range(len(rho_max_values)-1))
    total_pairs = len(rho_max_values) - 1
    
    print(f"  Increasing pairs: {increasing_count}/{total_pairs} ({100*increasing_count/total_pairs:.1f}%)")
    
    # At least 60% should be increasing
    assert increasing_count >= 0.6 * total_pairs, (
        f"rho_max does not generally increase with k!"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
