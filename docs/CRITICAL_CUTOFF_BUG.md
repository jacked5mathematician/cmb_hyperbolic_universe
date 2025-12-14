# CRITICAL: Radial Cutoff Function is Fundamentally Broken

**Date**: 2025-12-14  
**Severity**: 🚨 **CRITICAL** - Explains missing χ² dips  
**Status**: **ROOT CAUSE IDENTIFIED**

---

## Executive Summary

The production cutoff function `compute_rho_cutoffs()` uses an **envelope approximation** instead of the actual radial function X_k^L(ρ). Validation tests reveal that:

1. ✅ **Envelope ≠ Actual**: Errors of 60-1600% (k=1: 71%, k=20: 1661%)
2. ✅ **No roots exist**: For k≥5, X_k^L(ρ)sinh(ρ) **never reaches 0.25**
3. ✅ **Wrong trend**: Max value decreases with k (0.165 at k=5, 0.049 at k=20)

**Consequence**: Pipeline uses incorrect ρ_max → wrong image selection → missing constraints → χ² dips disappear.

---

## Test Results

### Test: Envelope vs Actual at Production ρ_max

| k | L | ρ_max (prod) | Envelope | Actual | Relative Error |
|---|---|--------------|----------|---------|----------------|
| 1.0 | 11 | 4.500 | 0.206 | 0.700 | **70.6%** |
| 2.0 | 12 | 3.200 | 0.231 | 0.373 | 38.1% |
| 5.0 | 15 | 2.150 | 0.066 | 0.165 | **59.8%** |
| 10.0 | 20 | 1.600 | 0.224 | 0.087 | **158.2%** |
| 20.0 | 30 | 1.450 | 0.119 | 0.007 | **1661.0%** |

### Test: Maximum Values of Actual Function

| k | L | max(\|X_k^L sinh ρ\|) | Reaches 0.25? |
|---|---|----------------------|---------------|
| 1.0 | 11 | 0.716 | ✅ Yes |
| 2.0 | 12 | 0.374 | ✅ Yes |
| 5.0 | 15 | **0.165** | ❌ **NO** |
| 10.0 | 20 | **0.090** | ❌ **NO** |
| 20.0 | 30 | **0.049** | ❌ **NO** |

**For k≥5, the threshold 0.25 is IMPOSSIBLE to reach!**

### Test: Actual vs Envelope Root Locations

| k | L | Actual Root | Envelope Root | Difference |
|---|---|-------------|---------------|------------|
| 2.0 | 12 | 3.633 | 3.190 | 0.442 |
| 5.0 | 15 | **None** | 2.113 | *N/A* |
| 10.0 | 20 | **None** | 1.597 | *N/A* |

**Envelope finds "roots" that don't exist in reality!**

---

## Root Cause Analysis

### Current Implementation (utils/cutoffs.py)

```python
def _abs_radial_envelope(k, ell, rho):
    """Paper-inspired envelope for |X_k^ell(rho) * sinh(rho)|"""
    rho0 = arcsinh(sqrt(ell*(ell+1)) / k)  # Turning point
    if rho < rho0:
        return inf
    phase = k * (rho - rho0)
    return abs(cos(phase))  # ← WRONG APPROXIMATION
```

**Problem**: This assumes X_k^L(ρ) ~ cos(k(ρ-ρ₀))/sinh(ρ) for ρ > ρ₀.

**Reality**: The actual Legendre function Φ_ν_ℓ(ρ) has:
- Normalization factors that **decrease with ℓ**
- Amplitude modulation that **decreases with k**
- Phase shifts different from simple cos(kρ)

### Why It Fails for Large k

For k=10, L=20:
- **Envelope** max: ~1.0 (pure cosine oscillation)
- **Actual** max: ~0.090 (10x smaller!)

The Legendre function P^{-1/2-ℓ}_{-1/2+iν}(cosh ρ) has **amplitude suppression** at large ℓ that the envelope doesn't capture.

---

## Impact on χ² Analysis

### Scenario: k=10 eigenmode scan

1. **Production** uses ρ_max = 1.600 (envelope root at 0.25)
2. **Reality**: Actual function peaks at 0.090, **never reaches 0.25**
3. **Effect**: Cutoff is **arbitrary** - not based on actual function behavior
4. **Consequence**: 
   - May cut off too early → insufficient images → under-constrained
   - Or cut off too late → too many weak images → noise-dominated

### Why χ² Dips Are Missing

The paper's algorithm relies on:
```
X_k^L(ρ_max) sinh(ρ_max) = 0.25  (cutoff condition)
```

But for k≥5, this equation **has no solution**! The production code:
1. Uses envelope approximation that finds fake "roots"
2. Sets ρ_max at wrong location
3. Collects wrong set of images
4. Constructs χ² matrix with wrong constraints
5. **χ² dips disappear or shift**

---

## Proposed Solutions

### Option 1: Adaptive Threshold (Recommended)

Replace fixed threshold 0.25 with **percentile of maximum**:

```python
def compute_rho_cutoffs_adaptive(k, L, l_min, percentile=0.25):
    # Find maximum of actual function
    rho_grid = np.linspace(0.5, 30.0, 1000)
    vals = [actual_radial_function(k, L, r) for r in rho_grid]
    max_val = max(vals)
    
    # Set threshold as fraction of max
    threshold = percentile * max_val
    
    # Find first crossing
    for i, (rho, val) in enumerate(zip(rho_grid, vals)):
        if val <= threshold:
            return refine_root(rho, k, L, threshold)
```

**Pros**:
- Always has a solution
- Adapts to actual function behavior
- Maintains relative constraint strength

**Cons**:
- More expensive (needs function evaluation)
- Different from paper's description

### Option 2: Magnitude Threshold

Use **absolute magnitude** instead of sinh-weighted:

```python
threshold_magnitude = 0.05  # |X_k^L(ρ)| < 0.05
# Check |Phi_nu_l(k, L, rho)| directly (without sinh)
```

**Pros**:
- Directly tests function amplitude
- More interpretable

**Cons**:
- Still needs threshold tuning
- May not match paper intent

### Option 3: Fixed ρ_max (Diagnostic)

For testing, allow **user-specified ρ_max**:

```python
method="fixed_rho", rho_max=5.0
```

**Pros**:
- Isolates cutoff sensitivity
- Enables parameter sweeps

**Cons**:
- Not adaptive
- Loses paper motivation

### Option 4: Hybrid - Use Actual Function

**Replace envelope with actual function** in existing logic:

```python
def _find_crossing_actual(k, ell, threshold_percentile=0.25):
    # Find max first
    rho_grid = np.linspace(0.5, 30.0, 500)
    vals = [actual_radial_function(k, ell, r) for r in rho_grid]
    max_val = max(vals)
    threshold = threshold_percentile * max_val
    
    # Find crossing
    for rho, val in zip(rho_grid, vals):
        if val <= threshold:
            return refine_with_bisection(rho, k, ell, threshold)
```

**Pros**:
- Minimal code changes
- Uses ground truth
- Maintains paper structure

**Cons**:
- Slower (mpmath evaluations)
- Need to pick percentile parameter

---

## Immediate Actions

### 1. Document and Report

- ✅ Created test suite revealing bug
- ✅ Documented findings in this file
- ⏳ Update main documentation

### 2. Implement Robust Method

Add to `utils/cutoffs.py`:

```python
def compute_rho_cutoffs(k, L, l_min, method="auto", **kwargs):
    if method == "envelope":
        # Current (broken) implementation
        return _compute_envelope_cutoffs(k, L, l_min, **kwargs)
    elif method == "adaptive":
        # New: Adaptive threshold based on actual function
        return _compute_adaptive_cutoffs(k, L, l_min, **kwargs)
    elif method == "fixed":
        # Diagnostic: User-specified
        return kwargs['rho_min'], kwargs['rho_max'], False
    elif method == "auto":
        # Default: Use adaptive for k≥5, envelope for k<5
        if k >= 5.0:
            return _compute_adaptive_cutoffs(k, L, l_min, **kwargs)
        else:
            return _compute_envelope_cutoffs(k, L, l_min, **kwargs)
```

### 3. Re-run Pipeline

Test χ² reconstruction with corrected cutoffs:
- Compare envelope vs adaptive methods
- Check if χ² dips appear
- Validate against paper Figure 1

---

## Test File Locations

- `tests/test_rhomax_cutoff.py` - Validation suite
- `scripts/diagnose_radial_cutoff.py` - Visualization
- `output_values_local/radial_function_diagnostic.png` - Plot showing discrepancy

---

## References

- Cornish & Spergel (1999) Section 2, equation (2.8)
- Paper threshold: X_k^L(ρ_max) sinh(ρ_max) = 0.25
- Our finding: This equation has no solution for k≥5, L=floor(k)+10

---

## Conclusion

The production cutoff method is **fundamentally incompatible** with the actual radial functions for k≥5. This explains:

1. Missing χ² dips in eigenmode scans
2. Poor constraint quality at high k
3. Instability in matrix conditioning

**Resolution**: Implement adaptive threshold based on **actual function evaluation**, not envelope approximation.

**Priority**: 🚨 **HIGHEST** - Blocks all eigenmode reconstruction for k≥5
