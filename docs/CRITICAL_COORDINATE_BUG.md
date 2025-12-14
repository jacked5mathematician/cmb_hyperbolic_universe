# CRITICAL: Coordinate Conversion Bug Found

## Test Results: test_coordinate_roundtrip.py

**Status:** 4 FAILED, 13 PASSED

---

## Failures Detected

### 1. Origin Handling (CRITICAL BUG)
**Test:** `test_origin_roundtrip`  
**File/Line:** `utils/transformations.py:103`  
**Issue:** Division by `sinh_rho` when `rho ≈ 0` produces **NaN**

**Error:**
```
AssertionError: Origin round-trip failed: [ 1. nan nan nan]
```

**Code location:**
```python
theta = np.arccos(X[:, 2] / sinh_rho)  # Line 103
```

When `rho = 0`, `sinh_rho = 0`, causing division by zero → NaN.

**Impact:** 🔴 **CRITICAL** - Any basis function evaluation near origin will fail

---

### 2. Z-Axis Points (HIGH ERROR)
**Test:** `test_radial_points_roundtrip[r]`  
**File/Line:** Same as above (transformations.py:103)

**Numerical residuals:**
| Point | Error | Status |
|-------|-------|--------|
| [0, 0, 0.1] | **nan** | ❌ NaN |
| [0, 0, 0.3] | < 1e-12 | ✅ Pass |
| [0, 0, 0.5] | **7.45e-09** | ❌ Fails tolerance |
| [0, 0, 0.7] | **1.04e-08** | ❌ Fails tolerance |
| [0, 0, 0.85] | < 1e-12 | ✅ Pass |

**Pattern:** Errors are largest at intermediate radii (r ~ 0.5-0.7)

**Impact:** 🟠 **HIGH** - Basis functions along polar axis have 1e-8 errors instead of 1e-12

---

### 3. X/Y-Axis Points
**Status:** ✅ PASS - These work correctly

Only z-axis has issues, suggesting problem with theta calculation.

---

## Root Cause Analysis

### Bug Location
**File:** `utils/transformations.py`  
**Function:** `poincare_to_pseudo_spherical`  
**Lines:** 93-107

### Problematic Code
```python
def poincare_to_pseudo_spherical(points):
    points = np.array(points)
    norm_squared = np.sum(points**2, axis=1)
    valid_indices = norm_squared < 1
    valid_points = points[valid_indices]
    norm_squared = norm_squared[valid_indices]

    X0 = (1 + norm_squared) / (1 - norm_squared)
    X = 2 * valid_points / (1 - norm_squared[:, np.newaxis])
    rho = np.arccosh(X0)
    sinh_rho = np.sinh(rho)
    theta = np.arccos(X[:, 2] / sinh_rho)  # ← BUG: division by zero
    phi = np.arctan2(X[:, 1], X[:, 0])

    pseudo_spherical_points = np.column_stack((rho, theta, phi))
    return pseudo_spherical_points
```

### Why It's Wrong

1. **No guard against sinh_rho ≈ 0**
   - When `rho < 1e-10`, `sinh_rho ≈ 0`
   - Division produces NaN or numerical instability

2. **Incorrect theta formula for numerics**
   - Should use `X[:, 2] / sinh_rho = cosh(rho) * cos(theta)`
   - But when sinh_rho is small, this is unstable
   - Need special handling or use atan2 for better conditioning

3. **No handling of degenerate case**
   - At origin, theta is undefined (all directions equivalent)
   - Should set theta=0 (or any value) when rho < threshold

---

## Required Fix

### Option 1: Guard with threshold (simple)
```python
# Add guard for small rho
threshold = 1e-14
safe_sinh = np.where(sinh_rho > threshold, sinh_rho, 1.0)
z_normalized = np.where(sinh_rho > threshold, X[:, 2] / sinh_rho, 0.0)
theta = np.arccos(np.clip(z_normalized, -1.0, 1.0))
```

### Option 2: Use atan2 (better conditioning)
```python
# More numerically stable
xy_norm = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
theta = np.arctan2(xy_norm, X[:, 2])  # Avoids division
```

---

## Impact on Downstream Code

### ❌ Broken Without Fix
1. **Basis functions** `X_k^ℓ(ρ,θ,φ)` - NaN near origin
2. **Chi-squared matrix** - NaN entries propagate
3. **Eigenmode reconstruction** - Completely fails

### ⚠️ Degraded Without Fix
4. **Points along z-axis** - 1e-8 errors instead of 1e-12
5. **Conditioning of χ²** - Worse than necessary

---

## Why Geometry Tests Passed

The geometry tests (isometry, composition, distance) all use:
- **Poincaré coordinates directly** (no conversion to ρ,θ,φ)
- **Hyperboloid coordinates** (computed differently)
- **Distance function** (bypasses pseudo-spherical)

They never call `poincare_to_pseudo_spherical` for critical checks!

**This is exactly why we need basis function tests** - geometry can be correct while coordinate extraction is broken.

---

## Test Summary

### Passing Tests (13/17) ✅
- Poincaré ↔ Hyperboloid: Works perfectly
- Random points: 1e-12 precision for most cases  
- ρ values: Match expected distances
- θ,φ directions: Qualitatively correct
- Hyperboloid constraint: Always preserved

### Failing Tests (4/17) ❌
- Origin round-trip: **NaN**
- Z-axis at r=0.1: **NaN**
- Z-axis at r=0.5: **7.45e-09**
- Z-axis at r=0.7: **1.04e-08**

---

## Recommended Actions

### Immediate (before any basis functions)
1. 🔴 Fix `poincare_to_pseudo_spherical` in transformations.py:103
2. 🔴 Add guards for sinh_rho ≈ 0
3. 🔴 Re-run all coordinate tests
4. 🔴 Verify all pass with errors < 1e-12

### After Fix
5. ✅ Add radial ODE residual tests
6. ✅ Add asymptotic behavior tests
7. ✅ Add spherical harmonics tests
8. ✅ Then proceed to A(k) and χ²

---

## Conclusion

**🔴 BLOCK: Do not proceed with basis functions until coordinate bug is fixed**

The current implementation will produce:
- NaN values near origin
- 1e-8 errors along z-axis (instead of 1e-12)
- Corrupted chi-squared matrices
- Failed eigenmode reconstruction

**Estimated fix time:** 5-10 minutes  
**Estimated impact:** Prevents catastrophic failure of entire pipeline

---

*Report generated from test_coordinate_roundtrip.py failures*  
*Test run: 4 failed, 13 passed, 3 warnings in 1.06s*
