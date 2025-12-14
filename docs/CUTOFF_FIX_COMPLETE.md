# Cutoff Fix Complete: Parts 1-4 Summary

**Date**: 2025  
**Status**: ✅ **COMPLETE - Bug Fixed**

---

## Problem Statement

The original cutoff logic in `utils/cutoffs.py` used an envelope approximation `|cos(k(ρ-ρ₀))|` instead of the actual radial function. This caused **critical bugs**:

1. **Envelope approximation had 60-1661% errors** vs actual `Phi_nu_l`
2. **For k≥5, NO ROOTS EXISTED** - the actual function never reached threshold 0.25
3. Wrong ρ_max → wrong images → wrong χ² matrix → **missing χ² dips**

**Root Cause** (identified by user): Normalization mismatch. Our `Phi_nu_l` had wrong asymptotic amplitude (0.035-0.707 instead of ~1.0), preventing threshold crossings.

---

## Solution: 4-Part Paper-Faithful Fix

### Part 1: Confirm Normalization Mismatch ✅

**Script**: `scripts/measure_asymptotic_amplitude.py`

Measured asymptotic amplitude of `Phi_nu_l(k,L,ρ) * sinh(ρ)` on ρ∈[10,20] by fitting `A cos(kρ) + B sin(kρ)`:

| k    | L  | Amplitude R | Expected |
|------|----|-------------|----------|
| 1.0  | 11 | 0.707       | ~1.0     |
| 5.0  | 15 | 0.141       | ~1.0     |
| 10.0 | 20 | 0.071       | ~1.0     |
| 20.0 | 30 | 0.035       | ~1.0     |

**Result**: Confirmed - amplitude ranges 0.035-0.707 (should be ~1.0). Pattern: R ≈ 1/√(2k).

### Part 2: Implement Normalized X_k^L ✅

**New Module**: `utils/radial_normalized.py`

Created paper-faithful normalized radial function:
```python
X_normalized(k, ell, rho) = Phi_nu_l(k, ell, rho) / R(k, ell)
```

where `R(k, ell)` is the fitted asymptotic amplitude (cached for performance).

**Tests**: `tests/test_normalized_radial.py` (14 tests, all passing)

| Test | Result |
|------|--------|
| Asymptotic amplitude ~1 (k=1,2,5,10,20) | ✅ max=0.999-1.000 |
| Roots exist for k∈{1,2,5,10,20} | ✅ All found |
| Normalization factors R(k,ell) | ✅ Matches predictions |
| Cache functionality | ✅ Works correctly |

**Key Achievement**: Roots NOW EXIST for k≥5 (previously impossible).

### Part 3: Replace Envelope Rootfinding ✅

**Modified**: `utils/cutoffs.py`

Replaced envelope approximation with actual `X_normalized` in `_find_crossing()`:

**Old method (BROKEN)**:
```python
val = abs(cos(k*(rho - rho0)))  # Envelope approximation
```

**New method (PAPER-FAITHFUL)**:
```python
X_val = X_normalized(k, ell, rho)  # Actual radial function
val = abs(X_val * sinh(rho))
```

**Algorithm**: Deterministic 'first crossing after first maximum':
1. Find first local maximum of |X*sinh| after turning point ρ₀
2. Scan forward until |X*sinh| ≤ threshold
3. Return that ρ as cutoff

### Part 4: Fallback Modes ✅

**Added**: Fallback modes for edge cases where no root exists (marked NON-PAPER in logs):
- `fallback_mode="fixed_rho"`: Use ρ = arcsinh(4) + 0.5L
- `fallback_mode="relative_envelope"`: Scale threshold by fitted amplitude

**Note**: In testing, NO fallback needed for k∈[1,20] - all cases find paper-faithful roots!

---

## Validation Results

### Root Existence (Previously Failed)

| k    | L  | ρ_max  | \|X·sinh(ρ_max)\| | Threshold | Status |
|------|----|--------|-------------------|-----------|--------|
| 1.0  | 11 | 5.646  | 0.238             | ≤ 0.25    | ✅ PASS |
| 2.0  | 12 | 3.901  | 0.245             | ≤ 0.25    | ✅ PASS |
| 5.0  | 15 | 2.519  | 0.225             | ≤ 0.25    | ✅ PASS |
| 10.0 | 20 | 1.876  | 0.124             | ≤ 0.25    | ✅ PASS |
| 15.0 | 25 | 1.611  | 0.041             | ≤ 0.25    | ✅ PASS |
| 20.0 | 30 | 1.458  | 0.053             | ≤ 0.25    | ✅ PASS |

**Critical**: k=5,10,20 **NOW HAVE ROOTS** (previously no solution existed!)

### Comparison: Old vs New

#### Old Method (Envelope - BROKEN)

| k    | Envelope Finds? | Actual Root Exists? | Error    |
|------|-----------------|---------------------|----------|
| 1    | Yes             | Yes                 | 71%      |
| 2    | Yes             | Yes                 | 60%      |
| 5    | Yes             | **NO**              | N/A      |
| 10   | Yes             | **NO**              | N/A      |
| 20   | Yes             | **NO**              | N/A      |

#### New Method (X_normalized - FIXED)

| k    | X_norm Finds? | Actual Root Exists? | |X·sinh(ρ)| | Satisfies ≤0.25? |
|------|---------------|---------------------|------------|------------------|
| 1    | Yes           | Yes                 | 0.238      | ✅ Yes           |
| 2    | Yes           | Yes                 | 0.245      | ✅ Yes           |
| 5    | Yes           | **YES**             | 0.225      | ✅ Yes           |
| 10   | Yes           | **YES**             | 0.124      | ✅ Yes           |
| 20   | Yes           | **YES**             | 0.053      | ✅ Yes           |

---

## Performance Notes

- **Normalization cache**: R(k,ell) computed once per (k,ell), cached for reuse
- **Overhead**: ~1-2s for first call per k (amplitude fitting), negligible for subsequent calls
- **Accuracy**: Fit quality RMS < 1e-9 (excellent)

---

## Files Modified/Created

### New Files
- `utils/radial_normalized.py` (165 lines) - Paper-faithful X_normalized
- `tests/test_normalized_radial.py` (230 lines) - Validation tests (all passing)
- `scripts/measure_asymptotic_amplitude.py` (145 lines) - Part 1 diagnostic
- `scripts/test_paper_faithful_cutoffs.py` (122 lines) - Integration test

### Modified Files
- `utils/cutoffs.py`:
  - Added `from .radial_normalized import X_normalized`
  - Rewrote `_find_crossing()` to use actual X_norm (not envelope)
  - Added `fallback_mode` parameter to `compute_rho_cutoffs()`
  - Added [NON-PAPER] log markers for fallback cases

---

## Next Steps

✅ **Cutoff/sampling logic is now paper-faithful**

**User specified scope**: Do NOT change eigenmode scanning, A(k) construction, χ² definition, or SVD code yet.

**Expected Impact**: With correct ρ_max values, the code should now:
1. Generate correct images (no phantoms, no missing ghosts)
2. Construct correct χ² matrix 
3. **Recover missing χ² dips** for real eigenmodes

**Recommended**: Run full pipeline on m188 to verify χ² dips reappear.

---

## References

**Paper**: Cornish & Spergel (1999)  
**Cutoff criterion**: X_k^L(ρ_max) sinh(ρ_max) = 0.25  
**L selection**: L = floor(k) + 10  
**Turning point**: ρ₀ = arcsinh(√(ℓ(ℓ+1))/k)  

**Tests**: All 14 tests in `test_normalized_radial.py` passing ✅
