# Paper-Faithful Implementation Changes

## Executive Summary

This document summarizes the changes made to bring the implementation closer to the algorithm described in `docs/eigenvalueprob.pdf`. The primary focus was on ensuring mathematical fidelity to the paper's equations while adding comprehensive diagnostics and improving code quality.

## Critical Finding: Non-Paper Normalization

### Issue
The original implementation applied row L2 normalization to the constraint matrix A before SVD:
```python
row_norms = np.linalg.norm(A, axis=1, keepdims=True)
A = A / row_norms
```

**This normalization is NOT mentioned in the paper.**

### Paper Specification
Equation 2.7 defines:
```
χ² = ||A·a||²
```
where A is the constraint matrix directly from the ghost equations, with no preprocessing.

### Solution
Added `--chi2-mode` flag:
- `paper` (default): No row normalization, matches equation 2.7
- `legacy`: Preserves old behavior for comparison

### Impact
Paper mode produces chi² values that are mathematically faithful to the algorithm. The normalization was artificially scaling values toward O(1), which obscured the true mathematical behavior.

## Verified: Algorithm Parameters Already Correct

The paper specifies in equations 2.10:
- **L = 10 + floor(k)** ✅ Already implemented in `_paper_L(k)`
- **c = 10 + floor(100/k)** ✅ Already implemented in `_paper_c(k)`
- **ℓ_min = 5** ✅ Already set in `PAPER_L_MIN`

No changes were needed for these core parameters.

## Enhanced Diagnostics

Added comprehensive per-k diagnostics to understand spectrum behavior:

### Matrix Diagnostics
- `A_frobenius`: Frobenius norm of A
- `A_max_abs`: Maximum absolute value in A
- `A_min_nonzero`: Minimum nonzero absolute value in A

These help understand matrix conditioning and scaling issues.

### Singular Value Diagnostics
- `sigma_min`: Smallest singular value
- `sigma_max`: Largest singular value

These help identify:
- Numerical rank of the system
- Condition number (sigma_max / sigma_min)
- Whether chi² ≈ 0 is due to numerical issues or true eigenvalues

### Geometry Diagnostics
- `images_per_point`: Average ghost images per base point

Helps understand ghost enumeration behavior.

### Usage Example
```python
import numpy as np
data = np.load('spectrum.npz')

# Check condition number
condition = data['sigma_max'] / (data['sigma_min'] + 1e-30)
print(f"Condition numbers: {condition}")

# Check matrix scaling
print(f"Matrix Frobenius norms: {data['A_frobenius']}")

# Check ghost enumeration
print(f"Images per point: {data['images_per_point']}")
```

## Ground Truth: m188(-1,1) Eigenvalues

Extracted Table I from paper for validation:

| q² Value | k = sqrt(q²-1) | Multiplicity |
|----------|----------------|--------------|
| 20.4     | 4.41          | 1            |
| 22.6     | 4.64          | 1            |
| 27.2     | 5.12          | 1            |
| 30.2     | 5.40          | 1            |
| 39.6     | 6.21          | 1            |
| 46.2     | 6.73          | 1            |
| 51.8     | 7.11          | 1            |
| 55.3     | 7.37          | 1            |
| 60.1     | 7.68          | 1            |
| 70.6     | 8.34          | 2            |
| 75.5     | 8.63          | 2            |
| 78.8     | 8.81          | 1            |
| 80.9     | 8.93          | 1            |
| 83.1     | 9.06          | 1            |
| 86.0     | 9.22          | 1            |
| 96.8     | 9.78          | 2            |
| 98.0     | 9.85          | 1            |
| 99.4     | 9.92          | 1            |

### Validation Command
```bash
python main.py \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 500 \
    --n-points 50 \
    --seed 42 \
    --chi2-mode paper \
    --word-depth 4 \
    --output-dir output_values_local/m188_validation \
    --require-snappy
```

### Expected Results
1. Clear minima in chi² spectrum near k = 4.41, 4.64, 5.12, 5.40, ...
2. Multiple solution vectors with similar chi² at k ≈ 8.34, 8.63 (multiplicity 2)
3. No spurious minima far from these values
4. Smooth spectrum without step discontinuities

## Code Quality Improvements

### 1. Cache Management
Added utilities for debugging and performance tuning:
```python
from utils import clear_special_function_caches, get_cache_stats

# Check cache usage
stats = get_cache_stats()
print(f"Phi cache: {stats['phi_cache_size']} entries")
print(f"Y_lm cache: {stats['y_lm_cache_size']} entries")

# Clear for fresh run
clear_special_function_caches()
```

### 2. Memory Efficiency
Improved min computation to avoid intermediate copies:
```python
# Before (creates copy)
A_nonzero = np.abs(A[A != 0])
min_val = A_nonzero.min()

# After (no copy)
nonzero_mask = (A != 0)
min_val = np.abs(A[nonzero_mask]).min()
```

### 3. Code Reusability
Created `_make_cache_key()` helper to avoid duplication in cache key construction.

## HPC Script Fixes

### Problem
```bash
$ python scripts/extract_eigenvalues.py ...
ModuleNotFoundError: No module named 'utils'
```

### Solution
1. Added `scripts/__init__.py` to make it a package
2. Added robust sys.path handling for direct execution
3. Now works both ways:
   - `python scripts/extract_eigenvalues.py` (direct)
   - `python -m scripts.extract_eigenvalues` (module)

## Testing

### Test Coverage
- `test_scalar_vs_vectorized.py`: Verifies mathematical equivalence
- `test_sanity.py`: 6 tests for spectrum quality:
  1. No pathological flatlining
  2. Fallback tracking works
  3. Base points retained
  4. Diagnostics computed
  5. Paper vs legacy modes differ
  6. (Plus test_kept_points_meets_minimum)

### Security
- CodeQL scan: **0 vulnerabilities**

### All Tests Pass
```
pytest tests/ -v
======================== 7 passed ========================
```

## Documentation

### New Documents
- `docs/paper_algorithm_summary.md`: Extracted algorithm details
- `docs/paper_m188_eigenvalues.md`: Ground truth eigenvalues
- `docs/PR_SUMMARY.md`: Comprehensive PR summary
- `PAPER_FAITHFUL_CHANGES.md`: This document

### Updated Documents
- `README.md`: Added paper-faithful sanity run section
- `docs/algorithm.md`: Added paper algorithm parameters

## Backward Compatibility

Default behavior is now paper-faithful. To get old behavior:
```bash
python main.py --chi2-mode legacy ...
```

## Future Work (Outside Scope)

1. **Sampling resolution**: Paper Figure 1 appears smooth, suggesting fine sampling. Could analyze figure to estimate num-k.
2. **Image deduplication**: Not explicitly discussed in paper, but could improve efficiency.
3. **Numerical rank**: Could add estimated rank with configurable tolerance.
4. **Eigenmode normalization**: Equation 2.12 describes normalizing final eigenmodes to unity. Not needed for chi² computation, but useful if outputting eigenfunctions.

## Files Modified

### Core Implementation
- `main.py`: Added chi2_mode, enhanced diagnostics
- `utils/svd.py`: Added normalize_rows parameter
- `utils/special_functions.py`: Cache management utilities
- `utils/__init__.py`: Export cache utilities

### Scripts
- `scripts/extract_eigenvalues.py`: Fixed imports
- `scripts/__init__.py`: NEW - enables module execution

### Tests
- `tests/test_sanity.py`: NEW - 6 sanity tests

### Documentation
- `README.md`: Paper-faithful run section
- `docs/algorithm.md`: Paper parameters
- `docs/paper_algorithm_summary.md`: NEW
- `docs/paper_m188_eigenvalues.md`: NEW
- `docs/PR_SUMMARY.md`: NEW
- `PAPER_FAITHFUL_CHANGES.md`: NEW (this file)

## Summary of Deviations Found

| Issue | Paper Says | Original Code | Status |
|-------|-----------|---------------|--------|
| Chi² normalization | χ² = ‖A·a‖² (eq 2.7) | Applied row normalization | ✅ FIXED (gated with flag) |
| L schedule | L = 10 + floor(k) | Same | ✅ Already correct |
| c schedule | c = 10 + floor(100/k) | Same | ✅ Already correct |
| ℓ_min | ℓ_min = 5 | Same | ✅ Already correct |
| Diagnostics | Not specified | Minimal | ✅ ENHANCED |

## Conclusion

The implementation is now mathematically faithful to the paper's algorithm. The critical fix was removing/gating the non-paper row normalization. Enhanced diagnostics provide insight into spectrum behavior and help validate against ground truth eigenvalues from Table I.
