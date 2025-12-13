# PR Summary: Paper-Faithful Implementation for CMB Hyperbolic Universe

## Overview

This PR implements paper-faithful computation modes and enhanced diagnostics to bring the implementation closer to the algorithm described in `docs/eigenvalueprob.pdf`.

## Key Changes

### 1. Chi²-Mode Flag (Paper-Faithful vs Legacy)

**Problem**: The original code applied row L2 normalization to the constraint matrix A before SVD, which is **not mentioned in the paper**. This changed the chi² values and made them artificially closer to O(1).

**Solution**: Added `--chi2-mode` flag with two modes:
- `paper` (default): Computes χ² = ||A·a||² exactly as in equation 2.7, without row normalization
- `legacy`: Preserves the old row normalization behavior

**Impact**: Paper mode produces chi² values that are more faithful to the mathematical formulation in the paper.

### 2. Enhanced Diagnostics

Added comprehensive per-k diagnostics to `spectrum.npz`:
- `sigma_min`, `sigma_max`: Smallest and largest singular values
- `A_frobenius`: Frobenius norm of constraint matrix
- `A_max_abs`: Maximum absolute value in A
- `A_min_nonzero`: Minimum nonzero absolute value in A
- `images_per_point`: Average number of ghost images per base point

These diagnostics help understand:
- Numerical conditioning of the problem
- Matrix structure and scaling
- Why chi² values behave as they do

### 3. Script Import Fixes for HPC Usage

**Problem**: Running `python scripts/extract_eigenvalues.py` from repo root failed with `ModuleNotFoundError: utils`.

**Solution**:
- Added `scripts/__init__.py` to make it a proper Python package
- Added robust `sys.path` insertion in scripts for direct execution
- Scripts now work both as `python scripts/extract_eigenvalues.py` and `python -m scripts.extract_eigenvalues`

### 4. Comprehensive Testing

Added `tests/test_sanity.py` with 6 tests:
- `test_spectrum_not_flatlined_at_machine_precision`: Verifies chi² values are physically reasonable
- `test_fallback_fraction_reasonable`: Verifies fallback tracking is working
- `test_kept_points_meets_minimum`: Verifies base point retention logic
- `test_diagnostics_are_finite`: Verifies new diagnostics are computed correctly
- `test_paper_mode_differs_from_legacy`: Verifies paper and legacy modes produce different results
- Existing `test_scalar_vs_vectorized`: Verifies mathematical equivalence (still passes)

### 5. Documentation Updates

#### README.md
- Added "Paper-Faithful Sanity Run" section with recommended commands
- Documented paper algorithm parameters (L, c, ℓ_min)
- Added expected output characteristics

#### docs/algorithm.md
- Added "Paper-Faithful Algorithm Parameters" section
- Documented chi²-mode flag and its implications

#### docs/paper_algorithm_summary.md (NEW)
- Complete extraction of algorithm details from paper
- Key equations and parameters
- Identified deviations in original implementation

#### docs/paper_m188_eigenvalues.md (NEW)
- Extracted Table I from paper with eigenvalues for m188(-1,1)
- Provides ground truth q² values for validation
- Conversion formulas to k-values

### 6. Special Function Cache Management

Added utilities for debugging and performance tuning:
- `clear_special_function_caches()`: Clear all cached values
- `get_cache_stats()`: Get cache size statistics
- `USE_CACHE` flag: Can disable caching for debugging

## Algorithm Verification

### Paper Parameters (Already Correct in Code)
From eigenvalueprob.pdf Section II, equations 2.10:
- ✅ **L = 10 + floor(k)** - Already implemented correctly
- ✅ **c = 10 + floor(100/k)** - Already implemented correctly
- ✅ **ℓ_min = 5** - Already used in `PAPER_L_MIN`
- ✅ **M_target = c × N** where N = (L+1)² - Already implemented

### Chi² Definition
From equation 2.7:
- ✅ **χ² = ||A·a||²** where a is the solution vector from SVD
- ✅ **NO row normalization** mentioned in paper
- ✅ Eigenmodes determined "up to overall normalization" (applies to final eigenfunctions, not chi² computation)

### Scalar vs Vectorized Equivalence
- ✅ Test passes: relative Frobenius error < 1e-8
- ✅ Both use same prefactor: sqrt(π * N_nu_l / (2 * sinh(rho)))
- ✅ Both use mp.legenp with same parameters
- ✅ Both take real part consistently

## Deviations Found and Fixed

1. **Row normalization in SVD** (FIXED)
   - Original: Applied row L2 normalization before SVD
   - Paper: No normalization mentioned
   - Fix: Made normalization optional via `--chi2-mode` flag, default to paper mode

2. **L and c schedules** (ALREADY CORRECT)
   - Implementation already matched paper exactly
   - No changes needed

3. **Insufficient diagnostics** (FIXED)
   - Original: Only basic metadata (L, M, N, rho_min, rho_max)
   - Now: Comprehensive diagnostics including singular values, matrix norms, etc.

## Testing m188(-1,1) Against Paper Results

The paper provides Table I with eigenvalues (q² values) for m188(-1,1). To validate:

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

Expected results:
- Clear minima in chi² near k ≈ 4.41, 4.64, 5.12, 5.40, 6.21, 6.73, ... (from q² = 20.4, 22.6, 27.2, 30.2, 39.6, 46.2, ...)
- Degeneracies at k ≈ 8.34 (q² ≈ 70.6, 75.5 with multiplicity 2)
- No extreme step discontinuities
- No pathological flatlining near machine precision

## Remaining Work (Outside Scope of This PR)

1. **Sampling resolution**: Paper doesn't specify exact num-k, but Figure 1 appears smooth suggesting fine sampling
2. **Image deduplication**: Not explicitly discussed in paper
3. **Numerical rank estimation**: Could add with configurable tolerance
4. **Base point sampling details**: Paper says "randomly selected" but doesn't specify exact number

## Reproducibility

All changes maintain backward compatibility via the `--chi2-mode` flag. Default behavior is now paper-faithful.

To get the old behavior:
```bash
python main.py --chi2-mode legacy ...
```

## Files Changed

- `main.py`: Added chi2_mode parameter and enhanced diagnostics
- `utils/svd.py`: Added normalize_rows parameter, enhanced return values
- `utils/special_functions.py`: Added cache management utilities
- `utils/__init__.py`: Export new cache management functions
- `scripts/extract_eigenvalues.py`: Fixed imports for direct execution
- `scripts/__init__.py`: NEW - enables module-style execution
- `README.md`: Added paper-faithful sanity run documentation
- `docs/algorithm.md`: Added paper algorithm parameters
- `docs/paper_algorithm_summary.md`: NEW - comprehensive paper analysis
- `docs/paper_m188_eigenvalues.md`: NEW - ground truth eigenvalues
- `tests/test_sanity.py`: NEW - 6 sanity tests

## Test Results

All tests pass:
```
tests/test_scalar_vs_vectorized.py::test_scalar_matches_vectorized_small_case PASSED
tests/test_sanity.py::test_spectrum_not_flatlined_at_machine_precision PASSED
tests/test_sanity.py::test_fallback_fraction_reasonable PASSED
tests/test_sanity.py::test_kept_points_meets_minimum PASSED
tests/test_sanity.py::test_diagnostics_are_finite PASSED
tests/test_sanity.py::test_paper_mode_differs_from_legacy PASSED
```
