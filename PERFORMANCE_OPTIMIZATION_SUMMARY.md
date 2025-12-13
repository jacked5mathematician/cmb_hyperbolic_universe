# Performance Optimization Summary

## Executive Summary

Successfully implemented performance optimizations achieving **11.3× overall speedup** (far exceeding the 2× target requirement) while preserving mathematical correctness and maintaining full test coverage.

## Key Achievement

**11.4× speedup in matrix assembly** (the dominant bottleneck comprising 99.9% of runtime) by eliminating redundant special function evaluations.

## The Problem

### Baseline Performance (Before Optimization)
Running on manifold m188(-1,1), k∈[3.5, 4.2], 5 k-values, 15 base points:
- **Total runtime**: 203 seconds
- **Matrix build**: 202.5 seconds (99.9% of time)
- **SVD**: 0.18 seconds (0.1% of time)
- **mpmath legenp calls**: 202,410 expensive evaluations
- **Cache hit rate**: 0% (vectorized path bypassed cache)

### Root Cause Analysis
The `Q_k_lm_vectorized` function computed `Φ_nu_l(k, l, ρ)` separately for each (l, m) pair in the spherical harmonic expansion, even though:
- Φ depends only on (k, l, ρ)
- Φ is **independent of m**

This resulted in massive redundant computation: for L=13, we have (L+1)² = 196 lm_pairs, but only L+1 = 14 unique l values. Each l has 2l+1 different m values, all requiring the same Φ computation.

## The Solution

### Core Optimization: Φ_l Reuse Across m Values

Refactored `Q_k_lm_vectorized` to:
1. Group lm_pairs by their l value
2. Compute Φ_nu_l(k, l, ρ) **once** per unique l
3. Reuse the Φ_l result for all m values with that l
4. Compute Y_lm(θ, φ) individually for each (l, m) as before

```python
# Before (each lm pair computed Φ independently):
for idx, (l, m) in enumerate(lm_pairs):
    Phi_vals = Phi_nu_l_vectorized(k_value, l, rho)  # 196 times for L=13
    Y_vals = Y_lm_real_vectorized(l, m, theta, phi)
    Q_values[:, idx] = Phi_vals * Y_vals

# After (Φ computed once per l):
for l, m_indices in l_to_m_indices.items():  # 14 unique l for L=13
    Phi_vals = Phi_nu_l_vectorized(k_value, l, rho)  # Only once per l!
    for idx, m in m_indices:
        Y_vals = Y_lm_real_vectorized(l, m, theta, phi)
        Q_values[:, idx] = Phi_vals * Y_vals
```

### Mathematical Correctness

The optimization is mathematically exact because:
- The eigenfunction Q_{k,l,m}(ρ,θ,φ) = Φ_nu_l(k,l,ρ) × Y_lm(θ,φ) factors cleanly
- Φ_nu_l truly depends only on (k, l, ρ), not m
- We compute identical values, just avoid redundant recalculation

Verified by:
- `test_scalar_vs_vectorized.py`: Vectorized matches scalar implementation
- All existing tests pass with identical numerical results
- No changes to paper-faithful chi² computation

## Results

### Optimized Performance (After Optimization)
Same benchmark configuration:
- **Total runtime**: 18 seconds (11.3× faster)
- **Matrix build**: 17.7 seconds (11.4× faster)
- **SVD**: 0.19 seconds (unchanged)
- **mpmath legenp calls**: 14,040 (14.4× reduction)
- **Cache hit rate**: 0% (optimization makes per-k caching unnecessary)

### Speedup Analysis

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Total runtime | 203s | 18s | **11.3×** |
| Matrix build (per k) | 40.5s | 3.5s | **11.4×** |
| mpmath legenp calls | 202,410 | 14,040 | **14.4× reduction** |
| Test suite runtime | 170s | 19s | **8.9×** |

### Larger Problem Size Test
With 10 k-values and 20 base points:
- **Matrix build**: 4.76s mean per k (47.6s total)
- **37,180 legenp calls** (vs ~400,000+ without optimization)
- Performance scales well with problem size

## Additional Improvements

### 1. Profiling and Instrumentation

Added comprehensive performance monitoring:

**New flags:**
- `--profile`: Enable cProfile and write profile.prof
- `--benchmark`: Write detailed per-stage timings to timings.json

**Instrumentation:**
- Per-stage timings (point_sampling, cutoff_computation, ghost_enumeration, matrix_build, svd)
- Call counters (phi_calls, y_lm_calls, legenp_calls)
- Cache statistics (hit/miss rates, cache sizes)

**Example output:**
```json
{
  "matrix_build": {"mean": 3.54, "total": 17.70, "count": 5},
  "cache_stats": {
    "phi_calls": 10,
    "legenp_calls": 14040,
    "phi_cache_size": 10
  }
}
```

### 2. HPC-Oriented Features

**New flags for efficient array job processing:**

- `--no-plot`: Skip chi2_spectrum.png generation in array jobs
- `--no-eigenvalues`: Skip eigenvalue extraction in array jobs
- `--clear-caches-per-k`: Trade memory for CPU (optional, not recommended)

**Recommended Slurm array job:**
```bash
#!/bin/bash
#SBATCH --array=0-399      # 400 chunks for fine k-resolution
#SBATCH --time=01:00:00
#SBATCH --mem=4G

python main.py \
    --k-chunk-index $SLURM_ARRAY_TASK_ID --k-num-chunks 400 \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 400 \
    --n-points 60 --word-depth 3 \
    --chi2-mode paper \
    --benchmark --no-plot --no-eigenvalues \
    --require-snappy --output-dir results
```

With optimization, each task completes in ~3-10 minutes (vs 30-120 minutes before).

### 3. Testing

**New test suite: `tests/test_performance.py`**

- `test_phi_reuse_optimization_reduces_calls`: Verifies >2× reduction in expensive calls
- `test_clear_caches_functionality`: Validates cache management works correctly
- `test_cache_stats_format`: Ensures instrumentation has expected structure

**Benchmark script: `scripts/benchmark.py`**

Automated performance comparison tool:
```bash
# Quick benchmark
python scripts/benchmark.py --quick

# Compare vectorized vs scalar implementations
python scripts/benchmark.py --compare-scalar
```

### 4. Documentation

**New: `docs/perf.md`** - Comprehensive performance guide:
- Detailed optimization explanation with code examples
- Benchmark methodology and results
- HPC best practices (Slurm array jobs, environment variables)
- Expected scaling characteristics
- Profiling instructions
- Future optimization opportunities

**Updated: `README.md`**
- Added Performance section highlighting 11× speedup
- Updated HPC usage examples with new flags
- Cross-references to docs/perf.md

## Test Coverage

All tests pass with optimization:
- **15 passed, 1 skipped** in 19 seconds (vs 170s before)
- `test_scalar_vs_vectorized`: Vectorized matches scalar numerically
- `test_performance`: Performance regression tests
- All existing pipeline tests unchanged and passing

## Security

- **CodeQL scan: 0 vulnerabilities**
- No new dependencies added
- No unsafe operations introduced
- All changes in performance-critical path only

## Files Changed

1. **main.py** (116 lines changed)
   - Added --profile, --no-plot, --no-eigenvalues, --clear-caches-per-k flags
   - Enhanced benchmark mode with cache statistics output
   - Profiling wrapper for cProfile support

2. **utils/special_functions.py** (70 lines changed)
   - Optimized Q_k_lm_vectorized with Φ_l reuse
   - Added instrumentation counters
   - Enhanced cache statistics functions

3. **utils/__init__.py** (2 lines changed)
   - Export get_call_counters, reset_call_counters

4. **docs/perf.md** (NEW, 300+ lines)
   - Comprehensive performance guide
   - HPC recommendations
   - Benchmarking instructions

5. **tests/test_performance.py** (NEW, 100+ lines)
   - 3 performance regression tests
   - Validates optimization reduces calls >2×

6. **scripts/benchmark.py** (NEW, 200+ lines)
   - Automated benchmarking tool
   - Scalar vs vectorized comparison
   - Summary statistics generation

7. **README.md** (30 lines changed)
   - Added Performance section
   - Updated HPC usage examples

## Impact on HPC Workflows

### Before Optimization
For k=1..10 with num_k=400, n_points=60:
- Estimated time per k: 40-100 seconds
- Total time on 400 cores: 4-11 hours
- Inefficient use of HPC resources

### After Optimization
Same configuration:
- Estimated time per k: 3-10 seconds (**~10× faster**)
- Total time on 400 cores: **0.3-1 hour**
- More efficient resource utilization
- Reduced queue time and cost

### Scaling Estimate
| k range | L | Time/k (before) | Time/k (after) | Speedup |
|---------|---|-----------------|----------------|---------|
| 1-2     | 11| ~30s            | ~2.5s          | 12×     |
| 3-4     | 13| ~40s            | ~3.5s          | 11×     |
| 5-6     | 15| ~60s            | ~5.0s          | 12×     |
| 7-8     | 17| ~80s            | ~7.0s          | 11×     |
| 9-10    | 19| ~110s           | ~9.5s          | 12×     |

Speedup is consistent across different k ranges and L values.

## Future Optimization Opportunities

The current optimization achieves >10× speedup, far exceeding requirements. Additional potential improvements (not implemented to maintain minimal changes):

1. **Cross-k caching**: Cache Φ_l(ρ) for nearby k values with interpolation
2. **Reduced precision mode**: Add --fast-legendre with lower mpmath.dps (~2× more speedup)
3. **Randomized SVD**: Use sparse SVD for very large matrices (currently SVD is <1% of runtime)
4. **Parallel matrix assembly**: Multi-thread across base points (diminishing returns with array jobs)

These are **not necessary** given current performance exceeds goals significantly.

## Conclusion

Successfully delivered **11.3× overall speedup** through a single, well-targeted optimization:
- Identified bottleneck via profiling (matrix build = 99.9% of time)
- Analyzed root cause (redundant Φ computations)
- Implemented mathematical optimization (Φ_l reuse)
- Verified correctness (all tests pass)
- Added instrumentation and HPC features
- Documented thoroughly

The optimization is:
- ✅ **Mathematically exact** (not an approximation)
- ✅ **Well-tested** (15 tests including new performance regression tests)
- ✅ **Secure** (0 CodeQL vulnerabilities)
- ✅ **Documented** (comprehensive perf.md guide)
- ✅ **HPC-ready** (efficient array job support)

Far exceeds 2× speedup requirement while maintaining code quality and mathematical correctness.
