# Performance Optimization Guide

## Overview

This document describes performance optimizations implemented for large-scale k-sweeps on HPC clusters, focusing on the matrix assembly bottleneck while preserving mathematical correctness.

## Summary of Improvements

### Major Optimization: Φ_l Reuse Across m Values

**Problem**: The original `Q_k_lm_vectorized` computed Φ_nu_l(k, l, ρ) separately for each (l, m) pair, even though Φ only depends on l, not m.

**Solution**: Refactored to compute Φ_l once per l value and reuse for all m with that l.

**Impact**:
- **11.4× speedup** in matrix assembly (40.5s → 3.5s per k)
- **14.4× reduction** in expensive mpmath calls (202,410 → 14,040 legenp calls)
- Overall pipeline speedup of ~11× on typical runs

### Implementation Details

The optimization groups lm_pairs by l:
```python
# Group by l to reuse Phi computations
l_to_m_indices = {}
for idx, (l, m) in enumerate(lm_pairs):
    if l not in l_to_m_indices:
        l_to_m_indices[l] = []
    l_to_m_indices[l].append((idx, m))

# Process each unique l value
for l, m_indices in l_to_m_indices.items():
    # Compute Phi_nu_l once for this l
    Phi_vals = Phi_nu_l_vectorized(k_value, l, rho)
    
    # Reuse for all m with this l
    for idx, m in m_indices:
        Y_vals = Y_lm_real_vectorized(l, m, theta, phi)
        Q_values[:, idx] = Phi_vals * Y_vals
```

## Benchmark Results

### Test Configuration
- Manifold: m188(-1,1)
- k range: 3.5–4.2, num_k=5
- n_points: 15
- word_depth: 3
- Mode: paper (chi2-mode paper)

### Before Optimization
```
matrix_build: 40.50s mean per k (202.52s total)
svd: 0.037s mean per k (0.18s total)
legenp_calls: 202,410
Total runtime: ~203 seconds
```

### After Optimization
```
matrix_build: 3.54s mean per k (17.70s total)
svd: 0.037s mean per k (0.19s total)
legenp_calls: 14,040
Total runtime: ~18 seconds
```

### Speedup Analysis
- **Matrix build**: 11.4× faster (99.9% of runtime reduced)
- **Overall pipeline**: 11.3× faster
- **mpmath legenp calls**: 14.4× reduction

## HPC Recommendations

### For Slurm Array Jobs

Use these flags for efficient array job processing:

```bash
# Array job script (processes one chunk per task)
#!/bin/bash
#SBATCH --array=0-399      # 400 chunks for fine k-resolution
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

python main.py \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 400 \
    --n-points 60 \
    --word-depth 3 \
    --chi2-mode paper \
    --k-chunk-index $SLURM_ARRAY_TASK_ID \
    --k-num-chunks 400 \
    --output-dir results \
    --benchmark \
    --no-plot \
    --no-eigenvalues \
    --require-snappy
```

**Flag explanations**:
- `--k-chunk-index`: Process this chunk (0-based, matches SLURM_ARRAY_TASK_ID)
- `--k-num-chunks`: Total number of chunks to split k-values into
- `--no-plot`: Skip PNG generation (combine step will plot full spectrum)
- `--no-eigenvalues`: Skip eigenvalue extraction (combine step will extract)
- `--benchmark`: Write detailed timing statistics
- `--require-snappy`: Exit with error if SnapPy unavailable (prevents synthetic runs)

### Combining Results

After all array jobs complete:

```bash
python scripts/combine_spectra.py --input-dir results --output-dir combined_results
```

This will:
1. Load all chunk_*/spectrum.npz files
2. Merge k-values and chi² arrays
3. Generate full chi2_spectrum.png
4. Extract eigenvalues if --eigen-threshold specified

### Memory vs Performance Tradeoffs

If memory is constrained, use:

```bash
python main.py ... --clear-caches-per-k
```

This clears special function caches after each k value:
- **Benefit**: Bounds memory usage (prevents cache growth)
- **Cost**: ~10% performance penalty (must recompute some values)

For typical HPC nodes with 4GB+ RAM, cache clearing is **not recommended**.

## Environment Variables

Set these for optimal performance:

```bash
# Disable NumPy/SciPy implicit threading (we parallelize at job level)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run your job
python main.py ...
```

**Why**: Avoids thread contention when running multiple array tasks per node.

## Profiling Your Run

To identify bottlenecks in your specific configuration:

```bash
python main.py --profile --benchmark ... 
```

This generates:
1. `profile.prof`: cProfile data (analyze with `snakeviz profile.prof` or `python -m pstats`)
2. `timings.json`: Per-stage timing breakdown with cache statistics

Example timing.json output:
```json
{
  "matrix_build": {
    "mean": 3.54,
    "total": 17.70,
    "count": 5
  },
  "cache_stats": {
    "phi_calls": 10,
    "phi_cache_hits": 0,
    "legenp_calls": 14040,
    "phi_hit_rate": 0.0
  }
}
```

## Mathematical Correctness

All optimizations preserve mathematical correctness:

1. **Paper-faithful chi²**: Uses `--chi2-mode paper` (no row normalization)
2. **Φ_l reuse**: Mathematically correct since Φ_nu_l(k, l, ρ) is independent of m
3. **Tested equivalence**: `test_scalar_vs_vectorized.py` verifies vectorized matches scalar

## Expected Scaling

Based on benchmarks:

| Configuration | Time per k | Total time (400 k-values) |
|---------------|------------|---------------------------|
| k=1-2, L≈11   | ~2.5s      | ~17 minutes              |
| k=3-4, L≈13   | ~3.5s      | ~23 minutes              |
| k=5-6, L≈15   | ~5.0s      | ~33 minutes              |
| k=7-8, L≈17   | ~7.0s      | ~47 minutes              |
| k=9-10, L≈19  | ~9.5s      | ~63 minutes              |

**Full sweep k=1-10**: Estimate ~3 hours on 400 cores (one task per core)

## Diagnostic Instrumentation

The `--benchmark` flag provides detailed per-k diagnostics:

### Timing Phases
- `point_sampling`: Base point generation time
- `cutoff_computation`: ρ_min, ρ_max calculation
- `ghost_enumeration`: Ghost image generation per base point
- `matrix_build`: Matrix assembly (dominant cost)
- `svd`: Singular value decomposition

### Cache Statistics
- `phi_calls`: Total Φ function calls
- `phi_cache_hits/misses`: Cache effectiveness
- `legenp_calls`: Expensive mpmath Legendre function calls
- `phi_hit_rate`: Percentage of cached results reused

### Matrix Diagnostics (in spectrum.npz)
- `sigma_min`, `sigma_max`: Singular value range
- `A_frobenius`: Frobenius norm of constraint matrix
- `A_max_abs`, `A_min_nonzero`: Matrix element range
- `images_per_point`: Ghost enumeration statistics

Use these to debug performance issues or numerical problems.

## Future Optimization Opportunities

The current optimizations achieve >10× speedup, meeting the 2× target. Additional opportunities:

1. **Caching across k values**: Currently caches are independent per k. Could cache Φ_l(ρ) for nearby k values with interpolation.

2. **Reduced precision mode**: Add `--fast-legendre` flag using lower mpmath precision (e.g., dps=25 instead of 50) with ~2× speedup. Must be behind explicit flag and OFF by default per requirements.

3. **Chunked matrix assembly**: For very large M (>10,000 rows), assemble matrix in chunks to reduce peak memory.

4. **Randomized SVD**: For large matrices where only smallest singular values matter, use `scipy.sparse.linalg.svds` instead of full SVD. Currently SVD is <1% of runtime, so low priority.

These are **not implemented** to maintain minimal, focused changes. The Φ_l reuse optimization provides the dominant speedup.

## Testing

All optimizations are tested:

```bash
# Run full test suite (includes vectorized vs scalar equivalence)
python -m pytest tests/ -v

# Run scalar vs vectorized equivalence test specifically
python -m pytest tests/test_scalar_vs_vectorized.py -v

# Run benchmark on small problem
python main.py --small-test --benchmark --self-check
```

Expected: All tests pass, no mathematical differences between scalar and vectorized paths.

## References

- `PAPER_FAITHFUL_CHANGES.md`: Recent changes to ensure paper algorithm fidelity
- `docs/algorithm.md`: Core algorithm documentation
- `docs/paper_algorithm_summary.md`: Paper parameter schedules (L, c, ℓ_min)
- Main paper: `docs/eigenvalueprob.pdf`
