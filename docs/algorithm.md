## Pipeline overview

This repository follows the Cornish & Spergel “method of ghosts” conventions:

- The scan variable is **k**.
- `nu == k` everywhere (no `sqrt(k^2+1)`), and `q^2 = k^2 + 1`.
- We work in pseudospherical coordinates `(rho, theta, phi)`; rho is the hyperbolic radial distance.

### Rho cutoff policy

`compute_rho_cutoffs(k, L, l_min, threshold)` scans for the first rho where
`|X_k^ell(rho) * sinh(rho)| <= threshold` for `ell = l_min` (rho_min) and `ell = L` (rho_max).
If no crossing is found up to a safety cap, it logs a warning, marks `fallback_used=True`, and falls
back to an envelope heuristic. The function always guarantees `0 <= rho_min < rho_max` or raises a
clear error.

### Ghost enumeration

`enumerate_ghost_images` loads SnapPy pairing matrices when available, augments them with inverses,
and performs a BFS over group words until images in the rho window are collected. Matrices are
deduplicated by hashing rounded entries. If SnapPy is unavailable, a deterministic synthetic
enumerator produces images in the requested rho band so tests and diagnostics still function.

### Points inside the Dirichlet domain

`sample_points_in_dirichlet_domain` applies the Dirichlet-domain inequality
`d(x, p0) <= d(x, gamma(p0))` for a finite set of group elements (words up to a small depth).
If generators are unavailable it falls back to rejection sampling inside the Poincaré ball. The
function returns both Poincaré coordinates and their pseudospherical transforms for diagnostics.

### Matrix assembly and invariants

`generate_matrix_system` expects `points_images` as `List[List[(rho, theta, phi)]]`. It builds
`A(k)` with `N = (L+1)^2` columns (one per `(l, m)`) and `M = sum_i n_i(n_i-1)/2` rows (one per
unordered pair of images for each base point). Assertions check both formulas.

### SVD diagnostics

`solve_system_via_svd_numeric` returns the smallest singular values (and their squared values
`chi^2`). These diagnostics are logged per-k to track multiplicities.

### Running locally vs HPC

#### Local quick run

For development and testing:
```bash
# Dry run to validate configuration
python main.py --small-test --dry-run

# Quick test with self-checks
python main.py --small-test --self-check

# Benchmark mode to measure performance
python main.py --manifold "m003(-2,3)" --k-min 1.0 --k-max 1.5 --num-k 5 \
    --n-points 10 --benchmark --output-dir output_bench
```

The `--benchmark` flag writes detailed timing information to `timings.json` in the output directory,
including mean/min/max times for cutoff computation, point sampling, ghost enumeration, matrix 
build, and SVD.

#### HPC chunked run (Slurm array)

For large k-sweeps on HPC clusters, use chunking to parallelize across nodes:

```bash
# In your Slurm script, use array indices to process chunks:
#SBATCH --array=0-9  # 10 chunks

python main.py --manifold "m003(-2,3)" \
    --k-min 1.0 --k-max 10.0 --num-k 1000 \
    --n-points 50 \
    --k-chunk-index $SLURM_ARRAY_TASK_ID \
    --k-num-chunks 10 \
    --output-dir results
```

Each chunk will create a subdirectory `results/chunk_N/` containing `spectrum.npz` for its 
k-value range.

#### Combine chunked results

After all chunks complete, merge them:

```bash
python scripts/combine_spectra.py --input-dir results --output-file results/spectrum.npz
```

This will combine all `chunk_*/spectrum.npz` files into a single `spectrum.npz`.

#### SnapPy dependency control

To ensure you're using real manifold data (not synthetic fallback) on HPC:

```bash
python main.py --manifold "m003(-2,3)" --require-snappy ...
```

The `--require-snappy` flag will exit with an error if SnapPy is unavailable or cannot load
the manifold, preventing accidental fallback to synthetic ghost images.

#### Expected outputs

For each run, the pipeline creates:
- `spectrum.npz`: k-values and chi² ranks (1-5), plus metadata (L, M, N, rho_min, rho_max, etc.)
- `chi2_spectrum.png`: Plot of chi² vs k
- `eigenvalues.csv`: Extracted eigenvalues (if `--eigen-threshold` set)
- `self_check_report.json`: Validation results (if `--self-check` used)
- `timings.json`: Performance timings (if `--benchmark` used)

Logging records per-k values of `(L, c, rho_min, rho_max)`, base-point retention, matrix shape,
and the smallest singular values.
