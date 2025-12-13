Code used for computing eigenvalues of the Laplace-Beltrami operator on M/Gamma using the method of ghosts.

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib mpmath joblib tqdm

# Optional: Install SnapPy for real manifold data
# pip install snappy

# Run a quick test
python main.py --small-test --self-check

# Run with benchmark mode
python main.py --manifold "m003(-2,3)" --k-min 1.0 --k-max 2.0 --num-k 10 \
    --n-points 20 --benchmark --output-dir output_test
```

## Structure

- `main.py` - Main pipeline entry point with HPC support
- `utils/` - Core utility functions (geometry, special functions, matrix generation)
- `scripts/` - Helper scripts for post-processing
- `tests/` - Test suite
- `docs/` - Algorithm documentation and usage guides
- `legacy/` - Deprecated code (for reference only)

## Performance

Recent optimizations achieve **11× speedup** in matrix assembly by reusing Φ_l(ρ) computations across all m values. This dramatically reduces expensive mpmath evaluations while preserving mathematical correctness.

For detailed performance information, benchmarking, and HPC best practices, see `docs/perf.md`.

## HPC Usage

For large-scale runs on HPC clusters, use chunking with optimized flags:

```bash
# In Slurm array job (e.g., #SBATCH --array=0-399)
python main.py \
    --k-chunk-index $SLURM_ARRAY_TASK_ID --k-num-chunks 400 \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 400 \
    --n-points 60 --word-depth 3 \
    --chi2-mode paper \
    --chi2-definition ratio \
    --require-snappy \
    --benchmark \
    --no-plot --no-eigenvalues \
    --output-dir results

# Combine results after all chunks complete
python scripts/combine_spectra.py --input-dir results

# Compare chi-squared definitions
python scripts/plot_chi2_definitions.py \
    --spectrum results/spectrum.npz \
    --output results/chi2_comparison.png \
    --all-ranks
```

**New flags for HPC**:
- `--no-plot`: Skip plotting in array jobs (combine step will plot)
- `--no-eigenvalues`: Skip eigenvalue extraction in array jobs
- `--benchmark`: Write detailed timing statistics

## Chi-Squared Definitions

The pipeline supports multiple chi-squared definitions via `--chi2-definition`:

- `raw_residual` (default): χ² = σ² (direct from SVD)
- `per_row`: χ² = σ²/M (average per constraint)
- `ratio`: χ² = (σ_min/σ_max)² (condition number, **recommended**)
- `frobenius`: χ² = σ²/||A||_F² (relative to total matrix norm)

The `ratio` definition is recommended as it likely matches the paper's Figure 1, producing O(1) values instead of ~1e-8.

For detailed explanation, see `docs/chi2_definitions.md`.

```bash
# Run with recommended ratio definition
python main.py --manifold "m003(-2,3)" \
    --k-min 1.0 --k-max 10.0 --num-k 100 \
    --n-points 20 \
    --chi2-definition ratio \
    --output-dir results
```

See `docs/perf.md` for HPC recommendations and `docs/algorithm.md` for detailed usage instructions.

## Paper-Faithful Sanity Run

To reproduce spectrum similar to Figure 1 in the paper (docs/eigenvalueprob.pdf) for manifold m188(-1,1):

```bash
# Fast sanity run with paper-faithful settings (seeded for reproducibility)
python main.py \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 100 \
    --n-points 20 \
    --seed 42 \
    --chi2-mode paper \
    --word-depth 3 \
    --output-dir output_values_local/paper_sanity \
    --self-check

# For HPC/production runs with finer resolution (closer to paper)
python main.py \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 500 \
    --n-points 50 \
    --seed 42 \
    --chi2-mode paper \
    --word-depth 4 \
    --output-dir output_values_local/paper_full \
    --require-snappy
```

**Paper algorithm parameters** (from eigenvalueprob.pdf Section II):
- L = 10 + floor(k) - Maximum spherical harmonic degree
- c = 10 + floor(100/k) - Oversampling ratio M/N
- ℓ_min = 5 - Minimum l for rho_min cutoff
- Chi² = ||A·a||² (no row normalization in paper mode)

The `--chi2-mode paper` flag ensures chi-squared is computed as in equation 2.7 of the paper, without ad-hoc row normalization. Use `--chi2-mode legacy` to preserve the old normalization behavior.

**Expected output:**
- Qualitatively resembles Figure 1 in paper: many narrow minima/dips (especially rank 1)
- No extreme step discontinuities
- No pathological flatlining near machine precision

See `docs/paper_algorithm_summary.md` for details on paper algorithm.

## Testing

```bash
python -m pytest tests/ -v
``` 
