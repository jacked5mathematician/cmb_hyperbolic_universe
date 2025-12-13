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

## HPC Usage

For large-scale runs on HPC clusters, use chunking:

```bash
# In Slurm array job (e.g., #SBATCH --array=0-9)
python main.py --k-chunk-index $SLURM_ARRAY_TASK_ID --k-num-chunks 10 \
    --require-snappy --output-dir results ...

# Combine results after all chunks complete
python scripts/combine_spectra.py --input-dir results
```

See `docs/algorithm.md` for detailed usage instructions.

## Testing

```bash
python -m pytest tests/ -v
``` 
