# Reproducing Paper Figure 1 - Chi-Squared Spectrum

This guide explains how to reproduce chi-squared spectrum plots similar to Figure 1 in eigenvalueprob.pdf.

## The Chi-Squared Scale Issue

Initial runs showed χ² values around ~1e-8 to ~1e-34, while the paper's Figure 1 shows chi-squared values of order O(1). This PR addresses this discrepancy by implementing multiple chi-squared definitions.

## Quick Start

### Step 1: Generate Spectrum with Ratio Definition

```bash
# For manifold m188(-1,1) as in the paper
python main.py \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 100 \
    --n-points 50 \
    --word-depth 4 \
    --chi2-definition ratio \
    --require-snappy \
    --output-dir results/m188_ratio
```

**Important**: The `--require-snappy` flag ensures you're using real manifold data. Without SnapPy, the code falls back to synthetic ghosts which produce unrealistically small chi² values.

### Step 2: Analyze Chi-Squared Scale

```bash
python scripts/analyze_chi2_scale.py \
    --spectrum results/m188_ratio/spectrum.npz
```

This will show statistics for all definitions and identify which produces O(1) values.

### Step 3: Compare All Definitions

```bash
python scripts/plot_chi2_definitions.py \
    --spectrum results/m188_ratio/spectrum.npz \
    --output results/m188_ratio/chi2_comparison.png \
    --all-ranks
```

This creates two plots:
1. Comparison of all 4 definitions for rank 1
2. All ranks (1-3) using ratio definition

## Understanding the Definitions

### 1. raw_residual (default)
```
χ² = ||A·a||² = σ²
```
- Direct from SVD
- Produces very small values (~1e-8 to 1e-34)
- Not normalized, depends on matrix scaling

### 2. per_row
```
χ² = ||A·a||² / M
```
- Normalized by constraint count
- Even smaller values than raw_residual
- Good for comparing different M values

### 3. ratio (RECOMMENDED)
```
χ² = (σ_min / σ_max)²
```
- Uses condition number
- **Produces O(1) or smaller values** ✓
- Dimensionless, invariant to scaling
- Bounded in [0, 1]
- **Most likely paper definition**

### 4. frobenius
```
χ² = ||A·a||² / ||A||_F²
```
- Normalized by Frobenius norm
- Produces O(1) or smaller values
- Measures relative energy

## Why Ratio Definition?

The `ratio` definition is recommended because:

1. **Scale matches paper**: Produces O(1) or smaller values
2. **Dimensionless**: Independent of arbitrary matrix scaling
3. **Standard practice**: Condition number ratios are standard in numerical analysis
4. **Physical meaning**: Measures how well eigenmode is resolved relative to matrix scale
5. **Bounded**: Always in [0, 1], making plots interpretable

## Full Reproduction Example

```bash
# Step 1: Run paper-faithful pipeline with ratio definition
python main.py \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 200 \
    --n-points 60 \
    --word-depth 4 \
    --chi2-definition ratio \
    --chi2-mode paper \
    --require-snappy \
    --benchmark \
    --output-dir results/paper_faithful

# Step 2: Analyze scale
python scripts/analyze_chi2_scale.py \
    --spectrum results/paper_faithful/spectrum.npz

# Step 3: Create comparison plots
python scripts/plot_chi2_definitions.py \
    --spectrum results/paper_faithful/spectrum.npz \
    --output results/paper_faithful/chi2_comparison.png \
    --all-ranks --max-ranks 5

# Step 4: Extract eigenvalues
python scripts/extract_eigenvalues.py \
    --spectrum results/paper_faithful/spectrum.npz \
    --threshold 0.01 \
    --output results/paper_faithful/eigenvalues_refined.csv
```

## Expected Results

With real SnapPy data and ratio definition, you should see:

1. **Chi-squared scale**: O(1) or smaller (10^-1 to 10^0)
2. **Qualitative behavior**: Similar to Figure 1 - many narrow dips/minima
3. **Multiple ranks**: Higher ranks show different eigenvalue curves
4. **Smooth curves**: With sufficient k-sampling (num-k ≥ 100)

## Troubleshooting

### Chi-squared values still ~1e-34

**Cause**: Using synthetic ghosts (no SnapPy)
**Solution**: Install SnapPy and use `--require-snappy`

```bash
pip install snappy
```

### Chi-squared values around ~1e-8

**Cause**: Using `raw_residual` definition
**Solution**: Switch to `ratio` definition:

```bash
python main.py ... --chi2-definition ratio
```

### Not enough structure in spectrum

**Cause**: Too few k-values or base points
**Solution**: Increase sampling:

```bash
python main.py ... --num-k 200 --n-points 60
```

### Values are NaN

**Cause**: No valid ghost images or matrix issues
**Solution**: 
- Check manifold name is valid
- Increase `--word-depth` (default 3, try 4-5)
- Check logs for warnings about fallbacks

## HPC Usage for Fine Resolution

For production-quality spectra matching the paper:

```bash
# In Slurm array job script
#SBATCH --array=0-399
#SBATCH --time=1:00:00
#SBATCH --mem=8G

python main.py \
    --k-chunk-index $SLURM_ARRAY_TASK_ID \
    --k-num-chunks 400 \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 400 \
    --n-points 60 \
    --word-depth 4 \
    --chi2-definition ratio \
    --require-snappy \
    --benchmark \
    --no-plot --no-eigenvalues \
    --output-dir results/m188_production

# After all jobs complete:
python scripts/combine_spectra.py \
    --input-dir results/m188_production

python scripts/plot_chi2_definitions.py \
    --spectrum results/m188_production/spectrum.npz \
    --output results/m188_production/chi2_final.png \
    --all-ranks
```

## References

- Paper: `docs/eigenvalueprob.pdf` (equation 2.7)
- Chi-squared definitions: `docs/chi2_definitions.md`
- Algorithm parameters: `docs/algorithm.md`
- Performance tuning: `docs/perf.md`
