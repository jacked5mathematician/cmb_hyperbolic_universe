# Scripts Directory

This directory contains utility scripts for analyzing and validating chi-squared spectra.

## Analysis and Debugging

### debug_singular_spectrum.py

Analyzes singular value spectrum and diagnoses eigenvalue sensitivity.

**Usage**:
```bash
# Analyze spectrum with known eigenvalues
python scripts/debug_singular_spectrum.py \
    --spectrum output_values/spectrum.npz \
    --manifold m188

# Or provide custom eigenvalues
python scripts/debug_singular_spectrum.py \
    --spectrum output_values/spectrum.npz \
    --paper-eigenvalues "4.41,4.64,5.12,5.40"

# Specify output location
python scripts/debug_singular_spectrum.py \
    --spectrum output_values/spectrum.npz \
    --manifold m188 \
    --output diagnostics.png
```

**Output**:
- Singular value analysis (condition numbers, rank estimates)
- Comparison to paper eigenvalues
- Chi-squared dynamic range analysis
- Diagnostic plots (chi², condition number, smallest singular value)

**Supported manifolds**:
- `m188`: m188(-1,1) from paper Table I
- `m003_thurston`: m003(-2,3) from paper Table II
- `m003_weeks`: m003(-3,1) from paper Table III

### paper_sanity_run.py

Runs paper sanity check for eigenvalue detection with automatic validation.

**Usage**:
```bash
# Run full validation for m188(-1,1)
python scripts/paper_sanity_run.py \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 200 \
    --n-points 50 \
    --output-dir output_values_local/paper_sanity

# Quick test with fewer samples
python scripts/paper_sanity_run.py \
    --manifold "m188(-1,1)" \
    --k-min 4.0 --k-max 6.0 --num-k 50 \
    --n-points 30

# Validate existing spectrum without re-running
python scripts/paper_sanity_run.py \
    --skip-run \
    --output-dir output_values_local/paper_sanity
```

**Output**:
- Runs main.py with paper-faithful parameters
- Validates detected minima against paper eigenvalues
- Reports success rate and false positives
- Calls debug_singular_spectrum.py for detailed analysis

**Exit codes**:
- 0: PASS or PARTIAL (acceptable)
- 1: FAIL or error

## Data Processing

### combine_spectra.py

Combines chunked spectrum files from HPC array jobs.

**Usage**:
```bash
python scripts/combine_spectra.py \
    --input-dir results \
    --output-file results/combined_spectrum.npz
```

### extract_eigenvalues.py

Extracts eigenvalues from chi-squared spectrum using local minima detection.

**Usage**:
```bash
python scripts/extract_eigenvalues.py \
    --spectrum output_values/spectrum.npz \
    --threshold 0.01 \
    --output eigenvalues.csv
```

## Plotting and Visualization

### plot_chi2_definitions.py

Creates comparison plots for different chi-squared definitions.

**Usage**:
```bash
# Compare all definitions for rank 1
python scripts/plot_chi2_definitions.py \
    --spectrum output_values/spectrum.npz \
    --output comparison.png

# Show multiple ranks
python scripts/plot_chi2_definitions.py \
    --spectrum output_values/spectrum.npz \
    --output comparison.png \
    --all-ranks --max-ranks 5
```

### analyze_chi2_scale.py

Analyzes chi-squared scale and provides statistics for all definitions.

**Usage**:
```bash
python scripts/analyze_chi2_scale.py \
    --spectrum output_values/spectrum.npz
```

## Benchmarking

### benchmark.py

Performance benchmarking and profiling utilities.

## Example Workflow

### 1. Run with diagnostics

```bash
python main.py \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 200 \
    --n-points 50 \
    --word-depth 4 \
    --chi2-mode paper \
    --require-snappy \
    --benchmark \
    --output-dir results/m188_run
```

### 2. Debug singular spectrum

```bash
python scripts/debug_singular_spectrum.py \
    --spectrum results/m188_run/spectrum.npz \
    --manifold m188 \
    --output results/m188_run/diagnostics.png
```

### 3. Validate eigenvalues

```bash
python scripts/paper_sanity_run.py \
    --skip-run \
    --output-dir results/m188_run
```

### 4. Extract eigenvalues

```bash
python scripts/extract_eigenvalues.py \
    --spectrum results/m188_run/spectrum.npz \
    --threshold 0.01 \
    --output results/m188_run/eigenvalues.csv
```

### 5. Plot comparisons

```bash
python scripts/plot_chi2_definitions.py \
    --spectrum results/m188_run/spectrum.npz \
    --output results/m188_run/chi2_comparison.png \
    --all-ranks
```

## Notes

- All scripts support `--help` for detailed usage information
- Scripts automatically add parent directory to Python path
- Most analysis scripts require scipy for peak finding
- Plotting scripts require matplotlib
