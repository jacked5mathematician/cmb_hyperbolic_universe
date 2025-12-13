# Chi-Squared Definitions

This document explains the different chi-squared definitions available in the pipeline and how they relate to the paper (eigenvalueprob.pdf).

## The Problem

Initial runs showed chi² values around ~1e-8, while the paper's Figure 1 shows chi² values of order O(1). This suggests a difference in how chi² is normalized or defined.

## Paper Definition (Equation 2.7)

The paper defines chi-squared as:
```
χ² = ||A·a||²
```

where:
- A is the constraint matrix (M × N)
- a is the solution vector from SVD (the right singular vector)
- For SVD: A = U·Σ·Vᵀ, the smallest singular values σ satisfy ||A·a||² = σ²

## Available Definitions

The `--chi2-definition` flag allows choosing between different interpretations:

### 1. `raw_residual` (default)
```
χ² = ||A·a||² = σ²
```

This is the direct interpretation of equation 2.7. The chi-squared value is simply the squared smallest singular value from SVD.

**Characteristics:**
- Units: Same as ||A||² (depends on matrix scaling)
- Scale: Can be very small (~1e-8) if matrix is well-conditioned
- Paper match: Literal interpretation of equation 2.7

### 2. `per_row`
```
χ² = ||A·a||² / M = σ² / M
```

Normalizes by the number of constraints M (rows of A). This gives the average squared residual per constraint.

**Characteristics:**
- Units: Average squared residual per constraint
- Scale: Smaller than raw_residual by factor of M
- Use case: Compare different M values

### 3. `ratio` (likely paper definition)
```
χ² = (σ_min / σ_max)²
```

Uses the condition number to measure how well-resolved the eigenmode is. This is the ratio of the smallest to largest singular value, squared.

**Characteristics:**
- Units: Dimensionless
- Scale: Always in range [0, 1]
- Naturally produces O(1) values (or smaller)
- **Most likely paper definition** because:
  - Produces O(1) scale matching Figure 1
  - Dimensionless (invariant to matrix scaling)
  - Standard measure of numerical conditioning

### 4. `frobenius`
```
χ² = ||A·a||² / ||A||_F² = σ² / Σ(σᵢ²)
```

Normalizes by the Frobenius norm of A. The Frobenius norm squared equals the sum of all squared singular values.

**Characteristics:**
- Units: Dimensionless
- Scale: Always in range [0, 1]
- Measures relative energy in smallest mode
- Use case: Compare how much "energy" is in the eigenmode vs total matrix

## Recommendation

Based on the scaling issue (1e-8 vs O(1)), we recommend using `--chi2-definition ratio` as the default. This is most likely what the paper intends because:

1. **Scale matches:** Produces O(1) or smaller values, matching Figure 1
2. **Physical meaning:** Measures how well the eigenmode is resolved relative to the overall matrix scale
3. **Dimensionless:** Independent of arbitrary matrix scaling
4. **Standard practice:** Condition number ratios are standard in numerical analysis

## Usage Examples

### Run pipeline with ratio definition (recommended)
```bash
python main.py --manifold "m003(-2,3)" \
    --k-min 1.0 --k-max 10.0 --num-k 100 \
    --n-points 20 \
    --chi2-definition ratio \
    --output-dir results
```

### Compare all definitions on existing spectrum
```bash
# First run pipeline with diagnostics (requires sigma_max, A_frobenius)
python main.py --manifold "m003(-2,3)" \
    --k-min 1.0 --k-max 10.0 --num-k 100 \
    --n-points 20 \
    --chi2-definition raw_residual \
    --output-dir results

# Then plot all definitions
python scripts/plot_chi2_definitions.py \
    --spectrum results/spectrum.npz \
    --output results/chi2_comparison.png \
    --all-ranks
```

### Legacy compatibility
The old `--chi2-mode` flag controls row normalization (separate from definition):
- `--chi2-mode paper` (default): No row normalization before SVD
- `--chi2-mode legacy`: Apply row normalization (old behavior)

Both modes work with all `--chi2-definition` options.

## Testing

Run unit tests to verify numerical correctness:
```bash
pytest tests/test_chi2_definitions.py -v
```

## Implementation Details

- Chi-squared computation: `utils/chi2.py`
- SVD integration: `utils/svd.py`
- Visualization: `scripts/plot_chi2_definitions.py`
- Unit tests: `tests/test_chi2_definitions.py`

## Note on Existing Spectrum Files

Older spectrum.npz files may not contain `sigma_max` and `A_frobenius` diagnostics required for ratio and frobenius definitions. In this case, the visualization script will only show raw_residual and per_row. Regenerate the spectrum with the updated code to get all definitions.
