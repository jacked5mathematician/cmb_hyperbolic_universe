# Paper Eigenvalues for m188(-1,1)

From Table I in eigenvalueprob.pdf, the computed eigenvalue spectrum (q²) for m188(-1,1):

| q² Value | Multiplicity |
|----------|--------------|
| 20.4     | 1            |
| 22.6     | 1            |
| 27.2     | 1            |
| 30.2     | 1            |
| 39.6     | 1            |
| 46.2     | 1            |
| 51.8     | 1            |
| 55.3     | 1            |
| 60.1     | 1            |
| 70.6     | 2            |
| 75.5     | 2            |
| 78.8     | 1            |
| 80.9     | 1            |
| 83.1     | 1            |
| 86.0     | 1            |
| 96.8     | 2            |
| 98.0     | 1            |
| 99.4     | 1            |

These values serve as ground truth for validating our implementation.

## Conversion to k-values

Since q² = k² + 1, we can compute the corresponding k-values:

k = sqrt(q² - 1)

Example conversions:
- q² = 20.4 → k ≈ 4.41
- q² = 22.6 → k ≈ 4.64
- q² = 27.2 → k ≈ 5.12
- q² = 30.2 → k ≈ 5.40
- q² = 39.6 → k ≈ 6.21
- q² = 46.2 → k ≈ 6.73

## Usage for Validation

When running the pipeline for m188(-1,1) with k ∈ [1, 10], we expect to see:
1. Clear minima in chi² spectrum near these k-values
2. Multiplicities matching the table (especially the degeneracies at k≈8.34, k≈8.65, etc.)
3. No spurious minima far from these values

Run with:
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
