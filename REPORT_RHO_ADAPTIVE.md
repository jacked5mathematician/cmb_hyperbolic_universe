# Adaptive Rho Window Comparison Report

Generated: 2025-12-15 02:32:50

## Configuration

- k range: [1.0, 10.0]
- Number of k samples: 50
- Base points: 30

## Summary

| Mode | Valid χ² | Success Rate | First Failure k | Mean Kept Points | Mean Images/Point |
|------|----------|--------------|-----------------|------------------|-------------------|
| paper | 29 | 58.0% | 6.33 | 9.9 | 23.3 |
| adaptive_images | 42 | 84.0% | 8.71 | 11.5 | 19.4 |
| adaptive_rank | 50 | 100.0% | None | 12.7 | 18.1 |

## Improvement Over Paper Baseline

- **adaptive_images mode**: +13 additional valid k-values
- **adaptive_rank mode**: +21 additional valid k-values

## Detailed Results (Sample)

### First 20 k-values

| k | Paper χ² | Images χ² | Rank χ² | Paper rho_max | Images rho_max | Rank rho_max |
|---|----------|-----------|---------|---------------|----------------|--------------|
| 1.000 | 3.91e-05 | 3.91e-05 | 3.91e-05 | 5.634 | 5.634 | 5.634 |
| 1.184 | 1.86e-03 | 1.86e-03 | 1.86e-03 | 5.113 | 5.113 | 5.113 |
| 1.367 | 8.89e-03 | 8.89e-03 | 8.89e-03 | 4.713 | 4.713 | 4.713 |
| 1.551 | 0.0174 | 0.0174 | 0.0174 | 4.393 | 4.393 | 4.393 |
| 1.735 | 0.0200 | 0.0200 | 0.0200 | 4.129 | 4.129 | 4.129 |
| 1.918 | 0.0160 | 0.0160 | 0.0160 | 3.905 | 3.905 | 3.905 |
| 2.102 | 7.51e-03 | 7.51e-03 | 7.51e-03 | 3.795 | 3.795 | 3.795 |
| 2.286 | 7.91e-03 | 7.91e-03 | 7.91e-03 | 3.625 | 3.625 | 3.625 |
| 2.469 | 0.0153 | 0.0153 | 0.0153 | 3.475 | 3.475 | 3.475 |
| 2.653 | 0.0290 | 0.0290 | 0.0290 | 3.340 | 3.340 | 3.340 |
| 2.837 | 0.0391 | 0.0391 | 0.0391 | 3.219 | 3.219 | 3.219 |
| 3.020 | 0.0447 | 0.0447 | 0.0447 | 3.183 | 3.183 | 3.183 |
| 3.204 | 0.0313 | 0.0313 | 0.0313 | 3.081 | 3.081 | 3.081 |
| 3.388 | 0.0285 | 0.0285 | 0.0285 | 2.987 | 2.987 | 2.987 |
| 3.571 | 0.0149 | 0.0149 | 0.0149 | 2.899 | 2.899 | 2.899 |
| 3.755 | 0.0119 | 0.0119 | 0.0119 | 2.818 | 2.818 | 2.818 |
| 3.939 | 5.34e-03 | 5.34e-03 | 5.34e-03 | 2.742 | 2.742 | 2.742 |
| 4.122 | 4.17e-03 | 4.17e-03 | 4.17e-03 | 2.740 | 2.740 | 2.740 |
| 4.306 | 3.20e-04 | 3.20e-04 | 3.20e-04 | 2.673 | 2.673 | 2.673 |
| 4.490 | 1.14e-31 | 1.14e-31 | 1.14e-31 | 2.609 | 2.609 | 2.609 |

### Last 5 k-values

| k | Paper χ² | Images χ² | Rank χ² | Paper rho_max | Images rho_max | Rank rho_max |
|---|----------|-----------|---------|---------------|----------------|--------------|
| 9.265 | NaN | NaN | 1.53e-34 | 1.907 | 2.157 | 2.407 |
| 9.449 | NaN | NaN | 2.59e-35 | 1.884 | 2.134 | 2.384 |
| 9.633 | NaN | NaN | 5.01e-35 | 1.860 | 2.110 | 2.360 |
| 9.816 | NaN | NaN | 2.45e-36 | 1.838 | 2.088 | 2.338 |
| 10.000 | NaN | NaN | 4.17e-35 | 1.861 | 2.111 | 2.361 |

## Adaptive Mode Statistics

### adaptive_images

- Max rho expansion: 0.250
- Mean expansion steps: 0.42

### adaptive_rank

- Max rho expansion: 0.500
- Mean expansion steps: 0.58

## Conclusions

1. **Paper mode failure pattern**: The paper-faithful cutoffs lead to insufficient ghost images at high k, causing point dropout and NaN chi² values.

2. **Adaptive strategies effectiveness**: Compare success rates to determine if adaptive window expansion recovers valid eigenvalue estimates.

3. **Trade-offs**: Expanding rho_max increases images but may include contributions from beyond the intended radial cutoff. Investigate if chi² values remain physically meaningful.
