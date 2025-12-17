# Adaptive rho Window Comparison Report

k in [1.0, 3.0], N=24, base_points=20

## Overall Summary
| Mode | Success Rate | Mean chi2 | Floor Fail |
|------|-------------|-----------|------------|
| paper | 100.0% | 0.0114 | 0.0% |
| adaptive_images | 100.0% | 0.0114 | 0.0% |
| adaptive_rank | 100.0% | 0.0114 | 0.0% |

## Band Statistics
### paper
| Band | N | Success | Mean chi2 | Mean rho | Images | sigma_min | tau | Floor Fail |
|------|---|---------|-----------|----------|--------|-----------|-----|------------|
| [1,3) | 23 | 100% | 0.0113 | 4.07 | 39.3 | 9.97e-02 | 2.00e-12 | 0% |
| [3,5) | 1 | 100% | 0.0140 | 3.19 | 19.4 | 1.18e-01 | 9.23e-13 | 0% |

### adaptive_images
| Band | N | Success | Mean chi2 | Mean rho | Images | sigma_min | tau | Floor Fail |
|------|---|---------|-----------|----------|--------|-----------|-----|------------|
| [1,3) | 23 | 100% | 0.0113 | 4.07 | 39.3 | 9.97e-02 | 2.00e-12 | 0% |
| [3,5) | 1 | 100% | 0.0140 | 3.19 | 19.4 | 1.18e-01 | 9.23e-13 | 0% |

### adaptive_rank
| Band | N | Success | Mean chi2 | Mean rho | Images | sigma_min | tau | Floor Fail |
|------|---|---------|-----------|----------|--------|-----------|-----|------------|
| [1,3) | 23 | 100% | 0.0113 | 4.07 | 39.3 | 9.97e-02 | 2.00e-12 | 0% |
| [3,5) | 1 | 100% | 0.0140 | 3.19 | 19.4 | 1.18e-01 | 9.23e-13 | 0% |
