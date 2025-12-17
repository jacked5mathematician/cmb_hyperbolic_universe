# Local Diagnostics Report: High-k SVD Floor Investigation

## Purpose

This report investigates whether high-k failures are caused by:
- **(A)** Paper rho cutoff starving constraints (insufficient images)
- **(B)** Insufficient diversity of base points in function space
- **(C)** Numerical scaling / redundancy that persists even with many images

## Configuration

- Seed: 0
- Default n_points: 30
- Output directory: `output_diagnostics_suite`

## Run Summaries

### Run A: Low-k sanity (k=1.5, paper mode) - expected NOT at floor

| k | χ² | M | N | kept_pts | total_img | σ_min | σ_2 | σ_max | τ | below_τ | rank | nullity | cond | failure |
|---|---:|---:|---:|--------:|--------:|------:|----:|-----:|---:|-------:|-----:|-------:|-----:|--------|
| 1.50 | 0.0166 | 14850 | 144 | 10 | 550 | 1.29e-01 | 8.67e-01 | 8.94e-01 | 2.95e-12 | 0 | 144 | 0 | 6.9 |  |

### Run B: Mid-k transition (k=3.5, paper mode)

| k | χ² | M | N | kept_pts | total_img | σ_min | σ_2 | σ_max | τ | below_τ | rank | nullity | cond | failure |
|---|---:|---:|---:|--------:|--------:|------:|----:|-----:|---:|-------:|-----:|-------:|-----:|--------|
| 3.50 | 0.0181 | 3253 | 196 | 30 | 451 | 1.35e-01 | 1.16e+00 | 1.32e+00 | 9.55e-13 | 0 | 196 | 0 | 9.8 |  |

### Run C: High-k paper mode (k=9.8) - expected floor behavior

| k | χ² | M | N | kept_pts | total_img | σ_min | σ_2 | σ_max | τ | below_τ | rank | nullity | cond | failure |
|---|---:|---:|---:|--------:|--------:|------:|----:|-----:|---:|-------:|-----:|-------:|-----:|--------|
| 9.80 | - | 0 | 400 | 0 | 0 | - | - | - | - | - | - | - | - | **no_viable_points** |

### Run D: High-k adaptive_images mode (k=9.8)

| k | χ² | M | N | kept_pts | total_img | σ_min | σ_2 | σ_max | τ | below_τ | rank | nullity | cond | failure |
|---|---:|---:|---:|--------:|--------:|------:|----:|-----:|---:|-------:|-----:|-------:|-----:|--------|
| 9.80 | 0.0000 | 1827 | 400 | 26 | 319 | 1.32e-16 | 8.45e-01 | 8.65e-01 | 3.51e-13 | 1 | 293 | 107 | 6.53e+15 |  |

### Run E: High-k nearby pair for stability (k=9.80, 9.82)

| k | χ² | M | N | kept_pts | total_img | σ_min | σ_2 | σ_max | τ | below_τ | rank | nullity | cond | failure |
|---|---:|---:|---:|--------:|--------:|------:|----:|-----:|---:|-------:|-----:|-------:|-----:|--------|
| 9.80 | - | 0 | 400 | 0 | 0 | - | - | - | - | - | - | - | - | **no_viable_points** |
| 9.82 | - | 0 | 400 | 0 | 0 | - | - | - | - | - | - | - | - | **no_viable_points** |

### Run F_15: Sensitivity test: n_points=15 at k=9.8

| k | χ² | M | N | kept_pts | total_img | σ_min | σ_2 | σ_max | τ | below_τ | rank | nullity | cond | failure |
|---|---:|---:|---:|--------:|--------:|------:|----:|-----:|---:|-------:|-----:|-------:|-----:|--------|
| 9.80 | - | 0 | 400 | 0 | 0 | - | - | - | - | - | - | - | - | **no_viable_points** |

### Run F_30: Sensitivity test: n_points=30 at k=9.8

| k | χ² | M | N | kept_pts | total_img | σ_min | σ_2 | σ_max | τ | below_τ | rank | nullity | cond | failure |
|---|---:|---:|---:|--------:|--------:|------:|----:|-----:|---:|-------:|-----:|-------:|-----:|--------|
| 9.80 | - | 0 | 400 | 0 | 0 | - | - | - | - | - | - | - | - | **no_viable_points** |

### Run F_60: Sensitivity test: n_points=60 at k=9.8

| k | χ² | M | N | kept_pts | total_img | σ_min | σ_2 | σ_max | τ | below_τ | rank | nullity | cond | failure |
|---|---:|---:|---:|--------:|--------:|------:|----:|-----:|---:|-------:|-----:|-------:|-----:|--------|
| 9.80 | - | 0 | 400 | 0 | 0 | - | - | - | - | - | - | - | - | **no_viable_points** |

### Run G: Diversity FPS test: sample 60 points, select 30 via FPS at k=9.8

| k | χ² | M | N | kept_pts | total_img | σ_min | σ_2 | σ_max | τ | below_τ | rank | nullity | cond | failure |
|---|---:|---:|---:|--------:|--------:|------:|----:|-----:|---:|-------:|-----:|-------:|-----:|--------|
| 9.80 | - | 0 | 400 | 0 | 0 | - | - | - | - | - | - | - | - | **no_viable_points** |

## Analysis

### Low-k sanity (Run A, k=1.5)
- σ_min = 1.29e-01, τ = 2.95e-12
- below_τ = 0 → ✅ NOT at floor

### Mid-k transition (Run B, k=3.5)
- σ_min = 1.35e-01, τ = 9.55e-13
- below_τ = 0 → not at floor

### High-k paper mode (Run C, k=9.8)
- **FAILURE**: no_viable_points - paper rho window is too narrow
- This confirms hypothesis (A): paper rho cutoff starves constraints

### High-k adaptive mode (Run D, k=9.8)
- σ_min = 1.32e-16, τ = 3.51e-13
- below_τ = 1, total_images = 319
- rank = 293, nullity = 107
- rho_max_original = 1.839558389849811, rho_max = 2.589558389849811
- adaptive_expansions = 3

**Comparison C vs D:**
- Paper mode FAILED (no points), adaptive mode SUCCEEDED
- Adaptive yielded 319 images, rank=293
- This confirms adaptive rho expansion is necessary for high-k ✅

### Sensitivity to base-point count (Run F)

| n_points | result | σ_min | τ | below_τ | rank | nullity | total_images |
|----------|--------|------:|---:|-------:|-----:|-------:|------------:|
| 15 | **FAIL** | - | - | - | - | - | 0 |
| 30 | **FAIL** | - | - | - | - | - | 0 |
| 60 | **FAIL** | - | - | - | - | - | 0 |

### Diversity FPS test (Run G)

- **FAILURE**: no_viable_points - paper mode fails even with FPS

## Conclusions

**Q1 (Paper rho at high-k)**: Paper rho FAILS completely at k=9.8 - **no viable points** retained. The paper rho window is too narrow to provide any usable images. This confirms hypothesis **(A)**: paper rho cutoff starves constraints.

**Q2 (Adaptive vs Paper)**: Adaptive rho **rescues** the computation - paper mode fails completely while adaptive produces 319 images with rank=293, nullity=107.

   → However, adaptive mode still hits numerical floor (σ_min < τ). Hypothesis **(C)** (redundancy) is also present.

**Q3 (Base-point count)**: All n_points values (15, 30, 60) fail at k=9.8 with paper rho. The issue is rho window, not point count.

**Q4 (Diversity FPS)**: FPS run also failed (no_viable_points). Diversity selection cannot help when paper rho is too narrow.
