# Local Function-Space Diagnostics Report

**Branch:** `diagnostics/local-gram-diversity`  
**Commit:** `c1c2fab86338113c15f3ba2a6de3ccf1d896bd79`  
**Machine:** Darwin arm64 (MacBook Air)  
**Date:** 2025-12-18  
**Artifacts:** `reports/artifacts_local/`

---

## Executive Summary

High-k rank deficiency is caused **primarily by rho-starvation (Hypothesis A)** when using paper-mode cutoffs. The paper's rho window is too narrow to collect enough images at high k, resulting in complete failure (0 viable points). When adaptive rho expansion is enabled, full rank is achieved consistently up to k≈14, at which point basis size (N=625) begins to challenge conditioning.

**Key findings:**

| Hypothesis | Verdict | Evidence |
|:-----------|:--------|:---------|
| **(A) Rho-starvation** | **CONFIRMED – Primary cause** | Paper mode: 0/20 points at k=9.8. Adaptive modes recover full rank. |
| **(B) Function-space diversity** | Minor factor | Random, FPS, and QR selection all achieve σ_min ≈ 0.023 (within 4%). |
| **(C) Numerical precision** | Emerging at L≥24 | L=24 (k=14) shows nullity=16 with float64. Precision probe needed for HPC. |

---

## Step 1: Rho Mode Comparison at k=9.8

Three rho-window strategies tested at k=9.8 (L=19, N=400):

| Run | Mode | kept_points | M | rank | nullity | σ_min | τ | σ_min/τ | rho_max |
|:----|:-----|:------------|:--|:-----|:--------|:------|:--|:--------|:--------|
| P | `paper` | 0 | 0 | 0 | 0 | – | – | – | 1.84 (fixed) |
| Aimg | `adaptive_images` | 19 | 1515 | 228 | **172** | 6.4e-17 | 2.7e-13 | 2.4e-4 | 2.59 |
| Arank | `adaptive_rank` | 20 | 5499 | 400 | **0** | 1.5e-2 | 1.4e-12 | 1.1e+10 | 3.09 |

### Interpretation

1. **Paper mode (P):** Complete failure. No images survive the narrow rho window [1.24, 1.84].  
2. **Adaptive images (Aimg):** Expands rho to 2.59, collects 1515 images, but 43% of basis directions (172/400) remain in the null space. σ_min ≈ machine epsilon → essentially rank-deficient.  
3. **Adaptive rank (Arank):** Expands rho to 3.09 (5 expansions), achieves full rank=400, σ_min=1.5e-2 is 10 orders of magnitude above τ.

**Conclusion:** The paper cutoff starves the system at high k. Adaptive rank expansion is necessary and sufficient.

---

## Step 2: Basis Size (L) Sweep with Adaptive Rank

Different k values produce different basis sizes L = ⌊k + 0.5⌋ (approximately). All runs use `--rho-window-mode adaptive_rank` with 20 points:

| k | L | N=(L+1)² | rank | nullity | σ_min | condition | rho_max | expansions | Health |
|:--|:--|:---------|:-----|:--------|:------|:----------|:--------|:-----------|:-------|
| 2.5 | ~9 | 169 | 169 | 0 | 9.8e-2 | 11 | 3.45 | 0 | ✅ Healthy |
| 5.0 | ~14 | 256 | 256 | 0 | 1.8e-2 | 63 | 2.76 | 1 | ✅ Healthy |
| 9.8 | ~19 | 400 | 400 | 0 | 1.5e-2 | 74 | 3.09 | 5 | ✅ Healthy |
| 14.0 | ~24 | 625 | 609 | **16** | 7.8e-16 | 1.8e+15 | 3.38 | 7 | ⚠️ Rank loss |

### Interpretation

- For L ≤ 19 (N ≤ 400), adaptive rank achieves full rank with condition numbers < 100.
- At L=24 (N=625), even with 7 rho expansions, **16 null directions** appear. This signals either:
  - Insufficient image count (need more points or deeper enumeration), or
  - Float64 precision limits for large L.

**Conclusion:** The method is reliable up to L≈20 on local hardware. Beyond that, HPC runs with more points and/or extended precision may be required.

---

## Step 3: Selection Method Comparison at k=9.8

Compared three point-selection strategies, starting with 30 base points and selecting 15:

| Method | σ_min | τ | σ_min/τ | condition | kept_points | rho_max | expansions |
|:-------|:------|:--|:--------|:----------|:------------|:--------|:-----------|
| Random (baseline) | 2.28e-2 | 1.03e-12 | **2.2e+10** | 47 | 30→30 | 2.84 | 4 |
| Geometric FPS | 2.36e-2 | 2.36e-12 | **1.0e+10** | 55 | 30→15 | 3.34 | 6 |
| Feature-QR | 2.31e-2 | 2.57e-12 | **9.0e+9** | 58 | 30→15 | 3.34 | 6 |

### Interpretation

All three methods achieve similar σ_min (~0.023) and full rank. The differences are within ~4%:

- **Random:** Slight edge in condition number (47) because more points were kept.
- **FPS/QR:** Slightly higher σ_min per point, but need more rho expansion.

**Conclusion:** Function-space diversity (Hypothesis B) is **not a bottleneck** at this scale. The adaptive rho mechanism dominates. For larger N, FPS or QR may offer marginal improvements.

---

## Step 4: Precision Probe (Deferred to HPC)

Float64 results at L=24 show emerging precision issues (nullity=16). A proper comparison of:
- `np.float64` vs `np.longdouble` Gram matrices
- `mpmath.mp.dps=50` for basis function evaluation

...is computationally expensive and deferred to HPC. The signature to watch is:

```
gram_smallest_20: [1.28e-05, ..., -3.1e-16]  # sign flips indicate numerical noise
```

---

## Step 5: Enumeration Depth (Paper Mode)

Testing whether increasing `--word-depth` from 14 to 20 helps paper mode was interrupted due to runtime (~minutes per depth level). This is HPC territory.

---

## Summary Table

| k | L | Paper Result | Adaptive Rank Result | Root Cause |
|:--|:--|:-------------|:---------------------|:-----------|
| 2.5 | 9 | – | ✅ Full rank | N/A (easy) |
| 5.0 | 14 | – | ✅ Full rank | N/A (moderate) |
| 9.8 | 19 | ❌ 0 points | ✅ Full rank | Rho-starvation (A) |
| 14.0 | 24 | – | ⚠️ nullity=16 | Precision/scaling (C) |

---

## HPC Handoff Recommendations

### Verified CLI Flags

```bash
# Healthy run at k=9.8 (full rank, σ_min >> τ)
python main.py \
  --k-min 9.8 --k-max 9.8 --num-k 1 \
  --n-points 20 --seed 42 \
  --rho-window-mode adaptive_rank \
  --extended-diagnostics \
  --output-dir output_k9p8
```

### Recommended HPC Sweep

```bash
# Full k-sweep with adaptive rank
python main.py \
  --k-min 2.0 --k-max 15.0 --num-k 50 \
  --n-points 50 \
  --rho-window-mode adaptive_rank \
  --extended-diagnostics \
  --output-dir output_sweep_adaptive

# For k > 12, consider:
#   --n-points 100    # More constraints
#   --word-depth 18   # Deeper enumeration
```

### Open Questions for HPC

1. **At what k does nullity become unavoidable even with many points?**  
   → Sweep k ∈ [12, 20] with n_points ∈ {50, 100, 200}.

2. **Does extended precision (longdouble or mpmath) eliminate L=24 nullity?**  
   → Implement `--precision longdouble` flag for Gram computation.

3. **Is the nullity due to true linear dependence or numerical loss?**  
   → Compare Gram eigenvalues between float64 and float128.

---

## Artifacts

All CSV diagnostics and logs are in `reports/artifacts_local/`:

```
reports/artifacts_local/
├── P/           # Paper mode (failed)
├── Aimg/        # Adaptive images
├── Arank/       # Adaptive rank
├── L9/          # k=2.5 sweep
├── L14/         # k=5.0 sweep
├── L19/         # k=9.8 sweep
├── L24/         # k=14.0 sweep
├── sel_random/  # Selection: random
├── sel_fps/     # Selection: geometric FPS
└── sel_qr/      # Selection: feature-QR
```

Each folder contains `svd_diag.csv` with columns:
- `k`, `M` (images), `N` (basis), `rank`, `nullity`
- `sigma_min`, `sigma_max`, `tau`, `condition_number`
- `gram_smallest_20`, `gram_largest_5`
- `rho_min`, `rho_max`, `adaptive_expansions`

---

## Conclusion

**Hypothesis A (rho-starvation) is the dominant failure mode.** The paper's fixed rho cutoff is too restrictive at high k, leaving no viable sampling points. The `--rho-window-mode adaptive_rank` flag, which expands rho until the matrix condition is healthy, completely resolves this for L ≤ 19.

At L=24 (k=14), conditioning degrades even with adaptive expansion, suggesting that either more constraints (larger n_points) or higher numerical precision will be needed for HPC runs targeting k > 12.

Function-space diversity (FPS, QR selection) provides marginal improvements but is not the primary bottleneck.
