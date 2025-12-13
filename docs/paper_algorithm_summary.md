# Paper Algorithm Summary (from eigenvalueprob.pdf)

## Key Algorithm Parameters (from Section II, equations 2.10)

The paper specifies:
- **L = 10 + floor(k)** - Maximum spherical harmonic degree
- **ℓ_min = 5** - Minimum l for rho_min cutoff
- **c = 10 + floor(100/k)** - Oversampling ratio M/N
- **M_target = c × N** where N = (L+1)²

## Chi-squared Definition (equation 2.7)

The paper defines chi-squared as:
```
χ² = ||A·a||²
```

where A is the constraint matrix and a is the solution vector from SVD.

**Important**: The paper does NOT mention any row normalization or rescaling of the A matrix before SVD. The chi-squared values come directly from the smallest singular values squared.

## Radial Eigenfunctions (equation 1.5)

The radial functions X_ℓ^k(ρ) are defined as hyperspherical Bessel functions:
```
X_ℓ^k(ρ) = [(-1)^(ℓ+1) sinh^ℓ(ρ)] / [√(Π(n²+k²))] × d^(ℓ+1)[cos(kρ)]/d(cosh ρ)^(ℓ+1)
```

These have the asymptotic behavior (equation 2.8):
- Near origin (ρ << ρ₀): X_ℓ^k(ρ) ≈ 0
- Far from origin (ρ >> ρ₀): X_ℓ^k(ρ) ≈ cos(kρ + φ₀) / sinh(ρ)

## Cutoff Selection (Section II)

The paper describes choosing ρ_max based on the structure of X_ℓ^k(ρ):
- Start of oscillatory regime at ρ_0
- The radial functions decay as 1/sinh(ρ) for large ρ
- The cutoff should capture the oscillatory regime

The paper mentions:
- "The inner cut-off helps to keep the Q_kℓm's of similar size"
- Uses both ℓ_min for rho_min and L for rho_max

## Base Points and Images

From Section II:
- "randomly selecting a collection of d points inside the Dirichlet domain"
- "Each point p_j yields n_j images"
- Uses face-pairing generators from SnapPea

## Normalization

The paper mentions (equation 1.7):
- Eigenmodes in H³ have delta-function normalization
- Equation 2.12 shows modes in compact space Σ normalized to unity
- **Critical**: The paper says eigenmodes are determined "up to an overall normalization"

This means the chi-squared values are relative, not absolute. There's no mention of rescaling chi² values to be O(1).

## Sampling Resolution for k

From Figure 1 caption and Section III.A:
- The example shows k=1→10
- Paper mentions checking eigenvalues up to q²=100 (k≈10)
- No explicit statement of number of k samples
- Figure 1 shows smooth curves suggesting fine sampling

## Key Deviations in Current Implementation

1. **Row normalization in SVD**: Current code (utils/svd.py line 22-24) does:
   ```python
   row_norms = np.linalg.norm(A, axis=1, keepdims=True)
   A = A / row_norms
   ```
   This is NOT in the paper and changes the chi² values.

2. **L schedule**: Current uses floor(k) in some places, paper uses 10+floor(k)

3. **No explicit discussion** in paper of:
   - Quantization precision for caching
   - Exact number of base points
   - Deduplication of images

## Implementation Plan

- Remove or gate row normalization (add --chi2-mode paper|legacy)
- Ensure L = 10 + floor(k) matches paper exactly
- Ensure c = 10 + floor(100/k) matches paper exactly
- Fix any vectorized vs scalar discrepancies in special functions
- Add comprehensive diagnostics to understand chi² behavior
