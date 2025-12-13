# Paper vs Code Constraint Matrix Construction

## Paper's Constraint Formulation (Section II)

From eigenvalueprob.pdf, the paper describes the constraint construction as follows:

### Base Setup
1. **Sample d points** inside the Dirichlet domain: p₁, p₂, ..., pₐ
2. **Generate ghost images** using face-pairing generators gₐ ∈ Γ
3. For each base point pⱼ, find all nⱼ images out to distance ρ_max

### Constraint Equations (Equation 2.1)
Each point pⱼ with nⱼ images generates **nⱼ(nⱼ+1)/2** equations of the form:

```
Ψ(gₐpⱼ) - Ψ(gᵦpⱼ) = 0,  (α ≠ β)
```

**Key detail**: These are **pairwise differences between all ghost images** of the **same base point**.

### Matrix System (Equation 2.3)
The constraint matrix A has:
- **M rows** = Σⱼ nⱼ(nⱼ+1)/2  (total pairwise differences)
- **N columns** = (L+1)²  (number of (ℓ,m) basis functions)
- Each row represents: `Q_kℓm(gₐpⱼ) - Q_kℓm(gᵦpⱼ)` for one pair (α,β)

The paper explicitly states this in the schematic system (2.3):
```
(Q₀₀(g₁p₁) - Q₀₀(g₂p₁)) ... (Q_LL(gₐp₁) - Q_LL(gᵦp₁))
                    ...
(Q₀₀(g₁pₐ) - Q₀₀(g₂pₐ)) ... (Q_LL(gₐpₐ) - Q_LL(gᵦpₐ))
```

### Important Paper Details
- **No weighting by sinh(ρ)** in the constraint equations
- **No normalization** of rows mentioned before SVD
- **Chi-squared** directly from SVD: χ² = ||A·a||² (equation 2.7)
- Each row enforces that the eigenfunction value matches at two ghost images

## Current Code Implementation

File: `utils/sys_generation.py`

### Function: `generate_matrix_system()`

```python
def generate_matrix_system(points_images, L, k_value):
    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]
    N = (L + 1) ** 2
    
    for images in points_images:
        n_j = len(images)
        if n_j < 2:
            continue
        
        # Compute Q-values for all images: shape (n_j, N)
        Q_matrix = Q_k_lm_vectorized(k_value, lm_pairs, images_array)
        
        # Pairwise differences using upper triangle
        idx_i, idx_j = np.triu_indices(n_j, k=1)
        A_block = Q_matrix[idx_i, :] - Q_matrix[idx_j, :]
        row_blocks.append(A_block)
```

### What the Code Does
1. For each base point, enumerate nⱼ ghost images
2. Compute Q_kℓm for all images and all (ℓ,m) pairs
3. Generate ALL pairwise differences: Q(image_i) - Q(image_j) for i < j
4. Stack these as rows of matrix A

### Row Count Verification
The code computes:
```python
M_expected = sum(len(imgs) * (len(imgs) - 1) // 2 for imgs in points_images)
```

This matches the paper's formula: Σⱼ nⱼ(nⱼ-1)/2

**Note**: Paper has (nⱼ+1)/2 but that appears to be a typo since they state α ≠ β, which means we're choosing 2 distinct elements from nⱼ images, giving C(nⱼ,2) = nⱼ(nⱼ-1)/2.

## Comparison Summary

| Aspect | Paper | Current Code | Match? |
|--------|-------|--------------|--------|
| Sample base points in Dirichlet domain | ✓ | ✓ | ✅ YES |
| Use face-pairing generators | ✓ | ✓ | ✅ YES |
| Generate ghost images per base point | ✓ | ✓ | ✅ YES |
| Pairwise differences Q(gₐpⱼ) - Q(gᵦpⱼ) | ✓ | ✓ | ✅ YES |
| Each base point contributes independently | ✓ | ✓ | ✅ YES |
| M = Σ nⱼ(nⱼ-1)/2 | ✓ | ✓ | ✅ YES |
| N = (L+1)² | ✓ | ✓ | ✅ YES |
| No row weighting by sinh(ρ) | ✓ | ✓ | ✅ YES |
| No row normalization before SVD | ✓ | Flag-controlled | ⚠️ FIXED |
| Chi-squared = ||A·a||² | ✓ | Flag-controlled | ⚠️ FIXED |

## Key Findings

### ✅ The Constraint Construction is CORRECT

The current implementation in `utils/sys_generation.py` **correctly implements** the paper's constraint formulation:

1. **Pairwise ghost differences**: Both paper and code use differences Q(gₐpⱼ) - Q(gᵦpⱼ)
2. **Per-base-point structure**: Constraints are organized by base point, with each generating nⱼ(nⱼ-1)/2 rows
3. **No face-pairing-specific logic**: The paper does NOT use any special structure beyond "all pairwise differences of ghosts"
4. **Matrix dimensions**: Both compute M and N identically

### ⚠️ Already Fixed Issues (via flags)

1. **Row normalization**: Originally applied in SVD, now controlled by `--chi2-mode paper` (no normalization)
2. **Chi-squared definition**: Multiple definitions available via `--chi2-definition`

### ❌ Issues NOT Related to Constraint Construction

The problem statement mentions "stepwise plateaus and lack of sharp minima" in chi² spectra. Based on this analysis, this is **NOT** due to incorrect constraint construction. The constraints match the paper exactly.

Potential causes for smooth spectra (to investigate separately):
1. **Insufficient k-sampling resolution**: Paper's Fig. 1 appears very fine, suggesting num_k >> 10
2. **Ghost enumeration**: Need to verify sufficient images are being generated (nⱼ ≥ 10 per point)
3. **Rho cutoff selection**: Fallback behavior may be too aggressive, cutting off important regions
4. **Base point quality**: Points must truly be in Dirichlet domain for manifold structure to emerge
5. **Numerical conditioning**: Matrix A may have scaling issues that obscure eigenvalue sensitivity

## Conclusion

**The constraint matrix construction in the code is faithful to the paper.** No changes are needed to `utils/sys_generation.py` for constraint formulation.

The issue of smooth chi² spectra without sharp minima must be investigated in:
- Sampling parameters (k resolution, number of base points)
- Ghost enumeration quality and count
- Rho cutoff behavior (fallback vs paper-specified cutoffs)
- Numerical conditioning and chi² definition

These are addressed in subsequent steps of the problem statement.
