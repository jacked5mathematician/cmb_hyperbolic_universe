# Constraint Matrix Verification Summary

## Executive Summary

**Key Finding**: The constraint matrix construction in `utils/sys_generation.py` **correctly implements** the paper's algorithm. The issue of smooth chi-squared spectra without sharp minima is **NOT** due to incorrect constraint formulation.

## Detailed Analysis

### Paper's Constraint Formulation (eigenvalueprob.pdf, Section II)

The paper describes the following algorithm:

1. **Sample d base points** p₁, p₂, ..., pₐ inside the Dirichlet domain
2. **Generate ghost images** using face-pairing generators gₐ ∈ Γ
3. For each base point pⱼ, find all nⱼ images within distance ρ_max
4. **Build constraint matrix** where each row represents:
   ```
   Q_kℓm(gₐpⱼ) - Q_kℓm(gᵦpⱼ) = 0  (for α ≠ β)
   ```
5. **Matrix dimensions**:
   - M rows = Σⱼ nⱼ(nⱼ-1)/2 (all pairwise differences)
   - N columns = (L+1)² (number of spherical harmonic basis functions)

### Current Implementation Match

The code in `utils/sys_generation.py` exactly implements this:

```python
def generate_matrix_system(points_images, L, k_value):
    # For each base point's images
    for images in points_images:
        n_j = len(images)
        # Compute Q_kℓm for all images
        Q_matrix = Q_k_lm_vectorized(k_value, lm_pairs, images_array)
        
        # Generate ALL pairwise differences
        idx_i, idx_j = np.triu_indices(n_j, k=1)
        A_block = Q_matrix[idx_i, :] - Q_matrix[idx_j, :]
```

**Verification**: Row count formula matches: M = Σ nⱼ(nⱼ-1)/2 ✅

## Root Causes of Smooth Spectra

Since constraint construction is correct, the smooth spectra must be caused by other factors:

### 1. K-Sampling Resolution

**Issue**: Paper's Figure 1 shows very fine structure with many narrow minima. This requires high k-sampling density.

**Evidence**: 
- Paper eigenvalues are closely spaced (e.g., k=8.34, 8.63, 8.81, 8.93, 9.06 for m188)
- Detecting minima separated by Δk ≈ 0.3 requires sampling much finer than this

**Recommendation**: Use num_k ≥ 200 for k ∈ [1,10] (Δk ≤ 0.045)

### 2. Ghost Enumeration Quality

**Issue**: Need sufficient ghost images per base point to construct well-conditioned constraints.

**Evidence from paper**:
- Paper mentions "at least 10 images of each point"
- Uses equation (2.9) to set ρ_max: X_k^L(ρ) sinh(ρ) = 0.25

**Current implementation**:
- Has min_images=10 parameter ✅
- Uses similar radial envelope for cutoff ✅
- But may be dropping points if insufficient images found

**Recommendation**: 
- Monitor `kept_points` diagnostic per k
- Ensure kept_points > 10 and ideally ≥ n_points * 0.8
- Check `images_per_point` diagnostic (should be > 10)

### 3. Rho Cutoff Fallback Behavior

**Issue**: Fallback rho cutoffs may be too conservative, reducing the number of valid ghost images.

**Evidence**:
- Code has `fallback_used` tracking
- Fallback uses heuristic: rho_max = arcsinh(1/threshold) + scale*L
- If fallback is used frequently, may not match paper's prescription

**Current diagnostics**: Already tracking `fallback_used`, `rho_min`, `rho_max` per k ✅

**Recommendation**:
- Check fallback_used fraction across k-scan
- If > 0.5, investigate cutoff threshold and rho_start parameters
- Consider adjusting threshold or step size in compute_rho_cutoffs

### 4. Base Point Quality

**Issue**: Base points must truly lie in Dirichlet domain for manifold topology to emerge.

**Evidence**:
- Code has `self_check` mode with Dirichlet domain verification
- Uses group element checking: d(x, p₀) ≤ d(x, γ(p₀))

**Current implementation**: 
- Has Dirichlet domain checker ✅
- Falls back to Poincaré ball sampling if no generators

**Recommendation**:
- Always use `--require-snappy` for real manifolds
- Run with `--self-check` to verify base points pass Dirichlet test
- Monitor sampling_meta['fallback_used']

### 5. Numerical Conditioning

**Issue**: Matrix scaling and condition number affect chi² sensitivity to eigenvalues.

**Evidence**:
- Paper uses raw chi² = ||A·a||² (no normalization)
- Large condition numbers can mask eigenvalue structure

**Current diagnostics**: Now tracking condition number (σ_max/σ_min), numerical rank ✅

**Recommendation**:
- Use `--chi2-definition raw_residual` or `ratio` (not per_row)
- Monitor condition numbers via diagnostics
- Use debug_singular_spectrum.py to visualize singular value behavior

## Enhanced Diagnostics Added

### Per-k Diagnostics (already in spectrum.npz)
- `sigma_min`, `sigma_max`: Range of singular values
- `A_frobenius`, `A_max_abs`, `A_min_nonzero`: Matrix scale info
- `kept_points`: Number of base points contributing
- `images_per_point`: Average ghost images per point
- `fallback_used`: Whether fallback cutoffs were used

### New Diagnostics (from this PR)
- `first_10_singular_values`: First 10 singular values per k
- `numerical_rank`: Estimated rank with tolerance 1e-10
- `rank_tolerance`: Tolerance used for rank estimation

### Validation Tools

1. **scripts/debug_singular_spectrum.py**
   - Analyzes singular value spectrum
   - Compares detected minima to paper eigenvalues
   - Checks for pathological flatness
   - Creates diagnostic plots

2. **scripts/paper_sanity_run.py**
   - Runs m188(-1,1) with paper-faithful parameters
   - Validates eigenvalue detection against Table I
   - Reports success rate and false positives

3. **tests/test_eigenvalue_sensitivity.py**
   - Tests SVD detects rank drops in synthetic matrices
   - Verifies chi² minima at parameter values
   - Validates multiple solution ordering

## Recommended Validation Workflow

### Step 1: Quick Diagnostic Run

```bash
python main.py \
    --manifold "m188(-1,1)" \
    --k-min 4.0 --k-max 5.5 --num-k 50 \
    --n-points 30 \
    --word-depth 4 \
    --chi2-mode paper \
    --chi2-definition raw_residual \
    --require-snappy \
    --self-check \
    --benchmark \
    --output-dir output_values_local/diagnostic_run
```

Check diagnostics:
```bash
python scripts/debug_singular_spectrum.py \
    --spectrum output_values_local/diagnostic_run/spectrum.npz \
    --manifold m188
```

Look for:
- [ ] kept_points ≥ 20 (most base points retained)
- [ ] images_per_point ≥ 15 (sufficient ghosts)
- [ ] fallback_used = False or < 0.3 (cutoffs working)
- [ ] Chi² shows variation > 3 orders of magnitude
- [ ] Clear minima near expected eigenvalues (k≈4.41, 4.64, 5.12, 5.40)

### Step 2: Full Paper Sanity Run

```bash
python scripts/paper_sanity_run.py \
    --manifold "m188(-1,1)" \
    --k-min 1.0 --k-max 10.0 --num-k 200 \
    --n-points 50 \
    --output-dir output_values_local/paper_sanity
```

This will:
- Run full k-scan with paper parameters
- Validate against all 18 eigenvalues from Table I
- Report success rate
- Generate diagnostic plots

Expected output:
```
✓ PASS: Spectrum shows eigenvalue-selective behavior
  Success rate: >80%
  False minima: <5
```

### Step 3: If Validation Fails

If Step 2 shows FAIL or PARTIAL:

1. **Check kept_points**: If low, increase word_depth or n_points
2. **Check fallback_used**: If high, investigate cutoff computation
3. **Check images_per_point**: If < 10, increase word_depth
4. **Check condition numbers**: If > 1e12, may have numerical issues
5. **Check k-sampling**: Increase num_k for finer resolution

## Conclusion

The constraint matrix construction is **mathematically faithful to the paper**. No changes to `utils/sys_generation.py` are needed for constraint formulation.

To achieve paper-quality chi² spectra with sharp eigenvalue minima:

1. Use **fine k-sampling** (num_k ≥ 200 for k ∈ [1,10])
2. Ensure **sufficient ghost images** (images_per_point > 10)
3. Verify **base points are valid** (use --require-snappy, --self-check)
4. Monitor **fallback behavior** (fallback_used < 0.5)
5. Use **paper-faithful chi² definition** (--chi2-mode paper, --chi2-definition raw_residual)

The new diagnostic tools provide visibility into each of these factors and enable systematic debugging of spectrum quality issues.
