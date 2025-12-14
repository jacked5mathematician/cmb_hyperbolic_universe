# Cutoff and Sampling Inventory (Step 1)

**Date**: 2025-12-14  
**Purpose**: Document code paths for radial cutoffs, point sampling, and image enumeration

---

## 1. Radial Functions

### Implementation
**File**: `utils/special_functions.py`  
**Function**: `Phi_nu_l(nu, l, chi)` (lines 51-90)

**Details**:
- Uses mpmath Legendre functions P^{-1/2-ℓ}_{-1/2+iν}(cosh(χ))
- 50-digit precision for high accuracy
- Returns mpmath.mpf objects
- Normalization: N_ν_ℓ = ∏_{n=1}^ℓ (ν² + n²)

**Formula**: 
```
Φ_ν_ℓ(χ) = √[π N_ν_ℓ / (2 sinh(χ))] · Re[P^{-1/2-ℓ}_{-1/2+iν}(cosh(χ))]
```

---

## 2. ρ_max Computation (Root-Finding)

### Implementation
**File**: `utils/cutoffs.py`  
**Function**: `compute_rho_cutoffs(k, L, l_min, threshold=0.25, ...)` (lines 77-148)

### Root Finding Method
**Uses paper-inspired envelope approximation** (not actual X_k^L evaluation):

```python
def _abs_radial_envelope(k, ell, rho):
    rho0 = arcsinh(sqrt(ell*(ell+1)) / k)  # Paper eq (2.8) turning point
    if rho < rho0:
        return inf
    phase = k * (rho - rho0)
    return abs(cos(phase))
```

**Root condition**: Find first ρ where `|X_k^L(ρ) sinh(ρ)| ≤ 0.25`  
**Implementation**: `envelope(k, L, ρ) ≤ 0.25` (approximation!)

### Search Algorithm
**File**: `utils/cutoffs.py`, `_find_crossing()` (lines 56-74)

```python
def _find_crossing(k, ell, threshold, rho_cap, step, rho_start):
    rho = rho_start  # Default: 0.75
    while rho <= rho_cap:  # Default: 120.0
        val = _abs_radial_envelope(k, ell, rho)
        if val <= threshold:
            return rho
        rho += step  # Default: 0.05
    return None
```

**Method**: Linear scan with fixed step size (not Brent!)  
**No bracketing or sign-change detection**  
**No bisection or Newton refinement**

### Parameters
- `rho_start`: 0.75 (avoid pathological near-zero crossings)
- `step`: 0.05 (coarse scan)
- `rho_cap`: 120.0 (upper limit)
- `rho_max_floor`: 1.0 (minimum acceptable ρ_max)
- `threshold`: 0.25 (paper value)

### Fallback Logic
If `rho_max` not found or < 1.0:
```python
rho_max = max(arcsinh(1.0 / 0.25), arcsinh(1.0 / 0.25) + 0.5 * L, 1.0)
```

**Fallback triggered** → sets `fallback_used = True`

---

## 3. L and ℓ_min Selection

### Implementation
**File**: `main.py` (lines 37-48)

```python
PAPER_L_MIN = 5

def _paper_L(k: float) -> int:
    return int(np.floor(k)) + 10

def _paper_c(k: float) -> int:
    return int(np.floor(100.0 / k)) + 10
```

**Rule**:
- **L** = floor(k) + 10
- **ℓ_min** = 5 (constant)
- **c** = floor(100/k) + 10 (# of points target)

**Examples**:
| k | L | ℓ_min | c |
|---|---|-------|---|
| 1.0 | 11 | 5 | 110 |
| 2.0 | 12 | 5 | 60 |
| 5.0 | 15 | 5 | 30 |
| 10.0 | 20 | 5 | 20 |
| 20.0 | 30 | 5 | 15 |

---

## 4. Point Sampling (Dirichlet Domain)

### Implementation
**File**: `utils/points.py`  
**Function**: `sample_points_in_dirichlet_domain()` (lines 48-125)

### Method
**Dirichlet domain criterion**: For point x, check
```
d(x, p₀) ≤ d(x, γ(p₀))  for all γ in word set
```

where p₀ = origin.

### Algorithm
1. **Load group elements** via `get_group_elements(manifold, word_depth=3)`
2. **Compute images** γ(p₀) for all words up to depth 3
3. **Sample candidates** uniformly in Poincaré ball (radius 0.85)
4. **Test criterion** for each candidate
5. **Accept/reject** based on distance comparison

### Parameters
- `word_depth`: 3 (default)
- `fallback_radius`: 0.85
- `tolerance`: 1e-6 (margin for d(x,p₀) ≤ d(x,γp₀) + tol)
- `MAX_ATTEMPT_MULTIPLIER`: 50 (try 50n samples before fallback)

### Fallback
If insufficient points accepted after 50n attempts:
```python
# Fall back to uniform sampling in ball
points = rng.uniform(-0.85, 0.85, size=(n, 3))
# Keep only ||x|| < 0.85
```

---

## 5. Image Enumeration (Ghost Images)

### Implementation
**File**: `utils/ghosts.py`  
**Function**: `enumerate_ghost_images()` (lines 90-158)

### Method
For each base point p_j:
1. **Load group elements** γ (words up to `max_word_length=6`)
2. **Apply transformations** γ(p_j) in hyperboloid coordinates
3. **Convert to pseudo-spherical** (ρ, θ, φ)
4. **Filter by radius**: Keep if `rho_min ≤ ρ ≤ rho_max + rho_margin`
5. **Deduplicate** images (round to 8 decimals)
6. **Cap at max_images** (default 200)

### Parameters
- `rho_min`, `rho_max`: From `compute_rho_cutoffs()`
- `rho_margin`: 0.5 (extra tolerance for collection)
- `min_images`: 10 (minimum per point, else drop point)
- `max_word_length`: 6 (BFS word depth)
- `tolerance`: 1e-8 (deduplication precision)
- `max_images`: 200 (cap per point)

### Fallback (Synthetic Images)
If no group elements available:
```python
def _synthetic_images(base_point, rho_min, rho_max, count):
    rng = np.random.default_rng(hash(base_point))
    rhos = rng.uniform(rho_min, rho_max, count)
    thetas = rng.uniform(0, π, count)
    phis = rng.uniform(-π, π, count)
    return [(ρ, θ, φ)]
```

**Synthetic count**: min(200, max(10, 12))

---

## 6. Image Selection in Pipeline

### Implementation
**File**: `main.py` (lines 260-291)

### Selection Logic
```python
points_images, ghost_meta = enumerate_ghost_images(
    manifold_name, base_points_pseudo, 
    rho_min, rho_max, 
    max_word_length=6, max_images=200
)

selected_points = []
for idx, imgs in enumerate(points_images):
    capped_imgs = imgs[:max_images_per_point]  # Cap at 200
    rows = len(capped_imgs) * (len(capped_imgs) - 1) // 2
    
    if rows < M_target or len(selected_points) < required_min_points:
        selected_points.append(capped_imgs)
    
    if rows >= M_target and len(selected_points) >= required_min_points:
        break
```

**Stopping criteria**:
1. Accumulated rows ≥ M_target
2. At least `required_min_points` points selected

**M_target** = c(c-1)/2, where c = floor(100/k) + 10

---

## 7. Key Assumptions and Issues

### Critical Assumptions
1. **Envelope approximation**: Uses `cos(k(ρ-ρ₀))` instead of actual `Φ_ν_ℓ(ρ)`
   - ❌ **No validation that envelope matches actual radial function**
   - ❌ **No residual check after root finding**

2. **Linear scan**: Step size 0.05, no refinement
   - ❌ **No bracketing verification**
   - ❌ **Resolution limited to 0.05 in ρ**

3. **"First root"**: Takes first crossing above ρ_start=0.75
   - ❌ **No deterministic "which root" rule for oscillatory function**
   - ❌ **Different roots possible depending on step size**

4. **Fallback threshold**: ρ_max < 1.0 triggers fallback
   - ❌ **No justification for threshold=1.0**
   - ❌ **Fallback uses heuristic arcsinh(4) + 0.5L**

### Potential Failure Modes
1. **Envelope != actual X_k^L**: Root may not satisfy paper condition
2. **Coarse step**: May miss actual root, overshoot by 0.05
3. **Oscillations**: Multiple roots exist, no stability across k
4. **Near-boundary images**: If most images cluster near ρ_max, constraints are noise-dominated

---

## 8. File Summary

| File | Functions | Purpose |
|------|-----------|---------|
| `utils/cutoffs.py` | `compute_rho_cutoffs()`, `_abs_radial_envelope()`, `_find_crossing()` | Compute ρ_min, ρ_max via envelope approximation |
| `utils/special_functions.py` | `Phi_nu_l()` | Actual radial functions (not used in cutoff!) |
| `utils/points.py` | `sample_points_in_dirichlet_domain()` | Sample base points in domain |
| `utils/ghosts.py` | `enumerate_ghost_images()`, `get_group_elements()` | Enumerate γ(p) images |
| `main.py` | `_paper_L()`, `_paper_c()`, image selection loop | Pipeline control |

---

## 9. Action Items for Steps 2-6

### Step 2: Root Validation Tests
- ✅ Compare envelope to actual `Phi_nu_l(nu, L, rho) * sinh(rho)`
- ✅ Verify residual |f(ρ_max)| ≤ 1e-10
- ✅ Implement deterministic sign-change scan reference
- ✅ Test continuity across k grid

### Step 3: Cutoff Effectiveness
- ✅ Evaluate actual radial function near ρ_max
- ✅ Check image distribution (how many near cutoff?)

### Step 4: Sampling Quality
- ✅ Pairwise distance distribution
- ✅ Boundary proximity check
- ✅ Conditioning proxy

### Step 5: Harden Fallback
- ✅ Add ALLOW_FALLBACK flag
- ✅ RuntimeError if fallback when disabled
- ✅ Export SnapPy data to offline cache

### Step 6: Robust Alternatives
- ✅ method="paper_first_root" (current)
- ✅ method="tracked_root" (continuity across k)
- ✅ method="magnitude_threshold" (direct |X| check)
- ✅ method="fixed_rho" (diagnostic mode)

---

## Summary

**Current implementation**:
- Uses **envelope approximation** instead of actual radial functions
- **Linear scan** with 0.05 step, no refinement
- **No validation** of root condition after finding
- **Fallback** when ρ_max < 1.0, using heuristic formula

**Risk**: Envelope may not match actual X_k^L behavior → wrong cutoff → missing χ² dips
