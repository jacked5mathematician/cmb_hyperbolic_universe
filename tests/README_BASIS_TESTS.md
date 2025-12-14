# Basis Function Tests

Validates the mathematical correctness of basis functions X_k^ℓ(ρ,θ,φ) = Φ_ν_ℓ(ρ) Y_ℓm(θ,φ) used in Cornish & Spergel (1999) eigenmode reconstruction.

## Quick Start

```bash
# Run all basis function tests
pytest tests/test_basis_functions.py -v

# Run specific test categories
pytest tests/test_basis_functions.py::test_radial_ode_residual -v
pytest tests/test_basis_functions.py::test_asymptotic_behavior -v
pytest tests/test_basis_functions.py::test_spherical_harmonics_laplacian -v
```

## Test Categories

### 1. Radial ODE Residual (5 tests)
Validates that Φ_ν_ℓ(ρ) satisfies the radial Helmholtz ODE:
```
Φ'' + 2 coth(ρ) Φ' - ℓ(ℓ+1)/sinh²(ρ) Φ + (k²+1) Φ = 0
```

**Method**: Finite differences with h=1e-5  
**Tolerance**: Residual < 5e-6  
**Status**: ✅ All passing

### 2. Asymptotic Behavior (5 tests)
Validates correct asymptotic form for large ρ:
```
Φ(ρ) sinh(ρ) ≈ A cos(kρ + φ)
```

**Tests**:
- Envelope bounded (< 10)
- No exponential growth (< 2x over Δρ=10)
- Oscillatory with period ~π/k

**Status**: ✅ All passing

### 3. Spherical Harmonics (8 tests)
Validates Y_ℓm(θ,φ) satisfies Laplacian eigenvalue equation:
```
Δ_S² Y_ℓm = -ℓ(ℓ+1) Y_ℓm
```

**Method**: Finite-difference Laplacian with h=1e-5  
**Tolerance**: Error < 3e-6  
**Status**: ✅ All passing

## Test Parameters

| Test | (k, ℓ) or (ℓ, m) | Notes |
|------|-------------------|-------|
| Radial ODE | (1,0), (2,0), (1,1), (2,1), (3,2) | Various wavenumbers |
| Asymptotic | (1,0), (2,0), (3,0), (1,1), (2,1) | Large ρ ∈ [10,20] |
| Harmonics | (0,0), (1,±1), (1,0), (2,0), (2,1) | Various ℓ, m |

## Key Results

### Radial ODE Residuals
```
(k=1, ℓ=0): 1.88e-06  ✅
(k=2, ℓ=0): 2.60e-06  ✅
(k=1, ℓ=1): 1.23e-06  ✅
(k=2, ℓ=1): 1.62e-06  ✅
(k=3, ℓ=2): 5.74e-07  ✅
```

### Asymptotic Envelopes
```
(k=1, ℓ=0): max=0.223, crossings=3  ✅
(k=2, ℓ=0): max=0.223, crossings=6  ✅
(k=3, ℓ=0): max=0.236, crossings=10 ✅
(k=1, ℓ=1): max=0.707, crossings=4  ✅
(k=2, ℓ=1): max=0.707, crossings=6  ✅
```

### Spherical Harmonic Laplacian Errors
```
(ℓ=0, m=0):  < 1e-6  ✅
(ℓ=1, m=-1): 1.23e-06 ✅
(ℓ=1, m=0):  1.08e-06 ✅
(ℓ=1, m=1):  1.23e-06 ✅
(ℓ=2, m=0):  < 1e-6  ✅
(ℓ=2, m=1):  2.53e-06 ✅
```

## Implementation Details

### Functions Tested
- `Phi_nu_l(nu, l, chi)` - Radial functions (utils/special_functions.py:51)
- `Y_lm_real(l, m, theta, phi)` - Real spherical harmonics (utils/special_functions.py:122)

### Numerical Precision
- Uses mpmath for 50-digit precision Legendre functions
- Test conversions: `float(mpf_value)` for comparisons
- Finite differences with h=1e-5

### Known Issues
- **FIXED**: Coordinate bug in utils/transformations.py:103 (arccos → atan2)
- scipy comparison skipped: Custom Y_ℓm normalization differs from scipy.special.sph_harm

## Related Tests

Run full validation suite:
```bash
# All geometry + basis function tests
pytest tests/test_geometry_isometry.py \
       tests/test_group_composition.py \
       tests/test_distance_sanity.py \
       tests/test_lorentz_structure.py \
       tests/test_tiling_enumeration.py \
       tests/test_coordinate_roundtrip.py \
       tests/test_basis_functions.py -v

# 113 tests passing
```

## References

- Cornish & Spergel (1999) - "Testing the Copernican Principle"
- utils/special_functions.py - Basis function implementations
- docs/BASIS_FUNCTION_VALIDATION.md - Detailed validation report
