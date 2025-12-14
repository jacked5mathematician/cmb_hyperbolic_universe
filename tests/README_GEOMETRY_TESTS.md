# Geometry Invariant Tests - Quick Reference

## Running the Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all geometry tests (RECOMMENDED)
pytest tests/test_geometry_isometry.py \
       tests/test_group_composition.py \
       tests/test_distance_sanity.py \
       tests/test_m188_geometry.py \
       tests/test_lorentz_structure.py \
       tests/test_tiling_enumeration.py -q

# Quick core tests only (1 second)
pytest tests/test_geometry_isometry.py \
       tests/test_group_composition.py \
       tests/test_distance_sanity.py \
       tests/test_m188_geometry.py -q

# With diagnostic output
pytest tests/test_lorentz_structure.py \
       tests/test_tiling_enumeration.py -v -s
```

## Test Summary

| Test File | Tests | Purpose | Time |
|-----------|-------|---------|------|
| `test_geometry_isometry.py` | 7 | Validate dist(x,y) = dist(g(x),g(y)) | ~0.2s |
| `test_group_composition.py` | 6 | Validate g(h(x)) = (g∘h)(x) | ~0.2s |
| `test_distance_sanity.py` | 20 | Validate metric axioms | ~0.4s |
| `test_m188_geometry.py` | 3 | Test m188(-1,1) specifically | ~0.2s |
| `test_lorentz_structure.py` | 16 | **Lorentz group SO(3,1)** | **~1.0s** |
| `test_tiling_enumeration.py` | 26 | **Tiling/enumeration correctness** | **~6.6s** |
| **TOTAL** | **78** | **Complete validation** | **~8.6s** |

## Expected Results

```
✅ 78 passed in 7.66s
```

All tests should pass with errors < 1e-10 (typically 1e-12 to 1e-16).

## What's Being Tested

### Core Invariants (36 tests)

#### Invariant A: Isometry Preservation
- Hyperbolic distances are preserved under group actions
- `dist(x, y) ≈ dist(g(x), g(y))` for all group elements g
- Critical for eigenmode reconstruction accuracy

#### Invariant B: Group Composition
- Sequential and composed applications agree
- `g(h(x)) ≈ (g∘h)(x)` for all group elements g, h
- Ensures group structure is correctly implemented

#### Invariant C: Metric Sanity
- Identity: `dist(x, x) = 0`
- Symmetry: `dist(x, y) = dist(y, x)`
- Triangle inequality: `dist(x, z) ≤ dist(x, y) + dist(y, z)`
- Validates distance function correctness

### Extended Tests (42 tests)

#### Lorentz Structure (16 tests)
- **G^T η G = η**: Generators preserve Minkowski metric (max error 5e-12)
- **det(G) ≈ 1**: Proper Lorentz group membership (max error 7e-14)
- **Model consistency**: Hyperboloid ↔ Poincaré distances agree (error ~1e-15)
- **Ball preservation**: Group actions keep points inside (margin > 0.1)

#### Tiling Enumeration (26 tests)
- **Determinism**: Same seed → same image count (100% stable)
- **No duplicates**: Min pairwise distance > 0.3 (no clustering)
- **Growth**: Images increase with radius (follows hyperbolic volume)
- **Deep words**: word_depth = 1,2,3,4 all work correctly
- **m188(-1,1)**: Depth 4 → 24,277 unique group elements ✅

## If Tests Fail

1. Check SnapPy installation:
   ```bash
   python -c "import snappy; print(snappy.__version__)"
   ```

2. Verify manifold loads:
   ```bash
   python -c "import snappy; m = snappy.Manifold('m188(-1,1)'); print(len(m.dirichlet_domain().pairing_matrices()))"
   ```

3. Check numerical precision:
   - Errors < 1e-10 are acceptable
   - Errors > 1e-8 indicate a problem

4. Run individual test files to isolate issues:
   ```bash
   pytest tests/test_distance_sanity.py -v  # Most basic checks
   pytest tests/test_geometry_isometry.py -v  # Group action checks
   ```

## Dependencies

- Python 3.13+
- pytest 9.0.2+
- numpy
- snappy (SnapPy)

All dependencies already installed in `.venv/`.

## Test Coverage

- **Manifolds tested:** m003(-2,3), m004(-2,3), m188(-1,1)
- **Group elements:** Up to depth 4 (tested 5,263 to 24,277 elements)
- **Points sampled:** Random Poincaré ball points with |x| < 0.7
- **Seeds:** Deterministic (42, 123, 456, 789, 999) for reproducibility
- **Total test cases:** >2500 individual geometric checks

## Files Added

```
tests/
├── test_geometry_isometry.py    # 185 lines, 7 tests - Invariant A
├── test_group_composition.py    # 241 lines, 6 tests - Invariant B
├── test_distance_sanity.py      # 233 lines, 20 tests - Invariant C
├── test_m188_geometry.py        # 143 lines, 3 tests - m188(-1,1) specific
├── test_lorentz_structure.py    # 333 lines, 16 tests - SO(3,1) structure ✨
└── test_tiling_enumeration.py   # 334 lines, 26 tests - Tiling/enumeration ✨
```

**Total:** 1,469 lines of test code, 78 tests, 0 failures

## Next Steps

After validating geometry (✅ DONE):
1. ✅ Lorentz structure validated
2. ✅ Tiling enumeration validated (word_depth up to 4)
3. ✅ Ready for basis functions and χ² construction
4. Proceed with eigenmode scanning on m188(-1,1)

---
*For detailed analysis, see `docs/extended_geometry_validation.md`*
