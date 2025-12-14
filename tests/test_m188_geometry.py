"""
Quick test of m188(-1,1) manifold geometry to validate it works with our invariant tests.
"""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.ghosts import get_group_elements
from utils.transformations import (
    apply_so31_action,
    poincare_distance,
    project_to_klein,
    klein_to_poincare,
)


def apply_group_action_full(matrix: np.ndarray, poincare_pt: np.ndarray) -> np.ndarray:
    """Apply SO(3,1) matrix to a Poincaré ball point, return result in Poincaré."""
    hyperboloid = apply_so31_action(matrix, poincare_pt)
    klein = project_to_klein(hyperboloid)
    return klein_to_poincare([klein])[0]


def test_m188_loads():
    """Test that m188(-1,1) manifold loads successfully."""
    group_elements, fallback = get_group_elements("m188(-1,1)", max_depth=1)
    
    if fallback:
        pytest.skip("SnapPy not available for m188(-1,1)")
    
    assert len(group_elements) > 0, "No group elements loaded for m188(-1,1)"
    print(f"\nm188(-1,1): Loaded {len(group_elements)} group elements at depth 1")


def test_m188_isometry():
    """Test isometry preservation for m188(-1,1)."""
    group_elements, fallback = get_group_elements("m188(-1,1)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available for m188(-1,1)")
    
    # Get first non-identity generator
    identity = np.eye(4)
    generator = None
    for elem in group_elements:
        if not np.allclose(elem, identity, atol=1e-10):
            generator = elem
            break
    
    if generator is None:
        pytest.skip("No non-identity generator found for m188(-1,1)")
    
    # Test with a few random points
    rng = np.random.default_rng(42)
    max_error = 0.0
    
    for _ in range(5):
        # Generate two random points
        pts = []
        while len(pts) < 2:
            candidate = rng.uniform(-0.6, 0.6, size=3)
            if np.linalg.norm(candidate) < 0.6:
                pts.append(candidate)
        
        x, y = pts[0], pts[1]
        
        # Original distance
        d_xy = poincare_distance(x, y)
        
        # Transformed distance
        gx = apply_group_action_full(generator, x)
        gy = apply_group_action_full(generator, y)
        d_gx_gy = poincare_distance(gx, gy)
        
        error = abs(d_xy - d_gx_gy)
        max_error = max(max_error, error)
        
        assert error < 1e-10, f"m188(-1,1) isometry violated: error={error:.2e}"
    
    print(f"m188(-1,1): Tested 5 point pairs, max isometry error={max_error:.2e}")


def test_m188_composition():
    """Test group composition for m188(-1,1)."""
    group_elements, fallback = get_group_elements("m188(-1,1)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available for m188(-1,1)")
    
    # Get two non-identity generators
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if len(generators) < 2:
        pytest.skip("Need at least 2 generators for m188(-1,1)")
    
    g, h = generators[0], generators[1]
    
    # Test composition with random points
    rng = np.random.default_rng(123)
    max_error = 0.0
    
    for _ in range(3):
        # Generate random point
        while True:
            candidate = rng.uniform(-0.5, 0.5, size=3)
            if np.linalg.norm(candidate) < 0.5:
                x = candidate
                break
        
        # Sequential application
        h_x = apply_group_action_full(h, x)
        g_h_x = apply_group_action_full(g, h_x)
        
        # Composed application
        gh = g @ h
        gh_x = apply_group_action_full(gh, x)
        
        error = np.linalg.norm(g_h_x - gh_x)
        max_error = max(max_error, error)
        
        assert error < 1e-10, f"m188(-1,1) composition violated: error={error:.2e}"
    
    print(f"m188(-1,1): Tested 3 composition cases, max error={max_error:.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
