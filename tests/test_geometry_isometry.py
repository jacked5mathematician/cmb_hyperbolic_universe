"""
Test that isometries preserve hyperbolic distance.

Validates Invariant A: For any points x,y in H³ and group element g,
    dist(x,y) ≈ dist(g(x), g(y))

This is fundamental to Cornish & Spergel (1999) eigenmode reconstruction.
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


def sample_random_poincare_points(n: int, seed: int, radius: float = 0.7) -> np.ndarray:
    """Sample random points in Poincaré ball with deterministic seed."""
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < n:
        candidate = rng.uniform(-radius, radius, size=3)
        if np.linalg.norm(candidate) < radius:
            points.append(candidate)
    return np.array(points)


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_isometry_preserves_distance_single_generator(seed):
    """Test that a single generator preserves distance between two random points."""
    # Get real generators for m003(-2,3)
    group_elements, fallback = get_group_elements("m003(-2,3)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available or no generators loaded")
    
    # Pick first non-identity generator
    generator = None
    identity = np.eye(4)
    for elem in group_elements:
        if not np.allclose(elem, identity, atol=1e-10):
            generator = elem
            break
    
    if generator is None:
        pytest.skip("No non-identity generator found")
    
    # Sample two random points
    points = sample_random_poincare_points(2, seed)
    x, y = points[0], points[1]
    
    # Compute original distance
    d_xy = poincare_distance(x, y)
    
    # Apply group action
    gx = apply_group_action_full(generator, x)
    gy = apply_group_action_full(generator, y)
    
    # Compute transformed distance
    d_gx_gy = poincare_distance(gx, gy)
    
    # Check isometry property
    error = abs(d_xy - d_gx_gy)
    assert error < 1e-10, (
        f"Isometry violated: dist(x,y)={d_xy:.15f}, "
        f"dist(g(x),g(y))={d_gx_gy:.15f}, error={error:.2e}"
    )


@pytest.mark.parametrize("manifold_name", ["m003(-2,3)", "m004(-2,3)"])
def test_isometry_preserves_distance_multiple_generators(manifold_name):
    """Test isometry property for multiple generators and point pairs."""
    group_elements, fallback = get_group_elements(manifold_name, max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip(f"SnapPy not available for {manifold_name}")
    
    # Get generators (exclude identity)
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if not generators:
        pytest.skip(f"No non-identity generators for {manifold_name}")
    
    # Test with multiple point pairs and generators
    points = sample_random_poincare_points(6, seed=42, radius=0.6)
    
    max_error = 0.0
    test_count = 0
    
    for i in range(0, len(points) - 1, 2):
        x, y = points[i], points[i + 1]
        d_xy = poincare_distance(x, y)
        
        for g in generators[:3]:  # Test first 3 generators
            gx = apply_group_action_full(g, x)
            gy = apply_group_action_full(g, y)
            d_gx_gy = poincare_distance(gx, gy)
            
            error = abs(d_xy - d_gx_gy)
            max_error = max(max_error, error)
            test_count += 1
            
            assert error < 1e-10, (
                f"Isometry violated for {manifold_name}: "
                f"dist(x,y)={d_xy:.15f}, dist(g(x),g(y))={d_gx_gy:.15f}, "
                f"error={error:.2e}"
            )
    
    print(f"\n{manifold_name}: Tested {test_count} cases, max error={max_error:.2e}")


def test_identity_preserves_distance_trivially():
    """Sanity check: identity element trivially preserves distance."""
    identity = np.eye(4)
    points = sample_random_poincare_points(4, seed=789)
    
    for i in range(len(points) - 1):
        x, y = points[i], points[i + 1]
        d_xy = poincare_distance(x, y)
        
        # Apply identity
        gx = apply_group_action_full(identity, x)
        gy = apply_group_action_full(identity, y)
        d_gx_gy = poincare_distance(gx, gy)
        
        # Should be exact (or numerical noise only)
        assert np.allclose(x, gx, atol=1e-14)
        assert np.allclose(y, gy, atol=1e-14)
        assert abs(d_xy - d_gx_gy) < 1e-14


def test_isometry_at_origin():
    """Test isometry for points including the origin."""
    group_elements, fallback = get_group_elements("m003(-2,3)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available")
    
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if not generators:
        pytest.skip("No generators found")
    
    origin = np.array([0.0, 0.0, 0.0])
    test_point = np.array([0.3, 0.2, -0.1])
    
    d_orig = poincare_distance(origin, test_point)
    
    g = generators[0]
    g_origin = apply_group_action_full(g, origin)
    g_test = apply_group_action_full(g, test_point)
    
    d_transformed = poincare_distance(g_origin, g_test)
    
    error = abs(d_orig - d_transformed)
    assert error < 1e-10, f"Isometry failed at origin: error={error:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
