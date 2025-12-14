"""
Test basic sanity properties of hyperbolic distance function.

Validates Invariant C:
- dist(x,x) = 0 (identity of indiscernibles)
- dist(x,y) = dist(y,x) (symmetry)
- dist(x,z) ≤ dist(x,y) + dist(y,z) (triangle inequality, approximate)

Critical for Cornish & Spergel (1999) eigenmode reconstruction.
"""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.transformations import poincare_distance


def sample_random_poincare_points(n: int, seed: int, radius: float = 0.7) -> np.ndarray:
    """Sample random points in Poincaré ball with deterministic seed."""
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < n:
        candidate = rng.uniform(-radius, radius, size=3)
        if np.linalg.norm(candidate) < radius:
            points.append(candidate)
    return np.array(points)


@pytest.mark.parametrize("seed", [42, 123, 456, 789, 999])
def test_distance_self_is_zero(seed):
    """Test that dist(x,x) = 0 for random points."""
    points = sample_random_poincare_points(5, seed)
    
    for x in points:
        d_xx = poincare_distance(x, x)
        assert abs(d_xx) < 1e-14, f"dist(x,x) = {d_xx:.2e}, expected 0"


def test_distance_origin_to_self():
    """Test that dist(origin, origin) = 0."""
    origin = np.array([0.0, 0.0, 0.0])
    d = poincare_distance(origin, origin)
    assert abs(d) < 1e-14, f"dist(0,0) = {d:.2e}, expected 0"


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_distance_symmetry(seed):
    """Test that dist(x,y) = dist(y,x) for random pairs."""
    points = sample_random_poincare_points(10, seed, radius=0.6)
    
    max_asymmetry = 0.0
    test_count = 0
    
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            x, y = points[i], points[j]
            
            d_xy = poincare_distance(x, y)
            d_yx = poincare_distance(y, x)
            
            asymmetry = abs(d_xy - d_yx)
            max_asymmetry = max(max_asymmetry, asymmetry)
            test_count += 1
            
            assert asymmetry < 1e-14, (
                f"Symmetry violated: dist(x,y)={d_xy:.15f}, "
                f"dist(y,x)={d_yx:.15f}, diff={asymmetry:.2e}"
            )
    
    print(f"\nTested {test_count} pairs, max asymmetry={max_asymmetry:.2e}")


def test_distance_symmetry_with_origin():
    """Test symmetry for distances involving the origin."""
    origin = np.array([0.0, 0.0, 0.0])
    points = sample_random_poincare_points(5, seed=321, radius=0.5)
    
    for x in points:
        d_ox = poincare_distance(origin, x)
        d_xo = poincare_distance(x, origin)
        
        assert abs(d_ox - d_xo) < 1e-14, (
            f"Symmetry with origin violated: "
            f"dist(0,x)={d_ox:.15f}, dist(x,0)={d_xo:.15f}"
        )


@pytest.mark.parametrize("seed", [42, 123, 789])
def test_triangle_inequality(seed):
    """Test approximate triangle inequality: dist(x,z) ≤ dist(x,y) + dist(y,z)."""
    points = sample_random_poincare_points(9, seed, radius=0.5)
    
    violations = 0
    max_violation = 0.0
    test_count = 0
    
    # Test all triples
    for i in range(len(points)):
        for j in range(len(points)):
            for k in range(len(points)):
                if i == j or j == k or i == k:
                    continue
                
                x, y, z = points[i], points[j], points[k]
                
                d_xy = poincare_distance(x, y)
                d_yz = poincare_distance(y, z)
                d_xz = poincare_distance(x, z)
                
                # Triangle inequality: d_xz ≤ d_xy + d_yz
                violation = d_xz - (d_xy + d_yz)
                test_count += 1
                
                if violation > 1e-10:  # Allow for numerical tolerance
                    violations += 1
                    max_violation = max(max_violation, violation)
                    
                    # Only assert if violation is significant
                    assert violation < 1e-8, (
                        f"Triangle inequality violated significantly:\n"
                        f"dist(x,z) = {d_xz:.15f}\n"
                        f"dist(x,y) + dist(y,z) = {d_xy:.15f} + {d_yz:.15f} = {d_xy + d_yz:.15f}\n"
                        f"violation = {violation:.2e}"
                    )
    
    print(f"\nTested {test_count} triples, violations > 1e-10: {violations}, max={max_violation:.2e}")


def test_distance_positivity():
    """Test that dist(x,y) > 0 for distinct points."""
    points = sample_random_poincare_points(5, seed=111, radius=0.6)
    
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            x, y = points[i], points[j]
            
            # Ensure points are actually distinct
            if np.allclose(x, y, atol=1e-12):
                continue
            
            d = poincare_distance(x, y)
            
            assert d > 0, f"Distance not positive for distinct points: dist={d:.2e}"


def test_distance_known_value_along_ray():
    """Test distance formula against known value for points along a ray."""
    # For points [r,0,0] and [0,0,0] in Poincaré ball:
    # dist = 2*arctanh(r)
    
    test_cases = [
        (0.1, 2 * np.arctanh(0.1)),
        (0.3, 2 * np.arctanh(0.3)),
        (0.5, 2 * np.arctanh(0.5)),
        (0.7, 2 * np.arctanh(0.7)),
    ]
    
    origin = np.array([0.0, 0.0, 0.0])
    
    for r, expected_dist in test_cases:
        point = np.array([r, 0.0, 0.0])
        computed_dist = poincare_distance(origin, point)
        
        error = abs(computed_dist - expected_dist)
        
        assert error < 1e-14, (
            f"Distance formula incorrect for r={r}:\n"
            f"Expected: {expected_dist:.15f}\n"
            f"Computed: {computed_dist:.15f}\n"
            f"Error: {error:.2e}"
        )


def test_distance_bounds():
    """Test that distances are bounded appropriately."""
    # Points well inside the ball should have finite distances
    points = sample_random_poincare_points(10, seed=222, radius=0.6)
    
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            d = poincare_distance(points[i], points[j])
            
            # Distance should be positive and finite
            assert 0 < d < np.inf, f"Distance out of bounds: {d}"
            
            # For points with radius < 0.6, distance should be reasonable
            assert d < 10.0, f"Distance unreasonably large: {d}"


def test_distance_near_boundary():
    """Test distance computation for points closer to the boundary."""
    origin = np.array([0.0, 0.0, 0.0])
    
    # Points at various radii
    radii = [0.5, 0.7, 0.85, 0.9, 0.95]
    
    for r in radii:
        point = np.array([r, 0.0, 0.0])
        d = poincare_distance(origin, point)
        expected = 2 * np.arctanh(r)
        
        error = abs(d - expected)
        
        # Allow slightly looser tolerance near boundary
        tolerance = 1e-12 if r < 0.9 else 1e-10
        
        assert error < tolerance, (
            f"Distance incorrect at r={r}: "
            f"expected={expected:.15f}, computed={d:.15f}, error={error:.2e}"
        )


def test_distance_scale_invariance_along_axes():
    """Test that distance scales correctly along coordinate axes."""
    origin = np.array([0.0, 0.0, 0.0])
    
    # Test along x, y, and z axes
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    
    r = 0.4
    expected_dist = 2 * np.arctanh(r)
    
    for axis in axes:
        point = r * axis
        d = poincare_distance(origin, point)
        
        error = abs(d - expected_dist)
        
        assert error < 1e-14, (
            f"Distance anisotropic along axis {axis}: "
            f"expected={expected_dist:.15f}, computed={d:.15f}, error={error:.2e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
