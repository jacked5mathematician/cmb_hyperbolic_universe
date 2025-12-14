"""
Test group composition consistency.

Validates Invariant B: For any point x and group elements g,h,
    g(h(x)) ≈ (g∘h)(x)

This ensures the group action is correctly implemented for Cornish & Spergel (1999).
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
def test_group_composition_two_generators(seed):
    """Test g(h(x)) = (g∘h)(x) for two generators and random point."""
    group_elements, fallback = get_group_elements("m003(-2,3)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available or insufficient generators")
    
    # Get two non-identity generators
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if len(generators) < 2:
        pytest.skip("Need at least 2 non-identity generators")
    
    g, h = generators[0], generators[1]
    
    # Sample random point
    x = sample_random_poincare_points(1, seed)[0]
    
    # Method 1: Apply h then g sequentially
    h_x = apply_group_action_full(h, x)
    g_h_x = apply_group_action_full(g, h_x)
    
    # Method 2: Compose matrices first, then apply
    g_compose_h = g @ h
    gh_x = apply_group_action_full(g_compose_h, x)
    
    # Compare results
    error = np.linalg.norm(g_h_x - gh_x)
    
    assert error < 1e-10, (
        f"Composition failed: ||g(h(x)) - (g∘h)(x)|| = {error:.2e}\n"
        f"g(h(x)) = {g_h_x}\n"
        f"(g∘h)(x) = {gh_x}"
    )


def test_group_composition_multiple_points():
    """Test composition consistency for multiple points."""
    group_elements, fallback = get_group_elements("m003(-2,3)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available")
    
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if len(generators) < 2:
        pytest.skip("Need at least 2 generators")
    
    g, h = generators[0], generators[1]
    g_compose_h = g @ h
    
    # Test with multiple points
    points = sample_random_poincare_points(5, seed=42, radius=0.6)
    
    max_error = 0.0
    for x in points:
        # Sequential application
        h_x = apply_group_action_full(h, x)
        g_h_x = apply_group_action_full(g, h_x)
        
        # Composed application
        gh_x = apply_group_action_full(g_compose_h, x)
        
        error = np.linalg.norm(g_h_x - gh_x)
        max_error = max(max_error, error)
        
        assert error < 1e-10, f"Composition failed at point {x}: error={error:.2e}"
    
    print(f"\nTested 5 points, max composition error={max_error:.2e}")


def test_composition_with_identity():
    """Test that composing with identity doesn't change the result."""
    group_elements, fallback = get_group_elements("m003(-2,3)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available")
    
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if not generators:
        pytest.skip("No generators found")
    
    g = generators[0]
    x = sample_random_poincare_points(1, seed=123)[0]
    
    # g(I(x)) should equal g(x)
    I_x = apply_group_action_full(identity, x)
    g_I_x = apply_group_action_full(g, I_x)
    g_x = apply_group_action_full(g, x)
    
    assert np.allclose(g_I_x, g_x, atol=1e-14), "g∘I ≠ g"
    
    # I(g(x)) should equal g(x)
    I_g_x = apply_group_action_full(identity, g_x)
    
    assert np.allclose(I_g_x, g_x, atol=1e-14), "I∘g ≠ g"
    
    # (g∘I)(x) should equal g(x)
    g_compose_I = g @ identity
    gI_x = apply_group_action_full(g_compose_I, x)
    
    assert np.allclose(gI_x, g_x, atol=1e-14), "Matrix composition g∘I ≠ g"


def test_composition_associativity():
    """Test that (g∘h)∘k = g∘(h∘k) for three generators."""
    group_elements, fallback = get_group_elements("m003(-2,3)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available")
    
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if len(generators) < 3:
        pytest.skip("Need at least 3 generators")
    
    g, h, k = generators[0], generators[1], generators[2]
    x = sample_random_poincare_points(1, seed=456)[0]
    
    # Method 1: (g∘h)∘k
    gh = g @ h
    gh_k = gh @ k
    result1 = apply_group_action_full(gh_k, x)
    
    # Method 2: g∘(h∘k)
    hk = h @ k
    g_hk = g @ hk
    result2 = apply_group_action_full(g_hk, x)
    
    error = np.linalg.norm(result1 - result2)
    
    assert error < 1e-10, (
        f"Associativity failed: ||(g∘h)∘k - g∘(h∘k)|| = {error:.2e}"
    )


def test_inverse_composition():
    """Test that g∘g⁻¹ ≈ I (identity) when applied to points."""
    group_elements, fallback = get_group_elements("m003(-2,3)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available")
    
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if not generators:
        pytest.skip("No generators found")
    
    g = generators[0]
    
    # Compute inverse
    try:
        g_inv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        pytest.skip("Generator is singular")
    
    # Compose g with its inverse
    g_compose_ginv = g @ g_inv
    
    # Should be close to identity matrix
    assert np.allclose(g_compose_ginv, identity, atol=1e-10), "g∘g⁻¹ ≠ I (matrix level)"
    
    # Test on points
    x = sample_random_poincare_points(1, seed=789)[0]
    
    # Apply g then g⁻¹
    g_x = apply_group_action_full(g, x)
    ginv_g_x = apply_group_action_full(g_inv, g_x)
    
    # Should return to original point
    error = np.linalg.norm(ginv_g_x - x)
    
    assert error < 1e-10, (
        f"Inverse composition failed: ||g⁻¹(g(x)) - x|| = {error:.2e}\n"
        f"Original: {x}\n"
        f"After g∘g⁻¹: {ginv_g_x}"
    )


def test_composition_at_origin():
    """Test composition with origin point."""
    group_elements, fallback = get_group_elements("m003(-2,3)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available")
    
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if len(generators) < 2:
        pytest.skip("Need at least 2 generators")
    
    g, h = generators[0], generators[1]
    origin = np.array([0.0, 0.0, 0.0])
    
    # Sequential
    h_origin = apply_group_action_full(h, origin)
    g_h_origin = apply_group_action_full(g, h_origin)
    
    # Composed
    gh = g @ h
    gh_origin = apply_group_action_full(gh, origin)
    
    error = np.linalg.norm(g_h_origin - gh_origin)
    
    assert error < 1e-10, f"Composition at origin failed: error={error:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
