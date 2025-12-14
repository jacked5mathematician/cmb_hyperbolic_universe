"""
Test Lorentz group structure properties.

Validates that:
1. Generator matrices are proper Lorentz isometries: G^T η G = η, det(G) ≈ 1
2. Distance computed in hyperboloid model matches Poincaré distance
3. Group actions keep points inside the Poincaré ball

Critical for ensuring SO(3,1) representation is mathematically correct.
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


# Minkowski metric η = diag(1, -1, -1, -1)
ETA = np.diag([1.0, -1.0, -1.0, -1.0])


def poincare_to_hyperboloid(poincare_pt: np.ndarray) -> np.ndarray:
    """Convert Poincaré ball point to hyperboloid coordinates."""
    norm_squared = np.dot(poincare_pt, poincare_pt)
    X0 = (1 + norm_squared) / (1 - norm_squared)
    X_spatial = 2 * poincare_pt / (1 - norm_squared)
    return np.array([X0, X_spatial[0], X_spatial[1], X_spatial[2]], dtype=float)


def hyperboloid_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute hyperbolic distance in hyperboloid model: arccosh(-η(X,Y))."""
    # Minkowski inner product: X^T η Y = X0*Y0 - X1*Y1 - X2*Y2 - X3*Y3
    minkowski_product = X[0]*Y[0] - X[1]*Y[1] - X[2]*Y[2] - X[3]*Y[3]
    # For points on the hyperboloid, -η(X,Y) ≥ 1, with equality when X=Y
    # Distance is arccosh(-η(X,Y))
    cosh_arg = max(1.0, minkowski_product)  # Guard against numerical errors
    return np.arccosh(cosh_arg)


def sample_random_poincare_points(n: int, seed: int, radius: float = 0.7) -> np.ndarray:
    """Sample random points in Poincaré ball with deterministic seed."""
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < n:
        candidate = rng.uniform(-radius, radius, size=3)
        if np.linalg.norm(candidate) < radius:
            points.append(candidate)
    return np.array(points)


def apply_group_action_full(matrix: np.ndarray, poincare_pt: np.ndarray) -> np.ndarray:
    """Apply SO(3,1) matrix to a Poincaré ball point, return result in Poincaré."""
    hyperboloid = apply_so31_action(matrix, poincare_pt)
    klein = project_to_klein(hyperboloid)
    return klein_to_poincare([klein])[0]


@pytest.mark.parametrize("manifold_name", ["m003(-2,3)", "m004(-2,3)", "m188(-1,1)"])
def test_generators_are_lorentz_isometries(manifold_name):
    """Test that generator matrices satisfy G^T η G = η and det(G) ≈ 1."""
    group_elements, fallback = get_group_elements(manifold_name, max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip(f"SnapPy not available for {manifold_name}")
    
    # Get generators (exclude identity for more interesting tests)
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if not generators:
        pytest.skip(f"No non-identity generators for {manifold_name}")
    
    max_metric_error = 0.0
    max_det_error = 0.0
    test_count = 0
    
    for G in generators:
        # Test 1: G^T η G = η (preserves Minkowski metric)
        G_T_eta_G = G.T @ ETA @ G
        metric_error = np.linalg.norm(G_T_eta_G - ETA, ord='fro')
        max_metric_error = max(max_metric_error, metric_error)
        
        assert metric_error < 1e-10, (
            f"{manifold_name}: Generator violates Lorentz condition G^T η G = η\n"
            f"Frobenius norm error: {metric_error:.2e}\n"
            f"G^T η G =\n{G_T_eta_G}\n"
            f"η =\n{ETA}"
        )
        
        # Test 2: det(G) ≈ 1 (proper Lorentz group, not time-reversing)
        det_G = np.linalg.det(G)
        det_error = abs(det_G - 1.0)
        max_det_error = max(max_det_error, det_error)
        
        assert det_error < 1e-10, (
            f"{manifold_name}: Generator determinant not ≈ 1\n"
            f"det(G) = {det_G:.15f}, error = {det_error:.2e}"
        )
        
        test_count += 1
    
    print(f"\n{manifold_name}: Tested {test_count} generators")
    print(f"  Max metric error (||G^T η G - η||): {max_metric_error:.2e}")
    print(f"  Max determinant error |det(G) - 1|: {max_det_error:.2e}")


@pytest.mark.parametrize("seed", [42, 123, 789])
def test_distance_hyperboloid_matches_poincare(seed):
    """Test that distances computed in hyperboloid and Poincaré models agree."""
    points = sample_random_poincare_points(10, seed, radius=0.6)
    
    max_error = 0.0
    test_count = 0
    
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            x_poincare = points[i]
            y_poincare = points[j]
            
            # Distance in Poincaré model
            d_poincare = poincare_distance(x_poincare, y_poincare)
            
            # Convert to hyperboloid
            X_hyp = poincare_to_hyperboloid(x_poincare)
            Y_hyp = poincare_to_hyperboloid(y_poincare)
            
            # Distance in hyperboloid model
            d_hyperboloid = hyperboloid_distance(X_hyp, Y_hyp)
            
            error = abs(d_poincare - d_hyperboloid)
            max_error = max(max_error, error)
            test_count += 1
            
            assert error < 1e-10, (
                f"Distance mismatch between models:\n"
                f"Poincaré: {d_poincare:.15f}\n"
                f"Hyperboloid: {d_hyperboloid:.15f}\n"
                f"Error: {error:.2e}"
            )
    
    print(f"\nSeed {seed}: Tested {test_count} pairs, max error={max_error:.2e}")


def test_distance_hyperboloid_matches_poincare_at_origin():
    """Test distance model agreement for origin-to-point distances."""
    origin = np.array([0.0, 0.0, 0.0])
    origin_hyp = poincare_to_hyperboloid(origin)
    
    # Test along ray [r, 0, 0]
    test_radii = [0.1, 0.3, 0.5, 0.7, 0.85]
    
    for r in test_radii:
        point = np.array([r, 0.0, 0.0])
        point_hyp = poincare_to_hyperboloid(point)
        
        d_poincare = poincare_distance(origin, point)
        d_hyperboloid = hyperboloid_distance(origin_hyp, point_hyp)
        
        error = abs(d_poincare - d_hyperboloid)
        
        assert error < 1e-12, (
            f"Distance mismatch at r={r}:\n"
            f"Poincaré: {d_poincare:.15f}\n"
            f"Hyperboloid: {d_hyperboloid:.15f}\n"
            f"Error: {error:.2e}"
        )


@pytest.mark.parametrize("manifold_name", ["m003(-2,3)", "m004(-2,3)", "m188(-1,1)"])
@pytest.mark.parametrize("seed", [42, 123])
def test_action_stays_in_ball(manifold_name, seed):
    """Test that group actions keep points inside the Poincaré ball."""
    group_elements, fallback = get_group_elements(manifold_name, max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip(f"SnapPy not available for {manifold_name}")
    
    # Get generators
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if not generators:
        pytest.skip(f"No generators for {manifold_name}")
    
    # Sample random points
    points = sample_random_poincare_points(5, seed, radius=0.7)
    
    max_norm = 0.0
    test_count = 0
    violations = 0
    
    for x in points:
        for g in generators[:3]:  # Test first 3 generators
            g_x = apply_group_action_full(g, x)
            norm_gx = np.linalg.norm(g_x)
            max_norm = max(max_norm, norm_gx)
            test_count += 1
            
            # Points must stay strictly inside the ball
            if norm_gx >= 1.0 - 1e-12:
                violations += 1
            
            assert norm_gx < 1.0 - 1e-12, (
                f"{manifold_name}: Group action pushed point outside ball\n"
                f"Original point: {x}, norm={np.linalg.norm(x):.15f}\n"
                f"Transformed point: {g_x}, norm={norm_gx:.15f}\n"
                f"Safety margin: {1.0 - norm_gx:.2e} (required > 1e-12)"
            )
    
    print(f"\n{manifold_name}, seed {seed}: Tested {test_count} actions")
    print(f"  Max transformed norm: {max_norm:.15f}")
    print(f"  Min safety margin: {1.0 - max_norm:.2e}")
    print(f"  Violations (norm ≥ 1-1e-12): {violations}")


def test_action_stays_in_ball_near_boundary():
    """Test group action on points closer to the boundary."""
    group_elements, fallback = get_group_elements("m003(-2,3)", max_depth=1)
    
    if fallback or len(group_elements) < 2:
        pytest.skip("SnapPy not available")
    
    identity = np.eye(4)
    generators = [elem for elem in group_elements if not np.allclose(elem, identity, atol=1e-10)]
    
    if not generators:
        pytest.skip("No generators found")
    
    # Test with points at increasing radii
    test_radii = [0.5, 0.7, 0.85, 0.9]
    g = generators[0]
    
    for r in test_radii:
        point = np.array([r, 0.0, 0.0])
        g_point = apply_group_action_full(g, point)
        norm_gp = np.linalg.norm(g_point)
        
        assert norm_gp < 1.0 - 1e-12, (
            f"Action violated ball constraint at r={r}:\n"
            f"Transformed norm: {norm_gp:.15f}\n"
            f"Safety margin: {1.0 - norm_gp:.2e}"
        )


def test_lorentz_condition_identity():
    """Sanity check: identity matrix satisfies Lorentz condition."""
    I = np.eye(4)
    I_T_eta_I = I.T @ ETA @ I
    
    assert np.allclose(I_T_eta_I, ETA, atol=1e-15), "Identity violates Lorentz condition"
    assert abs(np.linalg.det(I) - 1.0) < 1e-15, "Identity determinant not 1"


def test_hyperboloid_constraint_satisfied():
    """Test that converted points satisfy the hyperboloid constraint X^T η X = 1."""
    points = sample_random_poincare_points(10, seed=999, radius=0.7)
    
    for x in points:
        X = poincare_to_hyperboloid(x)
        
        # Compute X^T η X = X0^2 - X1^2 - X2^2 - X3^2
        X_T_eta_X = X[0]**2 - X[1]**2 - X[2]**2 - X[3]**2
        
        error = abs(X_T_eta_X - 1.0)
        
        assert error < 1e-12, (
            f"Hyperboloid constraint violated:\n"
            f"Poincaré: {x}\n"
            f"Hyperboloid: {X}\n"
            f"X^T η X = {X_T_eta_X:.15f} (should be 1.0)\n"
            f"Error: {error:.2e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
