"""
Test hyperbolic coordinate round-trip conversions.

Validates that coordinate transformations are invertible:
    Poincaré → Hyperboloid → Pseudo-spherical (ρ,θ,φ) → Hyperboloid → Poincaré

If (ρ,θ,φ) extraction is wrong, everything downstream fails
even if geometry tests pass.

Critical for basis function construction.
"""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.transformations import (
    apply_so31_action,
    poincare_to_pseudo_spherical,
    project_to_klein,
    klein_to_poincare,
)


def poincare_to_hyperboloid(poincare_pt: np.ndarray) -> np.ndarray:
    """Convert Poincaré ball point to hyperboloid coordinates."""
    norm_squared = np.dot(poincare_pt, poincare_pt)
    X0 = (1 + norm_squared) / (1 - norm_squared)
    X_spatial = 2 * poincare_pt / (1 - norm_squared)
    return np.array([X0, X_spatial[0], X_spatial[1], X_spatial[2]], dtype=float)


def pseudo_spherical_to_hyperboloid(rho: float, theta: float, phi: float) -> np.ndarray:
    """Convert pseudo-spherical (ρ,θ,φ) to hyperboloid coordinates."""
    X0 = np.cosh(rho)
    sinh_rho = np.sinh(rho)
    X1 = sinh_rho * np.sin(theta) * np.cos(phi)
    X2 = sinh_rho * np.sin(theta) * np.sin(phi)
    X3 = sinh_rho * np.cos(theta)
    return np.array([X0, X1, X2, X3], dtype=float)


def hyperboloid_to_poincare(X: np.ndarray) -> np.ndarray:
    """Convert hyperboloid to Poincaré ball coordinates."""
    # X = (X0, X1, X2, X3) with X0^2 - X1^2 - X2^2 - X3^2 = 1
    # Poincaré: project from (1,0,0,0) through X to unit ball
    # Formula: p = X_spatial / (X0 + 1)
    X0, X1, X2, X3 = X
    denom = X0 + 1.0
    if abs(denom) < 1e-14:
        # Point is at infinity (X0 ≈ -1), map to origin
        return np.array([0.0, 0.0, 0.0])
    return np.array([X1 / denom, X2 / denom, X3 / denom])


def sample_random_poincare_points(n: int, seed: int, radius: float = 0.7) -> np.ndarray:
    """Sample random points in Poincaré ball with deterministic seed."""
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < n:
        candidate = rng.uniform(-radius, radius, size=3)
        if np.linalg.norm(candidate) < radius:
            points.append(candidate)
    return np.array(points)


@pytest.mark.parametrize("seed", [42, 123, 456, 789])
def test_poincare_hyperboloid_roundtrip(seed):
    """Test Poincaré → Hyperboloid → Poincaré round-trip."""
    points = sample_random_poincare_points(10, seed, radius=0.7)
    
    max_error = 0.0
    for p_original in points:
        # Forward: Poincaré → Hyperboloid
        X = poincare_to_hyperboloid(p_original)
        
        # Check hyperboloid constraint
        constraint = X[0]**2 - X[1]**2 - X[2]**2 - X[3]**2
        assert abs(constraint - 1.0) < 1e-12, (
            f"Hyperboloid constraint violated: X^T η X = {constraint:.15f}"
        )
        
        # Backward: Hyperboloid → Poincaré
        p_reconstructed = hyperboloid_to_poincare(X)
        
        error = np.linalg.norm(p_original - p_reconstructed)
        max_error = max(max_error, error)
        
        assert error < 1e-12, (
            f"Round-trip error too large:\n"
            f"Original: {p_original}\n"
            f"Reconstructed: {p_reconstructed}\n"
            f"Error: {error:.2e}"
        )
    
    print(f"Seed {seed}: Tested {len(points)} points, max error={max_error:.2e}")


@pytest.mark.parametrize("seed", [42, 123, 789])
def test_poincare_pseudospherical_hyperboloid_roundtrip(seed):
    """Test full round-trip: Poincaré → Hyperboloid → (ρ,θ,φ) → Hyperboloid → Poincaré."""
    points = sample_random_poincare_points(10, seed, radius=0.7)
    
    max_error = 0.0
    max_rho_error = 0.0
    
    for p_original in points:
        # Step 1: Poincaré → Hyperboloid
        X1 = poincare_to_hyperboloid(p_original)
        
        # Step 2: Hyperboloid → Pseudo-spherical (ρ,θ,φ)
        # This is what poincare_to_pseudo_spherical does internally
        pseudo = poincare_to_pseudo_spherical(p_original.reshape(1, -1))[0]
        rho, theta, phi = pseudo
        
        # Validate (ρ,θ,φ) ranges
        assert rho >= 0.0, f"Invalid rho={rho} < 0"
        assert 0.0 <= theta <= np.pi, f"Invalid theta={theta}"
        assert -np.pi <= phi <= np.pi, f"Invalid phi={phi}"
        
        # Step 3: Pseudo-spherical → Hyperboloid
        X2 = pseudo_spherical_to_hyperboloid(rho, theta, phi)
        
        # Check that X1 ≈ X2 (same hyperboloid point)
        X_error = np.linalg.norm(X1 - X2)
        max_rho_error = max(max_rho_error, X_error)
        
        assert X_error < 1e-12, (
            f"Hyperboloid mismatch after (ρ,θ,φ) conversion:\n"
            f"Direct: {X1}\n"
            f"Via (ρ,θ,φ): {X2}\n"
            f"Error: {X_error:.2e}\n"
            f"(ρ,θ,φ) = ({rho:.6f}, {theta:.6f}, {phi:.6f})"
        )
        
        # Step 4: Hyperboloid → Poincaré
        p_reconstructed = hyperboloid_to_poincare(X2)
        
        # Final error
        error = np.linalg.norm(p_original - p_reconstructed)
        max_error = max(max_error, error)
        
        assert error < 1e-12, (
            f"Full round-trip error too large:\n"
            f"Original: {p_original}\n"
            f"Reconstructed: {p_reconstructed}\n"
            f"Error: {error:.2e}"
        )
    
    print(f"Seed {seed}: {len(points)} points, max error={max_error:.2e}, max ρ-path error={max_rho_error:.2e}")


def test_origin_roundtrip():
    """Test round-trip for the origin (special case)."""
    origin = np.array([0.0, 0.0, 0.0])
    
    # Poincaré → Hyperboloid
    X = poincare_to_hyperboloid(origin)
    expected_X = np.array([1.0, 0.0, 0.0, 0.0])
    
    assert np.allclose(X, expected_X, atol=1e-14), (
        f"Origin not mapped to (1,0,0,0): {X}"
    )
    
    # Hyperboloid → Pseudo-spherical
    pseudo = poincare_to_pseudo_spherical(origin.reshape(1, -1))[0]
    rho, theta, phi = pseudo
    
    assert abs(rho) < 1e-14, f"Origin rho={rho}, expected 0"
    
    # Pseudo-spherical → Hyperboloid
    X2 = pseudo_spherical_to_hyperboloid(rho, theta, phi)
    
    # At rho=0, theta and phi are undefined, but X2 should still be (1,0,0,0)
    assert np.allclose(X2, expected_X, atol=1e-12), (
        f"Origin round-trip failed: {X2}"
    )
    
    # Hyperboloid → Poincaré
    p_reconstructed = hyperboloid_to_poincare(X2)
    
    assert np.linalg.norm(p_reconstructed) < 1e-12, (
        f"Origin not recovered: {p_reconstructed}"
    )


@pytest.mark.parametrize("r", [0.1, 0.3, 0.5, 0.7, 0.85])
def test_radial_points_roundtrip(r):
    """Test round-trip for points along coordinate axes."""
    # Test along x, y, z axes
    test_points = [
        np.array([r, 0.0, 0.0]),
        np.array([0.0, r, 0.0]),
        np.array([0.0, 0.0, r]),
    ]
    
    for p in test_points:
        # Full round-trip
        X = poincare_to_hyperboloid(p)
        pseudo = poincare_to_pseudo_spherical(p.reshape(1, -1))[0]
        rho, theta, phi = pseudo
        X2 = pseudo_spherical_to_hyperboloid(rho, theta, phi)
        p_reconstructed = hyperboloid_to_poincare(X2)
        
        error = np.linalg.norm(p - p_reconstructed)
        
        assert error < 1e-12, (
            f"Radial point {p} round-trip error: {error:.2e}"
        )


def test_known_rho_values():
    """Test that ρ values match expected distances from origin."""
    # For point [r, 0, 0], the hyperbolic distance from origin is 2*arctanh(r)
    # This should equal ρ
    
    test_radii = [0.1, 0.3, 0.5, 0.7]
    
    for r in test_radii:
        p = np.array([r, 0.0, 0.0])
        pseudo = poincare_to_pseudo_spherical(p.reshape(1, -1))[0]
        rho, theta, phi = pseudo
        
        expected_rho = 2 * np.arctanh(r)
        error = abs(rho - expected_rho)
        
        assert error < 1e-13, (
            f"ρ mismatch for r={r}:\n"
            f"Computed ρ: {rho:.15f}\n"
            f"Expected ρ: {expected_rho:.15f}\n"
            f"Error: {error:.2e}"
        )


def test_theta_phi_consistency():
    """Test that θ and φ correctly encode spherical direction."""
    # Point [0, 0, r] should have θ ≈ 0 (north pole)
    # Point [r, 0, 0] should have θ ≈ π/2, φ ≈ 0
    # Point [0, r, 0] should have θ ≈ π/2, φ ≈ π/2
    
    r = 0.5
    
    # North pole (along positive z-axis in Poincaré)
    p_north = np.array([0.0, 0.0, r])
    pseudo = poincare_to_pseudo_spherical(p_north.reshape(1, -1))[0]
    _, theta, _ = pseudo
    assert abs(theta) < 0.1, f"North pole θ={theta}, expected ≈0"
    
    # Along x-axis
    p_x = np.array([r, 0.0, 0.0])
    pseudo = poincare_to_pseudo_spherical(p_x.reshape(1, -1))[0]
    _, theta, phi = pseudo
    assert abs(theta - np.pi/2) < 0.1, f"x-axis θ={theta}, expected ≈π/2"
    assert abs(phi) < 0.1 or abs(phi - 2*np.pi) < 0.1, f"x-axis φ={phi}, expected ≈0"
    
    # Along y-axis
    p_y = np.array([0.0, r, 0.0])
    pseudo = poincare_to_pseudo_spherical(p_y.reshape(1, -1))[0]
    _, theta, phi = pseudo
    assert abs(theta - np.pi/2) < 0.1, f"y-axis θ={theta}, expected ≈π/2"
    assert abs(phi - np.pi/2) < 0.1, f"y-axis φ={phi}, expected ≈π/2"


@pytest.mark.parametrize("seed", [42, 123])
def test_hyperboloid_constraint_preserved(seed):
    """Test that all coordinate conversions preserve the hyperboloid constraint."""
    points = sample_random_poincare_points(10, seed, radius=0.7)
    
    for p in points:
        # Poincaré → Hyperboloid
        X1 = poincare_to_hyperboloid(p)
        constraint1 = X1[0]**2 - X1[1]**2 - X1[2]**2 - X1[3]**2
        assert abs(constraint1 - 1.0) < 1e-12
        
        # Via pseudo-spherical
        pseudo = poincare_to_pseudo_spherical(p.reshape(1, -1))[0]
        X2 = pseudo_spherical_to_hyperboloid(*pseudo)
        constraint2 = X2[0]**2 - X2[1]**2 - X2[2]**2 - X2[3]**2
        assert abs(constraint2 - 1.0) < 1e-12, (
            f"Constraint violated via (ρ,θ,φ): {constraint2:.15f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
