"""
Test eigenvalue sensitivity using synthetic constraint systems.

This test verifies that the SVD-based chi-squared computation correctly
identifies rank deficiencies in constraint matrices, which is the fundamental
mechanism for detecting eigenvalues.
"""

import numpy as np
import pytest

from utils.svd import solve_system_via_svd_numeric


def test_rank_drop_detection_simple():
    """Test that SVD detects rank drop in a simple synthetic matrix."""
    # Create a matrix with known rank deficiency
    # Build a 10x5 matrix with rank 4 (one zero singular value)
    
    # Start with random matrix
    np.random.seed(42)
    U = np.random.randn(10, 5)
    V = np.random.randn(5, 5)
    
    # Create singular values with one near-zero
    singular_values = np.array([10.0, 5.0, 2.0, 1.0, 1e-10])
    S = np.diag(singular_values)
    
    # Construct matrix: A = U @ S @ V.T
    A = U @ S @ V.T
    
    # Solve with SVD
    chi2_values, solution_vectors, diagnostics = solve_system_via_svd_numeric(
        A, n_smallest=3, normalize_rows=False, chi2_definition='raw_residual'
    )
    
    # The smallest chi-squared should be very small (corresponding to zero singular value)
    assert chi2_values[0] < 1e-15, "Should detect rank deficiency with near-zero chi²"
    
    # Verify singular value is captured
    assert diagnostics['sigma_min'] < 1e-9, "Should report very small singular value"
    
    # Higher rank solutions should have larger chi²
    assert chi2_values[1] > chi2_values[0], "Higher ranks should have larger chi²"
    assert chi2_values[2] > chi2_values[1], "Chi² should increase with rank"


def test_rank_drop_at_specific_parameter():
    """
    Test that chi² varies with a parameter, showing minima at specific values.
    
    This simulates the k-scanning behavior where the constraint matrix becomes
    rank-deficient (eigenvalue) at specific parameter values.
    """
    # Simulate scanning over a parameter (like k in the real problem)
    chi2_values = []
    param_values = np.linspace(0, 10, 50)
    
    # Create matrices that have rank drop near param=5
    for param in param_values:
        # Build matrix where smallest singular value depends on parameter
        # It should be small near param=5
        np.random.seed(42)
        base_matrix = np.random.randn(20, 10)
        
        # Add a component that becomes singular at param=5
        # The closer param is to 5, the smaller the determinant-like quantity
        singularity_factor = (param - 5.0) ** 2 + 0.01
        
        # Create a matrix where one column is nearly dependent on others near param=5
        A = base_matrix.copy()
        # Make last column almost a linear combination when param ≈ 5
        A[:, -1] = A[:, 0] + A[:, 1] + singularity_factor * np.random.randn(20)
        
        chi2, _, _ = solve_system_via_svd_numeric(
            A, n_smallest=1, normalize_rows=False, chi2_definition='raw_residual'
        )
        chi2_values.append(chi2[0])
    
    chi2_array = np.array(chi2_values)
    
    # Find minimum
    min_idx = np.argmin(chi2_array)
    param_at_min = param_values[min_idx]
    
    # Verify minimum is near param=5
    assert abs(param_at_min - 5.0) < 1.0, f"Minimum should be near 5, got {param_at_min}"
    
    # Verify chi² increases away from minimum
    # Check that average chi² in the wings is larger than at minimum
    wing_indices = np.where((param_values < 2) | (param_values > 8))[0]
    assert chi2_array[wing_indices].mean() > chi2_array[min_idx], \
        "Chi² should be larger away from the minimum"


def test_chi2_definitions_consistency():
    """Test that different chi² definitions return consistent results."""
    # Test that all definitions can be computed without error
    # and return finite values
    
    np.random.seed(42)
    M, N = 25, 12
    A = np.random.randn(M, N)
    
    definitions = ['raw_residual', 'per_row', 'ratio', 'frobenius']
    
    for defn in definitions:
        chi2, _, diag = solve_system_via_svd_numeric(
            A, n_smallest=3, normalize_rows=False, chi2_definition=defn
        )
        
        # All chi² values should be finite
        assert all(np.isfinite(chi2)), f"Definition '{defn}' returned non-finite chi²"
        
        # Chi² should be non-negative
        assert all(c >= 0 for c in chi2), f"Definition '{defn}' returned negative chi²"
        
        # Should have diagnostics
        assert 'sigma_min' in diag and np.isfinite(diag['sigma_min'])
        assert 'numerical_rank' in diag


def test_multiple_rank_solutions():
    """Test that SVD returns multiple solutions ordered by chi²."""
    # Create matrix with defined rank structure
    np.random.seed(42)
    M, N = 30, 10
    
    # Build matrix with specific singular value structure
    U = np.random.randn(M, N)
    V = np.random.randn(N, N)
    S = np.diag([10, 8, 5, 3, 2, 1.5, 1.0, 0.5, 0.1, 0.01])
    A = U @ S @ V.T
    
    # Get 5 solutions
    chi2_values, solutions, _ = solve_system_via_svd_numeric(
        A, n_smallest=5, normalize_rows=False, chi2_definition='raw_residual'
    )
    
    # Verify ordering: chi² should be increasing
    for i in range(len(chi2_values) - 1):
        assert chi2_values[i] <= chi2_values[i+1], \
            f"Chi² should increase: {chi2_values[i]} > {chi2_values[i+1]}"
    
    # Verify we got 5 solutions
    assert len(chi2_values) == 5
    assert len(solutions) == 5
    
    # Verify solution shapes
    for sol in solutions:
        assert sol.shape == (N,), f"Solution should be length {N}"


def test_normalized_vs_unnormalized_rows():
    """Test that row normalization affects chi² values."""
    # Create a matrix with varied row norms
    np.random.seed(42)
    M, N = 20, 8
    A = np.random.randn(M, N)
    
    # Scale some rows to have different magnitudes
    for i in range(0, M, 3):
        A[i, :] *= 10.0  # Make some rows large
    
    # Test both modes
    chi2_norm, _, diag_norm = solve_system_via_svd_numeric(
        A, n_smallest=3, normalize_rows=True, chi2_definition='raw_residual'
    )
    chi2_unnorm, _, diag_unnorm = solve_system_via_svd_numeric(
        A, n_smallest=3, normalize_rows=False, chi2_definition='raw_residual'
    )
    
    # Both should return finite values
    assert all(np.isfinite(chi2_norm)), "Normalized chi² should be finite"
    assert all(np.isfinite(chi2_unnorm)), "Unnormalized chi² should be finite"
    
    # Values should differ significantly (normalization has an effect)
    # Don't check exact values but verify they're in different scales
    assert not np.allclose(chi2_normalized := chi2_norm, chi2_unnormalized := chi2_unnorm, rtol=0.1), \
        "Normalized and unnormalized chi² should have different scales"
    
    # Both should have valid diagnostics
    assert 'numerical_rank' in diag_norm
    assert 'numerical_rank' in diag_unnorm


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
