"""
Unit tests for chi-squared definitions.

This test verifies that different chi2 definitions produce numerically
correct values for known test matrices.
"""
import numpy as np
import pytest
import sys
import os

# Ensure utils is in the path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.chi2 import (
    compute_chi2,
    compute_chi2_raw_residual,
    compute_chi2_per_row,
    compute_chi2_ratio,
    compute_chi2_frobenius,
)
from scipy.linalg import svd


def test_chi2_raw_residual():
    """Test raw_residual definition: chi^2 = sigma^2."""
    # Create a simple test matrix with known singular values
    A = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5]
    ])
    
    _, s, Vt = svd(A, full_matrices=False)
    
    chi2 = compute_chi2_raw_residual(A, s, Vt, n_smallest=3)
    
    # Expected: squared smallest singular values
    # Singular values are [3.0, 2.0, 1.118...]
    # Smallest is 1.118..., so chi^2 should be ~1.25
    assert len(chi2) == 3
    assert chi2[0] < chi2[1] < chi2[2]
    assert np.allclose(chi2[0], 1.118**2, rtol=1e-2)


def test_chi2_per_row():
    """Test per_row definition: chi^2 = sigma^2 / M."""
    A = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5]
    ])
    M = A.shape[0]
    
    _, s, Vt = svd(A, full_matrices=False)
    
    chi2 = compute_chi2_per_row(A, s, Vt, n_smallest=3)
    chi2_raw = compute_chi2_raw_residual(A, s, Vt, n_smallest=3)
    
    # Should be raw divided by M
    assert len(chi2) == 3
    assert np.allclose(chi2, chi2_raw / M, rtol=1e-10)


def test_chi2_ratio():
    """Test ratio definition: chi^2 = (sigma_min / sigma_max)^2."""
    A = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5]
    ])
    
    _, s, Vt = svd(A, full_matrices=False)
    
    chi2 = compute_chi2_ratio(A, s, Vt, n_smallest=3)
    
    # Singular values in descending order are [4, 2, 1.118...]
    # sigma_max = 4.0
    # Smallest values: 1.118..., 2.0, 4.0
    # Ratios: (1.118/4)^2, (2/4)^2, (4/4)^2 = ~0.078, 0.25, 1.0
    assert len(chi2) == 3
    assert chi2[0] < chi2[1] < chi2[2]
    assert np.allclose(chi2[0], (1.118/4.0)**2, rtol=1e-2)
    assert np.allclose(chi2[1], (2.0/4.0)**2, rtol=1e-5)
    assert np.allclose(chi2[2], (4.0/4.0)**2, rtol=1e-5)


def test_chi2_frobenius():
    """Test frobenius definition: chi^2 = sigma^2 / ||A||_F^2."""
    A = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5]
    ])
    
    _, s, Vt = svd(A, full_matrices=False)
    
    chi2 = compute_chi2_frobenius(A, s, Vt, n_smallest=3)
    chi2_raw = compute_chi2_raw_residual(A, s, Vt, n_smallest=3)
    
    # Frobenius norm squared = sum of squared singular values
    frobenius_sq = np.sum(s ** 2)
    
    # Should be raw divided by frobenius_sq
    assert len(chi2) == 3
    assert np.allclose(chi2, chi2_raw / frobenius_sq, rtol=1e-10)


def test_chi2_dispatcher():
    """Test the main compute_chi2 dispatcher function."""
    A = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5]
    ])
    
    _, s, Vt = svd(A, full_matrices=False)
    
    # Test all definitions work through dispatcher
    chi2_raw = compute_chi2(A, s, Vt, definition='raw_residual', n_smallest=3)
    chi2_per_row = compute_chi2(A, s, Vt, definition='per_row', n_smallest=3)
    chi2_ratio = compute_chi2(A, s, Vt, definition='ratio', n_smallest=3)
    chi2_frob = compute_chi2(A, s, Vt, definition='frobenius', n_smallest=3)
    
    # All should return arrays of length 3
    assert len(chi2_raw) == 3
    assert len(chi2_per_row) == 3
    assert len(chi2_ratio) == 3
    assert len(chi2_frob) == 3
    
    # Verify they match direct calls
    assert np.allclose(chi2_raw, compute_chi2_raw_residual(A, s, Vt, n_smallest=3))
    assert np.allclose(chi2_per_row, compute_chi2_per_row(A, s, Vt, n_smallest=3))
    assert np.allclose(chi2_ratio, compute_chi2_ratio(A, s, Vt, n_smallest=3))
    assert np.allclose(chi2_frob, compute_chi2_frobenius(A, s, Vt, n_smallest=3))


def test_chi2_unknown_definition():
    """Test that unknown definition raises ValueError."""
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    _, s, Vt = svd(A, full_matrices=False)
    
    with pytest.raises(ValueError, match="Unknown chi2 definition"):
        compute_chi2(A, s, Vt, definition='invalid_name', n_smallest=1)


def test_chi2_edge_cases():
    """Test edge cases like zero matrices and single values."""
    # Zero matrix
    A = np.zeros((3, 3))
    _, s, Vt = svd(A, full_matrices=False)
    
    chi2_ratio = compute_chi2_ratio(A, s, Vt, n_smallest=2)
    assert len(chi2_ratio) == 2
    assert all(np.isnan(chi2_ratio))
    
    chi2_frob = compute_chi2_frobenius(A, s, Vt, n_smallest=2)
    assert len(chi2_frob) == 2
    assert all(np.isnan(chi2_frob))
    
    # Single nonzero value
    A = np.array([[5.0]])
    _, s, Vt = svd(A, full_matrices=False)
    
    chi2_raw = compute_chi2_raw_residual(A, s, Vt, n_smallest=1)
    assert len(chi2_raw) == 1
    assert np.allclose(chi2_raw[0], 25.0)
    
    chi2_ratio = compute_chi2_ratio(A, s, Vt, n_smallest=1)
    assert len(chi2_ratio) == 1
    assert np.allclose(chi2_ratio[0], 1.0)  # sigma_min/sigma_max = 1 for single value


def test_chi2_realistic_matrix():
    """Test with a more realistic constraint matrix."""
    # Simulate a constraint matrix from the pipeline
    np.random.seed(42)
    M, N = 100, 25
    A = np.random.randn(M, N) * 0.1  # Small values like in the actual problem
    
    # Add structure: most constraints should be nearly satisfied
    # but a few eigenmodes should be poorly satisfied
    _, s, Vt = svd(A, full_matrices=False)
    
    # Test all definitions produce reasonable results
    chi2_raw = compute_chi2_raw_residual(A, s, Vt, n_smallest=5)
    chi2_per_row = compute_chi2_per_row(A, s, Vt, n_smallest=5)
    chi2_ratio = compute_chi2_ratio(A, s, Vt, n_smallest=5)
    chi2_frob = compute_chi2_frobenius(A, s, Vt, n_smallest=5)
    
    # All should be sorted (smallest to largest)
    assert all(chi2_raw[i] <= chi2_raw[i+1] for i in range(4))
    assert all(chi2_per_row[i] <= chi2_per_row[i+1] for i in range(4))
    assert all(chi2_ratio[i] <= chi2_ratio[i+1] for i in range(4))
    assert all(chi2_frob[i] <= chi2_frob[i+1] for i in range(4))
    
    # Check scaling relationships
    assert np.allclose(chi2_per_row, chi2_raw / M)
    
    # Ratio should be bounded [0, 1]
    assert all(0 <= x <= 1 for x in chi2_ratio)
    
    # Frobenius normalization should also be bounded
    assert all(0 <= x <= 1 for x in chi2_frob)


def test_chi2_definitions_differ():
    """Test that different definitions produce different results."""
    np.random.seed(123)
    A = np.random.randn(50, 15)
    _, s, Vt = svd(A, full_matrices=False)
    
    chi2_raw = compute_chi2_raw_residual(A, s, Vt, n_smallest=3)
    chi2_per_row = compute_chi2_per_row(A, s, Vt, n_smallest=3)
    chi2_ratio = compute_chi2_ratio(A, s, Vt, n_smallest=3)
    chi2_frob = compute_chi2_frobenius(A, s, Vt, n_smallest=3)
    
    # All definitions should differ significantly
    assert not np.allclose(chi2_raw, chi2_per_row)
    assert not np.allclose(chi2_raw, chi2_ratio)
    assert not np.allclose(chi2_raw, chi2_frob)
    assert not np.allclose(chi2_per_row, chi2_ratio)
    assert not np.allclose(chi2_per_row, chi2_frob)
    assert not np.allclose(chi2_ratio, chi2_frob)
