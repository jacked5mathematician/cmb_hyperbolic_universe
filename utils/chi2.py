"""
Chi-squared computation with multiple definition options.

This module provides different chi-squared definitions to match various
interpretations from the paper (eigenvalueprob.pdf).

All chi2 computation functions share the same signature (A, singular_values, vectors, n_smallest)
to work with the dispatcher function compute_chi2(). The `vectors` parameter is unused in most
implementations but kept for API consistency and potential future extensions.
"""
import numpy as np
from typing import Tuple, Optional


def compute_chi2_raw_residual(
    A: np.ndarray,
    singular_values: np.ndarray,
    vectors: np.ndarray,
    n_smallest: int = 3
) -> np.ndarray:
    """
    Compute chi^2 = ||A·a||^2 = sigma^2.
    
    This is the direct interpretation of equation 2.7 from the paper,
    where chi^2 is simply the squared smallest singular values.
    
    Args:
        A: Constraint matrix (M x N)
        singular_values: All singular values from SVD
        vectors: Right singular vectors (Vt from SVD) - unused, kept for API consistency
        n_smallest: Number of smallest values to return
    
    Returns:
        Array of chi^2 values for the n_smallest eigenmodes
    """
    if len(singular_values) == 0:
        return np.array([np.nan] * n_smallest)
    
    order = np.argsort(singular_values)
    take = min(n_smallest, len(singular_values))
    smallest_sigma = singular_values[order[:take]]
    return smallest_sigma ** 2


def compute_chi2_per_row(
    A: np.ndarray,
    singular_values: np.ndarray,
    vectors: np.ndarray,
    n_smallest: int = 3
) -> np.ndarray:
    """
    Compute chi^2 = ||A·a||^2 / M, normalized by number of rows.
    
    This definition scales by the constraint count, making chi^2 
    represent average squared residual per constraint.
    
    Args:
        A: Constraint matrix (M x N)
        singular_values: All singular values from SVD
        vectors: Right singular vectors (Vt from SVD) - unused, kept for API consistency
        n_smallest: Number of smallest values to return
    
    Returns:
        Array of chi^2 values for the n_smallest eigenmodes
    """
    if len(singular_values) == 0:
        return np.array([np.nan] * n_smallest)
    
    M = A.shape[0]
    order = np.argsort(singular_values)
    take = min(n_smallest, len(singular_values))
    smallest_sigma = singular_values[order[:take]]
    return (smallest_sigma ** 2) / M


def compute_chi2_ratio(
    A: np.ndarray,
    singular_values: np.ndarray,
    vectors: np.ndarray,
    n_smallest: int = 3
) -> np.ndarray:
    """
    Compute chi^2 = (sigma_min / sigma_max)^2 for each rank.
    
    This definition uses the condition number to measure how well-resolved
    the eigenmode is relative to the matrix scale. This is likely the 
    paper's definition since it naturally produces O(1) values.
    
    Args:
        A: Constraint matrix (M x N)
        singular_values: All singular values from SVD
        vectors: Right singular vectors (Vt from SVD) - unused, kept for API consistency
        n_smallest: Number of smallest values to return
    
    Returns:
        Array of chi^2 values for the n_smallest eigenmodes
    """
    if len(singular_values) == 0:
        return np.array([np.nan] * n_smallest)
    
    sigma_max = singular_values.max()
    if sigma_max == 0:
        return np.array([np.nan] * n_smallest)
    
    order = np.argsort(singular_values)
    take = min(n_smallest, len(singular_values))
    smallest_sigma = singular_values[order[:take]]
    
    return (smallest_sigma / sigma_max) ** 2


def compute_chi2_frobenius(
    A: np.ndarray,
    singular_values: np.ndarray,
    vectors: np.ndarray,
    n_smallest: int = 3
) -> np.ndarray:
    """
    Compute chi^2 = ||A·a||^2 / ||A||_F^2.
    
    This definition normalizes by the Frobenius norm of A, scaling chi^2
    by the overall matrix magnitude. The Frobenius norm equals the sum
    of squared singular values.
    
    Args:
        A: Constraint matrix (M x N)
        singular_values: All singular values from SVD
        vectors: Right singular vectors (Vt from SVD) - unused, kept for API consistency
        n_smallest: Number of smallest values to return
    
    Returns:
        Array of chi^2 values for the n_smallest eigenmodes
    """
    frobenius_sq = np.sum(singular_values ** 2)
    if frobenius_sq == 0:
        return np.array([np.nan] * n_smallest)
    
    order = np.argsort(singular_values)
    take = min(n_smallest, len(singular_values))
    smallest_sigma = singular_values[order[:take]]
    
    return (smallest_sigma ** 2) / frobenius_sq


# Dispatch dictionary for chi2 definitions
CHI2_DEFINITIONS = {
    'raw_residual': compute_chi2_raw_residual,
    'per_row': compute_chi2_per_row,
    'ratio': compute_chi2_ratio,
    'frobenius': compute_chi2_frobenius,
}


def compute_chi2(
    A: np.ndarray,
    singular_values: np.ndarray,
    vectors: np.ndarray,
    definition: str = 'raw_residual',
    n_smallest: int = 3
) -> np.ndarray:
    """
    Compute chi-squared using the specified definition.
    
    Args:
        A: Constraint matrix (M x N)
        singular_values: All singular values from SVD
        vectors: Right singular vectors (Vt from SVD)
        definition: One of 'raw_residual', 'per_row', 'ratio', 'frobenius'
        n_smallest: Number of smallest values to return
    
    Returns:
        Array of chi^2 values for the n_smallest eigenmodes
    
    Raises:
        ValueError: If definition is not recognized
    """
    if definition not in CHI2_DEFINITIONS:
        valid_defs = ', '.join(CHI2_DEFINITIONS.keys())
        raise ValueError(
            f"Unknown chi2 definition '{definition}'. "
            f"Must be one of: {valid_defs}"
        )
    
    compute_func = CHI2_DEFINITIONS[definition]
    return compute_func(A, singular_values, vectors, n_smallest)
