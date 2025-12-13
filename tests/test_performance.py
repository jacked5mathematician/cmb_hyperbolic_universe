"""Tests for performance optimizations and caching behavior."""
import numpy as np

from utils.sys_generation import generate_matrix_system
from utils.special_functions import (
    clear_special_function_caches,
    get_cache_stats,
    reset_call_counters,
)


def test_phi_reuse_optimization_reduces_calls():
    """Verify that Phi_l reuse optimization reduces mpmath calls."""
    # Create a simple test case with multiple m values for same l
    # For L=2, we have l=0,1,2 with m=-l..l giving 9 lm pairs total
    # (0,0), (1,-1), (1,0), (1,1), (2,-2), (2,-1), (2,0), (2,1), (2,2)
    points_images = [[
        (1.0, 0.5, 0.3),
        (1.2, 0.6, 0.4),
        (1.4, 0.7, 0.5),
    ]]
    L = 2
    k_value = 2.0
    
    # Clear caches and reset counters
    clear_special_function_caches()
    reset_call_counters()
    
    # Generate matrix system (uses optimized vectorized path)
    M, N, A = generate_matrix_system(points_images, L, k_value)
    
    # Check that we computed the matrix correctly
    assert N == (L + 1) ** 2  # 9 columns
    expected_M = len(points_images[0]) * (len(points_images[0]) - 1) // 2  # 3 pairs
    assert M == expected_M
    
    # Get call statistics
    stats = get_cache_stats()
    
    # With optimization, we should have:
    # - L+1 unique l values (l=0,1,2) → 3 unique l values
    # - Each l needs evaluation at 3 rho values
    # - Total legenp calls should be 3 (unique l) * 3 (rho values) = 9
    # Without optimization, it would be 9 (lm pairs) * 3 (rho values) = 27
    
    # Allow some tolerance for implementation details
    assert stats['legenp_calls'] <= 12, f"Expected ≤12 legenp calls, got {stats['legenp_calls']}"
    assert stats['legenp_calls'] >= 9, f"Expected ≥9 legenp calls, got {stats['legenp_calls']}"
    
    # Verify the reduction is significant (at least 2× reduction)
    naive_calls = N * len(points_images[0])  # Would be 9*3=27 without optimization
    actual_calls = stats['legenp_calls']
    reduction_factor = naive_calls / actual_calls
    assert reduction_factor >= 2.0, f"Expected ≥2× reduction, got {reduction_factor:.1f}×"


def test_clear_caches_functionality():
    """Verify that cache clearing works correctly."""
    # Create a test case
    points_images = [[
        (1.0, 0.5, 0.3),
        (1.2, 0.6, 0.4),
    ]]
    L = 1
    k_value = 1.5
    
    # Clear and generate once
    clear_special_function_caches()
    reset_call_counters()
    M1, N1, A1 = generate_matrix_system(points_images, L, k_value)
    stats1 = get_cache_stats()
    
    # Clear caches
    clear_special_function_caches()
    
    # Check cache was cleared
    stats_after_clear = get_cache_stats()
    assert stats_after_clear['phi_cache_size'] == 0
    assert stats_after_clear['y_lm_cache_size'] == 0
    
    # Generate again with same inputs
    reset_call_counters()
    M2, N2, A2 = generate_matrix_system(points_images, L, k_value)
    stats2 = get_cache_stats()
    
    # Results should be identical
    assert M1 == M2
    assert N1 == N2
    assert np.allclose(A1, A2)
    
    # Call counts should be similar (cache was cleared)
    assert stats2['legenp_calls'] == stats1['legenp_calls']


def test_cache_stats_format():
    """Verify that cache stats have expected structure."""
    stats = get_cache_stats()
    
    # Check all expected keys exist
    expected_keys = [
        'phi_cache_size', 'y_lm_cache_size',
        'rho_precision', 'angle_precision',
        'phi_calls', 'phi_cache_hits', 'phi_cache_misses',
        'y_lm_calls', 'y_lm_cache_hits', 'y_lm_cache_misses',
        'legenp_calls'
    ]
    
    for key in expected_keys:
        assert key in stats, f"Expected key '{key}' in cache stats"
    
    # Check types
    assert isinstance(stats['phi_cache_size'], int)
    assert isinstance(stats['phi_calls'], int)
    assert isinstance(stats['legenp_calls'], int)
