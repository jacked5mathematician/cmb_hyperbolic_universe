"""
Tests for normalized radial functions X_k^ℓ(ρ).

Part 2 of the cutoff fix: Verify that X_normalized has correct asymptotic amplitude
and that the cutoff equation has roots.
"""
import numpy as np
import pytest

from utils.radial_normalized import X_normalized, get_normalization_factor, clear_normalization_cache


def L_from_k(k: float) -> int:
    """Paper rule: L = floor(k) + 10"""
    return int(np.floor(k)) + 10


class TestAsymptoticAmplitude:
    """Test that X_normalized * sinh(ρ) has envelope ~1 on ρ ∈ [10, 20]."""
    
    @pytest.mark.parametrize("k", [1.0, 2.0, 5.0, 10.0, 20.0])
    def test_amplitude_near_one(self, k: float):
        """
        For several (k, ell), X_norm * sinh(ρ) should have envelope ~1 
        on [10, 20] (within tolerance).
        """
        ell = L_from_k(k)
        
        # Evaluate on asymptotic regime
        rho_values = np.linspace(10.0, 20.0, 50)
        envelope_values = []
        
        for rho in rho_values:
            X = X_normalized(k, ell, rho)
            val = X * np.sinh(rho)
            envelope_values.append(abs(val))
        
        max_envelope = max(envelope_values)
        min_envelope = min(envelope_values)
        mean_envelope = np.mean(envelope_values)
        
        # The envelope should oscillate around 1.0
        # Max should be ~1, min should be ~0 (at nodes), mean should be ~0.7
        # We're more interested in the max value being close to 1
        
        print(f"\nk={k:.1f}, L={ell}: max={max_envelope:.4f}, min={min_envelope:.4f}, mean={mean_envelope:.4f}")
        
        # Check that maximum envelope is close to 1.0
        assert abs(max_envelope - 1.0) < 0.15, (
            f"k={k}, ell={ell}: max envelope {max_envelope:.4f} deviates from 1.0 by "
            f"{abs(max_envelope - 1.0):.4f} > 0.15"
        )
        
        # Mean should be reasonable (around 0.5-0.8 for oscillating function)
        assert 0.3 < mean_envelope < 0.9, (
            f"k={k}, ell={ell}: mean envelope {mean_envelope:.4f} outside [0.3, 0.9]"
        )
    
    def test_normalization_factor_reasonable(self):
        """Normalization factors R(k, ell) should be reasonable."""
        test_cases = [
            (1.0, 11, 1.0, 1.8),    # k=1: expect R = k√2 ≈ 1.414
            (5.0, 15, 6.0, 8.0),    # k=5: expect R = 5√2 ≈ 7.071
            (10.0, 20, 13.0, 15.0), # k=10: expect R = 10√2 ≈ 14.142
            (20.0, 30, 27.0, 30.0), # k=20: expect R = 20√2 ≈ 28.284
        ]
        
        for k, ell, min_expected, max_expected in test_cases:
            R = get_normalization_factor(k, ell)
            print(f"k={k:.1f}, L={ell}: R={R:.6f}")
            
            assert min_expected < R < max_expected, (
                f"k={k}, ell={ell}: R={R:.6f} outside expected range "
                f"[{min_expected}, {max_expected}]"
            )


class TestCutoffRootExists:
    """Test that the equation X_norm(k, L, ρ) * sinh(ρ) = 0.25 has roots."""
    
    def _has_sign_change(self, k: float, ell: int, rho_start: float, rho_end: float, 
                        threshold: float = 0.25, step: float = 0.05) -> bool:
        """
        Check if X_norm * sinh - threshold changes sign on [rho_start, rho_end].
        """
        rho = rho_start
        prev_val = X_normalized(k, ell, rho) * np.sinh(rho) - threshold
        
        while rho <= rho_end:
            rho += step
            curr_val = X_normalized(k, ell, rho) * np.sinh(rho) - threshold
            
            if prev_val * curr_val < 0:
                # Sign change detected
                return True
            
            prev_val = curr_val
        
        return False
    
    def _find_turning_point(self, k: float, ell: int) -> float:
        """Compute turning point ρ₀ = arcsinh(sqrt(ℓ(ℓ+1)) / k)."""
        return float(np.arcsinh(np.sqrt(ell * (ell + 1)) / k))
    
    @pytest.mark.parametrize("k", [1.0, 2.0, 5.0, 10.0, 20.0])
    def test_root_exists_after_turning_point(self, k: float):
        """
        For k ∈ {1, 2, 5, 10, 20} with L = 10 + floor(k),
        the equation X_norm(k, L, ρ) * sinh(ρ) = 0.25 should have a root.
        
        This is the critical test: if this passes, the envelope bug is fixed.
        """
        ell = L_from_k(k)
        rho0 = self._find_turning_point(k, ell)
        
        # Search from turning point + small offset to rho_cap=30
        rho_start = rho0 + 0.1
        rho_end = 30.0
        
        has_root = self._has_sign_change(k, ell, rho_start, rho_end, threshold=0.25)
        
        print(f"\nk={k:.1f}, L={ell}: ρ₀={rho0:.3f}, searching [{rho_start:.3f}, {rho_end:.1f}]")
        
        if has_root:
            # Find approximate root location for reporting
            rho = rho_start
            step = 0.05
            while rho <= rho_end:
                val = X_normalized(k, ell, rho) * np.sinh(rho)
                if abs(val - 0.25) < 0.01:  # Close to threshold
                    print(f"  Root near ρ={rho:.3f}, X*sinh(ρ)={val:.4f}")
                    break
                rho += step
            
            # Test passes if root exists
            assert True
        else:
            # Diagnostic: print max value reached
            rho_vals = np.linspace(rho_start, rho_end, 100)
            max_val = 0.0
            max_rho = rho_start
            
            for rho in rho_vals:
                val = X_normalized(k, ell, rho) * np.sinh(rho)
                if val > max_val:
                    max_val = val
                    max_rho = rho
            
            pytest.fail(
                f"k={k}, ell={ell}: NO ROOT found!\n"
                f"  Turning point: ρ₀={rho0:.3f}\n"
                f"  Searched: [{rho_start:.3f}, {rho_end:.1f}]\n"
                f"  Max value: {max_val:.4f} at ρ={max_rho:.3f}\n"
                f"  Threshold: 0.25\n"
                f"  → X_norm never crosses threshold"
            )
    
    def test_root_value_correctness(self):
        """
        For a simple case (k=2), verify that the root value is actually correct.
        """
        k = 2.0
        ell = L_from_k(k)
        rho0 = self._find_turning_point(k, ell)
        
        # Scan for root
        rho = rho0 + 0.1
        step = 0.02
        root_rho = None
        
        prev_val = X_normalized(k, ell, rho) * np.sinh(rho) - 0.25
        
        while rho < 30.0:
            rho += step
            curr_val = X_normalized(k, ell, rho) * np.sinh(rho) - 0.25
            
            if prev_val * curr_val < 0:
                # Found sign change - root is between rho-step and rho
                root_rho = rho - step / 2
                break
            
            prev_val = curr_val
        
        assert root_rho is not None, "Should find a root for k=2"
        
        # Verify the root
        val_at_root = X_normalized(k, ell, root_rho) * np.sinh(root_rho)
        print(f"\nk={k}, L={ell}: root at ρ={root_rho:.4f}, X*sinh(ρ)={val_at_root:.4f}")
        
        # Should be close to 0.25
        assert abs(val_at_root - 0.25) < 0.05, (
            f"Value at root {val_at_root:.4f} deviates from 0.25"
        )


class TestNormalizationCache:
    """Test the caching mechanism."""
    
    def test_cache_stores_values(self):
        """Cache should store computed normalization factors."""
        clear_normalization_cache()
        
        # Compute for several (k, ell)
        _ = get_normalization_factor(1.0, 11)
        _ = get_normalization_factor(5.0, 15)
        _ = get_normalization_factor(10.0, 20)
        
        from utils.radial_normalized import get_cache_size
        size = get_cache_size()
        
        assert size >= 3, f"Cache should have at least 3 entries, got {size}"
    
    def test_cache_quantizes_k(self):
        """Cache should quantize k to avoid explosion."""
        clear_normalization_cache()
        
        # These should map to the same cache entry
        R1 = get_normalization_factor(5.00001, 15)
        R2 = get_normalization_factor(5.00002, 15)
        
        # Should be identical (from cache)
        assert R1 == R2, "Quantization should produce identical cached values"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
