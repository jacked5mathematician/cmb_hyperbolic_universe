"""
Validate that Phi_nu_l matches the target X_k^ell radial eigenfunction.

For ell=0, the closed form is:
    X_k^0(rho) = sin(k*rho) / sinh(rho)
    => X_k^0(rho) * sinh(rho) = sin(k*rho)  (amplitude = 1)

Test: Does Phi_nu_l(k, 0, rho) * sinh(rho) match sin(k*rho)?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.special_functions import Phi_nu_l


def test_ell0_closed_form(k_values, rho_grid):
    """
    Compare Phi_nu_l(k, 0, rho) * sinh(rho) against target sin(k*rho).
    
    Returns:
        dict: Results for each k including amplitude, phase, and RMS error
    """
    results = {}
    
    for k in k_values:
        # Compute Phi_nu_l * sinh(rho) for ell=0
        phi_vals = []
        for rho in rho_grid:
            phi = Phi_nu_l(k, 0, rho)
            phi_sinh = float(phi * np.sinh(rho))
            phi_vals.append(phi_sinh)
        
        phi_vals = np.array(phi_vals)
        
        # Target: sin(k*rho)
        target_vals = np.sin(k * rho_grid)
        
        # Measure amplitude mismatch
        # Fit: phi_vals = A * sin(k*rho + phase)
        # For comparison, compute ratio at peaks
        phi_peaks = []
        target_peaks = []
        
        for i in range(1, len(rho_grid) - 1):
            # Detect peaks in target
            if target_vals[i-1] < target_vals[i] > target_vals[i+1]:
                if abs(target_vals[i]) > 0.5:  # Only clear peaks
                    phi_peaks.append(abs(phi_vals[i]))
                    target_peaks.append(abs(target_vals[i]))
        
        if phi_peaks:
            amplitude_ratio = np.mean(phi_peaks) / np.mean(target_peaks)
        else:
            amplitude_ratio = np.nan
        
        # RMS error
        rms_error = np.sqrt(np.mean((phi_vals - target_vals)**2))
        
        # Normalized RMS (vs target amplitude)
        target_amp = np.max(np.abs(target_vals))
        norm_rms = rms_error / target_amp if target_amp > 0 else np.nan
        
        results[k] = {
            'amplitude_ratio': amplitude_ratio,
            'rms_error': rms_error,
            'norm_rms': norm_rms,
            'phi_vals': phi_vals,
            'target_vals': target_vals,
        }
    
    return results


def main():
    print("=" * 80)
    print("VALIDATION: Does Phi_nu_l match paper's X_k^ell for ell=0?")
    print("=" * 80)
    print()
    print("Target: X_k^0(rho) = sin(k*rho) / sinh(rho)")
    print("        X_k^0(rho) * sinh(rho) = sin(k*rho)  [amplitude = 1]")
    print()
    
    # Test on moderate to large rho (where asymptotic form applies)
    rho_grid = np.linspace(3.0, 15.0, 200)
    k_values = [1.0, 2.0, 5.0, 10.0, 20.0]
    
    print("Testing on rho ∈ [3, 15] (asymptotic regime)")
    print()
    
    results = test_ell0_closed_form(k_values, rho_grid)
    
    print("-" * 80)
    print(f"{'k':<8} {'Amplitude Ratio':<20} {'Norm. RMS Error':<20} {'Status':<12}")
    print("-" * 80)
    
    all_match = True
    
    for k in k_values:
        res = results[k]
        amp_ratio = res['amplitude_ratio']
        norm_rms = res['norm_rms']
        
        # Check if it matches (amplitude ratio should be ~1, norm RMS small)
        matches = (0.95 <= amp_ratio <= 1.05) and (norm_rms < 0.05)
        status = "✅ MATCH" if matches else "❌ MISMATCH"
        
        if not matches:
            all_match = False
        
        print(f"{k:<8.1f} {amp_ratio:<20.6f} {norm_rms:<20.6f} {status:<12}")
    
    print("-" * 80)
    print()
    
    if all_match:
        print("✅ SUCCESS: Phi_nu_l(k, 0, rho) matches target X_k^0(rho) = sin(k*rho)/sinh(rho)")
        print("   No renormalization needed - Phi_nu_l IS the paper's X_k^ell!")
    else:
        print("❌ MISMATCH: Phi_nu_l does NOT match paper's X_k^ell")
        print()
        print("Detailed analysis:")
        for k in k_values:
            res = results[k]
            amp_ratio = res['amplitude_ratio']
            if not (0.95 <= amp_ratio <= 1.05):
                print(f"  k={k:.1f}: amplitude ratio = {amp_ratio:.6f}")
                print(f"    Expected: ~1.0 (since target is sin(k*rho))")
                print(f"    Scaling factor needed: {1.0/amp_ratio:.6f}")
    
    # Plot for visual inspection
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values[:6]):
        if idx >= len(k_values):
            axes[idx].axis('off')
            continue
        
        res = results[k]
        
        axes[idx].plot(rho_grid, res['target_vals'], 'k-', label='Target: sin(k·ρ)', linewidth=2)
        axes[idx].plot(rho_grid, res['phi_vals'], 'r--', label='Phi_nu_l·sinh(ρ)', linewidth=1.5)
        axes[idx].set_xlabel('ρ')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_title(f'k={k:.1f}: Amp ratio = {res["amplitude_ratio"]:.4f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    if len(k_values) < 6:
        for idx in range(len(k_values), 6):
            axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'output_values_local' / 'ell0_validation.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print()
    print(f"Plot saved to: {output_path}")
    print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()
