"""
Emergency diagnostic script to understand actual radial function behavior.

This script plots the actual X_k^L(rho) * sinh(rho) vs the envelope approximation
to understand why roots don't exist for large k.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils.special_functions import Phi_nu_l
from utils.cutoffs import abs_radial_envelope, rho_turning_point

def actual_radial(k, ell, rho):
    """Actual |X_k^ell(rho) * sinh(rho)|"""
    X = Phi_nu_l(k, ell, rho)
    return float(abs(X * np.sinh(rho)))

# Test cases
test_cases = [(1.0, 11), (2.0, 12), (5.0, 15), (10.0, 20), (20.0, 30)]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (k, L) in enumerate(test_cases):
    ax = axes[idx]
    
    # Generate rho values
    rho_values = np.linspace(0.5, 10.0, 200)
    
    # Evaluate both functions
    actual_vals = [actual_radial(k, L, rho) for rho in rho_values]
    envelope_vals = [abs_radial_envelope(k, L, rho) for rho in rho_values]
    
    # Find turning point
    rho0 = rho_turning_point(k, L)
    
    # Plot
    ax.plot(rho_values, actual_vals, 'b-', linewidth=2, label='Actual $|X_k^L \\sinh\\rho|$')
    ax.plot(rho_values, envelope_vals, 'r--', linewidth=1.5, label='Envelope approx')
    ax.axhline(0.25, color='green', linestyle=':', label='Threshold 0.25')
    ax.axvline(rho0, color='orange', linestyle=':', alpha=0.5, label=f'$\\rho_0$={rho0:.2f}')
    
    ax.set_xlabel('$\\rho$')
    ax.set_ylabel('$|X_k^L \\sinh\\rho|$')
    ax.set_title(f'k={k:.1f}, L={L}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.5])
    
    # Report max value
    max_actual = max(actual_vals)
    print(f"k={k:5.1f}, L={L:2d}: max(actual)={max_actual:.4f}, reaches 0.25? {max_actual >= 0.25}")

# Hide extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('output_values_local/radial_function_diagnostic.png', dpi=150)
print("\nPlot saved to: output_values_local/radial_function_diagnostic.png")
plt.close()

# Now test with varying ell to see if lower ell helps
print("\n" + "="*60)
print("Testing different ell values for k=10:")
print("="*60)

k = 10.0
for ell in [5, 10, 15, 20, 25, 30]:
    rho_values = np.linspace(0.5, 15.0, 300)
    actual_vals = [actual_radial(k, ell, rho) for rho in rho_values]
    max_val = max(actual_vals)
    min_val = min([v for v in actual_vals if v > 0])
    print(f"  ell={ell:2d}: max={max_val:.4f}, min={min_val:.4f}, reaches 0.25? {max_val >= 0.25}")
