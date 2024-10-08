import numpy as np
import matplotlib.pyplot as plt
from utils.special_functions import Phi_nu_l

def equation_to_solve(rho, k, L):
    """
    The equation X_k^L(rho) * sinh(rho) = 0.25, where X_k^L(rho) is Phi_nu_l(k, L, rho).
    """
    # Compute the Bessel function value and take only the real part
    bessel_value = float(Phi_nu_l(k, L, rho).real)  # Ensure this is a float
    return bessel_value * np.sinh(rho) - 0.25

def plot_equation(k, L, rho_range=(1e-5, 10), num_points=1000):
    """
    Plots the equation X_k^L(rho) * sinh(rho) - 0.25 over a given range of rho values.
    
    Parameters:
    - k: The parameter 'k' in X_k^L(rho).
    - L: The angular momentum parameter 'L'.
    - rho_range: The range of rho values to plot (default is [1e-5, 10]).
    - num_points: Number of points to evaluate the equation at (default is 1000).
    """
    rho_values = np.linspace(rho_range[0], rho_range[1], num_points)
    equation_values = [equation_to_solve(rho, k, L) for rho in rho_values]
    
    # Plot the function
    plt.figure(figsize=(8, 6))
    plt.plot(rho_values, equation_values, label=r'$X_k^L(\rho) \sinh(\rho) - 0.25$')
    plt.axhline(0, color='black', linestyle='--', label='y=0')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$X_k^L(\rho) \sinh(\rho) - 0.25$')
    plt.title(f'Plot of the equation for k={k}, L={L}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == "__main__":
    k = 4.0  # Example k value
    L = 2  # Example L value
    plot_equation(k, L, rho_range=(1e-5, 10), num_points=1000)