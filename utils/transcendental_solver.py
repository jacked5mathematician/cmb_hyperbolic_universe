import numpy as np
import matplotlib.pyplot as plt
from mpmath import hyp2f1, gamma, sqrt, sinh, cosh, pi
from scipy.optimize import minimize_scalar
from utils.special_functions import Phi_nu_l

# Transcendental equation using the real part of Phi_nu_l
def transcendental_eq(chi, k, ell):
    """Equation to find the root for rho_max given k and ell using the real part of Phi_nu_l."""
    nu = np.sqrt(k**2 + 1)
    return float(Phi_nu_l(nu, ell, chi).real)

# Approximation function for large rho
def approx_function(rho, k, phi_0):
    """Approximation function: cos(k*rho + phi_0)/sinh(rho)."""
    return np.cos(k * rho + phi_0) / np.sinh(rho)

# Derivative of Phi_nu_l with respect to rho (numerical)
def derivative_phi_nu_l(chi, k, ell, delta=1e-5):
    """Numerically compute the derivative of Phi_nu_l with respect to chi."""
    return (transcendental_eq(chi + delta, k, ell) - transcendental_eq(chi - delta, k, ell)) / (2 * delta)

# Derivative of the approximation function with respect to rho
def derivative_approx_function(rho, k, phi_0):
    """Analytical derivative of the approximation function."""
    return -k * np.sin(k * rho + phi_0) / np.sinh(rho) - np.cos(k * rho + phi_0) * np.cosh(rho) / np.sinh(rho)**2

# Search for the transition point rho_0 where the function values and derivatives match
def find_rho0(k, ell, rho_min=10, rho_max=20, step=0.1):
    """
    Find rho_0 where Phi_nu_l and the approximation function values are close.
    
    Parameters:
    - k: The wavenumber.
    - ell: The angular momentum quantum number.
    - rho_min: The minimum rho value for the search.
    - rho_max: The maximum rho value for the search.
    - step: The step size for scanning rho values.
    
    Returns:
    - rho_0: The rho value where the functions are similar.
    """
    for rho in np.arange(rho_min, rho_max, step):
        phi_nu_l_value = transcendental_eq(rho, k, ell)
        approx_value = approx_function(rho, k, 0)  # Initial phi_0 = 0 guess
        if np.isclose(phi_nu_l_value, approx_value, atol=0.1):
            return rho
    return rho_max  # If no match is found, return the upper bound

# Find the optimal phi_0 to match the derivative at rho_0
def find_phi0(rho_0, k, ell):
    """
    Find the optimal phi_0 to match the derivatives at rho_0.
    
    Parameters:
    - rho_0: The transition point where the values match.
    - k: The wavenumber.
    - ell: The angular momentum quantum number.
    
    Returns:
    - phi_0: The phase shift that aligns the derivative of the approximation with Phi_nu_l.
    """
    # Define an objective function to minimize the difference in derivatives
    def derivative_error(phi_0):
        exact_derivative = derivative_phi_nu_l(rho_0, k, ell)
        approx_derivative = derivative_approx_function(rho_0, k, phi_0)
        return np.abs(exact_derivative - approx_derivative)

    # Use a scalar minimizer to find phi_0 that minimizes the derivative error
    result = minimize_scalar(derivative_error, bounds=(-np.pi, np.pi), method='bounded')
    return result.x

# Plotting function to visualize the results
def plot_comparison_with_matching_derivatives(k, ell, rho_min=0.1, rho_max=20):
    """
    Plot the real part of Phi_nu_l and the matched approximation function.
    
    Parameters:
    - k: The wavenumber.
    - ell: The angular momentum quantum number.
    - rho_min: The minimum rho value for the plot.
    - rho_max: The maximum rho value for the plot.
    """
    # Find rho_0 where the functions match
    rho_0 = find_rho0(k, ell)
    phi_0_opt = find_phi0(rho_0, k, ell)

    # Define a range for rho (chi)
    rho_values = np.linspace(rho_min, rho_max, 500)

    # Compute the exact and approximate function values
    exact_values = [transcendental_eq(rho, k, ell) for rho in rho_values]
    approx_values = [approx_function(rho, k, phi_0_opt) for rho in rho_values]

    # Plot both Phi_nu_l and the matched approximation
    plt.figure(figsize=(10, 6))
    plt.plot(rho_values, exact_values, label=f"$\\Phi_{{\\nu,\\ell}}(\\rho)$", color='blue')
    plt.plot(rho_values, approx_values, label=f"Matched Approximation", linestyle='dashed', color='red')
    plt.axvline(x=rho_0, color='green', linestyle=':', label=f"Matched $\\rho_0 = {rho_0:.2f}$")
    plt.xlabel("$\\rho$")
    plt.ylabel("Function Value")
    plt.title("Comparison of $\\Phi_{\\nu,\\ell}(\\rho)$ and the Matched Approximation")
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Matched rho_0: {rho_0:.6f}")
    print(f"Matched phi_0: {phi_0_opt:.6f}")

if __name__ == "__main__":
    # Define parameters for testing
    k = 4  # Example value for k
    ell = 5  # Example value for ell

    # Call the plot function to visualize the results
    plot_comparison_with_matching_derivatives(k, ell)