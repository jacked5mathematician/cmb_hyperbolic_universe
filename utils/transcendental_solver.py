import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpmath import hyp2f1, gamma, sqrt, sinh, cosh, pi
import mpmath as mp
from utils.special_functions import Phi_nu_l

# Transcendental equation using the real part of Phi_nu_l
def transcendental_eq(chi, k, ell):
    """Equation to find the root for rho_max given k and ell using the real part of Phi_nu_l."""
    nu = np.sqrt(k**2 + 1)
    # Take the real part of Phi_nu_l and subtract 0.25 for the transcendental equation
    return float(Phi_nu_l(nu, ell, chi).real * np.sinh(chi)) - 0.25

# Separate function to search for interval and find rho (root)
def find_rho_solution(k, ell, lower=0.1, upper=50, step=0.1):
    """
    Find the first rho where the transcendental equation X^l_k(rho) * sinh(rho) = 0.25 holds.
    This function searches for an interval where the equation crosses 0.25 and then uses
    a root-finding method (brentq) to locate the root.

    Parameters:
    - k: The wavenumber.
    - ell: The angular momentum quantum number.
    - lower: The lower bound for rho search.
    - upper: The upper bound for rho search.
    - step: The step size to search for interval crossings.

    Returns:
    - rho: The first rho where the equation holds.
    """

    # Search for an interval where the transcendental_eq crosses zero
    def find_interval():
        for chi in np.arange(lower, upper, step):
            f_lower = transcendental_eq(chi, k, ell)
            f_upper = transcendental_eq(chi + step, k, ell)
            # Check if the function crosses zero between chi and chi + step
            if f_lower * f_upper < 0:
                return chi, chi + step
        return None, None  # No crossing found within the range

    # Find the interval
    lower_bound, upper_bound = find_interval()
    
    if lower_bound is not None and upper_bound is not None:
        # Use Brent's method to find the root in the identified interval
        rho = scipy.optimize.brentq(transcendental_eq, lower_bound, upper_bound, args=(k, ell))
        return rho
    else:
        return np.nan  # No root found within the given range

# New function to handle graphing
def plot_transcendental_eq(k, l_min, l_max, rho_min, rho_max):
    """
    Plot the transcendental equation for given k, l_min, and l_max, and show rho_min and rho_max.
    
    Parameters:
    - k: The wavenumber.
    - l_min: The lower angular momentum value.
    - l_max: The higher angular momentum value.
    - rho_min: The first root for l_min (to be displayed on the plot).
    - rho_max: The first root for l_max (to be displayed on the plot).
    """
    # Define a range for rho (chi)
    rho_values = np.linspace(0.1, 20, 500)

    # Compute the transcendental equation values for l_min and l_max
    transcendental_min = [transcendental_eq(rho, k, l_min) + 0.25 for rho in rho_values]  # Add 0.25 for visualization
    transcendental_max = [transcendental_eq(rho, k, l_max) + 0.25 for rho in rho_values]  # Add 0.25 for visualization

    # Plot the transcendental equation for l_min and l_max
    plt.figure(figsize=(10, 6))
    plt.plot(rho_values, transcendental_min, label=f"$l_{{min}} = {l_min}$", color='blue')
    plt.plot(rho_values, transcendental_max, label=f"$l_{{max}} = {l_max}$", color='red')
    plt.axhline(0.25, color='green', linestyle='--', label="$0.25$")
    if not np.isnan(rho_min):
        plt.axvline(rho_min, color='blue', linestyle='--', label=f"$\\rho_{{min}} = {rho_min:.2f}$")
    if not np.isnan(rho_max):
        plt.axvline(rho_max, color='red', linestyle='--', label=f"$\\rho_{{max}} = {rho_max:.2f}$")
    plt.xlabel("$\\rho$")
    plt.ylabel("$X^{\\ell}_k(\\rho) \\times \\sinh(\\rho)$")
    plt.title("Solution to the transcendental equation for different $l$ values")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Define parameters for testing
    k = 4  # Example value for k
    l_min = 5  # Example value for l_min
    l_max = 15  # Example value for l_max

    # Use the function to find rho_min for l_min and rho_max for l_max
    rho_min = find_rho_solution(k, l_min)
    rho_max = find_rho_solution(k, l_max)

    print(f"The first rho_min for l_min={l_min} is: {rho_min}")
    print(f"The first rho_max for l_max={l_max} is: {rho_max}")

    # Call the plot function to visualize the results
    plot_transcendental_eq(k, l_min, l_max, rho_min, rho_max)

# Function to create heatmap of rho_min or rho_max
def create_heatmap(k_values, l_values, mode="min"):
    """
    Create a heatmap of rho_min or rho_max values over a range of k and ell values.
    
    Parameters:
    - k_values: Array of k values.
    - l_values: Array of ell values.
    - mode: "min" for rho_min or "max" for rho_max.
    
    Returns:
    - A heatmap of rho_min or rho_max values.
    """
    heatmap_data = np.zeros((len(l_values), len(k_values)))

    # Loop through the grid of k and ell values
    for i, l in enumerate(l_values):
        for j, k in enumerate(k_values):
            # Find rho_min or rho_max depending on the mode
            if mode == "min":
                rho = find_rho_solution(k, l)
            else:
                rho = find_rho_solution(k, l)
            
            heatmap_data[i, j] = rho
    
    return heatmap_data

# Set parameters for the heatmap
#k_values = np.linspace(1, 10, 20)  # Range of k values
#l_values = np.arange(1, 20,1)  # Range of ell values

# Create heatmaps for rho_min and rho_max
#rho_min_heatmap = create_heatmap(k_values, l_values, mode="min")

# Plotting heatmap
def plot_heatmap(heatmap_data, k_values, l_values, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap_data, extent=[k_values[0], k_values[-1], l_values[0], l_values[-1]],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label="Rho")
    plt.title(title)
    plt.xlabel("k values")
    plt.ylabel("l values")
    plt.show()

# Plot rho_min heatmap
#plot_heatmap(rho_min_heatmap, k_values, l_values, title="Heatmap of rho_min values for changing k and l")