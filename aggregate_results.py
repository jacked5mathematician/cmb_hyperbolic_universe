# aggregate_results.py
import numpy as np
from utils import plot_chi_squared_spectrum

def aggregate_results(num_chunks):
    k_values_total = []
    chi_squared_total = []

    for i in range(num_chunks):
        filename = f"chi_squared_results_chunk_{i}.npz"
        data = np.load(filename)
        k_values_total.extend(data['k_values'])
        chi_squared_total.extend(data['chi_squared'])

    # Sort the results by k_values
    sorted_indices = np.argsort(k_values_total)
    k_values_sorted = np.array(k_values_total)[sorted_indices]
    chi_squared_values_sorted = np.array(chi_squared_total)[sorted_indices]

    # Save aggregated results
    np.savez("chi_squared_results_aggregated.npz", k_values=k_values_sorted, chi_squared=chi_squared_values_sorted)
    print("Aggregated results saved to chi_squared_results_aggregated.npz")

    # Optionally, plot the chi-squared spectrum
    manifold_name = 'm188(-1,1)'  # Use the same manifold name
    resolution = len(k_values_sorted)
    plot_chi_squared_spectrum(k_values_sorted, chi_squared_values_sorted, manifold_name, resolution)

if __name__ == "__main__":
    aggregate_results(num_chunks=16)