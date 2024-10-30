import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Import your plot function here
from utils import plot_chi_squared_spectrum

def combine_results(total_chunks, resolution, output_dir='output_values', manifold_name='m188(-1,1)'):
    all_k_values = []
    all_chi_squared_best = []
    all_chi_squared_second_best = []
    all_chi_squared_third_best = []

    # Gather and load each .npz file
    for chunk_index in range(total_chunks):
        file_path = os.path.join(output_dir, f"results_chunk_{chunk_index}.npz")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist.")
            continue
        data = np.load(file_path)
        
        # Extract k_values and chi_squared values for each type
        k_values = data['k_values']
        chi_squared_best = data['chi_squared_best']
        chi_squared_second_best = data['chi_squared_second_best']
        chi_squared_third_best = data['chi_squared_third_best']

        # Ensure length match between k_values and each chi_squared array
        if not (len(k_values) == len(chi_squared_best) == len(chi_squared_second_best) == len(chi_squared_third_best)):
            print(f"Error: Length mismatch in {file_path}. Arrays are not aligned.")
            continue

        all_k_values.extend(k_values)
        all_chi_squared_best.extend(chi_squared_best)
        all_chi_squared_second_best.extend(chi_squared_second_best)
        all_chi_squared_third_best.extend(chi_squared_third_best)

    # Sort the results by k_values
    sorted_indices = np.argsort(all_k_values)
    k_values_sorted = np.array(all_k_values)[sorted_indices]
    chi_squared_best_sorted = np.array(all_chi_squared_best)[sorted_indices]
    chi_squared_second_best_sorted = np.array(all_chi_squared_second_best)[sorted_indices]
    chi_squared_third_best_sorted = np.array(all_chi_squared_third_best)[sorted_indices]

    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_file = os.path.join(output_dir, f"combined_results_{manifold_name}_{resolution}res_{timestamp}.npz")
    np.savez(
        combined_file,
        k_values=k_values_sorted,
        chi_squared_best=chi_squared_best_sorted,
        chi_squared_second_best=chi_squared_second_best_sorted,
        chi_squared_third_best=chi_squared_third_best_sorted
    )

    # Plot the chi-squared spectrum
    plot_chi_squared_spectrum(
        k_values_sorted,
        chi_squared_best_sorted,
        manifold_name,
        resolution,
        chi_squared_second_best=chi_squared_second_best_sorted,
        chi_squared_third_best=chi_squared_third_best_sorted,
        show_second_best=True,
        show_third_best=True
    )
    print(f"Combined results saved to {combined_file}")

if __name__ == "__main__":
    # Read the configuration from the JSON file
    with open('output_values/config.json', 'r') as f:
        config = json.load(f)

    total_chunks = config["total_chunks"]
    resolution = config["resolution"]

    # Run combination and plotting
    combine_results(total_chunks=total_chunks, resolution=resolution)