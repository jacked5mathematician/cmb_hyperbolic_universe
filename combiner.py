import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

from utils import plot_chi_squared_spectrum, solve_system_via_svd_numeric

def sanity_check_matrices(total_chunks, matrices_dir):
    """Sanity check to ensure all expected matrices exist and are valid."""
    for chunk_index in range(total_chunks):
        matrix_files = [f for f in os.listdir(matrices_dir) if f.startswith(f'matrix_chunk_{chunk_index}_k_')]
        if not matrix_files:
            print(f"[Warning] No matrix files found for chunk {chunk_index} in {matrices_dir}.")
            continue
        
        for matrix_file in matrix_files:
            matrix_path = os.path.join(matrices_dir, matrix_file)
            try:
                matrix_data = np.load(matrix_path)
                if 'A' not in matrix_data or matrix_data['A'].size == 0:
                    print(f"[Error] Matrix in {matrix_file} is missing or empty.")
            except Exception as e:
                print(f"[Error] Failed to load matrix {matrix_file}: {e}")

def recompute_svd_and_update_results(
    total_chunks,
    output_dir='output_values',
    matrices_dir='output_matrices',
    num_best_to_compute=5  # Number of chi-squared values to compute
):
    for chunk_index in range(total_chunks):
        # Paths to the result and matrix files
        result_file = os.path.join(output_dir, f"results_chunk_{chunk_index}.npz")
        matrix_files = [f for f in os.listdir(matrices_dir) if f.startswith(f'matrix_chunk_{chunk_index}_k_')]
        if not matrix_files:
            print(f"No matrix files found for chunk {chunk_index} in {matrices_dir}")
            continue

        # Load existing results if any
        if os.path.exists(result_file):
            data = np.load(result_file)
            k_values = data['k_values']
        else:
            print(f"Result file {result_file} does not exist. Proceeding to create a new one.")
            k_values = []

        # Prepare lists to store new chi-squared values
        k_values_processed = []
        chi_squared_all = [[] for _ in range(num_best_to_compute)]

        # Process each matrix file
        for matrix_file in matrix_files:
            matrix_path = os.path.join(matrices_dir, matrix_file)

            # Extract k_value from filename
            try:
                k_value_str = matrix_file.split('_k_')[1].replace('.npz', '')
                k_value = float(k_value_str)
            except ValueError:
                print(f"Could not extract k_value from filename {matrix_file}")
                continue

            # Load the matrix A
            matrix_data = np.load(matrix_path)
            A = matrix_data['A']

            # Recompute SVD and chi-squared values
            chi_squared_values, _ = solve_system_via_svd_numeric(A)
            if len(chi_squared_values) < num_best_to_compute:
                print(f"Warning: Less chi-squared values computed than requested ({len(chi_squared_values)} vs {num_best_to_compute}).")

            # Append the results
            k_values_processed.append(k_value)
            for i in range(min(len(chi_squared_values), num_best_to_compute)):
                chi_squared_all[i].append(chi_squared_values[i])

            print(f"Chunk {chunk_index}, k={k_value}: Recomputed chi-squared values.")

        # Sort the results by k_values
        sorted_indices = np.argsort(k_values_processed)
        k_values_sorted = np.array(k_values_processed)[sorted_indices]
        chi_squared_sorted = [np.array(chi)[sorted_indices] for chi in chi_squared_all]

        # Save updated results back to the result file
        np.savez(
            result_file,
            k_values=k_values_sorted,
            **{f'chi_squared_rank_{i+1}': chi_sorted for i, chi_sorted in enumerate(chi_squared_sorted)}
        )

        print(f"Updated results saved to {result_file}")

def combine_results(
    total_chunks,
    resolution,
    output_dir='output_values',
    manifold_name='m188(-1,1)',
    num_best_to_combine=3  # Number of chi-squared ranks to combine
):
    all_k_values = []
    all_chi_squared = [[] for _ in range(num_best_to_combine)]

    # Gather and load each .npz file
    for chunk_index in range(total_chunks):
        file_path = os.path.join(output_dir, f"results_chunk_{chunk_index}.npz")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist.")
            continue

        data = np.load(file_path)
        k_values = data['k_values']
        all_k_values.extend(k_values)

        # Extract chi-squared values for each requested rank
        for i in range(num_best_to_combine):
            key = f'chi_squared_rank_{i+1}'
            if key in data:
                all_chi_squared[i].extend(data[key])
            else:
                print(f"Warning: {key} not found in {file_path}.")

    # Sort the results by k_values
    sorted_indices = np.argsort(all_k_values)
    k_values_sorted = np.array(all_k_values)[sorted_indices]
    chi_squared_sorted = [np.array(chi)[sorted_indices] for chi in all_chi_squared]

    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_file = os.path.join(output_dir, f"combined_results_{manifold_name}_{resolution}res_{timestamp}.npz")
    np.savez(
        combined_file,
        k_values=k_values_sorted,
        **{f'chi_squared_rank_{i+1}': chi_sorted for i, chi_sorted in enumerate(chi_squared_sorted)}
    )

    # Plot the chi-squared spectrum
    plot_chi_squared_spectrum(
        k_values_sorted,
        chi_squared_sorted,
        manifold_name,
        resolution,
        num_best_to_show=num_best_to_combine
    )
    print(f"Combined results saved to {combined_file}")

if __name__ == "__main__":
    # Read the configuration from the JSON file
    with open('output_values/config.json', 'r') as f:
        config = json.load(f)

    total_chunks = config["total_chunks"]
    resolution = config["resolution"]
    
    print("[Info] Starting sanity check for matrices.")
    sanity_check_matrices(total_chunks=total_chunks, matrices_dir='output_matrices')

    #recompute_svd_and_update_results(total_chunks=total_chunks)

    #combine_results(total_chunks=total_chunks, resolution=resolution)