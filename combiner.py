import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

from utils import plot_chi_squared_spectrum, solve_system_via_svd_numeric
from utils.svd import plot_chi_squared_spectrum  # Import the updated plot function

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

    if not all_k_values:
        print("No k_values found in the results. Exiting.")
        return

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

def extract_values_from_log(log_file):
    k_values = []
    chi_squared_values = [[] for _ in range(3)]  # Assuming we are interested in the top 3 chi-squared values

    with open(log_file, 'r') as f:
        for line in f:
            if "k =" in line and "chi_squared =" in line:
                try:
                    parts = line.split(":")
                    k_value = float(parts[1].split(",")[0].strip().split("=")[1])
                    chi_squared_str = parts[2].split(",")[0].strip().split("=")[1].strip("[]")
                    chi_squared_list = [float(x) for x in chi_squared_str.split(",")]
                    k_values.append(k_value)
                    for i, chi in enumerate(chi_squared_list):
                        if i < 3:
                            chi_squared_values[i].append(chi)
                except (IndexError, ValueError) as e:
                    print(f"Error parsing line: {line.strip()} - {e}")

    return k_values, chi_squared_values

def combine_log_results(output_dir='output_values_2', num_best_to_combine=3):
    all_k_values = []
    all_chi_squared = [[] for _ in range(num_best_to_combine)]

    # Gather and load each .log file
    for process_index in range(100):  # Assuming process indices range from 0 to 99
        log_file = os.path.join(output_dir, f"process_{process_index}.log")
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} does not exist.")
            continue

        k_values, chi_squared_values = extract_values_from_log(log_file)
        all_k_values.extend(k_values)
        for i in range(num_best_to_combine):
            all_chi_squared[i].extend(chi_squared_values[i])

    if not all_k_values:
        print("No k_values found in the log files. Exiting.")
        return

    # Sort the results by k_values
    sorted_indices = np.argsort(all_k_values)
    k_values_sorted = np.array(all_k_values)[sorted_indices]
    chi_squared_sorted = [np.array(chi)[sorted_indices] for chi in all_chi_squared]

    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_file = os.path.join(output_dir, f"combined_results_from_logs_{timestamp}.npz")
    np.savez(
        combined_file,
        k_values=k_values_sorted,
        **{f'chi_squared_rank_{i+1}': chi_sorted for i, chi_sorted in enumerate(chi_squared_sorted)}
    )

    # Plot the chi-squared spectrum
    plot_chi_squared_spectrum(
        k_values_sorted,
        chi_squared_sorted,
        "combined_logs",
        len(k_values_sorted),
        num_best_to_show=num_best_to_combine
    )
    print(f"Combined results saved to {combined_file}")

if __name__ == "__main__":
    # Read the configuration from the JSON file
    with open('output_values/config.json', 'r') as f:
        config = json.load(f)

    total_chunks = config["total_chunks"]
    resolution = config["resolution"]
    manifold_name = config.get("manifold_name", "unknown_manifold")

    # Ensure the output directories exist
    if not os.path.exists('output_matrices'):
        os.makedirs('output_matrices')
    if not os.path.exists('output_values'):
        os.makedirs('output_values')
    if not os.path.exists('output_plots'):
        os.makedirs('output_plots')

    # Call combine_results to process the data and plot chi-squared values
    '''
    combine_results(
        total_chunks=total_chunks,
        resolution=resolution,
        output_dir='output_values',
        manifold_name=manifold_name,
        num_best_to_combine=3
    )
    '''
    # Ensure the output directories exist
    if not os.path.exists('output_values_2'):
        os.makedirs('output_values_2')
    if not os.path.exists('output_plots'):
        os.makedirs('output_plots')

    # Call combine_log_results to process the log files and plot chi-squared values
    combine_log_results(output_dir='output_values_2', num_best_to_combine=3)