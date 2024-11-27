import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from scipy.linalg import svd
from joblib import Parallel, delayed
import numpy as np
import tqdm
import time
from utils.sys_generation import construct_numeric_matrix

# Function to solve the system using SVD and compute chi^2, with timing
def solve_system_via_svd_numeric(A):
    start_time = time.time()  # Start timing

    # Perform Singular Value Decomposition
    U, s, Vt = svd(A, full_matrices=False)

    # Iterate through all singular vectors (rows of Vt)
    chi_squared_values = []
    vectors = []

    for vec in Vt:
        chi_squared = np.linalg.norm(A @ vec) ** 2  # Compute chi^2
        chi_squared_values.append(chi_squared)
        vectors.append(vec)

    # Find the indices of the 3 smallest chi^2 values
    sorted_indices = np.argsort(chi_squared_values)[:3]

    # Extract the top 3 chi^2 values and corresponding vectors
    top_chi_squared_values = [chi_squared_values[idx] for idx in sorted_indices]
    top_vectors = [vectors[idx] for idx in sorted_indices]

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    #print(f"SVD computation completed in {elapsed_time:.4f} seconds.")

    return top_chi_squared_values, top_vectors

import matplotlib.pyplot as plt
from datetime import datetime
import os

def plot_chi_squared_spectrum(
    k_values,
    chi_squared_values_list,
    manifold_name,
    resolution,
    output_dir='output_plots',
    num_best_to_show=3,
    dpi=600  # Set a high DPI for detailed images
):
    """
    Plots the chi-squared spectrum for the best singular vectors, allowing the user to
    choose how many best values to display.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))

    # Plot the chi-squared spectra for the requested number of best values
    for i in range(num_best_to_show):
        if i < len(chi_squared_values_list):
            plt.plot(k_values, chi_squared_values_list[i], label=f'Best χ²(k), Rank {i+1}', linestyle='-', linewidth=1.5)

    # Labels and title
    plt.xlabel('k', fontsize=14)
    plt.ylabel('χ²', fontsize=14)
    plt.title(f'χ² Spectrum for {manifold_name}, with Resolution = {resolution}', fontsize=16)

    plt.grid(True)
    plt.legend()

    # Get current date and time for timestamped file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the plot with the timestamp in the filename
    filename = os.path.join(output_dir, f'chi_squared_spectrum_{manifold_name}_{resolution}res_{timestamp}.png')
    plt.savefig(filename, dpi=dpi)  # Save plot as a high-quality PNG image
    print(f"Plot saved as {filename}")


# Function to compute chi^2 for a given k, with progress bar update
def compute_chi_squared_for_k(k_value, matrix_system):
    A = construct_numeric_matrix(matrix_system, k_value)
    chi_squared, _ = solve_system_via_svd_numeric(A)
    return chi_squared

# Compute the chi^2 spectrum for a range of k values in parallel, with progress bars
def compute_chi_squared_spectrum_parallel(matrix_system, M, N, k_values):
    print("Starting parallel computation of the chi-squared spectrum...")

    start_time = time.time()  # Start timing the entire process

    # Use tqdm to track the progress of chi-squared computation for each k value
    chi_squared_values = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_chi_squared_for_k)(k_val, matrix_system) for k_val in tqdm.tqdm(k_values, desc="Computing Chi-Squared Spectrum")
    )

    end_time = time.time()  # End timing the entire process
    total_elapsed_time = end_time - start_time

    print(f"Total chi-squared spectrum computation completed in {total_elapsed_time:.2f} seconds.")  # Report total timing

    return chi_squared_values