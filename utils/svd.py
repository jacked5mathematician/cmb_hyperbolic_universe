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
def solve_system_via_svd_numeric(A, n_smallest: int = 3, normalize_rows: bool = False):
    """
    Compute the n_smallest singular values (and corresponding right singular vectors).
    
    Args:
        A: Constraint matrix
        n_smallest: Number of smallest singular values to return
        normalize_rows: If True, apply row L2 normalization (legacy mode).
                       If False (default), use paper-faithful mode without normalization.
    
    The default keeps backward compatibility with the previous top-3 reporting.
    Paper mode (normalize_rows=False) computes chi² = σ² directly from SVD as in equation 2.7.
    Legacy mode (normalize_rows=True) applies row normalization before SVD.
    """
    start_time = time.time()  # Start timing

    A = construct_numeric_matrix(A)
    
    if normalize_rows:
        # Legacy: Row L2 normalization to reduce dominance of any single block
        row_norms = np.linalg.norm(A, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        A = A / row_norms

    # Perform Singular Value Decomposition
    # Left singular vectors are not needed; we still request full SVD to retain Vt
    _, s, Vt = svd(A, full_matrices=False)

    # Smallest singular values (and corresponding vectors)
    order = np.argsort(s)
    take = min(n_smallest, len(s))
    smallest_sigma = s[order[:take]]
    vectors = Vt[order[:take]]

    chi_squared = smallest_sigma ** 2

    # Compute diagnostics
    diagnostics = {
        'sigma_min': float(s.min()) if len(s) > 0 else np.nan,
        'sigma_max': float(s.max()) if len(s) > 0 else np.nan,
        'all_singular_values': s.tolist(),
    }

    end_time = time.time()  # End timing
    return chi_squared.tolist(), vectors, diagnostics

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
