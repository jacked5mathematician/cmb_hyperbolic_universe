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

    # Get indices of the smallest, second smallest, and third smallest singular values
    sorted_indices = np.argsort(s)  # Sort indices based on singular values in ascending order
    
    # Get the corresponding singular vectors from Vt
    best_a = Vt[sorted_indices[0]]  # Vector for smallest singular value
    second_best_a = Vt[sorted_indices[1]]  # Vector for second smallest singular value
    third_best_a = Vt[sorted_indices[2]]  # Vector for third smallest singular value

    # Normalize the singular vectors
    best_a_normalized = best_a / np.linalg.norm(best_a)
    second_best_a_normalized = second_best_a / np.linalg.norm(second_best_a)
    third_best_a_normalized = third_best_a / np.linalg.norm(third_best_a)

    # Calculate chi^2 for each solution
    chi_squared_best = np.linalg.norm(A @ best_a_normalized) ** 2
    chi_squared_second_best = np.linalg.norm(A @ second_best_a_normalized) ** 2
    chi_squared_third_best = np.linalg.norm(A @ third_best_a_normalized) ** 2
    
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    # print(f"SVD computation completed in {elapsed_time:.4f} seconds.")  # Report timing

    return (chi_squared_best, chi_squared_second_best, chi_squared_third_best), (best_a_normalized, second_best_a_normalized, third_best_a_normalized)

import matplotlib.pyplot as plt
from matplotlib import rc

# Enable LaTeX for rendering text in the plot
rc('text', usetex=True)

import matplotlib.pyplot as plt
from datetime import datetime

def plot_chi_squared_spectrum(k_values, chi_squared_values, manifold_name, resolution, 
                              chi_squared_second_best=None, chi_squared_third_best=None, 
                              show_second_best=False, show_third_best=False):
    plt.figure(figsize=(8, 6))
    
    # Plot the main chi-squared values
    plt.plot(k_values, chi_squared_values, label=r'Best $\chi^2(k)$ Spectrum', color='blue')

    # Optionally plot the second best chi-squared values if requested
    if show_second_best and chi_squared_second_best is not None:
        plt.plot(k_values, chi_squared_second_best, label=r'Second Best $\chi^2(k)$ Spectrum', linestyle='--', color='green')
    
    # Optionally plot the third best chi-squared values if requested
    if show_third_best and chi_squared_third_best is not None:
        plt.plot(k_values, chi_squared_third_best, label=r'Third Best $\chi^2(k)$ Spectrum', linestyle=':', color='red')
    
    # LaTeX for axis labels
    plt.xlabel(r'$k$', fontsize=14)
    plt.ylabel(r'$\chi^2$', fontsize=14)
    
    # LaTeX for the title with manifold name
    plt.title(r'$\chi^2$ Spectrum for {}, with Resolution = {}'.format(manifold_name, resolution), fontsize=16)
    
    plt.grid(True)
    plt.legend()
    
    # Get current date and time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the plot with the timestamp in the filename
    filename = f'chi_squared_spectrum_{manifold_name}_{resolution}res_{timestamp}.png'
    plt.savefig(filename)  # Save plot as a PNG image
    
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