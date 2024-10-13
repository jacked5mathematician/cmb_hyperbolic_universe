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

    # Use full_matrices=False to avoid computing unnecessary large U and Vt
    U, s, Vt = svd(A, full_matrices=False)
    a = Vt[-1]  # The singular vector corresponding to the smallest singular value

    # Calculate chi^2 = |A * a|^2 efficiently using norm
    chi_squared = np.linalg.norm(A @ a) ** 2

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"SVD computation completed in {elapsed_time:.4f} seconds.")  # Report timing

    return chi_squared, a

import matplotlib.pyplot as plt
from matplotlib import rc

# Enable LaTeX for rendering text in the plot
rc('text', usetex=True)

def plot_chi_squared_spectrum(k_values, chi_squared_values, manifold_name, resolution):
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, chi_squared_values, label=r'$\chi^2(k)$ Spectrum', color='blue')
    
    # LaTeX for axis labels
    plt.xlabel(r'$k$', fontsize=14)
    plt.ylabel(r'$\chi^2$', fontsize=14)
    
    # LaTeX for the title with manifold name
    title_str = r'$\chi^2$ Spectrum for {}, with Resolution = {}'.format(manifold_name, resolution)
    plt.title(title_str, fontsize=16)
    
    plt.grid(True)
    plt.legend()
    
    # Instead of plt.show(), save the plot to a file
    plt.savefig(f'chi_squared_spectrum_{manifold_name}_{resolution}res.png')  # Save plot as a PNG image
    print(f"Process saved as chi_squared_spectrum_{manifold_name}_{resolution}res.png")

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