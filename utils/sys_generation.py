from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm  # Import for progress bars
from utils.special_functions import parallel_Q_k_lm_compute
import mpmath as mp
import sympy as sp
import numpy as np
from joblib import Parallel, delayed

# Ensure picklability for special numeric types
def ensure_picklable(value):
    if isinstance(value, mp.mpf):
        return float(value)
    elif isinstance(value, mp.mpc):
        return complex(value)
    elif isinstance(value, sp.Basic):
        return float(value.evalf())
    return value

# Parallelized and cache-optimized version of compute_colum

def compute_column(l, m, k_value, points_images, q_values, n_jobs=-1):
    """Optimized compute_column with joblib parallelization."""
    column = []

    # Helper function to compute the difference between two Q_k_lm values
    def process_image_pair(alpha, beta, images):
        rho_alpha, theta_alpha, phi_alpha = images[alpha]
        rho_beta, theta_beta, phi_beta = images[beta]

        Q_alpha = q_values[(rho_alpha, theta_alpha, phi_alpha)]
        Q_beta = q_values[(rho_beta, theta_beta, phi_beta)]

        return Q_alpha - Q_beta

    # Parallelize the column computation with Joblib
    for images in points_images:
        n_j = len(images)
        pairs = [(alpha, beta) for alpha in range(n_j) for beta in range(alpha + 1, n_j)]
        # Parallel processing of each pair with joblib
        results = Parallel(n_jobs=n_jobs)(delayed(process_image_pair)(alpha, beta, images) for alpha, beta in pairs)
        column.extend(results)

    return column

def generate_matrix_system(points_images, L, k_value, valid_points):
    # Convert k_value to float if it's not already to ensure it's picklable
    k_value = float(k_value)

    d = valid_points  # Number of points inside the domain

    # Check and debug structure of points_images
    if not isinstance(points_images, list) or not all(isinstance(img, list) for img in points_images):
        print("Invalid structure detected for points_images.")
        raise ValueError("points_images should be a list of lists of tuples (rho, theta, phi).")

    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]

    # Print progress for matrix generation
    print(f"Generating matrix system for k = {k_value}, L = {L}, with d = {d} points.")

    # Use the parallelized Q_k_lm computation
    q_values = parallel_Q_k_lm_compute(lm_pairs, k_value, points_images)

    # Joblib Parallel for parallel processing with a dynamic progress bar
    columns = Parallel(n_jobs=-1)(
        delayed(compute_column)(l, m, k_value, points_images, q_values)
        for l, m in tqdm(lm_pairs, total=len(lm_pairs), desc="Generating matrix columns", ascii=False)
    )

    print("Matrix generation completed.")

    # Transpose the result to get columns as needed
    matrix_system = list(map(list, zip(*columns)))

    N_calculated = (L + 1) ** 2  # Number of columns
    return len(matrix_system), N_calculated, matrix_system

# Function to construct the numerical matrix for a given value of k
def construct_numeric_matrix(matrix_system, k_value):
    M = len(matrix_system)
    N = len(matrix_system[0])
    A = np.zeros((M, N), dtype=complex)  # Ensure the matrix is complex
    
    # Iterate over rows and columns, converting to Python complex numbers if needed
    for i in range(M):
        for j in range(N):
            entry = matrix_system[i][j]
            if isinstance(entry, mp.mpc):  # If it's an mpc object from mpmath
                A[i, j] = complex(entry.real, entry.imag)  # Convert to Python complex
            else:
                A[i, j] = entry  # Assume already numeric

    print(f"Constructed matrix for k = {k_value}: size = {M} x {N}")
    return A