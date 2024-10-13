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

def precompute_special_functions(precomputed_tiling_data):
    """
    Precompute special function values (Q_k_lm) for each k_value using the precomputed tiling data.

    Parameters:
    - precomputed_tiling_data: Dictionary storing precomputed points_images, valid_points, and L_value for each k_value.

    Returns:
    - precomputed_data: Dictionary storing precomputed q_values for each k_value.
    """
    precomputed_data = {}

    # Iterate over each k_value in the precomputed_tiling_data
    for k_value, data in tqdm(precomputed_tiling_data.items(), desc="Precomputing Special Functions", dynamic_ncols=True):
        points_images = data['points_images']  # Extract points_images for this k_value
        L_value = data['L_value']              # Extract corresponding L_value for this k_value

        # Create list of (l, m) pairs based on L_value
        lm_pairs = [(l, m) for l in range(L_value + 1) for m in range(-l, l + 1)]

        # Compute q_values for this k_value using parallel_Q_k_lm_compute
        q_values = parallel_Q_k_lm_compute(lm_pairs, k_value, points_images)

        # Store the precomputed q_values and points_images for later use
        precomputed_data[k_value] = {
            'q_values': q_values,
            'points_images': points_images,
            'L_value': L_value  # Store L_value for later use
        }

    return precomputed_data

# Parallelized and cache-optimized version of compute_colum

def compute_column(l, m, k_value, points_images, q_values, n_jobs=-1):
    """Optimized compute_column with joblib parallelization."""
    column = []

    # Helper function to compute the difference between two Q_k_lm values
    def process_image_pair(alpha, beta, images, q_values_for_point):
        rho_alpha, theta_alpha, phi_alpha = images[alpha]
        rho_beta, theta_beta, phi_beta = images[beta]

        Q_alpha = q_values_for_point[(rho_alpha, theta_alpha, phi_alpha)][(l, m)]
        Q_beta = q_values_for_point[(rho_beta, theta_beta, phi_beta)][(l, m)]

        return Q_alpha - Q_beta

    # Parallelize the column computation with Joblib
    for original_idx, images in enumerate(points_images):
        n_j = len(images)
        pairs = [(alpha, beta) for alpha in range(n_j) for beta in range(alpha + 1, n_j)]
        q_values_for_point = q_values[original_idx]  # Access q_values for this point

        # Parallel processing of each pair with joblib
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_image_pair)(alpha, beta, images, q_values_for_point) for alpha, beta in pairs
        )
        column.extend(results)

    return column

def generate_matrix_system(precomputed_data, k_value):
    """
    Generate the matrix system using precomputed q_values and points_images.

    Parameters:
    - precomputed_data: The dictionary with precomputed q_values and points_images.
    - k_value: Current k value.

    Returns:
    - The generated matrix system.
    """
    # Retrieve precomputed q_values and points_images
    q_values = precomputed_data[k_value]['q_values']
    points_images = precomputed_data[k_value]['points_images']
    L_value = precomputed_data[k_value]['L_value']

    valid_points = len(points_images)  # Number of valid points used

    lm_pairs = [(l, m) for l in range(L_value + 1) for m in range(-l, l + 1)]

    # Print progress for matrix generation
    print(f"Generating matrix system for k = {k_value}, L = {L_value}, with {valid_points} points.")

    # Generate columns in parallel using precomputed q_values
    columns = Parallel(n_jobs=-1)(
        delayed(compute_column)(l, m, k_value, points_images, q_values)
        for l, m in tqdm(lm_pairs, total=len(lm_pairs), desc="Generating matrix columns")
    )

    # Transpose the result to get columns as needed
    matrix_system = list(map(list, zip(*columns)))

    N_calculated = (L_value + 1) ** 2  # Number of columns

    # Print matrix dimensions before returning
    print(f"Generated matrix system for k = {k_value}: {len(matrix_system)} rows, {N_calculated} columns")
    
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