from joblib import Parallel, delayed
import numpy as np
from utils.special_functions import parallel_Q_k_lm_compute
import mpmath as mp

# Ensure picklability for special numeric types
def ensure_picklable(value):
    if isinstance(value, mp.mpf):
        return float(value)
    elif isinstance(value, mp.mpc):
        return complex(value)
    elif isinstance(value, sp.Basic):
        return float(value.evalf())
    return value

def compute_column(l, m, k_value, points_images, q_values, n_jobs=1):
    """Optimized compute_column with joblib parallelization."""
    column = []

    # Helper function to compute the difference between two Q_k_lm values
    def process_image_pair(alpha, beta, images):
        rho_alpha, theta_alpha, phi_alpha = images[alpha]
        rho_beta, theta_beta, phi_beta = images[beta]

        Q_alpha = q_values[(rho_alpha, theta_alpha, phi_alpha)]
        Q_beta = q_values[(rho_beta, theta_beta, phi_beta)]

        return Q_alpha - Q_beta

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

    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]

    # Use the parallelized Q_k_lm computation
    q_values = parallel_Q_k_lm_compute(lm_pairs, k_value, points_images)

    # Generate matrix columns without redundant progress bars
    columns = Parallel(n_jobs=1)(
        delayed(compute_column)(l, m, k_value, points_images, q_values)
        for l, m in lm_pairs
    )

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

    return A