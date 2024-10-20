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

# Helper function to compute the difference between two Q_k_lm values
def process_image_pair(alpha, beta, images, q_values):
    rho_alpha, theta_alpha, phi_alpha = images[alpha]
    rho_beta, theta_beta, phi_beta = images[beta]

    Q_alpha = q_values[(rho_alpha, theta_alpha, phi_alpha)]
    Q_beta = q_values[(rho_beta, theta_beta, phi_beta)]

    return Q_alpha - Q_beta

def compute_column(l, m, k_value, points_images, q_values, n_jobs=1):
    """Compute a single column of the matrix system."""
    column = []

    for images in points_images:
        n_j = len(images)
        pairs = [(alpha, beta) for alpha in range(n_j) for beta in range(alpha + 1, n_j)]
        tasks = [(alpha, beta, images, q_values) for alpha, beta in pairs]
        # Process serially or with limited parallelism to avoid over-parallelization
        if n_jobs > 1 and len(tasks) > 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_image_pair)(alpha, beta, images, q_values) for alpha, beta in pairs
            )
        else:
            results = [process_image_pair(alpha, beta, images, q_values) for alpha, beta in pairs]
        column.extend(results)

    return column

def generate_matrix_system(points_images, L, k_value, valid_points):
    k_value = float(k_value)
    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]

    q_values = parallel_Q_k_lm_compute(lm_pairs, k_value, points_images)

    columns = []
    for l, m in lm_pairs:
        column = compute_column(l, m, k_value, points_images, q_values, n_jobs=1)
        columns.append(column)

    matrix_system = list(map(list, zip(*columns)))
    N_calculated = (L + 1) ** 2
    return len(matrix_system), N_calculated, matrix_system

def construct_numeric_matrix(matrix_system, k_value):
    M = len(matrix_system)
    N = len(matrix_system[0])
    A = np.zeros((M, N), dtype=complex)

    for i in range(M):
        for j in range(N):
            entry = matrix_system[i][j]
            A[i, j] = complex(entry)  # Ensure entry is converted to complex

    return A