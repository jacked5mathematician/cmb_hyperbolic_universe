from joblib import Parallel, delayed
import numpy as np
from utils.special_functions import parallel_Q_k_lm_compute
import mpmath as mp

def compute_column(l, m, k_value, points_images, q_values):
    """Vectorized compute_column without explicit loops over image pairs."""
    column = []

    for images in points_images:
        n_j = len(images)
        if n_j < 2:
            continue  # No pairs to process

        # Convert images to numpy arrays
        images_array = np.array(images)  # Shape (n_j, 3)

        # Fetch Q_values for all images in this set
        Q_values_images = np.array([q_values[tuple(img)] for img in images])  # Shape (n_j,)

        # Create indices for pairs where alpha < beta
        alpha_indices, beta_indices = np.triu_indices(n_j, k=1)

        Q_alpha = Q_values_images[alpha_indices]
        Q_beta = Q_values_images[beta_indices]

        differences = Q_alpha - Q_beta  # This is a vector of differences

        column.extend(differences)

    return column

def generate_matrix_system(points_images, L, k_value, valid_points):
    k_value = float(k_value)

    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]

    # Use the parallelized Q_k_lm computation
    q_values = parallel_Q_k_lm_compute(lm_pairs, k_value, points_images)

    # Determine the number of jobs based on your system's capabilities
    n_jobs = 1  # Use all available cores

    # Parallel processing over (l, m) pairs
    columns = Parallel(n_jobs=n_jobs)(
        delayed(compute_column)(l, m, k_value, points_images, q_values)
        for l, m in lm_pairs
    )

    # Transpose the result to get columns as needed
    matrix_system = list(map(list, zip(*columns)))

    N_calculated = (L + 1) ** 2  # Number of columns
    return len(matrix_system), N_calculated, matrix_system

def construct_numeric_matrix(matrix_system, k_value):
    # Convert the matrix_system to a numpy array
    A = np.array(matrix_system)

    # Define a vectorized function to convert entries
    def convert_entry(entry):
        if isinstance(entry, mp.mpc):
            return complex(entry.real, entry.imag)
        elif isinstance(entry, mp.mpf):
            return float(entry)
        else:
            return entry  # Assume already numeric

    # Vectorize the function
    vectorized_convert = np.vectorize(convert_entry)

    # Apply the vectorized conversion
    A = vectorized_convert(A)

    # Ensure the matrix is of type complex (if necessary)
    A = A.astype(np.complex128)

    return A