from joblib import Parallel, delayed
import numpy as np
from utils.special_functions import parallel_Q_k_lm_compute, Q_k_lm_vectorized
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

def generate_matrix_system(points_images, L, k_value):
    """
    Generate the matrix system using the provided points and their images.

    Parameters:
    - points_images: List of tuples where each tuple contains (original_point, [image_points])
    - L: Max angular momentum
    - k_value: Current k value

    Returns:
    - M: Number of rows in the matrix
    - N: Number of columns in the matrix
    - A: The constructed matrix as a numpy array
    """
    k_value = float(k_value)
    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]
    N = len(lm_pairs)  # Number of columns
    A_rows = []

    for original_point, images in points_images:
        n_j = len(images)
        
        if n_j < 2:
            continue

        images_array = np.array(images)
        alpha_indices, beta_indices = np.triu_indices(n_j, k=1)

        Q_values = Q_k_lm_vectorized(k_value, lm_pairs, images_array)
        Q_alpha = Q_values[alpha_indices]
        Q_beta = Q_values[beta_indices]
        differences = Q_alpha - Q_beta

        A_rows.extend(differences)

    A = np.array(A_rows, dtype=np.complex128)
    M = len(A)
    return M, N, A

def construct_numeric_matrix():
    return None