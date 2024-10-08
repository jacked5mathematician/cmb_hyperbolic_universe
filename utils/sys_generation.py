from utils.special_functions import Q_k_lm_cached
import mpmath as mp
import sympy as sp
import numpy as np


def ensure_picklable(value):
    """Convert mpmath or sympy types to standard Python types."""
    if isinstance(value, mp.mpf):
        return float(value)
    elif isinstance(value, mp.mpc):
        return complex(value)
    elif isinstance(value, sp.Basic):  # sympy symbolic types
        return float(value.evalf())
    return value
# Construct matrix system for solving coefficients a_{qlm}

def compute_column(l, m, k_value, points_images):
    """Optimized compute_column with caching and pre-computation."""
    column = []
    q_values = {}  # Cache for Q values of unique (rho, theta, phi) tuples

    # Precompute all Q_k_lm values for unique (rho, theta, phi)
    for point_images in points_images:
        for rho, theta, phi in point_images:
            if (rho, theta, phi) not in q_values:
                q_values[(rho, theta, phi)] = Q_k_lm_cached(k_value, l, m, rho, theta, phi)

    for j, images in enumerate(points_images):
        n_j = len(images)
        for alpha in range(n_j):
            for beta in range(alpha + 1, n_j):
                rho_alpha, theta_alpha, phi_alpha = images[alpha]
                rho_beta, theta_beta, phi_beta = images[beta]

                Q_alpha = q_values[(rho_alpha, theta_alpha, phi_alpha)]
                Q_beta = q_values[(rho_beta, theta_beta, phi_beta)]

                column.append(Q_alpha - Q_beta)
    return column

def generate_matrix_system(points_images, L, k_value, valid_points):
    # Convert k_value to float if it's not already to ensure it's picklable
    k_value = ensure_picklable(k_value)

    d = valid_points  # Number of points inside the domain

    # Check and debug structure of points_images
    if not isinstance(points_images, list) or not all(isinstance(img, list) for img in points_images):
        print("Invalid structure detected for points_images.")
        print("Expected structure: List of lists of tuples (rho, theta, phi).")
        print(f"Actual structure: {points_images}")
        # Uncomment below to see more information
        # for i, entry in enumerate(points_images):
        #     print(f"Entry {i}: {entry}")
        raise ValueError("points_images should be a list of lists of tuples (rho, theta, phi).")

    # Print a small sample to check the structure if needed (for debugging)
    # print(points_images[:2])  # Uncomment this line for debugging

    # Define a function to compute each column (l, m)
    def compute_column_wrapper(l, m):
        return compute_column(l, m, k_value, points_images)

    # Use a regular loop to compute the matrix columns without parallel processing
    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]

    # Print debug information before generating matrix system
    print(f"Generating matrix system for k = {k_value}, L = {L}, with d = {d} points.")

    # Sequentially compute each column without parallelism
    columns = []
    for l, m in lm_pairs:
        column = compute_column_wrapper(l, m)
        columns.append(column)

    # Transpose the result to get columns as needed
    matrix_system = list(map(list, zip(*columns)))

    N_calculated = (L + 1)**2  # Number of columns
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

def compute_filtered_column(l, m, k_value, points_images, rho_min, rho_max):
    """Compute column for a given (l, m) while filtering image pairs based on rho_min and rho_max."""
    column = []
    q_values = {}  # Cache for Q values of unique (rho, theta, phi) tuples

    # Precompute all Q_k_lm values for unique (rho, theta, phi) that fall within the (rho_min, rho_max) range
    for point_images in points_images:
        for rho, theta, phi in point_images:
            if rho_min <= rho <= rho_max:  # Filter by rho_min and rho_max
                if (rho, theta, phi) not in q_values:
                    q_values[(rho, theta, phi)] = Q_k_lm_cached(k_value, l, m, rho, theta, phi)

    # Now, only use valid pairs for each point that fall within the range
    for j, images in enumerate(points_images):
        valid_images = [img for img in images if rho_min <= img[0] <= rho_max]  # Filtered list of valid images
        
        n_j = len(valid_images)  # Number of valid images
        for alpha in range(n_j):
            for beta in range(alpha + 1, n_j):
                rho_alpha, theta_alpha, phi_alpha = valid_images[alpha]
                rho_beta, theta_beta, phi_beta = valid_images[beta]

                Q_alpha = q_values[(rho_alpha, theta_alpha, phi_alpha)]
                Q_beta = q_values[(rho_beta, theta_beta, phi_beta)]

                column.append(Q_alpha - Q_beta)

    return column


def generate_filtered_matrix_system(points_images, L, k_value, rho_min, rho_max):
    """
    Generate a filtered matrix system considering only images within the range [rho_min, rho_max].
    
    Parameters:
    - points_images: List of images (in pseudo-spherical coordinates) for each point.
    - L: Angular momentum parameter.
    - k_value: Value of k for the computation.
    - rho_min: Minimum rho value for the images.
    - rho_max: Maximum rho value for the images.

    Returns:
    - M: The number of rows in the matrix system.
    - N_calculated: The number of columns in the matrix system.
    - matrix_system: The filtered matrix system for the given k and L.
    """
    print(f"Generating filtered matrix system for k = {k_value}, L = {L}, with d = {len(points_images)} points.")

    # Filter points_images based on rho_min and rho_max
    filtered_points_images = []
    for images in points_images:
        filtered_images = [img for img in images if rho_min <= img[0] <= rho_max]
        if len(filtered_images) > 1:  # Need at least 2 images to form pairs
            filtered_points_images.append(filtered_images)

    # Debugging: Print the number of filtered points and pairs
    print(f"Number of points with valid images after filtering: {len(filtered_points_images)}")

    # Define a function to compute each column (l, m)
    def compute_filtered_column(l, m):
        column = []
        for images in filtered_points_images:
            n_j = len(images)
            for alpha in range(n_j):
                for beta in range(alpha + 1, n_j):
                    rho_alpha, theta_alpha, phi_alpha = images[alpha]
                    rho_beta, theta_beta, phi_beta = images[beta]

                    Q_alpha = Q_k_lm_cached(k_value, l, m, rho_alpha, theta_alpha, phi_alpha)
                    Q_beta = Q_k_lm_cached(k_value, l, m, rho_beta, theta_beta, phi_beta)

                    column.append(Q_alpha - Q_beta)
        return column

    # Compute columns for each (l, m) pair
    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]

    columns = []
    for l, m in lm_pairs:
        column = compute_filtered_column(l, m)
        columns.append(column)

    # Transpose the result to get columns as needed
    matrix_system = list(map(list, zip(*columns)))

    # Debugging: Print final matrix size
    M = len(matrix_system)
    N_calculated = (L + 1) ** 2  # Number of columns is (L + 1)^2

    return M, N_calculated, matrix_system



