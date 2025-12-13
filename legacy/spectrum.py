import cProfile
import pstats
import io
import snappy
import numpy as np
import sympy as sp
from scipy.special import sph_harm
from scipy.linalg import svd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
from mpmath import hyp2f1, gamma, sqrt, sinh, cosh, pi
import mpmath as mp
from functools import lru_cache
import random
from scipy.special import lpmv, sph_harm

# Constants
K = -1  # Hyperbolic space (K = -1)
epsilon = 1e-12  # Small number to avoid division by zero

# Set mpmath precision for better convergence
mp.dps = 100  # Decimal places of precision
mp.pretty = True  # Print results in a more readable format

# Ensure all values are picklable for parallel processing
def ensure_picklable(value):
    """Convert mpmath or sympy types to standard Python types."""
    if isinstance(value, mp.mpf):
        return float(value)
    elif isinstance(value, mp.mpc):
        return complex(value)
    elif isinstance(value, sp.Basic):  # sympy symbolic types
        return float(value.evalf())
    return value

# Radial function using Phi^nu_l (Hyperspherical Bessel function for K=-1)
def Phi_nu_l(nu, l, chi):
    """Compute the hyperspherical Bessel function Phi^nu_l(chi) for K = -1."""
    nu = float(nu)
    chi = float(chi)
    
    # Compute N^nu_l as a product from n=1 to l of (nu^2 + n^2)
    N_nu_l = np.prod([nu**2 + n**2 for n in range(1, l + 1)])
    
    # The Legendre function P^{-1/2-l}_{-1/2+i*nu}(cosh(chi)) using hyp2f1
    def legendre_P(alpha, beta, x):
        if x < 1 + epsilon:
            x = 1 + epsilon
        
        prefactor = ((x + 1) / (x - 1))**(beta / 2)
        hyp_part = hyp2f1(alpha + 1, -alpha, 1 - beta, (1 - x) / 2)
        return prefactor * hyp_part / gamma(1 - beta)
    
    alpha = -1/2 + 1j * nu
    beta = -1/2 - l
    x = cosh(chi)
    
    legendre_value = legendre_P(alpha, beta, x)
    
    Phi = sqrt(pi * N_nu_l / (2 * sinh(chi))) * legendre_value
    return Phi

    
def normalization_constant(l, m):
    return np.sqrt((2 * l + 1) / (4 * np.pi) * mp.factorial(l - np.abs(m)) / mp.factorial(l + np.abs(m)))

# Real-valued spherical harmonics using explicit formulas
def Y_lm_real(l, m, theta, phi):
    if m > 0:
        return np.sqrt(2) * normalization_constant(l, m) * np.cos(m * phi) * lpmv(m, l, np.cos(theta))
    elif m < 0:
        return np.sqrt(2) * normalization_constant(l, -m) * np.sin(-m * phi) * lpmv(-m, l, np.cos(theta))
    else:
        return normalization_constant(l, 0) * lpmv(0, l, np.cos(theta))
    
# Cached versions to speed up repeated computations
def Phi_nu_l_cached(nu, l, chi):
    """Cached version of the hyperspherical Bessel function Phi^nu_l(chi) for K = -1."""
    return Phi_nu_l(nu, l, chi)

def Y_lm_real_cached(l, m, theta, phi):
    """Cached version of the real-valued spherical harmonics function."""
    return Y_lm_real(l, m, theta, phi)

# Global cache dictionaries
phi_cache = {}
y_lm_cache = {}

def Phi_nu_l_cached(nu, l, chi):
    """Cached version of the hyperspherical Bessel function Phi^nu_l(chi) for K = -1."""
    key = (nu, l, chi)
    if key not in phi_cache:
        phi_cache[key] = Phi_nu_l(nu, l, chi)
    return phi_cache[key]

def Y_lm_real_cached(l, m, theta, phi):
    """Cached version of the real-valued spherical harmonics function."""
    key = (l, m, theta, phi)
    if key not in y_lm_cache:
        y_lm_cache[key] = Y_lm_real(l, m, theta, phi)
    return y_lm_cache[key]

# Full eigenfunction Q_{k,l,m}(rho, theta, phi) using Phi_nu_l
def Q_k_lm(k, l, m, rho, theta, phi):
    nu = np.sqrt(k**2 + 1)
    Phi_nu_l_value = Phi_nu_l_cached(nu, l, rho)
    Y_lm_real_value = Y_lm_real_cached(l, m, theta, phi)
    return Phi_nu_l_value * Y_lm_real_value

@lru_cache(maxsize=None)
def Q_k_lm_cached(k, l, m, rho, theta, phi):
    return Q_k_lm(k, l, m, rho, theta, phi)

# Transformation application in SO(3,1)
def apply_so31_action(matrix, point):
    norm_squared = np.dot(point, point)
    X0 = (1 + norm_squared) / (1 - norm_squared)
    X1, X2, X3 = 2 * point / (1 - norm_squared)
    hyperboloid_point = np.array([X0, X1, X2, X3], dtype=float)
    transformed_point = np.dot(matrix, hyperboloid_point)
    return transformed_point

# Project back to Klein coordinates
def project_to_klein(transformed_points):
    if transformed_points.ndim == 1:
        X0, X1, X2, X3 = transformed_points
        return np.array([X1 / X0, X2 / X0, X3 / X0], dtype=float)
    elif transformed_points.ndim == 2 and transformed_points.shape[1] == 4:
        X0 = transformed_points[:, 0]
        X1 = transformed_points[:, 1]
        X2 = transformed_points[:, 2]
        X3 = transformed_points[:, 3]
        return np.column_stack((X1 / X0, X2 / X0, X3 / X0))
    else:
        raise ValueError(f"Unexpected shape for transformed_points: {transformed_points.shape}")

# Convert Klein to pseudo-spherical coordinates (rho, theta, phi)
def klein_to_pseudo_spherical(points):
    pseudo_spherical_points = []
    for point in points:
        p_x, p_y, p_z = point
        norm_squared = p_x**2 + p_y**2 + p_z**2
        X0 = (1 + norm_squared) / (1 - norm_squared)
        X1 = 2 * p_x / (1 - norm_squared)
        X2 = 2 * p_y / (1 - norm_squared)
        X3 = 2 * p_z / (1 - norm_squared)
        rho = np.arccosh(X0)
        sinh_rho = np.sinh(rho)
        theta = np.arccos(X3 / sinh_rho) if sinh_rho != 0 else 0
        phi = np.arctan2(X2, X1)
        pseudo_spherical_points.append([rho, theta, phi])
    return np.array(pseudo_spherical_points, dtype=float)

# Ensures minimum number of images per point by varying rho_max dynamically
def ensure_minimum_images(inside_points, pairing_matrices, min_images_per_point=10):
    rho_max = 0.5  # Start with a small rho_max
    step_size = 0.1

    while True:
        classified_points = generate_transformed_points(inside_points, pairing_matrices, rho_max)
        all_sufficient = all(len(images) >= min_images_per_point for images in classified_points.values())
        
        if all_sufficient:
            return classified_points, rho_max
        else:
            rho_max += step_size


# Generate transformed points by applying the group generators
def generate_transformed_points(inside_points, pairing_matrices, rho_max):
    classified_points = {i: [] for i in range(len(inside_points))}
    inside_points = np.array(inside_points)
    
    for i, point in enumerate(inside_points):
        hyperboloid_points = np.array([apply_so31_action(matrix, point) for matrix in pairing_matrices])
        klein_points = project_to_klein(hyperboloid_points)
        pseudo_spherical_points = klein_to_pseudo_spherical(klein_points)
        
        for pseudo_spherical_point in pseudo_spherical_points:
            if pseudo_spherical_point[0] <= rho_max:
                classified_points[i].append(pseudo_spherical_point)
    
    return classified_points

# Converts pairing matrices to proper 4x4 NumPy arrays
def convert_to_4x4_matrices(pairs):
    matrices = []
    for pair in tqdm(pairs, desc="Converting Pairing Matrices"):
        matrix = np.array(list(pair), dtype=float)
        if matrix.shape == (16,):  
            matrix = matrix.reshape(4, 4)
        matrices.append(matrix)
    return matrices

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

def generate_matrix_system(points_images, L, k_value):
    # Convert k_value to float if it's not already to ensure it's picklable
    k_value = ensure_picklable(k_value)

    d = len(points_images)  # Number of points inside the domain

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

def convert_to_points_images(classified_transformed_points):
    """
    Converts the dictionary structure of classified_transformed_points to a list of lists of tuples.
    Each tuple represents a point in (rho, theta, phi) coordinates.
    """
    points_images = []
    
    for key, images in classified_transformed_points.items():
        point_list = []
        for image in images:
            # Convert array to tuple
            point_tuple = tuple(image)  # Assumes image is a numpy array like [rho, theta, phi]
            point_list.append(point_tuple)
        points_images.append(point_list)
    
    return points_images

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

# Function to solve the system using SVD and compute chi^2
def solve_system_via_svd_numeric(A):
    U, s, Vt = svd(A)
    a = Vt[-1]  # The singular vector corresponding to the smallest singular value
    
    # Calculate chi^2 = |A * a|^2
    chi_squared = np.linalg.norm(A @ a)**2
    return chi_squared, a

# Function to compute chi^2 for a given k
def compute_chi_squared_for_k(k_value, matrix_system):
    A = construct_numeric_matrix(matrix_system, k_value)
    chi_squared, _ = solve_system_via_svd_numeric(A)
    return chi_squared

# Compute the chi^2 spectrum for a range of k values in parallel
def compute_chi_squared_spectrum_parallel(matrix_system, M, N, k_values):
    chi_squared_values = Parallel(n_jobs=-1)(
        delayed(compute_chi_squared_for_k)(k_val, matrix_system) for k_val in tqdm(k_values, desc="Computing Chi-Squared Spectrum")
    )
    return chi_squared_values

# Function to plot the chi^2 spectrum
def plot_chi_squared_spectrum(k_values, chi_squared_values, L, num_points_used):
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, chi_squared_values, label=r'$\chi^2(k)$ Spectrum', color='blue')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\chi^2$')
    plt.title(r'$\chi^2$ Spectrum for Varying $k$, L = {}$, Points = {}'.format(L, num_points_used))
    plt.grid(True)
    plt.legend()
    plt.show()

# Main function to generate points, build the system, and find eigenvalue k, including chi^2 spectrum plot
def generate_and_solve_system_with_plot(manifold_name, L=1, num_points=1000, min_images_per_point=10, k_values=None, num_points_to_use=None, resolution=100):
    M = snappy.Manifold(manifold_name)
    
    try:
        D = M.dirichlet_domain()
    except RuntimeError as e:
        print(f"Failed to compute Dirichlet domain: {e}")
        return
    
    pairing_matrices = D.pairing_matrices()
    pairing_matrices = convert_to_4x4_matrices(pairing_matrices)

    vertex_details = D.vertex_list(details=True)
    vertices = np.array([list(v['position']) for v in vertex_details], dtype=float)

    faces = D.face_list()
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)

    # Generate random points and filter those inside the domain
    print(f"Generating {num_points} random points...")
    points = np.random.uniform(min_corner, max_corner, (num_points, 3))

    inside_points = []
    for point in tqdm(points, desc="Filtering Points Inside Domain"):
        is_inside = True
        for face in faces:
            face_vertices = vertices[face['vertex_indices']]
            if len(face_vertices) >= 3:
                v1 = face_vertices[1] - face_vertices[0]
                v2 = face_vertices[2] - face_vertices[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                if np.dot(point - face_vertices[0], normal) > 0:
                    is_inside = False
                    break
        if is_inside:
            inside_points.append(point)

    inside_points = np.array(inside_points)
    print(f"Number of points found inside the domain: {len(inside_points)}")

    # Select the desired number of points to use from those inside the domain
    if num_points_to_use is None or num_points_to_use > len(inside_points):
        num_points_to_use = len(inside_points)
    
    selected_points = random.sample(list(inside_points), num_points_to_use)
    print(f"Selected {num_points_to_use} points for the system.")
    
    if pairing_matrices is not None and len(selected_points) > 0:
        classified_transformed_points, rho_max = ensure_minimum_images(selected_points, pairing_matrices, min_images_per_point)
        print(f"Final rho_max used: {rho_max}")
        
        # Convert the dictionary classified_transformed_points to a list of lists of tuples
        points_images = convert_to_points_images(classified_transformed_points)
        
        # If k_values is None, create a default range of k values
        if k_values is None:
            k_values = np.linspace(1.0, 10.0, resolution)  # Adjust the range and resolution as needed

        # Generate matrix system for the k values
        chi_squared_values = []
        
        for k_value in tqdm(k_values, desc="Computing Matrix for each k"):
            # Generate the matrix system for the current k_value
            M, N, matrix_system = generate_matrix_system(points_images, L, k_value)
            
            # Construct the numeric matrix for the current k_value
            A = construct_numeric_matrix(matrix_system, k_value)
            
            # Perform SVD and compute chi^2
            chi_squared, _ = solve_system_via_svd_numeric(A)
            chi_squared_values.append(chi_squared)
        
        # Plot the chi^2 spectrum after all k-values are processed
        plot_chi_squared_spectrum(k_values, chi_squared_values, L, num_points_to_use)
    else:
        print("No points found inside the domain or no transformations applied.")

import cProfile
import pstats
import io

# Profiler function to wrap any function you want to profile
def profile_function(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()  # Start profiling
    result = func(*args, **kwargs)
    pr.disable()  # Stop profiling
    
    # Create a string buffer to hold the profile results
    s = io.StringIO()
    
    # Create a Stats object
    ps = pstats.Stats(pr, stream=s)
    
    # Sort and print by different criteria
    sort_criteria = ['cumulative', 'time', 'calls']
    for criteria in sort_criteria:
        s.write(f"\n---- Profile sorted by {criteria} ----\n")
        ps.sort_stats(criteria).print_stats(10)  # Print top 10 functions
    
    # Save profiling results to a file for later inspection
    with open("profiling_results.txt", "w") as f:
        f.write(s.getvalue())
    
    # Print the profile statistics to the console
    print(s.getvalue())  # Print the contents of the buffer to the console
    
    return result

# Run the function with profiling
profile_function(generate_and_solve_system_with_plot, 'm188(-1,1)', L=11, num_points=1000, num_points_to_use=40, resolution=100)