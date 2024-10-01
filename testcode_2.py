import snappy
import numpy as np
import sympy as sp
from scipy.special import sph_harm
from scipy.linalg import svd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar

k_sym = sp.symbols('k')
rho_sym = sp.symbols('rho_sym')

import numpy as np
import matplotlib.pyplot as plt
from mpmath import hyp2f1, gamma, sqrt, sinh, cosh, pi
import mpmath as mp

# Constants
K = -1  # Hyperbolic space (K = -1)
epsilon = 1e-12  # Small number to avoid division by zero

def ensure_picklable(value):
    """Convert mpmath or sympy types to standard Python types."""
    if isinstance(value, mp.mpf):
        return float(value)
    elif isinstance(value, mp.mpc):
        return complex(value)
    elif isinstance(value, mpq):
        return float(value)
    elif isinstance(value, sp.Basic):  # sympy symbolic types
        return float(value.evalf())
    return value

# Ensure picklable values before passing to the parallel function
k_value = ensure_picklable(k_value)

# Set mpmath precision for better convergence
mp.dps = 100  # Decimal places of precision
mp.pretty = True  # Print results in a more readable format

from functools import lru_cache

# Radial function using Phi^nu_l (Hyperspherical Bessel function for K=-1)
def Phi_nu_l(nu, l, chi):
    """Compute the hyperspherical Bessel function Phi^nu_l(chi) for K = -1."""
    
    # Ensure nu and chi are numeric
    if isinstance(nu, sp.Basic):  # If nu is symbolic, convert to numeric
        nu = float(nu)
    if isinstance(chi, sp.Basic):
        chi = float(chi)
    
    # Compute N^nu_l as a product from n=1 to l of (nu^2 + n^2)
    N_nu_l = np.prod([nu**2 + n**2 for n in range(1, l + 1)])
    
    # The Legendre function P^{-1/2-l}_{-1/2+i*nu}(cosh(chi)) using hyp2f1
    def legendre_P(alpha, beta, x):
        # Handle small x to avoid division by zero
        if x < 1 + epsilon:
            x = 1 + epsilon
        
        # Compute the prefactor (x+1)/(x-1) raised to beta/2
        prefactor = ((x + 1) / (x - 1))**(beta / 2)
        
        # Compute the hypergeometric part
        hyp_part = hyp2f1(alpha + 1, -alpha, 1 - beta, (1 - x) / 2)
        
        # Combine everything
        return prefactor * hyp_part / gamma(1 - beta)
    
    # Parameters for the Legendre function
    alpha = -1/2 + 1j * nu
    beta = -1/2 - l
    x = cosh(chi)
    
    # Legendre function P^beta_alpha(cosh(chi))
    legendre_value = legendre_P(alpha, beta, x)
    
    # Compute the full Phi^nu_l(chi)
    Phi = sqrt(pi * N_nu_l / (2 * sinh(chi))) * legendre_value
    
    return Phi

# Real-valued spherical harmonics function
def Y_lm_real(l, m, theta, phi):
    if m == 0:
        return sph_harm(0, l, phi, theta).real
    elif m > 0:
        return np.sqrt(2) * (-1)**m * sph_harm(m, l, phi, theta).real
    else:
        return np.sqrt(2) * (-1)**m * sph_harm(-m, l, phi, theta).imag
    
@lru_cache(maxsize=None)
def Phi_nu_l_cached(nu, l, chi):
    """Cached version of the hyperspherical Bessel function Phi^nu_l(chi) for K = -1."""
    return Phi_nu_l(nu, l, chi)

@lru_cache(maxsize=None)
def Y_lm_real_cached(l, m, theta, phi):
    """Cached version of the real-valued spherical harmonics function."""
    return Y_lm_real(l, m, theta, phi)

# Full eigenfunction Q_{k,l,m}(rho, theta, phi) using Phi_nu_l
def Q_k_lm(k, l, m, rho, theta, phi):
    # Ensure k is numeric when passed
    if isinstance(k, sp.Basic):  # If k is symbolic, evaluate it numerically
        k = float(k.evalf())

    # Compute nu based on the numeric value of k
    nu = np.sqrt(k**2 + 1)  # nu is related to k, and k should be numeric
    
    # Ensure that rho is also numeric
    if isinstance(rho, sp.Basic):
        rho = float(rho)
    
    # Compute the radial part and the spherical harmonics part
    Phi_nu_l_value = Phi_nu_l(nu, l, rho)  # Hyperspherical Bessel function
    Y_lm_real_value = Y_lm_real(l, m, theta, phi)  # Real-valued spherical harmonics
    return Phi_nu_l_value * Y_lm_real_value

# Transformation application in SO(3,1)
def apply_so31_action(matrix, point):
    norm_squared = np.dot(point, point)
    X0 = (1 + norm_squared) / (1 - norm_squared)
    X1, X2, X3 = 2 * point / (1 - norm_squared)
    hyperboloid_point = np.array([X0, X1, X2, X3], dtype=float)
    transformed_point = np.dot(matrix, hyperboloid_point)
    return transformed_point

# Project back to Klein coordinates
# Project back to Klein coordinates
def project_to_klein(transformed_points):
    # Check if input is a single point or a batch of points
    if transformed_points.ndim == 1:
        # Single point
        X0, X1, X2, X3 = transformed_points
        return np.array([X1 / X0, X2 / X0, X3 / X0], dtype=float)
    elif transformed_points.ndim == 2 and transformed_points.shape[1] == 4:
        # Batch of points
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
    step_size = 0.5

    while True:
        classified_points = generate_transformed_points(inside_points, pairing_matrices, rho_max)
        
        # Check if every point has at least the required number of images
        all_sufficient = all(len(images) >= min_images_per_point for images in classified_points.values())
        
        if all_sufficient:
            return classified_points, rho_max
        else:
            rho_max += step_size  # Increase rho_max iteratively until the condition is met

# Generate transformed points by applying the group generators
# Generate transformed points by applying the group generators
def generate_transformed_points(inside_points, pairing_matrices, rho_max):
    classified_points = {i: [] for i in range(len(inside_points))}

    # Convert to array for batch processing
    inside_points = np.array(inside_points)
    
    for i, point in enumerate(inside_points):
        # Apply action to all pairing matrices
        hyperboloid_points = np.array([apply_so31_action(matrix, point) for matrix in pairing_matrices])
        
        # Project to Klein coordinates
        klein_points = project_to_klein(hyperboloid_points)  # `project_to_klein` now handles batches

        # Convert Klein coordinates to pseudo-spherical coordinates
        pseudo_spherical_points = klein_to_pseudo_spherical(klein_points)
        
        # Filter the points by rho_max and assign
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
from joblib import Parallel, delayed

# Function to generate the matrix system with optimized joblib usage
# Function to generate the matrix system with optimized joblib usage
def generate_matrix_system(points_images, L, k_value):
    # Convert k_value to float if it's not already to ensure it's picklable
    k_value = ensure_picklable(k_value)

    d = len(points_images)  # Number of points inside the domain

    # Define a function to compute each column (l, m)
    def compute_column(l, m):
        column = []
        for j in range(d):
            images = points_images[j]
            n_j = len(images)

            for alpha in range(n_j):
                for beta in range(alpha + 1, n_j):
                    rho_alpha, theta_alpha, phi_alpha = images[alpha]
                    rho_beta, theta_beta, phi_beta = images[beta]

                    Q_alpha = Q_k_lm(k_value, l, m, rho_alpha, theta_alpha, phi_alpha)
                    Q_beta = Q_k_lm(k_value, l, m, rho_beta, theta_beta, phi_beta)

                    # Convert Q_alpha and Q_beta to complex or float to ensure picklability
                    Q_alpha = ensure_picklable(Q_alpha)
                    Q_beta = ensure_picklable(Q_beta)

                    column.append(Q_alpha - Q_beta)
        return column

    # Use Parallel processing with optimized settings
    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]

    # Print debug information before parallelization
    print(f"Generating matrix system for k = {k_value}, L = {L}, with d = {d} points.")

    # Using batch_size to control task granularity
    columns = Parallel(n_jobs=1, batch_size='auto', prefer="processes")(
        delayed(compute_column)(l, m) for l, m in lm_pairs
    )

    matrix_system = list(map(list, zip(*columns)))

    N_calculated = (L + 1)**2  # Number of columns
    return len(matrix_system), N_calculated, matrix_system

# Function to construct the numerical matrix for a given value of k
def construct_numeric_matrix(matrix_system, k_value):
    M = len(matrix_system)
    N = len(matrix_system[0])
    A = np.zeros((M, N), dtype=complex)  # Ensure the matrix is complex
    
    # Iterate over rows and columns, converting mpc to Python complex numbers
    for i in range(M):
        for j in range(N):
            entry = matrix_system[i][j]
            if isinstance(entry, mp.mpc):  # If it's an mpc object
                A[i, j] = complex(entry.real, entry.imag)  # Convert to Python complex
            else:
                A[i, j] = entry  # Assume already numeric
    
    print(f"Constructed matrix for k = {k_value}: size = {M} x {N}")
    return A

def solve_system_via_svd_numeric(A):
    # Perform SVD on A
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

def compute_chi_squared_spectrum_parallel(matrix_system, M, N, k_values):
    # Use parallel processing to compute chi^2 for each k in parallel
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
import random

# Main function to generate points, build the system, and find eigenvalue k, including chi^2 spectrum plot
def generate_and_solve_system_with_plot(manifold_name, L=1, num_points=1000, min_images_per_point=10, k_values=None, num_points_to_use=None):
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

    # Generate more random points than needed, then filter based on the number of points inside the domain
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
        
        # If k_values is None, create a default range of k values
        if k_values is None:
            k_values = np.linspace(1.0, 10.0, 100)  # Adjust the range and resolution as needed

        # Generate matrix system for the first k value (progress bar)
        matrix_progress = tqdm(k_values, desc="Computing Matrix for each k")
        
        chi_squared_values = []
        
        for k_value in matrix_progress:
            # Generate the matrix system for the current k_value
            M, N, matrix_system = generate_matrix_system(classified_transformed_points, L, k_value)
            
            # Construct the numeric matrix for the current k_value
            A = construct_numeric_matrix(matrix_system, k_value)
            
            # Perform SVD and compute chi^2
            chi_squared, _ = solve_system_via_svd_numeric(A)
            chi_squared_values.append(chi_squared)
        
        # Plot the chi^2 spectrum after all k-values are processed
        plot_chi_squared_spectrum(k_values, chi_squared_values, L, num_points_to_use)
    else:
        print("No points found inside the domain or no transformations applied.")

# Example usage
#generate_and_solve_system_with_plot('m188(-1,1)', L=4, num_points=1000, num_points_to_use=2)

for i in range(3,7,1):
    generate_and_solve_system_with_plot('m188(-1,1)', L=i, num_points=1000, num_points_to_use=2)