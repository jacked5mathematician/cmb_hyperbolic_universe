import numpy as np
import random
from utils import (
    build_dirichlet_domain,
    generate_random_points_in_domain,
    filter_points_in_domain,
    select_points,
    generate_transformed_points,
    convert_to_points_images,
    generate_matrix_system,
    construct_numeric_matrix,
    solve_system_via_svd_numeric,
    plot_chi_squared_spectrum,
    compute_target_M,
    filter_points_for_overconstraint
)
from tqdm import tqdm  # Import for progress bars

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

# Helper function to ensure minimum images per point
def ensure_minimum_images(inside_points, pairing_matrices, min_images_per_point=10, max_rho_max=5.0):
    rho_max = 0.5  # Start with an initial small rho_max
    step_size = 0.1  # Adjust rho_max in increments
    classified_points = None

    while rho_max <= max_rho_max:
        # Generate transformed points with the current rho_max
        classified_points = generate_transformed_points(inside_points, pairing_matrices, rho_max)

        # Check if all points have at least `min_images_per_point`
        all_sufficient = all(len(images) >= min_images_per_point for images in classified_points.values())

        if all_sufficient:
            break  # We have enough images, exit the loop
        else:
            rho_max += step_size  # Increase rho_max and try again

    if rho_max > max_rho_max:
        print(f"Warning: rho_max exceeded the limit of {max_rho_max}, stopping adjustment.")

    return classified_points, rho_max

def main():
    manifold_name = 'm188(-1,1)'  # Example manifold name
    L = 11  # Example value for angular momentum
    num_points = 1000  # Number of random points to generate
    c = 60  # Degree of over-constraint (c = M / N)
    min_images_per_point = 10  # Minimum number of images per point
    k_values = np.linspace(1.0, 10.0, 100)  # Range of k values
    resolution = 100  # Resolution for the k values

    # Step 1: Build Dirichlet domain
    domain_data = build_dirichlet_domain(manifold_name)
    if domain_data is None:
        print("Failed to build Dirichlet domain.")
        return
    vertices, faces, pairing_matrices = domain_data

    # Step 2: Generate random points
    points = generate_random_points_in_domain(vertices, num_points)

    # Step 3: Filter points inside the domain
    inside_points = filter_points_in_domain(points, faces, vertices)
    print(f"Number of points found inside the domain: {len(inside_points)}")

    # Step 4: Ensure minimum images per point and dynamically adjust rho_max
    if pairing_matrices is not None and len(inside_points) > 0:
        classified_transformed_points, rho_max = ensure_minimum_images(inside_points, pairing_matrices, min_images_per_point)
        print(f"Final rho_max used: {rho_max}")

        # Step 5: Calculate the target number of rows (M) based on the degree of over-constraint (c) and L
        M_desired, N = compute_target_M(L, c)
        print(f"Computed target number of rows (M_desired): {M_desired}")
        print(f"Computed number of columns (N): {N}")

        # Step 6: Filter and select points for the desired over-constraint
        selected_points, selected_transformed_points = filter_points_for_overconstraint(classified_transformed_points, inside_points, M_desired)

        # Convert the dictionary selected_transformed_points to a list of lists of tuples
        points_images = convert_to_points_images(selected_transformed_points)

        # Step 7: Compute matrix system and chi-squared values (Note: M remains the same for each k-value)
        chi_squared_values = []
        for k_value in tqdm(k_values, desc="Computing Matrix for each k"):
            _, _, matrix_system = generate_matrix_system(points_images, L, k_value)

            # Skip if the matrix system is empty
            if len(matrix_system) == 0 or len(matrix_system[0]) == 0:
                print(f"Error: matrix_system is empty for k = {k_value}")
                continue

            # Construct the numeric matrix
            A = construct_numeric_matrix(matrix_system, k_value)
            chi_squared, _ = solve_system_via_svd_numeric(A)
            chi_squared_values.append(chi_squared)

        # Step 8: Calculate actual degree of over-constraint and compare with target
        M_actual = A.shape[0]  # The number of rows in the matrix
        c_actual = M_actual / N  # Actual degree of over-constraint
        print(f"Actual number of rows (M_actual): {M_actual}")
        print(f"Actual degree of over-constraint (c_actual): {c_actual}")
        print(f"Difference between target and actual degree of over-constraint: {abs(c_actual - c)}")

        # Step 9: Plot the chi-squared spectrum
        plot_chi_squared_spectrum(k_values, chi_squared_values, L, len(selected_points))
    else:
        print("No points found inside the domain or no transformations applied.")

if __name__ == "__main__":
    profile_function(main)