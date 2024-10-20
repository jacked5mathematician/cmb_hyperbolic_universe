import numpy as np
from utils import (
    build_dirichlet_domain,
    generate_random_points_in_domain,
    filter_points_in_domain,
    generate_transformed_points,
    convert_to_points_images,
    solve_system_via_svd_numeric,
    plot_chi_squared_spectrum,
    compute_target_M,
    filter_points_for_overconstraint,
    determine_tiling_radius,
    generate_matrix_system,   
    construct_numeric_matrix 
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

def process_k_values_chunk(process_index, k_values_chunk, inside_points, pairing_matrices, min_images, tolerance, manifold_name):
    chi_squared_values_chunk = []
    k_values_processed = []

    # Initialize variables per process
    previous_c_value = None
    classified_transformed_points = None
    rho_min, rho_max = None, None
    selected_points = None
    selected_transformed_points = None
    L = None

    for k_value in tqdm(k_values_chunk, desc=f"Processing k_values (Chunk {process_index+1})"):
        # Compute new c and L based on the current k_value
        new_c_value = 10 + round(100 / k_value)
        new_L_value = 10 + round(k_value)

        # Only recompute the tiling radius if the value of c has changed
        if new_c_value != previous_c_value:
            # Compute the tiling radius for the new value of c
            classified_transformed_points, rho_min, rho_max, valid_points = determine_tiling_radius(
            inside_points, pairing_matrices, new_L_value, new_c_value, min_images, tolerance
            )
            # If the tiling radius failed, skip this k_value
            if classified_transformed_points is None:
                print(f"Error: Could not determine tiling radius for k = {k_value}.")
                continue

            # Calculate the target number of rows (M) based on the degree of over-constraint (c) and L
            M_desired, N = compute_target_M(new_L_value, new_c_value)

            # Filter and select points for the desired over-constraint
            selected_points, selected_transformed_points = filter_points_for_overconstraint(
                classified_transformed_points, inside_points, M_desired
            )

            # Convert the dictionary selected_transformed_points to a list of lists of tuples
            points_images = convert_to_points_images(selected_transformed_points)

            # Update the previous c value to the new one
            previous_c_value = new_c_value

        # Use the filtered points for matrix generation
        valid_points = len(selected_points)

        # Compute matrix system and chi-squared values using filtered matrix system
        _, _, matrix_system = generate_matrix_system(points_images, new_L_value, k_value, valid_points)

        # Skip if the matrix system is empty
        if len(matrix_system) == 0 or len(matrix_system[0]) == 0:
            print(f"Error: matrix_system is empty for k = {k_value}")
            continue

        # Construct the numeric matrix
        A = construct_numeric_matrix(matrix_system, k_value)
        chi_squared, _ = solve_system_via_svd_numeric(A)
        chi_squared_values_chunk.append(chi_squared)
        k_values_processed.append(k_value)

    return chi_squared_values_chunk, k_values_processed

def main():
    manifold_name = 'm188(-1,1)'  # Example manifold name
    num_points = 10000  # Number of random points to generate
    min_images = 20  # Minimum number of images required per point
    tolerance = 0.1  # Allow small deviations in rho
    resolution = 400  # Resolution for the k values
    k_values = np.linspace(1.0, 10.0, resolution)  # Range of k values

    num_chunks = 4  # Number of chunks to process sequentially
    k_values_chunks = np.array_split(k_values, num_chunks)

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

    chi_squared_values = []
    k_values_collected = []

    for process_index, k_values_chunk in enumerate(k_values_chunks):
        print(f"Process {process_index+1} starting with k_values_chunk of size {len(k_values_chunk)}")
        chi_squared_chunk, k_values_chunk_processed = process_k_values_chunk(
            process_index, k_values_chunk, inside_points, pairing_matrices, min_images, tolerance, manifold_name
        )
        k_values_collected.extend(k_values_chunk_processed)
        chi_squared_values.extend(chi_squared_chunk)

    # Sort the results by k_values
    sorted_indices = np.argsort(k_values_collected)
    k_values_sorted = np.array(k_values_collected)[sorted_indices]
    chi_squared_values_sorted = np.array(chi_squared_values)[sorted_indices]

    # Step 8: Plot the chi-squared spectrum
    plot_chi_squared_spectrum(k_values_sorted, chi_squared_values_sorted, manifold_name, resolution)

if __name__ == "__main__":
    profile_function(main)