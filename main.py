import numpy as np
from utils import (
    build_dirichlet_domain,
    generate_random_points_in_domain,
    filter_points_in_domain,
    convert_to_points_images,
    solve_system_via_svd_numeric,
    plot_chi_squared_spectrum,
    filter_points_for_overconstraint,
    determine_tiling_radius,
    generate_matrix_system,
    construct_numeric_matrix,
    precompute_special_functions
)
from tqdm import tqdm

import cProfile
import pstats
import io

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


def main():
    manifold_name = 'm188(-1,1)'  # Example manifold name
    num_points = 10000  # Number of random points to generate
    min_images = 20  # Minimum number of images required per point
    tolerance = 0.1  # Allow small deviations in rho
    resolution = 10  # Resolution for the k values
    k_values = np.linspace(1.0, 10.0, resolution)  # Range of k values

    # Specify how many chunks you want
    num_chunks = 2  # Example: You want 4 chunks

    # Calculate the chunk size based on the number of k_values and number of chunks
    chunk_size = len(k_values) // num_chunks
    remainder = len(k_values) % num_chunks  # Handle remainder values

    # Step 1: Build Dirichlet domain
    domain_data = build_dirichlet_domain(manifold_name)
    if domain_data is None:
        print("Failed to build Dirichlet domain.")
        return
    vertices, faces, pairing_matrices = domain_data

    # Step 2: Generate random points and filter them
    points = generate_random_points_in_domain(vertices, num_points)
    inside_points = filter_points_in_domain(points, faces, vertices)
    print(f"Number of points found inside the domain: {len(inside_points)}")

    chi_squared_values = []  # Store chi-squared values across chunks

    # Process `k_values` in the specified number of chunks
    start_idx = 0
    for chunk_idx in range(num_chunks):
        end_idx = start_idx + chunk_size
        if chunk_idx < remainder:
            end_idx += 1  # Distribute remainder evenly across chunks

        k_values_chunk = k_values[start_idx:end_idx]
        start_idx = end_idx  # Move to the next chunk

        # Step 3: Precompute tiling radius and points_images for the current chunk of k_values
        precomputed_tiling_data = {}
        for k_value in tqdm(k_values_chunk, desc=f"Precomputing chunk {chunk_idx+1}/{num_chunks}"):
            new_c_value = 10 + round(100 / k_value)
            new_L_value = 10 + round(k_value)
            print(f"Precomputing for k = {k_value}, L = {new_L_value}")

            # Compute tiling radius
            classified_transformed_points, rho_min, rho_max, valid_points, M_desired = determine_tiling_radius(
                inside_points, pairing_matrices, new_L_value, new_c_value, min_images, tolerance
            )

            if classified_transformed_points is None:
                print(f"Error: Could not determine tiling radius for k = {k_value}. Skipping.")
                continue

            # Filter points and select transformed points
            selected_points, selected_transformed_points = filter_points_for_overconstraint(
                classified_transformed_points, inside_points, M_desired
            )
            points_images = convert_to_points_images(selected_transformed_points)

            # Store precomputed tiling data
            precomputed_tiling_data[k_value] = {
                'points_images': points_images,
                'valid_points': len(selected_points),
                'L_value': new_L_value,
                'M_desired': M_desired
            }

        # Step 4: Precompute special function values (q_values) for the current chunk
        precomputed_data = precompute_special_functions(precomputed_tiling_data)

        # Step 5: Generate matrices for each k_value in the current chunk and compute chi-squared values
        for idx, k_value in enumerate(k_values_chunk):
            q_values, points_images, L_value = precomputed_data[idx]

            # Generate matrix system
            M, N, matrix_system = generate_matrix_system(q_values, points_images, L_value)
            A = construct_numeric_matrix(matrix_system, k_value)
            chi_squared, _ = solve_system_via_svd_numeric(A)
            chi_squared_values.append(chi_squared)

        # Clear precomputed data after each chunk to free memory
        precomputed_tiling_data.clear()
        precomputed_data.clear()

    # Step 6: Plot the chi-squared spectrum
    plot_chi_squared_spectrum(k_values, chi_squared_values, manifold_name, resolution)


if __name__ == "__main__":
    profile_function(main)