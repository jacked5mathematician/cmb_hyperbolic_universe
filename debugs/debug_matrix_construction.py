import numpy as np
from utils import (
    build_dirichlet_domain,
    generate_random_points_in_domain,
    filter_points_in_domain,
    generate_transformed_points,
    convert_to_points_images,
    determine_tiling_radius,
    generate_filtered_matrix_system,
    construct_filtered_numeric_matrix,
    compute_target_M
)

def debug_tiling_radius_and_matrix():
    manifold_name = 'm188(-1,1)'  # Example manifold name
    L = 11  # Example value for angular momentum
    num_points = 10000  # Number of random points to generate for quicker debugging
    c = 50  # Degree of over-constraint (c = M / N)
    min_images = 12  # Minimum number of images required per point
    max_images = 20  # Maximum number of images allowed per point
    tolerance = 0.1  # Allow small deviations in rho

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

    # Step 4: Determine tiling radius using min_images, max_images, and tolerance
    classified_transformed_points, rho_min, rho_max = determine_tiling_radius(
        inside_points, pairing_matrices, L, c, min_images=min_images, tolerance=tolerance
    )

    # Verify the number of rows expected based on the images (prior to matrix generation)
    total_rows = 0
    for idx, images in classified_transformed_points.items():
        n_j = len(images)
        total_rows += n_j * (n_j - 1) // 2  # Compute number of rows from image pairs
    print(f"Expected total rows based on the images: {total_rows}")

    # Step 5: Calculate the target number of rows (M) based on the degree of over-constraint (c) and L
    M_desired, N = compute_target_M(L, c)
    print(f"Computed target number of rows (M_desired): {M_desired}")
    print(f"Computed number of columns (N): {N}")

    # Step 6: Convert the dictionary classified_transformed_points to a list of lists of tuples
    points_images = convert_to_points_images(classified_transformed_points)

    # Step 7: Compute matrix system for the first k-value and compare row sizes
    k_value = 1.0  # Pick the first k-value for debugging
    total_rows_matrix_system, _, matrix_system = generate_filtered_matrix_system(points_images, L, k_value, rho_min, rho_max)
    print(f"Generated matrix system for k = {k_value}, total rows in matrix system: {total_rows_matrix_system}")

    # Step 8: Construct the numeric matrix for the first k-value
    A = construct_filtered_numeric_matrix(matrix_system, k_value)
    print(f"Constructed matrix size: {A.shape[0]} rows x {A.shape[1]} columns")

    # Check if the rows from the matrix construction match the expected number of rows from the images
    if A.shape[0] != total_rows:
        print(f"Discrepancy detected: Expected {total_rows} rows, but constructed matrix has {A.shape[0]} rows.")

if __name__ == "__main__":
    debug_tiling_radius_and_matrix()