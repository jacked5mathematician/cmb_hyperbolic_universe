import numpy as np
from utils import (
    build_dirichlet_domain,
    generate_random_points_in_domain,
    filter_points_in_domain,
    generate_transformed_points,
    convert_to_points_images,
    generate_matrix_system,
    construct_numeric_matrix, 
    compute_target_M, 
    construct_numeric_matrix,
    filter_points_for_overconstraint
)
def debug_parameter_control():
    manifold_name = 'm188(-1,1)'  # Example manifold name
    L = 11  # Example value for angular momentum
    num_points = 1000  # Use a smaller number of points for debugging
    c = 70  # Degree of over-constraint (c = M / N)
    rho_max = 4.5  # For generating transformed points
    test_k_value = 1.0  # Use a test k value for matrix generation

    # Step 1: Build Dirichlet domain
    print("Building Dirichlet domain...")
    domain_data = build_dirichlet_domain(manifold_name)
    if domain_data is None:
        print("Failed to build Dirichlet domain.")
        return
    vertices, faces, pairing_matrices = domain_data

    # Step 2: Generate random points
    points = generate_random_points_in_domain(vertices, num_points)

    # Step 3: Filter points inside the domain
    inside_points = filter_points_in_domain(points, faces, vertices)
    print(f"Number of points inside the domain: {len(inside_points)}")

    # Step 4: Generate transformed points
    print("Generating transformed points...")
    classified_transformed_points = generate_transformed_points(inside_points, pairing_matrices, rho_max)

    # Step 5: Compute target M based on c and L
    M_desired, N = compute_target_M(L, c)
    print(f"Computed target number of rows (M): {M_desired}")
    print(f"Computed number of columns (N): {N}")

    # Step 6: Filter and select points for over-constraint
    print("Filtering and selecting points based on over-constraint...")
    selected_points, selected_transformed_points = filter_points_for_overconstraint(classified_transformed_points, inside_points, M_desired)

    # Convert the dictionary selected_transformed_points to a list of lists of tuples
    points_images = convert_to_points_images(selected_transformed_points)

    # Debug Output:
    print(f"Number of selected points: {len(selected_points)}")

    # Step 7: Generate matrix system for a test k value
    print(f"Generating matrix system for test k value: {test_k_value}")
    _, _, matrix_system = generate_matrix_system(points_images, L, test_k_value)

    # Check the matrix system structure
    if len(matrix_system) == 0 or len(matrix_system[0]) == 0:
        print("Error: matrix_system is empty.")
        return

    print(f"Matrix system generated with dimensions: {len(matrix_system)} rows, {len(matrix_system[0])} columns")

    # Step 8: Construct the numerical matrix
    print("Constructing the numerical matrix...")
    A = construct_numeric_matrix(matrix_system, test_k_value)
    print(f"Matrix A shape: {A.shape}")

    # Step 9: Calculate and compare actual over-constraint
    M_actual = A.shape[0]  # The number of rows in the matrix
    c_actual = M_actual / N  # Actual degree of over-constraint
    print(f"Actual number of rows (M_actual): {M_actual}")
    print(f"Actual degree of over-constraint (c_actual): {c_actual}")
    print(f"Difference between target and actual degree of over-constraint: {abs(c_actual - c)}")
