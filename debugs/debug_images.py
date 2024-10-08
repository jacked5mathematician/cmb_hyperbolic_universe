import numpy as np
from utils.domain_building import build_dirichlet_domain, generate_random_points_in_domain, filter_points_in_domain
from utils.transformations import generate_transformed_points, convert_to_points_images
from utils.sys_generation import generate_matrix_system, construct_numeric_matrix

def select_points_for_c(n_j_values, target_M):
    """
    Select points in such a way that the sum of n_j(n_j - 1)/2 is as close as possible to the target number of rows M.

    Parameters:
    - n_j_values: List of (point_index, n_j) where n_j is the number of images for that point.
    - target_M: The target number of rows M based on the over-constraint parameter c.

    Returns:
    - selected_indices: List of selected point indices.
    """
    selected_indices = []
    current_M = 0
    
    # Sort the points by the number of images (n_j) in descending order to prioritize points with more images
    sorted_n_j_values = sorted(n_j_values, key=lambda x: x[1], reverse=True)

    for idx, n_j in sorted_n_j_values:
        current_M += n_j * (n_j - 1) / 2  # Add the contribution to M
        selected_indices.append(idx)
        
        # Stop when we have enough points to reach the target_M
        if current_M >= target_M:
            break
    
    return selected_indices

def compute_n_j_and_matrix_with_c(manifold_name, c, num_points=1000, rho_max=0.5, L=1, k_value=1.0):
    """
    Constructs matrix A based on the over-constraint parameter c, computes n_j(n_j - 1)/2 for each point,
    and adjusts the number of points used to match the desired value of c = M/N.

    Parameters:
    - manifold_name: The name of the manifold.
    - c: The degree of over-constraint (c = M/N).
    - num_points: Number of points to generate.
    - rho_max: Maximum value for rho for generating images.
    - L: Angular momentum parameter for generating the matrix system.
    - k_value: Value of k for constructing the matrix system.

    Returns:
    - A comparison between the computed sum of n_j(n_j - 1)/2 and the number of rows in the matrix.
    """
    # Step 1: Build the Dirichlet domain
    domain_data = build_dirichlet_domain(manifold_name)
    if domain_data is None:
        print("Failed to build Dirichlet domain.")
        return None
    vertices, faces, pairing_matrices = domain_data

    # Step 2: Generate random points inside the domain
    points = generate_random_points_in_domain(vertices, num_points)

    # Step 3: Filter points inside the domain
    inside_points = filter_points_in_domain(points, faces, vertices)
    print(f"Number of points found inside the domain: {len(inside_points)}")

    # Step 4: Generate transformed points and count the number of images for each point
    classified_transformed_points = generate_transformed_points(inside_points, pairing_matrices, rho_max)

    # Compute n_j(n_j - 1)/2 for each point
    n_j_values = [(idx, len(images)) for idx, images in classified_transformed_points.items()]

    # Step 5: Calculate the number of columns N based on the angular momentum parameter L
    N = (L + 1) ** 2  # Number of columns is (L + 1)^2
    print(f"Number of columns (N): {N}")

    # Step 6: Calculate the target number of rows M based on the desired over-constraint parameter c
    M_desired = int(c * N)
    print(f"Desired number of rows (M): {M_desired}")

    # Step 7: Select points to match the desired number of rows M
    selected_indices = select_points_for_c(n_j_values, M_desired)

    # Extract the selected points
    selected_points = [inside_points[idx] for idx in selected_indices]
    print(f"Selected {len(selected_points)} points for matrix computation.")

    # Step 8: Generate transformed points for the selected points
    selected_transformed_points = {idx: classified_transformed_points[idx] for idx in selected_indices}

    # Compute the actual number of rows M based on the selected points
    actual_M = sum(len(images) * (len(images) - 1) / 2 for images in selected_transformed_points.values())
    print(f"Actual number of rows (M): {actual_M}")

    # Convert the dictionary selected_transformed_points to a list of lists of tuples
    points_images = convert_to_points_images(selected_transformed_points)

    # Step 9: Construct matrix system
    _, _, matrix_system = generate_matrix_system(points_images, L, k_value)
    A = construct_numeric_matrix(matrix_system, k_value)

    # Step 10: Compare the actual M and N and the degree of over-constraint
    num_rows_A = A.shape[0]
    c_actual = num_rows_A / N
    print(f"Actual degree of over-constraint (c = M/N): {c_actual}")

    return actual_M, num_rows_A, c_actual

if __name__ == "__main__":
    # Example usage
    manifold_name = 'm188(-1,1)'  # Replace with your actual manifold name
    c = 70  # The desired degree of over-constraint
    num_points = 10000
    rho_max = 4.5  # Adjust as needed
    L = 11  # Angular momentum parameter
    k_value = 1.0  # Example k value for matrix construction

    # Run the computation with the degree of over-constraint parameter
    compute_n_j_and_matrix_with_c(manifold_name, c, num_points, rho_max, L, k_value)