# parameter_control.py

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

def compute_target_M(L, c):
    """
    Compute the target number of rows M based on the degree of over-constraint c and the angular momentum L.

    Parameters:
    - L: Angular momentum parameter.
    - c: The degree of over-constraint (c = M / N).

    Returns:
    - M: Target number of rows.
    - N: Number of columns in the matrix system.
    """
    N = (L + 1) ** 2  # Number of columns is (L + 1)^2
    M_desired = int(c * N)  # Desired number of rows
    return M_desired, N

def filter_points_for_overconstraint(classified_transformed_points, inside_points, M_desired):
    """
    Filter and select points to match the desired number of rows M.

    Parameters:
    - classified_transformed_points: Transformed points dictionary.
    - inside_points: Points inside the domain.
    - M_desired: Target number of rows M.

    Returns:
    - selected_points: The selected points for matrix computation.
    - selected_transformed_points: The transformed points for the selected points.
    """
    # Calculate n_j for each point
    n_j_values = [(idx, len(images)) for idx, images in classified_transformed_points.items()]
    
    # Select points to match the desired number of rows M
    selected_indices = select_points_for_c(n_j_values, M_desired)
    
    # Extract the selected points and their transformed images
    selected_points = [inside_points[idx] for idx in selected_indices]
    selected_transformed_points = {idx: classified_transformed_points[idx] for idx in selected_indices}
    
    return selected_points, selected_transformed_points