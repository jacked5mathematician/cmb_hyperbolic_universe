# parameter_control.py
import numpy as np

from .transformations import generate_transformed_points

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


import time

def determine_tiling_radius(inside_points, pairing_matrices, L, c, min_images=5, tolerance=0.1, initial_step_size=0.05, min_step_size=0.0005):
    """
    Adjust rho_min and rho_max to include enough images such that the total number of rows matches the target M,
    while ensuring each point has at least the minimum number of images within the range [rho_min, rho_max].

    This enhanced algorithm dynamically adjusts the step size for both rho_min and rho_max based on how close 
    we are to the target number of rows (M_desired) and how far the current range is from including enough points.

    Parameters:
    - inside_points: List of points inside the domain.
    - pairing_matrices: List of pairing matrices.
    - L: Angular momentum parameter.
    - c: Degree of over-constraint.
    - min_images: Minimum number of images required per point.
    - tolerance: Allow small deviations in rho.
    - initial_step_size: Starting increment for adjusting rho_min and rho_max.
    - min_step_size: Minimum allowed step size for fine-tuning adjustments.

    Returns:
    - classified_transformed_points: Dictionary of points with their corresponding images.
    - rho_min: The calculated minimum rho value.
    - rho_max: The calculated maximum rho value.
    """

    print(f"Starting determine_tiling_radius with min_images={min_images}, tolerance={tolerance}, initial_step_size={initial_step_size}")

    # Step 1: Compute the target number of rows M based on L and c
    M_desired, N = compute_target_M(L, c)
    print(f"Target number of rows (M_desired): {M_desired}")

    # Step 2: Compute the initial average rho value
    classified_transformed_points = generate_transformed_points(inside_points, pairing_matrices)
    all_rho_values = [image[0] for images in classified_transformed_points.values() for image in images]
    avg_rho = np.mean(all_rho_values)
    print(f"Initial average rho value: {avg_rho}")

    # Step 3: Initialize rho_min and rho_max at avg_rho
    rho_min = avg_rho
    rho_max = avg_rho
    step_size = initial_step_size

    iteration = 0
    max_iterations = 100
    start_time = time.time()
    overshoot = False  # Track whether we overshot the target

    # Step 4: Iterate to adjust rho_min and rho_max until M_desired rows are achieved
    while iteration < max_iterations:
        iteration += 1

        # Generate transformed points with the current rho_min and rho_max
        classified_transformed_points = generate_transformed_points(inside_points, pairing_matrices)
        
        total_rows = 0
        valid_points = 0

        for images in classified_transformed_points.values():
            # Filter images based on the current range
            filtered_images = [img for img in images if rho_min <= img[0] <= rho_max]
            num_images = len(filtered_images)
            
            # Each point must have at least `min_images` images in the range
            if num_images >= min_images:
                valid_points += 1
                # Each point contributes n_j(n_j - 1) / 2 rows
                total_rows += num_images * (num_images - 1) // 2

        print(f"Iteration {iteration}: rho_min = {rho_min}, rho_max = {rho_max}, valid points = {valid_points}/{len(inside_points)}, total_rows = {total_rows}/{M_desired}")

        # Stop if we satisfy both conditions: minimum images for each point and M_desired rows
        if total_rows >= M_desired:
            if total_rows > M_desired * 1.1:
                # If overshooting by more than 10%, reduce the range drastically
                step_size = max(step_size / 2, min_step_size)
                rho_min += step_size
                rho_max -= step_size
                print(f"Overshoot detected, reducing range: rho_min = {rho_min}, rho_max = {rho_max}, new step_size = {step_size}")
            else:
                break
        else:
            # If we are below M_desired, we expand the range but more conservatively
            rho_min -= step_size
            rho_max += step_size

            # Dynamically adjust step size based on proximity to target rows
            if total_rows > M_desired * 0.9:
                # Close to target, reduce step size for fine-tuning
                step_size = max(step_size / 2, min_step_size)
            elif total_rows < M_desired * 0.5:
                # Far from target, avoid overly aggressive expansion
                step_size = min(step_size * 1.1, initial_step_size)  # Slightly increase step size
            else:
                # Adjust step size conservatively when in mid-range
                step_size = min(step_size * 1.05, initial_step_size)

        # Break if max iterations reached
        if iteration >= max_iterations:
            print(f"Warning: Max iterations reached. Could not fully satisfy the M_desired condition or minimum image requirement.")
            break

    print(f"Final rho_min: {rho_min}, Final rho_max: {rho_max} (Time: {time.time() - start_time:.2f} seconds)")

    # Step 5: Filter images to ensure they are within the final [rho_min, rho_max]
    final_points = {}
    for idx, images in classified_transformed_points.items():
        filtered_images = [image for image in images if (rho_min - tolerance) <= image[0] <= (rho_max + tolerance)]
        if len(filtered_images) > 0:
            final_points[idx] = filtered_images

    return final_points, rho_min, rho_max, valid_points