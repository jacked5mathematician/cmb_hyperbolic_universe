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

import numpy as np
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
from bisect import bisect_left, bisect_right
import numpy as np
from numba import njit, prange


@njit
def binary_search_filter(images_rho_sorted, rho_min, rho_max):
    """Binary search to filter images efficiently within a rho range."""
    left_idx = 0
    right_idx = len(images_rho_sorted)

    # Binary search for the left bound (first index where images_rho_sorted[i] >= rho_min)
    while left_idx < right_idx:
        mid = (left_idx + right_idx) // 2
        if images_rho_sorted[mid][0] < rho_min:  # Assuming first element is rho
            left_idx = mid + 1
        else:
            right_idx = mid

    # Binary search for the right bound (first index where images_rho_sorted[i] > rho_max)
    right_idx = len(images_rho_sorted)
    left_bound = left_idx
    while left_idx < right_idx:
        mid = (left_idx + right_idx) // 2
        if images_rho_sorted[mid][0] > rho_max:  # Assuming first element is rho
            right_idx = mid
        else:
            left_idx = mid + 1

    # Now slice the array based on valid bounds
    return images_rho_sorted[left_bound:right_idx]

@njit(parallel=True)
def process_images(images_list, rho_min, rho_max, min_images):
    """Parallelized function to process image filtering."""
    total_rows = 0
    valid_points = 0
    d = len(images_list)
    
    for idx in prange(d):
        images = images_list[idx]
        filtered_images = binary_search_filter(images, rho_min, rho_max)
        num_images = len(filtered_images)

        # Only count points with enough images
        if num_images >= min_images:
            valid_points += 1
            total_rows += num_images * (num_images - 1) // 2

    return total_rows, valid_points

def determine_tiling_radius(inside_points, pairing_matrices, L, c, min_images=5, tolerance=0.1, initial_step_size=0.05, min_step_size=0.0005):
    """
    Adjust rho_min and rho_max to include enough images such that the total number of rows matches the target M,
    while ensuring each point has at least the minimum number of images within the range [rho_min, rho_max].
    """

    #print(f"Starting determine_tiling_radius with min_images={min_images}, tolerance={tolerance}, initial_step_size={initial_step_size}")
    
    # Step 1: Compute the target number of rows M based on L and c
    M_desired, N = compute_target_M(L, c)
    #print(f"Target number of rows (M_desired): {M_desired}")
    
    # Step 2: Compute the initial average rho value
    classified_transformed_points = generate_transformed_points(inside_points, pairing_matrices)
    all_rho_values = np.array([image[0] for images in classified_transformed_points.values() for image in images])
    avg_rho = np.mean(all_rho_values)
    #print(f"Initial average rho value: {avg_rho}")
    
    # Step 3: Initialize rho_min and rho_max at avg_rho
    rho_min = avg_rho
    rho_max = avg_rho
    step_size = initial_step_size

    iteration = 0
    max_iterations = 100
    start_time = time.time()

    # Convert dictionary to a list of lists of arrays (structure that Numba supports)
    images_list = [np.array([img for img in images]) for images in classified_transformed_points.values()]

    # Step 4: Iterate to adjust rho_min and rho_max until M_desired rows are achieved
    while iteration < max_iterations:
        iteration += 1

        # Process the points and count valid ones in parallel
        total_rows, valid_points = process_images(images_list, rho_min, rho_max, min_images)

        #print(f"Iteration {iteration}: rho_min = {rho_min}, rho_max = {rho_max}, valid points = {valid_points}/{len(inside_points)}, total_rows = {total_rows}/{M_desired}") #Uncomment for debugging

        # Stop if we satisfy both conditions: minimum images for each point and M_desired rows
        if total_rows >= M_desired:
            if total_rows > M_desired * 1.1:
                # If overshooting by more than 10%, reduce the range drastically
                step_size = max(step_size / 2, min_step_size)
                rho_min += step_size
                rho_max -= step_size
                #print(f"Overshoot detected, reducing range: rho_min = {rho_min}, rho_max = {rho_max}, new step_size = {step_size}") #Uncomment for debugging
            else:
                break
        else:
            # If we are below M_desired, we expand the range but more conservatively
            rho_min -= step_size
            rho_max += step_size

            # Dynamically adjust step size based on proximity to target rows
            if total_rows > M_desired * 0.9:
                step_size = max(step_size / 2, min_step_size)  # Reduce step size when close
            else:
                step_size = min(step_size * 1.05, initial_step_size)  # Conservative expansion

        # Break if max iterations reached
        if iteration >= max_iterations:
            print(f"Warning: Max iterations reached. Could not fully satisfy the M_desired condition or minimum image requirement.")
            break

    #print(f"Final rho_min: {rho_min}, Final rho_max: {rho_max} (Time: {time.time() - start_time:.2f} seconds)")

    # Step 5: Filter images to ensure they are within the final [rho_min, rho_max]
    final_points = {}
    for idx, images in classified_transformed_points.items():
        filtered_images = binary_search_filter(images, rho_min - tolerance, rho_max + tolerance)
        if len(filtered_images) > 0:
            final_points[idx] = filtered_images

    return final_points, rho_min, rho_max, valid_points, M_desired