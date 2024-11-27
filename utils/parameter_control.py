# parameter_control.py
import numpy as np
import time

from .transformations import poincare_distance

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
        current_M += n_j * (n_j - 1) // 2  # Add the contribution to M
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

def filter_points_for_overconstraint(final_points, data_with_distances, M_desired):
    """
    Filter and select points to match the desired number of rows M.

    Parameters:
    - final_points: Filtered transformed points dictionary.
    - data_with_distances: The original data dictionary.
    - M_desired: Target number of rows M.

    Returns:
    - selected_points: The selected points for matrix computation.
    - selected_transformed_points: The transformed points for the selected points.
    - points_images: List of tuples where each tuple contains an original point and its images.
    """
    pseudospherical_points = data_with_distances['pseudospherical_points']

    # Calculate n_j for each point
    n_j_values = [(idx, len(images)) for idx, images in final_points.items()]
    
    # Select points to match the desired number of rows M
    selected_indices = select_points_for_c(n_j_values, M_desired)
    
    # Extract the selected points and their transformed images
    selected_points = [pseudospherical_points[idx] for idx in selected_indices]
    selected_transformed_points = {idx: final_points[idx] for idx in selected_indices}
    
    # Create points_images in the desired format
    points_images = [(pseudospherical_points[idx], [img['point'] for img in final_points[idx]]) for idx in selected_indices]
    
    return selected_points, selected_transformed_points, points_images


import time

import numpy as np
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
from bisect import bisect_left, bisect_right
import numpy as np
from numba import njit, prange


def binary_search_filter(images_sorted_by_distance, rho_min, rho_max):
    """Binary search to filter images efficiently within a rho range."""
    distances = np.array([img['distance'] for img in images_sorted_by_distance])
    left_idx = np.searchsorted(distances, rho_min, side='left')
    right_idx = np.searchsorted(distances, rho_max, side='right')
    return images_sorted_by_distance[left_idx:right_idx]

def process_images(images_list, rho_min, rho_max, min_images):
    total_rows = 0
    valid_points = 0

    for images in images_list:
        # images is a list of transformed points with 'distance' keys
        distances = np.array([img['distance'] for img in images])
        # Filter images within [rho_min, rho_max]
        filtered_indices = np.where((distances >= rho_min) & (distances <= rho_max))[0]
        n_j = len(filtered_indices)
        if n_j >= min_images:
            total_rows += n_j * (n_j - 1) // 2
            valid_points += 1

    return total_rows, valid_points

def determine_tiling_radius(data_with_distances, L, c, min_images=5, tolerance=0.1, initial_step_size=0.05, min_step_size=0.0005):
    """
    Adjust rho_min and rho_max to include enough images such that the total number of rows matches the target M,
    while ensuring each point has at least the minimum number of images within the range [rho_min, rho_max].
    """

    # Extract data from the provided dictionary
    pseudospherical_points = data_with_distances['pseudospherical_points']
    distances = data_with_distances['distances']
    transformed_pseudospherical_points_list = data_with_distances['transformed_pseudospherical_points']
    transformed_distances_list = data_with_distances['transformed_distances']
    matrix_indices_list = data_with_distances['matrix_indices']

    # Step 1: Compute the target number of rows M based on L and c
    M_desired, N = compute_target_M(L, c)
    
    # Step 2: Compute the initial average distance from your data
    all_transformed_distances = np.concatenate(transformed_distances_list)
    avg_distance = np.mean(all_transformed_distances)
    
    # Step 3: Initialize rho_min and rho_max at avg_distance
    rho_min = avg_distance
    rho_max = avg_distance
    step_size = initial_step_size

    iteration = 0
    max_iterations = 100
    start_time = time.time()

    # Prepare images_list: a list of lists of transformed points with their distances and matrix indices
    images_list = []
    num_points = len(pseudospherical_points)
    num_generators = len(transformed_pseudospherical_points_list)

    for i in range(num_points):
        images = []
        for j in range(num_generators):
            point = transformed_pseudospherical_points_list[j][i]
            distance = transformed_distances_list[j][i]
            matrix_index = matrix_indices_list[j]  # Assuming matrix_indices_list[j] is the correct index
            images.append({'point': point, 'distance': distance, 'matrix_index': matrix_index})
        # Sort images by distance
        images_sorted = sorted(images, key=lambda x: x['distance'])
        images_list.append(images_sorted)

    # Step 4: Iterate to adjust rho_min and rho_max until M_desired rows are achieved
    while iteration < max_iterations:
        iteration += 1

        # Process the points and count valid ones
        total_rows, valid_points = process_images(images_list, rho_min, rho_max, min_images)

        # Stop if we satisfy both conditions: minimum images for each point and M_desired rows
        if total_rows >= M_desired:
            if total_rows > M_desired * 1.1:
                # If overshooting by more than 10%, reduce the range drastically
                step_size = max(step_size / 2, min_step_size)
                rho_min += step_size
                rho_max -= step_size
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

    # Step 5: Filter images to ensure they are within the final [rho_min, rho_max]
    final_points = {}
    for idx, images in enumerate(images_list):
        # images is a sorted list of transformed points with 'distance' keys
        filtered_images = binary_search_filter(images, rho_min - tolerance, rho_max + tolerance)
        if len(filtered_images) >= min_images:
            # Sort filtered images by matrix_index
            filtered_images_sorted = sorted(filtered_images, key=lambda x: x['matrix_index'])
            final_points[idx] = filtered_images_sorted

    return final_points, rho_min, rho_max, valid_points