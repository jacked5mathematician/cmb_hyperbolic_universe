import numpy as np


# Transformation application in SO(3,1)
def apply_so31_action(matrix, point):
    norm_squared = np.dot(point, point)
    X0 = (1 + norm_squared) / (1 - norm_squared)
    X1, X2, X3 = 2 * point / (1 - norm_squared)
    hyperboloid_point = np.array([X0, X1, X2, X3], dtype=float)
    transformed_point = np.dot(matrix, hyperboloid_point)
    return transformed_point

# Project back to Klein coordinates
def project_to_klein(transformed_points):
    if transformed_points.ndim == 1:
        X0, X1, X2, X3 = transformed_points
        return np.array([X1 / X0, X2 / X0, X3 / X0], dtype=float)
    elif transformed_points.ndim == 2 and transformed_points.shape[1] == 4:
        X0 = transformed_points[:, 0]
        X1 = transformed_points[:, 1]
        X2 = transformed_points[:, 2]
        X3 = transformed_points[:, 3]
        return np.column_stack((X1 / X0, X2 / X0, X3 / X0))
    else:
        raise ValueError(f"Unexpected shape for transformed_points: {transformed_points.shape}")

# Convert Klein to pseudo-spherical coordinates (rho, theta, phi)
def klein_to_pseudo_spherical(points):
    pseudo_spherical_points = []
    for point in points:
        p_x, p_y, p_z = point
        norm_squared = p_x**2 + p_y**2 + p_z**2
        X0 = (1 + norm_squared) / (1 - norm_squared)
        X1 = 2 * p_x / (1 - norm_squared)
        X2 = 2 * p_y / (1 - norm_squared)
        X3 = 2 * p_z / (1 - norm_squared)
        rho = np.arccosh(X0)
        sinh_rho = np.sinh(rho)
        theta = np.arccos(X3 / sinh_rho) if sinh_rho != 0 else 0
        phi = np.arctan2(X2, X1)
        pseudo_spherical_points.append([rho, theta, phi])
    return np.array(pseudo_spherical_points, dtype=float)


def klein_to_poincare(points):
    """
    Convert Klein coordinates back to Poincaré ball coordinates.
    """
    pts = np.array(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    norm_sq = np.sum(pts**2, axis=1)
    # Guard against slight numerical overflow
    norm_sq = np.clip(norm_sq, 0.0, 1.0 - 1e-12)
    scale = 1.0 / (1.0 + np.sqrt(1.0 - norm_sq))
    return pts * scale[:, None]


def convert_to_points_images(selected_transformed_points):
    """
    Combines selected_transformed_points such that each entry contains only the images.

    Parameters:
    - selected_transformed_points: Dictionary where keys are point indices and values are lists of images.

    Returns:
    - points_images: List of images for each original point index.
    """
    points_images = []
    for idx, images_list in selected_transformed_points.items():
        # Ensure the value is correctly formatted
        if not isinstance(images_list, list):
            print(f"Warning: Expected list for point {idx}, got {type(images_list)}")
            images_list = []

        # Extract points if 'point' key exists
        images = [img['point'] for img in images_list if 'point' in img]
        points_images.append(images)

    return points_images

def poincare_to_pseudo_spherical(points):
    """
    Transform a list of 3D Poincaré ball coordinates into pseudo-spherical coordinates.
    
    Parameters:
        points (list or np.ndarray): List or array of points in the Poincar�� ball model,
                                     where each point is [x, y, z].
    
    Returns:
        np.ndarray: Array of points in pseudo-spherical coordinates [rho, theta, phi].
    """
    points = np.array(points)
    norm_squared = np.sum(points**2, axis=1)
    valid_indices = norm_squared < 1
    valid_points = points[valid_indices]
    norm_squared = norm_squared[valid_indices]

    X0 = (1 + norm_squared) / (1 - norm_squared)
    X = 2 * valid_points / (1 - norm_squared[:, np.newaxis])
    rho = np.arccosh(X0)
    
    # Compute theta using atan2 for numerical stability
    # theta = arctan2(sqrt(X1^2 + X2^2), X3)
    # This avoids division by sinh(rho) which can be near zero
    xy_norm = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    theta = np.arctan2(xy_norm, X[:, 2])
    
    phi = np.arctan2(X[:, 1], X[:, 0])

    pseudo_spherical_points = np.column_stack((rho, theta, phi))
    return pseudo_spherical_points

def poincare_distance(point1, point2):
    """
    Compute the hyperbolic distance between two points in the Poincaré ball model.
    
    Parameters:
        point1 (list/tuple): Coordinates [x1, y1, z1] of the first point in the Poincaré ball.
        point2 (list/tuple): Coordinates [x2, y2, z2] of the second point in the Poincaré ball.
    
    Returns:
        float: The hyperbolic distance between the two points.
    """
    # Compute Euclidean norms of the points
    norm1_squared = sum(coord**2 for coord in point1)
    norm2_squared = sum(coord**2 for coord in point2)

    if norm1_squared >= 1 or norm2_squared >= 1:
        raise ValueError("One or both points are outside the Poincaré ball (norm >= 1).")

    diff_squared = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))

    denominator = (1 - norm1_squared) * (1 - norm2_squared)
    cosh_arg = 1 + (2 * diff_squared) / denominator
    hyperbolic_distance = np.arccosh(cosh_arg)

    return hyperbolic_distance
