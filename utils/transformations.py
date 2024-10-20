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


# Generate transformed points by applying the group generators
def generate_transformed_points(inside_points, pairing_matrices):
    classified_points = []
    inside_points = np.array(inside_points)
    
    for point in inside_points:
        hyperboloid_points = np.array([apply_so31_action(matrix, point) for matrix in pairing_matrices])
        klein_points = project_to_klein(hyperboloid_points)
        pseudo_spherical_points = klein_to_pseudo_spherical(klein_points)
        
        classified_points.append(pseudo_spherical_points)
    
    return classified_points

def convert_to_points_images(selected_transformed_points):
    """
    Converts the list of arrays to a list of lists of tuples.
    Each tuple represents a point in (rho, theta, phi) coordinates.
    """
    points_images = []
    
    for images in selected_transformed_points:
        point_list = [tuple(image) for image in images]
        points_images.append(point_list)
    
    return points_images