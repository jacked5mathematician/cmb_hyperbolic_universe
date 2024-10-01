# utils/domain_builder.py

import numpy as np
import snappy
from tqdm import tqdm
import random

# Converts pairing matrices to proper 4x4 NumPy arrays
def convert_to_4x4_matrices(pairs):
    matrices = []
    for pair in tqdm(pairs, desc="Converting Pairing Matrices"):
        matrix = np.array(list(pair), dtype=float)
        if matrix.shape == (16,):  
            matrix = matrix.reshape(4, 4)
        matrices.append(matrix)
    return matrices

def build_dirichlet_domain(manifold_name):
    """Computes the Dirichlet domain of the manifold and returns necessary data."""
    M = snappy.Manifold(manifold_name)
    
    try:
        D = M.dirichlet_domain()
    except RuntimeError as e:
        print(f"Failed to compute Dirichlet domain: {e}")
        return None
    
    # Extract pairing matrices, vertices, and faces
    pairing_matrices = D.pairing_matrices()
    pairing_matrices = convert_to_4x4_matrices(pairing_matrices)
    vertex_details = D.vertex_list(details=True)
    vertices = np.array([list(v['position']) for v in vertex_details], dtype=float)
    faces = D.face_list()
    
    return vertices, faces, pairing_matrices

# utils/domain_builder.py

def generate_random_points_in_domain(vertices, num_points):
    """Generates random points within the bounding box of the Dirichlet domain."""
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    
    # Generate random points within the bounds of the vertices
    print(f"Generating {num_points} random points...")
    points = np.random.uniform(min_corner, max_corner, (num_points, 3))
    
    return points

# utils/domain_builder.py

def filter_points_in_domain(points, faces, vertices):
    """Filters points to find those inside the Dirichlet domain by checking against the faces."""
    inside_points = []
    print("Filtering points inside the domain...")
    
    for point in tqdm(points, desc="Filtering Points Inside Domain"):
        is_inside = True
        for face in faces:
            face_vertices = vertices[face['vertex_indices']]
            if len(face_vertices) >= 3:
                v1 = face_vertices[1] - face_vertices[0]
                v2 = face_vertices[2] - face_vertices[0]
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                if np.dot(point - face_vertices[0], normal) > 0:
                    is_inside = False
                    break
        if is_inside:
            inside_points.append(point)
    
    return np.array(inside_points)

def select_points(points, num_points_to_use):
    """Selects a specified number of points from the list of points inside the domain."""
    if num_points_to_use is None or num_points_to_use > len(points):
        num_points_to_use = len(points)
    
    selected_points = random.sample(list(points), num_points_to_use)
    print(f"Selected {num_points_to_use} points for the system.")
    
    return selected_points