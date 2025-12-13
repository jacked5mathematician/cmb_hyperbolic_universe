import numpy as np

from utils.special_functions import Q_k_lm, Q_k_lm_vectorized


def generate_matrix_system(points_images, L, k_value):
    """
    Build matrix A(k) where columns correspond to (l, m) with 0<=l<=L and rows
    correspond to unordered pairs of images for each base point.

    points_images: List[List[(rho, theta, phi)]]
    
    Uses vectorized computation for performance:
    1. Build lm_pairs once per L
    2. For each basepoint, compute Q-values for all images in batch
    3. Use np.triu_indices for pairwise differences
    """
    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]
    N = (L + 1) ** 2
    
    # Collect all row blocks for each basepoint
    row_blocks = []
    
    for images in points_images:
        n_j = len(images)
        if n_j < 2:
            continue
        
        # Convert images to numpy array shape (n_j, 3)
        try:
            images_array = np.array(images, dtype=np.float64)
            if images_array.shape != (n_j, 3):
                raise ValueError(f"Expected images array shape ({n_j}, 3), got {images_array.shape}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert images to array: {e}. Expected list of (rho, theta, phi) tuples.")
        
        # Compute Q-values for all images and all (l,m) in one batch: shape (n_j, N)
        Q_matrix = Q_k_lm_vectorized(k_value, lm_pairs, images_array)
        
        # Compute pairwise differences for all pairs (alpha, beta) with alpha < beta
        # Using upper triangle indices
        idx_i, idx_j = np.triu_indices(n_j, k=1)
        
        # Build difference matrix: Q[alpha] - Q[beta] for each pair
        # Shape: (n_pairs, N) where n_pairs = n_j*(n_j-1)/2
        A_block = Q_matrix[idx_i, :] - Q_matrix[idx_j, :]
        
        row_blocks.append(A_block)
    
    # Stack all row blocks
    if row_blocks:
        A = np.vstack(row_blocks)
    else:
        A = np.zeros((0, N), dtype=np.complex128)
    
    M_expected = sum(len(imgs) * (len(imgs) - 1) // 2 for imgs in points_images)
    assert A.shape[0] == M_expected, f"M mismatch: expected {M_expected}, got {A.shape[0]}"
    assert A.shape[1] == N, f"N mismatch: expected {N}, got {A.shape[1]}"
    return A.shape[0], A.shape[1], A


def construct_numeric_matrix(matrix_system, k_value=None):
    """
    Legacy compatibility helper. Converts list-like systems to numpy arrays and
    returns pre-existing numeric arrays unchanged.
    """
    if isinstance(matrix_system, np.ndarray):
        return matrix_system
    return np.array(matrix_system, dtype=np.complex128)
