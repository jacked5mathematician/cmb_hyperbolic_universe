import numpy as np

from utils.special_functions import Q_k_lm


def generate_matrix_system(points_images, L, k_value):
    """
    Build matrix A(k) where columns correspond to (l, m) with 0<=l<=L and rows
    correspond to unordered pairs of images for each base point.

    points_images: List[List[(rho, theta, phi)]]
    """
    lm_pairs = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]
    N = (L + 1) ** 2
    rows = []
    for images in points_images:
        n_j = len(images)
        if n_j < 2:
            continue
        for alpha in range(n_j):
            for beta in range(alpha + 1, n_j):
                rho_a, th_a, ph_a = images[alpha]
                rho_b, th_b, ph_b = images[beta]
                row = [
                    Q_k_lm(k_value, l, m, rho_a, th_a, ph_a)
                    - Q_k_lm(k_value, l, m, rho_b, th_b, ph_b)
                    for l, m in lm_pairs
                ]
                rows.append(row)
    A = np.array(rows, dtype=np.complex128)
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
