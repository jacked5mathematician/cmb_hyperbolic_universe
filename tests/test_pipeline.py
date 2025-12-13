import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.cutoffs import compute_rho_cutoffs
from utils.ghosts import enumerate_ghost_images
from utils.sys_generation import generate_matrix_system
from utils.transformations import poincare_distance


def test_poincare_distance_origin_identity():
    r = 0.3
    d = poincare_distance([0.0, 0.0, 0.0], [r, 0.0, 0.0])
    assert np.isclose(d, 2 * np.arctanh(r))


def test_compute_rho_cutoffs_valid():
    rho_min, rho_max = compute_rho_cutoffs(k=1.0, L=2, l_min=0, threshold=0.25)
    assert 0 <= rho_min < rho_max


def test_enumerate_ghost_images_satisfies_count_and_window():
    rho_min, rho_max = 0.0, 1.0
    images = enumerate_ghost_images(
        "m003(-2,3)",
        base_points=[[0.0, 0.0, 0.0]],
        rho_min=rho_min,
        rho_max=rho_max,
        max_word_length=1,
    )
    assert images, "No images returned"
    first = images[0]
    assert len(first) >= 10
    assert all(rho_min <= img[0] <= rho_max for img in first)


def test_generate_matrix_system_shapes():
    points_images = [
        [(0.1, 0.0, 0.0), (0.2, 0.1, 0.0), (0.3, 0.2, 0.1)],
        [(0.4, 0.3, 0.2), (0.5, 0.4, 0.3)],
    ]
    L = 1
    M, N, A = generate_matrix_system(points_images, L, k_value=1.0)
    expected_M = sum(len(imgs) * (len(imgs) - 1) // 2 for imgs in points_images)
    expected_N = (L + 1) ** 2
    assert M == expected_M
    assert N == expected_N
    assert A.shape == (expected_M, expected_N)
