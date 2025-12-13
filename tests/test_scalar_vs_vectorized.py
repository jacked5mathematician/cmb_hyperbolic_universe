import numpy as np

from utils.sys_generation import generate_matrix_system, generate_matrix_system_scalar


def test_scalar_matches_vectorized_small_case():
    # Deterministic tiny dataset: one basepoint with four images
    points_images = [[
        (0.9, 0.4, 0.1),
        (1.1, 0.6, 0.2),
        (1.4, 0.2, 0.7),
        (1.8, 1.0, 0.9),
    ]]
    L = 3
    k_value = 2.3

    Mv, Nv, Av = generate_matrix_system(points_images, L, k_value)
    Ms, Ns, As = generate_matrix_system_scalar(points_images, L, k_value)

    assert (Mv, Nv) == (Ms, Ns)
    assert Av.shape == As.shape == (Mv, Nv)

    rel = np.linalg.norm(Av - As, ord="fro") / (np.linalg.norm(As, ord="fro") + 1e-30)
    assert rel < 1e-8, f"Relative difference too large: {rel}" 
