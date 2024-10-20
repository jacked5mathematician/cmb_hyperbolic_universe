import numpy as np
import mpmath as mp
from mpmath import hyp2f1, gamma, sqrt, sinh, cosh, pi
from functools import lru_cache
from scipy.special import lpmv
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import njit

epsilon = 1e-12  # Small number to avoid division by zero
mp.dps = 5

# Radial function using Phi^nu_l (Hyperspherical Bessel function for K=-1)
def Phi_nu_l(nu, l, chi):
    """Compute the hyperspherical Bessel function Phi^nu_l(chi) for K = -1."""
    nu = float(nu)
    chi = float(chi)
    
    # Compute N^nu_l as a product from n=1 to l of (nu^2 + n^2)
    N_nu_l = np.prod([nu**2 + n**2 for n in range(1, l + 1)])
    
    # The Legendre function P^{-1/2-l}_{-1/2+i*nu}(cosh(chi)) using hyp2f1
    def legendre_P(alpha, beta, x):
        if x < 1 + epsilon:
            x = 1 + epsilon
        
        prefactor = ((x + 1) / (x - 1))**(beta / 2)
        hyp_part = hyp2f1(alpha + 1, -alpha, 1 - beta, (1 - x) / 2)
        return prefactor * hyp_part / gamma(1 - beta)
    
    alpha = -1/2 + 1j * nu
    beta = -1/2 - l
    x = cosh(chi)
    
    legendre_value = legendre_P(alpha, beta, x)
    
    Phi = sqrt(pi * N_nu_l / (2 * sinh(chi))) * legendre_value
    return Phi

def normalization_constant(l, m):
    return np.sqrt((2 * l + 1) / (4 * np.pi) * mp.factorial(l - np.abs(m)) / mp.factorial(l + np.abs(m)))

# Real-valued spherical harmonics using explicit formulas
def Y_lm_real(l, m, theta, phi):
    if m > 0:
        return np.sqrt(2) * normalization_constant(l, m) * np.cos(m * phi) * lpmv(m, l, np.cos(theta))
    elif m < 0:
        return np.sqrt(2) * normalization_constant(l, -m) * np.sin(-m * phi) * lpmv(-m, l, np.cos(theta))
    else:
        return normalization_constant(l, 0) * lpmv(0, l, np.cos(theta))

# Global cache dictionaries for Q functions
phi_cache = {}
y_lm_cache = {}

def Phi_nu_l_cached(nu, l, chi):
    """Cached version of the hyperspherical Bessel function Phi^nu_l(chi) for K = -1."""
    key = (nu, l, chi)
    if key not in phi_cache:
        phi_cache[key] = Phi_nu_l(nu, l, chi)
    return phi_cache[key]

def Y_lm_real_cached(l, m, theta, phi):
    """Cached version of the real-valued spherical harmonics function."""
    key = (l, m, theta, phi)
    if key not in y_lm_cache:
        y_lm_cache[key] = Y_lm_real(l, m, theta, phi)
    return y_lm_cache[key]

# Full eigenfunction Q_{k,l,m}(rho, theta, phi) using Phi_nu_l
def Q_k_lm(k, l, m, rho, theta, phi):
    nu = k
    Phi_nu_l_value = Phi_nu_l_cached(nu, l, rho)
    Y_lm_real_value = Y_lm_real_cached(l, m, theta, phi)
    return Phi_nu_l_value * Y_lm_real_value

def Q_k_lm_cached(k, l, m, rho, theta, phi):
    return Q_k_lm(k, l, m, rho, theta, phi)

# Parallelize the expensive computation of Phi_nu_l and Y_lm_real
# Parallelize the Q_k_lm computation


def parallel_Phi_Y_lm(lm_pairs, k_value, all_images, n_jobs=1):
    """Parallel computation of Phi_nu_l and Y_lm_real for all lm_pairs and images."""

    def compute_Phi_Y_lm(args):
        l, m, rho, theta, phi = args
        nu = k_value
        Phi_nu_l_val = Phi_nu_l_cached(nu, l, rho)
        Y_lm_real_val = Y_lm_real_cached(l, m, theta, phi)
        return (rho, theta, phi), Phi_nu_l_val * Y_lm_real_val

    # Prepare arguments for parallel computation
    tasks = [(l, m, rho, theta, phi) for l, m in lm_pairs for rho, theta, phi in all_images]

    # Parallelizing the computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_Phi_Y_lm)(args) for args in tasks
    )

    # Convert the results into a dictionary
    q_values_dict = {result[0]: result[1] for result in results}
    return q_values_dict

def parallel_Q_k_lm_compute(lm_pairs, k_value, points_images):
    """Compute Q_k_lm for all lm pairs and points_images in parallel."""
    all_images = [(rho, theta, phi) for point_images in points_images for (rho, theta, phi) in point_images]

    # Use joblib to parallelize the computations of Phi_nu_l and Y_lm
    q_values_dict = parallel_Phi_Y_lm(lm_pairs, k_value, all_images)

    return q_values_dict