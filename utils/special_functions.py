import numpy as np
import mpmath as mp
from mpmath import hyp2f1, gamma, sqrt, sinh, cosh, pi
from functools import lru_cache
from scipy.special import lpmv

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

# Global cache dictionaries
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
    nu = np.sqrt(k**2 + 1)
    Phi_nu_l_value = Phi_nu_l_cached(nu, l, rho)
    Y_lm_real_value = Y_lm_real_cached(l, m, theta, phi)
    return Phi_nu_l_value * Y_lm_real_value

@lru_cache(maxsize=None)
def Q_k_lm_cached(k, l, m, rho, theta, phi):
    return Q_k_lm(k, l, m, rho, theta, phi)