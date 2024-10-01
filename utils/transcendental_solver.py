import cProfile
import pstats
import io
import matplotlib.pyplot as plt
import scipy.optimize
import snappy
import numpy as np
import sympy as sp
from scipy.special import sph_harm
from scipy.linalg import svd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
from mpmath import hyp2f1, gamma, sqrt, sinh, cosh, pi
import mpmath as mp
from functools import lru_cache
import random
from scipy.special import lpmv, sph_harm


## Your Phi_nu_l implementation
def Phi_nu_l(nu, l, chi):
    """Compute the hyperspherical Bessel function Phi^nu_l(chi) for K = -1."""
    nu = float(nu)
    chi = float(chi)
    
    # Compute N^nu_l as a product from n=1 to l of (nu^2 + n^2)
    N_nu_l = np.prod([nu**2 + n**2 for n in range(1, l + 1)])
    
    # The Legendre function P^{-1/2-l}_{-1/2+i*nu}(cosh(chi)) using hyp2f1
    def legendre_P(alpha, beta, x):
        epsilon = 1e-12
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

# Transcendental equation using the real part of Phi_nu_l
def transcendental_eq(chi, k, ell):
    """Equation to find the root for rho_max given k and ell using the real part of Phi_nu_l."""
    nu = np.sqrt(k**2 + 1)
    # Take the real part of Phi_nu_l and subtract 0.25 for the transcendental equation
    return float(Phi_nu_l(nu, ell, chi).real * np.sinh(chi)) - 0.25

# Function to find rho_max for a given k and ell
def find_rho_max(k, ell):
    """Find the solution to the transcendental equation for rho_max using Phi_nu_l."""
    try:
        # Find the first positive root numerically
        rho_max = scipy.optimize.brentq(transcendental_eq, 0.1,1050, args=(k, ell))  # Adjust bounds as necessary
    except ValueError:
        # If no root is found in the given range, set rho_max to a large default value
        rho_max = 10.0
    return rho_max

# Define range for k values
k_values = np.linspace(1, 20, 500)  # Define a range of k values

# Choose l_min or l_max
l_min = 15  # Adjust as needed
l_max = 20  # Adjust as needed

# Find rho_max solutions for each k and l_min
rho_max_solutions_l_min = [find_rho_max(k, l_min) for k in k_values]

# Find rho_max solutions for each k and l_max
rho_max_solutions_l_max = [find_rho_max(k, l_max) for k in k_values]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, rho_max_solutions_l_min, label=f"$\\rho_{{max}}$ for $\\ell_{{min}} = {l_min}$", color='blue')
plt.plot(k_values, rho_max_solutions_l_max, label=f"$\\rho_{{max}}$ for $\\ell_{{max}} = {l_max}$", color='red')
plt.xlabel("$k$")
plt.ylabel("$\\rho_{max}$")
plt.title("$\\rho_{max}$ Solutions vs $k$")
plt.grid(True)
plt.legend()
plt.show()