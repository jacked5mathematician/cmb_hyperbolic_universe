import matplotlib.pyplot as plt
from scipy.linalg import svd
from joblib import Parallel, delayed
import numpy as np

# Function to solve the system using SVD and compute chi^2
def solve_system_via_svd_numeric(A):
    U, s, Vt = svd(A)
    a = Vt[-1]  # The singular vector corresponding to the smallest singular value
    
    # Calculate chi^2 = |A * a|^2
    chi_squared = np.linalg.norm(A @ a)**2
    return chi_squared, a

# Function to plot the chi^2 spectrum
def plot_chi_squared_spectrum(k_values, chi_squared_values, L, num_points_used):
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, chi_squared_values, label=r'$\chi^2(k)$ Spectrum', color='blue')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\chi^2$')
    plt.title(r'$\chi^2$ Spectrum for Varying $k$, L = {}$, Points = {}'.format(L, num_points_used))
    plt.grid(True)
    plt.legend()
    plt.show()