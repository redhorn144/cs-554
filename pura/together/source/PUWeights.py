import numpy as np


######################################
#
######################################


######################################
# Helper function for the Wenland C2 
# weight function and its derivatives.
######################################

def C2Weight(x, center, radius):
    r = np.linalg.norm(x - center)/radius
    return (1 - r)**4 * (4*r + 1)

def C2WeightDerivatives(x, center, radius):
    gradw = np.zeros_like(x)
    r = np.linalg.norm(x - center)
    if r == 0:
        return gradw
    rho = r/radius
    factor = -20 * (1 - rho)**3 * rho / (radius * r)
    for i in range(len(x)):
        gradw[i] = factor * (x[i] - center[i])
    return gradw

def C2WeightLaplacian(x, center, radius):
    d = len(x)
    r = np.linalg.norm(x - center)
    if r == 0:
        return -20 * d / radius**2
    rho = r/radius
    psi_d = -20 * (1 - rho)**3 * rho 
    psi_dd = 20*(1 - rho)**2 *(4*rho - 1)
    return (psi_dd + (d - 1) * psi_d / rho) / radius**2