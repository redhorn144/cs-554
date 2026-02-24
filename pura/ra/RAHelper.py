import numpy as np
import scipy.linalg as spla
from scipy import optimize


#########################################
# Interpolation matrices for the kernel
# and its derivatives.
#########################################
def GenPhi(x, e):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    return np.exp(-(e * r) ** 2)

def GenPhixk(x, e, k):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    diff_k = diff[:, :, k]
    return -2 * e**2 * diff_k * np.exp(-(e * r) ** 2)

def GenLaplacian(x, e):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    d = x.shape[1]
    return 2 * e**2 * np.exp(-(e * r) ** 2) * (2 * e**2 * r**2 - d)

def GenEvalPhi(eval_points, x, e):
    diff = eval_points[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    return np.exp(-(e * r) ** 2)

##########################################
# Functions for the RBF-RA method
###########################################

def SolveSystems(phis, f):
    lams = [np.linalg.solve(phis[k], f) for k in range(len(phis))]
    return lams

def InterpolatedSamples(x, eval_points, f, es):
    lams = SolveSystems([GenPhi(x, e) for e in es], f)
    eval_phis = [GenEvalPhi(eval_points, x, es[k]) for k in range(len(es))]
    samples = np.array([eval_phis[k] @ lams[k] for k in range(len(es))])
    return samples

def GenE(es, m):
    K = len(es)
    E = np.zeros((K, m + 1))
    E[:, 0] = 1
    for k in range(K):
        row = np.array([es[k] ** (2*i) for i in range(1, m+1)])
        E[k, 1:] = row
    return E

def GenFj(fj, es, n):
    K = len(es)
    Fj = np.zeros((K, n))
    for k in range(K):
        row = np.array([es[k] ** (2*i) for i in range(1, n+1)])
        Fj[k, :] = row
    return -np.diag(fj)@Fj

def buildRA(fj_mates, es, n, m):
    E = GenE(es, m)
    

def GenEr(x):
    minimizer = optimize.fminbound(lambda e: ConditionObjective(e, x), 0.1, 20)
    return minimizer

def ConditionObjective(e, x):
    re_phi = GenPhi(x, e)
    im_phi = GenPhi(x, e * 1j)
    return np.linalg.norm(im_phi, ord=np.inf)/np.linalg.norm(np.linalg.inv(re_phi), ord=np.inf)

def GenEs(K):
    k = 2*np.arange(1, K/2 + 1)
    thetas = (np.pi/2)*(k/K)
    es = np.exp(1j*thetas)
    return es

