import numpy as np
import scipy.linalg as spla

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

def SolveSystems(phis, f):
    lams = [spla.solve(phis, f[:, i]) for i in range(f.shape[1])]
    return lams

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
    return np.diag(fj)@Fj

