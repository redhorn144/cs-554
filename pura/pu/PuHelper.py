#Helper functions for an RBF partition of unity method.
import random
import numpy as np
from scipy.spatial import cKDTree

def GenCenters(nodes, nodes_in_patch, overlap):
    P = overlap*nodes.shape[0]//nodes_in_patch

    centers = []
    
    ran_idx = np.random.choice(len(nodes))
    node = nodes[ran_idx]
    centers.append(node)

    minmaxdist = np.linalg.norm(nodes - node, axis=1)

    for i in range(1, P):
        node = nodes[np.argmax(minmaxdist)]
        centers.append(node)
        dist = np.linalg.norm(nodes - node, axis=1)
        minmaxdist = np.minimum(minmaxdist, dist)

    return centers

def GenPatches(nodes, centers, nodes_per_patch):
    tree = cKDTree(nodes)
    patches = []
    radii = []
    for center in centers:
        # Find the nearest nodes to the center
        distances, indices = tree.query(center, k=nodes_per_patch)
        patches.append(indices)
        radii.append(distances[-1])  # The radius is the distance to the farthest node in the patch
    
    return np.array(patches), np.array(radii)

def GenInterpolationMatrices(nodes, centers, patches, eps):
    matrices = []
    for i, patch in enumerate(patches):
        Phi = GenPhi(nodes[patch], eps)
        matrices.append(Phi)
    return matrices

def GenWeights(patches, matrices, f):
    weights = []
    for i, patch in enumerate(patches):
        A = matrices[i]
        #print(f"system size: {A.shape} for patch {i} with {len(f[patch])} nodes")
        w = np.linalg.solve(A, f[patch])
        weights.append(w)
    return weights

def GenPhi(x, e):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    return np.exp(-(e * r) ** 2)

