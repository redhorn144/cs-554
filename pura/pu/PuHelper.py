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
    node_to_patch = {}

    for i, center in enumerate(centers):
        # Find the nearest nodes to the center
        distances, indices = tree.query(center, k=nodes_per_patch)
        patches.append(indices)
        radii.append(distances[-1])  # The radius is the distance to the farthest node in the patch
        for ind in indices:
            if ind not in node_to_patch:
                node_to_patch[ind] = []
            node_to_patch[ind].append(i)
        
    return np.array(patches), np.array(radii), node_to_patch


def GenLocalPhi(nodes, patches, eps):
    matrices = []
    for i, patch in enumerate(patches):
        Phi = GenPhi(nodes[patch], eps)
        matrices.append(Phi)
    return matrices

def GenLocalPhixk(nodes, patches, eps, k):
    matrices = []
    for i, patch in enumerate(patches):
        Phixk = GenPhixk(nodes[patch], eps, k)
        
        matrices.append(Phixk)
    return matrices

def GenLocalLaplacian(nodes, patches, phis, eps):
    matrices = []   
    for i, patch in enumerate(patches):
        Lap = GenLaplacian(nodes[patch], eps)
        Lap = np.linalg.solve(phis[i], Lap.T).T
        matrices.append(Lap)
    return matrices

def GenLocalWeights(patches, matrices, f):
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

def PatchesCovering(x, centers, radii):
    tree = cKDTree(centers)
    nearby_centers = tree.query_ball_point(x, max(radii))
    covering = [j for j in nearby_centers if np.linalg.norm(x - centers[j]) <= radii[j]]
    return covering

def C2Weights(x, covering_patches, centers, radii):
    weights = np.zeros(len(covering_patches))
    for i, patch in enumerate(covering_patches):
        r = np.linalg.norm(x - centers[patch])/radii[patch]
        weights[i] = (1 - r)**4 * (4*r + 1)
    return weights

def Interpolate(x, nodes, patches, centers, radii, local_weights, epsilon):
    covering_patches = PatchesCovering(x, centers, radii)
    
    weights = C2Weights(x, covering_patches, centers, radii)

    local_values = []
    for i, patch in enumerate(covering_patches):
        local_nodes = nodes[patches[patch]]
        phi = np.array([np.exp(-(epsilon * np.linalg.norm(node - x)) ** 2) for node in local_nodes])
        w = local_weights[patch]
        local_values.append(np.dot(w, phi))
    local_values = np.array(local_values)
    return np.dot(weights, local_values) / np.sum(weights)

def ApplyL(nodes):
    for i in len()