from BaseHelpers import *
from Patch import Patch
from scipy.spatial import cKDTree

####################################
#SetupPatches: called on rank zero to generate the patches and distribute to other ranks
####################################

def SetupPatches(nodes, nodes_per_patch, overlap = 3):

    centers = GenCenters(nodes, nodes_per_patch, overlap)
    patch_node_inds, radii = GenPatches(nodes, centers, nodes_per_patch)

    return centers, radii, patch_node_inds


###################################
# Helper function to generate the patches and their associated data
###################################

def GenCenters(nodes, nodes_per_patch, overlap):
    P = int(overlap * nodes.shape[0] // nodes_per_patch)
    d = nodes.shape[1]
    centers = np.empty((P, d))

    ran_idx = np.random.choice(len(nodes))
    centers[0] = nodes[ran_idx]

    minmaxdist = np.linalg.norm(nodes - centers[0], axis=1)

    for i in range(1, P):
        idx = np.argmax(minmaxdist)
        centers[i] = nodes[idx]
        dist = np.linalg.norm(nodes - centers[i], axis=1)
        np.minimum(minmaxdist, dist, out=minmaxdist)

    return centers

def GenPatches(nodes, centers, nodes_per_patch, radius_scale=1.5):
    tree = cKDTree(nodes)
    patches = np.zeros((centers.shape[0], nodes_per_patch), dtype=int)
    radii = np.zeros(centers.shape[0])


    for i, center in enumerate(centers):
        # Find the nearest nodes to the center
        distances, indices = tree.query(center, k=nodes_per_patch)
        patches[i] = indices
        radii[i] = distances[-1] * radius_scale  # Scale up so all nodes are well inside the support
    return patches, radii