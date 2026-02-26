#Helper functions for an RBF partition of unity method.
import random
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.linalg import LinearOperator, gmres

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
        radii.append(distances[-1] * 1.5)  # Scale up so all nodes are well inside the support
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
    return np.array(matrices)

def GenLocalDxk(nodes, patches, phis, eps, k):
    matrices = []
    for i, patch in enumerate(patches):
        Phixk = GenPhixk(nodes[patch], eps, k)
        Dxk = np.linalg.solve(phis[i], Phixk.T).T
        matrices.append(Dxk)
    return np.array(matrices)

def GenLocalGrads(nodes, patches, phis, eps):
    grad_matrices = []
    for k in range(nodes.shape[1]):
        grad_k = GenLocalDxk(nodes, patches, phis, eps, k)
        grad_matrices.append(grad_k)
    return np.array(grad_matrices).transpose(1, 0, 2, 3)

def GenLocalLaps(nodes, patches, phis, eps):
    matrices = []   
    for i, patch in enumerate(patches):
        Lap = GenLaplacian(nodes[patch], eps)
        Lap = np.linalg.solve(phis[i], Lap.T).T
        matrices.append(Lap)
    return np.array(matrices)

def GenLocalWeights(patches, matrices, f):
    weights = []
    for i, patch in enumerate(patches):
        A = matrices[i]
        #print(f"system size: {A.shape} for patch {i} with {len(f[patch])} nodes")
        w = np.linalg.solve(A, f[patch])
        weights.append(w)
    return np.array(weights)

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

def ApplyLap(u, nodes, patches, centers, radii, grads, laps):
    N = len(nodes)
    d = nodes.shape[1]
    Lapu = np.zeros(N)

    # First pass: compute Shepard sum W, gradW, lapW at every node
    W = np.zeros(N)
    gradW = np.zeros((N, d))
    lapW = np.zeros(N)

    for patch_idx, patch in enumerate(patches):
        for k, node_idx in enumerate(patch):
            w = C2Weight(nodes[node_idx], centers[patch_idx], radii[patch_idx])
            gw = C2WeightDerivatives(nodes[node_idx], centers[patch_idx], radii[patch_idx])
            lw = C2WeightLaplacian(nodes[node_idx], centers[patch_idx], radii[patch_idx])
            W[node_idx] += w
            gradW[node_idx] += gw
            lapW[node_idx] += lw

    # Second pass: apply Laplacian with normalized PU weights
    for patch_idx, patch in enumerate(patches):
        lap_local = laps[patch_idx]
        grad_local = grads[patch_idx]
        u_local = u[patch]
        Lu_local = lap_local @ u_local
        gradu_local = np.array([grad_local[i] @ u_local for i in range(len(grad_local))])

        for k, node_idx in enumerate(patch):
            w = C2Weight(nodes[node_idx], centers[patch_idx], radii[patch_idx])
            gw = C2WeightDerivatives(nodes[node_idx], centers[patch_idx], radii[patch_idx])
            lw = C2WeightLaplacian(nodes[node_idx], centers[patch_idx], radii[patch_idx])

            Wn = W[node_idx]
            gWn = gradW[node_idx]
            lWn = lapW[node_idx]

            # Normalized weight and its derivatives via quotient rule
            w_bar = w / Wn
            gw_bar = gw / Wn - w * gWn / Wn**2
            lw_bar = (lw / Wn
                      - 2 * np.dot(gw, gWn) / Wn**2
                      - w * lWn / Wn**2
                      + 2 * w * np.dot(gWn, gWn) / Wn**3)

            Lapu[node_idx] += lw_bar * u_local[k]
            Lapu[node_idx] += 2 * np.dot(gw_bar, gradu_local[:, k])
            Lapu[node_idx] += w_bar * Lu_local[k]

    return Lapu

def SolvePoissonGMRES(
    rhs,
    nodes,
    patches,
    centers,
    radii,
    grads,
    laps,
    boundary_idx=None,
    boundary_values=0.0,
    x0=None,
    use_negative_laplacian=True,
    rtol=1e-8,
    atol=0.0,
    restart=None,
    maxiter=None,
):
    rhs = np.asarray(rhs, dtype=float)
    n = len(nodes)

    if rhs.shape != (n,):
        raise ValueError(f"rhs must have shape ({n},), got {rhs.shape}")

    if boundary_idx is None:
        boundary_idx = np.array([], dtype=int)
    else:
        boundary_idx = np.asarray(boundary_idx, dtype=int)

    if np.isscalar(boundary_values):
        boundary_values_vec = np.full(boundary_idx.shape, float(boundary_values))
    else:
        boundary_values_vec = np.asarray(boundary_values, dtype=float)
        if boundary_values_vec.shape != boundary_idx.shape:
            raise ValueError(
                f"boundary_values must be scalar or shape {boundary_idx.shape}, got {boundary_values_vec.shape}"
            )

    b = rhs.copy()
    if boundary_idx.size > 0:
        b[boundary_idx] = boundary_values_vec

    sign = -1.0 if use_negative_laplacian else 1.0

    def matvec(u):
        Au = sign * ApplyLap(u, nodes, patches, centers, radii, grads, laps)
        if boundary_idx.size > 0:
            Au[boundary_idx] = u[boundary_idx]
        return Au

    A = LinearOperator((n, n), matvec=matvec, dtype=float)
    residual_history = []

    def _callback(residual):
        if np.isscalar(residual):
            residual_history.append(float(residual))
        else:
            residual_history.append(float(np.linalg.norm(residual)))

    try:
        u, info = gmres(
            A,
            b,
            x0=x0,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            callback=_callback,
            callback_type="pr_norm",
        )
    except TypeError:
        u, info = gmres(
            A,
            b,
            x0=x0,
            tol=rtol,
            restart=restart,
            maxiter=maxiter,
            callback=_callback,
        )

    return u, info, np.array(residual_history)

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

