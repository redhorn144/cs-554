import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from nodes.SquareDomain import PoissonSquareOne
from source.Patch import Patch
from source.Setup import Setup
from source.Operators import ApplyLap
from source.Solver import gmres
from source.Plotter import PlotSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.025)
else:
    nodes = None
    normals = None
    groups = None

nodes = comm.bcast(nodes, root=0)
normals = comm.bcast(normals, root=0)
groups = comm.bcast(groups, root=0)

patches, patches_for_rank = Setup(comm, nodes, normals, 50)
print(f"Rank {rank} setup complete with {len(patches)} patches.")
BCs = np.array(["dirichlet"])
bc_groups = np.array([groups['boundary:all']])
Lap = ApplyLap(comm, patches, nodes.shape[0], bc_groups, BCs)

rhs = -2*np.pi**2*np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
rhs[bc_groups[0]] = 0.0

print(f"Rank {rank} starting GMRES solve...")
if rank == 0:
    t_start = MPI.Wtime()
solution, num_iters = gmres(comm, Lap, rhs, tol=1e-4, restart=100, maxiter=10)

if rank == 0:
    t_end = MPI.Wtime()
    print(f"GMRES solve complete in {t_end - t_start:.2f} seconds.")
    print(f"GMRES converged in {num_iters} iterations.")
    print("GMRES solve complete. ")
    u_exact = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    error = np.linalg.norm(solution - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error: {error:.2e}")
    PlotSolution(nodes, solution)

# --- Spectrum computation ---
N = nodes.shape[0]
if rank == 0:
    print(f"Building dense operator matrix ({N}x{N}) for spectrum analysis...")

# Build the full matrix by applying Lap to each basis vector
A_dense = np.zeros((N, N))
for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    A_dense[:, j] = Lap(e_j)

if rank == 0:
    print("Computing eigenvalues...")
<<<<<<< HEAD
    eigenvalues, eigenvectors = np.linalg.eig(A_dense)  
    problem_eig = -2*np.pi**2
    closest_idx = np.argmin(np.abs(eigenvalues - problem_eig))
    closest_eig = eigenvalues[closest_idx]
    print(f"Problem eigenvalue: {problem_eig:.4e}")
    print(f"Closest computed eigenvalue: {closest_eig:.4e}")
    print(f"Distance: {np.abs(closest_eig - problem_eig):.4e}")
=======
    # The full matrix has identity rows at BC nodes, breaking symmetry.
    # Restrict to interior nodes so the submatrix is the true symmetric Laplacian.
    bc_nodes = bc_groups[0]
    interior = np.setdiff1d(np.arange(N), bc_nodes)
    A_interior = A_dense[np.ix_(interior, interior)]

    # The PU collocation matrix is non-symmetric (Kansa method).
    # Symmetrizing (A + A.T)/2 destroys the spectral structure — use eig directly.
    raw_vals, raw_vecs = np.linalg.eig(A_interior)

    # Eigenvalues are generally complex with small imaginary parts; keep real part.
    # Determine sign convention: if median real part is negative, operator ≈ +Delta,
    # so we negate to obtain eigenvalues of -Delta (which should be positive).
    real_parts = raw_vals.real
    if np.median(real_parts) < 0:
        real_parts = -real_parts
        print("Operator ≈ +Delta: negating eigenvalues to get -Delta")
    else:
        print("Operator ≈ -Delta: eigenvalues used as-is")

    # Sort by ascending real part (lowest frequency first)
    order = np.argsort(real_parts)
    eigenvalues    = real_parts[order]
    evecs_interior = raw_vecs[:, order].real

    print(f"First 4 eigenvalues of -Delta: {eigenvalues[:4]}")
    print(f"  (exact: 2pi^2={2*np.pi**2:.4f}, 5pi^2={5*np.pi**2:.4f}, 8pi^2={8*np.pi**2:.4f})")

    # Pad eigenvectors back to full size (zero at boundary = Dirichlet BCs)
    eigenvectors = np.zeros((N, len(eigenvalues)))
    eigenvectors[interior, :] = evecs_interior
>>>>>>> 5e43a85 (stage)

    # Compute and plot the eigenvector associated with the closest eigenvalue
    
    closest_eigvec = np.real(eigenvectors[:, closest_idx])

    from matplotlib.tri import Triangulation
    tri = Triangulation(nodes[:, 0], nodes[:, 1])

    fig_eig = plt.figure(figsize=(10, 8))
    ax = fig_eig.add_subplot(111, projection='3d')
    ax.plot_trisurf(tri, closest_eigvec, cmap='RdBu_r', edgecolor='none', antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Eigenvector value")
    ax.set_title(f"Eigenvector for eigenvalue {closest_eig:.4e}")
    plt.tight_layout()
    plt.savefig("closest_eigenvector.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot eigenvalues (real, sorted ascending from eigh)
    axes[0].scatter(np.arange(len(eigenvalues)), eigenvalues, s=5, alpha=0.7)
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue of $-\\Delta$")
    axes[0].set_title("Spectrum of $-\\Delta$ (interior nodes)")
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    # Plot sorted eigenvalues (log scale for conditioning view)
    axes[1].plot(eigenvalues, 'o-', markersize=2)
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Eigenvalue of $-\\Delta$")
    axes[1].set_title("Sorted eigenvalues of $-\\Delta$")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("laplacian_spectrum.png", dpi=150)
    #plt.show()

    # --- Plot eigenvectors for the first 4 eigenvalues ---
    # eigh already returns eigenvalues sorted ascending, so first 4 = lowest modes
    from scipy.interpolate import griddata as _gd
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig2 = plt.figure(figsize=(14, 11))
    x, y = nodes[:, 0], nodes[:, 1]
    gx, gy = np.meshgrid(
        np.linspace(x.min(), x.max(), 200),
        np.linspace(y.min(), y.max(), 200),
    )
    for k in range(4):
        evec = eigenvectors[:, k]  # already real from eigh
        # Normalise sign: ensure the maximum absolute value is positive
        if np.abs(evec.min()) > np.abs(evec.max()):
            evec = -evec
        lam = eigenvalues[k]
        grid_v = _gd((x, y), evec, (gx, gy), method="cubic")
        ax = fig2.add_subplot(2, 2, k + 1, projection="3d")
        surf = ax.plot_surface(gx, gy, grid_v, cmap="RdBu_r", linewidth=0, antialiased=True)
        fig2.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
        ax.set_title(f"Mode {k+1}: $\\lambda = {lam:.4g}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("$\\phi$")

    fig2.suptitle("Eigenmodes of $-\\Delta$ (first 4, sorted by $\\lambda$ ascending)", fontsize=14)
    fig2.tight_layout()
    fig2.savefig("laplacian_eigenvectors.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved eigenvector plots to laplacian_eigenvectors.png")

    # Print conditioning info
    eig_nonzero = eigenvalues[np.abs(eigenvalues) > 1e-12]
    cond_estimate = np.max(np.abs(eig_nonzero)) / np.min(np.abs(eig_nonzero))
    print(f"Eigenvalue range of -Delta: [{eigenvalues.min():.4e}, {eigenvalues.max():.4e}]")
    print(f"Spectral condition number estimate: {cond_estimate:.4e}")