import numpy as np
from mpi4py import MPI
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
solution, num_iters = gmres(comm, Lap, rhs, tol=1e-6, restart=300, maxiter=10)

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
    eigenvalues = np.linalg.eigvals(A_dense)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot eigenvalues in the complex plane
    axes[0].scatter(eigenvalues.real, eigenvalues.imag, s=5, alpha=0.7)
    axes[0].set_xlabel("Real part")
    axes[0].set_ylabel("Imaginary part")
    axes[0].set_title("Spectrum of the Laplacian operator")
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].axvline(0, color='k', linewidth=0.5)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # Plot sorted real parts (useful for understanding conditioning)
    sorted_real = np.sort(eigenvalues.real)
    axes[1].plot(sorted_real, 'o-', markersize=2)
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Eigenvalue (real part)")
    axes[1].set_title("Sorted real parts of eigenvalues")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("laplacian_spectrum.png", dpi=150)
    plt.show()

    # Print conditioning info
    eig_abs = np.abs(eigenvalues)
    eig_abs_nonzero = eig_abs[eig_abs > 1e-12]
    cond_estimate = np.max(eig_abs_nonzero) / np.min(eig_abs_nonzero)
    print(f"Eigenvalue range: [{np.min(eigenvalues.real):.4e}, {np.max(eigenvalues.real):.4e}]")
    print(f"Spectral condition number estimate: {cond_estimate:.4e}")
    print(f"Max imaginary component: {np.max(np.abs(eigenvalues.imag)):.4e}")