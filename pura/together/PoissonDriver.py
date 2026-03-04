import sys


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
solution = gmres(comm, Lap, rhs, tol=1e-6, maxiter=1000)

if rank == 0:
    print("GMRES solve complete. ")
    PlotSolution(nodes, solution)

