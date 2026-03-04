import sys
sys.path.insert(0, './source')
sys.path.insert(0, './nodes')

import numpy as np
from mpi4py import MPI
from SquareDomain import PoissonSquareOne
from Patch import Patch
from Setup import Setup
from RAHelpers import StableFlatMatrices

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    nodes, normals, min_boundary_idx = PoissonSquareOne(0.025)
else:
    nodes = None
    normals = None
    min_boundary_idx = None

nodes = comm.bcast(nodes, root=0)
normals = comm.bcast(normals, root=0)
min_boundary_idx = comm.bcast(min_boundary_idx, root=0)

patches, patches_for_rank = Setup(comm, nodes, 50)



