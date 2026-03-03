import sys
sys.path.insert(0, './source')
sys.path.insert(0, './nodes')

import numpy as np
from mpi4py import MPI
from SquareDomain import PoissonSquareOne
from Patch import Patch
from Setup import SetupPatches
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

print(f"Rank {rank} has {len(nodes)} nodes.")

if rank == 0:
    centers, radii, patch_node_inds = SetupPatches(nodes, 50, overlap=3)
else:
    centers = None
    radii = None
    patch_node_inds = None

centers = comm.bcast(centers, root=0)
radii = comm.bcast(radii, root=0)
patch_node_inds = comm.bcast(patch_node_inds, root=0)

num_patches = len(centers)
patches_for_rank = [i for i in range(num_patches) if i % comm.Get_size() == rank]

print(f"Rank {rank} assigned {len(patches_for_rank)} patches: {patches_for_rank}")

patches = []
for i in patches_for_rank:
    patch_nodes = nodes[patch_node_inds[i]]
    patch_center = centers[i]
    patch_radius = radii[i]
    patch_nodes_indices = patch_node_inds[i]
    Patch_Phi, Patch_D, Patch_L = StableFlatMatrices(patch_nodes)
    patch = Patch(center=patch_center, radius=patch_radius, node_indices=patch_nodes_indices, 
                    nodes=patch_nodes, Phi=Patch_Phi, D=Patch_D, L=Patch_L)
    patches.append(patch)

print(f"Rank {rank} finished setting up its patches.")