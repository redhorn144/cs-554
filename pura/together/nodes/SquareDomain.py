import numpy as np
from rbf.pde.nodes import poisson_disc_nodes, min_energy_nodes

####################################
# A simple square domain with Poisson disc nodes.
# Only admits 1 boundary group
####################################
def PoissonSquareOne(r):
    vert = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    nodes, groups, normals = poisson_disc_nodes(r, (vert, edges))
    interior_nodes = nodes[groups['interior']]
    boundary_nodes = nodes[groups['boundary:all']]

    nodes = np.vstack((interior_nodes, boundary_nodes))
    normals = np.vstack((normals[groups['interior']], normals[groups['boundary:all']]))
    min_boundary_idx = len(interior_nodes)

    return nodes, normals, min_boundary_idx