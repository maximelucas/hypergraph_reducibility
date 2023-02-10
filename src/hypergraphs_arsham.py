"""
Useful functions to generate hypergraphs 
and compute the multiorder Laplacian
"""

import random
from itertools import combinations, permutations
from math import factorial
from scipy.special import comb 

import networkx as nx
import numpy as np

__all__ = [
    "sort_hyperedges",
    "hyperedges_of_order",
    "to_simplicial_complex_from_hypergraph",
    "random_hypergraph",
    "fully_connected_hypergraph",
    "random_nested_generator",
    "adj_tensor_of_order",
    "adj_matrix_of_order",
    "degree_of_order",
    "laplacian_of_order",
]


# =======
# UTILS
# =======


def sort_hyperedges(hyperedges, directed=False):
    """Returns list of hyperedges sorted by length and then alphabetically.
    If not directed, pre-sort nodes in each hyperedge alphabetically
    """

    if not directed:
        hyperedges = [tuple(sorted(he)) for he in hyperedges]

    return sorted(hyperedges, key=lambda x: (len(x), x))


def hyperedges_of_order(hyperedges, d):
    """Returns list of all d-hyperedges"""

    return [hyperedge for hyperedge in hyperedges if len(hyperedge) == d + 1]


def to_simplicial_complex_from_hypergraph(hyperedges, verbose=False):
    """Converts a hypergraph to a simplicial complex
    by adding all missing subfaces.

    Parameters
    ----------
    hyperedges : list of tuples
        List of hyperedges in the hypergraph to fill
    verbose : bool
        If True, print all added hyperedges

    Returns
    -------
    hyperedges_simplicial : list of tuples

    """

    hyperedges_simplicial = _add_all_subfaces(hyperedges, verbose=verbose)

    return hyperedges_simplicial


def _add_all_subfaces(hyperedges, verbose=False):
    """Adds all missing subfaces to hypergraph

    Goes through all hyperedges, from larger to smaller,
    and adds their subfaces if they do not exist.

    Parameters
    ----------
    hyperedges : list of tuples
        List of hyperedges in the hypergraph to fill
    verbose : bool
        If True, print all added hyperedges

    Returns
    -------
    hyperedges : list of tuples

    """

    hyperedges_to_add = []

    # check that all subfaces of each hyperedge exist
    for hedge in hyperedges[::-1]:  # loop over hyperedges, from larger to smaller

        d = len(hedge)  # number of node, i.e. order-1
        if d >= 3:  # nodes already checked
            for face in combinations(hedge, d - 1):  # check if all subfaces are present

                face = tuple(sorted(face))
                if face not in hyperedges:
                    hyperedges_to_add.append(face)

    if verbose:
        print(f"Info: the following hyperedges were added")
        print(hyperedges_to_add)

    hyperedges += hyperedges_to_add
    hyperedges = sort_hyperedges(hyperedges)

    return hyperedges


# ============
# GENERATORS
# ============


def random_hypergraph(N, ps):
    """Generates a random hypergraph

    Generate N nodes, and connect any d+1 nodes
    by a hyperedge with probability ps[d].

    Parameters
    ----------
    N : int
        Number of nodes
    ps : list of floats
        Probabilities (between 0 and 1) to create a hyperedge
        at each order d between any d+1 nodes. ps[0] is edges,
        ps[1] for triangles, etc.

    Returns
    -------
    List of tuples
        List of hypergraphs
    """

    # I first generate a standard ER graph with edges connected with probability p1
    G = nx.fast_gnp_random_graph(N, ps[0], seed=None)

    nodes = G.nodes()
    hyperedges = list(G.edges())

    for i, p in enumerate(ps[1:]):
        d = i + 2  # order (+2 because we started with [1:])
        for hyperedge in combinations(nodes, d + 1):
            if random.random() <= p:
                hyperedges.append(hyperedge)

    return sort_hyperedges(hyperedges)


def fully_connected_hypergraph(N, d_max):
    """Generates a fully connected hypergraphs

    Generate N nodes and connect any d+1 nodes
    by a hyperedge, up to order d_max.

    Parameters
    ----------
    N : int
        Number of nodes
    d_max : int
        Highest order of interactions. For example,
        d_max=2 means we go up to triangles.
    """

    nodes = range(N)

    hyperedges = []

    for d in range(1, d_max + 1):
        for hyperedge in combinations(nodes, d + 1):
            hyperedges.append(hyperedge)

    return sort_hyperedges(hyperedges)

def random_nested_generator(N, d_max, p_max, n_lower, p_swap) :
    """Generates a random hypergraph with maximum order d_max.
    
    The hypergraph is generated as follows:
        1. Create a d_max-hyperedge with probabilty p_max
           for any d_max+1 nodes.
        2. Go down one order to d=d_max-1
            a. For each d+1 hyperedge, create a d-hyperedge 
            for any of its d+1 nodes with probability p_lower
            b. For each of those d-hyperedges created, 
            replace each node by an outside node with 
            probability p_swap.
            c. Start again at 2. until reaching d=1.
    p_swap controls how nested or random the hypergraph is.
    
    Parameters
    ----------
    N : int 
        Number of nodes
    d_max : int 
        Maximum order of hyperedge in the hypergraph
    p_max : float
        Probability to create a d_max-hyperedge
        from any d_max+1 nodes
    n_lower : float
        Average number of face [(d-1)-hyperedge]
        per existing d-hyperedge. For any of the d subfaces, 
        the probability to create a hyperedge
        is p_lower = n_lower / d
    p_swap : float 
        Probability of each node in a hyperedge
        to be replaced by a node not in that hyperedge
        If p_swap=0, then the hypergraph is fully nested
        by construction: each hyperedge is the face of a hyperedge
        of the order above. If p_swap=1, then the hypergraph
        should be fully random. 
        
    Returns
    -------
    hyperedges : list of list of tuples
        Each element is a list hyperedges at a given order 
    hyperedges_flat : list of tuples
        List of hyperedges sorted by size (flattened version of 'hyperedges')
    """
    
    if p_max < 0 or p_max > 1 : 
        raise ValueError("p_max should be between 0 and 1")
    if n_lower < 0 or n_lower > d_max : 
        raise ValueError("n_lower between 0 and d_max so that p_lower is between 0 and 1.")
    if p_swap < 0 or p_swap > 1 : 
        raise ValueError("p_swap should be between 0 and 1")
    
    nodes = range(N) 

    hyperedges = []
    hyperedges_d = []

    # add hyperedges of order d_max with prob p_max
    for hyperedge in combinations(nodes, d_max+1) : 
        if random.random() <= p_max : 
            hyperedges_d.append(hyperedge)

    hyperedges.append(hyperedges_d)

    # now go down in the orders
    ds = range(d_max-1, 0, -1)
    for i,d in enumerate(ds) :

        hyperedges_up = hyperedges[i]
        hyperedges_d = []

        for hyperedge_up in hyperedges_up : 

            # add each face of order d-1 with prob p_lower
            for hyperedge in combinations(hyperedge_up, d+1) : 
                
                p_lower = n_lower / (d+2)
                if p_lower > 1:
                    raise ValueError("p_lower cannot be >1")
                    
                if random.random() <= p_lower : # add d-1 face
                    # replace each node in hyperedge
                    # by a node outside of it with prob p_swap
                    valid_add = False 
                    
                    if len(hyperedges_d)==comb(N,d+1) : 
                        raise ValueError(f"All {d}-hyperedges already exist. Not possible to add more.")
                
                    while not valid_add : 
                        hyperedge_swapped = []
                        nodes_outside = list(set(nodes).difference(hyperedge))
                        # modify hyperedge by replacing nodes
                        for node in hyperedge : 
                            if random.random() <= p_swap : # replace
                                node_new = random.choice(nodes_outside)
                                nodes_outside.remove(node_new) # avoid twice same node
                            else : 
                                node_new = node
                            hyperedge_swapped.append(node_new)
                        hyperedge_swapped = tuple(sorted(hyperedge_swapped))
                        # check that modified hyperedge is valid, i.e.
                        # it does not exist yet. If it does, start over.
                        if hyperedge_swapped not in hyperedges_d : 
                            valid_add = True
                            hyperedges_d.append(tuple(hyperedge_swapped))
                        else :
                            pass
                            if p_swap==0 : # no chance to change, infinite loop
                                print(f'Hyperedge {hyperedge_swapped} already exists',
                                      f'not added because p_swap=0')
                                break
                    
                else : # do not add face
                    continue 

        hyperedges.append(hyperedges_d)
        
        hyperedges_flat = [he for sublist in hyperedges for he in sublist]
        hyperedges_flat = sort_hyperedges(hyperedges_flat)
        
        # make sure there are no duplicates 
        uni, counts = np.unique(hyperedges_flat, return_counts=True)
        assert len(uni[counts>1])==0
    
    return hyperedges_flat, hyperedges    


# ======================
# Multiorder Laplacian
# ======================


def adj_tensor_of_order(d, N, hyperedges):
    """Returns the adjacency tensor of order d

    Parameters
    ----------
    d : int
        Order of the adjacency matrix
    d_simplices : list of tuples
        Sorted list of hyperedges of order d

    Returns
    -------
    M : numpy array
        Adjacency tensor of order d

    """

    d_hyperedges = hyperedges_of_order(hyperedges, d)

    assert len(d_hyperedges[0]) == d + 1

    dims = (N,) * (d + 1)
    M = np.zeros(dims)

    for d_hyperedge in d_hyperedges:
        for d_hyperedge_perm in permutations(d_hyperedge):

            M[d_hyperedge_perm] = 1
    return M


def adj_matrix_of_order(d, M):
    """Returns the adjacency matrix of order d

    Parameters
    ----------
    d : int
        Order of the adjacency matrix
    M : numpy array
        Adjacency tensor of order d

    Returns
    -------
    adj_d : numpy array
        Matrix of dim (N, N)

    """

    adj_d = (
        1 / factorial(d - 1) * np.sum(M, axis=tuple(range(d + 1)[2:]))
    )  # sum over all axes except first 2 (i,j)

    return adj_d


def degree_of_order(d, M):
    """Returns the degree vector of order d

    Parameters
    ----------
    d : int
        Order of the degree
    M : numpy array
        Adjacency tensor of order d

    Returns
    -------
    K_d : numpy array
        Vector of dim (N,)

    """

    K_d = (
        1 / factorial(d) * np.sum(M, axis=tuple(range(d + 1)[1:]))
    )  # sum over all axes except first 2 (i,j)

    return K_d


def laplacian_of_order(d, N, hyperedges, return_k=False, rescale_per_node=False) :
    """Returns the Laplacian matrix of order d
    
    Parameters
    ----------
    d : int 
        Order of the adjacency matrix 
    d_simplices : list of tuples 
        Sorted list of hyperedges of order d
    rescale_per_node : bool, optional
        If True, divide the Laplacian by d, i.e.
        by the number of neighbour-nodes in a d-simplex

    Returns
    -------
    L_d : numpy array
        Matrix of dim (N, N)        
    
    """
    
    d_hyperedges = hyperedges_of_order(hyperedges, d)
    
    M_d = adj_tensor_of_order(d, N, d_hyperedges)
    
    Adj_d = adj_matrix_of_order(d, M_d)
    K_d = degree_of_order(d, M_d) 
    
    L_d = d * np.diag(K_d) - Adj_d
    
    if rescale_per_node : 
        L_d /= d
    
    if return_k :
        return L_d, K_d
    else:
        return L_d
