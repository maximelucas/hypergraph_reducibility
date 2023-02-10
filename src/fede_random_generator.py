import random
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def random_nested_generator(N, d_max, p_max, p_lower, p_swap) :
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
    p_lower : float
        Probability to create a (d-1)-hyperedge
        for any face of an existing d-hyperedge
    p_swap : float 
        Probability of each node in a hyperedge
        to be replaced by a node not in that hyperedge
        If p_swap=0, then the hypergraph is fully nested
        by construction: each hyperedge is face of a hyperedge
        of the order above. If p_swap=1, then the hypergraph
        should be fully random. 
    """
    
    if p_max < 0 or p_max > 1 : 
        raise ValueError("p_max should be between 0 and 1")
    if p_lower < 0 or p_lower > 1 : 
        raise ValueError("p_max should be between 0 and 1")
    if p_swap < 0 or p_swap > 1 : 
        raise ValueError("p_max should be between 0 and 1")
    
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

                if random.random() <= p_lower : 
                    # replace each node in hyperedge
                    # by a node outside of it with prob p_swap
                    nodes_outside = list(set(nodes).difference(hyperedge))
                    hyperedge_swapped = []
                    for node in hyperedge : 
                        if random.random() <= p_swap : # replace
                            node_new = random.choice(nodes_outside)
                        else : 
                            node_new = node
                        hyperedge_swapped.append(node_new)

                    hyperedges_d.append(tuple(hyperedge_swapped))
                else : 
                    continue # do not add that face of order d-1

        hyperedges.append(hyperedges_d)
        
        hyperedges_flat = [he for sublist in hyperedges for he in sublist]
    
    return sort_hyperedges(hyperedges_flat)

def sort_hyperedges(hyperedges, directed=False) : 
    """Returns list of hyperedges sorted by length and then alphabetically. 
    If not directed, pre-sort nodes in each hyperedge alphabetically
    """
    
    if not directed : 
        hyperedges = [tuple(sorted(he)) for he in hyperedges]
    
    return sorted(hyperedges, key=lambda x: (len(x), x))    