"""
Script for computing and plotting hypergraph information
"""

import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sb
import scipy.sparse as sp
from numpy.linalg import eigvals, eigvalsh
from scipy.linalg import expm, logm
from scipy.sparse.linalg import eigsh

import xgi

__all__ = [
    "find_charact_tau",
    "construct_hg_multilayer",
    "density",
    "partition",
    "KL",
    "penalization",
    "optimization",
    "optimization_v2",
    "entropy",
    "compute_information",
    "pad_arr_list",
    "plot_information",
    "shuffle_hyperedges",
]


def find_charact_tau(H, orders, weights, rescale_per_node=False, sparse=False):
    """
    Find characteristic timescale tau

    Parameters
    ----------
    H : xgi Hypergraph
        Input hypergraph
    orders : list
        List of integers representing the orders of the laplacian matrices.
    weights : list
        List of float values representing the weights for each order.

    Returns
    -------
    float
        The value of tau calculated from the eigenvalues of the multi-order laplacian matrix.
    """
    L_multi = xgi.multiorder_laplacian(
        H, orders, weights, rescale_per_node=rescale_per_node, sparse=sparse
    )

    if sparse:
        lambdas = eigsh(L_multi, k=1, return_eigenvectors=False)
    else:
        lambdas = eigvalsh(L_multi)

    return 1 / lambdas[-1]


def construct_hg_multilayer(H, rescale_per_node=False):
    """
    Computes the list of Laplacians and the total Laplacian matrix of a hypergraph H.

    Parameters
    ----------
    H : xgi.Hypergraph
        The hypergraph to compute the Laplacian of

    Returns
    -------
    list: a list of Laplacians, ordered by their order
    np.ndarray: the total Laplacian matrix
    """
    max_d = xgi.max_edge_order(H)
    hg_m = []
    for d in range(1, max_d + 1):
        L = xgi.laplacian(H, d, rescale_per_node=rescale_per_node)
        hg_m.append(L)
    N = G.num_nodes
    hg_all = np.zeros((N, N))
    for l in range(len(hg_m)):
        hg_all = hg_all + hg_m[l]
    return hg_m, hg_all


def density(Lap, tau, sparse=False):
    """
    Computes the density matrix for a Laplacian with scale `tau`.

    Parameters
    ----------
    Lap : np.ndarray
        The Laplacian matrix
    tau : float
        The scale of the Laplacian

    Returns
    -------
    np.ndarray: the density matrix
    """
    if sparse:
        rho = sp.linalg.expm(-tau * Lap)
        rho = rho / rho.trace()
        rho = rho + sp.eye(rho.shape[0]) * 10**-10
    else:
        rho = expm(-tau * Lap)
        rho = rho / np.trace(rho) 
        rho = rho + np.eye(len(rho)) * 10**-10
    return rho


def partition(Lap, tau):
    """
    Computes the partition function of a Laplacian with scale `tau`.

    Parameters
    ----------
    Lap : np.ndarray
        The Laplacian matrix
    tau : float
        The scale of the Laplacian

    Returns
    -------
    float: the partition function
    """
    return np.trace(expm(-2 * tau * Lap))


def KL(rho_emp, rho_model, sparse=False):
    """
    Computes the Kullback-Leibler (KL) divergence between an empirical observation `rho_emp` and a model `rho_model`.

    Parameters
    ----------
    rho_emp : (np.ndarray)
        The empirical observation
    rho_model : np.ndarray
        The model

    Returns
    -------
    float: the KL divergence between `rho_emp` and `rho_model`
    """

    if sparse:
        rho_emp = rho_emp.toarray()
        rho_model = rho_model.toarray()
    
    return np.trace(rho_emp @ logm(rho_emp) - rho_emp @ logm(rho_model))


def penalization(Lap, tau, sparse=False):
    """
    Computes the partition function of a Laplacian with scale `tau`.

    Parameters
    ----------
    Lap : np.ndarray
        The Laplacian matrix
    tau : float
        The scale of the Laplacian

    Returns
    -------
    float: the partition function
    """
    N = Lap.shape[0]
    return np.log(N) - entropy(Lap, tau, sparse=sparse)


def optimization(H, tau, rescaling_factors=None, rescale_per_node=False, sparse=False):
    """
    Computes the gain and loss for modeling a hypergraph (up to order `d_max`),
    using a part of it, up to order `d < d_max`.

    Parameters
    ----------
    H: xgi Hypergraph
        The input hypergraph
    tau: float
        The scale of signal propagation
    rescaling_factors : array of float
        Factors to rescale to tau for each order

    Returns
    -------
    D: numpy.ndarray
        The learning error
    lZ: numpy.ndarray
        The penalization term for model complexity
    """
    orders = np.array(xgi.unique_edge_sizes(H)) - 1
    weights = np.ones(len(orders))
    if rescaling_factors is None:
        rescale_factors = np.ones_like(orders)

    L_multi = xgi.multiorder_laplacian(
        H, orders, weights, rescale_per_node=rescale_per_node, sparse=sparse
    )

    rho_all = density(L_multi, tau, sparse=sparse)

    D = []  # Learning error
    lZ = []  # Penalization term for model complexity
    N = H.num_nodes

    for l in range(len(orders)):
        L_l = xgi.multiorder_laplacian(
            H, orders[0 : l + 1], weights[0 : l + 1], rescale_per_node=rescale_per_node, sparse=sparse
        )

        rho_l = density(L_l, tau * rescaling_factors[l], sparse=sparse)
        d = KL(rho_all, rho_l, sparse=sparse)
        z = penalization(L_l, tau * rescaling_factors[l], sparse=sparse)

        D.append(d)
        lZ.append(z)

    lZ = np.array(lZ)
    D = np.array(D)

    return D, lZ


def optimization_v2(H, tau, rescale_per_node=False, sparse=False):
    """
    Computes the gain and loss for modeling a hypergraph (up to order `d_max`),
    using a part of it, up to order `d < d_max`.

    Parameters
    ----------
    H: xgi Hypergraph
        The input hypergraph
    tau: float
        The scale of signal propagation

    Returns
    -------
    D: numpy.ndarray
        The learning error
    lZ: numpy.ndarray
        The penalization term for model complexity
    """
    orders = np.array(xgi.unique_edge_sizes(H)) - 1
    weights = np.ones(len(orders))
    L_multi = xgi.multiorder_laplacian(
        H, orders, weights, rescale_per_node=rescale_per_node, sparse=True
    )

    #rho_all = density(L_multi, tau, sparse=sparse
    eigenvals, eigenvectors = np.linalg.eigh(L_multi.toarray())
    # Compute the matrix exponential using the eigendecomposition
    exp_vals_multi = np.exp(- tau * eigenvals)
    exp_Lmulti = np.dot(eigenvectors, np.dot(np.diag(exp_vals_multi), eigenvectors.T))
    rho_all = exp_Lmulti / np.trace(exp_Lmulti) 
    rho_all = rho_all + np.eye(len(rho_all)) * 10**-10

    D = []  # Learning error
    lZ = []  # Penalization term for model complexity
    N = H.num_nodes

    for l in range(len(orders)):
        L_l = xgi.multiorder_laplacian(
            H, orders[0 : l + 1], weights[0 : l + 1], rescale_per_node=rescale_per_node, sparse=True
        )

        #rho_l = density(L_l, tau, sparse=sparse)
        eigenvals, eigenvectors = np.linalg.eigh(L_l.toarray())
        # Compute the matrix exponential using the eigendecomposition
        exp_vals_l = np.exp(- tau * eigenvals)
        exp_L = np.dot(eigenvectors, np.dot(np.diag(exp_vals_l), eigenvectors.T))
        rho_l = exp_L / np.trace(exp_L) 
        rho_l = rho_l + np.eye(len(rho_l)) * 10**-10

        #d = KL(rho_all, rho_l, sparse=sparse)
        d = np.trace(rho_all @ logm(rho_all) - rho_all @ logm(rho_l))

        # entropy
        #expL = np.exp(-tau * lambdas)
        Z = np.sum(exp_vals_l)  # Calculates the partition function
        p = exp_vals_l / Z  # Calculates the probabilities
        p = np.delete(p, np.where(p < 10**-8))
        S = np.sum(-p * np.log(p))  # entropy

        #z = np.log(N) - entropy(L_l, tau, sparse=sparse)
        #z = penalization(L_l, tau, sparse=sparse)
        N = L_l.shape[0]
        z = np.log(N) - S

        D.append(d)
        lZ.append(z)

    lZ = np.array(lZ)
    D = np.array(D)

    return D, lZ


def entropy(L, tau, sparse=False):
    """
    Computes the entropy associated to the Laplacian matrix.

    Parameters
    ----------
    L: numpy.ndarray
        The Laplacian matrix
    tau: float
        The scale of signal propagation

    Returns
    -------
    S: float
        The entropy of the graph
    """

    N = L.shape[0]

    if sparse:
        lambdas, _ = eigsh(L, k=N-1)
    else:
        lambdas = eigvalsh(L)  # Calculate eigenvalues of L

    expL = np.exp(-tau * lambdas)
    Z = np.sum(expL)  # Calculates the partition function
    p = expL / Z  # Calculates the probabilities
    p = np.delete(p, np.where(p < 10**-8))
    S = np.sum(-p * np.log(p))  # entropy
    return S


def compute_information(H, tau, rescale_per_node=False, sparse=False):
    """
    Utility function to compute the information of the hypergraph.

    Parameters
    ----------
    H: nx.Graph
        The input hypergraph
    taus: list of float
        The scale of signal propagation

    Returns
    -------
    Ds: numpy.ndarray
        The learning errors
    lZs: numpy.ndarray
        The penalization terms for model complexity
    orders: list
        The orders of the hypergraph
    """
    orders = np.array(xgi.unique_edge_sizes(H)) - 1
    d_max = len(orders)

    Ds = np.zeros(d_max)
    lZs = np.zeros(d_max)

    Ds, lZs = optimization(H, tau, rescale_per_node=rescale_per_node, sparse=sparse)

    return Ds, lZs, orders


def pad_arr_list(arr_list, max_shape=None):
    """
    Pad a list of arrays with zeros to have the same shape.

    Parameters
    ----------
    arr_list : list of array-like
        The list of arrays to be padded.
    max_shape : int, optional
        The maximum length to pad the arrays to. If not provided (None), the maximum length
        among the arrays in the list will be used. Defaults to None.

    Returns
    -------
    list
        A list of padded arrays.

    Raises
    ------
    None

    Example
    -------
    arr_list = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6, 7, 8, 9])]
    padded_arr_list = pad_arr_list(arr_list)
    """

    if max_shape is None:
        max_shape = max([a.shape[0] for a in arr_list])

    # Pad the shorter arrays with zeros
    padded_arr_list = [
        np.pad(a, (0, max_shape - a.shape[0]), mode="constant", constant_values=None)
        for a in arr_list
    ]
    return padded_arr_list



def plot_information(H, taus, axs=None, label=None):
    """
    Plot information function

    Parameters
    ----------
    H : NetworkX Graph object
        The graph to be plotted
    taus : list of float
        The values of tau to plot
    axs : list of matplotlib axes, optional
        The list of matplotlib axes to plot the information function, by default None
    label : str, optional
        The label for the plot, by default None

    Returns
    -------
    tuple of matplotlib figure and list of matplotlib axes
        The matplotlib figure and the list of matplotlib axes used to plot the information function
    """
    if axs is None:
        fig, axs = plt.subplots(
            1, len(taus), figsize=(2 * len(taus), 2.1), constrained_layout=True
        )

    d_max = xgi.max_edge_order(H)
    orders = range(1, d_max + 1)

    for i, tau in enumerate(taus):

        D, lZ = optimization(H, tau)

        axs[i].plot(orders, lZ - D, "o-", label=label)

        axs[i].set_title(rf"$\tau = {tau}$", weight="bold")
        axs[i].set_xlabel("Max order")
        axs[i].set_xticks(orders)

    axs[0].set_ylabel("Quality function")

    sb.despine()

    return plt.gcf(), axs


def shuffle_hyperedges(S, order, p):
    """Shuffle existing hyperdeges of order d with probablity p
    Parameters
    ----------
    S: xgi.HyperGraph
        Hypergraph
    order: int
        Order of hyperedges to shuffle
    p: float
        Probability of shuffling each hyperedge
    Returns
    -------
    H: xgi.HyperGraph
        Hypergraph with edges of order d shuffled
    """

    nodes = S.nodes
    H = xgi.Hypergraph()
    #H = xgi.Hypergraph(S.edges.members(dtype=dict))
    H.add_nodes_from(nodes)
    H.add_edges_from(S.edges.members(dtype=dict))

    d_hyperedges = H.edges.filterby("order", order).members(dtype=dict)

    for id_, members in d_hyperedges.items():
        if random.random() <= p:
            H.remove_edge(id_)
            new_hyperedge = tuple(random.sample(nodes, order + 1))
            while new_hyperedge in H._edge.values():
                new_hyperedge = tuple(random.sample(nodes, order + 1))
            H.add_edge(new_hyperedge)

    assert H.num_nodes == S.num_nodes
    assert xgi.num_edges_order(H, 1) == xgi.num_edges_order(S, 1)
    assert xgi.num_edges_order(H, 2) == xgi.num_edges_order(S, 2)
    assert xgi.num_edges_order(H, order) == xgi.num_edges_order(S, order)

    return H
