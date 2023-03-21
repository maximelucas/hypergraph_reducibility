"""
Script for computing and plotting hypergraph information
"""

import random 

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sb
from scipy import linalg

import xgi

__all__ = [
    "find_charact_tau",
    "construct_hg_multilayer",
    "density",
    "partition",
    "KL",
    "penalization",
    "optimization",
    "entropy",
    "compute_information",
    "plot_information",
    "shuffle_hyperedges"
]

def find_charact_tau(H, orders, weights, rescale_per_node=False):
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
    L_multi = xgi.multiorder_laplacian(H, orders, weights, rescale_per_node=rescale_per_node)
    Ls = np.sort(np.linalg.eigvals(L_multi))

    return 1 / Ls[-1]

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


def density(Lap, tau):
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
    rho = linalg.expm(-tau * Lap)
    rho = rho / np.trace(rho) + np.eye(len(rho)) * 10**-10
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
    return np.trace(linalg.expm(-2 * tau * Lap))


def KL(rho_emp, rho_model):
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
    return np.trace(
        np.matmul(rho_emp, linalg.logm(rho_emp))
        - np.matmul(rho_emp, linalg.logm(rho_model))
    )

def penalization(Lap, tau):
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
    Z = np.trace(linalg.expm(- tau * Lap))
    N = len(Lap)
    return np.log(N) - entropy(Lap,tau)

def optimization(H, tau, rescale_per_node=False):
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
    L_multi = xgi.multiorder_laplacian(H, orders, weights, rescale_per_node=rescale_per_node)
    
    rho_all = density(L_multi, tau)
    
    D = [] # Learning error
    lZ = [] # Penalization term for model complexity
    N = H.num_nodes
    
    for l in range(len(orders)):
        L_l = xgi.multiorder_laplacian(H, orders[0:l+1], weights[0:l+1], rescale_per_node=rescale_per_node)
        rho_l = density(L_l, tau)
        d = KL(rho_all, rho_l)
        z = penalization(L_l,tau)
        
        D.append(d)   
        lZ.append(z)

    lZ = np.array(lZ)
    D = np.array(D)

    return D, lZ


def entropy(L, tau):
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
    Ls = np.linalg.eigvals(L)  # Calculate eigenvalues of L
    Z = np.sum(np.exp(-tau * Ls))  # Calculates the partition function
    p = np.exp(-tau * Ls) / Z  # Calculates the probabilities
    p = np.delete(p,np.where(p<10**-8))
    S = np.sum(-p * np.log(p))  # entropy
    return S


def compute_information(H, tau, rescale_per_node=False):
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
    d_max=len(orders)

    Ds = np.zeros(d_max)
    lZs = np.zeros(d_max)
    
    Ds, lZs = optimization(H, tau, rescale_per_node=rescale_per_node)

    return Ds, lZs, orders


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
    H = xgi.Hypergraph(S)

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

    return H
