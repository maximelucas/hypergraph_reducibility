"""
Functions for computing the functinoal reducibility of hypergraphs
"""

import numpy as np
import scipy.sparse as sp
import xgi
from numpy.linalg import eigvalsh
from tqdm import tqdm

from utils import symm_posdef_expm, symm_posdef_logm

__all__ = [
    "find_charact_tau",
    "density",
    "KL",
    "penalization",
    "entropy",
    "optimization",
]


def find_charact_tau(
    H,
    orders,
    weights,
    rescale_per_node=False,
    rescale_per_order=True,
    sparse_Lap=True,
    idx=-1,
):
    """
    Compute characteristic timescale tau.

    Tau is commputed as 1 / the idx-th eigenvalue of the multiorder Laplacian.

    Parameters
    ----------
    H : xgi Hypergraph
        Input hypergraph
    orders : list
        List of integers representing the orders of the laplacian matrices to consider.
    rescale_per_node : bool, optional
        Wether to rescale the Laplacian at each order per node
    rescale_per_order: bool, optional
        Wether to rescale the Laplacian at each order per order
    sparse_Lap: bool, optional
        Wheter to use a sparse version of the Laplacian to speed up computations.
    idx: int, optional
        Index of the eigenvalue to use.
    Returns
    -------
    float
        The value of tau calculated from the eigenvalues of the multi-order laplacian matrix.

    See also
    ---
    multiorder_laplacian
    """
    L_multi = xgi.multiorder_laplacian(
        H,
        orders,
        weights,
        rescale_per_node=rescale_per_node,
        rescale_per_order=rescale_per_order,
        sparse=sparse_Lap,
    )

    if sparse_Lap:
        L_multi = L_multi.todense()

    N = len(L_multi)

    # if sparse and idx==-1:
    #    raise ValueError("Cannot compute the last eigenvalue for sparse matrices.")

    # if sparse:
    #    lambdas = sp.eigsh(L_multi, k=N-1, return_eigenvectors=False)
    # else
    lambdas = eigvalsh(L_multi)

    return 1 / lambdas[idx]


def density(Lap, tau, sparse=False):
    """
    Computes the density matrix for a Laplacian with scale `tau`.

    Parameters
    ----------
    Lap : np.ndarray
        The Laplacian matrix
    tau : float
        The scale of the Laplacian
    sparse: bool, optional
        If True, return sparse matrix. Default: False.

    Returns
    -------
    np.ndarray: the density matrix
    """
    if sparse:
        rho = sp.linalg.expm(-tau * Lap)
        rho = rho / rho.trace()
        rho = rho + sp.eye(rho.shape[0]) * 10**-10
    else:
        # rho = expm(-tau * Lap)
        rho = symm_posdef_expm(-tau * Lap)
        rho = rho / np.trace(rho)
        rho = rho + np.eye(len(rho)) * 10**-10
    return rho


def KL(rho_emp, rho_model, sparse=False):
    """
    Computes the Kullback-Leibler (KL) divergence between density matrices.

    The first density matrix is associated with empirical observation `rho_emp` and
    the second density matrix is associated with a model `rho_model`.

    Parameters
    ----------
    rho_emp : (np.ndarray)
        Density matrix of empirical observation
    rho_model : np.ndarray
        Density matrix of model

    Returns
    -------
    float : the KL divergence between `rho_emp` and `rho_model`
    """

    if sparse:
        rho_emp = rho_emp.toarray()
        rho_model = rho_model.toarray()

    # return np.trace(rho_emp @ logm(rho_emp) - rho_emp @ logm(rho_model))
    log_emp = symm_posdef_logm(rho_emp)
    log_mod = symm_posdef_logm(rho_model)
    mul1 = np.matmul(rho_emp, log_emp)
    mul2 = np.matmul(rho_emp, log_mod)
    return np.trace(mul1 - mul2)


def penalization(Lap, tau, sparse=False):
    """
    Computes the partition function of a Laplacian with scale `tau`.

    Parameters
    ----------
    Lap : np.ndarray
        The Laplacian matrix
    tau : float
        The scale of the Laplacian
    sparse: bool, optional
        If True, compute the `entropy` as a sparse matrix for speed.
        Default: False.

    Returns
    -------
    float: the partition function
    """
    N = Lap.shape[0]
    return np.log(N) - entropy(Lap, tau, sparse=sparse)


def entropy(L, tau, sparse=False):
    """
    Computes the entropy associated to the Laplacian matrix.

    Parameters
    ----------
    L: numpy.ndarray
        The Laplacian matrix
    tau: float
        The scale of signal propagation
    sparse: bool, optional
        If True, computes N-1 eigenvalues from sparse matrix.
        Default: False.

    Returns
    -------
    S: float
        The entropy of the graph
    """

    N = L.shape[0]

    if sparse:
        lambdas, _ = sp.eigsh(
            L, k=N - 1
        )  # this uses all eigenvalues except the last one.. not fully exact
    else:
        lambdas = eigvalsh(L)  # Calculate eigenvalues of L

    expL = np.exp(-tau * lambdas)
    Z = np.sum(expL)  # Calculates the partition function
    p = expL / Z  # Calculates the probabilities
    p = np.delete(p, np.where(p < 10**-8))
    S = np.sum(-p * np.log(p))  # entropy
    return S


def optimization(
    H,
    tau,
    rescaling_factors=None,
    tau_per_order=False,
    rescale_per_node=False,
    rescale_per_order=True,
    sparse=False,
    sparse_Lap=True,
):
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
        rescaling_factors = np.ones_like(orders)

    if tau_per_order and not np.allclose(rescaling_factors, 1):
        raise UserWarning(
            "Computing `tau_per_order` but the rescaling_factors are not ones."
        )

    L_multi = xgi.multiorder_laplacian(
        H,
        orders,
        weights,
        rescale_per_node=rescale_per_node,
        rescale_per_order=rescale_per_order,
        sparse=sparse_Lap,
    )

    if sparse_Lap:
        L_multi = L_multi.todense()

    if tau_per_order:
        lambdas = eigvalsh(L_multi)
        tau = 1 / lambdas[-1]

    rho_all = density(L_multi, tau, sparse=sparse)

    D = []  # Learning error
    lZ = []  # Penalization term for model complexity
    N = H.num_nodes

    for l in tqdm(range(len(orders))):
        L_l = xgi.multiorder_laplacian(
            H,
            orders[0 : l + 1],
            weights[0 : l + 1],
            rescale_per_node=rescale_per_node,
            rescale_per_order=rescale_per_order,
            sparse=sparse_Lap,
        )

        if sparse_Lap:
            L_l = L_l.todense()

        if tau_per_order:
            lambdas = eigvalsh(L_l)
            tau = 1 / lambdas[-1]

        rho_l = density(L_l, tau * rescaling_factors[l], sparse=sparse)
        d = KL(rho_all, rho_l, sparse=sparse)
        z = penalization(L_l, tau * rescaling_factors[l], sparse=sparse)

        D.append(d)
        lZ.append(z)

    lZ = np.array(lZ)
    D = np.array(D)

    return D, lZ
