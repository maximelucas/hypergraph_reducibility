"""
Useful functions
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import xgi

from reducibility_hg import optimization

__all__ = [
    "symm_posdef_expm",
    "symm_posdef_logm",
    "pad_arr_list",
    "plot_information",
]


def symm_posdef_expm(matrix):
    """Matrix exponential for symmetric positive semidefinite matrix"""

    eigvals, eigvecs = np.linalg.eigh(
        matrix
    )  # Compute eigenvalues and eigenvectors of symmetric matrix
    exp_eigvals = np.exp(eigvals)  # Compute exponential of eigenvalues

    # Reconstruct matrix with new eigenvalues
    exp_matrix = eigvecs @ np.diag(exp_eigvals) @ eigvecs.T

    return exp_matrix


def symm_posdef_logm(matrix):
    """Matrix logarithm for symmetric positive semidefinite matrix"""

    eigvals, eigvecs = np.linalg.eigh(
        matrix
    )  # Compute eigenvalues and eigenvectors of symmetric matrix
    log_eigvals = np.log(eigvals)  # Compute logarithm of eigenvalues

    # Reconstruct matrix with new eigenvalues
    log_matrix = eigvecs @ np.diag(log_eigvals) @ eigvecs.T

    return log_matrix


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
