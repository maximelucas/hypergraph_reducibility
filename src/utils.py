"""
Useful functions
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import xgi

__all__ = [
    "symm_posdef_expm",
    "symm_posdef_logm",
    "pad_arr_list",
    "generate_geomspace_points",
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


import numpy as np


def generate_geomspace_points(
    start_value, end_value, num_points_between, num_points_before_after
):
    """
    Generate an array of points evenly spaced on a logarithmic scale between two specified values.

    Parameters
    ----------
    start_value : float
        The starting value of the range.
    end_value : float
        The ending value of the range.
    num_points_between : int
        The number of points to generate between start_value and end_value.
    num_points_before_after : int
        The total number of points to add before and after the specified range.

    Returns
    -------
    numpy.ndarray
        An array containing points evenly spaced on a logarithmic scale.

    Examples
    --------
    >>> generate_geomspace_points(1, 1000, 4, 1)
    array([   0.1,    1. ,   10. ,  100. , 1000. , 10000. ])
    """

    # Generate points between start and end values
    points_between = np.geomspace(start_value, end_value, num=num_points_between)

    # Compute the logarithmic distance between consecutive points
    log_distance = np.log10(points_between[1]) - np.log10(points_between[0])

    # Split num_points_before_after into points before and after
    num_points_before = num_points_before_after
    num_points_after = num_points_before_after

    # Extend the array by adding points before and after
    points_before = np.geomspace(
        start_value / (10 ** (num_points_before * log_distance)),
        start_value,
        num=num_points_before,
        endpoint=False,
    )
    points_after = np.geomspace(
        end_value * (10**log_distance),
        end_value * (10 ** (num_points_after * log_distance)),
        num=num_points_after,
    )

    # Concatenate all points
    final_array = np.concatenate((points_before, points_between, points_after))

    return final_array


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
    # here to avoid circular import
    from hypergraph_reducibility import optimization

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
