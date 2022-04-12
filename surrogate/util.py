import numpy as np


def bottomk_sorted(x, k, axis=0):
    """ Returns sorted bottom-K indices along an axis.

    Args:
        x (ndarray): the source array.
        k (int): number of bottom (smallest) elements to return.
        axis (int): axis over which to extract the bottom-k values.

    Returns:
        ndarray: (N,K)-shaped array of indices.
    """
    if k >= x.shape[axis]:  # full sort needed
        bottomk = x.argsort(axis=axis)
    else:
        unsorted_bottomk = x.argpartition(k, axis=axis).take(indices=range(k), axis=axis)
        sorted_topk_idx = np.take_along_axis(x, unsorted_bottomk, axis=axis).argsort(axis=axis)
        bottomk = np.take_along_axis(unsorted_bottomk, sorted_topk_idx, axis=axis)

    return bottomk


def topk_sorted(x, k, axis=0):
    """ Returns sorted top-K indices along an axis.

    Args:
        x (ndarray): the source array.
        k (int): number of top (largest) elements to return.
        axis (int): axis over which to extract the top-k values.

    Returns:
        ndarray: (N,K)-shaped array of indices.
    """
    return bottomk_sorted(-x, k, axis=axis)