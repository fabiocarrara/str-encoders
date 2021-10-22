import itertools
from pathlib import Path
import time

import h5py
import numpy as np
import requests
from tqdm import tqdm


def get_ann_benchmark(dataset_name, data_root='./data'):
    datapath = Path(data_root)
    datapath.mkdir(exist_ok=True)
    datapath = datapath / f'{dataset_name}.hdf5'
    if not datapath.exists():
        url = f'http://ann-benchmarks.com/{dataset_name}.hdf5'
        download_file(url, str(datapath))
    
    return h5py.File(datapath, 'r')


def download_file(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def nice_logspace(max_n):
    """ Returns log-spaces values like this: 1, 2, 5, 10, 20, 50, ... """
    def _inf_gen():
        yield from (1, 2, 5)
        i = 1
        while True:
            yield 10 * i
            yield 25 * i
            yield 50 * i
            i *= 10

    yield from itertools.takewhile(lambda x: x < max_n, _inf_gen())
    yield max_n


def compute_recalls(true_neighbors, predicted_neighbors):
    """ Computes recalls of (multiple) queries. """

    # the verbose version
    """
    recalls = []
    for t, p in zip(true_neighbors, predicted_neighbors):
        intersection = np.intersect1d(t, p)
        recall = len(intersection) / len(t)
        recalls.append(recall)

    return np.array(recalls)
    """

    # the one-liner
    return np.array([len(np.intersect1d(t, p)) / len(t) for t, p in zip(true_neighbors, predicted_neighbors)])


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


class Timer:

    def __init__(self, desc=None):
        self.desc = desc

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.stop = time.time()
        self.duration = self.stop - self.start
        prefix = (self.desc + ': ') if self.desc else ''
        print(f"{prefix}{self.duration} s")