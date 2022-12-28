import itertools
from pathlib import Path
import time

import h5py
import numpy as np
import requests
from sklearn.preprocessing import normalize
from tqdm import tqdm


def get_random_dataset(metric, d, nx, nq, data_root='./data'):
    assert metric in ('dot', 'angular'), "Random dataset metric must be one of ('dot', 'angular')."
    datapath = Path(data_root)
    datapath.mkdir(exist_ok=True)
    datapath = datapath / f'random-{d}-{metric}-{nx}db-{nq}q.hdf5'
    if not datapath.exists():

        test = 2 * np.random.rand(nq, d).astype(np.float32) - 1
        train = 2 * np.random.rand(nx, d).astype(np.float32) - 1
        if metric == 'angular':
            distances = normalize(test).dot(normalize(train).T)
        else:    
            distances = test.dot(train.T)
        neighbors = distances.argsort(axis=1)[:, ::-1].astype(np.int32)

        with h5py.File(datapath, 'w') as f:
            f.create_dataset('train', data=train)
            f.create_dataset('test', data=test)
            f.create_dataset('distances', data=distances)
            f.create_dataset('neighbors', data=neighbors)
    
    return h5py.File(datapath, 'r')


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


def get_dataset(dataset_name, data_root='./data'):
    if dataset_name.startswith('random'):
        attributes = dataset_name.split('-')
        metric = attributes[1]
        d, nx, nq = map(int, attributes[2:])
        return get_random_dataset(metric, d, nx, nq, data_root=data_root)
    
    return get_ann_benchmark(dataset_name, data_root=data_root)


def nice_logspace(max_n):
    """ Returns log-spaces values like this: 1, 2, 5, 10, 25, 50, ... """
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


class Timer:

    def __init__(self, desc=None, fmt='g'):
        self.desc = desc
        self.fmt = fmt

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.stop = time.time()
        self.duration = self.stop - self.start
        prefix = (self.desc + ': ') if self.desc else ''
        print(f"{prefix}{self.duration:{self.fmt}} s")