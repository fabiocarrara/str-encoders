import argparse
import logging
import math
from pathlib import Path
import time

from expman import Experiment
import faiss
import h5py
import pandas as pd
from tqdm import tqdm

from spqr import SPQR
from utils import download_file, nice_logspace, compute_recalls


def main(args):
    exp = Experiment(args, root=args.exp_root, ignore=('data_root', 'exp_root'))
    print(exp)

    logging.basicConfig(level=logging.DEBUG, filename=exp.path_to('log.txt'))

    datapath = Path(args.data_root)
    datapath.mkdir(exist_ok=True)
    datapath = datapath / f'{args.dataset}.hdf5'
    if not datapath.exists():
        url = f'http://ann-benchmarks.com/{args.dataset}.hdf5'
        download_file(url, str(datapath))
    
    # load data in RAM
    logging.info('Loading data ...')
    with h5py.File(datapath, 'r') as dataset:
        x = dataset['train'][:]
        q = dataset['test'][:]
        n, d = x.shape
        true_neighbors = dataset['neighbors'][:]
        _, lim = true_neighbors.shape
    
    C, M, F = args.n_coarse_centroids, args.n_subvectors, args.n_fine_centroids
    log2F = int(math.log2(F))

    # train common index
    logging.info('Training index ...')
    common_build_time = time.time()
    index = faiss.index_factory(d, f'IVF{C},PQ{M}x{log2F}')
    index.train(x)
    common_build_time = time.time() - common_build_time
    logging.info('done')

    # save trained index
    logging.info('Saving index ...')
    trained_index_path = exp.path_to('empty_trained_index.faiss')
    faiss.write_index(index, trained_index_path)
    logging.info('done')

    metrics = []
    metrics_path = exp.path_to('metrics.csv')

    ## FAISS
    # build index
    logging.info('Building FAISS index ...')
    faiss_build_time = time.time()
    index.add(x)
    faiss_build_time = time.time() - faiss_build_time + common_build_time
    logging.info('done')

    # search and evaluate

    logging.info('Searching and Evaluating FAISS index ...')
    progress = tqdm(list(nice_logspace(C)), desc='FAISS')
    for nprobe in progress:
        progress.set_postfix({'nprobe': nprobe})
        index.nprobe = nprobe

        faiss_search_time = time.time()
        _, nns = index.search(q, k=lim)
        faiss_search_time = time.time() - faiss_search_time

        metrics.extend([{
            'index': 'ivfpq',
            'n_coarse_centroids': C,
            'n_subvectors': M,
            'n_fine_centroids': F,
            'build_time': faiss_build_time,
            'nprobe_or_length': nprobe,
            'search_time': faiss_search_time,
            'query_idx': i,
            'k': k,
            'recall@k': r,
        } for k in nice_logspace(lim)
        for i, r in enumerate(compute_recalls(true_neighbors[:, :k], nns[:, :k]))
        ])

    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    logging.info('done')

    ## SPQR
    logging.info('Building, Searching, and Evaluating SPQR index ...')
    spqr = SPQR(index)

    progress = tqdm(list(nice_logspace(F)), desc='SPQR')
    for l in progress:
        progress.set_postfix({'prefix': l})

        # build index
        spqr.reset()
        spqr_build_time = time.time()
        spqr.add(x, permutation_length=l)
        spqr_build_time = time.time() - spqr_build_time + common_build_time

        # search and evaluate
        spqr_search_time = time.time()
        _, nns = spqr.search(q, permutation_length=l, k=lim)
        spqr_search_time = time.time() - spqr_search_time

        metrics.extend([{
            'index': 'spqr',
            'n_coarse_centroids': C,
            'n_subvectors': M,
            'n_fine_centroids': F,
            'build_time': spqr_build_time,
            'nprobe_or_length': l,
            'search_time': spqr_search_time,
            'query_idx': i,
            'k': k,
            'recall@k': r,
        } for k in nice_logspace(lim)
          for i, r in enumerate(compute_recalls(true_neighbors[:, :k], nns[:, :k]))
        ])

    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    logging.info('done')
    return


ANN_DATASETS = (
    'deep-image-96-angular',
    'fashion-mnist-784-euclidean',
    'gist-960-euclidean',
    'glove-25-angular',
    'glove-50-angular',
    'glove-100-angular',
    'glove-200-angular',
    'kosarak-jaccard',
    'mnist-784-euclidean',
    'nytimes-256-angular',
    'sift-128-euclidean',
    'lastfm-64-dot',
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SPQR Experiments')
    parser.add_argument('dataset', choices=ANN_DATASETS, help='dataset from ann-benchmarks.com')
    parser.add_argument('-c', '--n-coarse-centroids', type=int, default=None, help='no of coarse centroids')
    parser.add_argument('-m', '--n-subvectors', type=int, default=None, help='no of subvectors for PQ')
    parser.add_argument('-f', '--n-fine-centroids', type=int, choices=(2**4, 2**8, 2**12, 2**16), default=2**8, help='no of fine centroids')

    parser.add_argument('--data-root', default='data/', help='where to store downloaded data')
    parser.add_argument('--exp-root', default='runs/', help='where to store results')
    args = parser.parse_args()
    main(args)