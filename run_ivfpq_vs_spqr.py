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
from utils import get_ann_benchmark, nice_logspace, compute_recalls


def main(args):
    exp = Experiment(args, root=args.exp_root, ignore=('data_root', 'exp_root', 'force', 'batch_size'))
    print(exp)

    # setup logging
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(module)s.%(funcName)s:%(lineno)d - %(message)s',
        handlers=[logging.FileHandler(exp.path_to('log.txt'), mode='w'), stream_handler]
    )

    dataset = get_ann_benchmark(args.dataset, args.data_root)
    
    # load data in RAM
    logging.info('Loading data ...')
    x = dataset['train'][:]
    q = dataset['test'][:]
    n, d = x.shape
    true_neighbors = dataset['neighbors'][:]
    _, lim = true_neighbors.shape
    
    C, M, F = args.n_coarse_centroids, args.n_subvectors, args.n_fine_centroids
    log2F = int(math.log2(F))

    trained_index_path = exp.path_to('empty_trained_index.faiss')
    common_build_time_path = exp.path_to('common_build_time.txt')
    if not Path(trained_index_path).exists() or args.force:
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
        with open(common_build_time_path, 'w') as f:
            f.write(str(common_build_time))
        logging.info('done')
    else:
        logging.info(f'Reading prebuilt index from: {trained_index_path} ...')
        index = faiss.read_index(trained_index_path)
        with open(common_build_time_path, 'r') as f:
            common_build_time = float(f.read().strip())
        logging.info('done')

    metrics_path = exp.path_to('metrics.csv.gz')
    metrics = pd.read_csv(metrics_path) if Path(metrics_path).exists() else pd.DataFrame()
    
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

        # skip if already done
        if not args.force:
            if not metrics.empty and ((metrics['index'] == 'ivfpq') & (metrics['nprobe_or_length'] == nprobe)).any():
                logging.info(f'SKIPPING IVFPQ C={C} M={M} F={F} nprobe={nprobe}')
                continue

        index.nprobe = nprobe

        faiss_search_time = time.time()
        _, nns = index.search(q, k=lim)
        faiss_search_time = time.time() - faiss_search_time

        new_metrics = pd.DataFrame([{
            'index': 'ivfpq',
            'build_time': faiss_build_time,
            'nprobe_or_length': nprobe,
            'search_time': faiss_search_time,
            'k': k,
            'recall@k': compute_recalls(true_neighbors[:, :k], nns[:, :k]).mean(),
        } for k in nice_logspace(lim)])

        metrics = pd.concat((metrics, new_metrics), ignore_index=True)
        metrics.to_csv(metrics_path, index=False)

    logging.info('done')

    ## SPQR
    logging.info('Building, Searching, and Evaluating SPQR index ...')
    spqr = SPQR(index)

    progress = tqdm(list(nice_logspace(min(F, 25))), desc='SPQR')
    for L in progress:
        progress.set_postfix({'prefix': L})

        # skip if already done
        if not args.force:
            if not metrics.empty and ((metrics['index'] == 'spqr') & (metrics['nprobe_or_length'] == L)).any():
                logging.info(f'SKIPPING SPQR C={C} M={M} F={F} L={L}')
                continue

        # build index
        spqr.reset()
        spqr_build_time = time.time()
        if args.batch_size:
            for i in range(0, len(x), args.batch_size):
                spqr.add(x[i:i+args.batch_size], permutation_length=L)
        else:
            spqr.add(x, permutation_length=L)
        spqr_build_time = time.time() - spqr_build_time + common_build_time

        # search and evaluate
        spqr_search_time = time.time()
        _, nns = spqr.search(q, permutation_length=L, k=lim)
        spqr_search_time = time.time() - spqr_search_time

        new_metrics = pd.DataFrame([{
            'index': 'spqr',
            'build_time': spqr_build_time,
            'nprobe_or_length': L,
            'search_time': spqr_search_time,
            'k': k,
            'recall@k': compute_recalls(true_neighbors[:, :k], nns[:, :k]).mean(),
        } for k in nice_logspace(lim)])

        metrics = pd.concat((metrics, new_metrics), ignore_index=True)
        metrics.to_csv(metrics_path, index=False)

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

    parser.add_argument('--force', default=False, action='store_true', help='force index training')
    parser.add_argument('-b', '--batch-size', type=int, default=None, help='index data in batches with this size')
    args = parser.parse_args()
    main(args)