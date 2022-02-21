import argparse
from functools import partial
import logging
import math
from pathlib import Path
import pickle
import time

from expman import Experiment
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

tqdm = partial(tqdm, dynamic_ncols=True, leave=False)
trange = partial(trange, dynamic_ncols=True, leave=False)

from surrogate import IVFTopKSQ
from utils import get_ann_benchmark, nice_logspace, compute_recalls


def load_or_train_index(d, x, C, M, K, S, R, N, trained_index_path, train_time_path):
    if not Path(trained_index_path).exists() or args.force:
        # train common index
        logging.info('Training index ...')
        train_time = time.time()
        index = IVFTopKSQ(d, n_coarse_centroids=C, n_subvectors=M, k=K, sq_factor=S, rectify_negatives=R, l2_normalize=N)
        index.train(x)
        train_time = time.time() - train_time
        logging.info(f'Done in {train_time} s.')

        # save trained index
        logging.info(f'Saving trained index: {trained_index_path}')
        with open(trained_index_path, 'wb') as f:
            pickle.dump(index, f)

        with open(train_time_path, 'w') as f:
            f.write(str(train_time))
    else:
        logging.info(f'Reading pretrained index from: {trained_index_path}')
        with open(trained_index_path, 'rb') as f:
            index = pickle.load(f)
            # index.parallel = True

        with open(train_time_path, 'r') as f:
            train_time = float(f.read().strip())
    
    return index, train_time


def load_or_build_index(index, x, index_batch_size, force, built_index_path, build_time_path):
    if not Path(built_index_path).exists() or force:
        logging.info('Building index ...')
        index.reset()
        
        n = len(x)
        batch_size = n if index_batch_size is None else index_batch_size
        build_time = time.time()
        for i in trange(0, n, batch_size, desc='ADD'):
            index.add(x[i:i+batch_size])
        index.commit()
        build_time = time.time() - build_time
        logging.info(f'Done in {build_time} s.')

        # save built index
        logging.info(f'Saving built index: {built_index_path}')
        with open(built_index_path, 'wb') as f:
            pickle.dump(index, f)

        with open(build_time_path, 'w') as f:
            f.write(str(build_time))
    else:
        logging.info(f'Reading prebuilt index from: {built_index_path}')
        with open(built_index_path, 'rb') as f:
            index = pickle.load(f)

        with open(build_time_path, 'r') as f:
            build_time = float(f.read().strip())
    
    return index, build_time

# @tqdm_logging
def main(args):
    ignore = ('data_root', 'exp_root', 'force', 'index_batch_size', 'search_batch_size', 'search_timeout')
    exp = Experiment(args, root=args.exp_root, ignore=ignore)
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
    logging.info(f'Loading data: {args.dataset}.hdf5')
    x = dataset['train'][:]
    q = dataset['test'][:]
    n, d = x.shape
    true_neighbors = dataset['neighbors'][:]
    _, lim = true_neighbors.shape
    
    C, M, K, S, R, N = args.n_coarse_centroids, args.n_subvectors, args.keep, args.sq_factor, args.rectify_negatives, args.l2_normalize

    trained_index_path = exp.path_to('empty_trained_index.pickle')
    train_time_path = exp.path_to('train_time.txt')
    
    index, train_time = load_or_train_index(d, x, C, M, K, S, R, N, trained_index_path, train_time_path)

    built_index_path = exp.path_to(f'built_index.pickle')
    build_time_path = exp.path_to(f'build_time.txt')

    index, build_time = load_or_build_index(index, x, args.index_batch_size, args.force, built_index_path, build_time_path)

    metrics_path = exp.path_to('metrics.csv.gz')
    metrics = pd.read_csv(metrics_path) if Path(metrics_path).exists() else pd.DataFrame()

    ## IVFTHRSQ
    logging.info('Searching, and Evaluating IVFTopKSQ index ...')

    with logging_redirect_tqdm():
        logging.info(f'IVFTopKSQ C={C} M={M} K={K} S={S} R={R} N={N}')        

        search_time = -1
        progress_P = tqdm(list(nice_logspace(min(C, 50))))
        for nprobe in progress_P:
            progress_P.set_postfix({'nprobe': nprobe})

            # skip if already done
            if not args.force:
                if not metrics.empty and 'nprobe' in metrics.columns:
                    sel = (metrics['nprobe'] == nprobe)
                    if sel.any():
                        search_time = metrics[sel].iloc[0]['search_time']
                        logging.info(f'SKIPPING IVFTopKSQ C={C} M={M} K={K} S={S} R={R} N={N} nprobe={nprobe}')
                        continue

            if search_time > args.search_timeout:
                logging.info(f'Skipping nprobes >= {nprobe}, search() would take too long (> {args.search_timeout} s).')
                break
            
            index.nprobe = nprobe

            # search and evaluate
            batch_size = len(q) if args.search_batch_size is None else args.search_batch_size
            batch_size = math.ceil(batch_size / nprobe)

            search_time = time.time()
            nns = []
            for i in trange(0, len(q), batch_size, desc='SEARCH'):
                _, nns_batch = index.search(q[i:i+batch_size], k=lim)
                nns.append(nns_batch)
            nns = np.vstack(nns)
            search_time = time.time() - search_time

            recalls = {k: compute_recalls(true_neighbors[:, :k], nns[:, :k]).mean() for k in nice_logspace(lim)}

            new_metrics = pd.DataFrame([{
                'index_type': 'ivf-topk-sq',
                'train_time': train_time,
                'build_time': build_time,
                'num_entries': index.db.nnz,
                'nprobe': nprobe,
                'search_time': search_time,
                'k': k,
                'recall@k': r_at_k,
            } for k, r_at_k in recalls.items()])

            metrics = pd.concat((metrics, new_metrics), ignore_index=True)
            metrics.to_csv(metrics_path, index=False)

            k = 100
            logging.info(f'IVFTopKSQ C={C} M={M} K={K} S={S} R={R} N={N} nprobe={nprobe}: R@{k}={recalls[k]:.3f} search-time: {search_time:2.2f} s')

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
    parser = argparse.ArgumentParser(description='Run IVFTHRSQ Experiments')
    parser.add_argument('dataset', choices=ANN_DATASETS, help='dataset from ann-benchmarks.com')
    parser.add_argument('-c', '--n-coarse-centroids', type=int, default=None, help='no of coarse centroids')
    parser.add_argument('-m', '--n-subvectors', type=int, default=None, help='no of subvectors for PQ')
    parser.add_argument('-K', '--keep', type=float, default=0.75, help='Controls how many values are kept when encoding. Must be between 0.0 and 1.0 inclusive.')
    parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
    parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')

    parser.add_argument('--data-root', default='data/', help='where to store downloaded data')
    parser.add_argument('--exp-root', default='runs/', help='where to store results')

    parser.add_argument('--force', default=False, action='store_true', help='force index training')
    parser.add_argument('-b', '--index-batch-size', type=int, default=None, help='index data in batches with this size')
    parser.add_argument('-B', '--search-batch-size', type=int, default=None, help='search data in batches with this size')
    parser.add_argument('-t', '--search-timeout', type=int, default=1000, help='stop parameter search when search time (in seconds) is over this value')

    args = parser.parse_args()
    main(args)