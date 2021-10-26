import argparse
import logging
import math
from pathlib import Path
import pickle
import time

from expman import Experiment
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from surrogate import SPQR
from utils import get_ann_benchmark, nice_logspace, compute_recalls


def main(args):
    ignore = ('data_root', 'exp_root', 'force', 'index_batch_size', 'search_batch_size')
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
    
    C, M, F = args.n_coarse_centroids, args.n_subvectors, args.n_fine_centroids

    trained_index_path = exp.path_to('empty_trained_index.pickle')
    train_time_path = exp.path_to('train_time.txt')
    if not Path(trained_index_path).exists() or args.force:
        # train common index
        logging.info('Training index ...')
        train_time = time.time()
        index = SPQR(d, n_coarse_centroids=C, n_subvectors=M, n_fine_centroids=F)
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

    metrics_path = exp.path_to('metrics.csv.gz')
    metrics = pd.DataFrame() # pd.read_csv(metrics_path) if Path(metrics_path).exists() else pd.DataFrame()  #################### TOCHANGE

    ## SPQR
    logging.info('Building, Searching, and Evaluating SPQR index ...')

    progress = tqdm(list(nice_logspace(min(F, 25))), desc='SPQR')
    for L in progress:
        progress.set_postfix({'prefix': L})

        # skip if already done
        if False: # not args.force:  #################### TOCHANGE
            if not metrics.empty and (metrics['prefix_length'] == L).any():
                logging.info(f'SKIPPING SPQR C={C} M={M} F={F} L={L}')
                continue

        logging.info(f'SPQR C={C} M={M} F={F} L={L}')

        # build index
        built_index_path = exp.path_to(f'built_index_perm_length_{L}.pickle')
        build_time_path = exp.path_to(f'build_time_perm_length_{L}.txt')
        if not Path(built_index_path).exists() or True: # args.force:  #################### TOCHANGE
            logging.info('Building index ...')
            index.reset()
            index.prefix_length = L
            
            batch_size = n if args.index_batch_size is None else args.index_batch_size
            build_time = time.time()
            for i in trange(0, n, batch_size, desc='ADD'):
                index.add(x[i:i+batch_size])
            index.commit()
            build_time = time.time() - build_time
            logging.info(f'Done in {build_time} s.')

            # save built index
            # logging.info(f'Saving built index: {built_index_path}')
            # with open(built_index_path, 'wb') as f:
            #     pickle.dump(index, f)

            with open(build_time_path, 'w') as f:
                f.write(str(build_time))
        else:
            logging.info(f'Reading prebuilt index from: {built_index_path}')
            with open(built_index_path, 'rb') as f:
                index = pickle.load(f)

            with open(build_time_path, 'r') as f:
                build_time = float(f.read().strip())

        # search and evaluate
        batch_size = len(q) if args.search_batch_size is None else args.search_batch_size
        search_time = time.time()
        
        logging.info(f'Searching index ...')
        nns = []
        for i in trange(0, len(q), batch_size, desc='SEARCH'):
            _, nns_batch = index.search(q[i:i+batch_size], k=lim)
            nns.append(nns_batch)
        nns = np.vstack(nns)
        search_time = time.time() - search_time
        logging.info(f'Done in {search_time} s.')

        new_metrics = pd.DataFrame([{
            'index_type': 'spqr',
            'build_time': build_time,
            'num_entries': index.db.nnz,
            'prefix_length': L,
            'search_time': search_time,
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
    parser.add_argument('-f', '--n-fine-centroids', type=int, default=2**8, help='no of fine centroids')

    parser.add_argument('--data-root', default='data/', help='where to store downloaded data')
    parser.add_argument('--exp-root', default='runs/', help='where to store results')

    parser.add_argument('--force', default=False, action='store_true', help='force index training')
    parser.add_argument('-b', '--index-batch-size', type=int, default=None, help='index data in batches with this size')
    parser.add_argument('-B', '--search-batch-size', type=int, default=None, help='search data in batches with this size')
    args = parser.parse_args()
    main(args)