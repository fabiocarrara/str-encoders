import logging
import math

import numpy as np
from joblib import cpu_count, delayed, Parallel
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

from . import util
from .str_index import SurrogateTextIndex


def _ivf_topk_sq_encode(
    x,                  # featues to encode
    m,                  # number of subvectors
    k,                  # the number or fraction of high-value components to keep
    centroids,          # l1 quantizer centroids
    sq_factor,          # quantization factor
    rectify_negatives,  # whether to apply crelu
    l2_normalize,       # whether to l2-normalize vectors
    nprobe,             # how many coarse centroids to consider
    transpose,          # if True, transpose result (returns VxN)
    format,             # sparse format of result ('csr', 'csc', 'coo', etc.)
):
    n, d = x.shape
    c = len(centroids)
    nprobe = min(nprobe, c)

    if l2_normalize:
        x = normalize(x)
    
    l1_centroid_distances = cdist(x, centroids, metric='sqeuclidean')
    coarse_codes = util.bottomk_sorted(l1_centroid_distances, nprobe, axis=1)  # n x nprobe

    dsub = d // m
    x = x.reshape(n, m, dsub)  # n x m x d/m

    mult = 2 if rectify_negatives else 1
    xx = np.fabs(x) if rectify_negatives else x

    # keep the topk components per subvector
    k = int(k * dsub) if isinstance(k, float) else k

    cols = util.topk_sorted(xx, k, axis=2)  # n x m x k
    cols += np.arange(m).reshape(1, m, 1) * dsub  # shift indices to the right subvector
    cols = cols.reshape(n, -1)  # n x (m*k)

    rows = np.arange(n).reshape(n, 1)  # n x 1

    x = x.reshape(n, d)
    xx = xx.reshape(n, d)

    is_positive = x[rows, cols] > 0  # n x (m*k)
    data = xx[rows, cols]  # n x (m*k)
    
    if rectify_negatives:
        cols += np.where(is_positive, 0, d)  # shift indices of negatives after positives
    
    cols = np.stack([cols + coarse_codes[:, [i]] * mult * d for i in range(nprobe)], axis=-1)  # n x (m*k) x nprobe
    rows = np.expand_dims(rows, axis=-1)  # n x 1 x 1
    data = np.expand_dims(data, axis=-1)  # n x (m*k) x 1

    rows, cols, data = np.broadcast_arrays(rows, cols, data)  # n x (m*k) x nprobe

    rows = rows.flatten()
    cols = cols.flatten()
    data = data.flatten()

    shape = (n, c * mult * d)

    # scalar quantization
    data = np.fix(sq_factor * data.astype(np.float64)).astype('int')

    if transpose:
        rows, cols = cols, rows
        shape = shape[::-1]

    spclass = getattr(sparse, f'{format}_matrix')
    return spclass((data, (rows, cols)), shape=shape)


class IVFTopKSQ(SurrogateTextIndex):
    
    def __init__(
        self,
        d,
        n_coarse_centroids=1,  # FIXME: this should be a required positional argument
        n_subvectors=1,
        keep=0.75,
        sq_factor=1e5,
        rectify_negatives=True,
        l2_normalize=True,
        parallel=True
    ):
        """ Constructor
        Args:
            d (int): the number of dimensions of the vectors to be encoded.
            n_coarse_centroids (int): the number of coarse centroids of the level 1
                                      quantizer (voronoi partitioning).
            n_subvectors (int): the number of subvectors of the level 2 quantizer
                                (scalar quantization).
            k (int or float): if int, number of components per subvector to keep
                              (must be between 0 and (d / n_subvectors));
                              if float, the fraction of components to keep (must
                              be between 0.0 and 1.0). Defaults to 0.25.
            sq_factor (float): multiplicative factor controlling scalar quantization.
                               Defaults to 1000.
            rectify_negatives (bool): whether to reserve d additional dimensions
                                      to encode negative values separately 
                                      (a.k.a. apply CReLU transform).
                                      Defaults to True.
            l2_normalize (bool): whether to apply l2-normalization before processing vectors;
                                 set this to False if vectors are already normalized.
                                 Defaults to True.
        """

        self.d = d
        self.c = n_coarse_centroids
        self.m = n_subvectors
        self.keep = keep
        self.sq_factor = sq_factor
        self.rectify_negatives = rectify_negatives
        self.l2_normalize = l2_normalize
        self.nprobe = 1

        self._centroids = None

        vocab_size = self.c * 2 * d if self.rectify_negatives else self.c * d
        super().__init__(vocab_size, parallel)
    
    def add_subparser(subparsers, **kws):
        parser = subparsers.add_parser('ivf-topk-sq', help='Chunked TopK Scalar Quantization', **kws)
        parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')
        parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
        parser.add_argument('-c', '--n-coarse-centroids', type=int, default=512, help='no of coarse centroids')
        parser.add_argument('-m', '--n-subvectors', type=int, default=1, help='no of subvectors')
        parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
        parser.add_argument('-k', '--keep', type=float, default=0.25, help='Controls how many values are discarded when encoding. Must be between 0.0 and 1.0 inclusive.')
        parser.add_argument('-p', '--nprobe', type=int, default=1, help='how many partitions to visit at query time')
        parser.set_defaults(
            train_params=('l2_normalize', 'n_coarse_centroids', 'n_subvectors'),
            build_params=('rectify_negatives', 'sq_factor', 'keep'),
            query_params=('nprobe',)
        )

    def encode(self, x, inverted=True, query=False, nprobe=None):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """
        sparse_format = 'coo'
        transpose = inverted
        nprobe = nprobe or self.nprobe
        nprobe = nprobe if query else 1

        encode_args = (
            self.m,
            self.keep,
            self._centroids,
            self.sq_factor,
            self.rectify_negatives,
            self.l2_normalize,
            nprobe,
            transpose,
            sparse_format,
        )

        if self.parallel:
            func = delayed(_ivf_topk_sq_encode)
            batch_size = int(math.ceil(len(x) / cpu_count()))
            jobs = (func(x[i:i+batch_size], *encode_args) for i in range(0, len(x), batch_size))
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(jobs)
            results = sparse.hstack(results) if inverted else sparse.vstack(results)
            return results
        
        # non-parallel version
        return _ivf_topk_sq_encode(x, *encode_args)

    def train(
        self,
        x,
        max_samples_per_centroid=256,
        kmeans_kws={},
    ):
        """ Learn parameters from data.
        Args:
            x (ndarray): a (N,D)-shaped matrix of training vectors.
        """

        if self.l2_normalize:
            x = normalize(x)

        # run k-means for coarse quantization
        nx = len(x)
        xt = x
        max_samples = max_samples_per_centroid * self.c
        if nx > max_samples:  # subsample train set
            logging.info(f'subsampling {max_samples} / {nx} for coarse centroid training.')
            subset = np.random.choice(nx, size=max_samples, replace=False)
            xt = x[subset]

        # compute coarse centroids
        l1_kmeans = MiniBatchKMeans(
            n_clusters=self.c,
            batch_size=256*cpu_count(),
            compute_labels=False,
            n_init='auto',
            **kmeans_kws
        ).fit(xt)

        self._centroids = l1_kmeans.cluster_centers_

