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


def _ivf_deep_perm_encode(
    x,                  # featues to encode
    centroids,          # l1 quantizer centroids
    permutation_length, # length of permutation prefix
    rectify_negatives,  # whether to apply crelu
    l2_normalize,       # whether to l2-normalize vectors
    nprobe,             # how many coarse centroids to consider
    transpose,          # if True, transpose result (returns VxN)
    format,             # sparse format of result ('csr', 'csc', 'coo', etc.)
):
    n, d = x.shape
    c = len(centroids)
    nprobe = min(nprobe, c)
    permutation_length = min(permutation_length, d)

    if l2_normalize:
        x = normalize(x)
    
    l1_centroid_distances = cdist(x, centroids, metric='sqeuclidean')
    coarse_codes = util.bottomk_sorted(l1_centroid_distances, nprobe, axis=1)  # n x nprobe

    mult = 2 if rectify_negatives else 1
    xx = np.fabs(x) if rectify_negatives else x

    k = d if permutation_length is None else permutation_length

    cols = util.topk_sorted(xx, k, axis=1)  # n x k
    rows = np.arange(n).reshape(n, 1)  # n x 1

    is_positive = x[rows, cols] >= 0  # n x k
    
    if rectify_negatives:
        cols += np.where(is_positive, 0, d)  # shift indices of negatives after positives
    
    cols = np.stack([cols + coarse_codes[:, [i]] * mult * d for i in range(nprobe)], axis=-1)  # n x k x nprobe
    rows = np.expand_dims(rows, axis=-1)  # n x 1 x 1
    data = np.arange(k, 0, -1).reshape(1, -1, 1)  # 1 x k x 1

    rows, cols, data = np.broadcast_arrays(rows, cols, data)  # n x k x nprobe

    rows = rows.flatten()
    cols = cols.flatten()
    data = data.flatten()

    shape = (n, c * mult * d)

    if transpose:
        rows, cols = cols, rows
        shape = shape[::-1]

    spclass = getattr(sparse, f'{format}_matrix')
    return spclass((data, (rows, cols)), shape=shape)


class IVFDeepPermutation(SurrogateTextIndex):
    
    def __init__(
        self,
        d,
        n_coarse_centroids=512,
        permutation_length=32,
        rectify_negatives=True,
        l2_normalize=False,
        parallel=True,
    ):
        """ Constructor
        Args:
            d (int): the number of dimensions of the vectors to be encoded
            n_coarse_centroids (int): the number of coarse centroids of the level 1
                                      quantizer (voronoi partitioning).
            rectify_negatives (bool): whether to reserve d additional dimensions
                                      to encode negative values separately 
                                      (a.k.a. apply CReLU transform).
                                      Defaults to True.
            l2_normalize (bool): whether to apply l2-normalization before processing vectors;
                                 set this to False if vectors are already normalized.
                                 Defaults to True.
        Attributes:
            permutation_length (int): the length of the permutation prefix;
                                      if None, the whole permutation is used.
                                      Defaults to None.
        """

        self.d = d
        self.c = n_coarse_centroids
        self.permutation_length = permutation_length
        self.rectify_negatives = rectify_negatives
        self.l2_normalize = l2_normalize
        self.nprobe = 1

        self._centroids = None

        vocab_size = self.c * 2 * d if self.rectify_negatives else self.c * d
        super().__init__(vocab_size, parallel)
    
    def add_subparser(subparsers, **kws):
        parser = subparsers.add_parser('ivf-deep-perm', help='Chunked Deep Permutation', **kws)
        parser.add_argument('-c', '--n-coarse-centroids', type=int, default=512, help='no of coarse centroids')
        parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')
        parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
        parser.add_argument('-L', '--permutation-length', type=int, default=32, help='length of the permutation prefix (None for full permutation)')
        parser.add_argument('-p', '--nprobe', type=int, default=1, help='how many partitions to visit at query time')
        parser.set_defaults(
            train_params=('n_coarse_centroids', 'l2_normalize'),
            build_params=('rectify_negatives', 'permutation_length'),
            query_params=('nprobe',)
        )

    def encode(self, x, inverted=True, query=False, **kwargs):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """
        sparse_format = 'coo'
        transpose = inverted

        if self.parallel:
            batch_size = int(math.ceil(len(x) / cpu_count()))
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(
                delayed(_ivf_deep_perm_encode)(
                    x[i:i+batch_size],
                    self._centroids,
                    self.permutation_length,
                    self.rectify_negatives,
                    self.l2_normalize,
                    self.nprobe,
                    transpose,
                    sparse_format,
                ) for i in range(0, len(x), batch_size)
            )

            results = sparse.hstack(results) if inverted else sparse.vstack(results)
            return results
        
        # non-parallel version
        return _ivf_deep_perm_encode(
            x,
            self._centroids,
            self.permutation_length,
            self.rectify_negatives,
            self.l2_normalize,
            self.nprobe if query else 1,
            transpose,
            sparse_format,
        )
    
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
        