import logging
import math
from pathlib import Path

import numpy as np
from joblib import cpu_count, delayed, Parallel
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

from . import util
from .str_index import SurrogateTextIndex
from .topk_sq import _fast_random_semiorth


def _ivf_topk_sq_encode(
    x,                  # featues to encode
    # IVF parameters
    centroids,          # centroids
    nprobe,             # how many coarse centroids to consider
    # SQ parameters
    keep,               # the number or fraction of high-value components to keep
    shift_value,        # if not None, apply Negated Concatenation and add this value
    sq_factor,          # quantization factor
    rectify_negatives,  # whether to apply crelu
    l2_normalize,       # whether to l2-normalize vectors
    ortho_matrix,       # (semi-)orthogonal matrix used to shuffle feature information
    transpose,          # if True, transpose result (returns VxN)
    format,             # sparse format of result ('csr', 'csc', 'coo', etc.)
):
    n, d = x.shape
    c = len(centroids)
    nprobe = min(nprobe, c)
    k = int(keep * d) if isinstance(keep, float) else keep

    if l2_normalize:
        x = normalize(x)
    
    centroid_distances = cdist(x, centroids, metric='sqeuclidean')
    coarse_codes = util.bottomk_sorted(centroid_distances, nprobe, axis=1)  # n x nprobe

    if ortho_matrix is not None:
        x = x.dot(ortho_matrix)
        d = x.shape[1]

    # dsub = d // m
    # x = x.reshape(n, m, dsub)  # n x m x d/m

    apply_ncs = shift_value is not None  # apply negated concatenation and shift
    assert not (rectify_negatives and apply_ncs), "Cannot apply both CReLU and NCS"

    mult = 2 if rectify_negatives or apply_ncs else 1
    xx = np.fabs(x) if rectify_negatives or apply_ncs else x

    rows = np.arange(n).reshape(n, 1)  # n x 1
    cols = util.topk_sorted(xx, k, axis=1)  # n x k

    if rectify_negatives:
        data = xx[rows, cols]  # n x k
        is_positive = x[rows, cols] > 0  # n x k
        cols += np.where(is_positive, 0, d)  # shift indices of negatives after positives

    elif apply_ncs:
        data = x[rows, cols]  # n x k
        cols = np.hstack((cols, cols + d))  # n x 2*k
        data = np.hstack((data, -data)) + shift_value  # n x 2*k
    
    else:
        data = xx[rows, cols]
    
    cols = np.stack([cols + coarse_codes[:, [i]] * mult * d for i in range(nprobe)], axis=-1)  # n x k x nprobe
    rows = np.expand_dims(rows, axis=-1)  # n x 1 x 1
    data = np.expand_dims(data, axis=-1)  # n x (2*)k x 1

    rows, cols, data = np.broadcast_arrays(rows, cols, data)  # n x (2*)k x nprobe

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
        n_coarse_centroids=1,
        centroids_cache_dir=None,
        keep=0.75,
        shift_value=None,
        sq_factor=1e5,
        rectify_negatives=True,
        l2_normalize=True,
        dim_multiplier=None,
        seed=42,
        parallel=True,
    ):
        """ Constructor
        Args:
            d (int): the number of dimensions of the vectors to be encoded.
            n_coarse_centroids (int): the number of coarse centroids of the level 1
                                      quantizer (voronoi partitioning).
            centroids_cache_dir (Path): the directory where to cache the centroids. If None,
                                        no caching is performed. Defaults to None.
            keep (int or float): if int, number of components to keep;
                                 if float, the fraction of components to keep (must
                                 be between 0.0 and dim_multiplier). Defaults to 0.25.
            shift_value (float): if not None, applies negated concatenation and adds
                                 this value to components to make them positive.
                                 Defaults to None.
            sq_factor (float): multiplicative factor controlling scalar quantization.
                               Defaults to 10000.
            rectify_negatives (bool): whether to reserve d additional dimensions
                                      to encode negative values separately 
                                      (a.k.a. apply CReLU transform).
                                      Defaults to True.
            l2_normalize (bool): whether to apply l2-normalization before processing vectors;
                                 set this to False if vectors are already normalized.
                                 Defaults to True.
            dim_multiplier (float):  apply a random (semi-)orthogonal matrix to the input to expand
                                     the number of dimensions by this factor; if 0, no transformation is applied.
                                     Defaults to None.
            seed (int): the random state used to automatically generate the random matrix and the centroids.
        """

        self.d = d
        self.c = n_coarse_centroids
        self.centroids_cache_dir = centroids_cache_dir
        self.nprobe = 1
        self.keep = keep
        self.shift_value = shift_value
        self.sq_factor = sq_factor
        self.rectify_negatives = rectify_negatives
        self.l2_normalize = l2_normalize
        self.dim_multiplier = dim_multiplier
        self.seed = seed

        self._centroids = None
        self._R = None
        if self.dim_multiplier:
            assert self.dim_multiplier >= 1, "dim_multiplier must be >= 1.0"
            new_d = int(dim_multiplier * d)
            self._R = _fast_random_semiorth(new_d, d, seed=self.seed)
            d = new_d

        apply_ncs = shift_value is not None  # whether to apply negated concatenation and shift
        reserve_additional_dims = self.rectify_negatives or apply_ncs
        vocab_size = 2 * d if reserve_additional_dims else d
        vocab_size *= self.c

        discount = 0
        if apply_ncs:
            discount = np.fix((shift_value * sq_factor) ** 2).astype('int')
        super().__init__(vocab_size, parallel, discount=discount, is_trained=False)
    
    def add_subparser(subparsers, **kws):
        parser = subparsers.add_parser('ivf-topk-sq', help='Chunked TopK Scalar Quantization', **kws)
        parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')
        parser.add_argument('-c', '--n-coarse-centroids', type=int, default=512, help='no of coarse centroids')
        parser.add_argument('-a', '--centroids-cache-dir', type=Path, default=None, help='directory where to cache the centroids')
        parser.add_argument('-p', '--nprobe', type=int, default=1, help='how many partitions to visit at query time')
        parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
        parser.add_argument('-t', '--shift-value', type=float, default=None, help='Constant added to component values to make them positive.')
        parser.add_argument('-d', '--dim-multiplier', type=float, default=1.0, help='Expand input dimensionality by this factor applying a random semi-orthogonal transformation; if 0, no transformation is applied.')
        parser.add_argument('-r', '--seed', type=int, default=42, help='seed for generating a random orthogonal matrix to apply to vectors.')
        parser.add_argument('-k', '--keep', type=float, default=0.25, help='Controls how many values are discarded when encoding. Must be between 0.0 and 1.0 inclusive.')
        parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
        parser.set_defaults(
            train_params=('l2_normalize', 'n_coarse_centroids', 'centroids_cache_dir', 'rectify_negatives', 'shift_value', 'dim_multiplier', 'seed'),
            build_params=('sq_factor', 'keep'),
            query_params=('nprobe',),
            ignore_params=('centroids_cache_dir',)
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
            self._centroids,
            nprobe,
            self.keep,
            self.shift_value,
            self.sq_factor,
            self.rectify_negatives,
            self.l2_normalize,
            self._R,
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

        centroid_cache = None
        if self.centroids_cache_dir:
            centroid_cache = self.centroids_cache_dir / f'centroids_n{self.c}_seed{self.seed}.npy'
            if centroid_cache.exists():
                logging.info(f'Loading centroids from {centroid_cache}')
                self._centroids = np.load(centroid_cache)
                self.is_trained = True
                return

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
            random_state=self.seed,
            **kmeans_kws
        ).fit(xt)

        self._centroids = l1_kmeans.cluster_centers_
        self.is_trained = True

        if centroid_cache:
            logging.info(f'Saving centroids to {centroid_cache}')
            centroid_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(centroid_cache, self._centroids)

