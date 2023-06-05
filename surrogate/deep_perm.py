import math

import numpy as np
from joblib import cpu_count, delayed, Parallel
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans

from . import util
from .str_index import SurrogateTextIndex


def _deep_perm_encode(
    x,                  # featues to encode
    permutation_length, # length of permutation prefix
    rectify_negatives   # whether to apply crelu
):
    if rectify_negatives:
        x = np.hstack([np.maximum(x, 0), - np.minimum(x, 0)])

    n, d = x.shape

    k = d if permutation_length is None else permutation_length
    topk = util.topk_sorted(x, k, axis=1)  # n x k

    rows = np.arange(n).reshape(-1, 1)
    cols = topk
    data = np.arange(k, 0, -1).reshape(1, -1)
    
    rows, cols, data = np.broadcast_arrays(rows, cols, data)

    rows = rows.flatten()
    cols = cols.flatten()
    data = data.flatten()

    return rows, cols, data, n


class DeepPermutation(SurrogateTextIndex):
    
    def __init__(
        self,
        d,
        use_centroids=False,
        rectify_negatives=True,
        parallel=True
    ):
        """ Constructor
        Args:
            d (int): the number of dimensions of the vectors to be encoded
            use_centroids (bool): if True, computes centroids found with kmeans
                                  and uses them as pivots to compute permutations;
                                  if False, treat the vectors as precomputed
                                  distances to pivots.
            rectify_negatives (bool): whether to reserve d additional dimensions
                                      to encode negative values separately 
                                      (a.k.a. apply CReLU transform).
                                      Defaults to True.
        Attributes:
            permutation_length (int): the length of the permutation prefix;
                                      if None, the whole permutation is used.
                                      Defaults to None.
        """

        self.d = d
        self.use_centroids = use_centroids
        self.permutation_length = None
        self.rectify_negatives = rectify_negatives

        self._kmeans = None

        vocab_size = 2 * d if self.rectify_negatives else d
        super().__init__(vocab_size, parallel)
    
    def add_subparser(subparsers, **kws):
        parser = subparsers.add_parser('deep-perm', help='Deep Permutation', **kws)
        parser.add_argument('-c', '--use-centroids', action='store_true', default=False, help='Find Pivots with k-Means.')
        parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
        parser.add_argument('-L', '--permutation-length', type=int, default=None, help='length of the permutation prefix (None for full permutation)')
        parser.set_defaults(
            train_params=('use_centroids',),
            build_params=('rectify_negatives', 'permutation_length'),
            query_params=()
        )

    def encode(self, x, inverted=True, **kwargs):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """
        if self.use_centroids:
            x = - self._kmeans.transform(x)

        if self.parallel:
            batch_size = int(math.ceil(len(x) / cpu_count()))
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(
                delayed(_deep_perm_encode)(x[i:i+batch_size], self.permutation_length, self.rectify_negatives)
                for i in range(0, len(x), batch_size)
            )

            # use COO for fast sparse matrix concatenation
            results = [sparse.coo_matrix((data, (rows, cols)), shape=(n, self.vocab_size)) for rows, cols, data, n in results]
            results = sparse.vstack(results)
            results = results.T if inverted else results
            # then convert to CSR
            # results = results.tocsr()
            return results
        
        # non-parallel version
        rows, cols, data, n = _deep_perm_encode(x, self.permutation_length, self.rectify_negatives)
        if inverted:
            rows, cols = cols, rows
            shape = (self.vocab_size, n)
        else:
            shape = (n, self.vocab_size)

        return sparse.coo_matrix((data, (rows, cols)), shape=shape)
    
    def train(self, x):
        """ Learn parameters from data.
        Args:
            x (ndarray): a (N,D)-shaped matrix of training vectors.
        """
        if self.use_centroids:
            self._kmeans = MiniBatchKMeans(n_clusters=self.d, batch_size=256*cpu_count(), compute_labels=False).fit(x)
        