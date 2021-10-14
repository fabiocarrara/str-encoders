import math
import numpy as np
from joblib import cpu_count, delayed, Parallel
from scipy.sparse import csr_matrix, vstack

from surrogate.str_index import SurrogateTextIndex


def _deep_perm_encode(
    x,                  # featues to encode
    permutation_length, # length of permutation prefix
    rectify_negatives   # whether to apply crelu
):
    if rectify_negatives:
        x = np.hstack([np.maximum(x, 0), - np.minimum(x, 0)])

    n, d = x.shape

    k = d if permutation_length is None else permutation_length
    if k == d:
        topk = x.argsort(axis=1)[:, ::-1]
    else:
        unsorted_topk = x.argpartition(-k)[:, -k:]
        sorted_topk_idx = np.take_along_axis(x, unsorted_topk, axis=1).argsort(axis=1)[:, ::-1]
        topk = np.take_along_axis(unsorted_topk, sorted_topk_idx, axis=1)  # n x k

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
        permutation_length=None,
        rectify_negatives=True,
        parallel=True
    ):
        """ Constructor
        Args:
            d (int): the number of dimensions of the vectors to be encoded
            permutation_length (int): the length of the permutation prefix;
                                      if None, the whole permutation is used.
                                      Defaults to None.
            rectify_negatives (bool): whether to reserve d additional dimensions
                                      to encode negative values separately 
                                      (a.k.a. apply CReLU transform).
                                      Defaults to True.
        """

        self.d = d
        self.permutation_length = permutation_length
        self.rectify_negatives = rectify_negatives
        self.parallel = parallel

        vocab_size = 2 * d if self.rectify_negatives else d
        super().__init__(vocab_size)

    def encode(self, x):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """
        if self.parallel:
            return self.encode_parallel(x)
        
        # non-parallel version
        rows, cols, data, n = _deep_perm_encode(x, self.permutation_length, self.rectify_negatives)
        sparse_repr = csr_matrix((data, (rows, cols)), shape=(n, self.vocab_size))
        return sparse_repr
    
    def encode_parallel(self, x):

        batch_size = int(math.ceil(len(x) / cpu_count()))
        results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(
            delayed(_deep_perm_encode)(x[i:i+batch_size], self.permutation_length, self.rectify_negatives)
            for i in range(0, len(x), batch_size)
        )

        results = vstack([
            csr_matrix((data, (rows, cols)), shape=(n, self.vocab_size))
            for rows, cols, data, n in results
        ])

        return results
    
    def train(self, x):
        """ Learn parameters from data.
        Args:
            x (ndarray): a (N,D)-shaped matrix of training vectors.
        """
        pass  # there are no params to learn
        