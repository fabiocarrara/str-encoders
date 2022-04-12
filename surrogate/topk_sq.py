import math

import numpy as np
from joblib import cpu_count, delayed, Parallel
from scipy import sparse
from sklearn.preprocessing import normalize

from . import util
from .str_index import SurrogateTextIndex


def _topk_sq_encode(
    x,                  # featues to encode
    k,                  # the number or fraction of high-value components to keep
    sq_factor,          # quantization factor
    rectify_negatives,  # whether to apply crelu
    l2_normalize,       # whether to l2-normalize vectors
    transpose,          # if True, transpose result (returns VxN)
    format,             # sparse format of result ('csr', 'csc', 'coo', etc.)
):
    n, d = x.shape
    k = int(k * d) if isinstance(k, float) else k

    if l2_normalize:
        x = normalize(x)
    
    mult = 2 if rectify_negatives else 1
    xx = np.fabs(x) if rectify_negatives else x

    rows = np.arange(n).reshape(n, 1)  # n x 1
    cols = util.topk_sorted(xx, k, axis=1)  # n x k
    data = xx[rows, cols]  # n x k
    
    if rectify_negatives:
        is_positive = x[rows, cols] > 0  # n x k
        cols += np.where(is_positive, 0, d)  # shift indices of negatives after positives
    
    rows, cols, data = np.broadcast_arrays(rows, cols, data)  # n x (m*k) x nprobe

    rows = rows.flatten()
    cols = cols.flatten()
    data = data.flatten()

    shape = (n, mult * d)

    # scalar quantization
    data = np.fix(sq_factor * data.astype(np.float64)).astype('int')

    if transpose:
        rows, cols = cols, rows
        shape = shape[::-1]

    spclass = getattr(sparse, f'{format}_matrix')
    return spclass((data, (rows, cols)), shape=shape)


class TopKSQ(SurrogateTextIndex):
    
    def __init__(
        self,
        d,
        k=0.25,
        sq_factor=1000,
        rectify_negatives=True,
        l2_normalize=True,
        parallel=True
    ):
        """ Constructor
        Args:
            d (int): the number of dimensions of the vectors to be encoded.
            k (int or float): if int, number of components to keep (must be between 0 and d);
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
        self.k = k
        self.sq_factor = sq_factor
        self.rectify_negatives = rectify_negatives
        self.l2_normalize = l2_normalize

        vocab_size = 2 * d if self.rectify_negatives else d
        super().__init__(vocab_size, parallel)

    def encode(self, x, inverted=True, query=False):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """
        sparse_format = 'coo'
        transpose = inverted

        if self.parallel:
            batch_size = int(math.ceil(len(x) / cpu_count()))
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(
                delayed(_topk_sq_encode)(
                    x[i:i+batch_size],
                    self.k,
                    self.sq_factor,
                    self.rectify_negatives,
                    self.l2_normalize,
                    transpose,
                    sparse_format,
                ) for i in range(0, len(x), batch_size)
            )

            results = sparse.hstack(results) if inverted else sparse.vstack(results)
            return results
        
        # non-parallel version
        return _topk_sq_encode(
            x,
            self.k,
            self.sq_factor,
            self.rectify_negatives,
            self.l2_normalize,
            transpose,
            sparse_format,
        )

    def train(self, x):
        pass  # no train needed
