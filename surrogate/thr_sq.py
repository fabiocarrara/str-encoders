import math
import numpy as np
from joblib import cpu_count, delayed, Parallel
from scipy import sparse
from sklearn.preprocessing import normalize

from surrogate.str_index import SurrogateTextIndex


def _thr_sq_encode(
    x,                  # featues to encode
    threshold,          # thresholding factor
    sq_factor,          # quantization factor
    rectify_negatives,  # whether to apply crelu
    l2_normalize,       # whether to l2-normalize vectors
    transpose,          # if True, transpose result (returns VxN)
    format,             # sparse format of result ('csr', 'csc', 'coo', etc.)
):
    if l2_normalize:
        x = normalize(x)

    if rectify_negatives:
        x = np.hstack([np.maximum(x, 0), - np.minimum(x, 0)])
        keep = x > threshold  # assuming positive values only
    else:
        keep = np.fabs(x) > threshold

    # thresholding
    x = np.where(keep, x, 0)

    # scalar quantization
    x = np.fix(sq_factor * x).astype('int')

    if transpose:
        x = x.T

    spclass = getattr(sparse, f'{format}_matrix')
    return spclass(x)


class ThresholdSQ(SurrogateTextIndex):
    
    def __init__(
        self,
        d,
        threshold_percentile=75,
        sq_factor=1000,
        rectify_negatives=True,
        l2_normalize=True,
        parallel=True
    ):
        """ Constructor
        Args:
            d (int): the number of dimensions of the vectors to be encoded.
            threshold_percentile (int): percentile of train values to determine
                                        the threshold; if x, the (100 - x) top values are kept.
                                        Must be an integer between 1 and 99 (inclusive).
                                        Defaults to 75.
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
        self.threshold_percentile = threshold_percentile
        self.sq_factor = sq_factor
        self.rectify_negatives = rectify_negatives
        self.l2_normalize = l2_normalize

        self.threshold = None

        vocab_size = 2 * d if self.rectify_negatives else d
        super().__init__(vocab_size, parallel)

    def encode(self, x, inverted=True):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """

        transpose = inverted

        if self.parallel:
            batch_size = int(math.ceil(len(x) / cpu_count()))
            sparse_format = 'coo' if inverted else 'csr'
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(
                delayed(_thr_sq_encode)(
                    x[i:i+batch_size],
                    self.threshold,
                    self.sq_factor,
                    self.rectify_negatives,
                    self.l2_normalize,
                    transpose,
                    sparse_format)
                for i in range(0, len(x), batch_size)
            )

            if inverted:  # hstack COO matrices and then convert to CSR
                return sparse.hstack(results).tocsr()
            else:  # vstack CSR matrices
                return sparse.vstack(results)
        
        # non-parallel version
        sparse_format = 'csr'
        sparse_repr = _thr_sq_encode(x, self.threshold, self.sq_factor, self.rectify_negatives, self.l2_normalize, transpose, sparse_format)
        return sparse_repr
    
    def train(self, x):
        """ Learn parameters from data.
        Args:
            x (ndarray): a (N,D)-shaped matrix of training vectors.
        """
        if self.l2_normalize:
            x = normalize(x)

        if self.rectify_negatives:
            x = np.fabs(x)

        self.threshold = np.percentile(x, self.threshold_percentile)
