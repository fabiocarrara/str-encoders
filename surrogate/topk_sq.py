import math
import warnings

import numpy as np
from joblib import cpu_count, delayed, Parallel
from scipy import sparse
from sklearn.preprocessing import normalize

from . import util
from .str_index import SurrogateTextIndex


def _topk_sq_encode(
    x,                  # featues to encode
    keep,               # the number or fraction of high-value components to keep
    sq_factor,          # quantization factor
    rectify_negatives,  # whether to apply CReLU transformation
    l2_normalize,       # whether to l2-normalize vectors
    ortho_matrix,       # (semi-)orthogonal matrix used to shuffle feature information
    transpose,          # if True, transpose result (returns VxN)
    format,             # sparse format of result ('csr', 'csc', 'coo', etc.)
):
    n, d = x.shape
    k = int(keep * d) if isinstance(keep, float) else keep

    if l2_normalize:
        x = normalize(x)

    if ortho_matrix is not None:
        x = x.dot(ortho_matrix.T)
        d = x.shape[1]

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


def _fast_random_semiorth(m, n, seed=7):
    """ See Eq (1) of https://doi.org/10.21437/Interspeech.2018-1417 """
    rng = np.random.default_rng(seed)
    M = rng.normal(loc=0, scale=1 / np.sqrt(m), size=(m, n)).T
    I = np.eye(n)

    for i in range(100):
        M = M - 0.5 * (M.dot(M.T) - I).dot(M)
        iI = M.dot(M.T)
        if np.allclose(iI, I):
            return M

    warnings.warn("Semi-orthogonal matrix not converged after maximum iterations.")
    return M


class TopKSQ(SurrogateTextIndex):

    def __init__(
        self,
        d,
        keep=0.25,
        sq_factor=1000,
        rectify_negatives=True,
        l2_normalize=True,
        dim_multiplier=None,
        seed=42,
        parallel=True
    ):
        """ Constructor
        Args:
            d (int): the number of dimensions of the vectors to be encoded.
            keep (int or float): if int, number of components to keep;
                                 if float, the fraction of components to keep (must
                                 be between 0.0 and dim_multiplier). Defaults to 0.25.
            sq_factor (float): multiplicative factor controlling scalar quantization.
                               Defaults to 1000.
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
            seed (int): the random state used to automatically generate the random matrix.
        """

        self.d = d
        self.keep = keep
        self.sq_factor = sq_factor
        self.rectify_negatives = rectify_negatives
        self.l2_normalize = l2_normalize
        self.dim_multiplier = dim_multiplier
        self.seed = seed

        self._R = None
        if self.dim_multiplier:
            assert self.dim_multiplier >= 1, "dim_multiplier must be >= 1.0"
            new_d = int(dim_multiplier * d)
            self._R = _fast_random_semiorth(new_d, d, seed=self.seed).T
            d = new_d

        vocab_size = 2 * d if self.rectify_negatives else d
        super().__init__(vocab_size, parallel, is_trained=True)

    def encode(self, x, inverted=True, query=False):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """
        sparse_format = 'coo'
        transpose = inverted

        encode_args = (
            self.keep,
            self.sq_factor,
            self.rectify_negatives,
            self.l2_normalize,
            self._R,
            transpose,
            sparse_format,
        )

        if self.parallel:
            func = delayed(_topk_sq_encode)
            batch_size = int(math.ceil(len(x) / cpu_count()))
            jobs = (func(x[i:i+batch_size], *encode_args) for i in range(0, len(x), batch_size))
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(jobs)
            results = sparse.hstack(results) if inverted else sparse.vstack(results)
            return results

        # non-parallel version
        return _topk_sq_encode(x, *encode_args)

    def train(self, x):
        pass  # no train needed
