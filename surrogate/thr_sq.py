import math

import numpy as np
from joblib import cpu_count, delayed, Parallel
from scipy import sparse
from scipy.stats import ortho_group
from sklearn.preprocessing import normalize

from .str_index import SurrogateTextIndex


def _thr_sq_encode(
    x,                  # features to encode
    threshold,          # thresholding factor
    sq_factor,          # quantization factor
    rectify_negatives,  # whether to apply crelu
    l2_normalize,       # whether to l2-normalize vectors
    rotation_matrix,    # rotation matrix used to rotate features
    mean,               # mean vector computed on the whole dataset
    transpose,          # if True, transpose result (returns VxN)
    format,             # sparse format of result ('csr', 'csc', 'coo', etc.)
):
    if l2_normalize:
        x = normalize(x)

    if rotation_matrix is not None:
        x = x.dot(rotation_matrix.T)

    if mean is not None:
        x = x - mean

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
        subtract_mean=False,
        rotation_matrix=None,
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
            subtract_mean (bool): whether to subtract the mean from the dataset features
            rotation_matrix (ndarray or int): if ndarray: a (D,D)-shaped rotation matrix
                                              used to rotate dataset and query features
                                              to balance dimensions; if int: the random
                                              state used to automatically generate a random
                                              rotation matrix. Defaults to None.
        """

        self.d = d
        self.threshold_percentile = threshold_percentile
        self.sq_factor = sq_factor
        self.rectify_negatives = rectify_negatives
        self.l2_normalize = l2_normalize
        self.subtract_mean = subtract_mean
        self.rotation_matrix = rotation_matrix
        if isinstance(rotation_matrix, int):
            self.rotation_matrix = ortho_group.rvs(d, random_state=rotation_matrix)

        self.threshold = None
        self.mean = None

        vocab_size = 2 * d if self.rectify_negatives else d
        super().__init__(vocab_size, parallel, is_trained=False)
    
    def add_subparser(subparsers, **kws):
        parser = subparsers.add_parser('thr-sq', help='Threshold Scalar Quantization', **kws)
        parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')
        parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
        parser.add_argument('-r', '--rotation-matrix', type=int, default=None, help='seed for generating a random orthogonal matrix to apply to vectors; if omitted, no transformation is applied.')
        parser.add_argument('-m', '--subtract-mean', action='store_true', default=False, help='Compute and subtract mean database vector.')
        parser.add_argument('-Q', '--threshold-percentile', type=int, default=75, help='Controls how many values are discarded when encoding. Must be between 1 and 99 inclusive.')
        parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
        parser.set_defaults(
            train_params=('l2_normalize', 'rectify_negatives', 'rotation_matrix', 'subtract_mean', 'threshold_percentile'),
            build_params=('sq_factor',),
            query_params=()
        )

    def encode(self, x, inverted=True, query=False, **kwargs):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """

        sparse_format = 'coo'
        transpose = inverted
        mean = self.mean if not query else None

        # in case of the query encoding, we move the threshold up of the mean of the mean vector
        threshold = self.threshold + mean.mean() if mean is not None else self.threshold

        encode_args = (
            threshold,
            self.sq_factor,
            self.rectify_negatives,
            self.l2_normalize,
            self.rotation_matrix,
            mean,
            transpose,
            sparse_format,
        )

        if self.parallel:
            func = delayed(_thr_sq_encode)
            batch_size = int(math.ceil(len(x) / cpu_count()))
            jobs = (func(x[i:i+batch_size], *encode_args) for i in range(0, len(x), batch_size))
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(jobs)
            results = sparse.hstack(results) if inverted else sparse.vstack(results)
            return results

        # non-parallel version
        sparse_repr = _thr_sq_encode(x, *encode_args)
        return sparse_repr

    def train(self, x):
        """ Learn parameters from data.
        Args:
            x (ndarray): a (N,D)-shaped matrix of training vectors.
        """
        if self.l2_normalize:
            x = normalize(x)

        if self.rotation_matrix is not None:
            x = x.dot(self.rotation_matrix.T)

        if self.subtract_mean:
            self.mean = x.mean(axis=0)
            x = x - self.mean

        if self.rectify_negatives:
            x = np.fabs(x)

        self.threshold = np.percentile(x, self.threshold_percentile)
