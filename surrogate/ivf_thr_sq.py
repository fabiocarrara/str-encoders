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


def _ivf_thr_sq_encode(
    x,                  # featues to encode
    m,                  # number of subvectors
    centroids,          # l1 quantizer centroids
    thresholds,         # sparsification thresholds
    sq_factor,          # quantization factor
    rectify_negatives,  # whether to apply crelu
    l2_normalize,       # whether to l2-normalize vectors
    nprobe,             # how many coarse centroids to consider
    transpose,          # if True, transpose result (returns VxN)
    format,             # sparse format of result ('csr', 'csc', 'coo', etc.)
    query,              # whether to apply query encoding (no residual, returns correction scores)
):
    n, d = x.shape
    c = len(centroids)
    nprobe = min(nprobe, c)

    if l2_normalize:
        x = normalize(x)
    
    l1_centroid_distances = cdist(x, centroids, metric='sqeuclidean')
    coarse_codes = util.bottomk_sorted(l1_centroid_distances, nprobe, axis=1)  # n x nprobe

    if query:  # compute norms for later
        x_norm = np.linalg.norm(x, axis=1) if not l2_normalize else np.ones(n)
        c_norm = np.linalg.norm(centroids, axis=1)
    
    # repeat x for nprobe times (1st nprobe times, then 2nd nprobe times, etc.; enables batching)
    x = np.broadcast_to(x.reshape(n, 1, d), (n, nprobe, d)).reshape(n * nprobe, d)  # n * nprobe, d
    coarse_codes = coarse_codes.flatten()  # n * nprobe
    n *= nprobe

    # compute correction scores (will be used later when merging results)
    if query:
        # we exploit |A - B|_2^2 = |A|^2 + |B|^2 - 2A·B
        # ==> A·B = (|A|^2 + |B|^2 - |A - B|_2^2) / 2
        x_idx = np.arange(n) // nprobe
        xc = (x_norm[x_idx] + c_norm[coarse_codes] - l1_centroid_distances[x_idx, coarse_codes]) / 2
        correction_scores = (sq_factor * sq_factor) * xc

    # compute residuals
    if not query:
        x = x - centroids[coarse_codes]  # n x d

    # x = np.split(x, m, axis=1)  # m x n x d/m
    x = x.reshape(n, m, d // m)  # n x m x d/m
    thresholds = thresholds.reshape(1, m, 1)

    if rectify_negatives:
        keep_pos = (x > thresholds).reshape(n, d)
        keep_neg = (-x > thresholds).reshape(n, d)

        x = x.reshape(n, d)

        idx_pos = np.nonzero(keep_pos)
        idx_neg = np.nonzero(keep_neg)

        data_pos = x[idx_pos]
        data_neg = -x[idx_neg]

        rows_pos, cols_pos = idx_pos
        rows_neg, cols_neg = idx_neg

        cols_pos += coarse_codes[rows_pos] * 2 * d 
        cols_neg += coarse_codes[rows_neg] * 2 * d + d

        data = np.hstack((data_pos, data_neg))
        rows = np.hstack((rows_pos, rows_neg))
        cols = np.hstack((cols_pos, cols_neg))

        shape = (n, c * 2 * d)
    else:
        keep = (np.fabs(x) > thresholds).reshape(n, d)
        x = x.reshape(n, d)

        idx = np.nonzero(keep)
        data = x[idx]
        rows, cols = idx
        cols += coarse_codes[rows] * d

        shape = (n, c * d)

    # scalar quantization
    data = np.fix(sq_factor * data).astype('int')

    if transpose:
        rows, cols = cols, rows
        shape = shape[::-1]

    spclass = getattr(sparse, f'{format}_matrix')
    encoded = spclass((data, (rows, cols)), shape=shape)

    if query:
        return encoded, correction_scores
    
    return encoded


class IVFThresholdSQ(SurrogateTextIndex):
    
    def __init__(
        self,
        d,
        n_coarse_centroids=None,
        n_subvectors=None,
        threshold_percentile=75,
        sq_factor=1000,
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
        self.c = n_coarse_centroids
        self.m = n_subvectors
        self.threshold_percentile = threshold_percentile
        self.sq_factor = sq_factor
        self.rectify_negatives = rectify_negatives
        self.l2_normalize = l2_normalize
        self.nprobe = 1

        self._centroids = None
        self._thresholds = None

        vocab_size = self.c * 2 * d if self.rectify_negatives else self.c * d
        super().__init__(vocab_size, parallel)

    def encode(self, x, inverted=True, query=False):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """
        sparse_format = 'coo'
        transpose = inverted
        thresholds = self._thresholds if query else self._residual_thresholds

        if self.parallel:
            batch_size = int(math.ceil(len(x) / cpu_count()))
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(
                delayed(_ivf_thr_sq_encode)(
                    x[i:i+batch_size],
                    self.m,
                    self._centroids,
                    thresholds,
                    self.sq_factor,
                    self.rectify_negatives,
                    self.l2_normalize,
                    self.nprobe,
                    transpose,
                    sparse_format,
                    query,
                ) for i in range(0, len(x), batch_size)
            )

            if query:
                results, correction_scores = zip(*results)
                correction_scores = np.hstack(correction_scores)

            results = sparse.hstack(results) if inverted else sparse.vstack(results)

            if query:
                return results, correction_scores
            
            return results
        
        # non-parallel version
        return _ivf_thr_sq_encode(
            x,
            self.m,
            self._centroids,
            thresholds,
            self.sq_factor,
            self.rectify_negatives,
            self.l2_normalize,
            self.nprobe,
            transpose,
            sparse_format,
            query,
        )
    
    def train(
        self,
        x,
        max_samples_per_centroid=256,
        kmeans_kws={}
    ):
        """ Learn parameters from data.
        Args:
            x (ndarray): a (N,D)-shaped matrix of training vectors.
        """

        if self.l2_normalize:
            x = normalize(x)

        # compute non-residual thresholds for query encoding
        x_non_residual = np.fabs(x) if self.rectify_negatives else x
        x_non_residual = np.split(x_non_residual, self.m, axis=1)  # m x n x d/m
        x_non_residual = np.reshape(x_non_residual, (self.m, -1))  # m x (n x d/m)
        self._thresholds = np.percentile(x_non_residual, self.threshold_percentile, axis=1)

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
            **kmeans_kws
        ).fit(xt)

        self._centroids = l1_kmeans.cluster_centers_

        # compute residuals
        coarse_codes = l1_kmeans.predict(x)
        residuals = x - self._centroids[coarse_codes]  # n x d

        if self.rectify_negatives:
            residuals = np.fabs(residuals)

        residuals = np.split(residuals, self.m, axis=1)  # m x n x d/m
        residuals = np.reshape(residuals, (self.m, -1))  # m x (n x d/m)
        self._residual_thresholds = np.percentile(residuals, self.threshold_percentile, axis=1)

    def search(self, q, k, *args, **kwargs):
        q_enc, correction_scores = self.encode(q, inverted=False, query=True)
        sorted_scores, indices = self.search_encoded(q_enc, k, *args, **kwargs)
        sorted_scores += correction_scores[:, None]

        if self.nprobe > 1:  # merge results
            nq = len(q)

            sorted_scores = sorted_scores.reshape(nq, -1)
            indices = indices.reshape(nq, -1)

            idx = util.topk_sorted(sorted_scores, k, axis=1)
            sorted_scores = np.take_along_axis(sorted_scores, idx, axis=1)
            indices = np.take_along_axis(indices, idx, axis=1)

        return sorted_scores, indices
