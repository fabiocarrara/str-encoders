import math

import faiss
from joblib import cpu_count, delayed, Parallel
import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.distance import cdist

from surrogate.str_index import SurrogateTextIndex
# from utils import parallel_cdist as cdist

def _spqr_encode(
    x,             # features to encode
    l1_centroids,  # coarse centroids
    l2_centroids,  # fine centroids
    k,             # permutation prefix length
):
    n = len(x)
    m, f, _ = l2_centroids.shape

    # find nearest coarse centroids
    coarse_codes = cdist(x, l1_centroids, metric='sqeuclidean').argmin(axis=1) # n 

    # compute residual
    residuals = x - l1_centroids[coarse_codes]  # n x d
    residuals = np.split(residuals, m, axis=1)  # m x n x d/m

    # compute distances to fine centroids for each subvector
    fine_distances = np.array([ cdist(r, c, metric='sqeuclidean')
            for r, c in zip(residuals, l2_centroids) ])  # m x n x f

    # find top-k NNs for each subvector:
    # this sorts the entire array (inefficient), but needed if k == f ...
    if k == f:
        topk = fine_distances.argsort(axis=2)[:, :, :k]  # m x n x k
    else:
    # ... while this partitions the array and sorts only the topk
        unsorted_topk = fine_distances.argpartition(k, axis=2)[:, :, :k]
        sorted_topk_idx = np.take_along_axis(fine_distances, unsorted_topk, axis=2).argsort(axis=2)
        topk = np.take_along_axis(unsorted_topk, sorted_topk_idx, axis=2)  # m x n x k

    # build the representation
    rows = np.arange(n).reshape(1, -1, 1)
    m_shifts = np.arange(m).reshape(-1, 1, 1)
    c_shifts = coarse_codes.reshape(1, -1, 1)
    cols = topk + f * m_shifts + f * m * c_shifts
    data = np.arange(k, 0, -1).reshape(1, 1, -1)
    
    rows, cols, data = np.broadcast_arrays(rows, cols, data)

    rows = rows.flatten()
    cols = cols.flatten()
    data = data.flatten()

    return rows, cols, data, n


class SPQR(SurrogateTextIndex):
    """ Implements Surrogate of Product Quantizator Representation (SQPR). """

    def __init__(
        self,
        d_or_index,
        n_coarse_centroids=None,
        n_subvectors=None,
        n_fine_centroids_log2=None,
        parallel=False
    ):
        """ Constructor.
        Args:
            d_or_index (int or faiss.Index): the number of dimensions of the vectors
                                             or a prebuilt faiss Index.
            n_coarse_centroids (int): the number of coarse centroids of the level 1
                                      quantizer (voronoi partitioning).
            n_subvectors (int): the number of subvectors of the level 2 quantizer
                                (product quantization).
            n_fine_centroids_log2 (int): the log2() of number of fine centroids for
                                         the level 2 quantizer (product quantization);
                                         FAISS currently support 4, 8, 12, 16 on CPU.
        """

        assert (
            isinstance(d_or_index, int) and \
            isinstance(n_coarse_centroids, int) and \
            isinstance(n_subvectors, int) and \
            isinstance(n_fine_centroids_log2, int)
        ) or (
            isinstance(d_or_index, faiss.swigfaiss.IndexIVFPQ) and
            (n_coarse_centroids is None) and \
            (n_subvectors is None) and \
            (n_fine_centroids_log2 is None)
        ), "You must specify a prebuilt faiss index " \
            "or all the params (as ints) needed to build the index."

        self.parallel = parallel
        self.l1_centroids = None
        self.l2_centroids = None
        self.db = None

        if isinstance(d_or_index, int):
            self.d = d_or_index
            self.c = n_coarse_centroids
            self.m = n_subvectors
            self.f_log2 = n_fine_centroids_log2
            self.f = 2 ** self.f_log2

            index_factory_string = f'IVF{self.c},PQ{self.m}x{self.f_log2}'
            self.faiss_index = faiss.index_factory(self.d, index_factory_string)
        else:
            self.faiss_index = d_or_index
            self.d = d_or_index.d
            self.c = d_or_index.nlist
            self.m = d_or_index.pq.M
            self.f_log2 = int(math.log2(d_or_index.pq.ksub))
            self.f = d_or_index.pq.ksub
        
        self.vocab_size = self.c * self.m * self.f
        self._load_centroids()
        self.reset()

    def _load_centroids(self):
        if self.faiss_index.is_trained:
            index = self.faiss_index
            # get the level-1 centroids by reconstructing codes 0, ..., C-1
            self.l1_centroids = index.quantizer.reconstruct_n(0, index.nlist)
            # get the level-2 centroids and reshape them
            self.l2_centroids = faiss.vector_to_array(index.pq.centroids)
            # there are ksub centroids of length dsub for each of the M subvectors
            self.l2_centroids = self.l2_centroids.reshape(index.pq.M, index.pq.ksub, index.pq.dsub)
        else:
            self.l1_centroids = None
            self.l2_centroids = None

    def train(self, x):
        self.faiss_index.train(x)
        self._load_centroids()
    
    def encode(self, *args, **kwargs):
        if self.parallel:
            return self.encode_parallel(*args, **kwargs)
        
        return self.encode_sequential(*args, **kwargs)

    def encode_sequential(self, x, permutation_length=None):
        """ Encodes vectors and returns their SPQR representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
            permutation_length (int): the number of nearest fine centroids to keep
                                      when truncating the permutation; if None, the
                                      whole permutation is used. Defaults to None.
        """
        assert self.faiss_index.is_trained, "Index must be trained before encoding."

        k = self.f if permutation_length is None else permutation_length
        n, d = x.shape
        
        assert self.d == d, f"Dimension mismatch: expected {self.d}, got {d}"

        # find nearest coarse centroids
        coarse_codes = cdist(x, self.l1_centroids, metric='sqeuclidean').argmin(axis=1) # n 

        # compute residual
        residuals = x - self.l1_centroids[coarse_codes]  # n x d
        residuals = np.split(residuals, self.m, axis=1)  # m x n x d/m

        # compute distances to fine centroids for each subvector
        fine_distances = np.array([ cdist(r, c, metric='sqeuclidean')
                for r, c in zip(residuals, self.l2_centroids) ])  # m x n x f

        # find top-k NNs for each subvector:
        # this sorts the entire array (inefficient), but needed if k == f ...
        if k == self.f:
            topk = fine_distances.argsort(axis=2)[:, :, :k]  # m x n x k
        else:
        # ... while this partitions the array and sorts only the topk
            unsorted_topk = fine_distances.argpartition(k, axis=2)[:, :, :k]
            sorted_topk_idx = np.take_along_axis(fine_distances, unsorted_topk, axis=2).argsort(axis=2)
            topk = np.take_along_axis(unsorted_topk, sorted_topk_idx, axis=2)  # m x n x k

        # build the representation
        # the nested-for-loops way ...
        """
        rows, cols, data = [], [], []
        for subvector_index, sub_topk in enumerate(topk):  # 0 ...  m-1
            for sample_index, nns in enumerate(sub_topk):  # 0 ... n
                for importance, nn in enumerate(nns[::-1], start=1):  # 1 ... k
                    rows.append(sample_index)
                    cols.append(nn + self.f * subvector_index + self.f * self.m * coarse_codes[sample_index])
                    data.append(importance)

        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)
        
        sparse_repr = csr_matrix((data, (rows, cols)), shape=(n, self.vocab_size))
        """

        # ... or the vectorized way (WARNING: heavy broadcasting ahead)
        rows = np.arange(n).reshape(1, -1, 1)
        m_shifts = np.arange(self.m).reshape(-1, 1, 1)
        c_shifts = coarse_codes.reshape(1, -1, 1)
        cols = topk + self.f * m_shifts + self.f * self.m * c_shifts
        data = np.arange(k, 0, -1).reshape(1, 1, -1)
        
        rows, cols, data = np.broadcast_arrays(rows, cols, data)

        rows = rows.flatten()
        cols = cols.flatten()
        data = data.flatten()

        sparse_repr = csr_matrix((data, (rows, cols)), shape=(n, self.vocab_size))

        return sparse_repr

    def encode_parallel(self, x, permutation_length=None):

        k = self.f if permutation_length is None else permutation_length

        batch_size = int(math.ceil(len(x) / cpu_count()))
        results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(
            delayed(_spqr_encode)(x[i:i+batch_size], self.l1_centroids, self.l2_centroids, k)
            for i in range(0, len(x), batch_size)
        )

        results = vstack([
            csr_matrix((data, (rows, cols)), shape=(n, self.vocab_size))
            for rows, cols, data, n in results
        ])

        return results