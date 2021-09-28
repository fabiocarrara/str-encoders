import itertools
import math

import faiss
import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.distance import cdist


class SPQR:
    """ Implements Surrogate of Product Quantizator Representation (SQPR). """

    def __init__(
        self,
        d_or_index,
        n_coarse_centroids=None,
        n_subvectors=None,
        n_fine_centroids_log2=None,
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
    
    def encode_one(self, x, permutation_length=None):
        """ Encodes one vector and returns the SPQR representation.
        Args:
            x (ndarray): a (D,)-shaped vector to be encoded.
            permutation_length (int): the number of nearest fine centroids to keep
                                      when truncating the permutation; if None, the
                                      whole permutation is used. Defaults to None.
        """
        assert self.faiss_index.is_trained, "Index must be trained before encoding."

        k = self.f if permutation_length is None else permutation_length
        d = x.shape[0]

        assert self.d == d, f"Dimension mismatch: expected {self.d}, got {d}"

        # find nearest coarse centroid
        coarse_code = cdist(x.reshape(1, -1), self.l1_centroids, metric='euclidean').squeeze().argmin()

        # compute residual
        residual = x - self.l1_centroids[coarse_code]
        residual = np.split(residual, self.m)

        # compute distances to fine centroids for each subvector
        fine_distances = np.array([
            cdist(r.reshape(1, -1), c, metric='euclidean').squeeze()
                for r, c in zip(residual, self.l2_centroids)
        ])  # m x f

        # find top-k NNs for each subvector
        topk = fine_distances.argsort(axis=1)[:, :k]  # m x k

        # build the representation
        """
        # the nested-for-loops way ...
        rows, cols, data = [], []
        for i, nns in enumerate(topk):
            for importance, nn in enumerate(nns[::-1], start=1):
                rows.append(0)
                cols.append(nn + self.f * i + self.f * self.m * coarse_code)
                data.append(importance)

        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)
        sparse_repr = csr_matrix((data, (rows, cols)), shape=(1, self.vocab_size))
        """

        # ... or the vectorized way
        shifts = np.broadcast_to(np.arange(self.m).reshape(-1, 1), topk.shape)
        cols = (topk + self.f * shifts + self.f * self.m * coarse_code).flatten()
        data = np.broadcast_to(np.arange(k, 0, -1).reshape(1, -1), topk.shape).flatten()
        rows = np.zeros_like(cols)

        sparse_repr = csr_matrix((data, (rows, cols)), shape=(1, self.vocab_size))

        return sparse_repr

    def encode(self, x, permutation_length=None):
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
        coarse_codes = cdist(x, self.l1_centroids, metric='euclidean').argmin(axis=1) # n 

        # compute residual
        residuals = x - self.l1_centroids[coarse_codes]  # n x d
        residuals = np.split(residuals, self.m, axis=1)  # m x n x d/m

        # compute distances to fine centroids for each subvector
        fine_distances = np.array([ cdist(r, c, metric='euclidean')
                for r, c in zip(residuals, self.l2_centroids) ])  # m x n x f

        # find top-k NNs for each subvector
        topk = fine_distances.argsort(axis=2)[:, :, :k]  # m x n x k

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

    def add(self, x, permutation_length=None):
        """ Encodes and stores encoded vectors (in sparse format) for subsequent search.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
            permutation_length (int): the number of nearest fine centroids to keep
                                      when truncating the permutation; if None, the
                                      whole permutation is used. Defaults to None.
        """
        x_enc = self.encode(x, permutation_length=permutation_length)
        self.db = vstack((self.db, x_enc))

    def reset(self):
        """ Clears the index removing stored vectors. The learned centroids are kept. """
        del self.db
        self.db = csr_matrix((0, self.vocab_size), dtype='float32')
    
    def search(self, q, permutation_length=None, k=None):
        """ Performs bruteforce search with given queries.
        Args:
            q (ndarray): a (N,D)-shaped matrix of query vectors.
            permutation_length (int): the number of nearest fine centroids to keep
                                      when truncating the permutation; if None, the
                                      whole permutation is used. Defaults to None.
            k (int): the number of nearest neighbors to return; if None, all the
                     database will be returned in the result set. Defaults to None.
        """
        k = self.db.shape[0] if k is None else k
        nq = len(q)

        q_enc = self.encode(q, permutation_length=permutation_length)

        ## this materializes in RAM the entire score matrix ...
        # scores = q_enc.dot(self.db.T).toarray()
        # indices = scores.argsort(axis=1)[:, ::-1][:, :k]
        # sorted_scores = scores[np.arange(nq).reshape(-1, 1), indices]

        ## ... this instead materializes only nonzero scores
        indices = np.full((nq, k), -1, dtype='int')
        sorted_scores = np.zeros((nq, k), dtype='float32')

        scores = q_enc.dot(self.db.T)
        nonzero_scores = scores.nonzero()
        # group scores by query_idx and process each query independently
        for query_idx, nonzero_idx in itertools.groupby(zip(*nonzero_scores), key=lambda x: x[0]):
            nonzero_idx = np.array([e[1] for e in nonzero_idx])
            query_nonzero_scores = scores[query_idx, nonzero_idx].toarray()[0]
            query_neighbors = query_nonzero_scores.argsort()[::-1][:k]
            indices[query_idx, :len(query_neighbors)] = nonzero_idx[query_neighbors]
            sorted_scores[query_idx, :len(query_neighbors)] = query_nonzero_scores[query_neighbors]

        return sorted_scores, indices