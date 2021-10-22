from abc import ABC, abstractmethod
import math

import numpy as np
from joblib import cpu_count, delayed, Parallel
from scipy import sparse

import utils


def _csr_hstack(blocks):
    """ A faster version of hstack for CSR matrix. """
    num_rows, num_cols = np.array([b.shape for b in blocks]).T
    
    assert np.all(num_rows == num_rows[0]), "blocks have different row numbers"
    num_rows = num_rows[0]

    res_indptr = np.sum([b.indptr for b in blocks], axis=0)
    res_data = np.empty(res_indptr[-1], dtype=blocks[0].data.dtype)
    res_indices = np.empty(res_indptr[-1], dtype=blocks[0].indices.dtype)

    offsets = np.cumsum(num_cols)
    offsets = np.insert(offsets, 0, 0)
    num_cols = offsets[-1]
    
    for i in range(num_rows):
        res_start = res_indptr[i]
        for b, o in zip(blocks, offsets):
            b_start, b_end = b.indptr[i], b.indptr[i+1]
            b_nnz = b_end - b_start
            res_end = res_start + b_nnz
            res_data[res_start:res_end] = b.data[b_start:b_end]
            res_indices[res_start:res_end] = b.indices[b_start:b_end] + o
            res_start = res_end

    return sparse.csr_matrix((res_data, res_indices, res_indptr), shape=(num_rows, num_cols))


def _search(
    q_enc,
    db,
    k
):
    nq = q_enc.shape[0]
    indices = np.full((nq, k), -1, dtype='int')
    sorted_scores = np.zeros((nq, k), dtype='float32')
    
    scores = q_enc.dot(db)

    for query_idx, query_scores in enumerate(scores):
        if query_scores.nnz == 0:
            continue
        
        query_neighbors = utils.topk_sorted(query_scores.data, k)
        indices[query_idx, :len(query_neighbors)] = query_scores.indices[query_neighbors]
        sorted_scores[query_idx, :len(query_neighbors)] = query_scores.data[query_neighbors]
    
    return sorted_scores, indices 


class SurrogateTextIndex(ABC):
    """ An index producing term-frequency-like encodings (positive integer vectors)
        and computing scores as inner products using inverted lists.
    """

    def __init__(self, vocab_size, parallel):
        """ Constructor """
        self.db = None
        self.vocab_size = vocab_size
        self.parallel = parallel

        self._to_commit = []
        self.reset()

    def add(self, x, *args, **kwargs):
        """ Encodes and stores encoded vectors (in sparse format) for subsequent search.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
            args, kwargs: additional arguments for the encode() method.
        """
        x_enc = self.encode(x, inverted=True, *args, **kwargs)
        self._to_commit.append(x_enc)
    
    def commit(self):
        self.db = _csr_hstack([self.db] + self._to_commit)
        self._to_commit.clear()
    
    @property
    def density(self):
        """Returns the number of non-zero elements stored in the posting lists of the index."""
        return self.db.nnz / np.prod(self.db.shape)
    
    @property
    def dirty(self):
        """Returns whether there are added vectors to commit."""
        return bool(self._to_commit)

    @abstractmethod
    def encode(self, x, inverted=True):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
            inverted (bool): if True, returns the (V,N)-shaped inverted representation
        Returns:
            x_enc (sparse.csr_matrix): the encoded vectors
        """
        raise NotImplementedError

    def reset(self):
        """ Clears the index removing stored vectors. Values of learned parameters are kept. """
        del self.db
        self.db = sparse.csr_matrix((self.vocab_size, 0), dtype='int')

    def search(self, q, k, *args, **kwargs):
        """ Performs kNN search with given queries.
        Args:
            q (ndarray): a (N,D)-shaped matrix of query vectors.
            k (int): the number of nearest neighbors to return.
            args, kwargs: additional arguments for the encode() method.

        Returns:
            ndarray: a (N,D)-shaped matrix of sorted scores.
            ndarray: a (N,D)-shaped matrix of kNN indices.
        """
        k = self.db.shape[0] if k is None else k
        nq = q.shape[0]

        q_enc = self.encode(q, *args, inverted=False, **kwargs)

        if self.parallel:
            batch_size = int(math.ceil(nq / cpu_count()))
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(
                delayed(_search)(q_enc[i:i+batch_size], self.db, k)
                for i in range(0, nq, batch_size)
            )

            sorted_scores, indices = zip(*results)
            sorted_scores = np.vstack(sorted_scores)
            indices = np.vstack(indices)

        else:  # sequential version
            sorted_scores, indices = _search(q_enc, self.db, k)

        return sorted_scores, indices
    
    @abstractmethod
    def train(self, x):
        """ Learn parameters from data.
        Args:
            x (ndarray): a (N,D)-shaped matrix of training vectors.
        """
        raise NotImplementedError
    