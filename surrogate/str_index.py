from abc import ABC, abstractmethod
import itertools

import numpy as np
from scipy.sparse import csr_matrix, vstack


class SurrogateTextIndex(ABC):
    """ An index producing term-frequency-like encodings (positive integer vectors)
        and computing scores as inner products using inverted lists.
    """

    def __init__(self, vocab_size):
        """ Constructor """
        self.db = None
        self.vocab_size = vocab_size

    def add(self, x, *args, **kwargs):
        """ Encodes and stores encoded vectors (in sparse format) for subsequent search.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
            args, kwargs: additional arguments for the encode() method.
        """
        x_enc = self.encode(x, *args, **kwargs)
        self.db = vstack((self.db, x_enc))
    
    @property
    def num_elements(self):
        """Returns the number of non-zero elements stored in the posting lists of the index."""
        return self.db.nnz

    @abstractmethod
    def encode(self, x):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """
        raise NotImplementedError

    def reset(self):
        """ Clears the index removing stored vectors. Values of learned parameters are kept. """
        del self.db
        self.db = csr_matrix((0, self.vocab_size), dtype='int32')

    def search(self, q, k, *args, **kwargs):
        """ Performs kNN search with given queries.
        Args:
            q (ndarray): a (N,D)-shaped matrix of query vectors.
            k (int): the number of nearest neighbors to return.
            args, kwargs: additional arguments for the encode() method.
        """
        k = self.db.shape[0] if k is None else k
        nq = len(q)

        q_enc = self.encode(q, *args, **kwargs)

        ## this materializes in RAM the entire score matrix ...
        # scores = q_enc.dot(self.db.T).toarray()
        # indices = scores.argsort(axis=1)[:, ::-1][:, :k]
        # sorted_scores = scores[np.arange(nq).reshape(-1, 1), indices]

        breakpoint()

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
    
    @abstractmethod
    def train(self, x):
        """ Learn parameters from data.
        Args:
            x (ndarray): a (N,D)-shaped matrix of training vectors.
        """
        raise NotImplementedError
    