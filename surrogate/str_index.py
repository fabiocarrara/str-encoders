from abc import ABC, abstractmethod
import math

import numpy as np
from joblib import cpu_count, delayed, Parallel
from scipy import sparse

from . import util


def _search(
    q_enc,
    db,
    k,
    discount=0,
):
    nq = q_enc.shape[0]
    indices = np.full((nq, k), -1, dtype='int')
    sorted_scores = np.zeros((nq, k), dtype='float32')

    scores = q_enc.dot(db)

    if discount:
        q_nz = (q_enc > 0).astype('int')
        db_nz = (db > 0).astype('int')
        accessed_posts = (q_nz).dot(db_nz)
        discount_matrix = accessed_posts * discount
        scores -= discount_matrix

    for query_idx, query_scores in enumerate(scores):
        if query_scores.nnz == 0:
            continue

        query_neighbors = util.topk_sorted(query_scores.data, k)
        indices[query_idx, :len(query_neighbors)] = query_scores.indices[query_neighbors]
        sorted_scores[query_idx, :len(query_neighbors)] = query_scores.data[query_neighbors]

    return sorted_scores, indices


class SurrogateTextIndex(ABC):
    """ An index producing term-frequency-like encodings (positive integer vectors)
        and computing scores as inner products using inverted lists.
    """

    def __init__(self, vocab_size, parallel, discount=0, is_trained=True):
        """ Constructor """
        self.db = None
        self.vocab_size = vocab_size
        self.parallel = parallel
        self.discount = discount
        self.is_trained = is_trained

        self._to_commit = []
        self.reset()
    
    @staticmethod
    @abstractmethod
    def add_subparser(subparser, **kws):
        """ Adds a subparser to the given parser to add the index initialization options. """
        raise NotImplementedError

    def add(self, x, *args, **kwargs):
        """ Encodes and stores encoded vectors (in sparse format) for subsequent search.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
            args, kwargs: additional arguments for the encode() method.
        """
        x_enc = self.encode(x, inverted=True, *args, **kwargs)
        self._to_commit.append(x_enc)

    def commit(self):
        self.db = sparse.hstack([self.db] + self._to_commit).tocsr()
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
    def encode(self, x, inverted=True, query=False):
        """ Encodes vectors and returns their term-frequency representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
            inverted (bool): if True, returns the (V,N)-shaped inverted representation.
            query (bool): indicates wheather queries are to be encoded.
        Returns:
            x_enc (spmatrix): the encoded vectors
        """
        raise NotImplementedError

    def reset(self, build_params={}):
        """ Clears the index removing stored vectors. Values of learned parameters are kept.
            Optionally sets new build parameters.
        """
        del self.db
        self.db = sparse.coo_matrix((self.vocab_size, 0), dtype='int')

        for k, v in build_params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def search(self, q, k, *args, return_cost=False, **kwargs):
        """ Performs kNN search with given queries.
        Args:
            q (ndarray): a (N,D)-shaped matrix of query vectors.
            k (int): the number of nearest neighbors to return.
            return_cost (bool): whether to return query cost in terms of retrieved elements in the inverted index.
            args, kwargs: additional arguments for the encode() method.

        Returns:
            ndarray: a (N,D)-shaped matrix of sorted scores.
            ndarray: a (N,D)-shaped matrix of kNN indices.
        """
        q_enc = self.encode(q, *args, inverted=False, query=True, **kwargs)
        results = self.search_encoded(q_enc.tocsr(), k, *args, **kwargs)

        if return_cost:
            q_enc = q_enc.T.tocsr()  # we need inverted=True
            cost = self.search_cost_encoded(q_enc, *args, **kwargs)
            results += (cost,)

        return results

    def search_encoded(self, q_enc, k, *args, **kwargs):
        """ Performs kNN search with given already encoded queries.
        Args:
            q_env (sparse.csr_matrix): a (N,V)-shaped sparse matrix of encoded query vectors.
            k (int): the number of nearest neighbors to return.
            args, kwargs: additional arguments for the encode() method.

        Returns:
            ndarray: a (N,D)-shaped matrix of sorted scores.
            ndarray: a (N,D)-shaped matrix of kNN indices.
        """
        k = self.db.shape[0] if k is None else k
        nq = q_enc.shape[0]

        search_args = (
            self.db,
            k,
            self.discount,
        )

        if self.parallel:
            func = delayed(_search)
            batch_size = int(math.ceil(nq / cpu_count()))
            jobs = (func(q_enc[i:i+batch_size], *search_args) for i in range(0, nq, batch_size))
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(jobs)

            sorted_scores, indices = zip(*results)
            sorted_scores = np.vstack(sorted_scores)
            indices = np.vstack(indices)

        else:  # sequential version
            sorted_scores, indices = _search(q_enc, *search_args)

        return sorted_scores, indices

    def search_cost(self, q, *args, **kwargs):
        assert not self.dirty, "Search cost can be computed only on committed indices."
        q_enc = self.encode(q, *args, inverted=True, query=True, **kwargs).tocsr()
        return self.search_cost_encoded(q_enc, *args, **kwargs)

    def search_cost_encoded(self, q_enc, *args, **kwargs):
        assert not self.dirty, "Search cost can be computed only on committed indices."
        q_nnz = np.diff(q_enc.indptr).astype(np.int64)
        x_nnz = np.diff(self.db.indptr).astype(np.int64)
        return np.dot(q_nnz, x_nnz)

    @abstractmethod
    def train(self, x):
        """ Learn parameters from data.
        Args:
            x (ndarray): a (N,D)-shaped matrix of training vectors.
        """
        raise NotImplementedError
