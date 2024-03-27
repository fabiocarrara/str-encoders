import functools
import logging
import math

from joblib import cpu_count, delayed, Parallel
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans

from . import util
from .str_index import SurrogateTextIndex


def _spqr_train_sklearn(
    x,                            # vectors to train from
    c,                            # number of coarse centroids
    m,                            # number of subvectors
    f,                            # number of fine centroids
    c_redo=1,                     # number of kmeans runs with different seeds (coarse)
    c_iter=100,                   # maximum number of kmeans iterations (coarse)
    f_redo=1,                     # number of kmeans runs with different seeds (fine)
    f_iter=100,                   # maximum number of kmeans iterations (fine)
    max_samples_per_centroid=256, # max samples per centroid; if more, we subsample
    verbose=0,                    # verbosity level
):
    # kmeans = KMeans
    kmeans = functools.partial(MiniBatchKMeans, batch_size=256*cpu_count(), compute_labels=False)

    nx = len(x)
    xt = x
    max_samples = max_samples_per_centroid * c
    if nx > max_samples:  # subsample train set
        logging.info(f'subsampling {max_samples} / {nx} for coarse centroid training.')
        subset = np.random.choice(nx, size=max_samples, replace=False)
        xt = x[subset]

    # compute coarse centroids
    l1_kmeans = kmeans(n_clusters=c, n_init=c_redo, max_iter=c_iter, verbose=verbose)
    l1_kmeans.fit(xt)
    l1_centroids = l1_kmeans.cluster_centers_

    max_samples = max_samples_per_centroid * f
    if nx > max_samples:  # subsample train set
        logging.info(f'subsampling {max_samples} / {nx} for fine centroid training.')
        subset = np.random.choice(nx, size=max_samples, replace=False)
        xt = x[subset]

    # compute residuals
    coarse_codes = l1_kmeans.predict(xt)
    residuals = xt - l1_centroids[coarse_codes]  # n x d
    residuals = np.split(residuals, m, axis=1)  # m x n x d/m

    l2_centroids = []
    for sub_vector in residuals:
        l2_kmeans = kmeans(n_clusters=f, n_init=f_redo, max_iter=f_iter, verbose=verbose)
        l2_kmeans.fit(sub_vector)
        l2_centroids.append(l2_kmeans.cluster_centers_)  # f x d/m

    l2_centroids = np.stack(l2_centroids) # m x f x d/m

    return l1_centroids, l2_centroids


def _spqr_encode(
    x,             # features to encode
    l1_centroids,  # coarse centroids
    l2_centroids,  # fine centroids
    k,             # permutation prefix length
    nprobe,        # how many coarse centroids to consider
):
    n, d = x.shape
    m, f, _ = l2_centroids.shape

    # find nearest coarse centroids
    l1_centroid_distances = cdist(x, l1_centroids, metric='sqeuclidean')
    coarse_codes = util.bottomk_sorted(l1_centroid_distances, nprobe, axis=1)  # n x nprobe

    # repeat x for nprobe times (1st nprobe times, then 2nd nprobe times, etc.; enables batching)
    x = np.broadcast_to(x.reshape(n, 1, d), (n, nprobe, d)).reshape(n * nprobe, d)  # n * nprobe, d
    coarse_codes = coarse_codes.flatten() # n * nprobe
    n *= nprobe

    # compute residuals
    residuals = x - l1_centroids[coarse_codes]  # n x d
    residuals = np.split(residuals, m, axis=1)  # m x n x d/m

    # compute distances to fine centroids for each subvector
    # vectorized, but memory-hungry ...
    """ 
    fine_distances = np.array([ cdist(r, c, metric='sqeuclidean')
            for r, c in zip(residuals, l2_centroids) ])  # m x n x f

    # find kNNs for each subvector
    nns = util.bottomk_sorted(fine_distances, k, axis=2)  # m x n x k
    """

    # ... a subvector at a time, less memory consumption (also faster?!)
    nns = np.empty((m, n, k), dtype=int)
    for i, (sub_residuals, sub_centroids) in enumerate(zip(residuals, l2_centroids)):
        sub_fine_distances = cdist(sub_residuals, sub_centroids, metric='sqeuclidean')  # n x f
        nns[i] = util.bottomk_sorted(sub_fine_distances, k, axis=1)  # n x k

    # build the representation
    # the nested-for-loops way ...
    """
    rows, cols, data = [], [], []
    for subvector_index, sub_nns in enumerate(nns):  # 0 ...  m-1
        for sample_index, sample_sub_nns in enumerate(sub_nns):  # 0 ... n
            for importance, nn in enumerate(sample_sub_nns[::-1], start=1):  # 1 ... k
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
    m_shifts = np.arange(m).reshape(-1, 1, 1)
    c_shifts = coarse_codes.reshape(1, -1, 1)
    cols = nns + f * m_shifts + f * m * c_shifts
    data = np.arange(k, 0, -1).reshape(1, 1, -1)
    
    rows, cols, data = np.broadcast_arrays(rows, cols, data)

    rows = rows.flatten()
    cols = cols.flatten()
    data = data.flatten()

    return rows, cols, data, n


def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


class SPQR(SurrogateTextIndex):
    """ Implements Surrogate of Product Quantizator Representation (SQPR). """

    def __init__(
        self,
        d_or_index,
        n_coarse_centroids=None,
        n_subvectors=None,
        n_fine_centroids=None,
        engine='sklearn',
        parallel=True,
    ):
        """ Constructor.
        Args:
            d_or_index (int or faiss.Index): the number of dimensions of the vectors
                                             or a prebuilt faiss Index.
            n_coarse_centroids (int): the number of coarse centroids of the level 1
                                      quantizer (voronoi partitioning).
            n_subvectors (int): the number of subvectors of the level 2 quantizer
                                (product quantization).
            n_fine_centroids (int): the number of fine centroids for the level 2
                                    quantizer (product quantization); must be a power
                                    of two if engine='faiss'
            engine (str): k-means implementation ('faiss' or 'sklearn')
        """

        assert (
            isinstance(d_or_index, int) and \
            isinstance(n_coarse_centroids, int) and \
            isinstance(n_subvectors, int) and \
            isinstance(n_fine_centroids, int)
        ) or (
            isinstance(d_or_index, faiss.swigfaiss.IndexIVFPQ) and
            (n_coarse_centroids is None) and \
            (n_subvectors is None) and \
            (n_fine_centroids is None)
        ), "You must specify a prebuilt faiss index " \
            "or all the params (as ints) needed to build the index."

        supported = ('faiss', 'sklearn')
        assert engine in supported, f'Unknown engine: {engine}. Supported engines: {supported}.'

        assert engine != 'faiss' or is_power_of_two(n_fine_centroids), 'FAISS supports only power-of-two values for n_fine_centroids.'

        self.l1_centroids = None
        self.l2_centroids = None
        self.permutation_length = None
        self.nprobe = 1

        self.engine = engine

        if isinstance(d_or_index, int):
            self.d = d_or_index
            self.c = n_coarse_centroids
            self.m = n_subvectors
            self.f = n_fine_centroids
        else:
            self.faiss_index = d_or_index
            self.d = d_or_index.d
            self.c = d_or_index.nlist
            self.m = d_or_index.pq.M
            self.f = d_or_index.pq.ksub
            self._load_faiss_centroids(d_or_index)

        vocab_size = self.c * self.m * self.f
        super().__init__(vocab_size, parallel)

    def _load_faiss_centroids(self, index):
        import faiss
        if index.is_trained:
            # get the level-1 centroids by reconstructing codes 0, ..., C-1
            self.l1_centroids = index.quantizer.reconstruct_n(0, index.nlist)
            # get the level-2 centroids and reshape them
            self.l2_centroids = faiss.vector_to_array(index.pq.centroids)
            # there are ksub centroids of length dsub for each of the M subvectors
            self.l2_centroids = self.l2_centroids.reshape(index.pq.M, index.pq.ksub, index.pq.dsub)

    @property
    def is_trained(self):
        """ Whether the index is trained. """
        return self.l1_centroids is not None # and self.l2_centroids is not None

    def train(self, x):
        if self.engine == 'faiss':
            import faiss
            f_log2 = int(math.log2(self.f))
            index_factory_string = f'IVF{self.c},PQ{self.m}x{f_log2}'
            faiss_index = faiss.index_factory(self.d, index_factory_string)
            faiss_index.do_polysemous_training = False
            faiss_index.train(x)
            self._load_faiss_centroids(faiss_index)
        elif self.engine == 'sklearn':
            self.l1_centroids, self.l2_centroids = _spqr_train_sklearn(x, self.c, self.m, self.f)
    
    def add_subparser(subparsers, **kws):
        spqr_parser = subparsers.add_parser('spqr', help='SPQR', **kws)
        spqr_parser.add_argument('-c', '--n-coarse-centroids', type=int, default=None, help='no of coarse centroids')
        spqr_parser.add_argument('-m', '--n-subvectors', type=int, default=None, help='no of subvectors for PQ')
        spqr_parser.add_argument('-f', '--n-fine-centroids', type=int, default=2**8, help='no of fine centroids')
        spqr_parser.set_defaults(
            train_params=('n_coarse_centroids', 'n_subvectors', 'n_fine_centroids'),
            build_params=(),
            query_params=()
        )
        
    def encode(self, x, inverted=True):
        """ Encodes vectors and returns their SPQR representations.
        Args:
            x (ndarray): a (N,D)-shaped matrix of vectors to be encoded.
        """
        assert self.is_trained, "Index must be trained before encoding."
        assert self.d == x.shape[1], f"Dimension mismatch: expected {self.d}, got {x.shape[1]}"

        k = self.f if self.permutation_length is None else self.permutation_length

        if self.parallel:
            batch_size = int(math.ceil(len(x) / cpu_count()))
            results = Parallel(n_jobs=-1, prefer='threads', require='sharedmem')(
                delayed(_spqr_encode)(x[i:i+batch_size], self.l1_centroids, self.l2_centroids, k, self.nprobe)
                for i in range(0, len(x), batch_size)
            )

            # use COO for fast sparse matrix concatenation
            results = [sparse.coo_matrix((data, (rows, cols)), shape=(n, self.vocab_size)) for rows, cols, data, n in results]
            results = sparse.vstack(results)
            results = results.T if inverted else results
            # then convert to CSR
            # results = results.tocsr()
            return results
       
        rows, cols, data, n = _spqr_encode(x, self.l1_centroids, self.l2_centroids, k, self.nprobe)
        if inverted:
            rows, cols = cols, rows
            shape = (self.vocab_size, n)
        else:
            shape = (n, self.vocab_size)

        return sparse.coo_matrix((data, (rows, cols)), shape=shape)
    
    @property
    def prefix_length(self):
        """(int): 
            the number of nearest fine centroids to keep
            when truncating the permutation; if None, the
            whole permutation is used. Defaults to None.
        """
        return self.permutation_length
    
    @prefix_length.setter
    def prefix_length(self, l):
        if (self.db.shape[1] > 0) and (l != self.permutation_length):
            raise ValueError('Cannot change prefix_length of a populated index. Reset the index first.')
        
        self.permutation_length = l
    
    def search(self, q, k, *args, **kwargs):
        sorted_scores, indices = super().search(q, k, *args, **kwargs)

        if self.nprobe > 1:  # merge results
            nq = len(q)

            sorted_scores = sorted_scores.reshape(nq, -1)
            indices = indices.reshape(nq, -1)

            idx = util.topk_sorted(sorted_scores, k, axis=1)
            sorted_scores = np.take_along_axis(sorted_scores, idx, axis=1)
            indices = np.take_along_axis(indices, idx, axis=1)

        return sorted_scores, indices

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'nprobe'):
            self.nprobe = 1
