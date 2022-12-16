import argparse

from .deep_perm import DeepPermutation
from .thr_sq import ThresholdSQ
from .topk_sq import TopKSQ
from .topk_cnsq import TopKCNSQ
from .ivf_deep_perm import IVFDeepPermutation
from .ivf_thr_sq import IVFThresholdSQ
from .ivf_topk_sq import IVFTopKSQ
from .spqr import SPQR

from .io import load_index, save_index
from .util import generate_documents


def index_factory(d, index_type, index_params):
    if index_type == 'deep-perm':
        return DeepPermutation(d, **index_params)
    if index_type == 'ivf-deep-perm':
        return IVFDeepPermutation(d, **index_params)
    if index_type == 'thr-sq':
        return ThresholdSQ(d, **index_params)
    if index_type == 'ivf-thr-sq':
        return IVFThresholdSQ(d, **index_params)
    if index_type == 'topk-sq':
        return TopKSQ(d, **index_params)
    if index_type == 'ivf-topk-sq':
        return IVFTopKSQ(d, **index_params)
    if index_type == 'spqr':
        return SPQR(d, **index_params)

    raise NotImplementedError(f'{index_type} not implemented')


def add_index_argparser(parser):
    subparsers = parser.add_subparsers(dest='index_type', help='STR index type. Can be one of:', metavar='index', required=True)
    common = dict(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    deep_perm_parser = subparsers.add_parser('deep-perm', help='Deep Permutation', **common)
    deep_perm_parser.add_argument('-c', '--use-centroids', action='store_true', default=False, help='Find Pivots with k-Means.')
    deep_perm_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    deep_perm_parser.add_argument('-L', '--permutation-length', type=int, default=None, help='length of the permutation prefix (None for full permutation)')
    deep_perm_parser.set_defaults(
        train_params=('use_centroids',),
        build_params=('rectify_negatives', 'permutation_length'),
        query_params=()
    )

    ivf_deep_perm_parser = subparsers.add_parser('ivf-deep-perm', help='Chunked Deep Permutation', **common)
    ivf_deep_perm_parser.add_argument('-c', '--n-coarse-centroids', type=int, default=None, help='no of coarse centroids')
    ivf_deep_perm_parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')
    ivf_deep_perm_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    ivf_deep_perm_parser.add_argument('-L', '--permutation-length', type=int, default=None, help='length of the permutation prefix (None for full permutation)')
    ivf_deep_perm_parser.add_argument('-p', '--nprobe', type=int, default=None, help='how many partitions to visit at query time')
    ivf_deep_perm_parser.set_defaults(
        train_params=('n_coarse_centroids', 'l2_normalize'),
        build_params=('rectify_negatives', 'permutation_length'),
        query_params=('nprobe',)
    )

    thrsq_parser = subparsers.add_parser('thr-sq', help='Threshold Scalar Quantization', **common)
    thrsq_parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')
    thrsq_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    thrsq_parser.add_argument('-m', '--subtract-mean', action='store_true', default=False, help='Compute and subtract mean database vector.')
    thrsq_parser.add_argument('-r', '--rotation-matrix', type=int, default=None, help='seed for generating a random orthogonal matrix to apply to vectors; if omitted, no transformation is applied.')
    thrsq_parser.add_argument('-Q', '--threshold-percentile', type=int, default=75, help='Controls how many values are discarded when encoding. Must be between 1 and 99 inclusive.')
    thrsq_parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
    thrsq_parser.set_defaults(
        train_params=('l2_normalize', 'rectify_negatives', 'subtract_mean', 'rotation_matrix', 'threshold_percentile'),
        build_params=('sq_factor',),
        query_params=()
    )

    ivfthrsq_parser = subparsers.add_parser('ivf-thr-sq', help='Residual Chunked Threshold Scalar Quantization', **common)
    ivfthrsq_parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')
    ivfthrsq_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    ivfthrsq_parser.add_argument('-c', '--n-coarse-centroids', type=int, default=None, help='no of coarse centroids')
    ivfthrsq_parser.add_argument('-m', '--n-subvectors', type=int, default=None, help='no of subvectors')
    ivfthrsq_parser.add_argument('-Q', '--threshold-percentile', type=int, default=75, help='Controls how many values are discarded when encoding. Must be between 1 and 99 inclusive.')
    ivfthrsq_parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
    ivfthrsq_parser.add_argument('-p', '--nprobe', type=int, default=None, help='how many partitions to visit at query time')
    ivfthrsq_parser.set_defaults(
        train_params=('l2_normalize', 'rectify_negatives', 'n_coarse_centroids', 'n_subvectors', 'threshold_percentile'),
        build_params=('sq_factor',),
        query_params=('nprobe',)
    )

    topksq_parser = subparsers.add_parser('topk-sq', help='TopK Scalar Quantization', **common)
    topksq_parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')
    topksq_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    topksq_parser.add_argument('-r', '--rotation-matrix', type=int, default=None, help='seed for generating a random orthogonal matrix to apply to vectors; if omitted, no transformation is applied.')
    topksq_parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
    topksq_parser.add_argument('-k', '--keep', type=float, default=0.25, help='Controls how many values are discarded when encoding. Must be between 0.0 and 1.0 inclusive.')
    topksq_parser.set_defaults(
        train_params=(),
        build_params=('l2_normalize', 'rectify_negatives', 'rotation_matrix', 'sq_factor', 'keep'),
        query_params=()
    )

    ivftopksq_parser = subparsers.add_parser('ivf-topk-sq', help='Chunked TopK Scalar Quantization', **common)
    ivftopksq_parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')
    ivftopksq_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    ivftopksq_parser.add_argument('-c', '--n-coarse-centroids', type=int, default=512, help='no of coarse centroids')
    ivftopksq_parser.add_argument('-m', '--n-subvectors', type=int, default=1, help='no of subvectors')
    ivftopksq_parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
    ivftopksq_parser.add_argument('-k', '--keep', type=float, default=0.25, help='Controls how many values are discarded when encoding. Must be between 0.0 and 1.0 inclusive.')
    ivftopksq_parser.add_argument('-p', '--nprobe', type=int, default=1, help='how many partitions to visit at query time')
    ivftopksq_parser.set_defaults(
        train_params=('l2_normalize', 'n_coarse_centroids', 'n_subvectors'),
        build_params=('rectify_negatives', 'sq_factor', 'keep'),
        query_params=('nprobe',)
    )

    spqr_parser = subparsers.add_parser('spqr', help='SPQR', **common)  # TODO remove
    spqr_parser.add_argument('-c', '--n-coarse-centroids', type=int, default=None, help='no of coarse centroids')
    spqr_parser.add_argument('-m', '--n-subvectors', type=int, default=None, help='no of subvectors for PQ')
    spqr_parser.add_argument('-f', '--n-fine-centroids', type=int, default=2**8, help='no of fine centroids')
    spqr_parser.set_defaults(
        train_params=('n_coarse_centroids', 'n_subvectors', 'n_fine_centroids'),
        build_params=(),
        query_params=()
    )
    
    return parser