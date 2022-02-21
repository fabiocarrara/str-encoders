import argparse

from .deep_perm import DeepPermutation
from .thr_sq import ThresholdSQ
from .topk_sq import TopKSQ
from .ivf_thr_sq import IVFThresholdSQ
from .ivf_topk_sq import IVFTopKSQ
from .spqr import SPQR


def index_factory(d, index_type, index_params):
    if index_type == 'deep-perm':
        return DeepPermutation(d, **index_params)
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


def argparser():

    parser = argparse.ArgumentParser(description='Surrogate Text Index Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='index_type')

    deep_perm_parser = subparsers.add_parser('deep-perm', help='Deep Permutation STR Index')
    deep_perm_parser.add_argument('-c', '--use-centroids', action='store_true', default=False, help='Find Pivots with k-Means.')
    deep_perm_parser.add_argument('-L', '--permutation-length', type=int, default=None, help='length of the permutation prefix (None for full permutation)')
    deep_perm_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')

    thrsq_parser = subparsers.add_parser('thr-sq', help='Threshold Scalar Quantization STR Index')
    thrsq_parser.add_argument('-Q', '--threshold-percentile', type=int, default=75, help='Controls how many values are discarded when encoding. Must be between 1 and 99 inclusive.')
    thrsq_parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
    thrsq_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    thrsq_parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')

    ivfthrsq_parser = subparsers.add_parser('ivf-thr-sq', help='Residual Chunked Threshold Scalar Quantization STR Index')
    ivfthrsq_parser.add_argument('-c', '--n-coarse-centroids', type=int, default=None, help='no of coarse centroids')
    ivfthrsq_parser.add_argument('-m', '--n-subvectors', type=int, default=None, help='no of subvectors')
    ivfthrsq_parser.add_argument('-Q', '--threshold-percentile', type=int, default=75, help='Controls how many values are discarded when encoding. Must be between 1 and 99 inclusive.')
    ivfthrsq_parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
    ivfthrsq_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    ivfthrsq_parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')

    topksq_parser = subparsers.add_parser('topk-sq', help='TopK Scalar Quantization STR Index')
    topksq_parser.add_argument('-k', '--keep', type=float, default=0.25, help='Controls how many values are discarded when encoding. Must be between 0.0 and 1.0 inclusive.')
    topksq_parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
    topksq_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    topksq_parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')

    ivftopksq_parser = subparsers.add_parser('ivf-topk-sq', help='Chunked TopK Scalar Quantization STR Index')
    ivftopksq_parser.add_argument('-c', '--n-coarse-centroids', type=int, default=None, help='no of coarse centroids')
    ivftopksq_parser.add_argument('-m', '--n-subvectors', type=int, default=None, help='no of subvectors')
    ivftopksq_parser.add_argument('-k', '--keep', type=float, default=0.25, help='Controls how many values are discarded when encoding. Must be between 0.0 and 1.0 inclusive.')
    ivftopksq_parser.add_argument('-s', '--sq-factor', type=float, default=1000, help='Controls the quality of the scalar quantization.')
    ivftopksq_parser.add_argument('-C', '--rectify-negatives', action='store_true', default=False, help='Apply CReLU trasformation.')
    ivftopksq_parser.add_argument('-n', '--l2-normalize', action='store_true', default=False, help='L2-normalize vectors before processing.')

    spqr_parser = subparsers.add_parser('spqr', help='SPQR STR Index')
    spqr_parser.add_argument('-c', '--n-coarse-centroids', type=int, default=None, help='no of coarse centroids')
    spqr_parser.add_argument('-m', '--n-subvectors', type=int, default=None, help='no of subvectors for PQ')
    spqr_parser.add_argument('-f', '--n-fine-centroids', type=int, default=2**8, help='no of fine centroids')

    return parser