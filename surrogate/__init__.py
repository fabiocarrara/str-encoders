import argparse

from .deep_perm import DeepPermutation
from .spqr import SPQR


def index_factory(d, index_type, index_params):
    if index_type == 'deep-perm':
        return DeepPermutation(d, **index_params)
    if index_type == 'spqr':
        return SPQR(d, **index_params)

    raise NotImplementedError(f'{index_type} not implemented')


def argparser():

    parser = argparse.ArgumentParser(description='Surrogate Text Index Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='index_type')

    deep_perm_parser = subparsers.add_parser('deep-perm', help='Deep Permutation STR Index')
    deep_perm_parser.add_argument('-L', '--permutation-length', type=int, default=None, help='length of the permutation prefix (None for full permutation)')

    spqr_parser = subparsers.add_parser('spqr', help='SPQR STR Index')
    spqr_parser.add_argument('-c', '--n-coarse-centroids', type=int, default=None, help='no of coarse centroids')
    spqr_parser.add_argument('-m', '--n-subvectors', type=int, default=None, help='no of subvectors for PQ')
    spqr_parser.add_argument('-f', '--n-fine-centroids', type=int, choices=(2**4, 2**8, 2**12, 2**16), default=2**8, help='no of fine centroids')

    return parser