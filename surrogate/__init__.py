import argparse

from .deep_perm import DeepPermutation
from .thr_sq import ThresholdSQ
from .topk_sq import TopKSQ
from .ivf_deep_perm import IVFDeepPermutation
from .ivf_thr_sq import IVFThresholdSQ
from .ivf_topk_sq import IVFTopKSQ

from .io import load_index, save_index
from .util import generate_documents


REGISTERED_INDICES = {
    'deep-perm': DeepPermutation,
    'thr-sq': ThresholdSQ,
    'topk-sq': TopKSQ,
    'ivf-deep-perm': IVFDeepPermutation,
    'ivf-thr-sq': IVFThresholdSQ,
    'ivf-topk-sq': IVFTopKSQ,
}


def index_factory(d, index_type, index_params):
    """ Factory function to create an index object.
    Args:
        d (int): the number of dimensions of the vectors to be indexed.
        index_type (str): the type of index to create. See `surrogate.REGISTERED_INDICES`.
        index_params (dict): parameters to pass to the index constructor.
    Returns:
        SurrogateTextIndex: the index object.
    """
    if index_type in REGISTERED_INDICES:
        return REGISTERED_INDICES[index_type](d, **index_params)

    raise NotImplementedError(f'{index_type} not implemented')


def add_index_argparser(parser):
    """ Add an argument parser for the index type.
    Args:
        parser (argparse.ArgumentParser): the parser to which to add the subparsers.
    Returns:
        argparse.ArgumentParser: the parser with the added subparsers.
    """
    subparsers = parser.add_subparsers(dest='index_type', help='STR index type. Can be one of:', metavar='index', required=True)
    common = dict(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for name, index in REGISTERED_INDICES.items():
        index.add_subparser(subparsers, **common)

    return parser