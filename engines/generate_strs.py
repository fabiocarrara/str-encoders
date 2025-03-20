import argparse
from surrogate import TopKSQ, generate_documents
import utils

from tqdm import trange


def main(args):
    dataset = utils.get_ann_benchmark(args.dataset)
    x = dataset[args.split]
    n, d = x.shape

    encoder = TopKSQ(
        d,
        keep=args.keep,
        dim_multiplier=args.dim_multiplier,
        positivize='crelu',
        sq_factor=args.sq_factor,
        l2_normalize=True,
        parallel=True,
    )

    for start in trange(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        xi = x[start:end]
        xi_enc = encoder.encode(x, inverted=False)
        for doc in generate_documents(xi_enc, delimiter=args.delimiter):
            print(doc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset to use")
    parser.add_argument("-s", "--split", choices=('train', 'test'), default='train', help="Split to use")
    parser.add_argument("-d", "--delimiter", type=str, default='|', help="Delimiter for the generated documents")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Batch size for generating documents")
    parser.add_argument("-k", "--keep", type=int, default=0.25, help="Number of components to keep")
    parser.add_argument("-m", "--dim-multiplier", type=int, default=10, help="Dimensionality multiplier")
    parser.add_argument("-q", "--sq-factor", type=int, default=1e5, help="Scalar quantization factor")
    args = parser.parse_args()

    main(args)
