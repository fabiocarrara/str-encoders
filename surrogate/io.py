import joblib


def save_index(index, path):
    """ Save the index to a file. """
    joblib.dump(index, path)


def load_index(path):
    """ Load the index from a file. """
    return joblib.load(path)
