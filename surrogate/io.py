import joblib


def save_index(index, path):
    joblib.dump(index, path)


def load_index(path):
    return joblib.load(path)