# utils.py
import os
import pickle

def save_data(data, filepath):
    """
    Save data to a pickle file.

    Args:
        data:     any Python object to save
        filepath: full path including filename and .pkl extension
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_data(filepath):
    """
    Load data from a pickle file.

    Args:
        filepath: full path including filename and .pkl extension

    Returns:
        the loaded object
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)