"""Collection of various utility functions."""
import pickle
import yaml


def save_as_pickle(object: object, storage_path: str) -> None:
    """Save object to disk in pickle format.

    Args:
        object: Object to be stored as pickle.
        storage_path: File path and name to the disk location.

    Returns:
        None
    """
    pickle_file = open(storage_path, 'wb')
    pickle.dump(object, pickle_file)
    pickle_file.close()


def load_pickle(path_to_pickle_file: str) -> object:
    """Load the pickle file from storage.

    Args:
        path_to_pickle_file: File path and name to the disk location.

    Returns:
        Loaded object.
    """
    pickle_file = open(path_to_pickle_file, 'rb')
    obj = pickle.load(pickle_file)
    pickle_file.close()
    return obj


def load_config(path: str) -> dict:
    """Load config yaml files.

    Args:
        path: Path to config file.

    Returns:
        The config file as a dict.
    """
    with open(path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
