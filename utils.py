"""Utility functions."""
import collections


def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
