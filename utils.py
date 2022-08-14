"""Utility functions."""
from __future__ import annotations
from collections.abc import MutableMapping
from typing import Any, List, Tuple


def flatten(
    d: MutableMapping[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
