from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING

__all__ = ["PathWrapper"]

def create_path(path: str) -> Path:
    return Path(path)


@dataclass
class PathWrapper:
    _target_: str = "custom_types.create_path"
    path: str = MISSING
