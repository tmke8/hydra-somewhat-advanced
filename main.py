from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

from utils import flatten

# ============== model configs ================


@dataclass
class ModelConfig:
    """Base config for all models."""


@dataclass
class MlpConfig(ModelConfig):
    layers: int = 3
    hidden_units: int = 10


class Kern(Enum):
    linear = auto()
    RBF = auto()
    poly = auto()


@dataclass
class SVMConfig(ModelConfig):
    kernel: Kern = Kern.RBF
    C: float = 1.0


# ============== dataset configs ================


@dataclass
class DatasetConfig:
    """Base class of data configs."""

    dir: Path


@dataclass
class CmnistConfig(DatasetConfig):
    padding: int = 2
    color_background: bool = False


@dataclass
class AdultConfig(DatasetConfig):
    drop_native: bool = False
    drop_discrete: bool = False


# ============== main config ================


@dataclass
class Config:
    """Main config class."""

    model: ModelConfig
    dataset: DatasetConfig
    seed: int = 42
    data_pcnt: float = 1.0
    use_wandb: bool = False
    # unfortunately, hydra completely ignore the `init=False` directive
    use_cuda: bool = field(init=False, default=False)


# ============== register config classes ================

# ConfigStore enables type validation
cs = ConfigStore.instance()
cs.store(name="primary_schema", node=Config)
# register the subconfigs
cs.store(
    node=MlpConfig,  # class that defines the config
    name="mlp",  # the name under which this config will be accessible
    package="model",  # entry in `Config` where this should be stored
    group="model/schema",  # specify a key under which this scheme can be selected
)
cs.store(node=SVMConfig, name="svm", package="model", group="model/schema")
cs.store(node=CmnistConfig, name="cmnist", package="dataset", group="dataset/schema")
cs.store(node=AdultConfig, name="adult", package="dataset", group="dataset/schema")


# =============== main function =================


@hydra.main(config_path="conf", config_name="primary", version_base="1.2")
def my_app(hydra_cfg: DictConfig) -> None:
    # convert the dict-like hydra config into a real `Config` object
    cfg: Config = OmegaConf.to_object(hydra_cfg)  # type: ignore

    if isinstance(cfg.model, MlpConfig):
        print("using MLP")
        mlp = cfg.model
        print(f"{mlp.layers=}")
        print(f"{mlp.hidden_units=}")
    elif isinstance(cfg.model, SVMConfig):
        print("using SVM")
        svm = cfg.model
        print(f"{svm.kernel=}")
        print(f"{svm.C=}")

    print()

    print(f"data dir: '{cfg.dataset.dir}'")
    if isinstance(cfg.dataset, AdultConfig):
        adult_cfg = cfg.dataset
        print("using Adult dataset")
        print(f"{adult_cfg.drop_native=}")
    elif isinstance(cfg.dataset, CmnistConfig):
        cmnist_cfg = cfg.dataset
        print("using CMNIST dataset")
        print(f"{cmnist_cfg.padding=}")

    print()
    print(f"{cfg.seed=}")
    print(f"{cfg.use_wandb=}")
    print(f"{cfg.data_pcnt=}")
    print(f"{cfg.use_cuda=}")

    print()
    print("Config as flat dictionary:")
    print(flatten(OmegaConf.to_container(hydra_cfg, enum_to_str=True)))  # type: ignore


if __name__ == "__main__":
    my_app()
