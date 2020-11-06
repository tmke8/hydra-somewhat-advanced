from dataclasses import dataclass
from enum import Enum
from typing import cast

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING

from utils import flatten


# ============== model configs ================

@dataclass
class ModelConfig:
    """Base config for all models."""

@dataclass
class MlpConfig(ModelConfig):
    layers: int = 3
    hidden_units: int = 10

Kern = Enum("Kern", ["Linear", "RBF", "Poly"])

@dataclass
class SVMConfig(ModelConfig):
    kernel: Kern = Kern.RBF
    C: float = 1.0


# ============== dataset configs ================

@dataclass
class DatasetConfig:
    """Base class of data configs."""

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
    model: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    seed: int = 42
    data_pcnt: float = 1.0
    use_wandb: bool = False


# ============== register config classes ================

# ConfigStore enables type validation
cs = ConfigStore.instance()
cs.store(name="primary", node=Config)
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

@hydra.main(config_path="conf", config_name="primary")
def my_app(cfg: Config) -> None:
    if OmegaConf.get_type(cfg.model) is MlpConfig:
        mlp_cfg = cast(MlpConfig, cfg.model)
        print("using MLP")
        print(f"{mlp_cfg.layers=}")
        print(f"{mlp_cfg.hidden_units=}")
    elif OmegaConf.get_type(cfg.model) is SVMConfig:
        svm_cfg = cast(SVMConfig, cfg.model)
        print("using SVM")
        print(f"{svm_cfg.kernel=}")
        print(f"{svm_cfg.C=}")

    print()

    if OmegaConf.get_type(cfg.dataset) is AdultConfig:
        adult_cfg = cast(AdultConfig, cfg.dataset)
        print("using Adult dataset")
        print(f"{adult_cfg.drop_native=}")
    elif OmegaConf.get_type(cfg.dataset) is CmnistConfig:
        cmnist_cfg = cast(CmnistConfig, cfg.dataset)
        print("using CMNIST dataset")
        print(f"{cmnist_cfg.padding=}")
    
    print()
    print(f"{cfg.seed=}")
    print(f"{cfg.use_wandb=}")
    print(f"{cfg.data_pcnt=}")

    print()
    print("Config as flat dictionary:")
    print(flatten(OmegaConf.to_container(cfg)))


if __name__ == "__main__":
    my_app()
