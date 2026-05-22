from dataclasses import asdict, dataclass, field
from typing import Final, Literal

from src.utils import flatten

# ============== model configs ================


# Even though this class doesn't have any fields, we nevertheless need to mark it as a
# dataclass, because otherwise Hydra will complain when we use it as a type annotation
# in the `Config` class.
@dataclass
class ModelCfg:
    """Base config for all models."""


@dataclass(kw_only=True)
class MlpCfg(ModelCfg):
    """Config for MLP model."""

    layers: int = 3
    hidden_units: int = 10
    dropout: float = 0.0
    activation: Literal["relu", "selu"] = "relu"

    def __post_init__(self) -> None:
        if self.layers < 1:
            raise ValueError("Number of layers must be at least 1.")
        if self.hidden_units < 1:
            raise ValueError("Number of hidden units must be at least 1.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("Dropout must be in the range [0.0, 1.0).")


@dataclass(kw_only=True)
class SvmCfg(ModelCfg):
    """Config for SVM model."""

    kernel: Literal["linear", "rbf", "poly"] = "rbf"
    C: float = 1.0


# ============== dataset configs ================


@dataclass(kw_only=True)
class DatasetCfg:
    """Base class of data configs."""

    # this has no default, so it needs to be specified either in yaml or on commandline
    dir: str


@dataclass(kw_only=True)
class CmnistCfg(DatasetCfg):
    padding: int = 2
    color_background: bool = False


@dataclass(kw_only=True)
class CompasCfg(DatasetCfg):
    drop_native: bool = False
    drop_discrete: bool = False


@dataclass(kw_only=True)
class OptimizationCfg:
    """Config for optimization."""

    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 32


@dataclass
class Config:
    """Main config class."""

    # The first two fields refer to configuration groups.
    # For this reason, we cannot specify a default for them here.
    # The defaults can be specified in the root config yaml file (`configs/root.yaml`).
    model: ModelCfg
    data: DatasetCfg

    # This is a normal subconfig, for which we can specify defaults, but note that in
    # dataclasses, the default may not be mutable, so we use `default_factory`.
    opt: OptimizationCfg = field(default_factory=OptimizationCfg)

    # These are normal fields, for which we can (and should) specify defaults.
    seed: int = 42
    data_pcnt: float = 1.0
    wandb: Literal["online", "offline", "disabled"] = "online"
    use_cuda: bool = field(
        default=False, init=False, metadata={"omegaconf_ignore": True}
    )
    gpu: int = 0  # Set to -1 to use CPU.


# Config groups enable us to have different configurations for different subcomponents.
# For example, one subcomponent is the dataseet, and we can have different datasets,
# COMPAS and ColoredMNIST, which need different keys and values to be configured.
CONFIG_GROUPS: Final = {
    "model": {"mlp": MlpCfg, "svm": SvmCfg},
    "data": {"cmnist": CmnistCfg, "compas": CompasCfg},
}


# =============== main function =================


def run(cfg: Config) -> None:
    if isinstance(cfg.model, MlpCfg):
        print("Using MLP.")
    elif isinstance(cfg.model, SvmCfg):
        print("Using SVM.")

    if isinstance(cfg.data, CompasCfg):
        print("Using COMPAS data.")
    elif isinstance(cfg.data, CmnistCfg):
        print("Using CMNIST dataset.")

    print()
    print(cfg)

    print()
    print(f"Config as flat dictionary: {flatten(asdict(cfg))}")
