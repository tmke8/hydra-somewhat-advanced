import hydra
from omegaconf import OmegaConf, DictConfig

from src.config_store_utils import register_hydra_config
from src.run import CONFIG_GROUPS, Config, run


@hydra.main(
    config_path="configs",  # We use the `configs/` directory for the yaml files.
    config_name="root",  # The root config files is `configs/root.yaml`.
    version_base="1.3",
)
def main(hydra_config: DictConfig) -> None:
    # The `hydra_config` object we get is essentially a dictionary.
    # We convert it into a real `Config` object with the `OmegaConf.to_object` function.
    cfg = OmegaConf.to_object(hydra_config)
    assert isinstance(cfg, Config)
    run(cfg)


if __name__ == "__main__":
    # Before calling the main function, we need to register the main `Config` class and
    # the configuration groups. Without this, hydra doesn't know which keys and values
    # are valid in the configuration.
    # Whatever you set here as `schema_name` will need to be incluced as the first entry
    # in the `defaults` list in the main config yaml file (`configs/root.yaml`).
    register_hydra_config(Config, groups=CONFIG_GROUPS, schema_name="root_schema")
    main()
