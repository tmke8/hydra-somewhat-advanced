import dataclasses
from dataclasses import MISSING, is_dataclass
from typing import Any, Final, get_args, get_type_hints

from hydra.core.config_store import ConfigStore

NEED: Final = "there should be"
IF: Final = "if an entry has"


def register_hydra_config(
    main_cls: type,
    groups: dict[str, dict[str, type]],
    schema_name: str = "config_schema",
) -> None:
    """Check the given config and store everything in the ConfigStore.

    This function performs two tasks: 1) make the necessary calls to `ConfigStore`
    and 2) run some checks over the given config and if there are problems, try to give
    a nice error message.

    Args:
        main_cls: The main config class; needs to be a dataclass.
        groups: A dictionary that defines all the variants. The keys of top level of the
            dictionary should corresponds to the group names, and the keys in the nested
            dictionaries should correspond to the names of the options.
        schema_name: Name of the main schema. This name has to appear in the defaults
            list in the main config file.

    Raises:
        ValueError: If the config is malformed in some way.
        RuntimeError: If hydra itself is throwing an error.

    Example:

        .. code-block:: python

            @dataclass
            class DataModule:
                root: Path = Path()

            @dataclass
            class LinearModel:
                dim: int = 256

            @dataclass
            class CNNModel:
                kernel: int = 3

            @dataclass
            class Config:
                dm: DataModule = dataclasses.field(default_factory=DataModule)
                model: Any

            groups = {"model": {"linear": LinearModel, "cnn": CNNModel}}
            register_hydra_config(Config, groups)
    """
    assert isinstance(main_cls, type), "`main_cls` has to be a type."
    if not is_dataclass(main_cls):
        raise ValueError(f"The config class {main_cls.__name__} should be a dataclass.")
    entries = dataclasses.fields(main_cls)
    try:
        types = get_type_hints(main_cls)
    except NameError as exc:
        raise ValueError(
            f"Can't resolve type hints from the config class: `{main_cls.__name__}`."
        ) from exc

    for entry in entries:
        typ = types[entry.name]
        if typ == Any:
            if (group := groups.get(entry.name)) is not None:
                for var_name, var_class in group.items():
                    if not is_dataclass(var_class):
                        raise ValueError(
                            f"All variants should be dataclasses: type "
                            f"`{var_class.__name__}` of variant "
                            f"`{entry.name}={var_name}` is not a dataclass."
                        )
            else:
                raise ValueError(f"{IF} type `Any`, {NEED} variants: `{entry.name}`")
            if entry.default is not MISSING or entry.default_factory is not MISSING:
                raise ValueError(
                    f"{IF} type `Any`, {NEED} no default value: `{entry.name}`"
                )
        else:
            if is_dataclass(typ):
                if entry.default is MISSING and entry.default_factory is MISSING:
                    if (group := groups.get(entry.name)) is not None:
                        for var_name, var_class in group.items():
                            if not issubclass(var_class, typ):  # type: ignore
                                typ_name = typ.__name__  # type: ignore
                                raise ValueError(
                                    f"All variants should be subclasses of their "
                                    f"entry's type: type `{var_class.__name__}` of "
                                    f"variant `{entry.name}={var_name}` "
                                    f"is not a subclass of `{typ_name}`."
                                )
                    else:
                        raise ValueError(
                            f"{IF} a dataclass type, {NEED} a default value or "
                            f"registered variants: `{entry.name}`. You can specify a "
                            "default value with `field(default_factory=...)`."
                        )
                else:
                    if entry.name in groups:
                        raise ValueError(
                            "Can't have both a default value and variants: "
                            f"`{entry.name}`."
                        )
            elif entry.name in groups:
                raise ValueError(
                    f"Entry `{entry.name}` has registered variants, but its type "
                    f"annotation, `{getattr(typ, '__name__', str(typ))}`, is not a "
                    "dataclass. (Note that unions of dataclasses are not allowed "
                    "either.) You can always use `Any` for the type annotation."
                )

    cs = ConfigStore.instance()
    cs.store(node=main_cls, name=schema_name)
    for group, entries in groups.items():
        for var_name, var_type in entries.items():
            if (bases := getattr(var_type, "__orig_bases__", None)) is not None:
                if len(bases) > 0 and len(get_args(bases[0])) > 0:
                    raise ValueError(
                        f"Can't register a dataclass with generic base class: "
                        f"`{var_type.__name__}` with base class `{bases[0].__name__}`."
                    )
            try:
                cs.store(node=var_type, name=var_name, group=group)
            except Exception as exc:
                raise RuntimeError(
                    f"{main_cls=}, {var_type=}, {var_name=}, {group=}"
                ) from exc
