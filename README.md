# Somewhat advanced example of using hydra

This example is without the `instantiate()` functionality (which allows instantiating arbitrary classes), but otherwise uses all of the core hydra concepts in some way.

## Installation

Either with pip:

```sh
pip install -r pylock.toml
```

or with uv:

```sh
uv sync
```

## Usage

Try running these commands and observe the output.

```sh
python main.py
```

```sh
python main.py seed=12
```

```sh
python main.py model=svm_linear seed=12 data=cmnist_no_pad
```

```sh
python main.py model=mlp_small model.layers=10
```

```sh
python main.py +experiment=compas_small_mlp_lower_lr_longer model.dropout=0.2
```

Notice that the following fails because of the negative dropout probability:

```sh
python main.py +experiment=compas_small_mlp_lower_lr_longer model.dropout=-0.2
```
