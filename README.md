# Somewhat advanced example of using hydra

This example is without the `instantiate()` functionality.

It is mostly `main.py` that is important.

## Example output

```
$ python main.py
using SVM
svm_cfg.kernel=<Kern.RBF: 2>
svm_cfg.C=1.0

using Adult dataset
adult_cfg.drop_native=False

cfg.seed=42
cfg.use_wandb=False
cfg.data_pcnt=1.0

Config as flat dictionary:
{'model.kernel': <Kern.RBF: 2>, 'model.C': 1.0, 'dataset.drop_native': False, 'dataset.drop_discrete': False, 'seed': 42, 'data_pcnt': 1.0, 'use_wandb': False}
```
