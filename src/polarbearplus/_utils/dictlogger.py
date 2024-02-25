import os
import pickle
from typing import Any

import pandas as pd
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only


class _History(dict):
    def __missing__(self, key):
        df = pd.Series(name=key)
        df.index.name = "step"
        self[key] = df
        return df

    def __getstate__(self):
        return dict(self)


class DictLogger(Logger):
    def __init__(self, name: str | None = None, version: str | None = None, save_dir: str = "."):
        self._name = name
        self._version = version
        self._savedir = save_dir
        self.hyperparams = {}
        self.history = _History()

    @property
    def name(self):
        return "" if self._name is None else self._name

    @property
    def version(self):
        return "0" if self._version is None else self._version

    @property
    def save_dir(self):
        return self._savedir

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any]):
        self.hyperparams = params

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int):
        for k, v in metrics.items():
            self.history[k].loc[step] = v

    def after_save_checkpoint(self, checkpoint_callback):
        filename = os.path.basename(os.path.splitext(checkpoint_callback.last_model_path)[0])
        filename = os.path.join(self._savedir, f"{filename}_{self.name}_{self.version}_log.pkl")
        with open(filename, "wb") as f:
            pickle.dump(
                {"history": self.history, "hyperparams": self.hyperparams, "name": self.name, "version": self.version},
                f,
            )
