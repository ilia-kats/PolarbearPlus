import atexit
import os
import pickle
from pathlib import Path
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
    def __init__(
        self,
        save_dir: str | Path | None = None,
        filename: str = "log.pkl",
        name: str | None = None,
        version: str | None = None,
    ):
        super().__init__()
        self._name = name
        self._version = version
        self._savedir = save_dir
        self._filename = filename
        self.hyperparams = {}
        self.history = _History()
        self._needFullLog = False

        atexit.register(self._fullog)

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
        self.hyperparams = dict(params)

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int):
        self._needFullLog = True
        for k, v in metrics.items():
            self.history[k].loc[step] = v

    def _save(self):
        with open(os.path.join(self._savedir, self._filename), "wb") as f:
            pickle.dump(
                {
                    "history": dict(self.history),
                    "hyperparams": self.hyperparams,
                    "name": self.name,
                    "version": self.version,
                },
                f,
            )

    def _fullog(self):
        if self._needFullLog and self._savedir is not None:
            self._save()
            self._needFullLog = False

    def __del__(self):
        self._fullog()

    def after_save_checkpoint(self, checkpoint_callback):
        self._save()
