import math
import os
from collections.abc import Mapping

import zarr
from lightning.pytorch.callbacks import BasePredictionWriter


class StatsWriter(BasePredictionWriter):
    def __init__(self, output_dir: str | os.PathLike, chunk_max_mb=20):
        super().__init__("batch")
        self._output_path = os.path.join(output_dir, "stats.zarr")
        self._chunk_max_mb = chunk_max_mb
        self._zarrstore = None

    def _write_array(self, value, parent: zarr.Group, name: str):
        if name not in parent:
            ssize = value.nbytes / 1024**2
            if ssize < -self._chunk_max_mb:
                chunkshape = value.shape
            else:
                ndim = min(
                    value.shape[0],
                    max(1, math.ceil(self._chunk_max_mb / ssize * value.shape[0])),
                )
                chunkshape = (ndim, *value.shape[1:])
            parent.create_dataset(name, data=value, chunks=chunkshape)
        else:
            parent[name].append(value, axis=0)

    def _write_dict(self, value, parent: zarr.Group, name: str):
        parent = parent.require_group(name, overwrite=False)
        for k, s in value.items():
            if isinstance(s, Mapping):
                self._write_dict(s, parent, k)
            else:
                self._write_array(s, parent, k)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if not isinstance(prediction, Mapping):
            raise ValueError(
                f"{self.__class__.__name__} only works with predictions that are (nested) dicts of arrays."
            )
        if self._zarrstore is None:
            self._zarrstore = zarr.group(self._output_path, overwrite=True)

        for k, s in prediction.items():
            if isinstance(s, dict):
                self._write_dict(s, self._zarrstore, k)
            else:
                self._write_array(s, self._zarrstore, k)

    def on_predict_end(self, trainer, pl_module):
        if self._zarrstore is not None:
            self._zarrstore.store.close()
