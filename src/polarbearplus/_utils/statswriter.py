import math
import os

import zarr
from lightning.pytorch.callbacks import BasePredictionWriter


class StatsWriter(BasePredictionWriter):
    def __init__(self, output_dir: str | os.PathLike, chunk_max_mp=20):
        super().__init__("batch")
        self._output_path = os.path.join(output_dir, "stats.zarr")
        self._chunk_max_mb = 20

        self._zarrstore = zarr.group(self._output_path, overwrite=True)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if not isinstance(prediction, dict):
            raise ValueError(f"{self.__class__.__name__} only works with predictions that are dicts of arrays.")
        for k, s in prediction.items():
            if k not in self._zarrstore:
                ssize = s.nbytes / 1024**2
                if ssize < -self._chunk_max_mb:
                    chunkshape = s.shape
                else:
                    ndim = min(
                        s.shape[0],
                        max(1, math.ceil(self._chunk_max_mb / ssize * s.shape[0])),
                    )
                    chunkshape = (ndim, *s.shape[1:])
                self._zarrstore.create_dataset(k, data=s, chunks=chunkshape)
            else:
                self._zarrstore[k].append(s, axis=0)
