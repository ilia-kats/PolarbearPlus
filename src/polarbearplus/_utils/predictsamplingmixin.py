from abc import ABC, abstractmethod

import numpy_onlinestats as npo
from tqdm.auto import tqdm


class PredictSamplingMixin(ABC):
    _predict_quantiles = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)

    def __init__(self, predict_n_samples: int = 1000, *args, **kwargs):
        self._n_predict_samples = predict_n_samples

        super().__init__(*args, **kwargs)

    @abstractmethod
    def _one_sample(self, *args, **kwargs):
        pass

    def _sample(self, *args, **kwargs):
        for _ in tqdm(range(self._n_predict_samples), leave=False, dynamic_ncols=True, desc="Sampling"):
            sample = self._one_sample(*args, **kwargs)
            self._stats.add(sample.cpu().numpy())
        return self._collect_predict_stats()

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0 or batch_idx >= self.trainer.num_predict_batches[dataloader_idx] - 1:
            self._stats = npo.NpOnlineStats()

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._stats.reset()

    def on_predict_end(self):
        del self._stats

    def _collect_predict_stats(self):
        stats = {"mean": self._stats.mean(), "var": self._stats.var()}
        for q in self._predict_quantiles:
            stats[f"q{q}"] = self._stats.quantile(q)

        return stats
