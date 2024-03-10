import torch
from pyro.ops.streaming import CountMeanVarianceStats
from zuko.distributions import DiagNormal
from zuko.flows import Flow, MaskedAutoregressiveTransform, Unconditional
from zuko.transforms import MonotonicRQSTransform, RotationTransform

from .._vae import LightningVAEBase
from .base import TranslatorBase


class CouplingNSFWithRotation(Flow):
    def __init__(self, features: int, context: int = 0, transforms: int = 3, n_bins: int = 8):
        shapes = [(n_bins,), (n_bins,), (n_bins - 1,)]
        transform = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                passes=2,
                univariate=lambda *args: MonotonicRQSTransform(
                    *args,
                ),
                shapes=shapes,
            )
        ]
        if transforms > 1:
            for _ in range(transforms):
                transform.append(Unconditional(RotationTransform, torch.randn((features, features)), buffer=True))
                transform.append(
                    MaskedAutoregressiveTransform(
                        features=features, context=context, passes=2, univariate=MonotonicRQSTransform, shapes=shapes
                    )
                )

        base = Unconditional(DiagNormal, torch.zeros(features), torch.ones(features), buffer=True)

        super().__init__(transform, base)


class INNTranslatorLatent(TranslatorBase):
    def __init__(
        self, sourcevae: LightningVAEBase, destvae: LightningVAEBase, n_layers: int, n_bins: int, lr: float = 1e-3
    ):
        super().__init__(sourcevae, destvae, lr)

        self._flow = CouplingNSFWithRotation(
            features=2 * self._destvae.n_latent_dim,
            context=2 * self._sourcevae.n_latent_dim,
            transforms=n_layers,
            n_bins=n_bins,
        )
        self._latentstats = CountMeanVarianceStats()
        self._needLatentStats = True

        self.save_hyperparameters(ignore=["sourcevae", "destvae"])

    @property
    def translator(self):
        return self._flow

    def _step_impl(
        self,
        sourcebatch: torch.Tensor,
        sourcebatch_idx: torch.Tensor,
        destbatch: torch.Tensor,
        destbatch_idx: torch.Tensor,
    ):
        sourcelatent = torch.cat(self._sourcevae.encode_latent(sourcebatch, sourcebatch_idx), dim=-1)
        destlatent = torch.cat(self._destvae.encode_latent(destbatch, destbatch_idx), dim=-1)

        if self._needLatentStats:
            if self.training:
                for sample in destlatent:
                    self._latentstats.update(sample)
            stats = self._latentstats.get()
            latentmean = stats["mean"]
            latentstd = torch.sqrt(stats["variance"])
        else:
            latentmean = self._latentmean
            latentstd = self._latentstd

        destlatent = (destlatent - latentmean) / latentstd

        return -self._flow(sourcelatent).log_prob(destlatent).sum() / (destbatch.shape[0] * self._destvae.n_latent_dim)

    def on_train_epoch_end(self):
        if self._needLatentStats:
            stats = self._latentstats.get()
            self.register_buffer("_latentmean", stats["mean"])
            self.register_buffer("_latentstd", torch.sqrt(stats["variance"]))
            self._needLatentStats = False
            self._latentstats = None

    def state_dict(self):
        state = {"flow": self._flow.state_dict(), "needLatentStats": self._needLatentStats}
        if not self._needLatentStats:
            state["latentmean"] = self._latentmean
            state["latentstd"] = self._latentstd
        return state

    def load_state_dict(self, state: dict, strict=True, assign=False):
        self._flow.load_state_dict(state["flow"], strict=strict, assign=assign)
        self._needLatentStats = state["needLatentStats"]
        if strict and not self._needLatentStats and ("latentmean" not in state or "latentstd" not in state):
            raise RuntimeError("state_dict missing latent statistics.")
        for var in ("latentmean", "latentstd"):
            if var in state:
                if assign or not hasattr(self, f"_{var}"):
                    self.register_buffer(f"_{var}", state[var])
                else:
                    getattr(self, f"_{var}").copy_(state["latentmean"])
