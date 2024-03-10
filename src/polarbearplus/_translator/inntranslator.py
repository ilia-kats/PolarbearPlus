import inspect
from abc import abstractmethod

import torch
from pyro.ops.streaming import CountMeanVarianceStats
from zuko.distributions import DiagNormal
from zuko.flows import Flow, MaskedAutoregressiveTransform, Unconditional
from zuko.transforms import MonotonicRQSTransform, RotationTransform

from .._vae import LightningVAEBase
from .base import TranslatorBase


class CouplingNSFWithRotation(Flow):
    """Coupling neural spline flow with a rotation matrix after each couplig block.

    Args:
        features: Number of data features.
        context: Number of context features.
        transforms: Number of coupling blocks.
        n_bins: Number of spline bins.
        block_n_layers: Number of hidden layers in each coupling block.
        block_layer_width: Width of the hidden layers in a coupling block.
        block_residual: Whether to use residual layers in the coupling blocks.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        n_bins: int = 8,
        block_n_layers: int = 2,
        block_layer_width: int = 64,
        block_residual: bool = False,
    ):
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
                        features=features,
                        context=context,
                        passes=2,
                        univariate=MonotonicRQSTransform,
                        shapes=shapes,
                        hidden_features=[block_layer_width] * block_n_layers,
                        activation=torch.nn.GELU,
                        residual=block_residual,
                    )
                )

        base = Unconditional(DiagNormal, torch.zeros(features), torch.ones(features), buffer=True)

        super().__init__(transform, base)


class INNTranslatorBase(TranslatorBase):
    """Base class for the INN translator network.

    Uses a coupling neural spline flow.

    Args:
        sourcevae: The encoder VAE.
        destvae: The decoder VAE.
        n_layers: Number of coupling blocks in the invertible neural network.
        n_bins: Number of spline bins.
        block_n_layers: Number of hidden layers in each coupling block.
        block_layer_width: Width of the hidden layers in a coupling block.
        block_residual: Whether to use residual layers in the coupling blocks.
        n_latent_vars: Number of latent variables in each latent dimension. For example,
            if the latent distribution is a Gaussian, it has two latent variables per
            dimension: mean and standard deviation.
        lr: Learning rate.
    """

    def __init__(
        self,
        sourcevae: LightningVAEBase,
        destvae: LightningVAEBase,
        n_layers: int,
        n_bins: int,
        block_n_layers: int = 2,
        block_layer_width: int = 64,
        block_residual: bool = False,
        n_latent_vars: int = 1,
        lr: float = 1e-3,
    ):
        super().__init__(sourcevae, destvae, lr)

        self._n_latent_vars = n_latent_vars

        self._flow = CouplingNSFWithRotation(
            features=self._n_latent_vars * self._destvae.n_latent_dim,
            context=self._n_latent_vars * self._sourcevae.n_latent_dim,
            transforms=n_layers,
            n_bins=n_bins,
        )
        self._latentstats = CountMeanVarianceStats()
        self._needLatentStats = True

        current_frame = inspect.currentframe()
        self.save_hyperparameters(ignore=["sourcevae", "destvae"], frame=current_frame.f_back)

    @property
    def translator(self):
        return self._flow

    @abstractmethod
    def _step_impl(
        self,
        sourcebatch: torch.Tensor,
        sourcebatch_idx: torch.Tensor,
        destbatch: torch.Tensor,
        destbatch_idx: torch.Tensor,
    ):
        sourcelatent = torch.cat(
            self._sourcevae.encode_latent(sourcebatch, sourcebatch_idx)[: self._n_latent_vars], dim=-1
        )
        destlatent = torch.cat(self._destvae.encode_latent(destbatch, destbatch_idx)[: self._n_latent_vars], dim=-1)

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
        return sourcelatent, destlatent, latentmean, latentstd

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


class INNTranslatorLatent(INNTranslatorBase):
    """INN-based translator network that thranslates between parameters of the variational distribution.

    Uses a coupling neural spline flow.

    Args:
        sourcevae: The encoder VAE.
        destvae: The decoder VAE.
        n_layers: Number of coupling blocks in the invertible neural network.
        n_bins: Number of spline bins.
        block_n_layers: Number of hidden layers in each coupling block.
        block_layer_width: Width of the hidden layers in a coupling block.
        block_residual: Whether to use residual layers in the coupling blocks.
        lr: Learning rate.
    """

    def __init__(
        self,
        sourcevae: LightningVAEBase,
        destvae: LightningVAEBase,
        n_layers: int,
        n_bins: int,
        block_n_layers: int = 2,
        block_layer_width: int = 64,
        block_residual: bool = False,
        lr: float = 1e-3,
    ):
        super().__init__(sourcevae, destvae, n_layers, n_bins, 2, lr)

    def _step_impl(
        self,
        sourcebatch: torch.Tensor,
        sourcebatch_idx: torch.Tensor,
        destbatch: torch.Tensor,
        destbatch_idx: torch.Tensor,
    ):
        sourcelatent, destlatent, latentmean, latentstd = super()._step_impl(
            sourcebatch, sourcebatch_idx, destbatch, destbatch_idx
        )
        return -self._flow(sourcelatent).log_prob(destlatent).sum() / (destbatch.shape[0] * self._destvae.n_latent_dim)


class INNTranslatorSample(INNTranslatorBase):
    """INN-based translator network that thranslates between samples from the variational distribution.

    Uses a coupling neural spline flow.

    Args:
        sourcevae: The encoder VAE.
        destvae: The decoder VAE.
        n_layers: Number of coupling blocks in the invertible neural network.
        n_bins: Number of spline bins.
        block_n_layers: Number of hidden layers in each coupling block.
        block_layer_width: Width of the hidden layers in a coupling block.
        block_residual: Whether to use residual layers in the coupling blocks.
        lr: Learning rate.
    """

    def __init__(
        self,
        sourcevae: LightningVAEBase,
        destvae: LightningVAEBase,
        n_layers: int,
        n_bins: int,
        block_n_layers: int = 2,
        block_layer_width: int = 64,
        block_residual: bool = False,
        lr: float = 1e-3,
    ):
        super().__init__(sourcevae, destvae, n_layers, n_bins, 1, lr)

    def _step_impl(
        self,
        sourcebatch: torch.Tensor,
        sourcebatch_idx: torch.Tensor,
        destbatch: torch.Tensor,
        destbatch_idx: torch.Tensor,
    ):
        sourcelatent, destlatent, latentmean, latentstd = super()._step_impl(
            sourcebatch, sourcebatch_idx, destbatch, destbatch_idx
        )

        sourcesample = self._sourcevae.encode_and_sample_latent(sourcebatch, sourcebatch_idx)
        destsample = self._destvae.encode_and_sample_latent(destbatch, destbatch_idx)
        destsample = (destsample - latentmean) / latentstd

        return -self._flow(sourcesample).log_prob(destsample).sum() / (destbatch.shape[0] * self._destvae.n_latent_dim)
