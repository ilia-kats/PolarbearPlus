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
            for _ in range(transforms - 1):
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
        predict_n_samples: Number of samples to take during prediction.
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
        n_feature_vars: int = 1,
        n_context_vars: int = 1,
        lr: float = 1e-3,
        predict_n_samples: int = 1000,
    ):
        super().__init__(sourcevae, destvae, lr, predict_n_samples)

        self._n_feature_vars = n_feature_vars
        self._n_context_vars = n_context_vars

        self._flow = CouplingNSFWithRotation(
            features=self._n_feature_vars * self._destvae.n_latent_dim,
            context=self._n_context_vars * self._sourcevae.n_latent_dim,
            transforms=n_layers,
            n_bins=n_bins,
        )
        self._latentstats = None
        self._needLatentStats = True
        self._n_latentstats = 0

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
        sourcelatent = self._sourcevae.encode_latent(sourcebatch, sourcebatch_idx)
        destlatent = self._destvae.encode_latent(destbatch, destbatch_idx)

        if self._needLatentStats:
            if self.training:
                if self._latentstats is None:
                    self._n_latentstats = len(destlatent)
                    self._latentstats = tuple(CountMeanVarianceStats() for _ in range(self._n_latentstats))
                for i, samples in enumerate(destlatent):
                    for sample in samples:
                        self._latentstats[i].update(sample)
            stats = tuple(latentstats.get() for latentstats in self._latentstats)
            latentmean = tuple(s["mean"] for s in stats)
            latentvar = tuple(s["variance"] for s in stats)
        else:
            latentmean = self._latentmean
            latentvar = self._latentvar

        return sourcelatent, destlatent, latentmean, latentvar

    @property
    def _latentmean(self):
        return tuple(getattr(self, f"_latentmean_{i}") for i in range(self._n_latentstats))

    @property
    def _latentvar(self):
        return tuple(getattr(self, f"_latentvar_{i}") for i in range(self._n_latentstats))

    def on_train_epoch_end(self):
        if self._needLatentStats:
            stats = tuple(latentstats.get() for latentstats in self._latentstats)
            for i, stat in enumerate(stats):
                self.register_buffer(f"_latentmean_{i}", stat["mean"])
                self.register_buffer(f"_latentvar_{i}", stat["variance"])
            self._needLatentStats = False
            self._latentstats = None

    def state_dict(self):
        state = {
            "flow": self._flow.state_dict(),
            "needLatentStats": self._needLatentStats,
            "n_latentstats": self._n_latentstats,
        }
        if not self._needLatentStats:
            state["latentmean"] = self._latentmean
            state["latentvar"] = self._latentvar
        return state

    def load_state_dict(self, state: dict, strict=True, assign=False):
        self._flow.load_state_dict(state["flow"], strict=strict, assign=assign)
        self._needLatentStats = state["needLatentStats"]
        self._n_latentstats = state["n_latentstats"]
        if strict and not self._needLatentStats and ("latentmean" not in state or "latentvar" not in state):
            raise RuntimeError("state_dict missing latent statistics.")
        for var in ("latentmean", "latentvar"):
            if var in state:
                for i, tens in enumerate(state[var]):
                    if assign or not hasattr(self, f"_{var}_{i}"):
                        self.register_buffer(f"_{var}_{i}", tens)
                    else:
                        getattr(self, f"_{var}_{i}").copy_(tens)


class INNTranslatorLatent(INNTranslatorBase):
    """INN-based translator network that translates between parameters of the variational distribution.

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
        predict_n_samples: Number of samples to take during prediction.
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
        predict_n_samples: int = 1000,
    ):
        super().__init__(
            sourcevae=sourcevae,
            destvae=destvae,
            n_layers=n_layers,
            n_bins=n_bins,
            block_n_layers=block_n_layers,
            block_layer_width=block_layer_width,
            block_residual=block_residual,
            n_feature_vars=2,
            n_context_vars=2,
            lr=lr,
            predict_n_samples=predict_n_samples,
        )

    def _step_impl(
        self,
        sourcebatch: torch.Tensor,
        sourcebatch_idx: torch.Tensor,
        destbatch: torch.Tensor,
        destbatch_idx: torch.Tensor,
    ):
        sourcelatent, destlatent, latentmean, latentvar = (
            torch.cat(tens, dim=-1)
            for tens in super()._step_impl(sourcebatch, sourcebatch_idx, destbatch, destbatch_idx)
        )
        destlatent = (destlatent - latentmean) / torch.sqrt(latentvar)
        return -self._flow(sourcelatent).log_prob(destlatent).sum() / (destbatch.shape[0] * self._destvae.n_latent_dim)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        sourcebatch, sourcebatch_idx, destbatch, destbatch_idx = batch
        sourcelatent, destlatent, latentmean, latentvar = super()._step_impl(
            sourcebatch, sourcebatch_idx, destbatch, destbatch_idx
        )

        return self._sample(
            torch.cat(sourcelatent, dim=-1),
            destbatch_idx,
            torch.cat(latentmean, dim=-1),
            torch.sqrt(torch.cat(latentvar, dim=-1)),
        )

    def _one_sample(self, sourcelatent, destbatch_idx, latentmean, latentstd):
        flowsample = self._flow(sourcelatent).sample() * latentstd + latentmean
        return self._destvae.decode_and_sample_normalized(torch.tensor_split(flowsample, 2, dim=-1), destbatch_idx)


class INNTranslatorSample(INNTranslatorBase):
    """INN-based translator network that translates between samples from the variational distribution.

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
        predict_n_samples: Number of samples to take during prediction.
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
        predict_n_samples: int = 1000,
    ):
        super().__init__(
            sourcevae=sourcevae,
            destvae=destvae,
            n_layers=n_layers,
            n_bins=n_bins,
            block_n_layers=block_n_layers,
            block_layer_width=block_layer_width,
            block_residual=block_residual,
            n_feature_vars=1,
            n_context_vars=1,
            lr=lr,
            predict_n_samples=predict_n_samples,
        )

    def _step_impl(
        self,
        sourcebatch: torch.Tensor,
        sourcebatch_idx: torch.Tensor,
        destbatch: torch.Tensor,
        destbatch_idx: torch.Tensor,
    ):
        sourcelatent, destlatent, latentmean, latentvar = super()._step_impl(
            sourcebatch, sourcebatch_idx, destbatch, destbatch_idx
        )

        # add empirical variance of the latent mean and latent variance to get total variance of a sample from the variational distribution
        totalstd = torch.sqrt(latentvar[0] + latentmean[1] ** 2)

        sourcesample = self._sourcevae.encode_and_sample_latent(sourcebatch, sourcebatch_idx)
        destsample = self._destvae.encode_and_sample_latent(destbatch, destbatch_idx)
        destsample = (destsample - latentmean[0]) / totalstd

        return -self._flow(sourcesample).log_prob(destsample).sum() / (destbatch.shape[0] * self._destvae.n_latent_dim)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        sourcebatch, sourcebatch_idx, destbatch, destbatch_idx = batch
        sourcelatent, destlatent, latentmean, latentvar = super()._step_impl(
            sourcebatch, sourcebatch_idx, destbatch, destbatch_idx
        )
        totalstd = torch.sqrt(latentvar[0] + latentmean[1] ** 2)
        return self._sample(sourcebatch, sourcebatch_idx, destbatch_idx, latentmean[0], totalstd)

    def _one_sample(self, sourcebatch, sourcebatch_idx, destbatch_idx, latentmean, totalstd):
        sourcesample = self._sourcevae.encode_and_sample_latent(sourcebatch, sourcebatch_idx)
        flowsample = self._flow(sourcesample).sample() * totalstd + latentmean
        return self._destvae.decode_normalized(flowsample, destbatch_idx)


class INNTranslatorDoubleSample(INNTranslatorBase):
    """INN-based translator network that uses two sampling steps.

    Uses a coupling neural spline flow. This translates a sample from the source
    variational distribution to parameters of the target variational distribution.
    Prediction first samples from the source variational distribution, then samples
    from the normalizing flow, and finally samples from the target variational distribution.

    Args:
        sourcevae: The encoder VAE.
        destvae: The decoder VAE.
        n_layers: Number of coupling blocks in the invertible neural network.
        n_bins: Number of spline bins.
        block_n_layers: Number of hidden layers in each coupling block.
        block_layer_width: Width of the hidden layers in a coupling block.
        block_residual: Whether to use residual layers in the coupling blocks.
        lr: Learning rate.
        predict_n_samples: Number of samples to take during prediction.
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
        predict_n_samples: int = 1000,
    ):
        super().__init__(
            sourcevae=sourcevae,
            destvae=destvae,
            n_layers=n_layers,
            n_bins=n_bins,
            block_n_layers=block_n_layers,
            block_layer_width=block_layer_width,
            block_residual=block_residual,
            n_feature_vars=2,
            n_context_vars=1,
            lr=lr,
            predict_n_samples=predict_n_samples,
        )

    def _step_impl(
        self,
        sourcebatch: torch.Tensor,
        sourcebatch_idx: torch.Tensor,
        destbatch: torch.Tensor,
        destbatch_idx: torch.Tensor,
    ):
        sourcelatent, destlatent, latentmean, latentvar = (
            torch.cat(tens, dim=-1)
            for tens in super()._step_impl(sourcebatch, sourcebatch_idx, destbatch, destbatch_idx)
        )

        sourcesample = self._sourcevae.encode_and_sample_latent(sourcebatch, sourcebatch_idx)
        destlatent = (destlatent - latentmean) / torch.sqrt(latentvar)
        return -self._flow(sourcesample).log_prob(destlatent).sum() / (destbatch.shape[0] * self._destvae.n_latent_dim)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        sourcebatch, sourcebatch_idx, destbatch, destbatch_idx = batch
        sourcelatent, destlatent, latentmean, latentvar = super()._step_impl(
            sourcebatch, sourcebatch_idx, destbatch, destbatch_idx
        )

        return self._sample(
            sourcebatch, sourcebatch_idx, destbatch_idx, torch.cat(latentmean, dim=-1), torch.sqrt(torch.cat(latentvar))
        )

    def _one_sample(self, sourcebatch, sourcebatch_idx, destbatch_idx, latentmean, latentstd):
        sourcesample = self._sourcevae.encode_and_sample_latent(sourcebatch, sourcebatch_idx)
        flowsample = self._flow(sourcesample).sample() * latentstd + latentmean
        return self._destvae.decode_and_sample_normalized(torch.tensor_split(flowsample, 2, dim=-1), destbatch_idx)
