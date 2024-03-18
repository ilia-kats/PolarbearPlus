import inspect
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from numbers import Real

import lightning as L
import pyro
import torch
from pyro import poutine

from .._utils import PredictSamplingMixin
from .scalelatentmessenger import scale_latent

_InterpolationParams = namedtuple("_InterpolationParams", ["startval", "endval", "startepoch", "n_epochs"])


# http://www.phyast.pitt.edu/~micheles/python/metatype.html
class _VAEMeta(pyro.nn.module._PyroModuleMeta, ABCMeta):
    pass


class VAEBase(pyro.nn.PyroModule, metaclass=_VAEMeta):
    @property
    @abstractmethod
    def latent_name(self):
        pass

    @property
    @abstractmethod
    def observed_name(self):
        pass

    @property
    @abstractmethod
    def normalized_name(self):
        pass

    @abstractmethod
    def encode_latent(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the encoder network.

        Args:
            batch: Data matrix.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            Parameters of the variational distribution.
        """
        pass

    def encode_and_sample_latent(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Apply the encoder network and return a sample from the variational distribution.

        Args:
            batch: Data matrix.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            A sample of the latent variables.
        """
        trace = poutine.trace(
            poutine.block(self.guide, expose_fn=lambda msg: msg["name"] == self.latent_name)
        ).get_trace(batch, batch_idx)
        if len(trace.nodes) != 3:  # _INPUT, latent_name, _RETURN
            raise RuntimeError(f"Unexpected number of nodes in guide trace: {len(trace.nodes)}")
        return trace.nodes[self.latent_name]["value"]

    def decoded_likelihood(
        self,
        latent: tuple[torch.Tensor],
        auxiliary: tuple[torch.Tensor],
        observed: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> float:
        """Negative log-likelihood of observed data given the latent embedding.

        Args:
            latent: The latent embedding, usually the parameters of the variational posterior.
            auxiliary: The auxiliary variables.
            observed: Data matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        guide_trace = poutine.trace(self.reconstruction_guide).get_trace(*latent, *auxiliary)
        model_trace = poutine.trace(
            poutine.block(poutine.replay(self.model, guide_trace), expose_fn=lambda msg: msg["is_observed"])
        ).get_trace(observed, batch_idx)
        return -model_trace.log_prob_sum()

    def decode(self, latent: torch.Tensor, auxiliary: tuple[torch.Tensor], batch_idx: torch.Tensor) -> torch.Tensor:
        """Sample from the variational posterior given its parameters.

        Args:
            latent: The latent embedding, usually the parameters of the variational posterior.
            auxiliary: Auxiliary variables.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            A sample of observed data.
        """
        guide_trace = poutine.trace(self.reconstruction_guide).get_trace(*latent, *auxiliary)
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(None, batch_idx)
        return model_trace.nodes[self.observed_name]["value"]

    def decode_normalized(self, latent_sample: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Decode a sample from the variational posterior.

        Args:
            latent_sample: A sample from the variational posterior.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            Normalized (corrected for sequencing depth) decoded data.
        """
        model_trace = poutine.trace(poutine.condition(self.model, {self.latent_name: latent_sample})).get_trace(
            None, batch_idx
        )
        return model_trace.nodes[self.normalized_name]["value"]

    def _sample_guide_trace(self, latent_sample: torch.Tensor, auxiliary: tuple[torch.Tensor]) -> pyro.poutine.Trace:
        guide_trace = poutine.trace(self.sample_guide).get_trace(latent_sample, *auxiliary)
        guide_trace.nodes[self.latent_name][
            "value"
        ] = latent_sample  # we can't use poutine.condition here because it sets is_observed = True, which won't work with replay
        return guide_trace

    def decoded_sample_likelihood(
        self, latent_sample: torch.Tensor, observed: torch.Tensor, batch_idx: torch.Tensor
    ) -> float:
        """Negative log-likelihood of observed data given a sample from the variational distribution.

        Args:
            latent_sample: A sample from the variational distribution.
            observed: Data matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        auxiliary = self.encode_auxiliary(observed, batch_idx)
        guide_trace = self._sample_guide_trace(latent_sample, auxiliary)
        model_trace = poutine.trace(
            poutine.block(poutine.replay(self.model, guide_trace), expose_fn=lambda msg: msg["is_observed"])
        ).get_trace(observed, batch_idx)
        return -model_trace.log_prob_sum()

    def decode_and_sample(
        self, latent_sample: torch.Tensor, auxiliary: tuple[torch.Tensor], batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Sample data given a sample from the variational distribution.

        Args:
            latent_sample: A sample from the variational distribution.
            auxiliary: Auxiliary variables.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            A sample of observed data
        """
        guide_trace = self._sample_guide_trace(latent_sample, auxiliary)
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(None, batch_idx)
        return model_trace.nodes[self.observed_name]["value"]

    def decode_and_sample_normalized(self, latent: tuple[torch.Tensor], batch_idx: torch.Tensor) -> torch.Tensor:
        """Sample from the variational distribution given its parameters.

        Args:
            latent: The latent embedding, usually the parameters of the variational posterior.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            Normalized (corrected for sequencing depth) decoded data.
        """
        guide_trace = poutine.trace(self.normalized_guide).get_trace(*latent)
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(None, batch_idx)
        return model_trace.nodes[self.normalized_name]["value"]

    @abstractmethod
    def encode_auxiliary(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> tuple[torch.Tensor]:
        """Encode auxiliary variables.

        Auxiliary variables are variables that are not need for the latent embedding,
        but are required for correct decoding, e.g. to scale the decoded data.

        Args:
            batch: Data matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        pass

    @abstractmethod
    def reconstruction_guide(self, *latent_vars):
        """Variational posterior given the encoded data.

        This can be used for translating from another modality.

        Args:
            *latent_vars: Latent and auxiliary variables>
        """
        pass

    @abstractmethod
    def sample_guide(self, sample: torch.Tensor, *auxiliary_vars):
        """Variational posterior given a sample from the variatonal distribution.

        This can be used for translating from another modality.

        Args:
            sample: A sample from the variational distribution.
            *auxiliary_vars: Auxiliary variables.
        """
        pass

    @abstractmethod
    def normalized_guide(self, *latent_vars):
        """Variational posterior given the encoded data, without auxiliary variables.

        This is useful to draw normalized (corrected for sequencing depth) samples from
        the variational posterior. In this case the auxiliary variables will be set to
        a constant value, the unnormalized samples will be incorrect.

        Args:
            *latent_vars: Latent variables.
        """

    @abstractmethod
    def model(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        """Generative model."""
        pass

    @abstractmethod
    def guide(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        """Variational posterior."""
        pass

    @property
    @abstractmethod
    def encoder(self):
        pass

    @property
    @abstractmethod
    def decoder(self):
        pass

    @property
    @abstractmethod
    def n_latent_dim(self):
        pass


class LightningVAEBase(PredictSamplingMixin, L.LightningModule):
    def __init__(
        self,
        modulecls: type[VAEBase],
        lr: float,
        beta: float | tuple[float, float, int, int],
        predict_n_samples: int = 1000,
        *args,
        **kwargs,
    ):
        super().__init__(predict_n_samples=predict_n_samples)
        self._vae = modulecls(*args, **kwargs)
        self._lr = lr

        self._beta = beta if isinstance(beta, Real) else _InterpolationParams(*beta)

        current_frame = inspect.currentframe()
        self.save_hyperparameters(frame=current_frame.f_back)
        self._old_module_local_params = None

    def setup(self, stage):
        self._old_module_local_params = pyro.settings.get("module_local_params")
        pyro.settings.set(module_local_params=True)

    def teardown(self, stage):
        pyro.settings.set(module_local_params=self._old_module_local_params)
        self._old_module_local_params = None

    def configure_optimizers(self):
        return torch.optim.Adam(self._vae.parameters(), lr=self._lr)

    def _step(self, batch, batch_idx, dataloader_idx=0, log_name="-elbo"):
        with self.device:
            elbo = self._elbo(*batch) / (batch[0].shape[0] * batch[0].shape[1])
        self.log(log_name, elbo, on_step=True, on_epoch=True)
        return elbo

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, dataloader_idx, "-training_elbo")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, dataloader_idx, "-validation_elbo")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, dataloader_idx, "-test_elbo")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        latent = self.encode_latent(*batch)
        auxiliary = self.encode_auxiliary(*batch)

        samples = self._sample(latent, auxiliary, batch[1])
        ret = {f"latent_{i}": arr.cpu().numpy() for i, arr in enumerate(latent)} | {
            f"auxiliary_{i}": arr.cpu().numpy() for i, arr in enumerate(auxiliary)
        }
        ret["reconstruction_stats"] = samples
        return ret

    def _one_sample(self, latent, auxiliary, batch_idx):
        return self.decode(latent, auxiliary, batch_idx)

    def forward(self, batch):
        return self._vae.encode_latent(*batch)

    @property
    def encoder(self):
        return self._vae.encoder

    @property
    def decoder(self):
        return self._vae.decoder

    def encode_latent(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the encoder network.

        Args:
            batch: Data matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        return self._vae.encode_latent(batch, batch_idx)

    def encode_auxiliary(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> tuple[torch.Tensor]:
        """Encode auxiliary variables.

        Auxiliary variables are variables that are not need for the latent embedding,
        but are required for correct decoding, e.g. to scale the decoded data.

        Args:
            batch: Data matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        return self._vae.encode_auxiliary(batch, batch_idx)

    def encode_and_sample_latent(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Apply the encoder network and return a sample from the variational distribution.

        Args:
            batch: Data matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        return self._vae.encode_and_sample_latent(batch, batch_idx)

    def decoded_likelihood(
        self,
        latent: tuple[torch.Tensor],
        auxiliary: tuple[torch.Tensor],
        observed: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> float:
        """Negative log-likelihood of observed data given the latent embedding.

        Args:
            latent: The latent embedding, usually the parameters of the variational posterior.
            auxiliary: The auxiliary variables.
            observed: Data matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        return self._vae.decoded_likelihood(latent, auxiliary, observed, batch_idx)

    def decode(
        self, latent: tuple[torch.Tensor], auxiliary: tuple[torch.Tensor], batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Sample from the variational posterior given its parameters.

        Args:
            latent: The latent embedding, usually the parameters of the variational posterior.
            auxiliary: Auxiliary variables.
            batch_idx: Index of the experimental batch for each cell.
        """
        return self._vae.decode(latent, auxiliary, batch_idx)

    def decode_normalized(self, latent_sample: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Decode a sample from the variational posterior.

        Args:
            latent_sample: A sample from the variational posterior.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            Normalized (corrected for sequencing depth) decoded data.
        """
        return self._vae.decode_normalized(latent_sample, batch_idx)

    def decoded_sample_likelihood(
        self, latent_sample: torch.Tensor, observed: torch.Tensor, batch_idx: torch.Tensor
    ) -> float:
        """Negative log-likelihood of observed data given a sample from the variational distribution.

        Args:
            latent_sample: A sample from the variational distribution.
            observed: Data matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        return self._vae.decoded_sample_likelihood(latent_sample, observed, batch_idx)

    def decode_and_sample(
        self, latent_sample: torch.Tensor, auxiliary: tuple[torch.Tensor], batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Sample data given a sample from the variational distribution.

        Args:
            latent_sample: A sample from the variational distribution.
            auxiliary: Auxiliary variables.
            batch_idx: Index of the experimental batch for each cell.
        """
        return self._vae.decode_and_sample(latent_sample, auxiliary, batch_idx)

    def decode_and_sample_normalized(self, latent: tuple[torch.Tensor], batch_idx: torch.Tensor) -> torch.Tensor:
        """Sample from the variational distribution given its parameters.

        Args:
            latent: The latent embedding, usually the parameters of the variational posterior.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            Normalized (corrected for sequencing depth) decoded data.
        """
        return self._vae.decode_and_sample_normalized(latent, batch_idx)

    @property
    def n_latent_dim(self):
        return self._vae.n_latent_dim

    def on_train_epoch_start(self):
        beta = self._linear_interpolate(self._beta)
        self._elbo = pyro.infer.TraceMeanField_ELBO()(
            scale_latent(self._vae.model, beta), scale_latent(self._vae.guide, beta)
        )
        self.log("beta", beta)

    def on_validation_epoch_start(self):
        self._elbo = pyro.infer.TraceMeanField_ELBO()(self._vae.model, self._vae.guide)

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def on_predict_epoch_start(self):
        self.on_validation_epoch_start()

    def _linear_interpolate(self, val: _InterpolationParams | float):
        if isinstance(val, Real):
            return val
        if self.current_epoch >= val.startepoch + val.n_epochs:
            return val.endval
        elif self.current_epoch < val.startepoch:
            return val.startval
        else:
            return val.startval + (self.current_epoch - val.startepoch + 1) / val.n_epochs * (val.endval - val.startval)
