import inspect
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from numbers import Real

import lightning as L
import pyro
import torch
from pyro import poutine

from .scalelatentmessenger import scale_latent

_InterpolationParams = namedtuple("_InterpolationParams", ["startval", "endval", "startepoch", "n_epochs"])


# http://www.phyast.pitt.edu/~micheles/python/metatype.html
class _VAEMeta(pyro.nn.module._PyroModuleMeta, ABCMeta):
    pass


class VAEBase(pyro.nn.PyroModule, metaclass=_VAEMeta):
    @abstractmethod
    def encode_latent(self, batch, batch_idx):
        pass

    @abstractmethod
    def encode_auxiliary(self, batch, batch_idx):
        pass

    @abstractmethod
    def reconstruction_guide(self, *latent_vars):
        pass

    @abstractmethod
    def model(self, batch, batch_idx):
        pass

    @abstractmethod
    def guide(self, batch, batch_idx):
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


class LightningVAEBase(L.LightningModule):
    def __init__(
        self, modulecls: type[VAEBase], lr: float, beta: float | tuple[float, float, int, int], *args, **kwargs
    ):
        super().__init__()
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

    def forward(self, batch):
        return self._vae.encode_latent(*batch)

    @property
    def encoder(self):
        return self._vae.encoder

    @property
    def decoder(self):
        return self._vae.decoder

    def encode_latent(self, batch, batch_idx):
        return self._vae.encode_latent(batch, batch_idx)

    def encode_auxiliary(self, batch, batch_idx):
        return self._vae.encode_auxiliary(batch, batch_idx)

    def decoded_likelihood(self, latent, observed, batch_idx):
        with self.device:
            guide_trace = poutine.trace(self._vae.reconstruction_guide).get_trace(*latent)
            model_trace = poutine.trace(
                poutine.replay(poutine.block(self._vae.model, expose_fn=lambda msg: msg["is_observed"]), guide_trace)
            ).get_trace(observed, batch_idx)
            return -model_trace.log_prob_sum()

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
