import lightning as L
import pyro
import torch

from .scalelatentmessenger import scale_latent


class VAEBase(L.LightningModule):
    def __init__(self, modulecls: type[pyro.nn.PyroModule], lr: float, beta: float, *args, **kwargs):
        super().__init__()
        self._vae = modulecls(*args, **kwargs)
        self._elbo = pyro.infer.TraceMeanField_ELBO()(
            scale_latent(self._vae.model, beta), scale_latent(self._vae.guide, beta)
        )
        self._lr = lr

        self.save_hyperparameters()
        self._old_module_local_params = None

    def setup(self, stage):
        self._old_module_local_params = pyro.settings.get("module_local_params")
        pyro.settings.set(module_local_params=True)

    def teardown(self, stage):
        pyro.settings.set(module_local_params=self._old_module_local_params)
        self._old_module_local_params = None

    def configure_optimizers(self):
        return torch.optim.Adam(self._vae.parameters(), lr=self._lr)

    def _step(self, batch, batch_idx, dataloader_idx=0, log_name="elbo"):
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
        return self._vae.encode(*batch)

    @property
    def encoder(self):
        return self._vae.encoder

    @property
    def decoder(self):
        return self._vae.decoder
