import lightning as L
import torch

from .._utils import MLP
from .._vae import VAEBase


class MLPTranslator(L.LightningModule):
    def __init__(
        self,
        sourcevae: VAEBase,
        destvae: VAEBase,
        n_layers: int,
        layer_width: int,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self._sourcevae = sourcevae.freeze()
        self._destvae = destvae.freeze()

        self._translator = MLP(
            input_dim=self._sourcevae.n_latent_dim,
            output_dim=self._destvae.n_latent_dim,
            hidden_dim=layer_width,
            n_layers=n_layers,
            dropout=dropout,
        )
        self._lr = lr

        self.save_hyperparameters(ignore=["sourcevae", "destvae"])

    def state_dict(self):
        return self._translator.state_dict()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._translator.load_state_dict(state_dict, strict=strict, assign=assign)

    def on_validation_model_eval(self):
        self._translator.eval()

    def on_validation_model_train(self):
        self._translator.train()

    def on_test_model_eval(self):
        self._translator.eval()

    def configure_optimizers(self):
        return torch.optim.adam(self._translator.parameters(), lr=self._lr)

    def _step(self, sourcebatch, sourcebatch_idx, destbatch, destbatch_idx, dataloader_idx=0, log_name="-likelihood"):
        sourcelatent = self._sourcevae.encode_latent(sourcebatch, sourcebatch_idx)
        destauxiliary = self._destvae.encode_auxiliary(destbatch, destbatch_idx)

        decodedlikelihood = self._destvae.decoded_likelihood((*sourcelatent, *destauxiliary), destbatch, destbatch_idx)
        self.log(log_name, decodedlikelihood, on_step=True, on_epoch=True)
        return decodedlikelihood

    def training_step(self, sourcebatch, sourcebatch_idx, destbatch, destbatch_idx, dataloader_idx=0):
        return self._step(
            sourcebatch, sourcebatch_idx, destbatch, destbatch_idx, dataloader_idx, "-training_likelihood"
        )

    def validation_step(self, sourcebatch, sourcebatch_idx, destbatch, destbatch_idx, dataloader_idx=0):
        return self._step(
            sourcebatch, sourcebatch_idx, destbatch, destbatch_idx, dataloader_idx, "-validation_likelihood"
        )

    def test_step(self, sourcebatch, sourcebatch_idx, destbatch, destbatch_idx, dataloader_idx=0):
        return self._step(sourcebatch, sourcebatch_idx, destbatch, destbatch_idx, dataloader_idx, "-test_likelihood")
