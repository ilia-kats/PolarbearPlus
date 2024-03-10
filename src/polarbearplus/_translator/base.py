from abc import ABC, abstractmethod

import lightning as L
import torch

from .._vae import LightningVAEBase


class TranslatorBase(L.LightningModule, ABC):
    """Abstract base class for between-modality translators.

    Args:
        sourcevae: The encoder VAE.
        destvae: The decoder VAE.
        lr: Learning rate.
    """

    def __init__(self, sourcevae: LightningVAEBase, destvae: LightningVAEBase, lr: float = 1e-3):
        super().__init__()
        self._sourcevae = sourcevae
        self._destvae = destvae
        self._sourcevae.freeze()
        self._destvae.freeze()

        self._lr = lr

    @property
    @abstractmethod
    def translator(self):
        pass

    def on_validation_model_eval(self):
        self.translator.eval()

    def on_validation_model_train(self):
        self.translator.train()

    def on_test_model_eval(self):
        self.translator.eval()

    def configure_optimizers(self):
        return torch.optim.Adam(self.translator.parameters(), lr=self._lr)

    @abstractmethod
    def _step_impl(
        self,
        sourcebatch: torch.Tensor,
        sourcebatch_idx: torch.Tensor,
        destbatch: torch.Tensor,
        destbatch_idx: torch.Tensor,
    ) -> float:
        """Calculate loss for translation between modalities.

        Args:
            sourcebatch: Source data matrix.
            sourcebatch_idx: Index of the experimental batch for each cell in the source modality.
            destbatch: Target data matrix.
            destbatch_idx: Index of the experimental batch for each cell in the target modality.

        Returns:
            The negative log-likelihood of `destbatch` if the target decoder is applied using latent
                variables translated from the encoding of the source modality.
        """
        pass

    def _step(self, batch, batch_idx, dataloader_idx=0, log_name="-likelihood"):
        sourcebatch, sourcebatch_idx, destbatch, destbatch_idx = batch
        decodedlikelihood = self._step_impl(sourcebatch, sourcebatch_idx, destbatch, destbatch_idx)

        self.log(log_name, decodedlikelihood, on_step=True, on_epoch=True)
        return decodedlikelihood

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, dataloader_idx, "-training_likelihood")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, dataloader_idx, "-validation_likelihood")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, dataloader_idx, "-test_likelihood")
