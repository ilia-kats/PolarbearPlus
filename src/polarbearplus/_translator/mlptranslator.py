import inspect
from abc import ABC, abstractmethod

import lightning as L
import torch

from .._utils import MLP
from .._vae import LightningVAEBase


class MLPTranslatorBase(L.LightningModule, ABC):
    """Base class for the MLP translator network.

    This uses a standard dense neural network to translate between data modalities.

    Args:
        sourcevae: The encoder VAE.
        destvae: The decoder VAE.
        n_layers: Number of hidden layers. If 0, only one linear layer without
            any activation will be used.
        layer_width: Width of the hidden layers.
        n_latent_vars: Number of latent variables in each latent dimension. For example,
            if the latent distribution is a Gaussian, it has two latent variables per
            dimension: mean and standard deviation.
        dropout: Dropout probability in the translator. Used only if `n_layers > 0`.
        lr: Learning rate.
    """

    def __init__(
        self,
        sourcevae: LightningVAEBase,
        destvae: LightningVAEBase,
        n_layers: int,
        layer_width: int,
        n_latent_vars: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self._sourcevae = sourcevae
        self._destvae = destvae
        self._sourcevae.freeze()
        self._destvae.freeze()

        if n_layers == 0:
            self._translator = torch.nn.Linear(
                n_latent_vars * self._sourcevae.n_latent_dim, n_latent_vars * self._destvae.n_latent_dim
            )
        elif n_layers > 0:
            self._translator = MLP(
                input_dim=n_latent_vars * self._sourcevae.n_latent_dim,
                output_dim=n_latent_vars * self._destvae.n_latent_dim,
                hidden_dim=layer_width,
                n_layers=n_layers,
                dropout=dropout,
            )
        else:
            raise ValueError("number of layers must be non-negative")
        self._lr = lr

        current_frame = inspect.currentframe()
        self.save_hyperparameters(ignore=["sourcevae", "destvae"], frame=current_frame.f_back)

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
        return torch.optim.Adam(self._translator.parameters(), lr=self._lr)

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
        decodedlikelihood = self._step_impl(sourcebatch, sourcebatch_idx, destbatch, destbatch_idx) / destbatch.numel()

        self.log(log_name, decodedlikelihood, on_step=True, on_epoch=True)
        return decodedlikelihood

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, dataloader_idx, "-training_likelihood")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, dataloader_idx, "-validation_likelihood")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, dataloader_idx, "-test_likelihood")


class MLPTranslatorLatent(MLPTranslatorBase):
    """Translator network that translates between parameters of the variational distributions.

    Args:
        sourcevae: The encoder VAE.
        destvae: The decoder VAE.
        n_layers: Number of hidden layers. If 0, only one linear layer without
            any activation will be used.
        layer_width: Width of the hidden layers.
        dropout: Dropout probability in the translator. Used only if `n_layers > 0`.
        lr: Learning rate.
    """

    def __init__(
        self,
        sourcevae: LightningVAEBase,
        destvae: LightningVAEBase,
        n_layers: int,
        layer_width: int,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ):
        super().__init__(sourcevae, destvae, n_layers, layer_width, 2, dropout, lr)

    def _step_impl(
        self,
        sourcebatch: torch.Tensor,
        sourcebatch_idx: torch.Tensor,
        destbatch: torch.Tensor,
        destbatch_idx: torch.Tensor,
    ) -> float:
        sourcelatent = self._sourcevae.encode_latent(sourcebatch, sourcebatch_idx)
        destauxiliary = self._destvae.encode_auxiliary(destbatch, destbatch_idx)

        translatedmean, translatedstdev = torch.tensor_split(
            self._translator(torch.cat(sourcelatent, dim=-1)), 2, dim=-1
        )
        translatedstdev = translatedstdev.exp()

        return self._destvae.decoded_likelihood(
            (translatedmean, translatedstdev, *destauxiliary), destbatch, destbatch_idx
        )


class MLPTranslatorSample(MLPTranslatorBase):
    """Translator network that translates between samples from the latent distribution.

    Args:
        sourcevae: The encoder VAE.
        destvae: The decoder VAE.
        n_layers: Number of hidden layers. If 0, only one linear layer without
            any activation will be used.
        layer_width: Width of the hidden layers.
        dropout: Dropout probability in the translator. Used only if `n_layers > 0`.
        lr: Learning rate.
    """

    def __init__(
        self,
        sourcevae: LightningVAEBase,
        destvae: LightningVAEBase,
        n_layers: int,
        layer_width: int,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ):
        super().__init__(sourcevae, destvae, n_layers, layer_width, 1, dropout, lr)

    def _step_impl(
        self,
        sourcebatch: torch.Tensor,
        sourcebatch_idx: torch.Tensor,
        destbatch: torch.Tensor,
        destbatch_idx: torch.Tensor,
    ) -> float:
        sourcelatent = self._sourcevae.encode_and_sample_latent(sourcebatch, sourcebatch_idx)
        translatedlatent = self._translator(sourcelatent)
        return self._destvae.decoded_sample_likelihood(translatedlatent, destbatch, destbatch_idx)
