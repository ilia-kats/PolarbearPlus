import pyro
import pyro.distributions as dist
import torch
from lightning import pytorch as pl
from numpy.typing import ArrayLike
from pyro.nn import PyroModule, PyroParam
from torch.distributions import constraints
from torch.nn import functional as F

from .mlp import MLP
from .scalelatentmessenger import scale_latent


class _RNAVAE(PyroModule):
    """Pyro implementation of a VAE for scRNAseq data, based on scVI.

    Args:
        ngenes: Number of genes.
        nbatches: Number of experimental batches.
        logbatchmeans: Vector of means of log library sizes for all batches.
        logbatchvars: Vector of variances of log library sizes for all batches.
        n_latent_dim: Number of latent dimensions.
        encoder_n_layers: Number of hidden layers in the encoder.
        encoder_layer_width: Width of the hidden layers in the encoder.
        encoder_dropout: Dropout probability in the encoder.
        decoder_n_layers: Number of hidden layers in the decoder.
        decoder_layer_width: Width of the hidden layers in the decoder.
        decoder_dropout: Dropout probability in the decoder.
    """

    def __init__(
        self,
        ngenes: int,
        nbatches: int,
        logbatchmeans: ArrayLike,
        logbatchvars: ArrayLike,
        n_latent_dim: int,
        encoder_n_layers: int,
        encoder_layer_width: int,
        encoder_dropout: float = 0.1,
        decoder_n_layers: int | None = None,
        decoder_layer_width: int | None = None,
        decoder_dropout: float | None = None,
    ):
        super().__init__()

        self.ngenes = ngenes
        self.nbatches = nbatches
        self.n_latent_dim = n_latent_dim
        self.register_buffer("logbatchmeans", torch.as_tensor(logbatchmeans))
        self.register_buffer("logbatchstds", torch.sqrt(torch.as_tensor(logbatchvars)))

        self.register_buffer("zero", torch.as_tensor(0.0))
        self.register_buffer("one", torch.as_tensor(1.0))

        self.encoder = MLP(ngenes + nbatches, 2 * n_latent_dim, encoder_layer_width, encoder_n_layers, encoder_dropout)

        if decoder_n_layers is None:
            decoder_n_layers = encoder_n_layers
        if decoder_layer_width is None:
            decoder_layer_width = encoder_layer_width
        if decoder_dropout is None:
            decoder_dropout = encoder_dropout
        self.decoder = MLP(
            n_latent_dim + nbatches,
            ngenes,
            decoder_layer_width,
            decoder_n_layers,
            decoder_dropout,
            last_activation=torch.nn.Softmax(dim=-1),
        )
        self._l_encoder = MLP(ngenes + nbatches, 2, encoder_layer_width, 1, encoder_dropout)

        self.theta = PyroParam(torch.randn((self.nbatches, self.ngenes)).exp(), constraint=constraints.positive)

    def get_extra_state(self):
        return {"ngenes": self.ngenes, "nbatches": self.nbatches, "n_latent_dim": self.n_latent_dim}

    def set_extra_state(self, state):
        self.ngenes = state["ngenes"]
        self.nbatches = state["nbatches"]
        self.n_latent_dim = state["n_latent_dim"]

    def model(self, expression_mat: ArrayLike | None, batch_idx: ArrayLike):
        """Generative model.

        Args:
            expression_mat: Cells x genes expression matrix
            batch_idx: Index of the experimental batch for each cell
        """
        with pyro.plate("cells", size=expression_mat.shape[0], dim=-2):
            l_n = pyro.sample(
                "l_n", dist.LogNormal(self.logbatchmeans[batch_idx], self.logbatchstds[batch_idx])
            )  # (ncells, 1)
            with pyro.plate("latent", size=self.n_latent_dim, dim=-1):
                z_n = pyro.sample("z_n", dist.Normal(self.zero, self.one))  # (ncells, nlatent)

            z_n = torch.cat((z_n, F.one_hot(batch_idx, self.n_batches).to(z_n.dtype)), -1)
            rho_n = self.decoder(z_n)  # (ncells, ngenes)

            mu = l_n * rho_n
            theta_b = self.theta[batch_idx, :]  # (ncells, ngenes)
            with pyro.plate("genes", size=expression_mat.shape[1], dim=-1):
                pyro.sample(
                    "data", dist.GammaPoisson(concentration=1 / theta_b, rate=1 / (theta_b * mu)), obs=expression_mat
                )

    def encode(self, expression_mat: ArrayLike, batch_idx: ArrayLike, latentonly: bool = True):
        """Apply the encoder network.

        Args:
            expression_mat: Cells x genes expression matrix.
            batch_idx: Index of the experimental batch for each cell.
            latentonly: If True, encode only the latent variables. If False, also encode the size factors.

        Returns:
            If `latentonly=True`, a two-element tuple with means and standard deviations. Otherwise, a four-element
            tuple with means and standard deviations of the latent Normal, followed by locations and scales of the
            LogNormal size factor distribution.
        """
        concat = torch.cat((expression_mat, F.one_hot(batch_idx, self.n_batches).to(expression_mat.dtype)), -1)
        encoded = self.encoder(concat)
        latent_means = encoded[:, : self.n_latent_dim]
        latent_stdevs = encoded[:, self.n_latent_dim :].exp()

        if not latentonly:
            l_encoded = self._l_encoder(concat)
            sizefactor_means = l_encoded[:, 0]
            sizefactor_stdevs = l_encoded[:, 1].exp()
            return latent_means, latent_stdevs, sizefactor_means, sizefactor_stdevs
        else:
            return latent_means, latent_stdevs

    def guide(self, expression_mat, batch_idx):
        """Variational posterior.

        Args:
            expression_mat: Cells x genes expression matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        latent_means, latent_stdevs, sizefactor_means, sizefactor_stdevs = self.encode(
            expression_mat, batch_idx, latentonly=False
        )
        with pyro.plate("cells", size=expression_mat.shape[0], dim=-2):
            pyro.sample("l_n", dist.LogNormal(sizefactor_means, sizefactor_stdevs))  # (ncells, 1)
            with pyro.plate("latent", size=self.n_latent_dim, dim=-1):
                pyro.sample("z_n", dist.Normal(latent_means, latent_stdevs))  # (ncells, nlatent)


class RNAVAE(pl.LightningModule):
    """A beta-VAE for scRNAseq data.

    Args:
        ngenes: Number of genes.
        nbatches: Number of experimental batches.
        logbatchmeans: Vector of means of log library sizes for all batches.
        logbatchvars: Vector of variances of log library sizes for all batches.
        n_latent_dim: Number of latent dimensions.
        encoder_n_layers: Number of hidden layers in the encoder.
        encoder_layer_width: Width of the hidden layers in the encoder.
        encoder_dropout: Dropout probability in the encoder.
        decoder_n_layers: Number of hidden layers in the decoder.
        decoder_layer_width: Width of the hidden layers in the decoder.
        decoder_dropout: Dropout probability in the decoder.
        lr: Learning rate.
        beta: The scaling factor for the KL divergence.
    """

    def __init__(
        self,
        ngenes: int,
        nbatches: int,
        logbatchmeans: ArrayLike,
        logbatchvars: ArrayLike,
        n_latent_dim: int,
        encoder_n_layers: int,
        encoder_layer_width: int,
        encoder_dropout: float = 0.1,
        decoder_n_layers: int | None = None,
        decoder_layer_width: int | None = None,
        decoder_dropout: float | None = None,
        lr: float = 1e-3,
        beta: float = 1,
    ):
        self._vae = _RNAVAE(
            ngenes=ngenes,
            nbatches=nbatches,
            logbatchmeans=logbatchmeans,
            logbatchvars=logbatchvars,
            n_latent_dim=n_latent_dim,
            encoder_n_layers=encoder_n_layers,
            encoder_layer_width=encoder_layer_width,
            encoder_dropout=encoder_dropout,
            decoder_n_layers=decoder_n_layers,
            decoder_layer_width=decoder_layer_width,
            decoder_dropout=decoder_dropout,
        )
        self._elbo = pyro.infer.TraceMeanField_ELBO()(
            scale_latent(self._vae.model, beta), scale_latent(self._vae.guide, beta)
        )
        self._lr = lr

        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self._elbo.parameters(), lr=self._lr)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        elbo = self._elbo(*batch)
        self.log("-elbo", elbo, on_step=True, on_epoch=True)
        return elbo

    def forward(self, batch):
        return self._vae.encode(*batch)

    @property
    def encoder(self):
        return self._vae.encoder

    @property
    def decoder(self):
        return self._vae.decoder
