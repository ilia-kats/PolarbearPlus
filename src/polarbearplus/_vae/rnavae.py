import pyro
import pyro.distributions as dist
import torch
from numpy.typing import ArrayLike
from pyro.nn import PyroModule, PyroParam
from torch.distributions import constraints
from torch.nn import functional as F

from .mlp import MLP

class _RNAVAE(PyroModule):
    """
    Pyro implementation of a VAE for scRNAseq data, based on scVI.

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

        self.encoder = MLP(
            ngenes + nbatches, 2 * n_latent_dim + 2, encoder_layer_width, encoder_n_layers, encoder_dropout
        )

        if decoder_n_layers is None:
            decoder_n_layers = encoder_n_layers
        if decoder_layer_width is None:
            decoder_layer_width = encoder_layer_width
        if decoder_dropout is None:
            decoder_dropout = encoder_dropout
        self.decoder = MLP(n_latent_dim + nbatches, ngenes, decoder_layer_width, decoder_n_layers, decoder_dropout)

        self.theta = PyroParam(torch.randn((self.nbatches, self.ngenes)).exp(), constraint=constraints.positive)

    def model(self, expression_mat: ArrayLike | None, batch_idx: ArrayLike):
        """
        Generative model.

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
            rho_n = torch.softmax(self.decoder(z_n), -1)  # (ncells, ngenes)

            mu = l_n * rho_n
            theta_b = self.theta[batch_idx, :]  # (ncells, ngenes)
            with pyro.plate("genes", size=expression_mat.shape[1], dim=-1):
                pyro.sample(
                    "data", dist.GammaPoisson(concentration=1 / theta_b, rate=1 / (theta_b * mu)), obs=expression_mat
                )

    def guide(self, expression_mat, batch_idx):
        encoded = self.encoder(
            torch.cat((expression_mat, F.one_hot(batch_idx, self.n_batches).to(expression_mat.dtype)), -1)
        )
        latent_means = encoded[:, : self.n_latent_dim]
        latent_stdevs = encoded[:, self.n_latent_dim : -2].exp()
        sizefactor_means = encoded[:, -2]
        sizefactor_stdevs = encoded[:, -1].exp()

        with pyro.plate("cells", size=expression_mat.shape[0], dim=-2):
            pyro.sample("l_n", dist.LogNormal(sizefactor_means, sizefactor_stdevs))  # (ncells, 1)
            with pyro.plate("latent", size=self.n_latent_dim, dim=-1):
                pyro.sample("z_n", dist.Normal(latent_means, latent_stdevs))  # (ncells, nlatent)
