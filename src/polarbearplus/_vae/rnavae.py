from collections.abc import Callable

import pyro
import pyro.distributions as dist
import torch
from numpy.typing import ArrayLike
from pyro.nn import PyroParam
from torch import nn
from torch.distributions import constraints
from torch.nn import functional as F

from .._utils import MLP
from .vaebase import LightningVAEBase, VAEBase


class _RNAVAE(VAEBase):
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
        eps: float = 1e-3,
    ):
        super().__init__()

        self.ngenes = ngenes
        self.nbatches = nbatches
        self._n_latent_dim = n_latent_dim
        self.register_buffer("logbatchmeans", torch.as_tensor(logbatchmeans))
        self.register_buffer("logbatchstds", torch.sqrt(torch.as_tensor(logbatchvars)))

        self.register_buffer("zero", torch.as_tensor(0.0))
        self.register_buffer("one", torch.as_tensor(1.0))
        self.register_buffer("eps", torch.as_tensor(eps))

        self._encoder = MLP(ngenes + nbatches, 2 * n_latent_dim, encoder_layer_width, encoder_n_layers, encoder_dropout)

        if decoder_n_layers is None:
            decoder_n_layers = encoder_n_layers
        if decoder_layer_width is None:
            decoder_layer_width = encoder_layer_width
        if decoder_dropout is None:
            decoder_dropout = encoder_dropout
        self._decoder = MLP(
            n_latent_dim + nbatches,
            ngenes,
            decoder_layer_width,
            decoder_n_layers,
            decoder_dropout,
            last_activation=nn.Softmax(dim=-1),
        )
        self._l_encoder = MLP(ngenes + nbatches, 2, encoder_layer_width, 1, encoder_dropout)
        torch.nn.init.zeros_(self._l_encoder[-1].weight)

        self.theta = PyroParam(
            torch.randn((self.nbatches, self.ngenes)).exp(), constraint=constraints.interval(0.0, 1 / eps)
        )

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def n_latent_dim(self):
        return self._n_latent_dim

    @property
    def latent_name(self):
        return "z_n"

    @property
    def observed_name(self):
        return "data"

    @property
    def normalized_name(self):
        return "rho_n"

    def get_extra_state(self):
        return {"ngenes": self.ngenes, "nbatches": self.nbatches, "n_latent_dim": self._n_latent_dim}

    def set_extra_state(self, state):
        self.ngenes = state["ngenes"]
        self.nbatches = state["nbatches"]
        self._n_latent_dim = state["n_latent_dim"]

    def model(self, expression_mat: torch.Tensor | None, batch_idx: torch.Tensor):
        """Generative model.

        Args:
            expression_mat: Cells x genes expression matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        with pyro.plate("cells", size=batch_idx.shape[0], dim=-2):
            l_n = pyro.sample(
                "l_n", dist.LogNormal(self.logbatchmeans[batch_idx, None], self.logbatchstds[batch_idx, None])
            )  # (ncells, 1)
            with pyro.plate("latent", size=self.n_latent_dim, dim=-1):
                z_n = pyro.sample("z_n", dist.Normal(self.zero, self.one))  # (ncells, nlatent)

            z_n = torch.cat((z_n, F.one_hot(batch_idx, self.nbatches).to(z_n.dtype)), -1)
            rho_n = pyro.deterministic("rho_n", self._decoder(z_n))  # (ncells, ngenes)

            mu = pyro.deterministic("mu", l_n * rho_n)
            theta_b = self.theta[batch_idx, :]  # (ncells, ngenes)
            with pyro.plate("genes", size=self.ngenes, dim=-1):
                pyro.sample(
                    self.observed_name,
                    dist.GammaPoisson(concentration=1 / (theta_b + self.eps), rate=1 / (theta_b * mu + self.eps)),
                    obs=expression_mat,
                )

    def _encode_latent(self, concat: torch.Tensor):
        encoded = self._encoder(concat)
        latent_means = encoded[:, : self.n_latent_dim]
        latent_stdevs = encoded[:, self.n_latent_dim :].exp()

        return latent_means, latent_stdevs

    def encode_latent(self, expression_mat: torch.Tensor, batch_idx: torch.Tensor):
        """Apply the encoder network.

        Args:
            expression_mat: Cells x genes expression matrix.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            A two-element tuple with means and standard deviations, each of size n_cells x 1.
        """
        concat = torch.cat(
            (torch.log1p(expression_mat), F.one_hot(batch_idx, self.nbatches).to(expression_mat.dtype)), -1
        )
        return self._encode_latent(concat)

    def _encode_auxiliary(self, concat: ArrayLike):
        l_encoded = self._l_encoder(concat)
        sizefactor_means = l_encoded[:, 0]
        sizefactor_stdevs = l_encoded[:, 1].exp()
        return sizefactor_means, sizefactor_stdevs

    def encode_auxiliary(self, expression_mat: torch.Tensor, batch_idx: torch.Tensor):
        """Encode auxiliary variables.

        In this case, auxiliary variables are the means and scales of the cell-specific size factors.

        Args:
            expression_mat: Cells x genes expression matrix.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            A two-element tuple with means and scales, each of size n_cells x 1.

        """
        concat = torch.cat(
            (torch.log1p(expression_mat), F.one_hot(batch_idx, self.nbatches).to(expression_mat.dtype)), -1
        )
        return self._encode_auxiliary(concat)

    def guide(self, expression_mat: torch.Tensor, batch_idx: torch.Tensor):
        """Variational posterior.

        Args:
            expression_mat: Cells x genes expression matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        self.reconstruction_guide(
            *self.encode_latent(expression_mat, batch_idx), *self.encode_auxiliary(expression_mat, batch_idx)
        )

    def _baseguide(
        self, ncells: int, sizefactor_means: torch.Tensor, sizefactor_stdevs: torch.Tensor, latentsample: Callable
    ):
        with pyro.plate("cells", size=ncells, dim=-2):
            pyro.sample(
                "l_n", dist.LogNormal(sizefactor_means[:, None], sizefactor_stdevs[:, None] + self.eps)
            )  # (ncells, 1)
            with pyro.plate("latent", size=self.n_latent_dim, dim=-1):
                latentsample()

    def reconstruction_guide(
        self,
        latent_means: torch.Tensor,
        latent_stdevs: torch.Tensor,
        sizefactor_means: torch.Tensor,
        sizefactor_stdevs: torch.Tensor,
    ):
        """Variational posterior given the encoded data.

        This can be used for translating from another modality.

        Args:
            latent_means: Cells x latent_vars latent mean matrix.
            latent_stdevs: Cells x latent_vars latent standard deviation matrix.
            sizefactor_means: Cells x 1 latent sizefactor mean matrix.
            sizefactor_stdevs: Cells x 1 latent sizefactor scale matrix.
        """
        self._baseguide(
            latent_means.shape[0],
            sizefactor_means,
            sizefactor_stdevs,
            lambda: pyro.sample(self.latent_name, dist.Normal(latent_means, latent_stdevs + self.eps)),
        )

    def sample_guide(self, sample: torch.Tensor, sizefactor_means: torch.Tensor, sizefactor_stdevs: torch.Tensor):
        """Variational posterior given a sample from the variatonal distribution.

        This can be used for translating from another modality.

        Args:
            sample: A sample from the variational distribution
            sizefactor_means: Cells x 1 latent sizefactor mean matrix.
            sizefactor_stdevs: Cells x 1 latent sizefactor scale matrix.
        """
        self._baseguide(
            sample.shape[0],
            sizefactor_means,
            sizefactor_stdevs,
            lambda: pyro.sample(self.latent_name, dist.Delta(sample)),
        )

    def normalized_guide(self, latent_means: torch.Tensor, latent_stdevs: torch.Tensor):
        """Variational posterior given the encoded data, without auxiliary variables.

        This is useful to draw normalized (corrected for sequencing depth) samples from
        the variational posterior. In this case the auxiliary variables will be set to
        a constant value, the unnormalized samples will be incorrect.

        Args:
            latent_means: Cells x latent_vars latent mean matrix.
            latent_stdevs: Cells x latent_vars latent standard deviation matrix.
        """
        self._baseguide(
            latent_means.shape[0],
            self.zero[None],
            self.one[None],
            lambda: pyro.sample(self.latent_name, dist.Normal(latent_means, latent_stdevs + self.eps)),
        )


class RNAVAE(LightningVAEBase):
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
        beta: The scaling factor for the KL divergence. If a `float`, the scaling will stay constant.
            If a `tuple[float, float, int, int]`, the first entry is the starting value, the second
            entry the final value, the third entry the epoch at which the penalty starts to increase,
            and the fourth entry the number of epochs until the final value is reached.
        eps: Small value for numerical stability.
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
        beta: float | tuple[float, float, int, int] = 1,
        eps: float = 1e-3,
    ):
        super().__init__(
            _RNAVAE,
            lr,
            beta,
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
            eps=eps,
        )
