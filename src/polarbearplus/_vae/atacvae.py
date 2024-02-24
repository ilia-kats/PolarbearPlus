import pyro
import pyro.distributions as dist
import torch
from numpy.typing import ArrayLike
from pyro.nn import PyroModule, PyroParam
from torch import nn
from torch.distributions import constraints
from torch.nn import functional as F

from .mlp import MLP
from .vaebase import VAEBase


class _Encoder(nn.Module):
    """scATACseq encoder.

    Based on MLP, but separate encoders for each chromosome.

    Args:
        nconditions: Number of nconditions (e.g. experimental batches).
        chr_idx: List of index ranges in the data matrix belonging to separate chromosomes.
        output_dim: Number of output dimensions.
        n_layers: Number of hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self, nconditions: int, chr_idx: list[tuple[int, int]], output_dim: int, n_layers: int, dropout: float = 0.1
    ):
        super().__init__()

        self._encoders = nn.ModuleList()
        for idx_start, idx_end in chr_idx:
            indim = idx_end - idx_start
            hidden_dim = int((indim * output_dim) ** 0.5)
            self._encoders.append(
                MLP(
                    input_dim=indim + nconditions,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    dropout=dropout,
                )
            )

        self._output = nn.Sequential(nn.Linear(len(chr_idx) * output_dim, 2 * output_dim))

    def forward(self, x):
        return self._output(torch.cat([enc(x) for enc in self._encoders], -1))


class _Decoder(nn.Module):
    """scATACseq decoder.

    Based on MLP, but separate decoders for each chromosome.

    Args:
        nconditions: Number of nconditions (e.g. experimental batches).
        chr_idx: List of index ranges in the data matrix belonging to separate chromosomes.
        input_dim: Number of input dimensions.
        n_layers: Number of hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self, nconditions: int, chr_idx: list[tuple[int, int]], input_dim: int, n_layers: int, dropout: float = 0.1
    ):
        super().__init__()

        self._decoders = nn.ModuleList()
        for idx_start, idx_end in chr_idx:
            outdim = idx_end - idx_start
            hidden_dim = int((outdim * input_dim) ** 0.5)
            self._decoders.append(
                MLP(
                    input_dim=input_dim + nconditions,
                    output_dim=outdim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        return torch.cat([dec(x) for dec in self._decoders], -1)


class _ATACVAE(PyroModule):
    """Pyro implementation of a VAE for scATACseq data, based on PeakVI.

    Args:
        chr_idx: List of index ranges in the data matrix belonging to separate chromosomes.
        nbatches: Number of experimental batches.
        n_latent_dim: Number of latent dimensions.
        encoder_n_layers: Number of hidden layers in the encoder.
        encoder_dropout: Dropout probability in the encoder.
        decoder_n_layers: Number of hidden layers in the decoder.
        decoder_dropout: Dropout probability in the decoder.
    """

    def __init__(
        self,
        chr_idx: list[tuple[int, int]],
        nbatches: int,
        n_latent_dim: int | None = None,
        encoder_n_layers: int = 3,
        encoder_dropout: float = 0.1,
        decoder_n_layers: int | None = None,
        decoder_dropout: float | None = None,
    ):
        super().__init__()

        self.nregions = sum(end - start for start, end in chr_idx)
        self.nbatches = nbatches

        if n_latent_dim is None:
            n_latent_dim = int(self.nregions**0.25)
        self.n_latent_dim = n_latent_dim
        self.register_buffer("zero", torch.as_tensor(0.0))
        self.register_buffer("one", torch.as_tensor(1.0))

        if decoder_n_layers is None:
            decoder_n_layers = encoder_n_layers
        if decoder_dropout is None:
            decoder_dropout = encoder_dropout
        self.encoder = _Encoder(nbatches, chr_idx, n_latent_dim, encoder_n_layers, encoder_dropout)
        self.decoder = _Decoder(nbatches, chr_idx, n_latent_dim, decoder_n_layers, decoder_dropout)
        self._l_encoder = MLP(
            self.nregions + nbatches,
            1,
            (self.nregions * n_latent_dim) ** 0.5,
            encoder_n_layers,
            encoder_dropout,
            last_activation=nn.Sigmoid,
        )

        self.r = PyroParam(torch.randn((self.nregions,)).exp(), constraint=constraints.positive)

    def get_extra_state(self):
        return {"nregions": self.nregions, "nbatches": self.nbatches, "n_latent_dim": self.n_latent_dim}

    def set_extra_state(self, state):
        self.nregions = state["nregions"]
        self.nbatches = state["nbatches"]
        self.n_latent_dim = state["n_latent_dim"]

    def encode(self, region_mat: ArrayLike, batch_idx: ArrayLike):
        concat = torch.cat((region_mat, F.one_hot(batch_idx, self.nbatches).astype(region_mat)), -1)
        latents = self.encoder(concat)
        latent_means = latents[:, : self.n_latent_dim]
        latent_stdevs = latents[:, self.n_latent_dim :].exp()
        return latent_means, latent_stdevs

    def model(self, region_mat: ArrayLike | None, batch_idx: ArrayLike, ncells: int | None = None):
        """Generative model.

        Args:
            region_mat: Cells x regions binary peak matrix.
            batch_idx: Index of the experimental batch for each cell.
            ncells: Number of cells. Only required in generative mode (when no region matrix is given).
        """
        if region_mat is None and ncells is None:
            raise ValueError("Need either region_mat or ncells, but both are None.")
        if ncells is None:
            ncells = region_mat.shape[0]
        if region_mat is not None:
            concat = torch.cat((region_mat, F.one_hot(batch_idx, self.nbatches).to(region_mat.dtype)), -1)
            l_n = self._l_encoder(concat)  # (ncells, 1)
        else:
            l_n = torch.ones((ncells, 1))  # (ncells, 1)
        with pyro.plate("cells", size=ncells, dim=-2):
            with pyro.plate("latent", size=self.n_latent_dim, dim=-1):
                z_n = pyro.sample("z_n", dist.Normal(self.zero, self.one))  # (ncells, nlatent)

            z_n = torch.cat((z_n, F.one_hot(batch_idx, self.n_batches).to(z_n.dtype)), -1)
            y_n = self.decoder(z_n)  # (ncells, nregions)

            with pyro.plate("regions", size=self.nregions, dim=-1):
                pyro.sample("data", dist.Bernoulli(probs=y_n * l_n * self.r))

    def guide(self, region_mat: ArrayLike | None, batch_idx: ArrayLike):
        """Variational posterior.

        Args:
            region_mat: Cells x genes expression matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        latent_means, latent_stdevs = self.encode(region_mat, batch_idx)
        with pyro.plate("cells", size=region_mat.shape[0], dim=-2):  # noqa SIM117
            with pyro.plate("latent", size=self.n_latent_dim, dim=-1):
                pyro.sample("z_n", dist.Normal(latent_means, latent_stdevs))


class ATACVAE(VAEBase):
    """A beta-VAE for scATACseq data.

    Args:
        chr_idx: List of index ranges in the data matrix belonging to separate chromosomes.
        nbatches: Number of experimental batches.
        n_latent_dim: Number of latent dimensions.
        encoder_n_layers: Number of hidden layers in the encoder.
        encoder_dropout: Dropout probability in the encoder.
        decoder_n_layers: Number of hidden layers in the decoder.
        decoder_dropout: Dropout probability in the decoder.
        lr: Learning rate.
        beta: The scaling factor for the KL divergence.
    """

    def __init__(
        self,
        chr_idx: list[tuple[int, int]],
        nbatches: int,
        n_latent_dim: int,
        encoder_n_layers: int,
        encoder_dropout: float = 0.1,
        decoder_n_layers: int | None = None,
        decoder_dropout: float | None = None,
        lr: float = 1e-3,
        beta: float = 1,
    ):
        super().__init__(
            _ATACVAE,
            lr,
            beta,
            chr_idx=chr_idx,
            nbatches=nbatches,
            n_latent_dim=n_latent_dim,
            encoder_n_layers=encoder_n_layers,
            encoder_dropout=encoder_dropout,
            decoder_n_layers=decoder_n_layers,
            decoder_dropout=decoder_dropout,
        )
