from collections.abc import Callable

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroParam
from torch import nn
from torch.distributions import constraints
from torch.nn import functional as F

from .._utils import MLP
from .vaebase import LightningVAEBase, VAEBase


class _Encoder(nn.Module):
    """scATACseq encoder.

    Based on MLP, but separate encoders for each chromosome.

    Args:
        nconditions: Number of nconditions (e.g. experimental batches).
        chr_idx: List of index ranges in the data matrix belonging to separate chromosomes.
        output_dim: Number of output dimensions.
        n_layers: Number of hidden layers.
        dropout: Dropout probability.
        hidden_width_factor: Scaling factor for the width of the hidden layers.
    """

    def __init__(
        self,
        nconditions: int,
        chr_idx: list[tuple[int, int]],
        output_dim: int,
        n_layers: int,
        dropout: float = 0.1,
        hidden_width_factor: float = 1.0,
        noutputs: int = 1,
        last_activation: type[nn.Module] | nn.Module | None = None,
    ):
        super().__init__()

        self._chr_idx = chr_idx
        self._encoders = nn.ModuleList()
        self._nconditions = nconditions
        for idx_start, idx_end in chr_idx:
            indim = idx_end - idx_start
            hidden_dim = int((indim * output_dim) ** 0.5 * hidden_width_factor)
            self._encoders.append(
                MLP(
                    input_dim=indim + nconditions,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    dropout=dropout,
                )
            )

        output = nn.Linear(len(chr_idx) * output_dim, noutputs * output_dim)
        nn.init.kaiming_normal_(output.weight)
        nn.init.zeros_(output.bias)
        self._output = nn.Sequential(output)
        if last_activation is not None:
            if isinstance(last_activation, nn.Module):
                self._output.append(last_activation)
            else:
                self._output.append(last_activation())

    def forward(self, x, conditions):
        return self._output(
            torch.cat(
                [
                    enc(
                        torch.cat((x[..., idx_start:idx_end], F.one_hot(conditions, self._nconditions).to(x.dtype)), -1)
                    )
                    for (idx_start, idx_end), enc in zip(self._chr_idx, self._encoders, strict=False)
                ],
                -1,
            )
        )

    def get_extra_state(self):
        return {"chr_idx": self._chr_idx, "nconditions": self._nconditions}

    def set_extra_state(self, state):
        self._chr_idx = state["chr_idx"]
        self._nconditions = state["nconditions"]


class _Decoder(nn.Module):
    """scATACseq decoder.

    Based on MLP, but separate decoders for each chromosome.

    Args:
        nconditions: Number of nconditions (e.g. experimental batches).
        chr_idx: List of index ranges in the data matrix belonging to separate chromosomes.
        input_dim: Number of input dimensions.
        n_layers: Number of hidden layers.
        dropout: Dropout probability.
        hidden_width_factor: Scaling factor for the width of the hidden layers.
    """

    def __init__(
        self,
        nconditions: int,
        chr_idx: list[tuple[int, int]],
        input_dim: int,
        n_layers: int,
        dropout: float = 0.1,
        hidden_width_factor: float = 1.0,
    ):
        super().__init__()

        self._decoders = nn.ModuleList()
        self._nconditions = nconditions
        for idx_start, idx_end in sorted(chr_idx):
            outdim = idx_end - idx_start
            hidden_dim = int((outdim * input_dim) ** 0.5 * hidden_width_factor)
            self._decoders.append(
                MLP(
                    input_dim=input_dim + nconditions,
                    output_dim=outdim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    dropout=dropout,
                )
            )

    def forward(self, x, conditions):
        concat = torch.cat((x, F.one_hot(conditions, self._nconditions).to(x.dtype)), -1)
        return torch.sigmoid(torch.cat([dec(concat) for dec in self._decoders], -1))


class _ATACVAE(VAEBase):
    """Pyro implementation of a VAE for scATACseq data, based on PeakVI.

    Args:
        chr_idx: List of index ranges in the data matrix belonging to separate chromosomes.
        nbatches: Number of experimental batches.
        hidden_width_factor: Scaling factor for the width of the hidden layers.
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
        hidden_width_factor: float = 1.0,
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
        self._n_latent_dim = n_latent_dim
        self.register_buffer("zero", torch.as_tensor(0.0))
        self.register_buffer("one", torch.as_tensor(1.0))

        if decoder_n_layers is None:
            decoder_n_layers = encoder_n_layers
        if decoder_dropout is None:
            decoder_dropout = encoder_dropout
        self._encoder = _Encoder(
            nconditions=nbatches,
            chr_idx=chr_idx,
            output_dim=n_latent_dim,
            n_layers=encoder_n_layers,
            dropout=encoder_dropout,
            hidden_width_factor=hidden_width_factor,
            noutputs=2,
        )
        self._decoder = _Decoder(
            nconditions=nbatches,
            chr_idx=chr_idx,
            input_dim=n_latent_dim,
            n_layers=decoder_n_layers,
            dropout=decoder_dropout,
            hidden_width_factor=hidden_width_factor,
        )
        self._l_encoder = _Encoder(
            nconditions=nbatches,
            chr_idx=chr_idx,
            output_dim=1,
            n_layers=encoder_n_layers,
            dropout=encoder_dropout,
            hidden_width_factor=hidden_width_factor,
            last_activation=nn.Sigmoid,
        )
        # torch.nn.init.zeros_(self._l_encoder[-2].weight)

        self.r = PyroParam(torch.randn((self.nregions,)).sigmoid(), constraint=constraints.interval(0, 1))

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
        return "norm_y_n"

    def get_extra_state(self):
        return {"nregions": self.nregions, "nbatches": self.nbatches, "n_latent_dim": self._n_latent_dim}

    def set_extra_state(self, state):
        self.nregions = state["nregions"]
        self.nbatches = state["nbatches"]
        self._n_latent_dim = state["n_latent_dim"]

    def encode_latent(self, region_mat: torch.Tensor, batch_idx: torch.Tensor):
        """Apply the encoder network.

        Args:
            region_mat: Cells x genes expression matrix.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            A two-element tuple with means and standard deviations, each of size n_cells x 1.
        """
        latents = self._encoder(region_mat, batch_idx)
        latent_means = latents[:, : self.n_latent_dim]
        latent_stdevs = latents[:, self.n_latent_dim :].exp()

        return latent_means, latent_stdevs

    def encode_auxiliary(self, region_mat: torch.Tensor, batch_idx: torch.Tensor):
        """Encode auxiliary variables.

        In this case, auxiliary variables are the cell-specific factors.

        Args:
            region_mat: Cells x genes expression matrix.
            batch_idx: Index of the experimental batch for each cell.

        Returns:
            A 1-element tuple with an n_cells x 1 matrix.
        """
        return (self._l_encoder(region_mat, batch_idx),)  # (ncells, 1)

    def model(self, region_mat: torch.Tensor | None, batch_idx: torch.Tensor):
        """Generative model.

        Args:
            region_mat: Cells x regions binary peak matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        with pyro.plate("cells", size=batch_idx.shape[0], dim=-2):
            # this will be replaced by the encoded l_n from the guide during inference
            # can't use pyro.deterministic because that sets is_observed=True, so won't
            # be replaced by guide value. We want l_n in the trace for analysis
            l_n = pyro.sample("l_n", dist.Delta(self.one).mask(region_mat is None))  # (ncells, 1)

            with pyro.plate("latent", size=self.n_latent_dim, dim=-1):
                z_n = pyro.sample(self.latent_name, dist.Normal(self.zero, self.one))  # (ncells, nlatent)

            y_n = pyro.deterministic("y_n", self._decoder(z_n, batch_idx))  # (ncells, nregions)
            regionscaled = pyro.deterministic(self.normalized_name, y_n * self.r)
            probs = pyro.deterministic("mu", regionscaled * l_n)
            with pyro.plate("regions", size=self.nregions, dim=-1):
                pyro.sample(self.observed_name, dist.Bernoulli(probs=probs), obs=region_mat)

    def guide(self, region_mat: torch.Tensor, batch_idx: torch.Tensor):
        """Variational posterior.

        Args:
            region_mat: Cells x genes expression matrix.
            batch_idx: Index of the experimental batch for each cell.
        """
        self.reconstruction_guide(
            *self.encode_latent(region_mat, batch_idx), *self.encode_auxiliary(region_mat, batch_idx)
        )

    def _baseguide(self, ncells: int, l_n: torch.Tensor, latentsample: Callable):
        with pyro.plate("cells", size=ncells, dim=-2):
            pyro.sample("l_n", dist.Delta(l_n))
            with pyro.plate("latent", size=self.n_latent_dim, dim=-1):
                latentsample()

    def reconstruction_guide(self, latent_means: torch.Tensor, latent_stdevs: torch.Tensor, l_n: torch.Tensor):
        """Variational posterior given the encoded data.

        This can be used for translating from another modality.

        Args:
            latent_means: Cells x latent_vars latent mean matrix.
            latent_stdevs: Cells x latent_vars latent standard deviation matrix.
            l_n: Cells x 1 matrix of cell-specific factors.
        """
        self._baseguide(
            latent_means.shape[0], l_n, lambda: pyro.sample(self.latent_name, dist.Normal(latent_means, latent_stdevs))
        )

    def sample_guide(self, sample: torch.Tensor, l_n: torch.Tensor):
        """Variational posterior given a sample from the variatonal distribution.

        This can be used for translating from another modality.

        Args:
            sample: A sample from the variational distribution
            l_n: Cells x 1 matrix of cell-specific factors.
        """
        self._baseguide(sample.shape[0], l_n, lambda: pyro.sample(self.latent_name, dist.Delta(sample)))

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
            self.one,
            lambda: pyro.sample(self.latent_name, dist.Normal(latent_means, latent_stdevs)),
        )


class ATACVAE(LightningVAEBase):
    """A beta-VAE for scATACseq data.

    Args:
        chr_idx: List of index ranges in the data matrix belonging to separate chromosomes.
        nbatches: Number of experimental batches.
        n_latent_dim: Number of latent dimensions.
        encoder_n_layers: Number of hidden layers in the encoder.
        hidden_width_factor: Scaling factor for the width of the hidden layers.
        encoder_dropout: Dropout probability in the encoder.
        decoder_n_layers: Number of hidden layers in the decoder.
        decoder_dropout: Dropout probability in the decoder.
        lr: Learning rate.
        beta: The scaling factor for the KL divergence. If a `float`, the scaling will stay constant.
            If a `tuple[float, float, int, int]`, the first entry is the starting value, the second
            entry the final value, the third entry the epoch at which the penalty starts to increase,
            and the fourth entry the number of epochs until the final value is reached.
    """

    def __init__(
        self,
        chr_idx: list[tuple[int, int]],
        nbatches: int,
        n_latent_dim: int,
        encoder_n_layers: int,
        hidden_width_factor: float = 1.0,
        encoder_dropout: float = 0.1,
        decoder_n_layers: int | None = None,
        decoder_dropout: float | None = None,
        lr: float = 1e-3,
        beta: float | tuple[float, float, int, int] = 1,
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
