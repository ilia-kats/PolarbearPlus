from torch import nn


class MLP(nn.Sequential):
    """A simple feed-forward neural network.

    Args:
        input_dim: Number of input dimensions.
        output_dim: Number of output dimensions. If 0, no output layer will be included.
        hidden_dim: Width of hidden layers.
        n_layers: Number of hidden layers.
        dropout: Dropout probability.
        last_activation: Activation function of the output layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.1,
        last_activation: type[nn.Module] | nn.Module | None = None,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("Need at least one hidden layer")
        if input_dim < 1 or output_dim < 0 or hidden_dim < 1:
            raise ValueError("All layer dimensions must be positive")
        if dropout < 0 or dropout > 1:
            raise ValueError("Droput rate must be between 0 and 1")
        indim = input_dim
        for _ in range(n_layers):
            self.append(
                nn.Sequential(
                    nn.Linear(indim, hidden_dim),
                    nn.LayerNorm(hidden_dim, elementwise_affine=False),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            indim = hidden_dim
        if output_dim > 0:
            self.append(nn.Linear(hidden_dim, output_dim))
            if last_activation is not None:
                if isinstance(last_activation, nn.Module):
                    self.append(last_activation)
                else:
                    self.append(last_activation())

        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
