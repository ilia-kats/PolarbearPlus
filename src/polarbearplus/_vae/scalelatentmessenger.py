from collections.abc import Callable

from pyro.poutine.messenger import Messenger


class ScaleLatentMessenger(Messenger):
    """Scale the score of all unobserved sample sites.

    This is useful when implementing a beta-VAE.

    Args:
        scale: The scale (beta)
    """

    def __init__(self, scale: float):
        super().__init__()
        self._scale = scale

    def _pyro_sample(self, msg):
        if not msg["is_observed"]:
            msg["scale"] *= self._scale


def scale_latent(fn: Callable = None, scale: float = 1.0):
    """Convenient wrapper of `.ScaleLatentMessenger`.

    Args:
        fn: A stochastic function (callalbe containing Pyro primitive calls)
        scale: The scale.
    """
    msngr = ScaleLatentMessenger(scale=scale)
    return msngr(fn) if fn is not None else msngr
