"""
Shared activation factory.

Every model builds its activation through `get_activation()` so a single
`--activation` flag controls the whole pipeline. LeakyReLU is the default: with
return targets centred near zero, a plain ReLU zeroes the gradient for every
negative pre-activation, and a large fraction of units can go permanently dead
early in training. LeakyReLU keeps a small negative slope so those units recover.
"""

import torch.nn as nn

DEFAULT_ACTIVATION = "leaky_relu"

# Name -> zero-arg constructor. Kept as factories, not shared instances, so each
# layer owns its own module (important for state_dict keys and for activations
# that carry parameters, e.g. PReLU).
_ACTIVATIONS = {
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,      # alias
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
    "mish": nn.Mish,
}

ACTIVATION_CHOICES = sorted(_ACTIVATIONS)


def get_activation(name=DEFAULT_ACTIVATION):
    """Return a fresh activation module for `name`.

    Raises ValueError on an unknown name rather than silently falling back, so a
    typo in --activation cannot quietly train a different architecture than the
    one requested.
    """
    if name is None:
        name = DEFAULT_ACTIVATION
    key = str(name).strip().lower().replace("-", "_")
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {name!r}. Choose from: {', '.join(ACTIVATION_CHOICES)}"
        )
    return _ACTIVATIONS[key]()
