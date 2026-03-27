"""Unified network classes for actor-critic experiments.

Provides a single backbone (`Backbone`) that accepts either ReLU or FTA as the
hidden activation, eliminating the previous duplication between
`MultiLayerPerceptron` / `FTANetwork` and their actor wrappers.

Policy heads (`VxVyGaussianHead`, `NESWCategoricalHead`) are independent of
the backbone choice.
"""

import numpy as np
import torch
import torch.nn as nn

from fta import FTA


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    """A feedforward network whose hidden activation is configurable.

    For ReLU mode (the default), this builds:
        Linear -> ReLU -> ... -> Linear   (no activation after the last layer)

    For FTA mode, layers *before* the FTA use ReLU, and the FTA replaces the
    final hidden activation:
        Linear -> ReLU -> ... -> Linear -> LayerNorm -> FTA -> Linear

    A ``LayerNorm`` is inserted immediately before FTA so that the
    pre-FTA activations are normalized to zero mean / unit variance,
    removing sensitivity to the tiling bound hyperparameter.

    Args:
        n_in:  input dimension
        n_out: output dimension
        hidden: list of hidden-layer widths (ReLU mode), or ignored in FTA mode
        activation: ``nn.Module`` instance to use as the hidden activation.
            * ``nn.ReLU()`` (default) -- plain MLP
            * ``FTA(...)`` -- fuzzy tiling activation inserted after the
              pre-FTA layers; ``pre_fta`` / ``post_fta`` control the
              layer widths around it.
        pre_fta:  layer widths *before* the FTA (only used when
                  ``activation`` is an ``FTA`` instance).
        post_fta: layer widths *after* the FTA (only used when
                  ``activation`` is an ``FTA`` instance).  The last
                  element is connected to a final ``Linear(…, n_out)``.
    """

    def __init__(
        self,
        n_in: int = 20,
        n_out: int = 1,
        hidden: list | None = None,
        activation: nn.Module | None = None,
        pre_fta: list | None = None,
        post_fta: list | None = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()

        if isinstance(activation, FTA):
            self._build_fta(n_in, n_out, activation, pre_fta or [20], post_fta or [20])
        else:
            self._build_mlp(n_in, n_out, hidden or [20, 20], activation)

    # -- private builders ---------------------------------------------------

    def _build_mlp(self, n_in, n_out, hidden, activation):
        sizes = [n_in] + hidden + [n_out]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation)
        self.net = nn.Sequential(*layers)

    def _build_fta(self, n_in, n_out, fta_module, pre_fta, post_fta):
        fta_out = pre_fta[-1] * fta_module.n_tiles * fta_module.n_tilings
        sizes_pre = [n_in] + pre_fta
        # post_fta defines layers after FTA; last element is the output dim,
        # so n_out is appended only if it differs from post_fta[-1].
        sizes_post = [fta_out] + post_fta
        if post_fta[-1] != n_out:
            sizes_post.append(n_out)

        layers = []
        for i in range(len(sizes_pre) - 1):
            layers.append(nn.Linear(sizes_pre[i], sizes_pre[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.LayerNorm(sizes_pre[-1]))
        layers.append(fta_module)

        for i in range(len(sizes_post) - 1):
            layers.append(nn.Linear(sizes_post[i], sizes_post[i + 1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Policy heads -- these wrap a Backbone and add action-sampling logic
# ---------------------------------------------------------------------------

class VxVyGaussianHead(nn.Module):
    """Continuous 2-D velocity policy (Gaussian).

    Wraps a ``Backbone`` (or any ``nn.Module``) and adds ``tanh`` output
    bounding + Gaussian action sampling.
    """

    def __init__(self, backbone: nn.Module, max_speed: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.max_speed = max_speed
        self.n = 2  # action dimensionality

    def forward(self, x):
        return self.max_speed * torch.tanh(self.backbone(x))

    def sample_action(self, firingrate: torch.Tensor):
        """Sample from the 2-D Gaussian and return (action_np, log_prob)."""
        std = 0.1
        vx_dist = torch.distributions.Normal(firingrate[:, 0], scale=std)
        vy_dist = torch.distributions.Normal(firingrate[:, 1], scale=std)
        vx = vx_dist.sample()
        vy = vy_dist.sample()
        action = np.array([vx.item(), vy.item()])
        log_prob = vx_dist.log_prob(vx) + vy_dist.log_prob(vy)
        return action, log_prob


class NESWCategoricalHead(nn.Module):
    """Discrete 4-direction policy (Categorical: N-E-S-W).

    Wraps a ``Backbone`` and adds softmax + categorical sampling.
    """

    def __init__(self, backbone: nn.Module, speed: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.speed = speed
        self.n = 4

    def forward(self, x):
        return torch.softmax(self.backbone(x), dim=1)

    _DIRECTIONS = np.array([
        [0,  1],   # N
        [1,  0],   # E
        [0, -1],   # S
        [-1, 0],   # W
    ], dtype=np.float64)

    def sample_action(self, firingrate: torch.Tensor):
        """Sample a cardinal direction and return (action_np, log_prob)."""
        dist = torch.distributions.Categorical(firingrate)
        choice = dist.sample()
        action = self.speed * self._DIRECTIONS[choice.item()]
        log_prob = dist.log_prob(choice)
        return action, log_prob


# ---------------------------------------------------------------------------
# Convenience factories (match the old class constructors)
# ---------------------------------------------------------------------------

def make_mlp_critic(n_in: int, hidden: list | None = None) -> Backbone:
    """Return a plain MLP backbone suitable for the critic (1-D output)."""
    return Backbone(n_in=n_in, n_out=1, hidden=hidden or [20, 20])


def make_fta_critic(
    n_in: int,
    *,
    pre_fta: list | None = None,
    post_fta: list | None = None,
    input_min: float = -1.0,
    input_max: float = 1.0,
    n_tiles: int = 10,
    n_tilings: int = 1,
    eta: float | None = None,
) -> Backbone:
    """Return an FTA backbone suitable for the critic (1-D output).

    If *eta* is ``None`` (default), it is set to the tile width
    ``(input_max - input_min) / n_tiles`` so that ``eta == delta``.
    """
    if eta is None:
        eta = (input_max - input_min) / n_tiles
    fta = FTA(
        params={
            'n_tiles': n_tiles,
            'n_tilings': n_tilings,
            'fta_input_min': input_min,
            'fta_input_max': input_max,
            'fta_eta': eta,
        },
        input_dim=(pre_fta or [20])[-1],
    )
    return Backbone(
        n_in=n_in,
        n_out=1,
        activation=fta,
        pre_fta=pre_fta or [20],
        post_fta=post_fta or [20],
    )


def make_gaussian_actor(n_in: int, backbone_factory=None, **kw) -> VxVyGaussianHead:
    """Return a VxVy Gaussian actor head.

    ``backbone_factory`` defaults to a 1-hidden-layer MLP.
    """
    if backbone_factory is None:
        bb = Backbone(n_in=n_in, n_out=2, hidden=[50])
    else:
        bb = backbone_factory(n_in=n_in, n_out=2, **kw)
    return VxVyGaussianHead(bb)
