"""Actor-Critic classes with eligibility traces for continuous-time TD learning.

Provides ``BaseActorCritic``, ``Critic``, and ``Actor`` as RatInABox
``NeuralNetworkNeurons`` subclasses so they integrate with the framework's
history tracking and rate-map plotting.
"""

import numpy as np
import torch

from ratinabox.contribs.NeuralNetworkNeurons import NeuralNetworkNeurons


class BaseActorCritic(NeuralNetworkNeurons):
    """Shared logic for actor and critic: eligibility trace updates and
    gradient-based training with a continuous-time TD rule.

    This is a RatInABox ``Neurons`` subclass so you can call
    ``.plot_rate_map()`` and inspect ``.history``.
    """

    default_params = {
        "tau": 5.0,
        "tau_z": 5.0,
        "input_layers": [],
        "NeuralNetworkModule": None,
        "optimizer": lambda params: torch.optim.SGD(
            params, lr=0.01, maximize=True, weight_decay=0.0
        ),
    }

    def __init__(self, Agent, params=None):
        if params is None:
            params = {}
        self.params = __class__.default_params.copy()
        self.params.update(params)
        super().__init__(Agent, self.params)
        self.initialise_traces()
        self.firingrate = self.get_state(save_torch=True)
        self.firingrate_last = self.firingrate
        if self.params["optimizer"] is not None:
            self.optimizer = self.params["optimizer"](
                self.NeuralNetworkModule.parameters()
            )

    def initialise_traces(self):
        """Zero-initialise the eligibility traces for all parameters."""
        self.traces = []
        for param in self.NeuralNetworkModule.parameters():
            shape = param.detach().numpy().shape
            self.traces.append(np.zeros(shape))

    def _train_step(self, L, td_error):
        """Full training step: gradients -> trace update -> optimizer step."""
        self._calculate_gradients(L=L)
        self._update_traces(td_error=td_error)
        self.optimizer.step()

    def _calculate_gradients(self, L):
        """Back-propagate through *L* (value for critic, log-prob for actor)."""
        self.NeuralNetworkModule.zero_grad()
        L.backward(retain_graph=True)

    def _update_traces(self, td_error):
        """Exponential eligibility trace update and gradient replacement."""
        for j, param in enumerate(self.NeuralNetworkModule.parameters()):
            dt = self.Agent.dt
            tau_z = self.tau_z
            x = param.grad.detach().numpy()
            e = self.traces[j]
            self.traces[j] = (1 - dt / tau_z) * e + dt * x
            param.grad = torch.tensor(
                self.traces[j] * td_error * self.Agent.dt, dtype=torch.float
            )


class Critic(BaseActorCritic):
    """Value-function critic with continuous-time TD error."""

    default_params = {}

    def update(self, reward, train=True):
        self._update_td_error(reward)
        if train:
            super()._train_step(L=self.firingrate_torch, td_error=self.td_error)
        self.firingrate_last = self.firingrate
        super().update()

    def _update_td_error(self, reward):
        self.dfiringrate_dt = (self.firingrate - self.firingrate_last) / self.Agent.dt
        self.td_error = (
            reward + self.dfiringrate_dt - self.firingrate / self.tau
        ).item()


class Actor(BaseActorCritic):
    """Policy actor updated via the critic's TD error."""

    default_params = {}

    def update(self, log_prob=None, td_error=None, train=True):
        if train:
            super()._train_step(L=log_prob, td_error=td_error)
        super().update()
