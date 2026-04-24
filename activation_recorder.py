"""Clean hook-based activation recording for PyTorch modules.

Replaces the ad-hoc global-variable hook pattern in the notebook with a
reusable ``ActivationRecorder`` class that registers / removes forward
hooks properly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn



class ActivationRecorder:
    """Record forward-pass activations from one or more ``nn.Module`` layers.

    Usage::

        rec = ActivationRecorder()
        rec.attach(fta_module, name="fta_eta01")
        rec.attach(mlp_penultimate, name="mlp_rep")

        # --- run inference loop ---
        for obs in observations:
            rec.set_observation(obs)          # tag the upcoming activation
            with torch.inference_mode():
                net(state)

        # --- retrieve ---
        acts = rec.get("fta_eta01")           # dict[obs_key -> list[Tensor]]
        rec.detach_all()                      # clean up hooks

    Each recorded activation is a detached CPU tensor. Observations are
    rounded and converted to tuples so they can serve as dict keys.
    """

    def __init__(self, precision: int = 6):
        self._precision = precision
        self._current_obs: tuple | None = None
        self._stores: dict[str, dict[tuple, list[torch.Tensor]]] = {}
        self._handles: dict[str, torch.utils.hooks.RemovableHook] = {}

    # -- public API ---------------------------------------------------------

    def set_observation(self, observation) -> None:
        """Set the observation key that will tag the *next* forward pass."""
        self._current_obs = tuple(
            np.round(np.asarray(observation), self._precision).tolist()
        )

    def attach(self, module: nn.Module, name: str) -> None:
        """Register a forward hook on *module* under the given *name*."""
        if name in self._handles:
            raise ValueError(f"Name '{name}' is already attached")
        store: dict[tuple, list[torch.Tensor]] = defaultdict(list)
        self._stores[name] = store

        def _hook(mod, inputs, output, _store=store):
            if self._current_obs is None:
                return
            _store[self._current_obs].append(output.detach().cpu())

        self._handles[name] = module.register_forward_hook(_hook)

    def detach(self, name: str) -> None:
        """Remove the hook for *name*."""
        if name in self._handles:
            self._handles.pop(name).remove()

    def detach_all(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()

    def get(self, name: str) -> dict[tuple, list[torch.Tensor]]:
        """Return the activation store for *name*."""
        return dict(self._stores[name])

    def clear(self, name: str | None = None) -> None:
        """Clear recorded activations (all names, or just *name*)."""
        if name is None:
            for store in self._stores.values():
                store.clear()
        else:
            self._stores[name].clear()

    @property
    def names(self) -> list[str]:
        return list(self._stores.keys())
    


# ---------------------------------------------------------------------------
# Utility: find specific layers inside a network
# ---------------------------------------------------------------------------

def find_layer_module(net: torch.nn.Module, layer_name: str):
    '''Find layer in a network and print the type of each module as it searches'''

    print(f"Searching for {layer_name} layer in the network...")

    for m in net.modules():
        if isinstance(m, layer_name):
            return m
    raise ValueError(f"No {layer_name.__name__} module found in the network.")
    

def find_penultimate_layer(net: nn.Module) -> nn.Module:
    """Return the layer immediately before the last ``nn.Linear`` in *net*.

    Assumes the network exposes a ``.net`` attribute (``nn.Sequential``).
    """
    seq = net.net if hasattr(net, "net") else net
    layers = list(seq.children())

    last_linear_idx = None

   
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            last_linear_idx = i

    if last_linear_idx is None:
        raise ValueError("No Linear layer found.")
    if last_linear_idx == 0:
        raise ValueError("No penultimate layer exists.")
    
    penultimate = layers[last_linear_idx - 1]

    return penultimate
