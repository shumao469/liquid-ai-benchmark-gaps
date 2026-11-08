"""Memoryless, discrete recurrent, fixed-tau, and ODE-RNN observers."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .base import TemporalPolicy
from .topology import TopologySpec, recurrent_mask


class MLPPolicy(TemporalPolicy):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> None:
        del batch_size, device
        return None

    def forward_step(self, x: torch.Tensor, dt: torch.Tensor, state: None):
        del dt, state
        return self.network(x), None


class GRUPolicy(TemporalPolicy):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def initial_state(self, batch_size: int, device: torch.device | str):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward_step(self, x: torch.Tensor, dt: torch.Tensor, state: torch.Tensor):
        del dt
        state = self.cell(x, state)
        return self.head(state), state


class LSTMPolicy(TemporalPolicy):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.cell = nn.LSTMCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def initial_state(self, batch_size: int, device: torch.device | str):
        zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
        return zeros, zeros.clone()

    def forward_step(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ):
        del dt
        hidden, cell = self.cell(x, state)
        return self.head(hidden), (hidden, cell)


class MaskedLinear(nn.Linear):
    def __init__(self, features: int, mask: torch.Tensor) -> None:
        super().__init__(features, features, bias=False)
        if mask.shape != self.weight.shape:
            raise ValueError("Mask shape must equal recurrent weight shape")
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, None)


class FixedTauCTRNNPolicy(TemporalPolicy):
    """Stable fixed-time-constant control for the adaptive-time-constant ablation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        topology: TopologySpec,
        initial_tau: float = 0.20,
        min_tau: float = 0.001,
        max_step: float = 0.02,
    ) -> None:
        super().__init__()
        if initial_tau <= min_tau or min_tau <= 0:
            raise ValueError("Require initial_tau > min_tau > 0")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_tau = float(min_tau)
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        self.max_step = float(max_step)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.recurrent = MaskedLinear(hidden_dim, recurrent_mask(hidden_dim, topology))
        # This is the registered state-invariant time-scale control.  Keeping
        # tau as a buffer (rather than a learned parameter) prevents the
        # "fixed-tau" mechanism contrast from silently becoming adaptive during
        # optimization.
        self.register_buffer(
            "fixed_tau", torch.full((hidden_dim,), float(initial_tau))
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    @property
    def tau(self) -> torch.Tensor:
        return self.fixed_tau

    def initial_state(self, batch_size: int, device: torch.device | str):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward_step(self, x: torch.Tensor, dt: torch.Tensor, state: torch.Tensor):
        def field(value: torch.Tensor) -> torch.Tensor:
            equilibrium = torch.tanh(self.input_layer(x) + self.recurrent(value))
            return (equilibrium - value) / self.tau[None, :]

        step_count = max(
            1, int(math.ceil(float(dt.detach().max().cpu()) / self.max_step))
        )
        step = dt[:, None] / step_count
        for _ in range(step_count):
            k1 = field(state)
            k2 = field(state + 0.5 * step * k1)
            k3 = field(state + 0.5 * step * k2)
            k4 = field(state + step * k3)
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.head(state), state


class ODERNNPolicy(TemporalPolicy):
    """ODE evolution between observations followed by a GRU observation update.

    A deterministic fixed-step RK4 integrator is used and its maximum step is
    reported. This avoids conflating architecture with adaptive-solver tolerances.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        topology: TopologySpec,
        max_step: float = 0.02,
    ) -> None:
        super().__init__()
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_step = float(max_step)
        self.vector_field = MaskedLinear(hidden_dim, recurrent_mask(hidden_dim, topology))
        self.ode_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.observation_update = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def initial_state(self, batch_size: int, device: torch.device | str):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def _field(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.vector_field(state) + self.ode_bias)

    def _rk4(self, state: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        step_count = max(1, int(math.ceil(float(dt.detach().max().cpu()) / self.max_step)))
        step = dt[:, None] / step_count
        for _ in range(step_count):
            k1 = self._field(state)
            k2 = self._field(state + 0.5 * step * k1)
            k3 = self._field(state + 0.5 * step * k2)
            k4 = self._field(state + step * k3)
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state

    def forward_step(self, x: torch.Tensor, dt: torch.Tensor, state: torch.Tensor):
        evolved = self._rk4(state, dt)
        state = self.observation_update(x, evolved)
        return self.head(state), state

