"""Common step-wise observer interface used by every temporal architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class TemporalPolicy(nn.Module, ABC):
    """Historical class name retained for API stability; outputs estimated state."""

    input_dim: int
    output_dim: int

    @abstractmethod
    def initial_state(self, batch_size: int, device: torch.device | str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward_step(
        self, x: torch.Tensor, dt: torch.Tensor, state: Any
    ) -> tuple[torch.Tensor, Any]:
        """Advance one observation and return a state estimate plus recurrent state."""
        raise NotImplementedError

    def forward_sequence(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        state: Any | None = None,
    ) -> tuple[torch.Tensor, Any]:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected x [batch,time,{self.input_dim}], got {tuple(x.shape)}"
            )
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt[..., 0]
        if dt.ndim != 2 or dt.shape != x.shape[:2]:
            raise ValueError("dt must have shape [batch,time] or [batch,time,1]")
        if torch.any(~torch.isfinite(x)) or torch.any(~torch.isfinite(dt)):
            raise ValueError("x and dt must be finite")
        if torch.any(dt <= 0):
            raise ValueError("All time deltas must be positive")
        if state is None:
            state = self.initial_state(x.shape[0], x.device)
        actions = []
        for step in range(x.shape[1]):
            action, state = self.forward_step(x[:, step], dt[:, step], state)
            actions.append(action)
        return torch.stack(actions, dim=1), state

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        actions, _ = self.forward_sequence(x, dt)
        return actions

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

