"""Task-specific sequence classifiers for the corrected offline benchmarks.

The offline heads are intentionally separate from the closed-loop state observer.
All four model families consume the same value/mask/age feature tensor.  ODE-RNN,
LTC, and CfC additionally receive the explicit elapsed time for every update;
elapsed time is never replaced by a unit-step default inside this module.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch import nn


class SequenceEncoder(nn.Module, ABC):
    """Return one hidden feature vector per input time step."""

    input_dim: int
    hidden_dim: int
    uses_elapsed_time: bool

    @abstractmethod
    def forward(self, values: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def validate(values: torch.Tensor, dt: torch.Tensor, input_dim: int) -> None:
        if values.ndim != 3 or values.shape[-1] != input_dim:
            raise ValueError(
                f"values must have shape [batch,time,{input_dim}], got {tuple(values.shape)}"
            )
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt[..., 0]
        if dt.ndim != 2 or dt.shape != values.shape[:2]:
            raise ValueError("dt must have shape [batch,time] or [batch,time,1]")
        if not torch.isfinite(values).all() or not torch.isfinite(dt).all():
            raise ValueError("values and dt must be finite")
        if torch.any(dt <= 0):
            raise ValueError("elapsed times must be strictly positive")


class LSTMEncoder(SequenceEncoder):
    uses_elapsed_time = False

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.core = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, values: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        self.validate(values, dt, self.input_dim)
        output, _ = self.core(values)
        return output


class ODERNNEncoder(SequenceEncoder):
    """Continuous hidden evolution followed by a discrete observation update."""

    uses_elapsed_time = True

    def __init__(self, input_dim: int, hidden_dim: int, max_step: float = 1.0) -> None:
        super().__init__()
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_step = float(max_step)
        self.vector_field = nn.Linear(hidden_dim, hidden_dim)
        self.observation_update = nn.GRUCell(input_dim, hidden_dim)

    def _field(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.vector_field(state))

    def _evolve(self, state: torch.Tensor, elapsed: torch.Tensor) -> torch.Tensor:
        # A common number of substeps preserves vectorization while every sample
        # keeps its own elapsed time.  The integrator setting is logged in configs.
        substeps = max(
            1, int(math.ceil(float(elapsed.detach().max().cpu()) / self.max_step))
        )
        step = elapsed[:, None] / substeps
        for _ in range(substeps):
            k1 = self._field(state)
            k2 = self._field(state + 0.5 * step * k1)
            k3 = self._field(state + 0.5 * step * k2)
            k4 = self._field(state + step * k3)
            state = state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return state

    def forward(self, values: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        self.validate(values, dt, self.input_dim)
        if dt.ndim == 3:
            dt = dt[..., 0]
        state = values.new_zeros(values.shape[0], self.hidden_dim)
        sequence = []
        for step_index in range(values.shape[1]):
            state = self._evolve(state, dt[:, step_index])
            state = self.observation_update(values[:, step_index], state)
            sequence.append(state)
        return torch.stack(sequence, dim=1)


class LiquidEncoder(SequenceEncoder):
    """Dense ncps LTC/CfC encoder with exact per-sample time spans."""

    uses_elapsed_time = True

    def __init__(
        self,
        kind: Literal["ltc", "cfc"],
        input_dim: int,
        hidden_dim: int,
        *,
        ode_unfolds: int = 6,
        cfc_mode: str = "default",
    ) -> None:
        super().__init__()
        try:
            from ncps.torch import CfC, LTC
        except ImportError as exc:  # pragma: no cover - exercised by installation smoke
            raise RuntimeError("LTC/CfC offline models require ncps") from exc
        self.kind = str(kind)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        # Keep the registered numerical setting available for provenance and
        # compute accounting.  This plain Python attribute is intentionally not
        # part of the state dict and does not alter the ncps numerical path.
        self.ode_unfolds = int(ode_unfolds) if kind == "ltc" else None
        common = dict(
            input_size=input_dim,
            units=hidden_dim,
            return_sequences=True,
            batch_first=True,
        )
        if kind == "ltc":
            self.core = LTC(**common, ode_unfolds=int(self.ode_unfolds))
        elif kind == "cfc":
            self.core = CfC(**common, mode=str(cfc_mode))
        else:
            raise ValueError(f"Unsupported liquid encoder: {kind}")

    def forward(self, values: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        self.validate(values, dt, self.input_dim)
        if dt.ndim == 3:
            dt = dt[..., 0]

        # ncps 1.0.1 calls ``timespans[:, t].squeeze()`` inside its sequence
        # wrapper.  For B>1 that produces [B], which does not broadcast against
        # a [B,H] state unless B happens to equal H.  Calling the public cell
        # interface directly with [B,1] elapsed times preserves the exact ncps
        # equations, remains fully differentiable, and avoids a slow per-sample
        # fallback.  These constructors never enable mixed memory.
        if bool(getattr(self.core, "use_mixed", False)):
            raise RuntimeError("The registered dense LTC/CfC models must not use mixed memory")
        state = values.new_zeros(values.shape[0], int(self.core.state_size))
        outputs: list[torch.Tensor] = []
        for step_index in range(values.shape[1]):
            elapsed = dt[:, step_index : step_index + 1]
            try:
                output, state = self.core.rnn_cell.forward(
                    values[:, step_index], state, elapsed
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{self.kind.upper()} rejected explicit per-sample elapsed times "
                    f"at sequence step {step_index}; unit spacing is not an allowed fallback"
                ) from exc
            if self.kind == "cfc":
                output = self.core.fc(output)
            outputs.append(output)
        return torch.stack(outputs, dim=1)


def resolve_hidden_dim(
    model_config: dict[str, Any], dataset_name: str, model_name: str
) -> int:
    """Resolve a registered width, supporting task-specific parameter matching."""

    hidden_dims = model_config.get("hidden_dims", {})
    selected: Any = hidden_dims
    if isinstance(hidden_dims, dict) and dataset_name in hidden_dims:
        selected = hidden_dims[dataset_name]
    if isinstance(selected, dict) and model_name in selected:
        value = selected[model_name]
    else:
        value = model_config.get("hidden_dim", 64)
    width = int(value)
    if width <= 0:
        raise ValueError(
            f"hidden dimension for {dataset_name}/{model_name} must be positive"
        )
    return width


class OfflineSequenceClassifier(nn.Module):
    """Sequence-to-label model with an explicit task head."""

    def __init__(self, encoder: SequenceEncoder, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.encoder = encoder
        self.head = nn.Linear(encoder.hidden_dim, int(output_dim))
        self.output_dim = int(output_dim)

    @property
    def input_dim(self) -> int:
        return self.encoder.input_dim

    @property
    def uses_elapsed_time(self) -> bool:
        return bool(self.encoder.uses_elapsed_time)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward_features(self, values: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return self.encoder(values, dt)

    def forward(
        self,
        values: torch.Tensor,
        dt: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence = self.forward_features(values, dt)
        if lengths is None:
            final = sequence[:, -1]
        else:
            if lengths.ndim != 1 or lengths.shape[0] != sequence.shape[0]:
                raise ValueError("lengths must have shape [batch]")
            if torch.any(lengths <= 0) or torch.any(lengths > sequence.shape[1]):
                raise ValueError("lengths values fall outside the padded sequence")
            indices = (lengths.to(device=sequence.device, dtype=torch.long) - 1).view(-1, 1, 1)
            indices = indices.expand(-1, 1, sequence.shape[-1])
            final = sequence.gather(1, indices).squeeze(1)
        return self.head(final)


def build_offline_model(
    name: str,
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    ode_max_step: float = 1.0,
    ltc_ode_unfolds: int = 6,
    cfc_mode: str = "default",
) -> OfflineSequenceClassifier:
    """Build one mandatory comparator under a shared classifier interface."""

    normalized = name.lower().replace("_", "").replace("-", "")
    if normalized == "lstm":
        encoder: SequenceEncoder = LSTMEncoder(input_dim, hidden_dim)
    elif normalized in {"odernn", "ode"}:
        encoder = ODERNNEncoder(input_dim, hidden_dim, max_step=ode_max_step)
    elif normalized == "ltc":
        encoder = LiquidEncoder(
            "ltc", input_dim, hidden_dim, ode_unfolds=ltc_ode_unfolds
        )
    elif normalized == "cfc":
        encoder = LiquidEncoder("cfc", input_dim, hidden_dim, cfc_mode=cfc_mode)
    else:
        raise ValueError(f"Unknown offline model family: {name}")
    return OfflineSequenceClassifier(encoder, output_dim)

