"""Training-derived output scales for the three Graph-CfC observers."""

from __future__ import annotations

import numpy as np
import torch

from ..closed_loop.parameter_matching import build_registered_model
from ..closed_loop.training import Normalizer, fit_normalizer
from ..models.liquid import MotorSlice


GRAPH_CFC_MODELS = ("cfc", "ncp_cfc", "random_sparse_cfc")
MARGIN = 1.05


def train_range_radius(arrays: dict, normalizer: Normalizer | None = None) -> np.ndarray:
    """Set each symmetric motor range just beyond the standardized train targets."""
    if normalizer is None:
        normalizer = fit_normalizer(arrays)
    targets = np.asarray(arrays["target_state"], dtype=np.float32)
    standardized = (targets - normalizer.target_mean) / normalizer.target_std
    return (MARGIN * np.max(np.abs(standardized), axis=0)).astype(np.float32)


class TrainRangeMotorSlice(MotorSlice):
    def __init__(self, radius: list[float] | np.ndarray) -> None:
        value = torch.as_tensor(radius, dtype=torch.float32).detach().clone()
        if value.shape != (4,) or not torch.isfinite(value).all() or not (value > 1).all():
            raise ValueError("Expected four finite train-derived motor radii greater than one")
        super().__init__(len(value))
        self.register_buffer("target_radius", value, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value) * self.target_radius


def build_readout_model(
    name: str, config: dict, registry: dict, radius: list[float] | None
) -> torch.nn.Module:
    model = build_registered_model(name, config, registry)
    if radius is not None:
        if name not in GRAPH_CFC_MODELS:
            raise ValueError("Channel scales apply to the three Graph-CfC observers")
        if type(model.head) is not MotorSlice or model.core.rnn_cell._layers[0].mode != "default":
            raise ValueError("Channel scales require the default-mode Graph-CfC motor slice")
        model.head = TrainRangeMotorSlice(radius)
    return model
