"""Single auditable observer factory for training, ablation, and smoke tests."""

from __future__ import annotations

from typing import Any

from .base import TemporalPolicy
from .liquid import NCPSPolicy
from .native import (
    FixedTauCTRNNPolicy,
    GRUPolicy,
    LSTMPolicy,
    MLPPolicy,
    ODERNNPolicy,
)
from .topology import TopologySpec


def build_model(name: str, model_config: dict[str, Any]) -> TemporalPolicy:
    name = name.lower()
    input_dim = int(model_config["input_dim"])
    output_dim = int(model_config.get("state_dim", model_config.get("action_dim", 0)))
    if output_dim <= 0:
        raise ValueError("models.state_dim must be positive")
    hidden_dim = int(model_config["hidden_dim"])
    topology = TopologySpec.from_mapping(model_config.get("topology"))

    if name in {"mlp", "gru", "lstm"} and topology.kind != "dense":
        raise ValueError(f"{name} uses dense topology; use CT-RNN/ODE-RNN/liquid for topology ablation")
    if name == "mlp":
        return MLPPolicy(input_dim, hidden_dim, output_dim)
    if name == "gru":
        return GRUPolicy(input_dim, hidden_dim, output_dim)
    if name == "lstm":
        return LSTMPolicy(input_dim, hidden_dim, output_dim)
    if name == "ctrnn":
        config = dict(model_config.get("ctrnn", {}))
        return FixedTauCTRNNPolicy(
            input_dim,
            hidden_dim,
            output_dim,
            topology=topology,
            initial_tau=float(config.get("initial_tau", 0.20)),
            min_tau=float(config.get("min_tau", 0.001)),
            max_step=float(config.get("max_step", 0.02)),
        )
    if name == "odernn":
        config = dict(model_config.get("ode", {}))
        return ODERNNPolicy(
            input_dim,
            hidden_dim,
            output_dim,
            topology=topology,
            max_step=float(config.get("max_step", 0.02)),
        )
    if name in {"ltc", "cfc"}:
        liquid_config = dict(model_config.get("liquid", {}))
        return NCPSPolicy(
            name,
            input_dim,
            hidden_dim,
            output_dim,
            topology=topology,
            cfc_mode=str(liquid_config.get("cfc_mode", "default")),
            ode_unfolds=int(liquid_config.get("ode_unfolds", 6)),
            wiring_override=model_config.get("_wiring_override"),
            topology_audit=model_config.get("_topology_audit"),
        )
    raise ValueError(f"Unknown model: {name}")

