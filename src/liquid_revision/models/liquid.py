"""Strict ncps LTC/CfC observers that always pass explicit elapsed time."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import TemporalPolicy
from .topology import TopologySpec, make_ncps_wiring


def _direct_graph_mask_sha256(mask: torch.Tensor) -> str:
    """Hash a graph mask together with the tensor properties that define it."""
    value = mask.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def direct_graph_mask_manifest(module: nn.Module) -> dict[str, dict[str, Any]]:
    """Return deterministic evidence for every registry-built direct CfC mask."""
    manifest: dict[str, dict[str, Any]] = {}
    for name, child in module.named_modules():
        if not isinstance(child, DirectGraphCfCCell):
            continue
        mask = child.sparsity_mask
        manifest[name or "<root>"] = {
            "shape": [int(item) for item in mask.shape],
            "dtype": str(mask.dtype),
            "nonzero": int(torch.count_nonzero(mask).item()),
            "sha256": _direct_graph_mask_sha256(mask),
            "checkpoint_persistent": False,
        }
    return manifest


def _validate_direct_graph_mask_integrity(module: nn.Module) -> None:
    """Validate masks against the deterministic audit created with the model."""
    manifest = direct_graph_mask_manifest(module)
    expected_audits = [
        audit
        for child in module.modules()
        if isinstance((audit := getattr(child, "topology_audit", None)), dict)
        and "applied_mask_sha256" in audit
    ]
    if len(manifest) != len(expected_audits):
        raise RuntimeError("Direct graph mask/audit cardinality mismatch")
    observed = list(manifest.values())
    for evidence, expected in zip(observed, expected_audits, strict=True):
        if (
            evidence["shape"] != expected.get("applied_mask_shape")
            or evidence["dtype"] != expected.get("applied_mask_dtype")
            or evidence["nonzero"] != expected.get("applied_mask_nonzero")
            or evidence["sha256"] != expected.get("applied_mask_sha256")
            or expected.get("mask_checkpoint_persistent") is not False
        ):
            raise RuntimeError(
                "Direct graph mask differs from the registry-built topology audit"
            )


def load_graph_safe_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Any:
    """Load weights without permitting a checkpoint to supply graph topology.

    Direct Graph-CfC masks are rebuilt from the frozen parameter registry and
    are deliberately absent from checkpoints.  Rejecting every mask-named key
    before calling PyTorch also fails closed for legacy, forged, and
    ``strict=False`` callers.  The exact constructor-built tensor is snapshotted
    and revalidated (shape, dtype, value, and hash) after weight loading.
    """
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint model_state_dict must be a mapping")
    cells = {
        name or "<root>": child
        for name, child in module.named_modules()
        if isinstance(child, DirectGraphCfCCell)
    }
    if cells:
        forbidden = sorted(
            str(key)
            for key in state_dict
            if any("mask" in part.casefold() for part in str(key).split("."))
        )
        if forbidden:
            raise ValueError(
                "Checkpoint state_dict must not contain Direct Graph-CfC mask "
                f"keys: {forbidden}"
            )
    _validate_direct_graph_mask_integrity(module)
    before = {
        name: {
            "tensor": cell.sparsity_mask.detach().clone(),
            "shape": tuple(cell.sparsity_mask.shape),
            "dtype": cell.sparsity_mask.dtype,
            "sha256": _direct_graph_mask_sha256(cell.sparsity_mask),
        }
        for name, cell in cells.items()
    }
    result = module.load_state_dict(state_dict, strict=strict)
    after_cells = {
        name or "<root>": child
        for name, child in module.named_modules()
        if isinstance(child, DirectGraphCfCCell)
    }
    if set(after_cells) != set(before):
        raise RuntimeError("Direct graph cell set changed while loading weights")
    for name, expected in before.items():
        observed = after_cells[name].sparsity_mask
        if (
            tuple(observed.shape) != expected["shape"]
            or observed.dtype != expected["dtype"]
            or not torch.equal(observed, expected["tensor"].to(observed.device))
            or _direct_graph_mask_sha256(observed) != expected["sha256"]
        ):
            raise RuntimeError(
                "Checkpoint load changed the registry-built Direct Graph-CfC mask"
            )
    _validate_direct_graph_mask_integrity(module)
    return result


class MotorSlice(nn.Module):
    """Select the registered motor nodes from a synchronous global graph."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., : self.output_dim]


class DirectGraphCfCCell(nn.Module):
    """Backbone-free CfC cell with one explicit mask on every affine head.

    Dense CfC uses an all-one mask; NCP and degree-matched random CfC use their
    frozen graph masks.  Keeping the cell, update equation, initialization and
    motor-node readout identical makes the registered topology intervention the
    mask itself rather than an upstream layered-wiring execution path.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        mode: str,
        mask: torch.Tensor,
    ) -> None:
        super().__init__()
        if mode not in {"default", "pure", "no_gate"}:
            raise ValueError(f"Unsupported CfC mode: {mode}")
        expected = (int(hidden_size), int(input_size) + int(hidden_size))
        if tuple(mask.shape) != expected:
            raise ValueError(f"CfC graph mask {tuple(mask.shape)} != {expected}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.mode = str(mode)
        self.backbone_layers = 0
        frozen_mask = mask.detach().to(dtype=torch.float32).abs().clone()
        if not bool(torch.isfinite(frozen_mask).all()) or not bool(
            torch.logical_or(frozen_mask == 0, frozen_mask == 1).all()
        ):
            raise ValueError("CfC graph mask must be finite and binary")
        # The topology is reconstructed from the frozen registry, never from a
        # checkpoint.  ``persistent=False`` makes it impossible for ordinary
        # state-dict serialization to turn graph identity into learned state.
        self.register_buffer("sparsity_mask", frozen_mask, persistent=False)
        features = self.input_size + self.hidden_size
        self.ff1 = nn.Linear(features, self.hidden_size)
        if self.mode == "pure":
            self.w_tau = nn.Parameter(torch.zeros(1, self.hidden_size))
            self.A = nn.Parameter(torch.ones(1, self.hidden_size))
        else:
            self.ff2 = nn.Linear(features, self.hidden_size)
            self.time_a = nn.Linear(features, self.hidden_size)
            self.time_b = nn.Linear(features, self.hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for parameter in self.parameters():
            if parameter.dim() == 2 and parameter.requires_grad:
                nn.init.xavier_uniform_(parameter)

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Reject topology-bearing keys even if a caller requests non-strict load."""
        forbidden = sorted(
            str(key)
            for key in state_dict
            if str(key).startswith(prefix)
            and any(
                "mask" in part.casefold()
                for part in str(key)[len(prefix) :].split(".")
            )
        )
        if forbidden:
            error_msgs.append(
                "Direct Graph-CfC topology masks cannot be loaded from a "
                f"checkpoint: {forbidden}"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _masked(self, name: str, combined: torch.Tensor) -> torch.Tensor:
        layer = getattr(self, name)
        return F.linear(
            combined, layer.weight * self.sparsity_mask, layer.bias
        )

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor, elapsed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([inputs, hidden], dim=1)
        ff1 = self._masked("ff1", combined)
        if self.mode == "pure":
            new_hidden = (
                -self.A
                * torch.exp(-elapsed * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            ff2 = self._masked("ff2", combined)
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)
            time_a = self._masked("time_a", combined)
            time_b = self._masked("time_b", combined)
            interpolation = self.sigmoid(time_a * elapsed + time_b)
            if self.mode == "no_gate":
                new_hidden = ff1 + interpolation * ff2
            else:
                new_hidden = ff1 * (1.0 - interpolation) + interpolation * ff2
        return new_hidden, new_hidden


class _SingleCellStack(nn.Module):
    """Compatibility container exposing the audited cell as ``_layers[0]``."""

    def __init__(self, cell: nn.Module) -> None:
        super().__init__()
        self._layers = nn.ModuleList([cell])


class DirectGraphCfCCore(nn.Module):
    """Minimal ncps-wrapper-compatible container for the direct CfC cell."""

    def __init__(self, cell: DirectGraphCfCCell) -> None:
        super().__init__()
        self.rnn_cell = _SingleCellStack(cell)
        self.state_size = int(cell.hidden_size)
        self.use_mixed = False

    def forward(
        self,
        sequence: torch.Tensor,
        *,
        timespans: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Independent batch-one-compatible reference wrapper used in audits."""
        if sequence.ndim != 3 or timespans.shape != sequence.shape[:2]:
            raise ValueError("Expected sequence [B,T,F] and timespans [B,T]")
        hidden = (
            torch.zeros(
                sequence.shape[0],
                self.state_size,
                device=sequence.device,
                dtype=sequence.dtype,
            )
            if hx is None
            else hx
        )
        outputs = []
        cell = self.rnn_cell._layers[0]
        for step in range(sequence.shape[1]):
            output, hidden = cell(
                sequence[:, step], hidden, timespans[:, step, None]
            )
            outputs.append(output)
        return torch.stack(outputs, dim=1), hidden


class NCPSPolicy(TemporalPolicy):
    def __init__(
        self,
        kind: Literal["ltc", "cfc"],
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        topology: TopologySpec,
        cfc_mode: str = "default",
        ode_unfolds: int = 6,
        wiring_override: Any | None = None,
        topology_audit: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        try:
            from ncps.torch import LTC
        except ImportError as exc:
            raise RuntimeError("LTC/CfC policies require the ncps package") from exc
        self.kind = kind
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.topology = topology
        self.topology_audit = dict(topology_audit or {})
        self.elapsed_time_adapter = {
            "name": (
                "local_direct_graph_cfc_batch_loop"
                if kind == "cfc"
                else "ncps_1_0_1_batch_safe_cell_loop"
            ),
            "reason": (
                "use one explicit four-head mask and equation for dense and sparse CfC"
                if kind == "cfc"
                else "avoid upstream timespans[:, t].squeeze() batch broadcast bug"
            ),
            "upstream_package_modified": False,
            "cell_elapsed_time_shape": "[batch,1]",
        }
        self._timespan_rank: int | None = None
        self._timespan_mode: str | None = None

        if kind == "ltc":
            if wiring_override is not None:
                units: Any = wiring_override
                self.head = nn.Identity()
                if int(getattr(units, "units", -1)) != hidden_dim:
                    raise ValueError("wiring_override unit count does not match hidden_dim")
                if int(getattr(units, "output_dim", -1)) != output_dim:
                    raise ValueError("wiring_override output count does not match output_dim")
            elif topology.kind == "dense":
                units = hidden_dim
                self.head = nn.Linear(hidden_dim, output_dim)
            else:
                units = make_ncps_wiring(hidden_dim, output_dim, topology)
                self.head = nn.Identity()
            common = dict(
                input_size=input_dim,
                units=units,
                return_sequences=True,
                batch_first=True,
            )
            self.core = LTC(**common, ode_unfolds=int(ode_unfolds))
        elif kind == "cfc":
            if wiring_override is None:
                if topology.kind != "dense":
                    raise ValueError(
                        "Sparse CfC requires an explicitly frozen wiring_override"
                    )
                mask = torch.ones(
                    hidden_dim, input_dim + hidden_dim, dtype=torch.float32
                )
                self.topology_audit.update(
                    {
                        "selected_graph": "dense_all_one",
                        "recurrent_edges": hidden_dim * hidden_dim,
                        "sensory_edges": input_dim * hidden_dim,
                    }
                )
            else:
                if int(getattr(wiring_override, "units", -1)) != hidden_dim:
                    raise ValueError("wiring_override unit count does not match hidden_dim")
                if int(getattr(wiring_override, "output_dim", -1)) != output_dim:
                    raise ValueError("wiring_override output count does not match output_dim")
                combined = np.concatenate(
                    [
                        np.asarray(wiring_override.sensory_adjacency_matrix),
                        np.asarray(wiring_override.adjacency_matrix),
                    ],
                    axis=0,
                )
                mask = torch.as_tensor(np.abs(combined.T), dtype=torch.float32)
                expected_edges = int(self.topology_audit["sensory_edges"]) + int(
                    self.topology_audit["recurrent_edges"]
                )
                if int(torch.count_nonzero(mask).item()) != expected_edges:
                    raise RuntimeError("Topology edge count differs from frozen audit")
            cell = DirectGraphCfCCell(
                input_dim,
                hidden_dim,
                mode=cfc_mode,
                mask=mask,
            )
            self.core = DirectGraphCfCCore(cell)
            self.head = MotorSlice(output_dim)
            active_edges = int(torch.count_nonzero(mask).item())
            self.topology_audit.update(
                {
                    "global_graph_mask_applied": True,
                    "applied_mask_nonzero": active_edges,
                    "masked_cfc_heads": ["ff1", "ff2", "time_a", "time_b"],
                    "all_cfc_paths_respect_graph_mask": True,
                    "motor_node_indices": list(range(output_dim)),
                    "graph_cell_implementation": (
                        "local_full_state_synchronous_direct_graph_cfc"
                    ),
                    "dense_sparse_common_cell": True,
                    "applied_mask_shape": [int(item) for item in mask.shape],
                    "applied_mask_dtype": str(cell.sparsity_mask.dtype),
                    "applied_mask_sha256": _direct_graph_mask_sha256(
                        cell.sparsity_mask
                    ),
                    "mask_checkpoint_persistent": False,
                }
            )
        else:
            raise ValueError(kind)

    def _global_graph_cfc_step(
        self, inputs: torch.Tensor, hidden: torch.Tensor, elapsed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CfC step whose state and both time-gate heads share one graph mask.

        ncps 1.0.1 masks ``ff1`` and ``ff2`` but leaves ``time_a`` and
        ``time_b`` dense.  That behavior is valid for the upstream implementation
        but would introduce forbidden global paths into our topology contrast.
        This local, batch-safe implementation mirrors CfCCell while applying the
        registered graph to all four affine heads.  Site-packages are unchanged.
        """
        cell = self.core.rnn_cell._layers[0]
        if not isinstance(cell, DirectGraphCfCCell):
            raise RuntimeError("Direct graph CfC cell is not installed")
        return cell(inputs, hidden, elapsed)

    def initial_state(self, batch_size: int, device: torch.device | str):
        del batch_size, device
        return None

    @staticmethod
    def _state_slice(state: Any, index: int) -> Any:
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(item[index : index + 1] for item in state)
        return state[index : index + 1]

    @staticmethod
    def _state_cat(states: list[Any]) -> Any:
        if states[0] is None:
            return None
        if isinstance(states[0], tuple):
            return tuple(
                torch.cat([state[index] for state in states], dim=0)
                for index in range(len(states[0]))
            )
        return torch.cat(states, dim=0)

    def _run_core(
        self, sequence: torch.Tensor, dt: torch.Tensor, state: Any
    ) -> tuple[torch.Tensor, Any]:
        """Run ncps cells with a batch-safe explicit elapsed-time column.

        ncps 1.0.1 uses ``timespans[:, t].squeeze()``.  With batch size greater
        than one this produces ``[B]``, which does not broadcast against
        ``[B,H]`` and also makes the intended per-sample time axis ambiguous.
        This local adapter mirrors the upstream wrapper while passing ``[B,1]``
        directly to the unchanged cell.  It keeps batching efficient and never
        discards elapsed time.
        """
        if sequence.ndim != 3 or dt.ndim != 2 or sequence.shape[:2] != dt.shape:
            raise ValueError("sequence and dt must have shapes [B,T,F] and [B,T]")
        batch_size = sequence.shape[0]
        if state is None:
            h_state = torch.zeros(
                (batch_size, self.core.state_size),
                device=sequence.device,
                dtype=sequence.dtype,
            )
            c_state = (
                torch.zeros_like(h_state) if bool(self.core.use_mixed) else None
            )
        else:
            if bool(self.core.use_mixed):
                if not isinstance(state, tuple) or len(state) != 2:
                    raise ValueError("Mixed-memory ncps state must be (hidden, cell)")
                h_state, c_state = state
            else:
                h_state, c_state = state, None
        outputs: list[torch.Tensor] = []
        for step in range(sequence.shape[1]):
            inputs = sequence[:, step]
            elapsed = dt[:, step, None]
            if bool(self.core.use_mixed):
                h_state, c_state = self.core.lstm(inputs, (h_state, c_state))
            if self.kind == "cfc":
                h_out, h_state = self._global_graph_cfc_step(
                    inputs, h_state, elapsed
                )
            else:
                h_out, h_state = self.core.rnn_cell.forward(
                    inputs, h_state, elapsed
                )
            readout = self.core.fc(h_out) if hasattr(self.core, "fc") else h_out
            outputs.append(readout)
        self._timespan_rank = 2
        self._timespan_mode = "batch_safe_cell_loop"
        next_state = (h_state, c_state) if bool(self.core.use_mixed) else h_state
        return torch.stack(outputs, dim=1), next_state

    def forward_step(self, x: torch.Tensor, dt: torch.Tensor, state: Any):
        output, state = self._run_core(x[:, None, :], dt[:, None], state)
        action_features = output[:, -1]
        return self.head(action_features), state

    def forward_sequence(
        self, x: torch.Tensor, dt: torch.Tensor, state: Any | None = None
    ) -> tuple[torch.Tensor, Any]:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected x [batch,time,{self.input_dim}], got {tuple(x.shape)}")
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt[..., 0]
        if dt.ndim != 2 or dt.shape != x.shape[:2]:
            raise ValueError("dt must have shape [batch,time] or [batch,time,1]")
        if torch.any(~torch.isfinite(x)) or torch.any(~torch.isfinite(dt)):
            raise ValueError("x and dt must be finite")
        if torch.any(dt <= 0):
            raise ValueError("All time deltas must be positive")
        output, state = self._run_core(x, dt, state)
        return self.head(output), state

