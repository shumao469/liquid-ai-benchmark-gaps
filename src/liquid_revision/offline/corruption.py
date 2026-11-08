"""Deterministic fixed-checkpoint missingness perturbations.

Corruptions only remove observations.  They can never turn a naturally missing
PhysioNet entry into an observed value, and they always return the updated mask
and age-since-last-observation tensor used by the classifier.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal

import numpy as np


CorruptionKind = Literal["mcar", "temporal_block"]


@dataclass(frozen=True)
class CorruptionSpec:
    kind: CorruptionKind
    severity: float
    seed: int
    packetwise: bool = False

    def validate(self) -> None:
        if self.kind not in {"mcar", "temporal_block"}:
            raise ValueError(f"Unsupported corruption kind: {self.kind}")
        if not 0.0 <= float(self.severity) <= 1.0:
            raise ValueError("severity must be in [0, 1]")


@dataclass(frozen=True)
class CorruptedBatch:
    values: np.ndarray
    mask: np.ndarray
    age: np.ndarray
    metadata: dict[str, float | int | str | bool]


def derive_corruption_seed(base_seed: int, *tokens: object) -> int:
    """Stable seed derivation independent of Python's randomized hash()."""

    digest = hashlib.sha256(
        "|".join([str(int(base_seed)), *(str(token) for token in tokens)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def recompute_age(mask: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Compute per-variable elapsed time since its most recent observation."""

    observed = np.asarray(mask, dtype=bool)
    elapsed = np.asarray(dt, dtype=np.float32)
    if observed.ndim != 3:
        raise ValueError("mask must have shape [sample,time,feature]")
    if elapsed.ndim == 3 and elapsed.shape[-1] == 1:
        elapsed = elapsed[..., 0]
    if elapsed.shape != observed.shape[:2]:
        raise ValueError("dt must have shape [sample,time]")
    if not np.isfinite(elapsed).all() or np.any(elapsed <= 0):
        raise ValueError("dt must be finite and strictly positive")

    running = np.zeros((observed.shape[0], observed.shape[2]), dtype=np.float32)
    age = np.empty(observed.shape, dtype=np.float32)
    for index in range(observed.shape[1]):
        running += elapsed[:, index, None]
        running[observed[:, index]] = 0.0
        age[:, index] = running
    return age


def _validate_arrays(
    values: np.ndarray, mask: np.ndarray, dt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_array = np.asarray(values, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=bool)
    dt_array = np.asarray(dt, dtype=np.float32)
    if values_array.ndim != 3 or mask_array.shape != values_array.shape:
        raise ValueError("values and mask must share shape [sample,time,feature]")
    if dt_array.ndim == 3 and dt_array.shape[-1] == 1:
        dt_array = dt_array[..., 0]
    if dt_array.shape != values_array.shape[:2]:
        raise ValueError("dt must have shape [sample,time]")
    if not np.isfinite(values_array).all():
        raise ValueError("values must be finite before corruption")
    if not np.isfinite(dt_array).all() or np.any(dt_array <= 0):
        raise ValueError("dt must be finite and strictly positive")
    return values_array, mask_array, dt_array


def apply_corruption(
    values: np.ndarray,
    original_mask: np.ndarray,
    dt: np.ndarray,
    spec: CorruptionSpec,
) -> CorruptedBatch:
    """Remove observations according to a deterministic perturbation schedule."""

    spec.validate()
    values_array, original, dt_array = _validate_arrays(values, original_mask, dt)
    rng = np.random.default_rng(int(spec.seed))
    retained = original.copy()
    sample_count, time_count, feature_count = retained.shape

    if spec.kind == "mcar":
        if spec.packetwise:
            keep = rng.random((sample_count, time_count, 1)) >= spec.severity
        else:
            keep = rng.random(retained.shape) >= spec.severity
        retained &= np.broadcast_to(keep, retained.shape)
    else:
        block_length = min(time_count, int(round(float(spec.severity) * time_count)))
        if block_length > 0:
            for sample_index in range(sample_count):
                start = int(rng.integers(0, time_count - block_length + 1))
                retained[sample_index, start : start + block_length, :] = False

    # This invariant is the core protection for naturally missing clinical data.
    if np.any(retained & ~original):
        raise AssertionError("corruption relabelled a naturally missing entry as observed")

    corrupted = values_array.copy()
    corrupted[~retained] = 0.0
    original_observations = int(original.sum())
    retained_observations = int(retained.sum())
    removed = original_observations - retained_observations
    actual = removed / original_observations if original_observations else 0.0
    metadata: dict[str, float | int | str | bool] = {
        "kind": spec.kind,
        "requested_severity": float(spec.severity),
        "seed": int(spec.seed),
        "packetwise": bool(spec.packetwise),
        "samples": int(sample_count),
        "time_steps": int(time_count),
        "features": int(feature_count),
        "original_observations": original_observations,
        "removed_observations": int(removed),
        "actual_removed_fraction_of_observed": float(actual),
    }
    return CorruptedBatch(
        values=corrupted,
        mask=retained,
        age=recompute_age(retained, dt_array),
        metadata=metadata,
    )


def default_corruption_specs(
    *,
    base_seed: int,
    dataset: str,
    packetwise: bool,
    mcar_levels: Sequence[float] | None = None,
    temporal_block_levels: Sequence[float] | None = None,
) -> list[CorruptionSpec]:
    """Registered MCAR and temporal-block curves with fixed, auditable seeds."""

    def validated_levels(
        supplied: Sequence[float] | None, defaults: np.ndarray, name: str
    ) -> list[float]:
        levels = [float(value) for value in (defaults if supplied is None else supplied)]
        if not levels:
            raise ValueError(f"{name} must contain at least one severity")
        if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in levels):
            raise ValueError(f"{name} severities must be finite and in [0, 1]")
        rounded = [float(np.round(value, 10)) for value in levels]
        if len(set(rounded)) != len(rounded):
            raise ValueError(f"{name} contains duplicate severities")
        return rounded

    mcar = validated_levels(mcar_levels, np.arange(0.0, 1.0, 0.1), "mcar_levels")
    blocks = validated_levels(
        temporal_block_levels, np.arange(0.0, 0.7, 0.1), "temporal_block_levels"
    )

    specs: list[CorruptionSpec] = []
    for level in mcar:
        specs.append(
            CorruptionSpec(
                "mcar",
                float(level),
                derive_corruption_seed(base_seed, dataset, "mcar", f"{level:.1f}"),
                packetwise=packetwise,
            )
        )
    for level in blocks:
        specs.append(
            CorruptionSpec(
                "temporal_block",
                float(level),
                derive_corruption_seed(
                    base_seed, dataset, "temporal_block", f"{level:.1f}"
                ),
                packetwise=packetwise,
            )
        )
    return specs

