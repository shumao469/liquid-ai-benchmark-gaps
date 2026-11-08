"""Pre-generated, paired perturbation schedules for closed-loop evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PerturbationSpec:
    iid_dropout_probability: float = 0.0
    burst_start_probability: float = 0.0
    burst_stationary_probability: float = 0.0
    burst_length_steps: int = 0
    controlled_single_burst: bool = False
    delay_steps: int = 0
    sensor_noise_std: float = 0.0
    timestamp_measurement_jitter_std: float = 0.0
    actuator_gain: float = 1.0
    impulse_magnitude: float = 0.0
    impulse_duration_steps: int = 1
    channel_outage_channels: tuple[int, ...] = ()

    def validate(self, sensor_dim: int = 4) -> None:
        for name in (
            "iid_dropout_probability",
            "burst_start_probability",
            "burst_stationary_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.burst_length_steps < 0 or self.delay_steps < 0:
            raise ValueError("burst_length_steps and delay_steps must be non-negative")
        if self.sensor_noise_std < 0 or self.timestamp_measurement_jitter_std < 0:
            raise ValueError("Noise and timestamp jitter must be non-negative")
        if not 0.0 <= self.actuator_gain <= 2.0:
            raise ValueError("actuator_gain must be in [0, 2]")
        if self.impulse_magnitude < 0 or self.impulse_duration_steps <= 0:
            raise ValueError("Impulse magnitude/duration are invalid")
        if any(index < 0 or index >= sensor_dim for index in self.channel_outage_channels):
            raise ValueError("channel_outage_channels contains an invalid sensor index")
        active_burst_generators = sum(
            [
                self.burst_start_probability > 0,
                self.burst_stationary_probability > 0,
                bool(self.controlled_single_burst),
            ]
        )
        if active_burst_generators > 1:
            raise ValueError("Burst generators are mutually exclusive")
        if active_burst_generators and self.burst_length_steps <= 0:
            raise ValueError("An active burst generator requires burst_length_steps > 0")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "PerturbationSpec":
        values = dict(mapping or {})
        if "channel_outage_channels" in values:
            values["channel_outage_channels"] = tuple(values["channel_outage_channels"])
        spec = cls(**values)
        spec.validate()
        return spec


@dataclass(frozen=True)
class PerturbationSchedule:
    sensor_mask: np.ndarray
    sensor_noise: np.ndarray
    observed_timestamp_dt: np.ndarray
    actuator_gain: np.ndarray
    external_force: np.ndarray
    delay_steps: int
    seed: int
    spec: PerturbationSpec

    def metadata(self) -> dict[str, Any]:
        runs = packet_loss_run_lengths(self.sensor_mask)
        return {
            "seed": self.seed,
            "delay_steps": self.delay_steps,
            "spec": asdict(self.spec),
            "masked_fraction": float(1.0 - self.sensor_mask.mean()),
            "packet_loss_run_count": len(runs),
            "packet_loss_run_mean_steps": float(np.mean(runs)) if runs else 0.0,
            "packet_loss_run_steps_total": int(sum(runs)),
            "impulse_steps": int(np.any(self.external_force != 0.0, axis=1).sum()),
        }


def make_perturbation_schedule(
    *,
    horizon_steps: int,
    nominal_dt: float,
    spec: PerturbationSpec,
    seed: int,
    sensor_dim: int = 4,
) -> PerturbationSchedule:
    if horizon_steps <= 0 or nominal_dt <= 0:
        raise ValueError("horizon_steps and nominal_dt must be positive")
    spec.validate(sensor_dim=sensor_dim)
    rng = np.random.default_rng(int(seed))

    # The registered sensor fault is loss of one complete proprioceptive packet,
    # not four independent channel faults.  Draw once per control step and
    # broadcast to all q/dq channels; explicit channel-outage experiments remain
    # available through ``channel_outage_channels`` below.
    packet_observed = (
        rng.random((horizon_steps, 1)) >= spec.iid_dropout_probability
    )
    mask = np.broadcast_to(packet_observed, (horizon_steps, sensor_dim)).copy()
    if spec.controlled_single_burst:
        start = max(0, horizon_steps // 2)
        stop = min(horizon_steps, start + spec.burst_length_steps)
        mask[start:stop, :] = False
    elif spec.burst_length_steps > 0 and spec.burst_stationary_probability > 0:
        stationary = float(spec.burst_stationary_probability)
        exit_probability = 1.0 / float(spec.burst_length_steps)
        enter_probability = stationary * exit_probability / (1.0 - stationary)
        if enter_probability > 1.0:
            raise ValueError("Burst stationary probability/mean imply alpha > 1")
        missing = bool(rng.random() < stationary)
        for step in range(horizon_steps):
            if missing:
                mask[step, :] = False
                missing = not (rng.random() < exit_probability)
            else:
                missing = bool(rng.random() < enter_probability)
    elif spec.burst_length_steps > 0 and spec.burst_start_probability > 0:
        starts = np.flatnonzero(
            rng.random(horizon_steps) < spec.burst_start_probability
        )
        for start in starts:
            stop = min(horizon_steps, start + spec.burst_length_steps)
            mask[start:stop, :] = False
    if spec.channel_outage_channels:
        mask[:, list(spec.channel_outage_channels)] = False

    noise = rng.normal(0.0, spec.sensor_noise_std, size=(horizon_steps, sensor_dim))
    noise = noise.astype(np.float32)
    noise[~mask] = 0.0

    dt = nominal_dt + rng.normal(
        0.0, spec.timestamp_measurement_jitter_std, size=horizon_steps
    )
    dt = np.clip(dt, nominal_dt * 0.25, nominal_dt * 2.0).astype(np.float32)
    gain = np.full(horizon_steps, spec.actuator_gain, dtype=np.float32)

    force = np.zeros((horizon_steps, 2), dtype=np.float32)
    if spec.impulse_magnitude > 0:
        # Keep the impulse away from the boundaries when possible, while still
        # making the one-step smoke-test case well defined.
        low = min(horizon_steps - 1, max(0, int(round(0.20 * horizon_steps))))
        high_exclusive = min(
            horizon_steps,
            max(low + 1, int(round(0.80 * horizon_steps)) + 1),
        )
        start = int(rng.integers(low, high_exclusive))
        angle = float(rng.uniform(-np.pi, np.pi))
        vector = spec.impulse_magnitude * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        stop = min(horizon_steps, start + spec.impulse_duration_steps)
        force[start:stop] = vector

    return PerturbationSchedule(
        sensor_mask=mask.astype(bool),
        sensor_noise=noise,
        observed_timestamp_dt=dt,
        actuator_gain=gain,
        external_force=force,
        delay_steps=int(spec.delay_steps),
        seed=int(seed),
        spec=spec,
    )


def packet_loss_run_lengths(sensor_mask: np.ndarray) -> list[int]:
    """Lengths of contiguous complete-packet loss runs."""
    mask = np.asarray(sensor_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] <= 0:
        raise ValueError("sensor_mask must have shape [time,channels]")
    packet_missing = ~mask.all(axis=1)
    starts = np.flatnonzero(packet_missing & ~np.r_[False, packet_missing[:-1]])
    ends = np.flatnonzero(packet_missing & ~np.r_[packet_missing[1:], False])
    return [int(end - start + 1) for start, end in zip(starts, ends)]


class ObservationPerturber:
    """Apply a fixed schedule while leaving reference signals uncorrupted."""

    def __init__(self, schedule: PerturbationSchedule, sensor_dim: int = 4) -> None:
        self.schedule = schedule
        self.sensor_dim = int(sensor_dim)
        self._history: list[np.ndarray] = []
        self._channel_age = np.zeros(self.sensor_dim, dtype=np.float32)
        self._held_sensor = np.zeros(self.sensor_dim, dtype=np.float32)
        self._held_initialized = np.zeros(self.sensor_dim, dtype=bool)

    def reset(self) -> None:
        self._history.clear()
        self._channel_age[:] = 0.0
        self._held_sensor[:] = 0.0
        self._held_initialized[:] = False

    def observe(
        self, clean_observation: np.ndarray, step: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        clean = np.asarray(clean_observation, dtype=np.float32)
        if clean.ndim != 1 or clean.shape[0] <= self.sensor_dim:
            raise ValueError("clean_observation has an invalid shape")
        if step < 0 or step >= len(self.schedule.observed_timestamp_dt):
            raise IndexError(step)
        self._history.append(clean[: self.sensor_dim].copy())
        delayed_index = max(0, len(self._history) - 1 - self.schedule.delay_steps)
        candidate = self._history[delayed_index].copy()
        mask = self.schedule.sensor_mask[step]
        candidate += self.schedule.sensor_noise[step]
        self._held_sensor[mask] = candidate[mask]
        self._held_initialized[mask] = True
        # Learned observers receive causal per-channel zero-order hold plus the
        # explicit validity mask and age.  If an episode begins with loss, the
        # raw fallback is zero; the training/evaluation normalizers explicitly
        # map this never-observed fallback to normalized zero.
        sensor = self._held_sensor.copy()
        measured = np.concatenate([sensor, clean[self.sensor_dim :]]).astype(np.float32)
        observed_dt = float(self.schedule.observed_timestamp_dt[step])
        self._channel_age += observed_dt
        if self.schedule.delay_steps > 0:
            first = max(0, step - self.schedule.delay_steps + 1)
            delay_age = float(
                self.schedule.observed_timestamp_dt[first : step + 1].sum()
            )
        else:
            delay_age = 0.0
        self._channel_age[mask] = delay_age
        return (
            measured,
            mask.astype(np.float32),
            self._channel_age.copy(),
            observed_dt,
        )


def build_observer_input(
    measured_observation: np.ndarray,
    sensor_mask: np.ndarray,
    channel_age: np.ndarray,
    observed_dt: float,
    previous_control: np.ndarray,
) -> np.ndarray:
    measured = np.asarray(measured_observation, dtype=np.float32)
    mask = np.asarray(sensor_mask, dtype=np.float32)
    age = np.asarray(channel_age, dtype=np.float32)
    previous = np.asarray(previous_control, dtype=np.float32)
    if measured.ndim != 1 or measured.shape[0] <= 4:
        raise ValueError("measured_observation must be a vector with state and reference")
    if mask.shape != (4,) or age.shape != (4,) or previous.shape != (2,):
        raise ValueError("Expected mask/age/control shapes (4,), (4,), and (2,)")
    if observed_dt <= 0 or not np.isfinite(observed_dt):
        raise ValueError("observed_dt must be finite and positive")
    if np.any(age < 0) or not np.all(np.isfinite(age)):
        raise ValueError("channel_age must be finite and non-negative")
    return np.concatenate(
        [
            measured[:4],
            mask,
            age,
            measured[4:],
            previous,
            np.array([observed_dt], dtype=np.float32),
        ]
    ).astype(np.float32)


def build_policy_input(
    measured_observation: np.ndarray, sensor_mask: np.ndarray, observed_dt: float
) -> np.ndarray:
    """Legacy helper retained for exploratory action-policy notebooks only."""
    return build_observer_input(
        measured_observation,
        sensor_mask,
        np.zeros(4, dtype=np.float32),
        observed_dt,
        np.zeros(2, dtype=np.float32),
    )

