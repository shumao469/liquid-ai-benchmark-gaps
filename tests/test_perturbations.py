from __future__ import annotations

import numpy as np

from liquid_revision.perturbations import (
    ObservationPerturber,
    PerturbationSpec,
    build_observer_input,
    make_perturbation_schedule,
)


def test_schedule_is_seed_deterministic() -> None:
    spec = PerturbationSpec(
        iid_dropout_probability=0.2,
        burst_start_probability=0.1,
        burst_length_steps=3,
        sensor_noise_std=0.01,
        timestamp_measurement_jitter_std=0.001,
        impulse_magnitude=0.2,
    )
    first = make_perturbation_schedule(
        horizon_steps=50, nominal_dt=0.02, spec=spec, seed=4
    )
    second = make_perturbation_schedule(
        horizon_steps=50, nominal_dt=0.02, spec=spec, seed=4
    )
    np.testing.assert_array_equal(first.sensor_mask, second.sensor_mask)
    np.testing.assert_array_equal(first.sensor_noise, second.sensor_noise)
    np.testing.assert_array_equal(first.external_force, second.external_force)
    assert np.all(first.observed_timestamp_dt > 0)


def test_iid_dropout_is_one_complete_four_channel_packet() -> None:
    schedule = make_perturbation_schedule(
        horizon_steps=200,
        nominal_dt=0.02,
        spec=PerturbationSpec(iid_dropout_probability=0.35),
        seed=123,
    )
    # Every time row is either fully observed or fully absent.
    assert np.all(schedule.sensor_mask == schedule.sensor_mask[:, :1])
    assert (~schedule.sensor_mask[:, 0]).any()
    assert schedule.sensor_mask[:, 0].any()


def test_reference_is_not_corrupted() -> None:
    spec = PerturbationSpec(
        iid_dropout_probability=1.0, delay_steps=1, sensor_noise_std=1.0
    )
    schedule = make_perturbation_schedule(
        horizon_steps=3, nominal_dt=0.02, spec=spec, seed=3
    )
    perturber = ObservationPerturber(schedule)
    clean = np.arange(10, dtype=np.float32)
    measured, mask, age, dt = perturber.observe(clean, 0)
    assert not mask.any()
    np.testing.assert_array_equal(measured[:4], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(measured[4:], clean[4:])
    observer_input = build_observer_input(
        measured, mask, age, dt, np.zeros(2, dtype=np.float32)
    )
    assert observer_input.shape == (21,)
    assert np.all(age > 0)


def test_channel_outage_is_exact() -> None:
    spec = PerturbationSpec(channel_outage_channels=(1, 3))
    schedule = make_perturbation_schedule(
        horizon_steps=8, nominal_dt=0.02, spec=spec, seed=8
    )
    assert not schedule.sensor_mask[:, [1, 3]].any()
    assert schedule.sensor_mask[:, [0, 2]].all()


def test_learned_observation_uses_causal_zero_order_hold() -> None:
    schedule = make_perturbation_schedule(
        horizon_steps=4,
        nominal_dt=0.02,
        spec=PerturbationSpec(),
        seed=8,
    )
    schedule.sensor_mask[:] = np.asarray(
        [[True] * 4, [False] * 4, [False] * 4, [True] * 4]
    )
    perturber = ObservationPerturber(schedule)
    observed = []
    masks = []
    for step in range(4):
        clean = np.arange(8, dtype=np.float32) + 10 * step
        measured, mask, _, _ = perturber.observe(clean, step)
        observed.append(measured[:4])
        masks.append(mask)
    np.testing.assert_array_equal(observed[1], observed[0])
    np.testing.assert_array_equal(observed[2], observed[0])
    np.testing.assert_array_equal(observed[3], np.arange(4) + 30)
    assert not np.asarray(masks[1:3]).any()


def test_one_step_impulse_schedule_is_well_defined() -> None:
    schedule = make_perturbation_schedule(
        horizon_steps=1,
        nominal_dt=0.02,
        spec=PerturbationSpec(impulse_magnitude=0.2),
        seed=2,
    )
    assert schedule.external_force.shape == (1, 2)
    assert np.isfinite(schedule.external_force).all()

