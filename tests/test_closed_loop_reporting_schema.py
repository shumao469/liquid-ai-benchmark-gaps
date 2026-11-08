from __future__ import annotations

import math

import pytest

from liquid_revision.closed_loop.benchmark import _summarize


METRICS = (
    "tracking_rmse",
    "tracking_p95",
    "failure",
    "recovery_steps",
    "recovery_seconds",
    "control_effort",
    "saturation_fraction",
    "control_observed_fraction",
    "plant_completion_fraction",
    "state_rmse",
    "joint_position_rmse",
    "joint_velocity_rmse",
    "state_accuracy_finite_fraction",
    "latency_median_ms",
    "latency_p95_ms",
    "ode_solver_function_evaluations",
    "ltc_internal_unfolds",
)


def _row(*, training_seed: int, trajectory_id: str | None = None) -> dict:
    row = {
        "model_name": "cfc",
        "condition_id": "iid_drop_0.1",
        "family": "iid_dropout",
        "severity": 0.1,
        "training_seed": training_seed,
    }
    if trajectory_id is not None:
        row["trajectory_id"] = trajectory_id
    row.update({metric: 1.0 for metric in METRICS})
    return row


def test_condition_summary_distinguishes_rows_seeds_and_unique_episodes() -> None:
    rows = [
        _row(training_seed=seed, trajectory_id=trajectory_id)
        for seed in (11, 12)
        for trajectory_id in ("test-000001", "test-000002")
    ]

    [summary] = _summarize(
        rows,
        {"evaluation": {"bootstrap_replicates": 4}},
    )

    assert summary["model_seed_count"] == 2
    assert summary["unique_test_episodes"] == 2
    assert summary["row_count"] == 4
    assert "episodes" not in summary
    assert "training_seeds" not in summary
    assert math.isfinite(summary["tracking_rmse_mean"])


def test_condition_summary_uses_episode_id_only_when_trajectory_id_is_absent() -> None:
    rows = [_row(training_seed=11), _row(training_seed=12)]
    rows[0]["episode_id"] = 7
    rows[1]["episode_id"] = 7

    [summary] = _summarize(
        rows,
        {"evaluation": {"bootstrap_replicates": 4}},
    )

    assert summary["unique_test_episodes"] == 1
    assert summary["row_count"] == 2


def test_condition_summary_rejects_rows_without_an_episode_identity() -> None:
    with pytest.raises(
        ValueError,
        match="require trajectory_id or episode_id",
    ):
        _summarize(
            [_row(training_seed=11)],
            {"evaluation": {"bootstrap_replicates": 4}},
        )
