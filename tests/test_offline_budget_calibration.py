from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from liquid_revision.offline.budget_calibration import (
    _plateau_diagnostics,
    calibrate_one,
)


@dataclass(frozen=True)
class TinyArrays:
    values: np.ndarray
    mask: np.ndarray
    age: np.ndarray
    dt: np.ndarray
    labels: np.ndarray
    ids: np.ndarray
    lengths: np.ndarray


def _tiny(samples: int, seed: int) -> TinyArrays:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(samples, 5, 3)).astype(np.float32)
    return TinyArrays(
        values=values,
        mask=np.ones_like(values, dtype=bool),
        age=np.zeros_like(values, dtype=np.float32),
        dt=np.full((samples, 5), 0.02, dtype=np.float32),
        labels=(np.arange(samples) % 6).astype(np.int64),
        ids=np.asarray([f"tiny-{seed}-{index}" for index in range(samples)]),
        lengths=np.full(samples, 5, dtype=np.int64),
    )


def test_plateau_rule_distinguishes_flat_and_rising_validation_curves() -> None:
    config = {
        "plateau_window_validations": 5,
        "slope_threshold_metric_per_1000_steps": 0.005,
        "recent_best_gain_threshold": 0.005,
        "minimum_margin_after_best_validations": 2,
    }
    flat = [
        {"optimizer_step": 100 * (index + 1), "selection_metric": 0.7}
        for index in range(10)
    ]
    rising = [
        {"optimizer_step": 100 * (index + 1), "selection_metric": 0.5 + 0.01 * index}
        for index in range(10)
    ]
    assert _plateau_diagnostics(flat, config)["plateau"] is True
    rising_diagnostics = _plateau_diagnostics(rising, config)
    assert rising_diagnostics["plateau"] is False
    assert rising_diagnostics["clearly_not_plateau"] is True


def test_budget_calibration_never_accesses_test_arrays(tmp_path) -> None:
    class TestAccessGuard:
        def __getattribute__(self, name):
            raise AssertionError(f"budget calibration accessed test arrays: {name}")

    data = {
        "train": _tiny(18, 1),
        "val": _tiny(12, 2),
        "test": TestAccessGuard(),
        "manifest_sha256": "synthetic-manifest",
    }
    config = {
        "experiment": {
            "device": "cpu",
            "deterministic_torch": True,
            "test_metrics_permitted": False,
        },
        "features": {"channels": ["values", "mask", "age", "dt"]},
        "models": {"hidden_dim": 4, "ode_max_step": 1.0},
        "training": {
            "optimizer_steps": 100,
            "validation_interval": 100,
            "batch_size": 6,
            "evaluation_batch_size": 6,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
        },
        "budget_calibration": {
            "tuning_seed": 17,
            "test_evaluation": "forbidden",
            "corruption_evaluation": "forbidden",
        },
    }
    result = calibrate_one(
        dataset_name="har",
        model_name="lstm",
        seed=17,
        data=data,
        config=config,
        config_hash="e" * 64,
        output_root=tmp_path,
    )
    assert result["status"] == "completed"
    assert result["test_metrics_computed"] is False
    assert result["corruption_metrics_computed"] is False
    assert len(result["validation_history"]) == 1
