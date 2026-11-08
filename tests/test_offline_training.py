from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import liquid_revision.offline.training as offline_training
from liquid_revision.offline.training import (
    compose_model_features,
    deterministic_batch_indices,
    train_fixed_budget,
)


@dataclass(frozen=True)
class TinyArrays:
    values: np.ndarray
    mask: np.ndarray
    age: np.ndarray
    dt: np.ndarray
    labels: np.ndarray
    ids: np.ndarray
    group_ids: np.ndarray
    lengths: np.ndarray


def _tiny_har(samples: int, seed: int) -> TinyArrays:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(samples, 5, 3)).astype(np.float32)
    mask = np.ones_like(values, dtype=bool)
    age = np.zeros_like(values, dtype=np.float32)
    dt = np.full((samples, 5), 0.02, dtype=np.float32)
    labels = (np.arange(samples) % 6).astype(np.int64)
    ids = np.array([f"sample-{seed}-{index}" for index in range(samples)])
    lengths = np.full(samples, 5, dtype=np.int64)
    return TinyArrays(values, mask, age, dt, labels, ids, ids.copy(), lengths)


def test_feature_composition_and_deterministic_resume_batches() -> None:
    arrays = _tiny_har(9, 1)
    features = compose_model_features(
        arrays.values, arrays.mask, arrays.age, arrays.dt,
        channels=["values", "mask", "age", "dt"],
    )
    assert features.shape == (9, 5, 10)
    first = deterministic_batch_indices(9, 4, seed=12, optimizer_step=3)
    resumed = deterministic_batch_indices(9, 4, seed=12, optimizer_step=3)
    np.testing.assert_array_equal(first, resumed)


def test_two_step_run_freezes_checkpoint_without_test_and_rehashes_on_reuse(tmp_path) -> None:
    data = {
        "train": _tiny_har(18, 2),
        "val": _tiny_har(12, 3),
        "test": _tiny_har(12, 4),
        "manifest_sha256": "synthetic-manifest",
    }
    config = {
        "experiment": {"device": "cpu", "deterministic_torch": True},
        "features": {"channels": ["values", "mask", "age", "dt"]},
        "models": {"hidden_dim": 4, "ode_max_step": 1.0},
        "training": {
            "optimizer_steps": 2,
            "validation_interval": 1,
            "batch_size": 6,
            "evaluation_batch_size": 6,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
        },
        "corruption": {
            "base_seed": 99,
            "mcar_levels": [0.0, 0.3],
            "temporal_block_levels": [0.0, 0.4],
            "retrain_per_severity": False,
        },
        "latency": {"warmup": 0, "repeats": 1},
    }
    result = train_fixed_budget(
        dataset_name="har",
        model_name="lstm",
        seed=101,
        data=data,
        config=config,
        config_hash="a" * 64,
        output_root=tmp_path,
    )
    assert result["status"] == "checkpoint_frozen"
    assert result["best_step"] in {1, 2}
    assert result["test_metrics_computed"] is False
    assert result["optimizer_steps_planned"] == result["optimizer_steps_completed"] == 2
    assert result["accelerator_identity"]["training_precision"] == "float32"
    assert result["accelerator_identity"]["autocast_enabled"] is False
    assert result["compute_accounting"]["train"]["model_family"] == "lstm"
    assert result["compute_accounting"]["train"]["valid_updates"]["total"] == 18 * 5
    assert "fixed_checkpoint_evaluation" not in result
    run_dir = tmp_path / result["run_id"]
    assert (run_dir / "completion.json").is_file()
    assert not (run_dir / "fixed_checkpoint_evaluation").exists()

    repeated = train_fixed_budget(
        dataset_name="har",
        model_name="lstm",
        seed=101,
        data=data,
        config=config,
        config_hash="a" * 64,
        output_root=tmp_path,
    )
    assert repeated["best_checkpoint_sha256"] == result["best_checkpoint_sha256"]

    with (run_dir / "best.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="rehash failed"):
        train_fixed_budget(
            dataset_name="har",
            model_name="lstm",
            seed=101,
            data=data,
            config=config,
            config_hash="a" * 64,
            output_root=tmp_path,
        )


def test_resume_restores_validation_history_without_duplicates(tmp_path, monkeypatch) -> None:
    data = {
        "train": _tiny_har(18, 21),
        "val": _tiny_har(12, 22),
        "test": _tiny_har(12, 23),
        "manifest_sha256": "synthetic-manifest",
    }
    config = {
        "experiment": {"device": "cpu", "deterministic_torch": True},
        "features": {"channels": ["values", "mask", "age", "dt"]},
        "models": {"hidden_dim": 4, "ode_max_step": 1.0},
        "training": {
            "optimizer_steps": 3,
            "validation_interval": 1,
            "batch_size": 6,
            "evaluation_batch_size": 6,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
        },
        "corruption": {
            "base_seed": 99,
            "mcar_levels": [0.0],
            "temporal_block_levels": [0.0],
            "retrain_per_severity": False,
        },
        "latency": {"warmup": 0, "repeats": 1},
    }
    original_batches = offline_training.deterministic_batch_indices

    def interrupt_before_second_step(sample_count, batch_size, seed, optimizer_step):
        if optimizer_step == 1:
            raise RuntimeError("synthetic interruption after checkpoint")
        return original_batches(sample_count, batch_size, seed, optimizer_step)

    monkeypatch.setattr(
        offline_training, "deterministic_batch_indices", interrupt_before_second_step
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        train_fixed_budget(
            dataset_name="har",
            model_name="lstm",
            seed=101,
            data=data,
            config=config,
            config_hash="b" * 64,
            output_root=tmp_path,
        )
    monkeypatch.setattr(
        offline_training, "deterministic_batch_indices", original_batches
    )
    result = train_fixed_budget(
        dataset_name="har",
        model_name="lstm",
        seed=101,
        data=data,
        config=config,
        config_hash="b" * 64,
        output_root=tmp_path,
    )
    assert [item["optimizer_step"] for item in result["validation_history"]] == [1, 2, 3]
    manifest = __import__("json").loads(
        (tmp_path / result["run_id"] / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["attempt_number"] == 2
    assert manifest["resumed_from_step"] == 1
    assert manifest["restored_validation_records"] == 1


def test_training_and_checkpoint_selection_do_not_read_test_arrays(tmp_path, monkeypatch) -> None:
    class TestAccessGuard:
        def __getattribute__(self, name):
            raise AssertionError(f"test arrays accessed during training: {name}")

    data = {
        "train": _tiny_har(12, 31),
        "val": _tiny_har(12, 32),
        "test": TestAccessGuard(),
        "manifest_sha256": "synthetic-manifest",
    }
    config = {
        "experiment": {"device": "cpu", "deterministic_torch": True},
        "features": {"channels": ["values", "mask", "age", "dt"]},
        "models": {"hidden_dim": 4},
        "training": {
            "optimizer_steps": 1,
            "validation_interval": 1,
            "batch_size": 6,
            "evaluation_batch_size": 6,
        },
    }
    result = train_fixed_budget(
        dataset_name="har",
        model_name="lstm",
        seed=101,
        data=data,
        config=config,
        config_hash="c" * 64,
        output_root=tmp_path,
    )
    assert result["test_metrics_computed"] is False
    assert "fixed_checkpoint_evaluation" not in result


def test_aggregate_manifest_records_interruption(tmp_path, monkeypatch) -> None:
    config = {
        "experiment": {"datasets": ["har"], "seeds": [101]},
        "models": {"names": ["lstm"]},
    }
    monkeypatch.setattr(
        offline_training, "load_config", lambda path: (config, "d" * 64)
    )
    monkeypatch.setattr(
        offline_training,
        "load_all_datasets",
        lambda config, path, **kwargs: {"har": {}},
    )

    def interrupt(**kwargs):
        raise KeyboardInterrupt("synthetic user interruption")

    monkeypatch.setattr(offline_training, "train_fixed_budget", interrupt)
    with pytest.raises(KeyboardInterrupt, match="synthetic user interruption"):
        offline_training.run_offline_benchmark(
            tmp_path / "config.yaml", tmp_path / "output"
        )
    aggregate = __import__("json").loads(
        (tmp_path / "output" / "offline_benchmark_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert aggregate["status"] == "interrupted"
    assert aggregate["current_attempt"] is None
    assert aggregate["attempts"][-1]["status"] == "interrupted"
    assert aggregate["attempts"][-1]["exception_type"] == "KeyboardInterrupt"

