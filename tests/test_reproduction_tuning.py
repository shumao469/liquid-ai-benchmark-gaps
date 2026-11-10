from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from liquid_revision.io_utils import canonical_json_hash, sha256_file
from liquid_revision.reproduction import tuning


def _config() -> dict:
    return {
        "experiment": {"development_only": True},
        "formal": {"required": False},
        "tuning": {
            "seed": 17,
            "learning_rates": [0.0003, 0.001],
            "weight_decays": [0.0, 0.0001],
            "epochs": 2,
            "optimizer_steps_per_epoch": 3,
            "planned_optimizer_steps": 6,
        },
        "matching": {"registered_models": list(tuning.UNIT_MODELS)},
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    config = _config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config_sha256 = canonical_json_hash(config)
    registry_core = {
        "source_config_sha256": config_sha256,
        "models": {name: {} for name in tuning.UNIT_MODELS},
    }
    registry = {
        **registry_core,
        "registry_sha256": canonical_json_hash(registry_core),
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    return config_path, data_root, registry_path, config


def test_build_plan_preserves_unit_and_scaled_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, data_root, registry_path, _ = _inputs(tmp_path)
    # Registry rebuilding itself is covered by the closed-loop registry tests;
    # this unit test isolates the scheduler's frozen grid construction.
    monkeypatch.setattr(
        tuning, "_load_registry_identity", lambda _path, _config: "b" * 64
    )
    unit_plan, _, unit_candidates = tuning.build_plan(
        config_path=config_path,
        data_root=data_root,
        registry_path=registry_path,
        output_root=tmp_path / "runs",
        readout="unit",
    )
    scaled_plan, _, scaled_candidates = tuning.build_plan(
        config_path=config_path,
        data_root=data_root,
        registry_path=registry_path,
        output_root=tmp_path / "runs",
        readout="scaled",
    )

    assert unit_plan["scope"] == "validation_only_seed17"
    assert unit_plan["seed"] == 17
    assert unit_plan["grid"]["models"] == list(tuning.UNIT_MODELS)
    assert unit_plan["grid"]["candidate_count"] == len(unit_candidates) == 28
    assert unit_plan["budget"] == {"epochs": 2, "optimizer_steps": 6}
    assert scaled_plan["grid"]["models"] == list(tuning.SCALED_MODELS)
    assert scaled_plan["grid"]["candidate_count"] == len(scaled_candidates) == 12
    assert not (tmp_path / "runs").exists(), "planning must remain read-only"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["tuning"].update(seed=18), "seed"),
        (
            lambda value: value["tuning"].update(learning_rates=[0.001]),
            "learning-rate",
        ),
        (lambda value: value["formal"].update(required=True), "cannot bypass"),
        (
            lambda value: value["experiment"].update(development_only=False),
            "cannot bypass",
        ),
    ],
)
def test_frozen_design_and_formal_boundary_are_enforced(
    mutation, message: str
) -> None:
    config = _config()
    mutation(config)
    with pytest.raises(ValueError, match=message):
        tuning._validate_tuning_design(config, readout="unit")


def test_candidate_state_requires_full_budget_and_bound_checkpoint(
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    checkpoint = candidate_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    candidate = tuning.Candidate(
        candidate_id="lstm__lr00__wd00",
        model="lstm",
        learning_rate=0.0003,
        weight_decay=0.0,
        learning_rate_index=0,
        weight_decay_index=0,
        directory=candidate_dir,
        spec_path=candidate_dir / "candidate_spec.json",
    )
    context = {
        "config_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "readout": "unit",
        "expected_epochs": 2,
        "expected_optimizer_steps": 6,
    }
    spec = tuning._candidate_spec(candidate, context)
    state = {
        "scope": tuning.SCOPE,
        "phase": "tune",
        "model": "lstm",
        "seed": 17,
        "readout": "unit",
        "config_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "learning_rate": 0.0003,
        "weight_decay": 0.0,
        "optimizer_hyperparameters": {
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
        },
        "epochs_completed": 2,
        "optimizer_steps_completed": 6,
        "best_val_loss": 0.25,
        "best_checkpoint_sha256": sha256_file(checkpoint),
        "train_data_sha256": "c" * 64,
        "val_data_sha256": "d" * 64,
    }
    validated = tuning._validate_state_identity(state, spec, checkpoint)
    assert validated["optimizer_steps_completed"] == 6

    state["optimizer_steps_completed"] = 5
    with pytest.raises(ValueError, match="optimizer-step budget"):
        tuning._validate_state_identity(state, spec, checkpoint)


def test_selection_uses_historical_loss_weight_decay_learning_rate_tie_break() -> None:
    models = list(tuning.UNIT_MODELS)
    plan = {
        "readout": "unit",
        "config_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "tuning_plan_sha256": "e" * 64,
        "grid": {
            "models": models,
            "learning_rates": list(tuning.LEARNING_RATES),
            "weight_decays": list(tuning.WEIGHT_DECAYS),
            "candidate_count": 28,
        },
        "budget": {"epochs": 2, "optimizer_steps": 6},
    }
    rows = []
    for model in models:
        for lr_index, learning_rate in enumerate(tuning.LEARNING_RATES):
            for wd_index, weight_decay in enumerate(tuning.WEIGHT_DECAYS):
                candidate_id = tuning._candidate_id(model, lr_index, wd_index)
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_result_sha256": canonical_json_hash(candidate_id),
                        "model": model,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "learning_rate_index": lr_index,
                        "weight_decay_index": wd_index,
                        "best_val_loss": 0.5,
                        "best_checkpoint_sha256": "f" * 64,
                        "run_state_sha256": "1" * 64,
                        "config_sha256": "a" * 64,
                        "registry_sha256": "b" * 64,
                        "train_data_sha256": "c" * 64,
                        "val_data_sha256": "d" * 64,
                    }
                )
    selection = tuning._selection_from_results(plan=plan, results=rows)
    for model in models:
        winner = selection["selected"][model]
        assert winner["weight_decay"] == 0.0
        assert winner["learning_rate"] == 0.0003
    assert selection["data_sha256"] == {"train": "c" * 64, "val": "d" * 64}
    assert selection["tuning_selection_sha256"] == canonical_json_hash(
        {
            key: value
            for key, value in selection.items()
            if key != "tuning_selection_sha256"
        }
    )


def test_gpu_lane_parser_supports_two_cuda_lanes_or_one_cpu_lane() -> None:
    assert tuning._parse_gpu_slots("0,1") == ("0", "1")
    assert tuning._parse_gpu_slots("GPU-a,GPU-b") == ("GPU-a", "GPU-b")
    assert tuning._parse_gpu_slots("cpu") == ("cpu",)
    with pytest.raises(ValueError, match="cannot be mixed"):
        tuning._parse_gpu_slots("cpu,0")
    with pytest.raises(ValueError, match="unique"):
        tuning._parse_gpu_slots("0,0")
