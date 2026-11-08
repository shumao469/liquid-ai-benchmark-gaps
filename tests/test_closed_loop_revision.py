from __future__ import annotations

import json
import math
import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch")
pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")
pytest.importorskip("ncps")

from liquid_revision.closed_loop import benchmark, formal
from liquid_revision.closed_loop.benchmark import (
    _curve_naurc,
    _estimate_state_for_control,
    _exact_paired_seed_sign_flip_p,
    _mechanical_endpoint_contrasts,
    _mechanism_contrasts,
    _recovery_steps,
    build_response_conditions,
    evaluate_response_surface,
)
from liquid_revision.closed_loop.baselines import FullStateEstimator
from liquid_revision.data.closed_loop import _make_controller
from liquid_revision.closed_loop.manifests import (
    build_trajectory_manifest,
    load_trajectory_manifest,
    prepare_split_datasets,
)
from liquid_revision.closed_loop.formal import (
    create_controller_selection,
    create_environment_freeze,
    current_environment_identity,
    current_run_accelerator_identity,
    validate_controller_selection,
    validate_environment_freeze,
    create_tuning_selection,
    tuning_design_identity,
    validate_tuning_selection,
)


def test_environment_freeze_is_independent_of_cuda_visible_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    identity_all = current_environment_identity(config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    identity_worker = current_environment_identity(config)
    assert identity_worker == identity_all


def test_run_accelerator_records_physical_cvd_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = [
        {
            "physical_index": "0",
            "uuid": "GPU-zero",
            "pci_bus_id": "0000:01:00.0",
            "name": "GPU zero",
            "driver_version": "999.0",
        },
        {
            "physical_index": "1",
            "uuid": "GPU-one",
            "pci_bus_id": "0000:02:00.0",
            "name": "GPU one",
            "driver_version": "999.0",
        },
    ]
    from liquid_revision.closed_loop import formal

    monkeypatch.setattr(formal, "_physical_gpu_inventory", lambda *args: inventory)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"visible-{index}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (9, 0))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    recorded = current_run_accelerator_identity("cuda:0")
    assert recorded["cuda_visible_device_token"] == "1"
    assert recorded["selected_physical_gpu"]["uuid"] == "GPU-one"
    with pytest.raises(RuntimeError, match="select the GPU by UUID"):
        current_run_accelerator_identity(
            "cuda:0", require_physical_mapping=True
        )


def test_run_accelerator_rejects_ambiguous_uuid_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from liquid_revision.closed_loop import formal

    inventory = [
        {
            "physical_index": "0",
            "uuid": "GPU-abcd1111",
            "pci_bus_id": "0000:01:00.0",
            "name": "GPU zero",
            "driver_version": "999.0",
        },
        {
            "physical_index": "1",
            "uuid": "GPU-abcd2222",
            "pci_bus_id": "0000:02:00.0",
            "name": "GPU one",
            "driver_version": "999.0",
        },
    ]
    monkeypatch.setattr(formal, "_physical_gpu_inventory", lambda *args: inventory)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "visible-0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (9, 0))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd")
    with pytest.raises(RuntimeError, match="abbreviated GPU UUID"):
        current_run_accelerator_identity(
            "cuda:0", require_physical_mapping=True
        )

    monkeypatch.setattr(formal, "_physical_gpu_inventory", lambda *args: inventory[:1])
    with pytest.raises(RuntimeError, match="abbreviated GPU UUID"):
        current_run_accelerator_identity(
            "cuda:0", require_physical_mapping=True
        )


def test_paper_rollout_information_flow_and_exact_19d_schema() -> None:
    class SpyObserver(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_input = None

        def forward_sequence(self, x, dt, state):
            del dt
            self.last_input = x.detach().clone()
            return x[..., :4], state

    learned = SpyObserver()
    estimator = {"model_name": "spy", "model": learned}
    clean_a = np.asarray([1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    clean_b = clean_a.copy()
    clean_b[:4] += 100.0
    measured = np.asarray([5, 6, 7, 8, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    mask = np.asarray([1, 0, 1, 0], dtype=np.float32)
    age = np.asarray([0.0, 0.02, 0.0, 0.04], dtype=np.float32)
    previous_torque = np.asarray([0.5, -0.5], dtype=np.float32)
    first, _, input_a = _estimate_state_for_control(
        estimator,
        clean=clean_a,
        measured=measured,
        mask=mask,
        age=age,
        observed_dt=0.021,
        previous_control=previous_torque,
        recurrent_state=None,
        device="cpu",
    )
    second, _, input_b = _estimate_state_for_control(
        estimator,
        clean=clean_b,
        measured=measured,
        mask=mask,
        age=age,
        observed_dt=0.021,
        previous_control=previous_torque,
        recurrent_state=None,
        device="cpu",
    )
    assert input_a.shape == input_b.shape == (19,)
    np.testing.assert_array_equal(input_a, input_b)
    np.testing.assert_array_equal(first, second)
    assert learned.last_input.shape == (1, 1, 19)

    oracle = FullStateEstimator()
    oracle_a, _, _ = _estimate_state_for_control(
        {"model_name": "full_state_pd", "baseline": oracle},
        clean=clean_a,
        measured=measured,
        mask=mask,
        age=age,
        observed_dt=0.021,
        previous_control=previous_torque,
        recurrent_state=None,
        device="cpu",
    )
    corrupted_measured = measured.copy()
    corrupted_measured[:4] = -999.0
    oracle_same, _, _ = _estimate_state_for_control(
        {"model_name": "full_state_pd", "baseline": oracle},
        clean=clean_a,
        measured=corrupted_measured,
        mask=np.zeros(4, dtype=np.float32),
        age=age,
        observed_dt=0.021,
        previous_control=previous_torque,
        recurrent_state=None,
        device="cpu",
    )
    np.testing.assert_array_equal(oracle_a, oracle_same)
    controller = _make_controller("reacher_v5", _tiny_config()["controller"], 1.0)
    action_a = controller.control(first, measured)
    action_b = controller.control(first, corrupted_measured)
    np.testing.assert_array_equal(action_a, action_b)


def test_naurc_and_exact_seed_test_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _curve_naurc(np.asarray([0.0, 1.0]), np.asarray([1.0, np.nan]))
    with pytest.raises(ValueError, match="denominator"):
        _curve_naurc(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
    # Five same-signed nonzero effects have the exact two-sided floor 2/32.
    assert _exact_paired_seed_sign_flip_p(np.ones(5)) == pytest.approx(0.0625)
from liquid_revision.closed_loop.parameter_matching import load_parameter_registry
from liquid_revision.io_utils import canonical_json_hash, sha256_file
from liquid_revision.closed_loop.parameter_matching import (
    build_parameter_registry,
    build_registered_model,
    create_parameter_registry,
)
from liquid_revision.closed_loop.training import (
    _validate_checkpoint_tuning_binding,
    load_trained_observer,
    train_registered_model,
    train_tuning_candidate,
)


def test_checkpoint_tuning_binding_rejects_optimizer_and_registry_forgery() -> None:
    candidate_sha = "candidate-sha"
    tuning = {
        "trajectory_manifest_sha256": "trajectory",
        "parameter_registry_sha256": "registry",
        "implementation_sha256": "implementation",
        "environment_sha256": "environment",
        "controller_selection_sha256": "controller",
        "selected": {
            "lstm": {
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "checkpoint_sha256": candidate_sha,
            }
        },
        "candidates": [
            {
                "registered_name": "lstm",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "checkpoint_sha256": candidate_sha,
            }
        ],
    }
    checkpoint = {
        "registered_name": "lstm",
        "optimizer_hyperparameters": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "trajectory_manifest_sha256": "trajectory",
        "parameter_registry_sha256": "registry",
        "parameter_registry_implementation_sha256": "implementation",
        "environment_sha256": "environment",
        "controller_selection_sha256": "controller",
    }
    _validate_checkpoint_tuning_binding(checkpoint, tuning)
    forged_optimizer = json.loads(json.dumps(checkpoint))
    forged_optimizer["optimizer_hyperparameters"]["learning_rate"] = 0.0003
    with pytest.raises(ValueError, match="optimizer differs"):
        _validate_checkpoint_tuning_binding(forged_optimizer, tuning)
    forged_registry = dict(checkpoint)
    forged_registry["parameter_registry_sha256"] = "other-registry"
    with pytest.raises(ValueError, match="parameter_registry_sha256"):
        _validate_checkpoint_tuning_binding(forged_registry, tuning)
from liquid_revision.closed_loop.profiling import (
    profile_optimizer_steps,
    profile_rollout_throughput,
)
from liquid_revision.models.topology import make_degree_matched_wiring_pair
from liquid_revision.models import build_model


def _tiny_config() -> dict:
    return {
        "experiment": {"seed": 8, "deterministic_torch": True},
        "environment": {
            "kind": "reacher_v5",
            "horizon_steps": 4,
            "frame_skip": 2,
            "amplitude_range": [0.15, 0.20],
            "frequency_range_hz": [0.05, 0.08],
        },
        "controller": {
            "joint_kp": 10.0,
            "joint_kd": 1.0,
            "actuator_gear": 200.0,
        },
        "dataset": {
            "plant_perturbations_during_rollout": False,
            "perturbation": {"iid_dropout_probability": 0.1},
        },
        "splits": {
            "train": {
                "episodes": 2,
                "episode_seed_start": 100,
                "schedule_seed_start": 1000,
            },
            "val": {
                "episodes": 1,
                "episode_seed_start": 200,
                "schedule_seed_start": 2000,
            },
            "test": {
                "episodes": 1,
                "episode_seed_start": 300,
                "schedule_seed_start": 3000,
            },
        },
        "models": {
            "input_dim": 19,
            "state_dim": 4,
            "ode": {"max_step": 0.02},
            "ctrnn": {"initial_tau": 0.2, "min_tau": 0.001},
        },
        "matching": {
            "target_parameters": 404,
            "tolerance_fraction": 0.05,
            "minimum_hidden_dim": 4,
            "maximum_hidden_dim": 4,
            "registered_models": ["lstm"],
        },
        "training": {
            "epochs": 1,
            "patience": 1,
            "batch_size": 2,
            "window_steps": 2,
            "learning_rate": 0.001,
        },
        "evaluation": {
            "max_test_episodes": 1,
            "latency_warmup_calls": 0,
            "latency_measured_calls": 2,
            "iid_dropout_probabilities": [0.0, 0.2],
            "burst_lengths": [0, 2],
            "burst_start_probability": 0.5,
            "delay_steps": [0, 1],
            "nonlearned_baselines": ["full_state_pd", "zoh_pd", "kalman_pd"],
            "bootstrap_replicates": 5,
            "recovery_consecutive_steps": 1,
        },
    }


def test_trajectory_manifest_is_disjoint_and_tamper_evident(tmp_path: Path) -> None:
    config = _tiny_config()
    manifest = build_trajectory_manifest(config)
    assert manifest["counts"] == {"train": 2, "val": 1, "test": 1}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_trajectory_manifest(path)["manifest_sha256"] == manifest["manifest_sha256"]
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["splits"]["test"][0]["episode_seed"] = 100
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_trajectory_manifest(path)


def test_degree_matched_random_wiring_preserves_every_degree() -> None:
    ncp, random_wiring, audit = make_degree_matched_wiring_pair(
        hidden_dim=24,
        input_dim=19,
        output_dim=4,
        sparsity=0.5,
        ncp_seed=41,
        random_seed=42,
    )
    assert audit["exact_per_node_degree_match"] is True
    assert audit["graphs_differ"] is True
    np.testing.assert_array_equal(
        np.count_nonzero(ncp.adjacency_matrix, axis=0),
        np.count_nonzero(random_wiring.adjacency_matrix, axis=0),
    )
    np.testing.assert_array_equal(
        np.count_nonzero(ncp.adjacency_matrix, axis=1),
        np.count_nonzero(random_wiring.adjacency_matrix, axis=1),
    )


def test_parameter_registry_enforces_tolerance_and_topology_is_used() -> None:
    config = _tiny_config()
    config["matching"] = {
        "target_parameters": 20000,
        "tolerance_fraction": 0.05,
        "minimum_hidden_dim": 7,
        "maximum_hidden_dim": 192,
        "registered_models": ["ncp_cfc", "random_sparse_cfc"],
        "ncp_sparsity": 0.5,
        "ncp_seed": 51,
        "random_topology_seed": 52,
    }
    registry = build_parameter_registry(config)
    ncp = build_registered_model("ncp_cfc", config, registry)
    random_model = build_registered_model("random_sparse_cfc", config, registry)
    assert ncp.trainable_parameters == random_model.trainable_parameters
    assert (
        registry["entries"]["ncp_cfc"]["effective_active_trainable_parameters"]
        == registry["entries"]["random_sparse_cfc"][
            "effective_active_trainable_parameters"
        ]
    )
    assert abs(registry["entries"]["ncp_cfc"]["relative_error"]) <= 0.05
    assert (
        registry["entries"]["ncp_cfc"]["nominal_allocated_trainable_parameters"]
        > registry["entries"]["ncp_cfc"]["effective_active_trainable_parameters"]
    )
    ncp_entry = registry["entries"]["ncp_cfc"]
    assert ncp_entry["trainable_parameters"] == ncp_entry[
        "nominal_allocated_trainable_parameters"
    ]
    assert ncp_entry["trainable_parameters_semantics"] == (
        "deprecated_alias_of_nominal_allocated_trainable_parameters"
    )
    assert ncp_entry["parameter_matching_basis"] == (
        "effective_active_trainable_parameters"
    )
    assert ncp_entry["cell_implementation"] == (
        "local_full_state_synchronous_direct_graph_cfc"
    )
    assert ncp_entry["adamw_moment_state_scalars"] == 2 * ncp_entry[
        "nominal_allocated_trainable_parameters"
    ]
    assert ncp_entry["masked_affine_dense_compute_overhead_ratio"] > 1.0
    assert ncp_entry[
        "masked_affine_allocated_weight_slots_per_recurrent_update"
    ] > ncp_entry["masked_affine_active_weight_slots_per_recurrent_update"]
    assert ncp.topology_audit["selected_graph"] == "ncp"
    assert random_model.topology_audit["selected_graph"] == "degree_matched_random"
    assert ncp.topology_audit["global_graph_mask_applied"] is True
    assert ncp.topology_audit["all_cfc_paths_respect_graph_mask"] is True
    assert ncp.topology_audit["motor_node_indices"] == [0, 1, 2, 3]
    assert registry["topology_pair"]["ncp_pathway_constraints_preserved"] is False
    assert (
        ncp.topology_audit["ncp_recurrent_sha256"]
        != ncp.topology_audit["random_recurrent_sha256"]
    )
    import torch

    trainable_ncp = dict(ncp.named_parameters())
    trainable_random = dict(random_model.named_parameters())
    for name, parameter in trainable_ncp.items():
        if parameter.requires_grad:
            trainable_random[name].data.copy_(parameter.data)
    x_ncp = torch.randn(2, 4, 19, requires_grad=True)
    x_random = x_ncp.detach().clone().requires_grad_(True)
    dt = torch.full((2, 4), 0.02)
    output_ncp = ncp(x_ncp, dt)
    output_random = random_model(x_random, dt)
    assert float((output_ncp - output_random).abs().max().detach()) > 1e-7
    output_ncp.sum().backward()
    output_random.sum().backward()
    assert float((x_ncp.grad - x_random.grad).abs().max().detach()) > 1e-7

    ncp.zero_grad(set_to_none=True)
    cell = ncp.core.rnn_cell._layers[0]
    graph_input = torch.randn(3, 19)
    graph_hidden = torch.randn(3, ncp.hidden_dim)
    graph_dt = torch.full((3, 1), 0.02)
    hidden, _ = ncp._global_graph_cfc_step(graph_input, graph_hidden, graph_dt)
    hidden.square().sum().backward()
    allowed = cell.sparsity_mask.bool()
    forbidden = ~allowed
    for head_name in ("ff1", "ff2", "time_a", "time_b"):
        gradient = getattr(cell, head_name).weight.grad
        assert torch.count_nonzero(gradient[forbidden]) == 0
        assert torch.all(gradient[allowed].abs() > 0)

    dense = build_model(
        "cfc",
        {
            "input_dim": 19,
            "state_dim": 4,
            "hidden_dim": ncp.hidden_dim,
            "topology": {"kind": "dense"},
            "liquid": {"cfc_mode": "default"},
        },
    )
    dense_cell = dense.core.rnn_cell._layers[0]
    assert type(dense_cell) is type(cell)
    assert dense.topology_audit["dense_sparse_common_cell"] is True
    assert torch.all(dense_cell.sparsity_mask == 1)
    assert dense.topology_audit["motor_node_indices"] == [0, 1, 2, 3]


@pytest.mark.skipif(
    not formal.TRUSTED_NVIDIA_SMI_PATH.is_file(),
    reason="Formal environment identity test needs the Windows System32 NVIDIA binary.",
)
def test_controller_selection_can_replace_placeholder_before_final_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config()
    controller_artifact = tmp_path / "controller_selection.json"
    environment_artifact = tmp_path / "environment.json"
    config["experiment"]["development_only"] = False
    config["formal"] = {
        "required": True,
        "environment_artifact": str(environment_artifact),
    }
    config["trusted_nvidia_smi"] = formal.trusted_nvidia_smi_identity()
    config["controller_selection"] = {
        "artifact": str(controller_artifact),
        "kp_grid": [10.0, 20.0],
        "kd_grid": [1.0, 2.0],
    }

    def fake_rollout(estimator, candidate_config, condition, **kwargs):
        del estimator, condition, kwargs
        kp = float(candidate_config["controller"]["joint_kp"])
        kd = float(candidate_config["controller"]["joint_kd"])
        return {
            "failure": 0.0,
            "tracking_rmse": (kp - 20.0) ** 2 + (kd - 2.0) ** 2,
            "control_effort": kp + kd,
        }

    import liquid_revision.closed_loop.benchmark as benchmark_module

    monkeypatch.setattr(benchmark_module, "_rollout", fake_rollout)
    before = current_environment_identity(config)
    selected = create_controller_selection(config, controller_artifact)
    assert selected["selected"]["joint_kp"] == 20.0
    assert selected["selected"]["joint_kd"] == 2.0
    config["controller"]["joint_kp"] = 20.0
    config["controller"]["joint_kd"] = 2.0
    assert current_environment_identity(config) == before
    create_environment_freeze(config, environment_artifact)
    assert validate_environment_freeze(config) is not None
    assert validate_controller_selection(config) is not None
    forged = json.loads(controller_artifact.read_text(encoding="utf-8"))
    forged["selected"] = forged["candidates"][0]
    config["controller"]["joint_kp"] = float(forged["selected"]["joint_kp"])
    config["controller"]["joint_kd"] = float(forged["selected"]["joint_kd"])
    forged_core = {
        key: value
        for key, value in forged.items()
        if key not in {"controller_selection_sha256", "created_at_utc"}
    }
    forged["controller_selection_sha256"] = canonical_json_hash(forged_core)
    controller_artifact.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic grid winner"):
        validate_controller_selection(config)


def test_tuning_selection_rejects_incomplete_wrong_seed_test_touch_and_tamper(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    selection_path = tmp_path / "tuning_selection.json"
    config["tuning"] = {
        "artifact": str(selection_path),
        "seed": 17,
        "learning_rates": [0.0003, 0.001],
        "weight_decays": [0.0, 0.0001],
        "epochs": 1,
        "optimizer_steps_per_epoch": 2,
        "planned_optimizer_steps": 2,
    }
    data_root = tmp_path / "data"
    registry_path = tmp_path / "registry.json"
    prepare_split_datasets(config, data_root)
    create_parameter_registry(config, registry_path)
    registry = load_parameter_registry(registry_path)
    trajectory = load_trajectory_manifest(data_root / "trajectory_manifest.json")
    candidate_paths = []
    for index, (lr, wd) in enumerate(
        ((0.0003, 0.0), (0.0003, 0.0001), (0.001, 0.0), (0.001, 0.0001))
    ):
        result = train_tuning_candidate(
            registered_name="lstm",
            config=config,
            data_root=data_root,
            registry_path=registry_path,
            output_dir=tmp_path / f"candidate_{index}",
            learning_rate=lr,
            weight_decay=wd,
            device="cpu",
        )
        candidate_paths.append(Path(result["tuning_candidate_result"]))

    with pytest.raises(ValueError, match="Incomplete tuning grid"):
        create_tuning_selection(
            config, registry_path, data_root, candidate_paths[:-1], tmp_path / "missing.json"
        )

    original = json.loads(candidate_paths[-1].read_text(encoding="utf-8"))
    wrong_seed = dict(original)
    wrong_seed["seed"] = 18
    wrong_core = {
        key: value for key, value in wrong_seed.items() if key != "tuning_candidate_sha256"
    }
    wrong_seed["tuning_candidate_sha256"] = canonical_json_hash(wrong_core)
    wrong_seed_path = tmp_path / "wrong_seed.json"
    wrong_seed_path.write_text(json.dumps(wrong_seed), encoding="utf-8")
    with pytest.raises(ValueError, match="Tuning seed mismatch"):
        create_tuning_selection(
            config,
            registry_path,
            data_root,
            [*candidate_paths[:-1], wrong_seed_path],
            tmp_path / "wrong_seed_selection.json",
        )

    test_touched = dict(original)
    test_touched["test_episodes_used"] = 1
    touched_core = {
        key: value for key, value in test_touched.items() if key != "tuning_candidate_sha256"
    }
    test_touched["tuning_candidate_sha256"] = canonical_json_hash(touched_core)
    touched_path = tmp_path / "test_touched.json"
    touched_path.write_text(json.dumps(test_touched), encoding="utf-8")
    with pytest.raises(ValueError, match="touched test"):
        create_tuning_selection(
            config,
            registry_path,
            data_root,
            [*candidate_paths[:-1], touched_path],
            tmp_path / "test_touched_selection.json",
        )

    tampered = dict(original)
    tampered["validation_loss"] = -999.0
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        create_tuning_selection(
            config,
            registry_path,
            data_root,
            [*candidate_paths[:-1], tampered_path],
            tmp_path / "tampered_selection.json",
        )

    forged = dict(original)
    forged_checkpoint_path = tmp_path / "forged_consistent.pt"
    forged_checkpoint = torch.load(
        original["checkpoint_path"], map_location="cpu", weights_only=False
    )
    forged_checkpoint["run_phase"] = "confirmatory"
    torch.save(forged_checkpoint, forged_checkpoint_path)
    from liquid_revision.io_utils import sha256_file

    forged["checkpoint_path"] = str(forged_checkpoint_path)
    forged["checkpoint_sha256"] = sha256_file(forged_checkpoint_path)
    forged_core = {
        key: value for key, value in forged.items() if key != "tuning_candidate_sha256"
    }
    forged["tuning_candidate_sha256"] = canonical_json_hash(forged_core)
    forged_path = tmp_path / "forged_consistent.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="run_phase"):
        create_tuning_selection(
            config,
            registry_path,
            data_root,
            [*candidate_paths[:-1], forged_path],
            tmp_path / "forged_consistent_selection.json",
        )

    create_tuning_selection(
        config, registry_path, data_root, candidate_paths, selection_path
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection["selected"]["lstm"]
    alternative = next(
        row
        for row in selection["candidates"]
        if (
            float(row["learning_rate"]),
            float(row["weight_decay"]),
        )
        != (
            float(chosen["learning_rate"]),
            float(chosen["weight_decay"]),
        )
    )
    selection["selected"]["lstm"] = {
        "learning_rate": float(alternative["learning_rate"]),
        "weight_decay": float(alternative["weight_decay"]),
        "validation_loss": float(alternative["validation_loss"]),
        "checkpoint_sha256": alternative["checkpoint_sha256"],
    }
    selection_core = {
        key: value
        for key, value in selection.items()
        if key not in {"tuning_selection_sha256", "created_at_utc"}
    }
    selection["tuning_selection_sha256"] = canonical_json_hash(selection_core)
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic grid winners"):
        validate_tuning_selection(config, registry, trajectory)


def test_tiny_train_checkpoint_and_response_surface(tmp_path: Path) -> None:
    config = _tiny_config()
    data_root = tmp_path / "data"
    prepare_split_datasets(config, data_root)
    registry_path = tmp_path / "registry.json"
    create_parameter_registry(config, registry_path)
    training = train_registered_model(
        registered_name="lstm",
        seed=13,
        config=config,
        data_root=data_root,
        registry_path=registry_path,
        output_dir=tmp_path / "training",
        device="cpu",
    )
    assert training["status"] == "complete"
    assert training["optimizer_steps_planned"] == 2
    assert training["optimizer_steps_completed"] == 2
    assert training["fixed_budget_gate_passed"] is True
    observer, checkpoint = load_trained_observer(
        training["best_checkpoint"], expected_config=config
    )
    assert checkpoint["validation_split"] == "val"
    assert observer.training is False
    report = evaluate_response_surface(
        config=config,
        data_root=data_root,
        checkpoints=[training["best_checkpoint"]],
        output_dir=tmp_path / "evaluation",
        device="cpu",
    )
    assert report["fixed_checkpoints_no_retraining"] is True
    assert report["paired_episode_and_schedule_seeds"] is True
    assert set(report["models"]) == {
        "full_state_pd",
        "zoh_pd",
        "kalman_pd",
        "lstm",
    }
    assert (tmp_path / "evaluation" / "tracking_naurc.csv").is_file()
    assert (tmp_path / "evaluation" / "mechanism_contrasts.json").is_file()
    with (tmp_path / "evaluation" / "episode_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        episode_rows = list(csv.DictReader(handle))
    assert {
        "joint_position_rmse",
        "joint_velocity_rmse",
        "state_accuracy_finite_steps",
        "state_accuracy_finite_fraction",
        "recovery_seconds",
        "control_observed_steps",
        "plant_advanced_steps",
        "plant_completion_fraction",
    } <= set(episode_rows[0])
    assert all(
        float(row["control_observed_steps"])
        == float(row["plant_advanced_steps"])
        for row in episode_rows
    )
    for row in episode_rows:
        if row["family"] == "controlled_single_burst":
            assert float(row["recovery_seconds"]) == pytest.approx(
                float(row["recovery_steps"]) * 0.02
            )
    resumed = evaluate_response_surface(
        config=config,
        data_root=data_root,
        checkpoints=[training["best_checkpoint"]],
        output_dir=tmp_path / "evaluation",
        device="cpu",
        resume=True,
    )
    assert resumed["files"] == report["files"]


def test_tuning_candidate_runner_emits_hashed_validation_only_evidence(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    config["tuning"] = {
        "seed": 17,
        "learning_rates": [0.001, 0.002],
        "weight_decays": [0.0, 0.0001],
        "epochs": 1,
        "optimizer_steps_per_epoch": 2,
        "planned_optimizer_steps": 2,
    }
    data_root = tmp_path / "data"
    registry_path = tmp_path / "registry.json"
    prepare_split_datasets(config, data_root)
    create_parameter_registry(config, registry_path)
    result = train_tuning_candidate(
        registered_name="lstm",
        config=config,
        data_root=data_root,
        registry_path=registry_path,
        output_dir=tmp_path / "candidate",
        learning_rate=0.001,
        weight_decay=0.0,
        device="cpu",
    )
    artifact = json.loads(
        Path(result["tuning_candidate_result"]).read_text(encoding="utf-8")
    )
    core = {
        key: value
        for key, value in artifact.items()
        if key != "tuning_candidate_sha256"
    }
    assert artifact["tuning_candidate_sha256"] == canonical_json_hash(core)
    assert artifact["validation_only"] is True
    assert artifact["test_episodes_used"] == 0
    assert artifact["seed"] == 17
    assert artifact["optimizer_steps_completed"] == 2
    from liquid_revision.closed_loop.formal import (
        _validate_tuning_candidate_checkpoint,
    )

    registry = load_parameter_registry(registry_path, expected_config=config)
    trajectory = load_trajectory_manifest(
        data_root / "trajectory_manifest.json", expected_config=config
    )
    design = tuning_design_identity(config, registry, trajectory)
    epoch_path = next((tmp_path / "candidate" / "epoch_checkpoints").glob("*.pt"))
    epoch_path.unlink()
    with pytest.raises(ValueError, match="epoch ledger is incomplete or divergent"):
        _validate_tuning_candidate_checkpoint(
            artifact,
            config=config,
            registry=registry,
            trajectory=trajectory,
            design=design,
        )


def test_completed_evaluation_resume_rejects_tampered_derived_file(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    data_root = tmp_path / "data"
    prepare_split_datasets(config, data_root)
    evaluation = tmp_path / "evaluation"
    report = evaluate_response_surface(
        config=config,
        data_root=data_root,
        checkpoints=[],
        output_dir=evaluation,
        device="cpu",
    )
    assert report["status"] == "complete"
    with (evaluation / "condition_summary.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        evaluate_response_surface(
            config=config,
            data_root=data_root,
            checkpoints=[],
            output_dir=evaluation,
            device="cpu",
            resume=True,
        )


def test_running_evaluation_resume_is_terminal_before_test_schedule_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config()
    data_root = tmp_path / "data"
    prepare_split_datasets(config, data_root)
    evaluation = tmp_path / "evaluation"
    evaluate_response_surface(
        config=config,
        data_root=data_root,
        checkpoints=[],
        output_dir=evaluation,
        device="cpu",
    )
    state_path = evaluation / "evaluation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state.pop("evaluation_manifest_sha256", None)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    touched = False

    def _forbidden_audit(*args: object, **kwargs: object) -> object:
        nonlocal touched
        touched = True
        raise AssertionError("running resume touched the registered test schedule")

    monkeypatch.setattr(benchmark, "_audit_burst_realization", _forbidden_audit)
    with pytest.raises(RuntimeError, match="may not resume"):
        evaluate_response_surface(
            config=config,
            data_root=data_root,
            checkpoints=[],
            output_dir=evaluation,
            device="cpu",
            resume=True,
        )
    assert touched is False


def test_preoutput_claim_crash_can_resume_exact_bound_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config()
    data_root = tmp_path / "data"
    prepare_split_datasets(config, data_root)
    evaluation = tmp_path / "evaluation"
    monkeypatch.setattr(
        formal,
        "claim_test_release",
        lambda *args, **kwargs: {
            "status": "claimed",
            "externally_custodied": True,
            "test_release_state_sha256": "0" * 64,
        },
    )
    report = evaluate_response_surface(
        config=config,
        data_root=data_root,
        checkpoints=[],
        output_dir=evaluation,
        device="cpu",
        resume=True,
    )
    state = json.loads(
        (evaluation / "evaluation_state.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "complete"
    assert state["claim_recovery_after_preoutput_crash"] is True


def test_rehashed_episode_shard_forgery_is_recomputed_and_rejected(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    data_root = tmp_path / "data"
    prepare_split_datasets(config, data_root)
    evaluation = tmp_path / "evaluation"
    evaluate_response_surface(
        config=config,
        data_root=data_root,
        checkpoints=[],
        output_dir=evaluation,
        device="cpu",
    )
    shard_directory = next(path for path in (evaluation / "shards").iterdir() if path.is_dir())
    episode_file = shard_directory / "episode_metrics.csv"
    with episode_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["tracking_rmse"] = str(float(rows[0]["tracking_rmse"]) + 0.25)
    with episode_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    shard_manifest_path = shard_directory / "manifest.json"
    shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
    shard_manifest["episode_metrics_sha256"] = sha256_file(episode_file)
    core = {
        key: value
        for key, value in shard_manifest.items()
        if key != "shard_manifest_sha256"
    }
    shard_manifest["shard_manifest_sha256"] = canonical_json_hash(core)
    shard_manifest_path.write_text(
        json.dumps(shard_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Derived episode metric mismatch"):
        evaluate_response_surface(
            config=config,
            data_root=data_root,
            checkpoints=[],
            output_dir=evaluation,
            device="cpu",
            resume=True,
        )


def test_fully_rehashed_raw_metric_forgery_fails_physical_plant_replay(
    tmp_path: Path,
) -> None:
    """CSV and NPZ metrics cannot be forged together and blessed by rehashing."""
    config = _tiny_config()
    data_root = tmp_path / "data"
    prepare_split_datasets(config, data_root)
    evaluation = tmp_path / "evaluation"
    evaluate_response_surface(
        config=config,
        data_root=data_root,
        checkpoints=[],
        output_dir=evaluation,
        device="cpu",
    )
    shard_directory = next(
        path
        for path in (evaluation / "shards").iterdir()
        if path.is_dir()
        and json.loads((path / "manifest.json").read_text(encoding="utf-8"))[
            "condition_id"
        ]
        == "iid_dropout-0"
    )
    step_file = shard_directory / "step_log.npz"
    with np.load(step_file, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    target_id = str(arrays["trajectory_id"][0])
    arrays["tracking_error"][0] = float(arrays["tracking_error"][0]) + 0.05
    np.savez_compressed(step_file, **arrays)

    episode_file = shard_directory / "episode_metrics.csv"
    with episode_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    row = next(item for item in rows if item["trajectory_id"] == target_id)
    selected = np.asarray(arrays["trajectory_id"]).astype(str) == target_id
    errors = np.asarray(arrays["tracking_error"], dtype=float)[selected]
    penalty_steps = int(float(row["tracking_penalty_steps"]))
    if penalty_steps:
        errors = np.concatenate(
            [
                errors,
                np.full(
                    penalty_steps,
                    float(row["tracking_failure_penalty_m"]),
                    dtype=float,
                ),
            ]
        )
    row["tracking_rmse"] = str(float(np.sqrt(np.mean(np.square(errors)))))
    row["tracking_p95"] = str(float(np.quantile(errors, 0.95)))
    with episode_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = shard_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["step_log_sha256"] = sha256_file(step_file)
    manifest["episode_metrics_sha256"] = sha256_file(episode_file)
    core = {
        key: value
        for key, value in manifest.items()
        if key != "shard_manifest_sha256"
    }
    manifest["shard_manifest_sha256"] = canonical_json_hash(core)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Rehash every mutable wrapper in the completed output so that rejection
    # must come from deterministic MuJoCo replay rather than a stale checksum.
    step_manifest_path = evaluation / "step_log_manifest.json"
    step_manifest = json.loads(step_manifest_path.read_text(encoding="utf-8"))
    matching = next(
        item
        for item in step_manifest["shards"]
        if Path(item["path"]).resolve() == step_file.resolve()
    )
    matching["sha256"] = sha256_file(step_file)
    step_core = {
        key: value
        for key, value in step_manifest.items()
        if key != "step_log_manifest_sha256"
    }
    step_manifest["step_log_manifest_sha256"] = canonical_json_hash(step_core)
    step_manifest_path.write_text(
        json.dumps(step_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    release_path = evaluation / "evaluation_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["files"]["step_log_manifest.json"] = sha256_file(step_manifest_path)
    release["step_log_manifest_sha256"] = step_manifest[
        "step_log_manifest_sha256"
    ]
    release_core = {
        key: value
        for key, value in release.items()
        if key != "evaluation_release_sha256"
    }
    release["evaluation_release_sha256"] = canonical_json_hash(release_core)
    release_path.write_text(
        json.dumps(release, indent=2, sort_keys=True), encoding="utf-8"
    )
    state_path = evaluation / "evaluation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["evaluation_manifest_sha256"] = sha256_file(release_path)
    for item in state["completed_shards"]:
        if item["shard_name"] == shard_directory.name:
            item["shard_manifest_sha256"] = manifest["shard_manifest_sha256"]
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="Physical tracking_error replay mismatch"):
        evaluate_response_surface(
            config=config,
            data_root=data_root,
            checkpoints=[],
            output_dir=evaluation,
            device="cpu",
            resume=True,
        )


def _rehash_completed_evaluation_wrappers(
    evaluation: Path, shard_directory: Path
) -> None:
    """Simulate an attacker who can recompute every local JSON checksum."""
    step_file = shard_directory / "step_log.npz"
    episode_file = shard_directory / "episode_metrics.csv"
    shard_manifest_path = shard_directory / "manifest.json"
    shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
    shard_manifest["step_log_sha256"] = sha256_file(step_file)
    shard_manifest["episode_metrics_sha256"] = sha256_file(episode_file)
    shard_core = {
        key: value
        for key, value in shard_manifest.items()
        if key != "shard_manifest_sha256"
    }
    shard_manifest["shard_manifest_sha256"] = canonical_json_hash(shard_core)
    shard_manifest_path.write_text(
        json.dumps(shard_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    step_manifest_path = evaluation / "step_log_manifest.json"
    step_manifest = json.loads(step_manifest_path.read_text(encoding="utf-8"))
    matching = next(
        item
        for item in step_manifest["shards"]
        if Path(item["path"]).resolve() == step_file.resolve()
    )
    matching["sha256"] = sha256_file(step_file)
    step_core = {
        key: value
        for key, value in step_manifest.items()
        if key != "step_log_manifest_sha256"
    }
    step_manifest["step_log_manifest_sha256"] = canonical_json_hash(step_core)
    step_manifest_path.write_text(
        json.dumps(step_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    release_path = evaluation / "evaluation_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["files"]["step_log_manifest.json"] = sha256_file(step_manifest_path)
    release["step_log_manifest_sha256"] = step_manifest[
        "step_log_manifest_sha256"
    ]
    release_core = {
        key: value
        for key, value in release.items()
        if key != "evaluation_release_sha256"
    }
    release["evaluation_release_sha256"] = canonical_json_hash(release_core)
    release_path.write_text(
        json.dumps(release, indent=2, sort_keys=True), encoding="utf-8"
    )
    state_path = evaluation / "evaluation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["evaluation_manifest_sha256"] = sha256_file(release_path)
    for item in state["completed_shards"]:
        if item["shard_name"] == shard_directory.name:
            item["shard_manifest_sha256"] = shard_manifest[
                "shard_manifest_sha256"
            ]
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def test_rehashed_prediction_forgery_is_rejected_by_observer_replay(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    data_root = tmp_path / "data"
    prepare_split_datasets(config, data_root)
    evaluation = tmp_path / "evaluation"
    evaluate_response_surface(
        config=config,
        data_root=data_root,
        checkpoints=[],
        output_dir=evaluation,
        device="cpu",
    )
    shard_directory = next(
        path
        for path in (evaluation / "shards").iterdir()
        if path.is_dir()
        and json.loads((path / "manifest.json").read_text(encoding="utf-8"))[
            "condition_id"
        ]
        == "iid_dropout-0"
        and "full_state_pd" in path.name
    )
    step_file = shard_directory / "step_log.npz"
    with np.load(step_file, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    target_id = str(arrays["trajectory_id"][0])
    # Move in a PD-control null direction: kp*dq + kd*ddq = 0.  The physical
    # action/path can remain unchanged, but the state estimate is forged.
    delta = 1e-3
    kp = float(config["controller"]["joint_kp"])
    kd = float(config["controller"]["joint_kd"])
    arrays["predicted_state"][0, 0] += delta
    arrays["predicted_state"][0, 2] -= (kp / kd) * delta
    np.savez_compressed(step_file, **arrays)

    episode_file = shard_directory / "episode_metrics.csv"
    with episode_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    row = next(item for item in rows if item["trajectory_id"] == target_id)
    selected = np.asarray(arrays["trajectory_id"]).astype(str) == target_id
    prediction = np.asarray(arrays["predicted_state"], dtype=float)[selected]
    clean = np.asarray(arrays["clean_state_and_reference"], dtype=float)[selected, :4]
    row["state_rmse"] = str(float(np.sqrt(np.mean(np.square(prediction - clean)))))
    with episode_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _rehash_completed_evaluation_wrappers(evaluation, shard_directory)
    with pytest.raises(ValueError, match="Observer prediction replay mismatch"):
        evaluate_response_surface(
            config=config,
            data_root=data_root,
            checkpoints=[],
            output_dir=evaluation,
            device="cpu",
            resume=True,
        )


def test_completed_training_resume_is_strictly_read_only(tmp_path: Path) -> None:
    config = _tiny_config()
    data_root = tmp_path / "data"
    registry_path = tmp_path / "registry.json"
    output = tmp_path / "training"
    prepare_split_datasets(config, data_root)
    create_parameter_registry(config, registry_path)
    first = train_registered_model(
        registered_name="lstm",
        seed=13,
        config=config,
        data_root=data_root,
        registry_path=registry_path,
        output_dir=output,
        device="cpu",
    )
    paths = sorted(path for path in output.rglob("*") if path.is_file())
    before = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in paths}
    resumed = train_registered_model(
        registered_name="lstm",
        seed=13,
        config=config,
        data_root=data_root,
        registry_path=registry_path,
        output_dir=output,
        device="cpu",
        resume=True,
    )
    after_paths = sorted(path for path in output.rglob("*") if path.is_file())
    assert after_paths == paths
    after = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in paths}
    assert after == before
    assert resumed["best_checkpoint_sha256"] == first["best_checkpoint_sha256"]


def test_epoch_checkpoint_chain_is_write_once_and_resume_repairs_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config()
    config["training"]["epochs"] = 2
    data_root = tmp_path / "data"
    registry_path = tmp_path / "registry.json"
    output = tmp_path / "training"
    prepare_split_datasets(config, data_root)
    create_parameter_registry(config, registry_path)
    import liquid_revision.closed_loop.training as training_module

    original_write = training_module._write_epoch_checkpoint_ledger
    interrupted = {"done": False}

    def interrupt_after_epoch_commit(*args, **kwargs):
        if not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt("synthetic crash after immutable epoch commit")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(
        training_module, "_write_epoch_checkpoint_ledger", interrupt_after_epoch_commit
    )
    with pytest.raises(KeyboardInterrupt, match="synthetic crash"):
        train_registered_model(
            registered_name="lstm",
            seed=13,
            config=config,
            data_root=data_root,
            registry_path=registry_path,
            output_dir=output,
            device="cpu",
        )
    epoch_zero = output / "epoch_checkpoints" / "epoch_0000.pt"
    before = (sha256_file(epoch_zero), epoch_zero.stat().st_mtime_ns)
    monkeypatch.setattr(
        training_module, "_write_epoch_checkpoint_ledger", original_write
    )
    result = train_registered_model(
        registered_name="lstm",
        seed=13,
        config=config,
        data_root=data_root,
        registry_path=registry_path,
        output_dir=output,
        device="cpu",
        resume=True,
    )
    assert result["status"] == "complete"
    assert (sha256_file(epoch_zero), epoch_zero.stat().st_mtime_ns) == before
    epochs = sorted((output / "epoch_checkpoints").glob("epoch_*.pt"))
    assert [path.name for path in epochs] == ["epoch_0000.pt", "epoch_0001.pt"]
    ledger = json.loads((output / "checkpoint_ledger.json").read_text(encoding="utf-8"))
    assert ledger["checkpoint_count"] == 2
    assert ledger["checkpoints"][0]["previous_checkpoint_sha256"] is None
    assert ledger["checkpoints"][1]["previous_checkpoint_sha256"] == sha256_file(
        epoch_zero
    )
    with pytest.raises(FileExistsError):
        training_module._atomic_torch_save_new(epoch_zero, {"forged": True})
    assert (sha256_file(epoch_zero), epoch_zero.stat().st_mtime_ns) == before


def test_resume_rejects_self_rehashed_divergent_epoch_ledger(tmp_path: Path) -> None:
    config = _tiny_config()
    config["training"]["epochs"] = 2
    data_root = tmp_path / "data"
    registry_path = tmp_path / "registry.json"
    output = tmp_path / "training"
    prepare_split_datasets(config, data_root)
    create_parameter_registry(config, registry_path)
    train_registered_model(
        registered_name="lstm",
        seed=13,
        config=config,
        data_root=data_root,
        registry_path=registry_path,
        output_dir=output,
        device="cpu",
    )
    state_path = output / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    ledger_path = output / "checkpoint_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["checkpoints"][0]["validation_loss"] += 0.25
    core = {
        key: value
        for key, value in ledger.items()
        if key != "checkpoint_ledger_sha256"
    }
    ledger["checkpoint_ledger_sha256"] = canonical_json_hash(core)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    forged_hash = sha256_file(ledger_path)
    with pytest.raises(ValueError, match="exact immutable prefix"):
        train_registered_model(
            registered_name="lstm",
            seed=13,
            config=config,
            data_root=data_root,
            registry_path=registry_path,
            output_dir=output,
            device="cpu",
            resume=True,
        )
    assert sha256_file(ledger_path) == forged_hash


def test_release_recomputes_every_epoch_and_rejects_fully_rehashed_weight_forgery(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    config["training"]["epochs"] = 2
    data_root = tmp_path / "data"
    registry_path = tmp_path / "registry.json"
    output = tmp_path / "training"
    prepare_split_datasets(config, data_root)
    create_parameter_registry(config, registry_path)
    result = train_registered_model(
        registered_name="lstm",
        seed=13,
        config=config,
        data_root=data_root,
        registry_path=registry_path,
        output_dir=output,
        device="cpu",
    )
    from liquid_revision.closed_loop import formal as formal_module
    from liquid_revision.closed_loop import training as training_module

    registry = load_parameter_registry(registry_path, expected_config=config)
    ledger_path = output / "checkpoint_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    state_path = output / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    best_path = Path(result["best_checkpoint"])
    metadata = torch.load(best_path, map_location="cpu", weights_only=False)
    receipt = formal_module._validate_epoch_checkpoint_release(
        config=config,
        checkpoint_path=best_path,
        metadata=metadata,
        run_state=state,
        data_root=data_root,
        registry=registry,
        expected_identity=ledger["identity"],
        recompute_validation=True,
    )
    assert receipt["validation_recomputed"] is True
    assert receipt["checkpoint_count"] == 2
    assert receipt["selected_epoch_checkpoint_sha256"] == state[
        "selected_epoch_checkpoint_sha256"
    ]

    target_epoch = 1 - int(state["best_epoch"])
    epoch_paths = [
        output / "epoch_checkpoints" / f"epoch_{epoch:04d}.pt"
        for epoch in range(2)
    ]
    target = torch.load(epoch_paths[target_epoch], map_location="cpu", weights_only=False)
    first_key = next(iter(target["model_state_dict"]))
    target["model_state_dict"][first_key] = (
        target["model_state_dict"][first_key] + 10.0
    )
    training_module._atomic_torch_save(epoch_paths[target_epoch], target)
    for epoch in range(target_epoch + 1, 2):
        downstream = torch.load(
            epoch_paths[epoch], map_location="cpu", weights_only=False
        )
        downstream["previous_epoch_checkpoint_sha256"] = sha256_file(
            epoch_paths[epoch - 1]
        )
        training_module._atomic_torch_save(epoch_paths[epoch], downstream)
    forged_ledger = training_module._epoch_checkpoint_ledger(
        output / "epoch_checkpoints", expected_identity=ledger["identity"]
    )
    ledger_path.write_text(json.dumps(forged_ledger), encoding="utf-8")
    selected_path = epoch_paths[int(state["best_epoch"])]
    metadata["checkpoint_ledger_sha256"] = forged_ledger[
        "checkpoint_ledger_sha256"
    ]
    metadata["checkpoint_ledger_file_sha256"] = sha256_file(ledger_path)
    metadata["selected_epoch_checkpoint_sha256"] = sha256_file(selected_path)
    training_module._atomic_torch_save(best_path, metadata)
    state["checkpoint_ledger_sha256"] = forged_ledger[
        "checkpoint_ledger_sha256"
    ]
    state["checkpoint_ledger_file_sha256"] = sha256_file(ledger_path)
    state["selected_epoch_checkpoint_sha256"] = sha256_file(selected_path)
    state["best_checkpoint_sha256"] = sha256_file(best_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    forged_metadata = torch.load(best_path, map_location="cpu", weights_only=False)
    with pytest.raises(ValueError, match="independently recomputed validation loss"):
        formal_module._validate_epoch_checkpoint_release(
            config=config,
            checkpoint_path=best_path,
            metadata=forged_metadata,
            run_state=state,
            data_root=data_root,
            registry=registry,
            expected_identity=forged_ledger["identity"],
            recompute_validation=True,
        )


def test_training_failure_writes_immutable_ledger_and_forbids_in_place_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config()
    data_root = tmp_path / "data"
    registry_path = tmp_path / "registry.json"
    output = tmp_path / "failed_training"
    prepare_split_datasets(config, data_root)
    create_parameter_registry(config, registry_path)
    import liquid_revision.closed_loop.training as training_module

    def fail_epoch(*args, **kwargs):
        del args, kwargs
        raise FloatingPointError("synthetic nonfinite loss")

    monkeypatch.setattr(training_module, "_run_epoch", fail_epoch)
    with pytest.raises(FloatingPointError, match="synthetic"):
        train_registered_model(
            registered_name="lstm",
            seed=13,
            config=config,
            data_root=data_root,
            registry_path=registry_path,
            output_dir=output,
            device="cpu",
        )
    attempt = output / "failed_attempts" / "attempt_0001.json"
    payload = json.loads(attempt.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "failed_attempt_sha256"}
    assert payload["failed_attempt_sha256"] == canonical_json_hash(core)
    assert json.loads((output / "run_state.json").read_text(encoding="utf-8"))["status"] == "failed"
    with pytest.raises(ValueError, match="cannot be resumed in place"):
        train_registered_model(
            registered_name="lstm",
            seed=13,
            config=config,
            data_root=data_root,
            registry_path=registry_path,
            output_dir=output,
            device="cpu",
            resume=True,
        )
    assert list((output / "failed_attempts").glob("attempt_*.json")) == [attempt]


def test_incomplete_checkpoint_set_never_materializes_test_schedules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config()
    config["evaluation"].pop("max_test_episodes", None)
    config["evaluation"]["require_complete_registered_checkpoint_set"] = True
    config["training"]["seeds"] = [13]
    data_root = tmp_path / "data"
    prepare_split_datasets(config, data_root)
    from liquid_revision.closed_loop import benchmark as benchmark_module

    touched = False

    def forbidden_audit(*args, **kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("test schedule was materialized before release")

    monkeypatch.setattr(benchmark_module, "_audit_burst_realization", forbidden_audit)
    with pytest.raises(ValueError, match="checkpoint set is incomplete"):
        evaluate_response_surface(
            config=config,
            data_root=data_root,
            checkpoints=[],
            output_dir=tmp_path / "evaluation",
            device="cpu",
        )
    assert touched is False


def test_training_rejects_current_config_that_differs_from_frozen_sources(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    data_root = tmp_path / "data"
    prepare_split_datasets(config, data_root)
    registry_path = tmp_path / "registry.json"
    create_parameter_registry(config, registry_path)
    changed = json.loads(json.dumps(config))
    changed["training"]["learning_rate"] = 0.123
    with pytest.raises(
        ValueError,
        match="trajectory manifest source|deterministic rebuild of the frozen config",
    ):
        train_registered_model(
            registered_name="lstm",
            seed=13,
            config=changed,
            data_root=data_root,
            registry_path=registry_path,
            output_dir=tmp_path / "training",
            device="cpu",
        )


def test_response_condition_axes_are_isolated() -> None:
    conditions = build_response_conditions(_tiny_config())
    for condition in conditions:
        spec = condition["spec"]
        active = sum(
            [
                float(spec.get("iid_dropout_probability", 0)) > 0,
                int(spec.get("burst_length_steps", 0)) > 0,
                int(spec.get("delay_steps", 0)) > 0,
            ]
        )
        assert active <= 1


def test_recovery_keeps_unrecovered_event_as_right_censored_duration() -> None:
    errors = np.asarray([0.01, 0.01, 0.20, 0.20, 0.20, 0.20], dtype=float)
    disrupted = np.asarray([False, False, True, True, False, False])
    duration, censored, events, censored_events, threshold = _recovery_steps(
        errors,
        disrupted,
        floor_m=0.02,
        baseline_multiplier=1.0,
        baseline_window_steps=50,
        consecutive=2,
        horizon_steps=10,
    )
    assert duration == 6.0  # registered horizon remainder after disrupted step 3
    assert censored is True
    assert events == censored_events == 1
    assert threshold == pytest.approx(0.02)


def test_recovery_preburst_failure_uses_registered_schedule_and_finite_censor() -> None:
    errors = np.asarray([0.01, 0.015], dtype=float)
    observed_disrupted = np.asarray([False, False])
    scheduled = np.asarray(
        [False, False, False, False, True, True, False, False, False, False]
    )
    duration, censored, events, censored_events, threshold = _recovery_steps(
        errors,
        observed_disrupted,
        scheduled_disrupted=scheduled,
        floor_m=0.02,
        baseline_multiplier=1.0,
        baseline_window_steps=50,
        consecutive=2,
        horizon_steps=10,
    )
    assert duration == 4.0  # registered remainder after scheduled burst end at step 5
    assert censored is True
    assert events == censored_events == 1
    assert np.isfinite(duration)
    assert threshold == pytest.approx(0.02)


def test_mechanical_contrast_rejects_nonfinite_recovery() -> None:
    raw = [
        {
            "model_name": model,
            "training_seed": 101,
            "trajectory_id": "test-0",
            "family": "controlled_single_burst",
            "severity": 10.0,
            "recovery_steps": value,
            "recovery_seconds": value * 0.02,
            "failure": 0.0,
        }
        for model, value in (("ltc", math.nan), ("ctrnn", 4.0))
    ]
    with pytest.raises(ValueError, match="Non-finite paired recovery_seconds"):
        _mechanical_endpoint_contrasts(raw, _tiny_config())


def test_mechanism_contrast_sign_is_explicit_first_minus_second() -> None:
    raw = []
    target_naurc = {
        "ctrnn": 1.0,
        "ltc": 2.0,
        "cfc": 4.0,
        "ncp_cfc": 8.0,
        "random_sparse_cfc": 3.0,
        "lstm": 1.5,
        "odernn": 2.5,
    }
    for model, value in target_naurc.items():
        for seed in (101, 202):
            for trajectory_id in ("test-0", "test-1"):
                for severity, tracking_rmse in ((0.0, 1.0), (1.0, 2.0 * value - 1.0)):
                    raw.append(
                        {
                            "model_name": model,
                            "training_seed": seed,
                            "trajectory_id": trajectory_id,
                            "family": "iid_dropout",
                            "severity": severity,
                            "tracking_rmse": tracking_rmse,
                                "failure": 0.0,
                        }
                    )
    config = _tiny_config()
    config["evaluation"]["bootstrap_replicates"] = 20
    contrasts = _mechanism_contrasts(raw, config)
    iid = {
        row["contrast"]: row for row in contrasts["primary_packet_dropout"]
    }
    assert iid["adaptive_time_scale_ltc_minus_ctrnn"]["first_model"] == "ltc"
    assert iid["adaptive_time_scale_ltc_minus_ctrnn"]["second_model"] == "ctrnn"
    assert iid["adaptive_time_scale_ltc_minus_ctrnn"]["estimate_first_minus_second"] == 1.0
    assert iid["solver_free_cfc_minus_ltc"]["estimate_first_minus_second"] == 2.0
    assert iid["structured_ncp_minus_degree_matched_random"]["estimate_first_minus_second"] == 5.0
    assert all("holm_adjusted_p" in row for row in iid.values())
    assert all(row["pairing"].startswith("seed then episode") for row in iid.values())


def test_optimizer_profile_has_auditable_projection(tmp_path: Path) -> None:
    config = _tiny_config()
    registry_path = tmp_path / "registry.json"
    create_parameter_registry(config, registry_path)
    report = profile_optimizer_steps(
        config=config,
        registry_path=registry_path,
        output_path=tmp_path / "training_profile.json",
        device="cpu",
        steps=1,
        warmup_steps=0,
    )
    assert report["purpose"].startswith("capacity profile only")
    assert report["rows"][0]["measured_optimizer_steps"] == 1
    assert report["rows"][0]["projected_training_hours_per_seed"] >= 0
    assert (tmp_path / "training_profile.csv").is_file()


def test_declared_optimizer_budget_mismatch_fails_before_training(tmp_path: Path) -> None:
    config = _tiny_config()
    config["training"]["optimizer_steps_per_epoch"] = 999
    data_root = tmp_path / "data"
    prepare_split_datasets(config, data_root)
    registry_path = tmp_path / "registry.json"
    create_parameter_registry(config, registry_path)
    with pytest.raises(RuntimeError, match="Frozen optimizer-budget gate"):
        train_registered_model(
            registered_name="lstm",
            seed=13,
            config=config,
            data_root=data_root,
            registry_path=registry_path,
            output_dir=tmp_path / "training",
            device="cpu",
        )


def test_rollout_profile_never_uses_frozen_test_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config()
    from liquid_revision.closed_loop import profiling as profiling_module

    seen: list[tuple[int, int]] = []

    def fake_rollout(estimator, config, condition, *, episode_seed, schedule_seed, device):
        del estimator, config, condition, device
        seen.append((int(episode_seed), int(schedule_seed)))
        return {"steps": 4}

    monkeypatch.setattr(profiling_module, "_rollout", fake_rollout)
    report = profile_rollout_throughput(
        config=config,
        checkpoints=[],
        output_path=tmp_path / "rollout_profile.json",
        device="cpu",
        episodes=1,
    )
    assert report["nonlearned_estimators_profiled"] == 3
    assert report["projected_confirmatory_plant_steps"] > 0
    assert report["formal_projection_complete"] is False
    assert all(row["plant_steps_per_second"] > 0 for row in report["rows"])
    test_spec = config["splits"]["test"]
    frozen_test_episode_ids = set(
        range(
            test_spec["episode_seed_start"],
            test_spec["episode_seed_start"] + test_spec["episodes"],
        )
    )
    frozen_test_schedule_ids = set(
        range(
            test_spec["schedule_seed_start"],
            test_spec["schedule_seed_start"] + test_spec["episodes"],
        )
    )
    assert all(ep not in frozen_test_episode_ids for ep, _ in seen)
    assert all(schedule not in frozen_test_schedule_ids for _, schedule in seen)
    assert report["validation_or_synthetic_only"] is True
    assert report["test_episodes_used"] == 0
    assert report["profile_seed_namespace_sha256"]


def test_root_cli_help_renders_all_closed_loop_subcommands() -> None:
    from liquid_revision.cli import _build_parser

    help_text = _build_parser().format_help()
    assert "closed-loop-profile-training" in help_text
    assert "closed-loop-profile-rollout" in help_text
    assert "closed-loop-freeze-checkpoints" in help_text
    assert "closed-loop-freeze-pretraining" in help_text
    assert "closed-loop-authorize-test" in help_text
