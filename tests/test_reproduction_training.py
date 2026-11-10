"""Check the scaled motor readout and epoch recovery in the public workflow."""

from pathlib import Path

import numpy as np
import pytest
import torch

from liquid_revision.closed_loop import training as core
from liquid_revision.closed_loop.manifests import prepare_split_datasets
from liquid_revision.closed_loop.parameter_matching import create_parameter_registry
from liquid_revision.config import load_config
from liquid_revision.reproduction.readout import TrainRangeMotorSlice, train_range_radius
from liquid_revision.reproduction.training import fit_observer, load_observer


def test_train_range_readout_uses_channelwise_radius():
    targets = np.array(
        [[0, 0, 0, 0], [1, 2, 0, 2], [1, 5, 2, 2], [6, 7, 5, 2]],
        dtype=np.float32,
    )
    arrays = {"target_state": targets, "observer_input": targets}
    z = (targets - targets.mean(axis=0)) / targets.std(axis=0)
    expected = 1.05 * np.abs(z).max(axis=0)
    radius = train_range_radius(arrays)
    np.testing.assert_allclose(radius, expected)
    head = TrainRangeMotorSlice(radius)
    torch.testing.assert_close(head(torch.ones(1, 6)), torch.tensor(radius)[None, :])
    assert not head.state_dict()


def test_interrupted_scaled_fit_resumes_from_committed_epoch(tmp_path, monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "configs/closed_loop_paper_smoke.yaml"
    config, _ = load_config(config_path)
    config["training"].update(epochs=2, planned_optimizer_steps=6)
    data_root = tmp_path / "data"
    registry_path = tmp_path / "registry.json"
    prepare_split_datasets(config, data_root)
    create_parameter_registry(config, registry_path)
    args = {
        "config": config,
        "data_root": data_root,
        "registry_path": registry_path,
        "model": "cfc",
        "seed": 101,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "readout": "scaled",
    }
    run_epoch = core._run_epoch
    training_epochs = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal training_epochs
        if kwargs["optimizer"] is not None:
            training_epochs += 1
            if training_epochs == 2:
                raise RuntimeError("Simulated interruption")
        return run_epoch(*args, **kwargs)

    resumed = tmp_path / "resumed"
    with monkeypatch.context() as patch:
        patch.setattr(core, "_run_epoch", interrupt_second_epoch)
        with pytest.raises(RuntimeError, match="Simulated interruption"):
            fit_observer(output_dir=resumed, **args)
    assert (resumed / "epoch_checkpoints" / "epoch_0000.pt").is_file()
    state = fit_observer(output_dir=resumed, **args)
    reference = fit_observer(output_dir=tmp_path / "continuous", **args)
    assert state["optimizer_steps_completed"] == reference["optimizer_steps_completed"] == 6
    assert state["best_val_loss"] == reference["best_val_loss"]
    assert fit_observer(output_dir=resumed, **args) == state

    observer, metadata = load_observer(
        resumed / "best.pt", config=config, registry_path=registry_path,
        model="cfc", seed=101, readout="scaled",
    )
    torch.testing.assert_close(observer.model.head.target_radius, torch.tensor(metadata["readout_radius"]))
    assert not observer.training
