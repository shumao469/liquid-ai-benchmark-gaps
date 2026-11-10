"""Fit portable unit-range and channel-scaled observers on train/validation splits."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time

import torch

from ..closed_loop import training as core
from ..closed_loop.formal import apply_and_validate_precision_policy, is_development_only, is_formal
from ..closed_loop.manifests import load_trajectory_manifest
from ..closed_loop.parameter_matching import load_parameter_registry
from ..config import load_config
from ..data.closed_loop import load_closed_loop_dataset
from ..io_utils import atomic_write_json, canonical_json_hash, sha256_file
from . import readout as motor


SCOPE = "exploratory_closed_loop_reproduction"
TUNING_SCOPE = "validation_only_seed17"
TUNING_SCHEMA = "liquid_revision.reproduction.tuning_selection.v1"
SELECTION_RULE = "minimum_finite_validation_loss_then_earliest_epoch"
SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_reproduction(config: dict) -> None:
    if is_formal(config) or not is_development_only(config):
        raise ValueError("Use an exploratory reproduction config with formal.required=false and experiment.development_only=true")


@contextmanager
def _run_lock(path: Path):
    """Use an OS lock that is released when a worker exits or is interrupted."""
    with path.open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another worker owns the run directory: {path.parent}") from exc
        yield


def _split_arrays(data_root: Path, config: dict) -> tuple[dict, dict]:
    manifest = load_trajectory_manifest(data_root / "trajectory_manifest.json", expected_config=config)
    datasets, digests = {}, {}
    for split in ("train", "val"):
        arrays, binding = load_closed_loop_dataset(data_root / split)
        if (
            binding.get("source_split") != split
            or binding.get("source_trajectory_manifest_sha256") != manifest["manifest_sha256"]
            or binding.get("source_trajectory_rows") != manifest["splits"][split]
        ):
            raise ValueError(f"Dataset differs from the configured {split} trajectories")
        datasets[split] = arrays
        digests[split] = binding["dataset_sha256"]
    return datasets, digests


def _precision(config: dict, device: torch.device) -> dict:
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Select a CPU or CUDA device")
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Match the numeric policy used for the paper's paired comparison.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    return apply_and_validate_precision_policy(config)


def _phase_settings(
    config: dict, *, phase: str, seed: int, learning_rate: float, weight_decay: float
) -> dict:
    tuning = config["tuning"]
    if learning_rate not in tuning["learning_rates"] or weight_decay not in tuning["weight_decays"]:
        raise ValueError("Select learning rate and weight decay from the configured grid")
    if phase == "tune" and seed == tuning["seed"]:
        return tuning
    if phase == "train" and seed in config["training"]["seeds"]:
        return config["training"]
    raise ValueError("Use the configured tuning seed or a configured final-fit seed")


def _require_identity(payload: dict, identity: dict) -> None:
    for field, expected in identity.items():
        if payload.get(field) != expected:
            raise ValueError(f"Run identity differs for {field}; choose a new output directory")


def _require_completed(directory: Path, state: dict) -> None:
    if (
        state.get("status") != "complete"
        or state.get("optimizer_steps_completed") != state.get("optimizer_steps_planned")
        or state.get("epochs_completed") != state.get("epochs_planned")
        or state.get("fixed_budget_gate_passed") is not True
        or not math.isfinite(float(state.get("best_val_loss", math.nan)))
    ):
        raise ValueError(f"Expected a completed fixed-budget fit at {directory}")
    for filename, field in (
        ("best.pt", "best_checkpoint_sha256"),
        ("latest.pt", "latest_checkpoint_sha256"),
        ("checkpoint_ledger.json", "checkpoint_ledger_file_sha256"),
    ):
        if sha256_file(directory / filename) != state.get(field):
            raise ValueError(f"Completed fit changed: {directory / filename}")


def fit_observer(
    *,
    config: dict,
    data_root: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
    model: str,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    phase: str = "train",
    readout: str = "unit",
    device: str = "cpu",
    selection_sha256: str | None = None,
    expected_data_sha256: dict | None = None,
) -> dict:
    """Run a fixed-budget fit and select the earliest minimum validation loss."""
    _require_reproduction(config)
    if readout not in ("unit", "scaled") or (readout == "scaled" and model not in motor.GRAPH_CFC_MODELS):
        raise ValueError("Choose unit readout or scaled readout for a Graph-CfC model")
    budget = _phase_settings(config, phase=phase, seed=seed, learning_rate=learning_rate, weight_decay=weight_decay)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    with _run_lock(directory / ".worker.lock"):
        datasets, data_sha256 = _split_arrays(Path(data_root), config)
        if expected_data_sha256 is not None and data_sha256 != expected_data_sha256:
            raise ValueError("The fit data differ from the validation tuning selection")
        registry = load_parameter_registry(registry_path)
        normalizer = core.fit_normalizer(datasets["train"])
        radius = None
        if readout == "scaled":
            radius = motor.train_range_radius(datasets["train"], normalizer).tolist()
        settings = config["training"]
        train_windows, val_windows = (
            core.SequenceWindows(datasets[split], normalizer, settings["window_steps"], settings["burn_in_steps"])
            for split in ("train", "val")
        )
        del datasets
        batch_size = int(settings["batch_size"])
        steps_per_epoch = math.ceil(len(train_windows) / batch_size)
        epochs = int(budget["epochs"])
        optimizer_steps = epochs * steps_per_epoch
        if (
            epochs < 1
            or int(budget["optimizer_steps_per_epoch"]) != steps_per_epoch
            or int(budget["planned_optimizer_steps"]) != optimizer_steps
        ):
            raise ValueError("The training data and config do not match the specified optimizer budget")
        precision = _precision(config, device)
        pair = {"learning_rate": float(learning_rate), "weight_decay": float(weight_decay)}
        identity = {
            "schema_version": SCHEMA_VERSION,
            "scope": TUNING_SCOPE if phase == "tune" else SCOPE,
            "phase": phase,
            "model": model,
            "registered_name": model,
            "seed": seed,
            "readout": readout,
            "readout_radius": radius,
            "config_sha256": canonical_json_hash(config),
            "registry_sha256": registry["registry_sha256"],
            "data_sha256": data_sha256,
            "train_data_sha256": data_sha256["train"],
            "val_data_sha256": data_sha256["val"],
            "selection_sha256": selection_sha256,
            **pair,
            "optimizer_hyperparameters": pair,
            "steps_per_epoch": steps_per_epoch,
            "epochs_planned": epochs,
            "optimizer_steps_planned": optimizer_steps,
            "device_type": device.type,
            "torch_version": str(torch.__version__),
            "precision_policy": precision,
            "fit_implementation_sha256": canonical_json_hash([
                Path(__file__).read_text(encoding="utf-8"),
                Path(motor.__file__).read_text(encoding="utf-8"),
            ]),
        }
        state_path = directory / "run_state.json"
        if state_path.is_file():
            state = _read_json(state_path)
            _require_identity(state, identity)
            if state.get("status") == "complete":
                _require_completed(directory, state)
                return state
        core._seed_everything(seed, config["experiment"]["deterministic_torch"])
        observer = motor.build_readout_model(model, config, registry, radius).to(device)
        optimizer = torch.optim.AdamW(observer.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return _fit_epochs(
            directory, identity, observer, optimizer, normalizer, train_windows, val_windows,
            settings, device,
        )


def _fit_epochs(directory, identity, observer, optimizer, normalizer, train_windows, val_windows, settings, device):
    state_path = directory / "run_state.json"
    latest, best = directory / "latest.pt", directory / "best.pt"
    epoch_dir, ledger_path = directory / "epoch_checkpoints", directory / "checkpoint_ledger.json"
    ledger_args = {
        "checkpoint_directory": epoch_dir, "ledger_path": ledger_path,
        "latest_path": latest, "best_path": best, "expected_identity": identity,
    }
    newest = core._repair_incomplete_epoch_transaction(**ledger_args)
    history, previous = [], None
    if newest is not None:
        core.load_graph_safe_state_dict(observer, newest["model_state_dict"], strict=True)
        optimizer.load_state_dict(newest["optimizer_state_dict"])
        core._restore_rng(newest["rng_state"])
        history = list(newest["history"])
        previous = sha256_file(epoch_dir / f"epoch_{len(history) - 1:04d}.pt")
    epochs = identity["epochs_planned"]
    steps_per_epoch = identity["steps_per_epoch"]
    if len(history) > epochs:
        raise ValueError("Checkpoint sequence exceeds the requested budget")
    state = {**identity, "status": "running"}
    epoch_args = {
        "batch_size": int(settings["batch_size"]), "device": device,
        "gradient_clip": settings["gradient_clip_norm"],
    }
    for epoch in range(len(history), epochs):
        started = time.perf_counter()
        train_loss, steps = core._run_epoch(
            observer, train_windows, optimizer=optimizer,
            order_seed=identity["seed"] * 1_000_003 + epoch, **epoch_args,
        )
        val_loss, _ = core._run_epoch(observer, val_windows, optimizer=None, order_seed=0, **epoch_args)
        if steps != steps_per_epoch or not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise RuntimeError("Observer epoch requires finite losses and the configured optimizer budget")
        count = (epoch + 1) * steps_per_epoch
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "optimizer_steps_completed": count, "epoch_wall_seconds": time.perf_counter() - started,
        })
        best_epoch = min(range(len(history)), key=lambda i: (history[i]["val_loss"], i))
        payload = {
            **identity, "epoch": epoch, "history": history, "history_sha256": canonical_json_hash(history),
            "previous_epoch_checkpoint_sha256": previous, "selection_rule": SELECTION_RULE,
            "best_epoch": best_epoch, "best_val_loss": history[best_epoch]["val_loss"],
            "model_state_dict": observer.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": core._rng_state(), "normalizer": normalizer.to_dict(),
            "optimizer_steps_completed": count,
        }
        epoch_file = epoch_dir / f"epoch_{epoch:04d}.pt"
        core._atomic_torch_save_new(epoch_file, payload)
        previous = sha256_file(epoch_file)
        atomic_write_json(state_path, {
            **state, "epochs_completed": len(history), "optimizer_steps_completed": count,
            "best_epoch": best_epoch, "best_val_loss": payload["best_val_loss"],
        })

    newest = core._repair_incomplete_epoch_transaction(**ledger_args)
    ledger = _read_json(ledger_path)
    if newest is None or newest["optimizer_steps_completed"] != identity["optimizer_steps_planned"] or ledger["checkpoint_count"] != epochs:
        raise RuntimeError("The fixed-budget fit cannot finalize an incomplete checkpoint sequence")
    best_epoch = min(range(epochs), key=lambda i: (ledger["checkpoints"][i]["validation_loss"], i))
    selected = ledger["checkpoints"][best_epoch]
    picked = torch.load(selected["checkpoint_path"], map_location="cpu", weights_only=False)
    picked.update({
        "selection_finalized_after_training_run": True,
        "training_run_optimizer_steps_completed": newest["optimizer_steps_completed"],
        "training_run_optimizer_steps_planned": identity["optimizer_steps_planned"],
        "training_run_epochs_completed": epochs,
        "training_run_fixed_budget_gate_passed": True,
        "training_run_stopped_early": False,
        "selected_epoch_checkpoint_sha256": selected["checkpoint_sha256"],
        "checkpoint_ledger_file_sha256": sha256_file(ledger_path),
    })
    core._atomic_torch_save(best, picked)
    completed = {
        **identity, "status": "complete", "epochs_completed": epochs,
        "optimizer_steps_completed": newest["optimizer_steps_completed"],
        "best_epoch": best_epoch, "best_val_loss": selected["validation_loss"],
        "fixed_budget_gate_passed": True, "best_checkpoint_sha256": sha256_file(best),
        "latest_checkpoint_sha256": sha256_file(latest),
        "checkpoint_ledger_file_sha256": picked["checkpoint_ledger_file_sha256"],
        "selected_epoch_checkpoint_sha256": selected["checkpoint_sha256"],
    }
    atomic_write_json(state_path, completed)
    return completed


def load_observer(
    checkpoint_path: str | Path, *, config: dict, registry_path: str | Path,
    model: str, seed: int, readout: str = "unit", device: str = "cpu",
) -> tuple[torch.nn.Module, dict]:
    """Rebuild a frozen observer from a completed portable final fit."""
    _require_reproduction(config)
    checkpoint = Path(checkpoint_path)
    if checkpoint.name != "best.pt":
        raise ValueError("Evaluation requires the validation-selected best.pt")
    registry = load_parameter_registry(registry_path)
    state = _read_json(checkpoint.with_name("run_state.json"))
    identity = {
        "schema_version": SCHEMA_VERSION, "scope": SCOPE, "phase": "train",
        "model": model, "seed": seed, "readout": readout, "config_sha256": canonical_json_hash(config),
        "registry_sha256": registry["registry_sha256"],
    }
    _require_identity(state, identity)
    _require_completed(checkpoint.parent, state)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require_identity(metadata, {
        **identity, "epoch": state["best_epoch"], "readout_radius": state["readout_radius"],
        "data_sha256": state["data_sha256"], "selection_sha256": state["selection_sha256"],
        "training_run_optimizer_steps_completed": state["optimizer_steps_completed"],
        "training_run_epochs_completed": state["epochs_completed"],
        "training_run_fixed_budget_gate_passed": True, "selection_finalized_after_training_run": True,
        "checkpoint_ledger_file_sha256": state["checkpoint_ledger_file_sha256"],
        "selected_epoch_checkpoint_sha256": state["selected_epoch_checkpoint_sha256"],
    })
    radius = metadata["readout_radius"]
    if (readout == "scaled") != (radius is not None):
        raise ValueError("The checkpoint readout and training-derived radius disagree")
    observer = motor.build_readout_model(model, config, registry, radius)
    core.load_graph_safe_state_dict(observer, metadata["model_state_dict"], strict=True)
    wrapped = core.NormalizedObserver(observer, core.Normalizer.from_dict(metadata["normalizer"]))
    return wrapped.to(device).eval(), metadata


def _load_selection(path: Path, config: dict, registry: dict, readout: str) -> dict:
    selection = _read_json(path)
    hash_field = "tuning_selection_sha256"
    selection_core = {key: value for key, value in selection.items() if key != hash_field}
    if canonical_json_hash(selection_core) != selection.get(hash_field):
        raise ValueError("The tuning selection hash differs")
    _require_identity(selection, {
        "schema": TUNING_SCHEMA, "scope": TUNING_SCOPE, "complete": True, "phase": "tune",
        "seed": config["tuning"]["seed"], "readout": readout,
        "config_sha256": canonical_json_hash(config), "registry_sha256": registry["registry_sha256"],
    })
    models = config["matching"]["registered_models"] if readout == "unit" else motor.GRAPH_CFC_MODELS
    if set(selection["grid"]["models"]) != set(models) or set(selection["selected"]) != set(models):
        raise ValueError("The tuning selection must cover every observer for the readout")
    if set(selection["data_sha256"]) != {"train", "val"}:
        raise ValueError("The tuning selection requires train and validation dataset hashes")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readout", choices=("unit", "scaled"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", action="append", help="Repeat to choose particular observers")
    parser.add_argument("--seed", type=int, action="append", help="Repeat to choose particular final-fit seeds")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config, _ = load_config(args.config)
    _require_reproduction(config)
    registry = load_parameter_registry(args.registry)
    selection = _load_selection(args.selection, config, registry, args.readout)
    models = args.model or list(selection["grid"]["models"])
    seeds = args.seed or config["training"]["seeds"]
    for model in models:
        pair = selection["selected"][model]
        for seed in seeds:
            output = args.output_dir / args.readout / model / f"seed_{seed}"
            fit_observer(
                config=config, data_root=args.data_root, registry_path=args.registry, output_dir=output,
                model=model, seed=seed, learning_rate=pair["learning_rate"], weight_decay=pair["weight_decay"],
                phase="train", readout=args.readout, device=args.device,
                selection_sha256=selection["tuning_selection_sha256"], expected_data_sha256=selection["data_sha256"],
            )
            print(output / "best.pt", flush=True)


if __name__ == "__main__":
    main()
