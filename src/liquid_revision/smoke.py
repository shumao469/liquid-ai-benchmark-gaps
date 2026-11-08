"""End-to-end foundation smoke test; intentionally performs no optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .data.closed_loop import generate_closed_loop_dataset, load_closed_loop_dataset
from .evaluation import evaluate_observer_episode
from .io_utils import atomic_write_json
from .repro import capture_environment, seed_everything


TIME_AWARE_MODELS = {"ctrnn", "odernn", "ltc", "cfc"}


def _batched_episode_prefixes(
    arrays: dict[str, np.ndarray], batch_size: int, sequence_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    offsets = arrays["episode_offsets"]
    inputs = []
    dts = []
    available_episodes = len(offsets) - 1
    if batch_size > available_episodes:
        raise ValueError(
            f"Smoke batch_size={batch_size} exceeds generated episodes={available_episodes}"
        )
    for episode in range(batch_size):
        start, stop = int(offsets[episode]), int(offsets[episode + 1])
        if stop - start < sequence_steps:
            raise ValueError(f"Episode {episode} is shorter than smoke sequence")
        inputs.append(arrays["observer_input"][start : start + sequence_steps])
        dts.append(
            arrays["observed_timestamp_dt"][start : start + sequence_steps, 0]
        )
    return np.stack(inputs), np.stack(dts)


def run_smoke(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    import torch

    from .models import build_model

    config, config_hash = load_config(config_path)
    experiment = dict(config.get("experiment", {}))
    seed_report = seed_everything(
        int(experiment.get("seed", 0)),
        bool(experiment.get("deterministic_torch", True)),
    )

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        dataset_manifest = generate_closed_loop_dataset(config, output / "dataset")
        arrays, _ = load_closed_loop_dataset(output / "dataset")
        smoke_config = dict(config.get("smoke", {}))
        batch_size = int(smoke_config.get("batch_size", 2))
        sequence_steps = int(smoke_config.get("sequence_steps", 8))
        input_array, dt_array = _batched_episode_prefixes(
            arrays, batch_size, sequence_steps
        )
        x = torch.from_numpy(input_array)
        dt = torch.from_numpy(dt_array)

        model_config = dict(config.get("models", {}))
        model_reports = []
        for index, name in enumerate(model_config.get("names", [])):
            torch.manual_seed(int(experiment.get("seed", 0)) + 100 * index)
            model = build_model(str(name), model_config)
            model.eval()
            with torch.no_grad():
                action, _ = model.forward_sequence(x, dt)
                changed_action, _ = model.forward_sequence(x, dt * 1.50)
            finite = bool(torch.isfinite(action).all().item())
            if action.shape != (batch_size, sequence_steps, int(model_config["state_dim"])):
                raise AssertionError(f"Unexpected {name} output shape: {tuple(action.shape)}")
            if not finite:
                raise AssertionError(f"Non-finite output from {name}")
            dt_delta = float(torch.max(torch.abs(action - changed_action)).item())
            if str(name).lower() in TIME_AWARE_MODELS and dt_delta <= 1e-8:
                raise AssertionError(
                    f"{name} produced identical outputs when all elapsed times changed; "
                    "the time path may be disconnected"
                )
            report = {
                "name": str(name),
                "trainable_parameters": model.trainable_parameters,
                "output_shape": list(action.shape),
                "finite": finite,
                "max_state_estimate_abs": float(torch.max(torch.abs(action)).item()),
                "dt_sensitivity_max_abs": dt_delta,
            }
            report["closed_loop"] = evaluate_observer_episode(
                model,
                config,
                episode_seed=int(smoke_config.get("closed_loop_episode_seed", 81000)),
                schedule_seed=int(smoke_config.get("closed_loop_schedule_seed", 91000)),
                max_steps=int(smoke_config.get("closed_loop_steps", 12)),
            )
            if hasattr(model, "_timespan_rank"):
                report["ncps_timespan_rank"] = getattr(model, "_timespan_rank")
                report["ncps_timespan_mode"] = getattr(model, "_timespan_mode")
            model_reports.append(report)

        environment_report = capture_environment(
            output / "environment.json",
            config_paths=[config_path],
            repo_path=Path(__file__).resolve().parents[3],
            extra={"smoke_seed_report": seed_report},
        )
        result = {
            "status": "PASS",
            "training_performed": False,
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": config_hash,
            "dataset": {
                "transitions": dataset_manifest["transitions"],
                "episodes": dataset_manifest["episodes"],
                "sha256": dataset_manifest["dataset_sha256"],
            },
            "models": model_reports,
            "torch_runtime": environment_report.get("torch_runtime"),
        }
        atomic_write_json(output / "smoke_report.json", result)
        return result
    except Exception:
        # Preserve the output directory and any diagnostics for forensic inspection.
        raise

