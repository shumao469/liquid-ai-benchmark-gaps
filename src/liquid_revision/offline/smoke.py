"""Fast, non-training smoke gate for the corrected offline benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from liquid_revision.config import load_config
from liquid_revision.io_utils import atomic_write_json
from liquid_revision.repro import seed_everything

from .corruption import CorruptionSpec, apply_corruption
from .models import build_offline_model, resolve_hidden_dim
from .training import TASKS, compose_model_features, load_all_datasets, utc_now


def _subset(arrays: Any, limit: int) -> Any:
    indices = np.arange(min(limit, len(arrays.labels)))
    return arrays.subset(indices)


def _assert_disjoint(data: dict[str, Any]) -> None:
    train = {str(item) for item in data["train"].ids}
    validation = {str(item) for item in data["val"].ids}
    test = {str(item) for item in data.get("test", ()).ids} if "test" in data else set()
    if train & validation or train & test or validation & test:
        raise AssertionError("offline split IDs overlap")


def run_offline_smoke(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, config_hash = load_config(config_path)
    smoke_config = dict(config.get("smoke", {}))
    experiment = dict(config.get("experiment", {}))
    model_config = dict(config.get("models", {}))
    feature_channels = list(
        dict(config.get("features", {})).get(
            "channels", ["values", "mask", "age", "dt"]
        )
    )
    max_examples = int(smoke_config.get("max_examples_per_split", 4))
    seed = int(smoke_config.get("seed", 909))
    seed_everything(seed, deterministic_torch=bool(experiment.get("deterministic_torch", True)))
    requested_device = str(experiment.get("device", "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("offline smoke requested CUDA but CUDA is unavailable")
    device = torch.device(requested_device)
    # Smoke checks are validation-only.  Test arrays remain unavailable until a
    # complete checkpoint release has passed the formal two-stage lock.
    datasets = load_all_datasets(config, config_path, include_test=False)
    report: dict[str, Any] = {
        "status": "running",
        "training_performed": False,
        "config": str(config_path),
        "config_sha256": config_hash,
        "started_at_utc": utc_now(),
        "device": str(device),
        "datasets": {},
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "offline_smoke.json"
    atomic_write_json(report_path, report)

    for dataset_name in [str(item) for item in experiment.get("datasets", datasets.keys())]:
        data = datasets[dataset_name]
        _assert_disjoint(data)
        task = TASKS[dataset_name]
        train = _subset(data["train"], max_examples)
        validation = _subset(data["val"], max_examples)
        features = compose_model_features(
            train.values,
            train.mask,
            train.age,
            train.dt,
            channels=feature_channels,
        )
        values_tensor = torch.from_numpy(features).to(device)
        dt_tensor = torch.from_numpy(np.asarray(train.dt, dtype=np.float32)).to(device)
        lengths_tensor = torch.from_numpy(np.asarray(train.lengths, dtype=np.int64)).to(device)
        dataset_report: dict[str, Any] = {
            "prepared_data_verification": data.get("prepared_data_verification"),
            "split_manifest_sha256": data.get("manifest_sha256"),
            "normalization_fit_split": "train_only",
            "split_sizes": {
                split: int(len(data[split].labels)) for split in ("train", "val")
            },
            "split_ids_disjoint": True,
            "input_shape": list(features.shape),
            "models": {},
        }
        for model_name in [str(item) for item in model_config.get(
            "names", ["lstm", "ode-rnn", "ltc", "cfc"]
        )]:
            hidden_dim = resolve_hidden_dim(model_config, dataset_name, model_name)
            model = build_offline_model(
                model_name,
                input_dim=features.shape[-1],
                hidden_dim=hidden_dim,
                output_dim=task.output_dim,
                ode_max_step=float(model_config.get("ode_max_step", 1.0)),
                ltc_ode_unfolds=int(model_config.get("ltc_ode_unfolds", 3)),
                cfc_mode=str(model_config.get("cfc_mode", "default")),
            ).to(device)
            logits = model(values_tensor, dt_tensor, lengths_tensor)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"{dataset_name}/{model_name} produced nonfinite logits")
            logits.square().mean().backward()
            finite_gradients = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            if not finite_gradients:
                raise FloatingPointError(f"{dataset_name}/{model_name} produced nonfinite gradients")
            changed_with_dt: bool | None = None
            if model.uses_elapsed_time:
                model.eval()
                with torch.inference_mode():
                    alternate = model(values_tensor, dt_tensor * 1.7, lengths_tensor)
                changed_with_dt = not torch.allclose(
                    logits.detach(), alternate, atol=1e-7, rtol=1e-6
                )
                if not changed_with_dt:
                    raise AssertionError(
                        f"{dataset_name}/{model_name} did not respond to changed elapsed time"
                    )
            dataset_report["models"][model_name] = {
                "logit_shape": list(logits.shape),
                "finite_forward": True,
                "finite_gradients": True,
                "uses_elapsed_time": model.uses_elapsed_time,
                "changed_with_elapsed_time": changed_with_dt,
                "trainable_parameters": model.trainable_parameters,
            }

        corrupted = apply_corruption(
            validation.values,
            validation.mask,
            validation.dt,
            CorruptionSpec(
                "mcar", 0.3, seed=seed + (0 if dataset_name == "har" else 1),
                packetwise=task.packetwise_corruption,
            ),
        )
        if np.any(corrupted.mask & ~validation.mask):
            raise AssertionError("smoke corruption created observations")
        dataset_report["corruption"] = corrupted.metadata
        dataset_report["corruption_split"] = "validation"
        dataset_report["test_accessed"] = False
        report["datasets"][dataset_name] = dataset_report
        atomic_write_json(report_path, report)

    report["status"] = "passed"
    report["completed_at_utc"] = utc_now()
    atomic_write_json(report_path, report)
    return report

