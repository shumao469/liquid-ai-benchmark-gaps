"""Determinism controls and narrowly scoped environment capture."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, sha256_file, source_tree_manifest


CAPTURE_PACKAGES = (
    "numpy",
    "PyYAML",
    "torch",
    "ncps",
    "gymnasium",
    "mujoco",
    "pytest",
)


def seed_everything(seed: int, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch without silently weakening determinism."""
    seed = int(seed)
    random.seed(seed)
    result: dict[str, Any] = {"seed": seed, "python": True}

    try:
        import numpy as np

        np.random.seed(seed)
        result["numpy"] = True
    except ImportError:
        result["numpy"] = False

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        result.update(
            {
                "torch": True,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "deterministic_torch": bool(deterministic_torch),
            }
        )
    except ImportError:
        result["torch"] = False
    return result


def seed_worker(worker_id: int) -> None:
    """PyTorch DataLoader worker initializer compatible with seeded generators."""
    del worker_id
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("seed_worker requires NumPy and PyTorch") from exc
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("make_torch_generator requires PyTorch") from exc
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _run_text(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in CAPTURE_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _torch_runtime() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": [props.major, props.minor],
                }
            )
    return {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "devices": devices,
    }


def _git_state(repo: Path) -> dict[str, Any]:
    return {
        "root": str(repo.resolve()),
        "commit": _run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "branch": _run_text(["git", "branch", "--show-current"], cwd=repo),
        "status_porcelain": _run_text(["git", "status", "--porcelain"], cwd=repo),
    }


def capture_environment(
    output_path: str | Path,
    *,
    config_paths: list[str | Path] | None = None,
    repo_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configs = []
    for item in config_paths or []:
        path = Path(item).resolve()
        configs.append({"path": str(path), "sha256": sha256_file(path)})

    nvidia_query = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    payload: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "packages": _package_versions(),
        "torch_runtime": _torch_runtime(),
        "nvidia_smi": nvidia_query.splitlines() if nvidia_query else None,
        "configs": configs,
        "source_tree": source_tree_manifest(Path(__file__).resolve().parent),
    }
    if repo_path is not None:
        payload["git"] = _git_state(Path(repo_path).resolve())
    if extra:
        payload["extra"] = extra
    atomic_write_json(output_path, payload)
    return payload

