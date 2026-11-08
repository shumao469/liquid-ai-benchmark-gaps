"""Immutable pre-formal artifacts and hard gates for the closed-loop study."""

from __future__ import annotations

import importlib.metadata
import copy
import csv
import hashlib
import hmac
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..io_utils import (
    atomic_write_json,
    canonical_json_hash,
    sha256_file,
    source_tree_manifest,
)
from ..models.liquid import load_graph_safe_state_dict
from ..envs.reacher import MovingTargetReacherEnv
from .parameter_matching import (
    REGISTERED_MODELS,
    _closed_loop_implementation_manifest,
    build_registered_model,
    load_parameter_registry,
)
from .manifests import build_trajectory_manifest, load_trajectory_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAL_PACKAGES = (
    "numpy",
    "PyYAML",
    "torch",
    "ncps",
    "gymnasium",
    "mujoco",
)
TRUSTED_NVIDIA_SMI_PATH = Path("C:/Windows/System32/nvidia-smi.exe")
GPU_UUID_PATTERN = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
GPU_PCI_BUS_ID_PATTERN = re.compile(
    r"^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$"
)
MAX_TUNING_IDLE_GPU_UTILIZATION_PERCENT = 5.0
TUNING_IDLE_SAMPLE_COUNT = 3
SHARED_GPU_POLICY_PATH = PROJECT_ROOT / "control/shared_gpu_training_policy_20261105.json"
SHARED_GPU_MEMORY_QUERY = "index,uuid,memory.total,memory.used,memory.free,utilization.gpu,compute_mode"
SHARED_RUNTIME_REVIEW_PATH = (
    PROJECT_ROOT / "private-runtime-evidence" / "shared_runtime_compatibility_review.json"
)


def _shared_runtime_compatibility_anchor(data_root: str | Path) -> dict[str, Any]:
    complete = json.loads((_resolve(data_root) / "COMPLETE.json").read_text(encoding="utf-8"))
    anchor = complete.get("runtime_compatibility_review")
    if (not isinstance(anchor, dict) or set(anchor) != {"path", "file_sha256"}
            or _resolve(anchor["path"]) != SHARED_RUNTIME_REVIEW_PATH.resolve()
            or sha256_file(anchor["path"]) != anchor["file_sha256"]):
        raise PermissionError("Shared namespace lacks the exact data-reuse review anchor")
    review = json.loads(_resolve(anchor["path"]).read_text(encoding="utf-8"))
    if review.get("status") != "PASS":
        raise PermissionError("Shared data-reuse review is not PASS")
    return anchor


def _load_shared_gpu_policy_anchor(*, required: bool = False) -> dict[str, Any] | None:
    """Load an explicit runtime policy, never reinterpret it as a quiet-GPU claim."""
    path = SHARED_GPU_POLICY_PATH.resolve()
    if not path.is_file():
        if required:
            raise PermissionError("Shared GPU policy is absent")
        return None
    policy = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = {"schema_version": 1, "mode": "shared_gpu_memory_admission",
        "minimum_gpu_free_mib": 12288, "minimum_host_free_mib": 8192,
        "sample_count": 3, "allow_existing_gpu_processes": True,
        "utilization_is_admission_threshold": False,
        "automatic_external_process_termination": False}
    if not isinstance(policy, dict) or any(type(policy.get(k)) is not type(v)
            or policy.get(k) != v for k, v in expected.items()):
        raise PermissionError("Shared GPU policy does not match the explicit registered resource boundary")
    if not GPU_UUID_PATTERN.fullmatch(str(policy.get("gpu_uuid", ""))):
        raise PermissionError("Shared GPU policy lacks a complete GPU UUID")
    return {"path": str(path), "policy_sha256": canonical_json_hash(policy),
            "policy_file_sha256": sha256_file(path), "policy": policy}


def _parse_shared_gpu_resource_snapshot(snapshot: dict[str, Any], gpu_uuid: str) -> dict[str, Any]:
    """Parse real aggregate VRAM, retaining reserved/rounding gaps explicitly."""
    rows = [[value.strip() for value in row] for row in csv.reader(
        io.StringIO(str(snapshot.get("stdout", ""))), skipinitialspace=True) if row]
    parsed = []
    for row in rows:
        if len(row) != 7 or not row[0].isdigit() or not GPU_UUID_PATTERN.fullmatch(row[1]):
            raise PermissionError("Malformed shared GPU resource identity")
        if any(not value.isdigit() for value in row[2:5]):
            raise PermissionError("Shared GPU memory must be finite nonnegative integer MiB")
        total, used, free = (int(value) for value in row[2:5])
        try:
            utilization = float(row[5])
        except ValueError as error:
            raise PermissionError("Malformed shared GPU utilization") from error
        if (total <= 0 or used > total or free > total or used + free > total + 2
                or not math.isfinite(utilization) or not 0 <= utilization <= 100
                or row[6] != "Default"):
            raise PermissionError("Inconsistent shared GPU memory/utilization/mode")
        parsed.append({"physical_index": int(row[0]), "gpu_uuid": row[1],
            "memory_total_mib": total, "memory_used_mib": used, "memory_free_mib": free,
            "reserved_or_rounding_gap_mib": total - used - free,
            "utilization_percent": utilization, "compute_mode": row[6]})
    if (len({row["physical_index"] for row in parsed}) != len(parsed)
            or len({row["gpu_uuid"].lower() for row in parsed}) != len(parsed)):
        raise PermissionError("Duplicate shared GPU identity")
    selected = [row for row in parsed if row["gpu_uuid"].lower() == gpu_uuid.lower()]
    if len(selected) != 1 or selected[0]["physical_index"] != 0:
        raise PermissionError("Shared GPU resource snapshot does not identify exact physical GPU0")
    return selected[0]
# Frozen independently here: replay does not import or trust the launcher's
# classification policy or a classification string carried in its evidence.
TUNING_WDDM_ALLOWED_IMAGES = frozenset(
    {
        "applicationframehost.exe",
        "awesun.exe",
        "chatgpt.exe",
        "chrome.exe",
        "claude.exe",
        "codex.exe",
        "codex-computer-use-swift.exe",
        "dwm.exe",
        "excel.exe",
        "explorer.exe",
        "lockapp.exe",
        "msedge.exe",
        "msedgewebview2.exe",
        "mstsc.exe",
        "notepad3.exe",
        "nvcontainer.exe",
        "nvidia overlay.exe",
        "outlook.exe",
        "powerpnt.exe",
        "rdpclip.exe",
        "searchhost.exe",
        "shellexperiencehost.exe",
        "shellhost.exe",
        "snippingtool.exe",
        "startmenuexperiencehost.exe",
        "taskmgr.exe",
        "teams.exe",
        "textinputhost.exe",
        "wechatappex.exe",
        "weixin.exe",
        "windowsterminal.exe",
        "winword.exe",
    }
)
TUNING_EXPLICIT_GPU_CONFLICT_IMAGES = frozenset(
    {
        "main.exe",
        "promecefpluginhost.exe",
        "python.exe",
        "pythonw.exe",
        "torch.exe",
        "torchrun.exe",
    }
)


def trusted_nvidia_smi_identity(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and rehash the one trusted Windows NVSMI executable.

    Formal configurations must bind both its normalized absolute path and byte
    hash.  Non-formal CPU-only development remains importable on hosts without
    NVIDIA tooling, but may not claim a trusted GPU inventory.
    """
    configured = dict((config or {}).get("trusted_nvidia_smi", {}))
    path = TRUSTED_NVIDIA_SMI_PATH.resolve()
    configured_path = configured.get("executable_path")
    configured_sha256 = str(configured.get("executable_sha256", "")).lower()
    formal_required = bool(config and is_formal(config))
    if configured_path is not None and Path(str(configured_path)).resolve() != path:
        raise ValueError(
            "trusted_nvidia_smi.executable_path must be the exact Windows System32 binary"
        )
    if formal_required and (
        configured_path is None or len(configured_sha256) != 64
    ):
        raise ValueError(
            "Formal config must bind trusted_nvidia_smi path and SHA-256"
        )
    if not path.is_file():
        if formal_required:
            raise FileNotFoundError(f"Trusted nvidia-smi is absent: {path}")
        return {
            "executable_path": str(path),
            "executable_sha256": None,
        }
    live_sha256 = sha256_file(path)
    if configured_sha256 and live_sha256.lower() != configured_sha256:
        raise ValueError("Trusted nvidia-smi byte SHA-256 differs from config")
    return {
        "executable_path": str(path),
        "executable_sha256": live_sha256,
    }


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return (PROJECT_ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def is_formal(config: dict[str, Any]) -> bool:
    formal = dict(config.get("formal", {}))
    return bool(formal.get("required", False))


def is_development_only(config: dict[str, Any]) -> bool:
    return bool(config.get("experiment", {}).get("development_only", False))


def _run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in FORMAL_PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _physical_gpu_inventory(
    trusted_nvidia_smi: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return the host GPU inventory independently of CUDA_VISIBLE_DEVICES.

    ``nvidia-smi`` enumerates the physical host devices, whereas the CUDA
    runtime enumerates only the process-visible subset.  The former belongs in
    the cross-run release identity; the latter is recorded per run.
    """
    trusted = (
        trusted_nvidia_smi_identity()
        if trusted_nvidia_smi is None
        else dict(trusted_nvidia_smi)
    )
    if trusted.get("executable_sha256") is None:
        return []
    executable_path = Path(str(trusted["executable_path"])).resolve()
    if (
        str(executable_path).casefold()
        != str(TRUSTED_NVIDIA_SMI_PATH.resolve()).casefold()
        or sha256_file(executable_path).lower()
        != str(trusted["executable_sha256"]).lower()
    ):
        raise PermissionError(
            "Trusted nvidia-smi path/hash drifted before inventory capture"
        )
    raw = _run_text(
        [
            str(executable_path),
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        raise RuntimeError(
            "Trusted nvidia-smi returned an empty physical GPU inventory"
        )
    fields = (
        "physical_index",
        "uuid",
        "pci_bus_id",
        "name",
        "driver_version",
    )
    rows: list[dict[str, Any]] = []
    for values in csv.reader(io.StringIO(raw)):
        if len(values) != len(fields):
            raise ValueError(f"Unexpected nvidia-smi inventory row: {values!r}")
        rows.append({key: value.strip() for key, value in zip(fields, values)})
    return _validate_physical_gpu_inventory_rows(rows)


def _validate_physical_gpu_inventory_rows(
    rows: Any,
) -> list[dict[str, str]]:
    """Validate and canonicalize a complete physical NVSMI inventory.

    Every row is semantic evidence.  A malformed non-target GPU must therefore
    fail the inventory just as a malformed GPU0 row does; silently filtering it
    would permit a forged or truncated host identity.
    """
    required = {
        "physical_index",
        "uuid",
        "pci_bus_id",
        "name",
        "driver_version",
    }
    if not isinstance(rows, list):
        raise ValueError("Physical GPU inventory must be a list")
    if not rows:
        raise ValueError("Physical GPU inventory must be non-empty")
    canonical: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("Physical GPU inventory row schema is invalid")
        index_text = str(row["physical_index"]).strip()
        uuid_text = str(row["uuid"]).strip()
        pci_text = str(row["pci_bus_id"]).strip()
        name = str(row["name"]).strip()
        driver_version = str(row["driver_version"]).strip()
        if not re.fullmatch(r"0|[1-9][0-9]*", index_text):
            raise ValueError("Physical GPU index must be canonical non-negative decimal")
        if not GPU_UUID_PATTERN.fullmatch(uuid_text):
            raise ValueError("Physical GPU UUID is not canonical")
        if not GPU_PCI_BUS_ID_PATTERN.fullmatch(pci_text):
            raise ValueError("Physical GPU PCI bus ID is not canonical")
        if not name:
            raise ValueError("Physical GPU name must be non-empty")
        if not driver_version:
            raise ValueError("Physical GPU driver version must be non-empty")
        canonical.append(
            {
                "physical_index": str(int(index_text)),
                "uuid": "GPU-" + uuid_text[4:].lower(),
                "pci_bus_id": pci_text.lower(),
                "name": name,
                "driver_version": driver_version,
            }
        )
    canonical.sort(key=lambda row: int(row["physical_index"]))
    indices = [int(row["physical_index"]) for row in canonical]
    if indices != list(range(len(canonical))):
        raise ValueError(
            "Physical GPU indices must be unique and contiguous from zero"
        )
    uuids = [row["uuid"] for row in canonical]
    if len(set(uuids)) != len(uuids):
        raise ValueError("Physical GPU UUIDs must be unique")
    pci_bus_ids = [row["pci_bus_id"] for row in canonical]
    if len(set(pci_bus_ids)) != len(pci_bus_ids):
        raise ValueError("Physical GPU PCI bus IDs must be unique")
    return canonical


def _torch_precision_policy(torch: Any) -> dict[str, Any]:
    return {
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "policy_source": "registered_config_plus_runtime_numeric_policy",
    }


def apply_and_validate_precision_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Apply registered deterministic flags and verify the frozen numeric policy."""
    import torch

    deterministic = bool(
        config.get("experiment", {}).get("deterministic_torch", True)
    )
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False if deterministic else bool(
            torch.backends.cudnn.benchmark
        )
    current = _torch_precision_policy(torch)
    if is_formal(config):
        path = config.get("formal", {}).get("environment_artifact")
        if not path:
            raise ValueError("Formal config lacks formal.environment_artifact")
        frozen = _load_hashed(path, "environment_sha256")
        expected = frozen.get("torch_runtime", {}).get("precision_policy")
        if not isinstance(expected, dict):
            raise ValueError("Environment freeze lacks a precision policy")
        for key, value in current.items():
            if expected.get(key) != value:
                raise ValueError(
                    f"Runtime precision policy differs from environment freeze for {key}"
                )
    return current


def current_run_accelerator_identity(
    device: str | Any,
    *,
    require_physical_mapping: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record the selected accelerator and CVD mapping without freezing it."""
    try:
        import torch
    except ImportError:
        return {"requested_device": str(device), "torch_available": False}

    resolved = torch.device(device)
    result: dict[str, Any] = {
        "requested_device": str(device),
        "resolved_device": str(resolved),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "visible_cuda_device_count": int(torch.cuda.device_count()),
        "precision_policy": _torch_precision_policy(torch),
    }
    if resolved.type != "cuda":
        if require_physical_mapping:
            raise RuntimeError(
                "Formal closed-loop training/evaluation requires a CUDA device "
                "with a resolved physical UUID/PCI identity"
            )
        result["selected_physical_gpu"] = None
        return result

    visible_index = int(resolved.index or 0)
    result.update(
        {
            "selected_visible_index": visible_index,
            "torch_device_name": torch.cuda.get_device_name(visible_index),
            "compute_capability": list(torch.cuda.get_device_capability(visible_index)),
        }
    )
    trusted = trusted_nvidia_smi_identity(config)
    inventory = _physical_gpu_inventory(trusted)
    result["trusted_nvidia_smi"] = trusted
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    token: str | None = None
    if cvd is not None:
        tokens = [item.strip() for item in cvd.split(",") if item.strip()]
        if visible_index < len(tokens):
            token = tokens[visible_index]
    selected: dict[str, str] | None = None
    if require_physical_mapping and (token is None or token.isdigit()):
        raise RuntimeError(
            "Formal CUDA runs require CUDA_VISIBLE_DEVICES to select the GPU "
            "by UUID (GPU-...), because numeric CUDA ordinals cannot be safely "
            "equated with nvidia-smi/NVML physical indices"
        )
    if token is None:
        selected = next(
            (row for row in inventory if row["physical_index"] == str(visible_index)),
            None,
        )
    elif token.isdigit():
        selected = next(
            (row for row in inventory if row["physical_index"] == token), None
        )
    else:
        normalized = token.removeprefix("GPU-").lower()
        exact_matches = [
            row
            for row in inventory
            if row["uuid"].lower() == token.lower()
        ]
        prefix_matches = [
            row
            for row in inventory
            if row["uuid"].removeprefix("GPU-").lower().startswith(normalized)
        ]
        if len(exact_matches) != 1 and prefix_matches:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES contains an abbreviated GPU UUID; "
                "use the complete UUID"
            )
        selected = exact_matches[0] if len(exact_matches) == 1 else None
    result["cuda_visible_device_token"] = token
    result["selected_physical_gpu"] = selected
    if require_physical_mapping and selected is None:
        raise RuntimeError(
            "Formal CUDA run could not map its visible device to a physical "
            "nvidia-smi UUID/PCI identity"
        )
    return result


def _mujoco_xml_identity(config: dict[str, Any]) -> dict[str, Any]:
    environment = dict(config.get("environment", {}))
    if str(environment.get("kind", "reacher_v5")) != "reacher_v5":
        raise ValueError("Formal closed-loop freeze supports only reacher_v5")
    accepted = {
        key: environment[key]
        for key in (
            "horizon_steps",
            "frame_skip",
            "trajectory_family",
            "amplitude_range",
            "frequency_range_hz",
            "action_limit",
            "velocity_failure_limit",
            "joint_position_failure_limit",
        )
        if key in environment
    }
    env = MovingTargetReacherEnv(**accepted)
    try:
        xml = Path(env.fullpath).resolve()
        return {"path": str(xml), "sha256": sha256_file(xml)}
    finally:
        env.close()


def current_environment_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Return the stable environment/source identity used by every formal stage."""
    try:
        import torch

        registered_deterministic = bool(
            config.get("experiment", {}).get("deterministic_torch", True)
        )
        frozen_precision_policy = _torch_precision_policy(torch)
        frozen_precision_policy.update(
            {
                "cudnn_deterministic": registered_deterministic,
                "cudnn_benchmark": False if registered_deterministic else bool(
                    torch.backends.cudnn.benchmark
                ),
                "deterministic_algorithms_enabled": registered_deterministic,
                "policy_source": "registered_config_plus_runtime_numeric_policy",
            }
        )
        torch_runtime: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "precision_policy": frozen_precision_policy,
        }
    except ImportError:
        torch_runtime = {"available": False}
    documents = {}
    for relative in ("docs/PROTOCOL.md", "docs/ANALYSIS_PLAN.md"):
        path = PROJECT_ROOT / relative
        documents[relative] = sha256_file(path)
    design_config = copy.deepcopy(config)
    design_config.setdefault("controller", {}).pop("joint_kp", None)
    design_config.setdefault("controller", {}).pop("joint_kd", None)
    design_config.setdefault("formal", {}).pop("environment_artifact", None)
    design_config.setdefault("controller_selection", {}).pop("artifact", None)
    design_config.setdefault("tuning", {}).pop("artifact", None)
    trusted_nvidia_smi = trusted_nvidia_smi_identity(config)
    return {
        "schema_version": 3,
        # Selected controller gains are deliberately excluded: the validation
        # grid must be able to replace an initial placeholder before the final
        # release freeze without creating a circular identity dependency.
        "preselection_design_config_sha256": canonical_json_hash(design_config),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "torch_runtime": torch_runtime,
        "trusted_nvidia_smi": trusted_nvidia_smi,
        "physical_gpu_inventory": _physical_gpu_inventory(
            trusted_nvidia_smi
        ),
        "mujoco_xml": _mujoco_xml_identity(config),
        "documents": documents,
        "implementation": _closed_loop_implementation_manifest(),
    }


def create_environment_freeze(
    config: dict[str, Any], output_path: str | Path
) -> dict[str, Any]:
    destination = _resolve(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite formal environment freeze: {destination}")
    core = current_environment_identity(config)
    artifact = {
        **core,
        "environment_sha256": canonical_json_hash(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(destination, artifact)
    return artifact


def _load_hashed(path: str | Path, hash_field: str) -> dict[str, Any]:
    source = _resolve(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key not in {hash_field, "created_at_utc"}
    }
    if canonical_json_hash(core) != payload.get(hash_field):
        raise ValueError(f"Artifact hash mismatch for {source}")
    return payload


def validate_environment_freeze(config: dict[str, Any]) -> dict[str, Any] | None:
    if not is_formal(config):
        return None
    if is_development_only(config):
        raise ValueError("A formal config cannot be marked development_only")
    path = config.get("formal", {}).get("environment_artifact")
    if not path:
        raise ValueError("Formal config lacks formal.environment_artifact")
    artifact = _load_hashed(path, "environment_sha256")
    current = current_environment_identity(config)
    if canonical_json_hash(current) != artifact["environment_sha256"]:
        raise ValueError(
            "Current config/software/CUDA/XML/protocol/source identity differs "
            "from the formal environment freeze"
        )
    return artifact


def _controller_selection_spec(config: dict[str, Any]) -> dict[str, Any]:
    selection = dict(config.get("controller_selection", {}))
    controller = dict(config.get("controller", {}))
    controller.pop("joint_kp", None)
    controller.pop("joint_kd", None)
    manifest = build_trajectory_manifest(config)
    return {
        "schema_version": 1,
        "environment": config.get("environment", {}),
        "controller_without_selected_gains": controller,
        "kp_grid": list(selection.get("kp_grid", [5.0, 10.0, 20.0, 40.0])),
        "kd_grid": list(selection.get("kd_grid", [0.5, 1.0, 2.0, 4.0])),
        "selection_rule": (
            "zero failures preferred; then minimum mean end-effector RMSE; "
            "then minimum mean control effort; then lower Kp; then lower Kd"
        ),
        "validation_rows_sha256": canonical_json_hash(manifest["splits"]["val"]),
        "validation_episode_count": len(manifest["splits"]["val"]),
    }


def create_controller_selection(
    config: dict[str, Any], output_path: str | Path, *, device: str = "cpu"
) -> dict[str, Any]:
    """Run the preregistered 16-grid on validation trajectories only."""
    del device  # full-state baseline is NumPy/MuJoCo and has no learned device
    from .baselines import build_nonlearned_baseline
    from .benchmark import _rollout

    destination = _resolve(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite controller selection: {destination}")
    environment_identity = current_environment_identity(config)
    environment_design_sha256 = canonical_json_hash(environment_identity)
    spec = _controller_selection_spec(config)
    manifest = build_trajectory_manifest(config)
    validation_rows = manifest["splits"]["val"]
    condition = {
        "id": "controller_selection_nominal",
        "family": "controller_selection_nominal",
        "severity": 0.0,
        "spec": {},
    }
    candidates = []
    validation_episode_rows: list[dict[str, Any]] = []
    for kp in spec["kp_grid"]:
        for kd in spec["kd_grid"]:
            candidate_config = json.loads(json.dumps(config))
            candidate_config.setdefault("controller", {})["joint_kp"] = float(kp)
            candidate_config["controller"]["joint_kd"] = float(kd)
            estimator = {
                "name": "full_state_pd",
                "training_seed": -1,
                "model": None,
                "baseline": build_nonlearned_baseline("full_state_pd"),
            }
            rows = [
                _rollout(
                    estimator,
                    candidate_config,
                    condition,
                    episode_seed=int(row["episode_seed"]),
                    schedule_seed=int(row["schedule_seed"]),
                    device="cpu",
                )
                for row in validation_rows
            ]
            validation_episode_rows.extend(
                {
                    "joint_kp": float(kp),
                    "joint_kd": float(kd),
                    "trajectory_id": validation_row["trajectory_id"],
                    "episode_seed": int(validation_row["episode_seed"]),
                    "schedule_seed": int(validation_row["schedule_seed"]),
                    "failure": float(result["failure"]),
                    "tracking_rmse": float(result["tracking_rmse"]),
                    "control_effort": float(result["control_effort"]),
                }
                for validation_row, result in zip(validation_rows, rows)
            )
            candidates.append(
                {
                    "joint_kp": float(kp),
                    "joint_kd": float(kd),
                    "failure_count": int(sum(item["failure"] for item in rows)),
                    "failure_rate": float(sum(item["failure"] for item in rows) / len(rows)),
                    "episode_count": len(rows),
                    "mean_tracking_rmse": float(
                        sum(item["tracking_rmse"] for item in rows) / len(rows)
                    ),
                    "mean_control_effort": float(
                        sum(item["control_effort"] for item in rows) / len(rows)
                    ),
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            row["failure_count"] != 0,
            row["failure_rate"],
            row["mean_tracking_rmse"],
            row["mean_control_effort"],
            row["joint_kp"],
            row["joint_kd"],
        ),
    )
    if int(selected["failure_count"]) != 0:
        raise RuntimeError(
            "No full-state PD candidate achieved zero validation failures; "
            "the controller gate is not satisfied"
        )
    core = {
        "schema_version": 1,
        "selection_spec": spec,
        "selection_spec_sha256": canonical_json_hash(spec),
        "environment_design_sha256": environment_design_sha256,
        "candidates": candidates,
        "validation_episode_rows": validation_episode_rows,
        "validation_episode_rows_sha256": canonical_json_hash(
            validation_episode_rows
        ),
        "selected": selected,
        "validation_only": True,
        "test_episodes_used": 0,
        "learned_model_results_used": False,
    }
    artifact = {
        **core,
        "controller_selection_sha256": canonical_json_hash(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(destination, artifact)
    return artifact


def validate_controller_selection(config: dict[str, Any]) -> dict[str, Any] | None:
    if not is_formal(config):
        return None
    selection_path = config.get("controller_selection", {}).get("artifact")
    if not selection_path:
        raise ValueError("Formal config lacks controller_selection.artifact")
    artifact = _load_hashed(selection_path, "controller_selection_sha256")
    expected_spec = _controller_selection_spec(config)
    if artifact.get("selection_spec_sha256") != canonical_json_hash(expected_spec):
        raise ValueError("Controller selection design/validation split identity changed")
    candidates = artifact.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Controller selection candidate table is missing")
    expected_pairs = {
        (float(kp), float(kd))
        for kp in expected_spec["kp_grid"]
        for kd in expected_spec["kd_grid"]
    }
    observed_pairs = [
        (float(row.get("joint_kp", math.nan)), float(row.get("joint_kd", math.nan)))
        for row in candidates
    ]
    if len(observed_pairs) != len(expected_pairs) or set(observed_pairs) != expected_pairs:
        raise ValueError("Controller selection is not the exact unique Cartesian grid")
    episode_count = int(expected_spec["validation_episode_count"])
    required_numeric = (
        "joint_kp",
        "joint_kd",
        "failure_count",
        "failure_rate",
        "mean_tracking_rmse",
        "mean_control_effort",
        "episode_count",
    )
    for row in candidates:
        if any(not math.isfinite(float(row.get(key, math.nan))) for key in required_numeric):
            raise ValueError("Controller selection contains a non-finite candidate")
        if int(row["episode_count"]) != episode_count:
            raise ValueError("Controller selection candidate episode count mismatch")
        if int(row["failure_count"]) < 0 or int(row["failure_count"]) > episode_count:
            raise ValueError("Controller selection candidate failure count is invalid")
        if not math.isclose(
            float(row["failure_rate"]),
            int(row["failure_count"]) / episode_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Controller selection failure rate/count mismatch")
    episode_rows = artifact.get("validation_episode_rows")
    if not isinstance(episode_rows, list) or artifact.get(
        "validation_episode_rows_sha256"
    ) != canonical_json_hash(episode_rows):
        raise ValueError("Controller validation episode shard identity mismatch")
    if len(episode_rows) != episode_count * len(expected_pairs):
        raise ValueError("Controller validation episode shard is incomplete")
    expected_ids = {
        (
            row["trajectory_id"],
            int(row["episode_seed"]),
            int(row["schedule_seed"]),
        )
        for row in build_trajectory_manifest(config)["splits"]["val"]
    }
    for kp, kd in expected_pairs:
        pair_rows = [
            row
            for row in episode_rows
            if float(row["joint_kp"]) == kp and float(row["joint_kd"]) == kd
        ]
        observed_ids = {
            (
                row["trajectory_id"],
                int(row["episode_seed"]),
                int(row["schedule_seed"]),
            )
            for row in pair_rows
        }
        if observed_ids != expected_ids or len(pair_rows) != episode_count:
            raise ValueError("Controller validation episode identities are incomplete")
        candidate = next(
            row
            for row in candidates
            if float(row["joint_kp"]) == kp and float(row["joint_kd"]) == kd
        )
        aggregates = {
            "failure_count": int(sum(float(row["failure"]) for row in pair_rows)),
            "failure_rate": float(np.mean([float(row["failure"]) for row in pair_rows])),
            "mean_tracking_rmse": float(
                np.mean([float(row["tracking_rmse"]) for row in pair_rows])
            ),
            "mean_control_effort": float(
                np.mean([float(row["control_effort"]) for row in pair_rows])
            ),
        }
        if int(candidate["failure_count"]) != aggregates["failure_count"] or any(
            not math.isclose(
                float(candidate[key]), float(aggregates[key]), rel_tol=0.0, abs_tol=1e-12
            )
            for key in ("failure_rate", "mean_tracking_rmse", "mean_control_effort")
        ):
            raise ValueError("Controller candidate aggregate differs from episode shard")
    registered_winner = min(
        candidates,
        key=lambda row: (
            int(row["failure_count"]) != 0,
            float(row["failure_rate"]),
            float(row["mean_tracking_rmse"]),
            float(row["mean_control_effort"]),
            float(row["joint_kp"]),
            float(row["joint_kd"]),
        ),
    )
    selected = artifact["selected"]
    if selected != registered_winner:
        raise ValueError("Controller selected row is not the deterministic grid winner")
    if int(selected.get("failure_count", -1)) != 0:
        raise ValueError("Frozen controller selection did not pass the zero-failure gate")
    controller = config.get("controller", {})
    if float(selected["joint_kp"]) != float(controller.get("joint_kp")) or float(
        selected["joint_kd"]
    ) != float(controller.get("joint_kd")):
        raise ValueError("Configured PD gains differ from validation-only selection")
    if artifact.get("environment_design_sha256") != canonical_json_hash(
        current_environment_identity(config)
    ):
        raise ValueError("Controller selection environment/XML/source design changed")
    if not artifact.get("validation_only") or artifact.get("test_episodes_used") != 0:
        raise ValueError("Controller selection is not validation-only")
    if artifact.get("learned_model_results_used"):
        raise ValueError("Controller selection consulted learned-model results")
    return artifact


def tuning_design_identity(
    config: dict[str, Any],
    registry: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.setdefault("tuning", {}).pop("artifact", None)
    normalized.setdefault("training", {}).pop("learning_rate", None)
    normalized.setdefault("training", {}).pop("weight_decay", None)
    tuning = dict(config.get("tuning", {}))
    environment = validate_environment_freeze(config) if is_formal(config) else None
    controller = validate_controller_selection(config) if is_formal(config) else None
    gpu_binding = None
    output_root = tuning.get("output_root")
    namespace_artifact = tuning.get("namespace_artifact")
    gpu_lease_root = tuning.get("gpu_lease_root")
    lease_custody_receipt = tuning.get("lease_custody_receipt")
    lease_session_slots = int(tuning.get("lease_session_slots", 0))
    if is_formal(config):
        if not output_root or not Path(str(output_root)).is_absolute():
            raise ValueError("Formal tuning.output_root must be an absolute fixed path")
        if not namespace_artifact or not Path(str(namespace_artifact)).is_absolute():
            raise ValueError(
                "Formal tuning.namespace_artifact must be an absolute fixed path"
            )
        if not gpu_lease_root or not Path(str(gpu_lease_root)).is_absolute():
            raise ValueError("Formal tuning.gpu_lease_root must be an absolute path")
        if not lease_custody_receipt or not Path(
            str(lease_custody_receipt)
        ).is_absolute():
            raise ValueError(
                "Formal tuning.lease_custody_receipt must be an absolute fixed path"
            )
        if lease_session_slots != 28:
            raise ValueError(
                "Formal tuning requires exactly one precommitted session per 28 candidates"
            )
        resolved_output_root = _resolve(str(output_root))
        resolved_lease_root = _resolve(str(gpu_lease_root))
        resolved_namespace_artifact = _resolve(str(namespace_artifact))
        resolved_custody_receipt = _resolve(str(lease_custody_receipt))
        if (
            resolved_output_root == resolved_lease_root
            or resolved_lease_root in resolved_output_root.parents
            or resolved_output_root in resolved_lease_root.parents
        ):
            raise ValueError(
                "Formal GPU lease root must be outside the tuning output tree"
            )
        for label, fixed_artifact in (
            ("namespace artifact", resolved_namespace_artifact),
            ("lease custody receipt", resolved_custody_receipt),
        ):
            if (
                fixed_artifact == resolved_output_root
                or resolved_output_root in fixed_artifact.parents
            ):
                raise ValueError(
                    f"Formal tuning {label} must be outside the candidate output tree"
                )
        if environment is None:
            raise ValueError("Formal tuning design lacks the environment freeze")
        physical_index = str(tuning.get("physical_gpu_index", ""))
        gpu_uuid = str(tuning.get("gpu_uuid", ""))
        gpu_pci_bus_id = str(tuning.get("gpu_pci_bus_id", ""))
        if physical_index != "0":
            raise ValueError("Formal validation tuning is restricted to physical GPU0")
        matches = [
            row
            for row in environment.get("physical_gpu_inventory", [])
            if str(row.get("physical_index")) == "0"
            and str(row.get("uuid", "")).lower() == gpu_uuid.lower()
            and str(row.get("pci_bus_id", "")).lower() == gpu_pci_bus_id.lower()
        ]
        if len(matches) != 1:
            raise ValueError(
                "Formal tuning GPU UUID/PCI identity differs from frozen physical GPU0"
            )
        precision_policy = environment.get("torch_runtime", {}).get(
            "precision_policy"
        )
        if not isinstance(precision_policy, dict):
            raise ValueError("Formal tuning environment lacks a precision policy")
        trusted_nvidia_smi = environment.get("trusted_nvidia_smi")
        if (
            not isinstance(trusted_nvidia_smi, dict)
            or trusted_nvidia_smi != trusted_nvidia_smi_identity(config)
            or len(str(trusted_nvidia_smi.get("executable_sha256", ""))) != 64
        ):
            raise ValueError(
                "Formal tuning environment lacks the exact trusted nvidia-smi identity"
            )
        gpu_binding = {
            "physical_index": "0",
            "uuid": matches[0]["uuid"],
            "pci_bus_id": matches[0]["pci_bus_id"],
            "precision_policy": precision_policy,
        }
    core = {
        "schema_version": 1,
        "phase": "validation_tuning",
        "normalized_design_config_sha256": canonical_json_hash(normalized),
        "tuning_seed": int(tuning.get("seed", 17)),
        "learning_rates": [
            float(value) for value in tuning.get("learning_rates", [3e-4, 1e-3])
        ],
        "weight_decays": [
            float(value) for value in tuning.get("weight_decays", [0.0, 1e-4])
        ],
        "epochs": int(tuning.get("epochs", 20)),
        "optimizer_steps_per_epoch": tuning.get("optimizer_steps_per_epoch"),
        "planned_optimizer_steps": tuning.get("planned_optimizer_steps"),
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "validation_rows_sha256": canonical_json_hash(trajectory["splits"]["val"]),
        "parameter_registry_sha256": registry["registry_sha256"],
        "implementation_sha256": registry["implementation"]["aggregate_sha256"],
        "environment_sha256": environment["environment_sha256"] if environment else None,
        "controller_selection_sha256": (
            controller["controller_selection_sha256"] if controller else None
        ),
        "tuning_output_root": (
            str(_resolve(str(output_root))) if output_root else None
        ),
        "tuning_namespace_artifact": (
            str(_resolve(str(namespace_artifact))) if namespace_artifact else None
        ),
        "tuning_gpu_lease_root": (
            str(_resolve(str(gpu_lease_root))) if gpu_lease_root else None
        ),
        "tuning_lease_custody_receipt": (
            str(_resolve(str(lease_custody_receipt)))
            if lease_custody_receipt
            else None
        ),
        "tuning_lease_session_slot_count": lease_session_slots,
        "tuning_gpu_identity": gpu_binding,
        "trusted_nvidia_smi": (
            environment.get("trusted_nvidia_smi") if environment else None
        ),
        "validation_only": True,
        "test_episodes_used": 0,
    }
    return {**core, "tuning_design_sha256": canonical_json_hash(core)}


def _tuning_float_slug(value: float) -> str:
    text = format(float(value), ".12g")
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _tuning_gpu_lease_stem(gpu_uuid: str) -> str:
    return str(gpu_uuid).lower().replace("-", "_")


def _tuning_lease_namespace_root(
    design: dict[str, Any], namespace_instance_id: str
) -> Path:
    gpu_identity = design.get("tuning_gpu_identity") or {}
    lease_root = _resolve(str(design.get("tuning_gpu_lease_root", "")))
    return (
        lease_root
        / "evidence"
        / _tuning_gpu_lease_stem(str(gpu_identity.get("uuid", "")))
        / "namespaces"
        / str(namespace_instance_id)
    ).resolve()


def _tuning_namespace_slots(
    config: dict[str, Any],
    design: dict[str, Any],
    *,
    namespace_instance_id: str,
) -> list[dict[str, Any]]:
    """Return the exact immutable 7x2x2 path/attempt namespace."""
    tuning = dict(config.get("tuning", {}))
    output_root = _resolve(str(tuning.get("output_root", "")))
    models = [
        str(value)
        for value in config.get("matching", {}).get(
            "registered_models", REGISTERED_MODELS
        )
    ]
    learning_rates = [float(value) for value in design["learning_rates"]]
    weight_decays = [float(value) for value in design["weight_decays"]]
    lease_namespace_root = _tuning_lease_namespace_root(
        design, namespace_instance_id
    )
    if len(models) != 7 or len(set(models)) != 7:
        raise ValueError("Formal tuning namespace requires seven unique models")
    if len(learning_rates) != 2 or len(set(learning_rates)) != 2:
        raise ValueError("Formal tuning namespace requires two learning rates")
    if len(weight_decays) != 2 or len(set(weight_decays)) != 2:
        raise ValueError("Formal tuning namespace requires two weight decays")
    slots: list[dict[str, Any]] = []
    for model in models:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                index = len(slots) + 1
                candidate_id = (
                    f"candidate_{index:02d}_{model}_"
                    f"lr_{_tuning_float_slug(learning_rate)}_"
                    f"wd_{_tuning_float_slug(weight_decay)}"
                )
                output_dir = (output_root / candidate_id).resolve()
                evidence_dir = (lease_namespace_root / "candidates" / candidate_id).resolve()
                if output_dir.parent != output_root:
                    raise ValueError("Tuning candidate path escapes the fixed output root")
                if evidence_dir.parent.parent != lease_namespace_root:
                    raise ValueError("Tuning lease evidence path escapes its namespace root")
                attempt_core = {
                    "namespace_instance_id": namespace_instance_id,
                    "tuning_design_sha256": design["tuning_design_sha256"],
                    "slot_index": index,
                    "candidate_id": candidate_id,
                    "registered_name": model,
                    "seed": int(design["tuning_seed"]),
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "output_dir": str(output_dir),
                }
                slots.append(
                    {
                        **attempt_core,
                        "attempt_id": "tuning-"
                        + canonical_json_hash(attempt_core)[:32],
                        "result_path": str(
                            (output_dir / "tuning_candidate_result.json").resolve()
                        ),
                        "authorization_token_path": str(
                            (evidence_dir / "authorization.json").resolve()
                        ),
                        "completion_receipt_path": str(
                            (evidence_dir / "completion.json").resolve()
                        ),
                        "worker_ready_path": str(
                            (evidence_dir / "worker_ready.json").resolve()
                        ),
                        "worker_launch_path": str(
                            (evidence_dir / "worker_launch.json").resolve()
                        ),
                        "output_reservation_path": str(
                            (evidence_dir / "output_reservation.json").resolve()
                        ),
                    }
                )
    if (
        len(slots) != 28
        or len({row["result_path"] for row in slots}) != 28
        or len({row["authorization_token_path"] for row in slots}) != 28
        or len({row["completion_receipt_path"] for row in slots}) != 28
        or len({row["worker_ready_path"] for row in slots}) != 28
        or len({row["worker_launch_path"] for row in slots}) != 28
        or len({row["output_reservation_path"] for row in slots}) != 28
    ):
        raise ValueError("Tuning namespace must contain exactly 28 unique slots")
    return slots


def _tuning_lease_session_slots(
    design: dict[str, Any],
    *,
    namespace_instance_id: str,
    candidate_slots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Precommit a finite append-only session chain outside the output tree."""
    count = int(design.get("tuning_lease_session_slot_count", 0))
    if count != 28 or len(candidate_slots) != 28:
        raise ValueError("Tuning lease namespace requires exactly 28 session slots")
    root = _tuning_lease_namespace_root(design, namespace_instance_id)
    slots: list[dict[str, Any]] = []
    for session_index, candidate_slot in enumerate(candidate_slots, start=1):
        session_core = {
            "namespace_instance_id": namespace_instance_id,
            "tuning_design_sha256": design["tuning_design_sha256"],
            "session_index": session_index,
            "candidate_slot_index": int(candidate_slot["slot_index"]),
            "candidate_attempt_id": candidate_slot["attempt_id"],
            "candidate_id": candidate_slot["candidate_id"],
        }
        session_id = "lease-session-" + canonical_json_hash(session_core)[:32]
        session_dir = (root / "sessions" / f"session_{session_index:03d}").resolve()
        slots.append(
            {
                **session_core,
                "session_id": session_id,
                "acquisition_path": str((session_dir / "acquisition.json").resolve()),
                "release_path": str((session_dir / "release.json").resolve()),
            }
        )
    return slots


def _tuning_lease_custody_spec(
    config: dict[str, Any],
    design: dict[str, Any],
    *,
    namespace_instance_id: str,
    candidate_slots: list[dict[str, Any]],
    session_slots: list[dict[str, Any]],
) -> dict[str, Any]:
    lease_root = _resolve(str(design["tuning_gpu_lease_root"]))
    gpu_stem = _tuning_gpu_lease_stem(design["tuning_gpu_identity"]["uuid"])
    namespace_root = _tuning_lease_namespace_root(design, namespace_instance_id)
    output_root = _resolve(str(design["tuning_output_root"]))
    if namespace_root == output_root or output_root in namespace_root.parents:
        raise ValueError("Tuning lease custody root overlaps the candidate output tree")
    core = {
        "schema_version": 1,
        "mode": "auditor_managed_create_only_append_only",
        "acl_policy": "create_only_no_modify_no_delete_existing_objects",
        "same_privilege_administrator_deletion_out_of_scope": True,
        "lease_root": str(lease_root),
        "lease_namespace_root": str(namespace_root),
        "global_mutex_path": str((lease_root / f"{gpu_stem}.mutex").resolve()),
        "global_history_path": str(
            (lease_root / f"{gpu_stem}.history.json").resolve()
        ),
        "global_record_path": str(
            (lease_root / f"{gpu_stem}.lease.json").resolve()
        ),
        "terminal_failure_receipt_path": str(
            (namespace_root / "terminal_failure.json").resolve()
        ),
        "trusted_nvidia_smi": design["trusted_nvidia_smi"],
        "session_authentication": (
            "sha256_commitment_then_hmac_sha256_token_completion_release_"
            "with_32_byte_secret_revealed_only_in_release"
        ),
        "one_candidate_per_session": True,
        "custody_receipt_path": str(
            _resolve(str(config["tuning"]["lease_custody_receipt"]))
        ),
        "candidate_authorization_paths_sha256": canonical_json_hash(
            [row["authorization_token_path"] for row in candidate_slots]
        ),
        "candidate_completion_paths_sha256": canonical_json_hash(
            [row["completion_receipt_path"] for row in candidate_slots]
        ),
        "candidate_worker_ready_paths_sha256": canonical_json_hash(
            [row["worker_ready_path"] for row in candidate_slots]
        ),
        "candidate_worker_launch_paths_sha256": canonical_json_hash(
            [row["worker_launch_path"] for row in candidate_slots]
        ),
        "candidate_output_reservation_paths_sha256": canonical_json_hash(
            [row["output_reservation_path"] for row in candidate_slots]
        ),
        "session_slots_sha256": canonical_json_hash(session_slots),
        "session_chain_initial_head_sha256": None,
    }
    return {**core, "lease_custody_spec_sha256": canonical_json_hash(core)}


def _tuning_launcher_identity() -> dict[str, Any]:
    files = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in (
            "scripts/run_closed_loop_tuning.py",
            "scripts/start_closed_loop_tuning.ps1",
        )
    }
    return {
        "files": files,
        "aggregate_sha256": canonical_json_hash(files),
    }


def create_tuning_namespace(
    config: dict[str, Any],
    registry_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    *,
    runtime_resource_policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the write-once namespace that must be externally anchored."""
    if not is_formal(config):
        raise ValueError("A tuning namespace is only defined for the formal design")
    destination = _resolve(output_path)
    configured = config.get("tuning", {}).get("namespace_artifact")
    if not configured or destination != _resolve(str(configured)):
        raise ValueError("Tuning namespace output differs from the fixed config path")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite tuning namespace: {destination}")
    output_root = _resolve(str(config.get("tuning", {}).get("output_root", "")))
    if output_root.exists():
        raise FileExistsError(
            "Refusing to initialize a tuning namespace over an existing output root"
        )
    registry = load_parameter_registry(registry_path, expected_config=config)
    trajectory = load_trajectory_manifest(
        _resolve(data_root) / "trajectory_manifest.json", expected_config=config
    )
    design = tuning_design_identity(config, registry, trajectory)
    instance_id = str(uuid.uuid4())
    slots = _tuning_namespace_slots(
        config, design, namespace_instance_id=instance_id
    )
    session_slots = _tuning_lease_session_slots(
        design,
        namespace_instance_id=instance_id,
        candidate_slots=slots,
    )
    custody_spec = _tuning_lease_custody_spec(
        config,
        design,
        namespace_instance_id=instance_id,
        candidate_slots=slots,
        session_slots=session_slots,
    )
    core = {
        "schema_version": 4,
        "purpose": "validation_only_seed17_optimizer_tuning_namespace",
        "namespace_instance_id": instance_id,
        "source_config_sha256": canonical_json_hash(config),
        "tuning_design_sha256": design["tuning_design_sha256"],
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "output_root": str(output_root),
        "selection_output": str(_resolve(config["tuning"]["artifact"])),
        "gpu_identity": design["tuning_gpu_identity"],
        "trusted_nvidia_smi": design["trusted_nvidia_smi"],
        "gpu_lease_root": design["tuning_gpu_lease_root"],
        "lease_custody_receipt_path": design[
            "tuning_lease_custody_receipt"
        ],
        "lease_custody_spec": custody_spec,
        "terminal_failure_receipt_path": custody_spec[
            "terminal_failure_receipt_path"
        ],
        "lease_session_slot_count": len(session_slots),
        "lease_session_slots": session_slots,
        "launcher_implementation": _tuning_launcher_identity(),
        "candidate_count": len(slots),
        "candidate_slots": slots,
        "write_once": True,
        "external_anchor_required": True,
        "validation_only": True,
        "test_episodes_used": 0,
    }
    if runtime_resource_policy_path is not None:
        if _resolve(runtime_resource_policy_path) != SHARED_GPU_POLICY_PATH.resolve():
            raise PermissionError("Shared policy path is not the explicitly fixed runtime policy")
        resource_policy = _load_shared_gpu_policy_anchor(required=True)
        if resource_policy["policy"]["gpu_uuid"] != core["gpu_identity"]["uuid"]:
            raise PermissionError("Shared policy GPU differs from the tuning namespace")
        core.update({"schema_version": 5, "runtime_resource_policy": resource_policy})
        core["runtime_compatibility_review"] = _shared_runtime_compatibility_anchor(data_root)
    artifact = {
        **core,
        "tuning_namespace_sha256": canonical_json_hash(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(destination, artifact)
    return artifact


def validate_tuning_namespace(
    config: dict[str, Any],
    registry: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    namespace_path: str | Path,
    expected_namespace_sha256: str,
    expected_namespace_file_sha256: str,
) -> dict[str, Any]:
    """Validate both canonical and byte-level externally supplied anchors."""
    configured = config.get("tuning", {}).get("namespace_artifact")
    source = _resolve(namespace_path)
    if not configured or source != _resolve(str(configured)):
        raise ValueError("Tuning namespace path differs from the fixed config path")
    if sha256_file(source) != str(expected_namespace_file_sha256):
        raise ValueError("Tuning namespace external file anchor mismatch")
    artifact = _load_hashed(source, "tuning_namespace_sha256")
    if artifact.get("tuning_namespace_sha256") != str(
        expected_namespace_sha256
    ):
        raise ValueError("Tuning namespace external canonical anchor mismatch")
    design = tuning_design_identity(config, registry, trajectory)
    expected = {
        "schema_version": 5 if "runtime_resource_policy" in artifact else 4,
        "source_config_sha256": canonical_json_hash(config),
        "tuning_design_sha256": design["tuning_design_sha256"],
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "output_root": design["tuning_output_root"],
        "selection_output": str(_resolve(config["tuning"]["artifact"])),
        "gpu_identity": design["tuning_gpu_identity"],
        "trusted_nvidia_smi": design["trusted_nvidia_smi"],
        "gpu_lease_root": design["tuning_gpu_lease_root"],
        "lease_custody_receipt_path": design[
            "tuning_lease_custody_receipt"
        ],
        "lease_session_slot_count": 28,
        "launcher_implementation": _tuning_launcher_identity(),
        "candidate_count": 28,
        "write_once": True,
        "external_anchor_required": True,
        "validation_only": True,
        "test_episodes_used": 0,
    }
    if "runtime_resource_policy" in artifact:
        policy = _load_shared_gpu_policy_anchor(required=True)
        if policy["policy"]["gpu_uuid"] != artifact["gpu_identity"]["uuid"]:
            raise PermissionError("Shared policy GPU differs from the tuning namespace")
        expected["runtime_resource_policy"] = policy
        expected["runtime_compatibility_review"] = _shared_runtime_compatibility_anchor(
            PROJECT_ROOT / "runs/closed_loop_confirmatory/data")
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"Tuning namespace identity mismatch for {key}")
    instance_id = str(artifact.get("namespace_instance_id", ""))
    if not instance_id:
        raise ValueError("Tuning namespace lacks its unique instance identity")
    expected_slots = _tuning_namespace_slots(
        config, design, namespace_instance_id=instance_id
    )
    if artifact.get("candidate_slots") != expected_slots:
        raise ValueError("Tuning namespace slot/path/attempt registry mismatch")
    expected_session_slots = _tuning_lease_session_slots(
        design,
        namespace_instance_id=instance_id,
        candidate_slots=expected_slots,
    )
    if artifact.get("lease_session_slots") != expected_session_slots:
        raise ValueError("Tuning namespace lease-session registry mismatch")
    expected_custody_spec = _tuning_lease_custody_spec(
        config,
        design,
        namespace_instance_id=instance_id,
        candidate_slots=expected_slots,
        session_slots=expected_session_slots,
    )
    if artifact.get("lease_custody_spec") != expected_custody_spec:
        raise ValueError("Tuning namespace custody specification mismatch")
    if artifact.get("terminal_failure_receipt_path") != expected_custody_spec[
        "terminal_failure_receipt_path"
    ]:
        raise ValueError("Tuning namespace terminal-failure path mismatch")
    return artifact


def validate_tuning_runtime_identity(
    runtime_device: dict[str, Any], gpu_identity: dict[str, Any]
) -> None:
    """Reject GPU ordinal/UUID/PCI or precision drift for tuning candidates."""
    selected_gpu = runtime_device.get("selected_physical_gpu") or {}
    if (
        str(selected_gpu.get("uuid", "")).lower()
        != str(gpu_identity.get("uuid", "")).lower()
        or str(selected_gpu.get("pci_bus_id", "")).lower()
        != str(gpu_identity.get("pci_bus_id", "")).lower()
        or str(gpu_identity.get("physical_index")) != "0"
        or runtime_device.get("precision_policy")
        != gpu_identity.get("precision_policy")
    ):
        raise ValueError("Validation-tuning runtime GPU0 UUID/PCI/precision mismatch")


def validate_terminal_tuning_failure(
    namespace: dict[str, Any], custody: dict[str, Any]
) -> None:
    """Reject a namespace once its precommitted terminal receipt exists.

    Absence is the only admissible state.  A present receipt is deeply checked
    for auditability and then rejected unconditionally; a malformed receipt is
    also a hard failure rather than an opportunity to continue.
    """
    configured_path = Path(namespace["terminal_failure_receipt_path"]).resolve()
    custody_path = Path(
        namespace["lease_custody_spec"]["terminal_failure_receipt_path"]
    ).resolve()
    if configured_path != custody_path:
        raise PermissionError("Tuning terminal-failure path binding is divergent")
    if not configured_path.exists():
        return
    if not configured_path.is_file():
        raise PermissionError("Tuning terminal-failure evidence is not a file")
    try:
        receipt = _load_hashed(
            configured_path, "tuning_namespace_terminal_failure_sha256"
        )
    except Exception as exc:
        raise PermissionError(
            "Tuning terminal-failure receipt is malformed or tampered"
        ) from exc
    expected = {
        "schema_version": 1,
        "purpose": "terminal_validation_tuning_namespace_failure",
        "status": "terminal_failed_no_rerun_permitted",
        "terminal_failure_receipt_path": str(configured_path),
        "tuning_namespace_sha256": namespace["tuning_namespace_sha256"],
        "namespace_instance_id": namespace["namespace_instance_id"],
        "tuning_lease_custody_receipt_path": custody["_validated_path"],
        "tuning_lease_custody_receipt_sha256": custody[
            "tuning_lease_custody_receipt_sha256"
        ],
        "tuning_lease_custody_receipt_file_sha256": custody[
            "_validated_file_sha256"
        ],
        "recovery_policy": (
            "one_session_per_candidate; any candidate start without completion, "
            "or any session whose owner dies without a valid authenticated release, "
            "permanently terminates the namespace; no resume, rerun, continuation "
            "token, synthesized release, or replacement attempt is permitted"
        ),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise PermissionError(
                f"Tuning terminal-failure receipt mismatch for {key}"
            )
    allowed_reasons = {
        "authorization_without_completion_after_owner_death",
        "authorization_without_completion_before_session_open",
        "orchestrator_or_candidate_failed_after_authorization",
        "attempted_reuse_of_inactive_or_incomplete_token",
        "lease_session_closed_with_authorization_without_completion",
        "candidate_started_without_completion_after_owner_death",
        "candidate_started_without_completion_before_session_open",
        "orchestrator_or_candidate_failed_after_candidate_start",
        "lease_session_closed_with_started_candidate_without_completion",
        "unreleased_session_after_owner_death",
        "invalid_session_release_after_owner_death",
        "lease_session_failed_before_candidate_completion",
        "active_gpu_lease_lost_before_session_release",
    }
    if receipt.get("reason") not in allowed_reasons:
        raise PermissionError("Tuning terminal-failure receipt has an unknown reason")
    indices = receipt.get("incomplete_slot_indices")
    attempts = receipt.get("incomplete_attempt_ids")
    if not isinstance(indices, list) or not isinstance(attempts, list):
        raise PermissionError("Tuning terminal-failure slot evidence is invalid")
    try:
        normalized_indices = [int(index) for index in indices]
    except (TypeError, ValueError) as exc:
        raise PermissionError(
            "Tuning terminal-failure slot evidence is invalid"
        ) from exc
    if (
        not normalized_indices
        or len(normalized_indices) != len(set(normalized_indices))
        or len(attempts) != len(normalized_indices)
        or any(not isinstance(attempt, str) or not attempt for attempt in attempts)
    ):
        raise PermissionError("Tuning terminal-failure slot evidence is invalid")
    registered = {
        int(slot["slot_index"]): slot["attempt_id"]
        for slot in namespace["candidate_slots"]
    }
    if any(
        registered.get(index) != attempt
        for index, attempt in zip(normalized_indices, attempts)
    ):
        raise PermissionError("Tuning terminal-failure slot/attempt binding is invalid")
    session_id = receipt.get("lease_session_id")
    acquisition_sha256 = receipt.get("lease_session_acquisition_sha256")
    before_session_open = receipt.get("reason") in {
        "authorization_without_completion_before_session_open",
        "candidate_started_without_completion_before_session_open",
    }
    if before_session_open:
        if session_id is not None or acquisition_sha256 is not None:
            raise PermissionError(
                "Tuning terminal-failure pre-session receipt has a session binding"
            )
    elif session_id is None or acquisition_sha256 is None:
        raise PermissionError(
            "Tuning terminal-failure receipt lacks its lease-session binding"
        )
    else:
        matches = [
            slot
            for slot in namespace["lease_session_slots"]
            if slot["session_id"] == session_id
        ]
        if len(matches) != 1 or not Path(matches[0]["acquisition_path"]).is_file():
            raise PermissionError(
                "Tuning terminal-failure lease session is not precommitted"
            )
        try:
            acquisition = _load_hashed(
                matches[0]["acquisition_path"],
                "tuning_lease_session_acquisition_sha256",
            )
        except Exception as exc:
            raise PermissionError(
                "Tuning terminal-failure lease acquisition is malformed"
            ) from exc
        if (
            acquisition.get("tuning_lease_session_acquisition_sha256")
            != acquisition_sha256
            or acquisition.get("session_id") != session_id
            or acquisition.get("session_index") != int(matches[0]["session_index"])
            or Path(str(acquisition.get("acquisition_path", ""))).resolve()
            != Path(matches[0]["acquisition_path"]).resolve()
            or acquisition.get("tuning_namespace_sha256")
            != namespace["tuning_namespace_sha256"]
            or acquisition.get("namespace_instance_id")
            != namespace["namespace_instance_id"]
            or acquisition.get("tuning_lease_custody_receipt_sha256")
            != custody["tuning_lease_custody_receipt_sha256"]
        ):
            raise PermissionError(
                "Tuning terminal-failure lease acquisition binding is invalid"
            )
    raise PermissionError(
        "Tuning namespace is permanently terminal; freeze/training is forbidden"
    )


def validate_tuning_lease_custody_receipt(
    config: dict[str, Any],
    namespace: dict[str, Any],
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    expected_receipt_file_sha256: str,
    namespace_file_sha256: str,
) -> dict[str, Any]:
    """Consume the external create-only custody attestation before GPU work."""
    source = _resolve(receipt_path)
    configured = config.get("tuning", {}).get("lease_custody_receipt")
    if not configured or source != _resolve(str(configured)):
        raise PermissionError("Tuning lease custody receipt path is not config-fixed")
    if (
        len(str(expected_receipt_file_sha256)) != 64
        or not source.is_file()
        or sha256_file(source) != str(expected_receipt_file_sha256)
    ):
        raise PermissionError(
            "Tuning lease custody receipt differs from its external file anchor"
        )
    receipt = _load_hashed(source, "tuning_lease_custody_receipt_sha256")
    if receipt.get("tuning_lease_custody_receipt_sha256") != str(
        expected_receipt_sha256
    ):
        raise PermissionError(
            "Tuning lease custody receipt differs from its external canonical anchor"
        )
    custody_spec = namespace["lease_custody_spec"]
    expected = {
        "schema_version": 1,
        "purpose": "tuning_lease_evidence_external_custody",
        "mode": "auditor_managed_create_only_append_only",
        "acl_policy": "create_only_no_modify_no_delete_existing_objects",
        "same_privilege_administrator_deletion_out_of_scope": True,
        "tuning_namespace_sha256": namespace["tuning_namespace_sha256"],
        "tuning_namespace_file_sha256": str(namespace_file_sha256),
        "namespace_instance_id": namespace["namespace_instance_id"],
        "lease_custody_spec_sha256": custody_spec[
            "lease_custody_spec_sha256"
        ],
        "lease_namespace_root": custody_spec["lease_namespace_root"],
        "candidate_authorization_paths_sha256": custody_spec[
            "candidate_authorization_paths_sha256"
        ],
        "candidate_completion_paths_sha256": custody_spec[
            "candidate_completion_paths_sha256"
        ],
        "candidate_worker_ready_paths_sha256": custody_spec[
            "candidate_worker_ready_paths_sha256"
        ],
        "candidate_worker_launch_paths_sha256": custody_spec[
            "candidate_worker_launch_paths_sha256"
        ],
        "candidate_output_reservation_paths_sha256": custody_spec[
            "candidate_output_reservation_paths_sha256"
        ],
        "session_slots_sha256": custody_spec["session_slots_sha256"],
        "gpu_identity": namespace["gpu_identity"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise PermissionError(f"Tuning lease custody mismatch for {key}")
    if not str(receipt.get("external_attestation_id", "")).strip() or not str(
        receipt.get("external_custodian", "")
    ).strip():
        raise PermissionError(
            "Tuning lease custody lacks an external custodian/attestation identity"
        )
    evidence_root = Path(custody_spec["lease_namespace_root"])
    if not evidence_root.is_absolute():
        raise PermissionError("Tuning lease custody evidence root is not absolute")
    if not evidence_root.is_dir():
        raise PermissionError(
            "Externally custodied tuning lease evidence root is not provisioned"
        )
    validated = {
        **receipt,
        "_validated_path": str(source),
        "_validated_file_sha256": str(expected_receipt_file_sha256),
    }
    validate_terminal_tuning_failure(namespace, validated)
    return validated


def _tuning_pid_is_running(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        get_exit_code.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        handle = open_process(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() != 87
        try:
            code = wintypes.DWORD()
            return bool(get_exit_code(handle, ctypes.byref(code))) and int(
                code.value
            ) == 259
        finally:
            close_handle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tuning_mutex_is_held(path: str | Path) -> bool:
    """Probe that another process still owns the UUID mutex."""
    source = _resolve(path)
    if not source.is_file():
        return False
    handle = source.open("r+b")
    acquired = False
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError:
                return True
            finally:
                if acquired:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - production target is Windows
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                return True
            finally:
                if acquired:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _load_tuning_global_lease_state(
    namespace: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = namespace["lease_custody_spec"]
    history_path = Path(spec["global_history_path"])
    record_path = Path(spec["global_record_path"])
    if not history_path.is_file() or not record_path.is_file():
        raise PermissionError("Tuning global lease history/record is absent")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history_core = {
        "schema_version": history.get("schema_version"),
        "events": history.get("events"),
    }
    if (
        not isinstance(history_core["events"], list)
        or canonical_json_hash(history_core) != history.get("history_sha256")
    ):
        raise PermissionError("Tuning global lease history hash mismatch")
    previous = None
    for index, event in enumerate(history_core["events"], start=1):
        event_core = {
            key: value for key, value in event.items() if key != "event_sha256"
        }
        if (
            int(event.get("event_index", -1)) != index
            or event.get("previous_event_sha256") != previous
            or canonical_json_hash(event_core) != event.get("event_sha256")
        ):
            raise PermissionError("Tuning global lease event chain mismatch")
        previous = event["event_sha256"]
    if not history_core["events"]:
        raise PermissionError("Tuning global lease history is empty")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record != history_core["events"][-1]:
        raise PermissionError("Tuning global lease two-file state diverged")
    return history, record


def _tuning_session_evidence_inventory(
    namespace: dict[str, Any], session_id: str
) -> dict[str, Any]:
    """Reconstruct the exact append-only token/completion set for one session."""
    authorizations: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    for slot in namespace["candidate_slots"]:
        authorization_path = Path(slot["authorization_token_path"])
        completion_path = Path(slot["completion_receipt_path"])
        if completion_path.exists() and not authorization_path.is_file():
            raise PermissionError(
                "Tuning completion exists without its authorization token"
            )
        token: dict[str, Any] | None = None
        if authorization_path.is_file():
            token = _load_hashed(
                authorization_path, "tuning_lease_authorization_sha256"
            )
            if token.get("lease_session_id") == session_id:
                authorizations.append(
                    {
                        "slot_index": int(slot["slot_index"]),
                        "attempt_id": slot["attempt_id"],
                        "path": slot["authorization_token_path"],
                        "canonical_sha256": token[
                            "tuning_lease_authorization_sha256"
                        ],
                        "session_hmac_sha256": token.get(
                            "session_hmac_sha256"
                        ),
                        "file_sha256": sha256_file(authorization_path),
                    }
                )
        if completion_path.is_file():
            completion = _load_hashed(
                completion_path, "tuning_lease_completion_sha256"
            )
            if completion.get("lease_session_id") == session_id:
                if (
                    token is None
                    or token.get("lease_session_id") != session_id
                    or completion.get("tuning_lease_authorization_sha256")
                    != token.get("tuning_lease_authorization_sha256")
                ):
                    raise PermissionError(
                        "Tuning completion/session authorization/HMAC evidence binding is invalid"
                    )
                completions.append(
                    {
                        "slot_index": int(slot["slot_index"]),
                        "attempt_id": slot["attempt_id"],
                        "path": slot["completion_receipt_path"],
                        "canonical_sha256": completion[
                            "tuning_lease_completion_sha256"
                        ],
                        "file_sha256": sha256_file(completion_path),
                        "authorization_sha256": token[
                            "tuning_lease_authorization_sha256"
                        ],
                        "session_hmac_sha256": completion.get(
                            "session_hmac_sha256"
                        ),
                    }
                )
    authorizations.sort(key=lambda row: int(row["slot_index"]))
    completions.sort(key=lambda row: int(row["slot_index"]))
    core = {
        "issued_authorizations": authorizations,
        "completion_receipts": completions,
        "issued_authorization_count": len(authorizations),
        "completion_receipt_count": len(completions),
        "issued_authorizations_sha256": canonical_json_hash(authorizations),
        "completion_receipts_sha256": canonical_json_hash(completions),
        "all_issued_authorizations_completed": (
            len(authorizations) == len(completions)
            and [row["slot_index"] for row in authorizations]
            == [row["slot_index"] for row in completions]
        ),
    }
    return {**core, "session_evidence_set_sha256": canonical_json_hash(core)}


def _tuning_session_hmac_sha256(
    secret: bytes, core: dict[str, Any]
) -> str:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise PermissionError("Tuning session HMAC secret is not 32 bytes")
    message = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _verify_tuning_session_artifact_hmac(
    payload: dict[str, Any], *, self_hash_field: str, secret: bytes
) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            self_hash_field,
            "created_at_utc",
            "session_hmac_sha256",
        }
    }
    expected = _tuning_session_hmac_sha256(secret, core)
    if not hmac.compare_digest(
        str(payload.get("session_hmac_sha256", "")), expected
    ):
        raise PermissionError("Tuning session artifact HMAC mismatch")


def _validate_tuning_session_chain(
    namespace: dict[str, Any],
    *,
    session_index: int,
    custody: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    slots = namespace["lease_session_slots"]
    if not 1 <= int(session_index) <= len(slots):
        raise PermissionError("Tuning lease session index is outside the namespace")
    previous_event_sha256 = None
    selected_acquisition: dict[str, Any] | None = None
    selected_release: dict[str, Any] | None = None
    for index, slot in enumerate(slots[: int(session_index)], start=1):
        acquisition_path = Path(slot["acquisition_path"])
        if not acquisition_path.is_file():
            raise PermissionError("Tuning lease session chain has a missing acquisition")
        acquisition = _load_hashed(
            acquisition_path, "tuning_lease_session_acquisition_sha256"
        )
        expected_acquisition = {
            "schema_version": 1,
            "purpose": "tuning_gpu_lease_session_acquisition",
            "tuning_namespace_sha256": namespace["tuning_namespace_sha256"],
            "namespace_instance_id": namespace["namespace_instance_id"],
            "tuning_lease_custody_receipt_sha256": custody[
                "tuning_lease_custody_receipt_sha256"
            ],
            "tuning_lease_custody_receipt_file_sha256": custody[
                "_validated_file_sha256"
            ],
            "session_index": index,
            "session_id": slot["session_id"],
            "acquisition_path": slot["acquisition_path"],
            "release_path": slot["release_path"],
            "previous_session_event_sha256": previous_event_sha256,
            "candidate_slot_index": int(slot["candidate_slot_index"]),
            "candidate_attempt_id": slot["candidate_attempt_id"],
            "candidate_id": slot["candidate_id"],
            "gpu_identity": namespace["gpu_identity"],
        }
        for key, value in expected_acquisition.items():
            if acquisition.get(key) != value:
                raise PermissionError(
                    f"Tuning lease session acquisition mismatch for {key}"
                )
        assigned_slots = [
            candidate_slot
            for candidate_slot in namespace["candidate_slots"]
            if int(candidate_slot["slot_index"])
            == int(acquisition.get("candidate_slot_index", -1))
            and candidate_slot["attempt_id"]
            == acquisition.get("candidate_attempt_id")
            and candidate_slot["candidate_id"]
            == acquisition.get("candidate_id")
        ]
        if (
            len(assigned_slots) != 1
            or len(str(acquisition.get("session_hmac_commitment_sha256", "")))
            != 64
        ):
            raise PermissionError(
                "Tuning lease session candidate/HMAC commitment is invalid"
            )
        assigned_slot = assigned_slots[0]
        release_path = Path(slot["release_path"])
        release = None
        if release_path.is_file():
            release = _load_hashed(
                release_path, "tuning_lease_session_release_sha256"
            )
            inventory = _tuning_session_evidence_inventory(
                namespace, slot["session_id"]
            )
            expected_release = {
                "schema_version": 1,
                "purpose": "tuning_gpu_lease_session_release",
                "tuning_namespace_sha256": namespace[
                    "tuning_namespace_sha256"
                ],
                "namespace_instance_id": namespace["namespace_instance_id"],
                "session_index": index,
                "session_id": slot["session_id"],
                "acquisition_path": slot["acquisition_path"],
                "acquisition_sha256": acquisition[
                    "tuning_lease_session_acquisition_sha256"
                ],
                "candidate_slot_index": int(assigned_slot["slot_index"]),
                "candidate_attempt_id": assigned_slot["attempt_id"],
                "candidate_id": assigned_slot["candidate_id"],
                "session_hmac_commitment_sha256": acquisition[
                    "session_hmac_commitment_sha256"
                ],
                "global_active_event_sha256": acquisition[
                    "global_active_event_sha256"
                ],
                "global_history_file_sha256_at_release": acquisition[
                    "global_history_file_sha256_at_acquisition"
                ],
                "global_record_file_sha256_at_release": acquisition[
                    "global_record_file_sha256_at_acquisition"
                ],
                "active_uuid_mutex_verified": True,
                **inventory,
            }
            for key, value in expected_release.items():
                if release.get(key) != value:
                    raise PermissionError(
                        f"Tuning lease session release mismatch for {key}"
                    )
            try:
                secret = bytes.fromhex(
                    str(release.get("session_hmac_secret_hex", ""))
                )
            except ValueError as exc:
                raise PermissionError(
                    "Tuning lease session release secret is invalid"
                ) from exc
            if (
                len(secret) != 32
                or hashlib.sha256(secret).hexdigest()
                != acquisition["session_hmac_commitment_sha256"]
            ):
                raise PermissionError(
                    "Tuning lease session release secret/commitment mismatch"
                )
            _verify_tuning_session_artifact_hmac(
                release,
                self_hash_field="tuning_lease_session_release_sha256",
                secret=secret,
            )
            authorization_rows = inventory["issued_authorizations"]
            completion_rows = inventory["completion_receipts"]
            if (
                len(authorization_rows) not in {0, 1}
                or len(completion_rows) not in {0, 1}
                or len(authorization_rows) != len(completion_rows)
                or (
                    authorization_rows
                    and int(authorization_rows[0]["slot_index"])
                    != int(assigned_slot["slot_index"])
                )
                or (
                    completion_rows
                    and int(completion_rows[0]["slot_index"])
                    != int(assigned_slot["slot_index"])
                )
            ):
                raise PermissionError(
                    "Tuning lease session evidence set is not its one assigned candidate"
                )
            for evidence_row, path_key, hash_field in (
                (
                    authorization_rows,
                    "authorization_token_path",
                    "tuning_lease_authorization_sha256",
                ),
                (
                    completion_rows,
                    "completion_receipt_path",
                    "tuning_lease_completion_sha256",
                ),
            ):
                if not evidence_row:
                    continue
                payload = _load_hashed(assigned_slot[path_key], hash_field)
                if (
                    payload.get("session_hmac_commitment_sha256")
                    != acquisition["session_hmac_commitment_sha256"]
                    or payload.get("lease_session_id") != slot["session_id"]
                ):
                    raise PermissionError(
                        "Tuning session evidence HMAC commitment is divergent"
                    )
                _verify_tuning_session_artifact_hmac(
                    payload, self_hash_field=hash_field, secret=secret
                )
            previous_event_sha256 = release[
                "tuning_lease_session_release_sha256"
            ]
        elif index < int(session_index):
            raise PermissionError(
                "Tuning lease session chain has an unreleased prior session"
            )
        else:
            previous_event_sha256 = acquisition[
                "tuning_lease_session_acquisition_sha256"
            ]
        selected_acquisition = acquisition
        selected_release = release
    assert selected_acquisition is not None
    return selected_acquisition, selected_release


def _parse_tuning_idle_compute_rows(
    stdout: Any, gpu_uuid: str
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    seen_pids: set[int] = set()
    for fields in csv.reader(io.StringIO(str(stdout)), skipinitialspace=True):
        fields = [item.strip() for item in fields]
        if not fields or all(not item for item in fields):
            continue
        if len(fields) != 4:
            raise PermissionError(
                "Tuning idle-gate compute-process row is malformed"
            )
        try:
            pid = int(fields[0])
        except ValueError as exc:
            raise PermissionError(
                "Tuning idle-gate compute-process PID is invalid"
            ) from exc
        if (
            pid <= 0
            or pid in seen_pids
            or not fields[1]
            or not fields[2]
            or not GPU_UUID_PATTERN.fullmatch(fields[3])
        ):
            raise PermissionError(
                "Tuning idle-gate compute-process row is invalid"
            )
        if fields[3].lower() == gpu_uuid.lower():
            seen_pids.add(pid)
            parsed.append(
                {
                    "pid": pid,
                    "nvidia_process_name": fields[1],
                    "used_gpu_memory": fields[2],
                    "gpu_uuid": fields[3],
                }
            )
    return sorted(parsed, key=lambda row: int(row["pid"]))


def _classify_tuning_wddm_processes(
    raw_processes: list[dict[str, Any]], resolutions: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute classifications from PID resolution evidence and fixed policy."""
    if not isinstance(resolutions, list) or len(resolutions) != len(raw_processes):
        raise PermissionError("Tuning idle-gate process resolutions are incomplete")
    by_pid: dict[int, str | None] = {}
    for row in resolutions:
        if not isinstance(row, dict) or set(row) != {
            "pid",
            "resolved_image",
            "resolution_source",
        }:
            raise PermissionError("Tuning idle-gate process resolution is invalid")
        try:
            pid = int(row["pid"])
        except (TypeError, ValueError) as exc:
            raise PermissionError(
                "Tuning idle-gate resolution PID is invalid"
            ) from exc
        if (
            pid in by_pid
            or row.get("resolution_source") != "os_openprocess_or_toolhelp"
        ):
            raise PermissionError("Tuning idle-gate process resolution is invalid")
        value = row.get("resolved_image")
        if value is None:
            resolved_image = None
        elif isinstance(value, str) and value.strip():
            resolved_image = value.strip().casefold()
            if Path(resolved_image).name != resolved_image:
                raise PermissionError(
                    "Tuning idle-gate resolved image must be a canonical basename"
                )
        else:
            raise PermissionError("Tuning idle-gate resolved image is invalid")
        by_pid[pid] = resolved_image
    if set(by_pid) != {int(row["pid"]) for row in raw_processes}:
        raise PermissionError("Tuning idle-gate process resolutions are divergent")
    classified: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for raw in raw_processes:
        resolved_image = by_pid[int(raw["pid"])]
        if resolved_image is None:
            classification = "conflict"
            reason = "unresolved_os_process_image"
        elif resolved_image in TUNING_WDDM_ALLOWED_IMAGES:
            classification = "allowed_wddm_client"
            reason = "resolved_image_in_frozen_wddm_allowlist"
        elif resolved_image in TUNING_EXPLICIT_GPU_CONFLICT_IMAGES:
            classification = "conflict"
            reason = "explicit_compute_or_training_process"
        else:
            classification = "conflict"
            reason = "resolved_image_not_in_frozen_wddm_allowlist"
        item = {
            **raw,
            "resolved_image": resolved_image,
            "classification": classification,
            "reason": reason,
        }
        classified.append(item)
        (allowed if classification == "allowed_wddm_client" else conflicts).append(
            item
        )
    return classified, allowed, conflicts


def _parse_shared_gpu_compute_rows(stdout: Any, gpu_uuid: str) -> list[dict[str, Any]]:
    """A PID may use both GPUs; only duplicate (PID, GPU UUID) rows conflict."""
    parsed: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for fields in csv.reader(io.StringIO(str(stdout)), skipinitialspace=True):
        fields = [value.strip() for value in fields]
        if not fields or all(not value for value in fields):
            continue
        if len(fields) != 4:
            raise PermissionError("Shared GPU compute-process row is malformed")
        try:
            pid = int(fields[0])
        except ValueError as error:
            raise PermissionError("Shared GPU compute-process PID is invalid") from error
        identity = (pid, fields[3].lower())
        if (pid <= 0 or not fields[1] or not fields[2]
                or not GPU_UUID_PATTERN.fullmatch(fields[3]) or identity in seen):
            raise PermissionError("Shared GPU compute-process identity is invalid or duplicated")
        seen.add(identity)
        if fields[3].lower() == gpu_uuid.lower():
            parsed.append({"pid": pid, "nvidia_process_name": fields[1],
                "used_gpu_memory": fields[2], "gpu_uuid": fields[3]})
    return sorted(parsed, key=lambda row: row["pid"])


def _validate_tuning_shared_gpu_memory_evidence(
    evidence: dict[str, Any], namespace: dict[str, Any]
) -> None:
    """Replay shared resource admission, not the legacy quiet-GPU assertion."""
    anchor = namespace.get("runtime_resource_policy")
    if not isinstance(anchor, dict) or anchor != _load_shared_gpu_policy_anchor(required=True):
        raise PermissionError("Shared GPU evidence lacks the namespace-bound live policy anchor")
    policy = anchor["policy"]
    gpu_uuid = namespace["gpu_identity"]["uuid"]
    samples = evidence.get("samples")
    if (evidence.get("schema_version") != 4
            or evidence.get("purpose") != "pre_candidate_shared_gpu_memory_admission"
            or evidence.get("gpu_uuid") != gpu_uuid or policy["gpu_uuid"] != gpu_uuid
            or evidence.get("runtime_resource_policy") != anchor
            or evidence.get("sample_count") != 3
            or evidence.get("dedicated_or_idle_gpu_claimed") is not False
            or evidence.get("utilization_is_admission_threshold") is not False
            or evidence.get("process_exclusion_by_basename") is not False
            or evidence.get("gate_passed") is not True
            or not isinstance(samples, list) or len(samples) != 3
            or evidence.get("samples_sha256") != canonical_json_hash(samples)):
        raise PermissionError("Invalid shared GPU admission evidence")
    interval = evidence.get("sample_interval_seconds")
    if (type(interval) not in (int, float) or not math.isfinite(float(interval))
            or interval < 0 or not str(evidence.get("checked_at_utc", "")).strip()):
        raise PermissionError("Invalid shared GPU sampling time evidence")
    source_sha = namespace["launcher_implementation"]["files"]["scripts/run_closed_loop_tuning.py"]
    if (evidence.get("idle_gate_implementation_sha256") != source_sha
            or sha256_file(PROJECT_ROOT / "scripts/run_closed_loop_tuning.py") != source_sha):
        raise PermissionError("Shared GPU admission implementation differs from namespace")
    trusted = namespace["trusted_nvidia_smi"]
    executable = Path(trusted["executable_path"]).resolve()
    if (executable != TRUSTED_NVIDIA_SMI_PATH.resolve()
            or sha256_file(executable) != trusted["executable_sha256"]):
        raise PermissionError("Shared GPU trusted NVSMI binary identity differs")
    expected_commands = {
        "compute_processes": [str(executable),
            "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid", "--format=csv,noheader,nounits"],
        "resources": [str(executable), f"--query-gpu={SHARED_GPU_MEMORY_QUERY}", "--format=csv,noheader,nounits"],
    }
    gpu_free, host_free, utilization = [], [], []
    for index, sample in enumerate(samples, 1):
        if not isinstance(sample, dict):
            raise PermissionError("Malformed shared GPU sample")
        core = {key: value for key, value in sample.items() if key != "sample_sha256"}
        if (sample.get("sample_index") != index or not sample.get("sampled_at_utc")
                or sample.get("sample_sha256") != canonical_json_hash(core)):
            raise PermissionError("Shared GPU sample integrity differs")
        snapshots = sample.get("nvidia_smi_snapshots", {})
        if set(snapshots) != set(expected_commands):
            raise PermissionError("Shared GPU raw snapshot set differs")
        for name, command in expected_commands.items():
            snapshot = snapshots[name]
            if not isinstance(snapshot, dict):
                raise PermissionError("Malformed shared GPU raw snapshot")
            snapshot_core = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
            if (snapshot.get("schema_version") != 1 or not snapshot.get("captured_at_utc")
                    or snapshot.get("command") != command
                    or snapshot.get("command_sha256") != canonical_json_hash(command)
                    or snapshot.get("snapshot_sha256") != canonical_json_hash(snapshot_core)
                    or snapshot.get("executable_sha256") != trusted["executable_sha256"]
                    or snapshot.get("returncode") != 0
                    or snapshot.get("stdout_sha256") != canonical_json_hash(str(snapshot.get("stdout", "")))
                    or snapshot.get("stderr_sha256") != canonical_json_hash(str(snapshot.get("stderr", "")))):
                raise PermissionError("Shared GPU raw NVSMI evidence is divergent")
        resources = _parse_shared_gpu_resource_snapshot(snapshots["resources"], gpu_uuid)
        if sample.get("resources") != resources:
            raise PermissionError("Shared GPU parsed resources diverge from raw NVSMI")
        raw_processes = _parse_shared_gpu_compute_rows(snapshots["compute_processes"].get("stdout", ""), gpu_uuid)
        if sample.get("raw_compute_processes") != raw_processes:
            raise PermissionError("Shared GPU process list diverges from raw NVSMI")
        resolutions = sample.get("process_image_resolutions")
        if (not isinstance(resolutions, list)
                or [item.get("pid") for item in resolutions] != [item["pid"] for item in raw_processes]
                or any(item.get("resolution_source") != "os_openprocess_or_toolhelp" for item in resolutions)):
            raise PermissionError("Shared GPU OS process resolutions are incomplete")
        for item in resolutions:
            image = item.get("resolved_image")
            if image is not None and (not isinstance(image, str) or not image
                    or image != Path(image).name.casefold()):
                raise PermissionError("Shared GPU OS image resolution is malformed")
        memory = sample.get("host_memory", {})
        total, available = memory.get("total_bytes"), memory.get("available_bytes")
        if (memory.get("source") != "GlobalMemoryStatusEx" or type(total) is not int
                or type(available) is not int or total <= 0 or not 0 <= available <= total):
            raise PermissionError("Shared GPU host-memory evidence is invalid")
        available_mib = available / (1024 * 1024)
        if (resources["memory_free_mib"] < policy["minimum_gpu_free_mib"]
                or available_mib < policy["minimum_host_free_mib"]):
            raise PermissionError("Shared GPU memory headroom is below the bound policy")
        gpu_free.append(resources["memory_free_mib"])
        host_free.append(available_mib)
        utilization.append(resources["utilization_percent"])
    if (evidence.get("gpu_free_samples_mib") != gpu_free
            or evidence.get("host_free_samples_mib") != host_free
            or evidence.get("utilization_samples_percent") != utilization):
        raise PermissionError("Shared GPU resource summary diverges from samples")


def _validate_tuning_idle_gate_evidence_v3(
    idle: dict[str, Any], namespace: dict[str, Any]
) -> None:
    """Independently replay three raw WDDM admission snapshots."""
    command_source = idle.get("command_source")
    samples = idle.get("samples")
    gpu_uuid = str(namespace.get("gpu_identity", {}).get("uuid", ""))
    if (
        idle.get("purpose") != "pre_candidate_gpu_idle_gate"
        or idle.get("gpu_uuid") != gpu_uuid
        or not str(idle.get("checked_at_utc", "")).strip()
        or idle.get("sample_count") != TUNING_IDLE_SAMPLE_COUNT
        or not isinstance(samples, list)
        or len(samples) != TUNING_IDLE_SAMPLE_COUNT
        or idle.get("samples_sha256") != canonical_json_hash(samples)
        or not isinstance(command_source, dict)
        or idle.get("command_source_sha256")
        != canonical_json_hash(command_source)
        or idle.get("allowed_wddm_image_allowlist")
        != sorted(TUNING_WDDM_ALLOWED_IMAGES)
        or len(str(idle.get("idle_gate_implementation_sha256", ""))) != 64
    ):
        raise PermissionError("Tuning authorization idle-gate evidence is invalid")
    try:
        interval = float(idle.get("sample_interval_seconds"))
    except (TypeError, ValueError) as exc:
        raise PermissionError("Tuning idle-gate sample interval is invalid") from exc
    if not math.isfinite(interval) or interval < 0.0:
        raise PermissionError("Tuning idle-gate sample interval is invalid")
    executable_path = Path(
        str(command_source.get("nvidia_smi_executable_path", ""))
    )
    executable_sha256 = str(
        command_source.get("nvidia_smi_executable_sha256", "")
    ).lower()
    trusted = namespace.get("trusted_nvidia_smi")
    if (
        not isinstance(trusted, dict)
        or not executable_path.is_absolute()
        or len(executable_sha256) != 64
        or str(executable_path.resolve()).casefold()
        != str(TRUSTED_NVIDIA_SMI_PATH.resolve()).casefold()
        or str(executable_path.resolve()).casefold()
        != str(Path(str(trusted.get("executable_path", ""))).resolve()).casefold()
        or executable_sha256
        != str(trusted.get("executable_sha256", "")).lower()
    ):
        raise PermissionError(
            "Tuning idle-gate command source is not the trusted executable"
        )
    try:
        live_executable_sha256 = sha256_file(executable_path.resolve()).lower()
    except OSError as exc:
        raise PermissionError(
            "Trusted nvidia-smi executable cannot be rehashed"
        ) from exc
    if live_executable_sha256 != executable_sha256:
        raise PermissionError("Trusted nvidia-smi executable byte hash drifted")
    expected_suffixes = {
        "compute_processes": [
            "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        "utilization": [
            "--query-gpu=index,uuid,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        "compute_mode": [
            "--query-gpu=index,uuid,compute_mode",
            "--format=csv,noheader,nounits",
        ],
    }
    all_raw: list[dict[str, Any]] = []
    all_classified: list[dict[str, Any]] = []
    all_allowed: list[dict[str, Any]] = []
    all_conflicts: list[dict[str, Any]] = []
    utilization_values: list[float] = []
    command_rows: list[dict[str, Any]] = []
    for expected_index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise PermissionError("Tuning idle-gate sample is invalid")
        sample_core = {key: value for key, value in sample.items() if key != "sample_sha256"}
        snapshots = sample.get("nvidia_smi_snapshots")
        if (
            sample.get("schema_version") != 1
            or sample.get("sample_index") != expected_index
            or not str(sample.get("sampled_at_utc", "")).strip()
            or sample.get("sample_sha256") != canonical_json_hash(sample_core)
            or not isinstance(snapshots, dict)
            or set(snapshots) != set(expected_suffixes)
            or sample.get("nvidia_smi_snapshots_sha256")
            != canonical_json_hash(snapshots)
        ):
            raise PermissionError("Tuning idle-gate sample is divergent")
        for name, suffix in expected_suffixes.items():
            snapshot = snapshots.get(name)
            if not isinstance(snapshot, dict):
                raise PermissionError("Tuning idle-gate raw snapshot is invalid")
            snapshot_core = {
                key: value for key, value in snapshot.items() if key != "snapshot_sha256"
            }
            command = snapshot.get("command")
            if (
                snapshot.get("schema_version") != 1
                or not str(snapshot.get("captured_at_utc", "")).strip()
                or snapshot.get("snapshot_sha256")
                != canonical_json_hash(snapshot_core)
                or command != [str(executable_path), *suffix]
                or snapshot.get("command_sha256") != canonical_json_hash(command)
                or snapshot.get("executable_sha256") != executable_sha256
                or int(snapshot.get("returncode", -1)) != 0
                or snapshot.get("stdout_sha256")
                != canonical_json_hash(str(snapshot.get("stdout", "")))
                or snapshot.get("stderr_sha256")
                != canonical_json_hash(str(snapshot.get("stderr", "")))
            ):
                raise PermissionError("Tuning idle-gate raw snapshot is divergent")
        row_commands = {
            "sample_index": expected_index,
            "compute_process_command_sha256": snapshots["compute_processes"][
                "command_sha256"
            ],
            "utilization_command_sha256": snapshots["utilization"][
                "command_sha256"
            ],
            "compute_mode_command_sha256": snapshots["compute_mode"][
                "command_sha256"
            ],
        }
        command_rows.append(row_commands)
        raw = _parse_tuning_idle_compute_rows(
            snapshots["compute_processes"].get("stdout", ""), gpu_uuid
        )
        classified, allowed, conflicts = _classify_tuning_wddm_processes(
            raw, sample.get("process_image_resolutions")
        )
        if (
            sample.get("raw_compute_processes") != raw
            or sample.get("classified_compute_processes") != classified
            or sample.get("allowed_wddm_clients") != allowed
            or sample.get("conflicts") != conflicts
        ):
            raise PermissionError(
                "Tuning idle-gate process classification is divergent"
            )
        utilization_rows = [
            [item.strip() for item in row]
            for row in csv.reader(
                io.StringIO(str(snapshots["utilization"].get("stdout", ""))),
                skipinitialspace=True,
            )
            if row
        ]
        if any(
            len(row) != 3
            or not row[0].isdigit()
            or not GPU_UUID_PATTERN.fullmatch(row[1])
            for row in utilization_rows
        ):
            raise PermissionError("Tuning idle-gate utilization row is malformed")
        utilization_matches = [
            row for row in utilization_rows if row[1].lower() == gpu_uuid.lower()
        ]
        try:
            utilization = (
                float(utilization_matches[0][2])
                if len(utilization_matches) == 1
                else math.inf
            )
            reported_utilization = float(sample.get("utilization_percent"))
        except (TypeError, ValueError):
            utilization = math.inf
            reported_utilization = math.inf
        if (
            not math.isfinite(utilization)
            or utilization < 0.0
            or utilization > 100.0
            or reported_utilization != utilization
        ):
            raise PermissionError("Tuning idle-gate utilization is invalid")
        mode_rows = [
            [item.strip() for item in row]
            for row in csv.reader(
                io.StringIO(str(snapshots["compute_mode"].get("stdout", ""))),
                skipinitialspace=True,
            )
            if row
        ]
        if any(
            len(row) != 3
            or not row[0].isdigit()
            or not GPU_UUID_PATTERN.fullmatch(row[1])
            or not row[2]
            for row in mode_rows
        ):
            raise PermissionError("Tuning idle-gate compute-mode row is malformed")
        mode_matches = [row for row in mode_rows if row[1].lower() == gpu_uuid.lower()]
        parsed_mode = mode_matches[0][2] if len(mode_matches) == 1 else None
        if parsed_mode != "Default" or sample.get("compute_mode") != parsed_mode:
            raise PermissionError("Tuning idle-gate compute mode is invalid")
        utilization_values.append(utilization)
        all_raw.extend({"sample_index": expected_index, **row} for row in raw)
        all_classified.extend(
            {"sample_index": expected_index, **row} for row in classified
        )
        all_allowed.extend({"sample_index": expected_index, **row} for row in allowed)
        all_conflicts.extend(
            {"sample_index": expected_index, **row} for row in conflicts
        )
    try:
        top_utilization = float(idle.get("utilization_percent"))
        top_maximum = float(idle.get("maximum_utilization_percent"))
    except (TypeError, ValueError) as exc:
        raise PermissionError("Tuning idle-gate utilization summary is invalid") from exc
    if (
        command_source.get("sample_command_sha256s") != command_rows
        or idle.get("raw_compute_processes") != all_raw
        or idle.get("compute_processes") != all_classified
        or idle.get("allowed_wddm_clients") != all_allowed
        or idle.get("conflicts") != all_conflicts
        or all_conflicts
        or idle.get("utilization_samples_percent") != utilization_values
        or not math.isfinite(top_utilization)
        or top_utilization != max(utilization_values)
        or not math.isfinite(top_maximum)
        or top_maximum != MAX_TUNING_IDLE_GPU_UTILIZATION_PERCENT
        or any(value > top_maximum for value in utilization_values)
        or idle.get("compute_mode") != "Default"
        or idle.get("compute_process_policy")
        != "os_resolved_fixed_wddm_allowlist_otherwise_conflict"
        or idle.get("gate_passed") is not True
    ):
        raise PermissionError("Tuning authorization idle-gate evidence is invalid")


def _validate_tuning_idle_gate_evidence(
    idle: Any, namespace: dict[str, Any]
) -> None:
    """Deep-check the raw launch-point nvidia-smi evidence embedded in a token."""
    if not isinstance(idle, dict):
        raise PermissionError("Tuning authorization idle-gate evidence is invalid")
    if "runtime_resource_policy" in namespace and idle.get("schema_version") != 4:
        raise PermissionError("Shared namespace requires shared resource evidence, not a legacy idle claim")
    if idle.get("schema_version") == 4:
        _validate_tuning_shared_gpu_memory_evidence(idle, namespace)
        return
    if idle.get("schema_version") == 3:
        _validate_tuning_idle_gate_evidence_v3(idle, namespace)
        return
    command_source = idle.get("command_source")
    snapshots = idle.get("nvidia_smi_snapshots")
    if (
        idle.get("schema_version") != 2
        or idle.get("purpose") != "pre_candidate_gpu_idle_gate"
        or idle.get("gpu_uuid") != namespace["gpu_identity"]["uuid"]
        or not str(idle.get("checked_at_utc", "")).strip()
        or not isinstance(command_source, dict)
        or idle.get("command_source_sha256")
        != canonical_json_hash(command_source)
        or not isinstance(snapshots, dict)
        or set(snapshots) != {
            "compute_processes",
            "utilization",
            "compute_mode",
        }
        or any(
            not isinstance(snapshots.get(name), dict)
            for name in ("compute_processes", "utilization", "compute_mode")
        )
        or command_source.get("compute_process_command_sha256")
        != snapshots["compute_processes"].get("command_sha256")
        or command_source.get("utilization_command_sha256")
        != snapshots["utilization"].get("command_sha256")
        or command_source.get("compute_mode_command_sha256")
        != snapshots["compute_mode"].get("command_sha256")
        or idle.get("nvidia_smi_snapshots_sha256")
        != canonical_json_hash(snapshots)
        or len(str(idle.get("idle_gate_implementation_sha256", ""))) != 64
    ):
        raise PermissionError("Tuning authorization idle-gate evidence is invalid")
    executable_path = Path(
        str(command_source.get("nvidia_smi_executable_path", ""))
    )
    executable_sha256 = str(
        command_source.get("nvidia_smi_executable_sha256", "")
    ).lower()
    trusted = namespace.get("trusted_nvidia_smi")
    if (
        not isinstance(trusted, dict)
        or not executable_path.is_absolute()
        or len(executable_sha256) != 64
        or str(executable_path.resolve()).casefold()
        != str(TRUSTED_NVIDIA_SMI_PATH.resolve()).casefold()
        or str(executable_path.resolve()).casefold()
        != str(Path(str(trusted.get("executable_path", ""))).resolve()).casefold()
        or executable_sha256
        != str(trusted.get("executable_sha256", "")).lower()
    ):
        raise PermissionError(
            "Tuning idle-gate command source is not the trusted executable"
        )
    try:
        live_executable_sha256 = sha256_file(executable_path.resolve()).lower()
    except OSError as exc:
        raise PermissionError(
            "Trusted nvidia-smi executable cannot be rehashed"
        ) from exc
    if live_executable_sha256 != executable_sha256:
        raise PermissionError("Trusted nvidia-smi executable byte hash drifted")
    expected_suffixes = {
        "compute_processes": [
            "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        "utilization": [
            "--query-gpu=index,uuid,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        "compute_mode": [
            "--query-gpu=index,uuid,compute_mode",
            "--format=csv,noheader,nounits",
        ],
    }
    for name, suffix in expected_suffixes.items():
        snapshot = snapshots[name]
        if not isinstance(snapshot, dict):
            raise PermissionError("Tuning idle-gate raw snapshot is invalid")
        snapshot_core = {
            key: value
            for key, value in snapshot.items()
            if key != "snapshot_sha256"
        }
        command = snapshot.get("command")
        if (
            snapshot.get("schema_version") != 1
            or not str(snapshot.get("captured_at_utc", "")).strip()
            or snapshot.get("snapshot_sha256")
            != canonical_json_hash(snapshot_core)
            or command != [str(executable_path), *suffix]
            or snapshot.get("command_sha256") != canonical_json_hash(command)
            or snapshot.get("executable_sha256") != executable_sha256
            or int(snapshot.get("returncode", -1)) != 0
            or snapshot.get("stdout_sha256")
            != canonical_json_hash(str(snapshot.get("stdout", "")))
            or snapshot.get("stderr_sha256")
            != canonical_json_hash(str(snapshot.get("stderr", "")))
        ):
            raise PermissionError("Tuning idle-gate raw snapshot is divergent")
    parsed_processes: list[dict[str, Any]] = []
    for fields in csv.reader(
        io.StringIO(str(snapshots["compute_processes"].get("stdout", ""))),
        skipinitialspace=True,
    ):
        fields = [item.strip() for item in fields]
        if not fields or all(not item for item in fields):
            continue
        if len(fields) != 4:
            raise PermissionError(
                "Tuning idle-gate compute-process row is malformed"
            )
        try:
            pid = int(fields[0])
        except ValueError as exc:
            raise PermissionError(
                "Tuning idle-gate compute-process PID is invalid"
            ) from exc
        if (
            pid <= 0
            or not fields[1]
            or not fields[2]
            or not GPU_UUID_PATTERN.fullmatch(fields[3])
        ):
            raise PermissionError(
                "Tuning idle-gate compute-process row is invalid"
            )
        if fields[3].lower() == str(
            namespace["gpu_identity"]["uuid"]
        ).lower():
            parsed_processes.append(
                {
                    "pid": pid,
                    "nvidia_process_name": fields[1],
                    "used_gpu_memory": fields[2],
                    "gpu_uuid": fields[3],
                }
            )
    parsed_processes.sort(key=lambda row: int(row["pid"]))
    process_evidence = idle.get("compute_processes")
    utilization_rows = [
        [item.strip() for item in row]
        for row in csv.reader(
            io.StringIO(str(snapshots["utilization"].get("stdout", ""))),
            skipinitialspace=True,
        )
        if row
    ]
    if any(
        len(row) != 3
        or not row[0].isdigit()
        or not GPU_UUID_PATTERN.fullmatch(row[1])
        for row in utilization_rows
    ):
        raise PermissionError("Tuning idle-gate utilization row is malformed")
    utilization_matches = [
        row
        for row in utilization_rows
        if row[1].lower()
        == str(namespace["gpu_identity"]["uuid"]).lower()
    ]
    try:
        parsed_utilization = (
            float(utilization_matches[0][2])
            if len(utilization_matches) == 1
            else math.inf
        )
    except ValueError:
        parsed_utilization = math.inf
    compute_mode_rows = [
        [item.strip() for item in row]
        for row in csv.reader(
            io.StringIO(str(snapshots["compute_mode"].get("stdout", ""))),
            skipinitialspace=True,
        )
        if row
    ]
    if any(
        len(row) != 3
        or not row[0].isdigit()
        or not GPU_UUID_PATTERN.fullmatch(row[1])
        or not row[2]
        for row in compute_mode_rows
    ):
        raise PermissionError("Tuning idle-gate compute-mode row is malformed")
    compute_mode_matches = [
        row
        for row in compute_mode_rows
        if row[1].lower()
        == str(namespace["gpu_identity"]["uuid"]).lower()
    ]
    parsed_compute_mode = (
        compute_mode_matches[0][2]
        if len(compute_mode_matches) == 1
        else None
    )
    if (
        parsed_processes != []
        or process_evidence != []
        or idle.get("conflicts") != []
        or idle.get("gate_passed") is not True
        or float(idle.get("maximum_utilization_percent", -1.0)) != 5.0
        or not math.isfinite(parsed_utilization)
        or parsed_utilization < 0.0
        or parsed_utilization > 100.0
        or float(idle.get("utilization_percent", math.inf))
        != parsed_utilization
        or float(idle.get("utilization_percent", math.inf))
        > float(idle.get("maximum_utilization_percent", -1.0))
        or parsed_compute_mode != "Default"
        or idle.get("compute_mode") != parsed_compute_mode
    ):
        raise PermissionError("Tuning authorization idle-gate evidence is invalid")


def validate_tuning_lease_authorization(
    config: dict[str, Any],
    namespace: dict[str, Any],
    slot: dict[str, Any],
    *,
    authorization_path: str | Path,
    expected_authorization_sha256: str,
    expected_authorization_file_sha256: str,
    custody_receipt_path: str | Path,
    expected_custody_receipt_sha256: str,
    expected_custody_receipt_file_sha256: str,
    namespace_file_sha256: str,
    require_active: bool = True,
    expected_worker_pid: int | None = None,
    expected_worker_capability_nonce_sha256: str | None = None,
    require_calling_worker: bool = True,
) -> dict[str, Any]:
    """Validate a slot authorization and, when requested, its live OS lease."""
    custody = validate_tuning_lease_custody_receipt(
        config,
        namespace,
        receipt_path=custody_receipt_path,
        expected_receipt_sha256=expected_custody_receipt_sha256,
        expected_receipt_file_sha256=expected_custody_receipt_file_sha256,
        namespace_file_sha256=namespace_file_sha256,
    )
    validate_terminal_tuning_failure(namespace, custody)
    if require_active and (
        expected_worker_pid is None
        or expected_worker_capability_nonce_sha256 is None
    ):
        raise PermissionError(
            "Active authorization requires an explicit single-worker capability"
        )
    source = _resolve(authorization_path)
    if source != Path(slot["authorization_token_path"]).resolve():
        raise PermissionError("Tuning authorization path differs from its slot")
    if (
        len(str(expected_authorization_file_sha256)) != 64
        or not source.is_file()
        or sha256_file(source) != str(expected_authorization_file_sha256)
    ):
        raise PermissionError("Tuning authorization file anchor mismatch")
    token = _load_hashed(source, "tuning_lease_authorization_sha256")
    if token.get("tuning_lease_authorization_sha256") != str(
        expected_authorization_sha256
    ):
        raise PermissionError("Tuning authorization canonical anchor mismatch")
    expected = {
        "schema_version": 1,
        "purpose": "authorize_one_validation_tuning_candidate",
        "status": "authorized",
        "tuning_namespace_sha256": namespace["tuning_namespace_sha256"],
        "tuning_namespace_file_sha256": str(namespace_file_sha256),
        "namespace_instance_id": namespace["namespace_instance_id"],
        "tuning_lease_custody_receipt_sha256": custody[
            "tuning_lease_custody_receipt_sha256"
        ],
        "tuning_lease_custody_receipt_file_sha256": custody[
            "_validated_file_sha256"
        ],
        "slot_index": int(slot["slot_index"]),
        "attempt_id": slot["attempt_id"],
        "candidate_id": slot["candidate_id"],
        "output_dir": slot["output_dir"],
        "result_path": slot["result_path"],
        "authorization_token_path": slot["authorization_token_path"],
        "completion_receipt_path": slot["completion_receipt_path"],
        "worker_ready_path": slot["worker_ready_path"],
        "worker_launch_path": slot["worker_launch_path"],
        "output_reservation_path": slot["output_reservation_path"],
        "gpu_identity": namespace["gpu_identity"],
    }
    for key, value in expected.items():
        if token.get(key) != value:
            raise PermissionError(f"Tuning authorization mismatch for {key}")
    worker_pid = token.get("authorized_worker_pid")
    worker_nonce_sha256 = str(
        token.get("worker_capability_nonce_sha256", "")
    )
    if (
        not isinstance(worker_pid, int)
        or worker_pid <= 0
        or len(worker_nonce_sha256) != 64
        or (
            expected_worker_pid is not None
            and worker_pid != int(expected_worker_pid)
        )
        or (
            expected_worker_capability_nonce_sha256 is not None
            and worker_nonce_sha256
            != str(expected_worker_capability_nonce_sha256)
        )
        or (
            require_active
            and require_calling_worker
            and os.getpid() != worker_pid
        )
    ):
        raise PermissionError("Tuning authorization worker capability mismatch")
    ready_path = Path(slot["worker_ready_path"])
    reservation_path = Path(slot["output_reservation_path"])
    if not ready_path.is_file() or not reservation_path.is_file():
        raise PermissionError("Tuning authorization worker/reservation proof is absent")
    ready = _load_hashed(ready_path, "tuning_worker_ready_sha256")
    reservation = _load_hashed(
        reservation_path, "tuning_output_reservation_sha256"
    )
    ready_expected = {
        "schema_version": 1,
        "purpose": "single_tuning_candidate_worker_ready",
        "tuning_namespace_sha256": namespace["tuning_namespace_sha256"],
        "namespace_instance_id": namespace["namespace_instance_id"],
        "slot_index": int(slot["slot_index"]),
        "attempt_id": slot["attempt_id"],
        "worker_pid": worker_pid,
        "worker_capability_nonce_sha256": worker_nonce_sha256,
        "worker_ready_path": slot["worker_ready_path"],
        "worker_launch_path": slot["worker_launch_path"],
    }
    reservation_expected = {
        "schema_version": 1,
        "purpose": "exclusive_tuning_candidate_output_reservation",
        "status": "reserved",
        "tuning_namespace_sha256": namespace["tuning_namespace_sha256"],
        "namespace_instance_id": namespace["namespace_instance_id"],
        "slot_index": int(slot["slot_index"]),
        "attempt_id": slot["attempt_id"],
        "candidate_id": slot["candidate_id"],
        "output_dir": slot["output_dir"],
        "reservation_path": slot["output_reservation_path"],
        "worker_capability_nonce_sha256": worker_nonce_sha256,
    }
    for key, value in ready_expected.items():
        if ready.get(key) != value:
            raise PermissionError(f"Tuning worker-ready mismatch for {key}")
    for key, value in reservation_expected.items():
        if reservation.get(key) != value:
            raise PermissionError(f"Tuning output reservation mismatch for {key}")
    token_proof_expected = {
        "worker_ready_sha256": ready["tuning_worker_ready_sha256"],
        "worker_ready_file_sha256": sha256_file(ready_path),
        "output_reservation_sha256": reservation[
            "tuning_output_reservation_sha256"
        ],
        "output_reservation_file_sha256": sha256_file(reservation_path),
        "output_directory_device": reservation.get("output_directory_device"),
        "output_directory_inode": reservation.get("output_directory_inode"),
    }
    for key, value in token_proof_expected.items():
        if token.get(key) != value:
            raise PermissionError(f"Tuning authorization proof mismatch for {key}")
    output_directory = Path(slot["output_dir"])
    if not output_directory.is_dir():
        raise PermissionError("Tuning reserved output directory is absent")
    output_stat = output_directory.stat()
    if (
        int(reservation.get("output_directory_device", -1))
        != int(output_stat.st_dev)
        or int(reservation.get("output_directory_inode", -1))
        != int(output_stat.st_ino)
    ):
        raise PermissionError("Tuning reserved output directory inode changed")
    session_index = int(token.get("lease_session_index", -1))
    acquisition, release = _validate_tuning_session_chain(
        namespace, session_index=session_index, custody=custody
    )
    session_slot = namespace["lease_session_slots"][session_index - 1]
    session_expected = {
        "lease_session_id": session_slot["session_id"],
        "lease_session_acquisition_path": session_slot["acquisition_path"],
        "lease_session_release_path": session_slot["release_path"],
        "lease_session_acquisition_sha256": acquisition[
            "tuning_lease_session_acquisition_sha256"
        ],
        "lease_session_acquisition_file_sha256": sha256_file(
            session_slot["acquisition_path"]
        ),
        "lease_owner_pid": int(acquisition.get("owner_pid", -1)),
        "lease_owner_session_id": acquisition.get("owner_session_id"),
        "global_active_event_sha256": acquisition.get(
            "global_active_event_sha256"
        ),
        "global_history_file_sha256_at_issue": acquisition.get(
            "global_history_file_sha256_at_acquisition"
        ),
        "global_record_file_sha256_at_issue": acquisition.get(
            "global_record_file_sha256_at_acquisition"
        ),
        "session_hmac_commitment_sha256": acquisition.get(
            "session_hmac_commitment_sha256"
        ),
    }
    for key, value in session_expected.items():
        if token.get(key) != value:
            raise PermissionError(
                f"Tuning authorization session mismatch for {key}"
            )
    if (
        int(acquisition.get("candidate_slot_index", -1))
        != int(slot["slot_index"])
        or acquisition.get("candidate_attempt_id") != slot["attempt_id"]
        or acquisition.get("candidate_id") != slot["candidate_id"]
        or len(str(token.get("session_hmac_sha256", ""))) != 64
    ):
        raise PermissionError(
            "Tuning authorization is not authenticated for its assigned session candidate"
        )
    idle = token.get("idle_gate_evidence")
    if token.get("idle_gate_evidence_sha256") != canonical_json_hash(idle):
        raise PermissionError("Tuning authorization idle-gate hash is invalid")
    _validate_tuning_idle_gate_evidence(idle, namespace)
    history, active = _load_tuning_global_lease_state(namespace)
    historical_events = [
        event
        for event in history["events"]
        if event.get("event_sha256") == token.get("global_active_event_sha256")
    ]
    if (
        len(historical_events) != 1
        or historical_events[0].get("status") != "active"
        or historical_events[0].get("owner_pid") != token.get("lease_owner_pid")
        or historical_events[0].get("tuning_namespace_sha256")
        != namespace["tuning_namespace_sha256"]
        or historical_events[0].get("namespace_instance_id")
        != namespace["namespace_instance_id"]
        or historical_events[0].get("gpu_identity") != namespace["gpu_identity"]
    ):
        raise PermissionError(
            "Tuning authorization lease acquisition is absent from current history"
        )
    if release is not None:
        globally_closed = any(
            event.get("status") == "released"
            and event.get("acquired_event_sha256")
            == token.get("global_active_event_sha256")
            for event in history["events"]
        ) or any(
            (event.get("crash_recovery") or {}).get("prior_event_sha256")
            == token.get("global_active_event_sha256")
            for event in history["events"]
        )
        still_same_active_head = (
            active.get("status") == "active"
            and active.get("event_sha256")
            == token.get("global_active_event_sha256")
        )
        if not globally_closed and not still_same_active_head:
            raise PermissionError(
                "Released tuning session lacks its global lease closure event"
            )
    if require_active:
        if release is not None:
            raise PermissionError("Tuning authorization session is already released")
        if (
            active.get("status") != "active"
            or active.get("event_sha256")
            != token.get("global_active_event_sha256")
            or active.get("owner_pid") != token.get("lease_owner_pid")
            or active.get("tuning_namespace_sha256")
            != namespace["tuning_namespace_sha256"]
            or active.get("namespace_instance_id")
            != namespace["namespace_instance_id"]
            or active.get("gpu_identity") != namespace["gpu_identity"]
            or sha256_file(namespace["lease_custody_spec"]["global_history_path"])
            != token.get("global_history_file_sha256_at_issue")
            or sha256_file(namespace["lease_custody_spec"]["global_record_path"])
            != token.get("global_record_file_sha256_at_issue")
        ):
            raise PermissionError("Tuning authorization global lease is not active")
        if not _tuning_pid_is_running(int(token.get("lease_owner_pid", -1))):
            raise PermissionError("Tuning authorization lease owner is not running")
        if not _tuning_mutex_is_held(
            namespace["lease_custody_spec"]["global_mutex_path"]
        ):
            raise PermissionError("Tuning authorization UUID mutex is not held")
        if history["events"][-1] != active:
            raise PermissionError("Tuning authorization global lease head changed")
    return {
        **token,
        "_validated_path": str(source),
        "_validated_file_sha256": str(expected_authorization_file_sha256),
        "_validated_custody": custody,
        "_validated_session_acquisition": acquisition,
        "_validated_session_release": release,
    }


def validate_tuning_lease_completion(
    config: dict[str, Any],
    namespace: dict[str, Any],
    slot: dict[str, Any],
    row: dict[str, Any],
    *,
    custody_receipt_path: str | Path,
    expected_custody_receipt_sha256: str,
    expected_custody_receipt_file_sha256: str,
    namespace_file_sha256: str,
    require_released: bool = True,
) -> dict[str, Any]:
    """Deeply bind one completed row to its precommitted lease evidence."""
    custody = validate_tuning_lease_custody_receipt(
        config,
        namespace,
        receipt_path=custody_receipt_path,
        expected_receipt_sha256=expected_custody_receipt_sha256,
        expected_receipt_file_sha256=expected_custody_receipt_file_sha256,
        namespace_file_sha256=namespace_file_sha256,
    )
    validate_terminal_tuning_failure(namespace, custody)
    completion_path = Path(slot["completion_receipt_path"]).resolve()
    if not completion_path.is_file():
        raise PermissionError("Tuning slot lacks its completion receipt")
    completion = _load_hashed(
        completion_path, "tuning_lease_completion_sha256"
    )
    token = validate_tuning_lease_authorization(
        config,
        namespace,
        slot,
        authorization_path=slot["authorization_token_path"],
        expected_authorization_sha256=str(
            completion.get("tuning_lease_authorization_sha256", "")
        ),
        expected_authorization_file_sha256=str(
            completion.get("tuning_lease_authorization_file_sha256", "")
        ),
        custody_receipt_path=custody_receipt_path,
        expected_custody_receipt_sha256=expected_custody_receipt_sha256,
        expected_custody_receipt_file_sha256=(
            expected_custody_receipt_file_sha256
        ),
        namespace_file_sha256=namespace_file_sha256,
        require_active=False,
    )
    if token.get("_validated_session_release") is None and require_released:
        raise PermissionError(
            "Tuning completion is not sealed by an authenticated session release"
        )
    if token.get("_validated_session_release") is None:
        token = validate_tuning_lease_authorization(
            config,
            namespace,
            slot,
            authorization_path=slot["authorization_token_path"],
            expected_authorization_sha256=str(
                completion.get("tuning_lease_authorization_sha256", "")
            ),
            expected_authorization_file_sha256=str(
                completion.get("tuning_lease_authorization_file_sha256", "")
            ),
            custody_receipt_path=custody_receipt_path,
            expected_custody_receipt_sha256=expected_custody_receipt_sha256,
            expected_custody_receipt_file_sha256=(
                expected_custody_receipt_file_sha256
            ),
            namespace_file_sha256=namespace_file_sha256,
            require_active=True,
            expected_worker_pid=int(token["authorized_worker_pid"]),
            expected_worker_capability_nonce_sha256=token[
                "worker_capability_nonce_sha256"
            ],
            require_calling_worker=False,
        )
    row_core = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "tuning_candidate_sha256",
            "created_at_utc",
            "selection_validation_receipt",
            "tuning_lease_completion_receipt",
        }
    }
    expected = {
        "schema_version": 1,
        "purpose": "complete_one_validation_tuning_candidate",
        "status": "complete",
        "validated_while_lease_active": True,
        "tuning_namespace_sha256": namespace["tuning_namespace_sha256"],
        "tuning_namespace_file_sha256": str(namespace_file_sha256),
        "namespace_instance_id": namespace["namespace_instance_id"],
        "slot_index": int(slot["slot_index"]),
        "attempt_id": slot["attempt_id"],
        "candidate_id": slot["candidate_id"],
        "authorization_token_path": slot["authorization_token_path"],
        "tuning_lease_authorization_sha256": token[
            "tuning_lease_authorization_sha256"
        ],
        "tuning_lease_authorization_file_sha256": token[
            "_validated_file_sha256"
        ],
        "lease_session_id": token["lease_session_id"],
        "lease_session_index": int(token["lease_session_index"]),
        "lease_session_acquisition_sha256": token[
            "lease_session_acquisition_sha256"
        ],
        "session_hmac_commitment_sha256": token[
            "session_hmac_commitment_sha256"
        ],
        "global_active_event_sha256": token["global_active_event_sha256"],
        "idle_gate_evidence_sha256": token["idle_gate_evidence_sha256"],
        "authorized_worker_pid": token["authorized_worker_pid"],
        "worker_capability_nonce_sha256": token[
            "worker_capability_nonce_sha256"
        ],
        "worker_ready_sha256": token["worker_ready_sha256"],
        "output_reservation_sha256": token["output_reservation_sha256"],
        "candidate_result_path": slot["result_path"],
        "tuning_candidate_sha256": row.get("tuning_candidate_sha256"),
        "tuning_candidate_file_sha256": sha256_file(slot["result_path"]),
        "checkpoint_path": row.get("checkpoint_path"),
        "checkpoint_sha256": row.get("checkpoint_sha256"),
        "candidate_row_core_sha256": canonical_json_hash(row_core),
        "gpu_identity": namespace["gpu_identity"],
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise PermissionError(f"Tuning completion mismatch for {key}")
    if len(str(completion.get("session_hmac_sha256", ""))) != 64:
        raise PermissionError("Tuning completion lacks its session HMAC")
    launch_path = Path(slot["worker_launch_path"])
    if not launch_path.is_file():
        raise PermissionError("Tuning completion lacks its worker-launch receipt")
    launch = _load_hashed(launch_path, "tuning_worker_launch_sha256")
    if (
        completion.get("worker_launch_sha256")
        != launch.get("tuning_worker_launch_sha256")
        or completion.get("worker_launch_file_sha256")
        != sha256_file(launch_path)
        or launch.get("worker_pid") != token.get("authorized_worker_pid")
        or launch.get("worker_capability_nonce_sha256")
        != token.get("worker_capability_nonce_sha256")
        or launch.get("tuning_lease_authorization_sha256")
        != token.get("tuning_lease_authorization_sha256")
    ):
        raise PermissionError("Tuning completion worker-launch proof is invalid")
    row_expected = {
        "tuning_lease_authorization_path": slot["authorization_token_path"],
        "tuning_lease_authorization_sha256": token[
            "tuning_lease_authorization_sha256"
        ],
        "tuning_lease_authorization_file_sha256": token[
            "_validated_file_sha256"
        ],
        "tuning_lease_custody_receipt_sha256": token[
            "tuning_lease_custody_receipt_sha256"
        ],
        "tuning_lease_custody_receipt_file_sha256": token[
            "tuning_lease_custody_receipt_file_sha256"
        ],
        "tuning_lease_session_id": token["lease_session_id"],
        "tuning_lease_session_index": int(token["lease_session_index"]),
        "tuning_lease_session_acquisition_sha256": token[
            "lease_session_acquisition_sha256"
        ],
        "tuning_lease_global_active_event_sha256": token[
            "global_active_event_sha256"
        ],
        "tuning_idle_gate_evidence_sha256": token[
            "idle_gate_evidence_sha256"
        ],
        "tuning_authorized_worker_pid": token["authorized_worker_pid"],
        "tuning_worker_capability_nonce_sha256": token[
            "worker_capability_nonce_sha256"
        ],
        "tuning_worker_ready_sha256": token["worker_ready_sha256"],
        "tuning_output_reservation_sha256": token[
            "output_reservation_sha256"
        ],
        "tuning_output_reservation_file_sha256": token[
            "output_reservation_file_sha256"
        ],
    }
    for key, value in row_expected.items():
        if row.get(key) != value:
            raise PermissionError(f"Tuning candidate lease proof mismatch for {key}")
    return {
        **completion,
        "_validated_path": str(completion_path),
        "_validated_file_sha256": sha256_file(completion_path),
        "_validated_authorization": token,
    }


def _validate_tuning_candidate_checkpoint(
    row: dict[str, Any],
    *,
    config: dict[str, Any],
    registry: dict[str, Any],
    trajectory: dict[str, Any],
    design: dict[str, Any],
    namespace: dict[str, Any] | None = None,
    namespace_file_sha256: str | None = None,
) -> None:
    """Deeply bind a self-hashed candidate row to its runner checkpoint."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training requires torch
        raise RuntimeError("Tuning checkpoint validation requires PyTorch") from exc

    checkpoint_path = _resolve(row["checkpoint_path"])
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != row.get(
        "checkpoint_sha256"
    ):
        raise ValueError("Tuning candidate checkpoint hash mismatch")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise ValueError("Tuning candidate checkpoint is not loadable") from exc
    source_config_sha256 = canonical_json_hash(config)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict) or canonical_json_hash(
        checkpoint_config
    ) != source_config_sha256:
        raise ValueError("Tuning checkpoint immutable config mismatch")
    expected = {
        "run_phase": "validation_tuning",
        "registered_name": row.get("registered_name"),
        "seed": int(row.get("seed", -1)),
        "config_sha256": source_config_sha256,
        "parameter_registry_source_config_sha256": source_config_sha256,
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "parameter_registry_implementation_sha256": registry["implementation"][
            "aggregate_sha256"
        ],
        "tuning_design_sha256": design["tuning_design_sha256"],
        "environment_sha256": design["environment_sha256"],
        "controller_selection_sha256": design["controller_selection_sha256"],
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Tuning checkpoint identity mismatch for {key}")
        row_value = row.get(key)
        if key in {
            "config_sha256",
            "trajectory_manifest_sha256",
            "parameter_registry_sha256",
            "environment_sha256",
            "controller_selection_sha256",
        } and row_value != value:
            raise ValueError(f"Tuning candidate row identity mismatch for {key}")
    if namespace is not None:
        result_path = str(_resolve(row.get("tuning_candidate_result_path", "")))
        slots = [
            slot
            for slot in namespace.get("candidate_slots", [])
            if slot.get("result_path") == result_path
        ]
        if len(slots) != 1:
            raise ValueError("Tuning candidate result path is not a unique namespace slot")
        slot = slots[0]
        namespace_expected = {
            "tuning_namespace_sha256": namespace["tuning_namespace_sha256"],
            "tuning_namespace_file_sha256": namespace_file_sha256,
            "tuning_namespace_instance_id": namespace["namespace_instance_id"],
            "tuning_attempt_id": slot["attempt_id"],
            "tuning_slot_index": int(slot["slot_index"]),
            "tuning_candidate_id": slot["candidate_id"],
            "tuning_candidate_output_dir": slot["output_dir"],
            "tuning_candidate_result_path": slot["result_path"],
            "tuning_gpu_identity": namespace["gpu_identity"],
            "tuning_trusted_nvidia_smi": namespace[
                "trusted_nvidia_smi"
            ],
        }
        for key, value in namespace_expected.items():
            if row.get(key) != value:
                raise ValueError(f"Tuning candidate namespace mismatch for {key}")
            if checkpoint.get(key) != value:
                raise ValueError(f"Tuning checkpoint namespace mismatch for {key}")
        lease_proof_keys = {
            "tuning_lease_authorization_path",
            "tuning_lease_authorization_sha256",
            "tuning_lease_authorization_file_sha256",
            "tuning_lease_custody_receipt_sha256",
            "tuning_lease_custody_receipt_file_sha256",
            "tuning_lease_session_id",
            "tuning_lease_session_index",
            "tuning_lease_session_acquisition_sha256",
            "tuning_lease_global_active_event_sha256",
            "tuning_idle_gate_evidence_sha256",
            "tuning_authorized_worker_pid",
            "tuning_worker_capability_nonce_sha256",
            "tuning_worker_ready_sha256",
            "tuning_output_reservation_sha256",
            "tuning_output_reservation_file_sha256",
            "tuning_output_directory_device",
            "tuning_output_directory_inode",
        }
        for key in lease_proof_keys:
            if row.get(key) in {None, ""} or checkpoint.get(key) != row.get(key):
                raise ValueError(
                    f"Tuning checkpoint lease-proof mismatch for {key}"
                )
        if row.get("tuning_lease_authorization_path") != slot[
            "authorization_token_path"
        ]:
            raise ValueError("Tuning candidate authorization path differs from slot")
        if str(checkpoint_path.parent) != slot["output_dir"]:
            raise ValueError("Tuning checkpoint path differs from its namespace slot")
        validate_tuning_runtime_identity(
            checkpoint.get("runtime_device", {}), namespace["gpu_identity"]
        )
        if checkpoint.get("runtime_device", {}).get("trusted_nvidia_smi") != (
            namespace["trusted_nvidia_smi"]
        ):
            raise ValueError(
                "Tuning checkpoint trusted nvidia-smi identity mismatch"
            )
    if row.get("implementation_sha256") != registry["implementation"][
        "aggregate_sha256"
    ]:
        raise ValueError("Tuning candidate row implementation identity mismatch")
    optimizer = checkpoint.get("optimizer_hyperparameters", {})
    if checkpoint.get("optimizer_selection_source") != (
        "registered_validation_tuning_candidate"
    ):
        raise ValueError("Tuning checkpoint bypassed the registered candidate runner")
    for key in ("learning_rate", "weight_decay"):
        if float(optimizer.get(key, float("nan"))) != float(row[key]):
            raise ValueError(f"Tuning checkpoint optimizer mismatch for {key}")
    if float(checkpoint.get("best_val_loss", float("nan"))) != float(
        row.get("validation_loss", float("nan"))
    ):
        raise ValueError("Tuning checkpoint validation loss mismatch")
    planned = int(row.get("optimizer_steps_planned", -1))
    completed = int(row.get("optimizer_steps_completed", -1))
    registered_planned = design.get("planned_optimizer_steps")
    if registered_planned is not None and planned != int(registered_planned):
        raise ValueError("Tuning candidate planned-step budget mismatch")
    if planned <= 0 or completed != planned:
        raise ValueError("Tuning candidate did not complete its fixed optimizer budget")
    if (
        int(checkpoint.get("training_run_optimizer_steps_planned", -1)) != planned
        or int(checkpoint.get("training_run_optimizer_steps_completed", -1))
        != completed
        or checkpoint.get("training_run_fixed_budget_gate_passed") is not True
        or checkpoint.get("training_run_stopped_early") is not False
        or checkpoint.get("selection_finalized_after_training_run") is not True
    ):
        raise ValueError("Tuning checkpoint failed its finalized fixed-budget gate")
    if checkpoint.get("validation_split") != "val":
        raise ValueError("Tuning checkpoint was not selected on validation data")
    required_payload = {
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "normalizer",
        "parameter_entry",
        "data_sha256",
        "training_run_history",
    }
    missing_payload = sorted(required_payload - set(checkpoint))
    if missing_payload:
        raise ValueError(
            f"Tuning checkpoint lacks runner payload keys: {missing_payload}"
        )
    if checkpoint["parameter_entry"] != registry["entries"][row["registered_name"]]:
        raise ValueError("Tuning checkpoint parameter entry differs from registry")
    model = build_registered_model(row["registered_name"], config, registry)
    try:
        load_graph_safe_state_dict(
            model, checkpoint["model_state_dict"], strict=True
        )
    except Exception as exc:
        raise ValueError("Tuning checkpoint model state is not loadable") from exc
    optimizer_state = checkpoint["optimizer_state_dict"]
    if not isinstance(optimizer_state, dict) or not {
        "state",
        "param_groups",
    } <= set(optimizer_state):
        raise ValueError("Tuning checkpoint optimizer state is incomplete")
    normalizer = checkpoint["normalizer"]
    if not isinstance(normalizer, dict) or {
        "input_mean",
        "input_std",
        "target_mean",
        "target_std",
    } - set(normalizer):
        raise ValueError("Tuning checkpoint normalizer payload is incomplete")
    if len(normalizer["input_mean"]) != int(config.get("models", {}).get("input_dim", 19)):
        raise ValueError("Tuning checkpoint input normalizer dimension mismatch")
    if len(normalizer["target_mean"]) != int(config.get("models", {}).get("state_dim", 4)):
        raise ValueError("Tuning checkpoint target normalizer dimension mismatch")
    history = checkpoint["training_run_history"]
    if (
        not isinstance(history, list)
        or len(history) != int(design["epochs"])
        or int(history[-1].get("optimizer_steps_completed", -1)) != completed
        or int(history[-1].get("epoch", -1)) != int(design["epochs"]) - 1
    ):
        raise ValueError("Tuning checkpoint full training history is incomplete")
    from .training import _epoch_checkpoint_ledger

    ledger_identity = {
        "run_phase": "validation_tuning",
        "registered_name": row["registered_name"],
        "seed": int(row["seed"]),
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "config_sha256": source_config_sha256,
        **formal_artifact_identities(
            config,
            registry=registry,
            trajectory_manifest=trajectory,
            require_tuning=False,
        ),
        "tuning_design_sha256": design["tuning_design_sha256"],
    }
    ledger_path = checkpoint_path.parent / "checkpoint_ledger.json"
    if not ledger_path.is_file():
        raise ValueError("Tuning candidate lacks its immutable epoch ledger")
    stored_ledger = _load_hashed(ledger_path, "checkpoint_ledger_sha256")
    rebuilt_ledger = _epoch_checkpoint_ledger(
        checkpoint_path.parent / "epoch_checkpoints",
        expected_identity=ledger_identity,
    )
    if stored_ledger != rebuilt_ledger or int(
        rebuilt_ledger.get("checkpoint_count", -1)
    ) != int(design["epochs"]):
        raise ValueError("Tuning candidate epoch ledger is incomplete or divergent")
    winner = min(
        rebuilt_ledger["checkpoints"],
        key=lambda item: (float(item["validation_loss"]), int(item["epoch"])),
    )
    if (
        int(checkpoint.get("best_epoch", -1)) != int(winner["epoch"])
        or not math.isclose(
            float(checkpoint.get("best_val_loss", math.nan)),
            float(winner["validation_loss"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "Tuning candidate best checkpoint is not the immutable-ledger winner"
        )
    selection_receipt = row.get("selection_validation_receipt")
    if isinstance(selection_receipt, dict) and (
        selection_receipt.get("checkpoint_ledger_sha256")
        != rebuilt_ledger["checkpoint_ledger_sha256"]
        or selection_receipt.get("ledger_epoch_rows_sha256")
        != canonical_json_hash(rebuilt_ledger["checkpoints"])
    ):
        raise ValueError("Tuning validation receipt differs from its epoch ledger")


def _recompute_tuning_validation_loss(
    row: dict[str, Any],
    *,
    config: dict[str, Any],
    registry: dict[str, Any],
    data_root: str | Path,
) -> dict[str, Any]:
    """Independently reload selected weights and recompute frozen-val loss."""
    import torch

    from ..data.closed_loop import load_closed_loop_dataset
    from .training import (
        Normalizer,
        SequenceWindows,
        _epoch_checkpoint_ledger,
        _run_epoch,
    )

    checkpoint_path = _resolve(row["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arrays, dataset_manifest = load_closed_loop_dataset(_resolve(data_root) / "val")
    if checkpoint.get("data_sha256", {}).get("val") != dataset_manifest.get(
        "dataset_sha256"
    ):
        raise ValueError("Tuning checkpoint validation dataset hash mismatch")
    training = dict(config.get("training", {}))
    design = tuning_design_identity(
        config,
        registry,
        load_trajectory_manifest(
            _resolve(data_root) / "trajectory_manifest.json",
            expected_config=config,
        ),
    )
    ledger_identity = {
        "run_phase": "validation_tuning",
        "registered_name": row["registered_name"],
        "seed": int(row["seed"]),
        "trajectory_manifest_sha256": row["trajectory_manifest_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "config_sha256": canonical_json_hash(config),
        **formal_artifact_identities(
            config,
            registry=registry,
            trajectory_manifest=load_trajectory_manifest(
                _resolve(data_root) / "trajectory_manifest.json",
                expected_config=config,
            ),
            require_tuning=False,
        ),
        "tuning_design_sha256": design["tuning_design_sha256"],
    }
    ledger = _epoch_checkpoint_ledger(
        checkpoint_path.parent / "epoch_checkpoints",
        expected_identity=ledger_identity,
    )
    stored_ledger = _load_hashed(
        checkpoint_path.parent / "checkpoint_ledger.json",
        "checkpoint_ledger_sha256",
    )
    if stored_ledger != ledger:
        raise ValueError("Tuning stored epoch ledger differs from immutable files")
    replay_rows: list[dict[str, Any]] = []
    for epoch_row in ledger["checkpoints"]:
        epoch_path = Path(epoch_row["checkpoint_path"])
        epoch_payload = torch.load(epoch_path, map_location="cpu", weights_only=False)
        normalizer = Normalizer.from_dict(epoch_payload["normalizer"])
        windows = SequenceWindows(
            arrays,
            normalizer,
            int(training.get("window_steps", 64)),
            int(training.get("burn_in_steps", 0)),
        )
        model = build_registered_model(row["registered_name"], config, registry)
        load_graph_safe_state_dict(
            model, epoch_payload["model_state_dict"], strict=True
        )
        recomputed_loss, _ = _run_epoch(
            model,
            windows,
            batch_size=int(training.get("batch_size", 32)),
            device=torch.device("cpu"),
            optimizer=None,
            gradient_clip=float(training.get("gradient_clip_norm", 1.0)),
            order_seed=0,
        )
        recorded_epoch_loss = float(epoch_row["validation_loss"])
        epoch_tolerance = max(1e-7, 1e-5 * abs(recorded_epoch_loss))
        if not math.isclose(
            recomputed_loss,
            recorded_epoch_loss,
            rel_tol=1e-5,
            abs_tol=1e-7,
        ):
            raise ValueError(
                "Tuning epoch independently recomputed validation loss differs "
                f"at epoch {epoch_row['epoch']}: {recomputed_loss} vs "
                f"{recorded_epoch_loss}"
            )
        replay_rows.append(
            {
                "epoch": int(epoch_row["epoch"]),
                "epoch_checkpoint_sha256": epoch_row["checkpoint_sha256"],
                "recorded_validation_loss": recorded_epoch_loss,
                "recomputed_validation_loss": float(recomputed_loss),
                "acceptance_tolerance": float(epoch_tolerance),
            }
        )
    winner = min(
        replay_rows,
        key=lambda item: (
            float(item["recomputed_validation_loss"]),
            int(item["epoch"]),
        ),
    )
    recomputed = float(winner["recomputed_validation_loss"])
    recorded = float(checkpoint["best_val_loss"])
    tolerance = max(1e-7, 1e-5 * abs(recorded))
    if not math.isclose(recomputed, recorded, rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError(
            "Tuning checkpoint independently recomputed validation loss differs "
            f"from selected loss: {recomputed} vs {recorded}"
        )
    if int(checkpoint.get("best_epoch", -1)) != int(winner["epoch"]):
        raise ValueError("Tuning selected epoch is not the recomputed earliest minimum")
    return {
        "schema_version": 1,
        "recomputed_validation_loss": float(recomputed),
        "recorded_validation_loss": recorded,
        "absolute_error": float(abs(recomputed - recorded)),
        "acceptance_tolerance": float(tolerance),
        "validation_dataset_sha256": dataset_manifest["dataset_sha256"],
        "validation_windows": len(windows),
        "checkpoint_ledger_sha256": ledger["checkpoint_ledger_sha256"],
        "checkpoint_ledger_file_sha256": sha256_file(
            checkpoint_path.parent / "checkpoint_ledger.json"
        ),
        "ledger_epoch_rows_sha256": canonical_json_hash(
            ledger["checkpoints"]
        ),
        "per_epoch_replay": replay_rows,
        "per_epoch_replay_sha256": canonical_json_hash(replay_rows),
        "selected_epoch": int(winner["epoch"]),
        "selection_rule": "minimum_finite_validation_loss_then_earliest_epoch",
        "device": "cpu",
        "test_episodes_used": 0,
    }


def create_tuning_selection(
    config: dict[str, Any],
    registry_path: str | Path,
    data_root: str | Path,
    candidate_result_paths: list[str | Path],
    output_path: str | Path,
    *,
    tuning_namespace_path: str | Path | None = None,
    expected_tuning_namespace_sha256: str | None = None,
    expected_tuning_namespace_file_sha256: str | None = None,
    tuning_lease_custody_receipt_path: str | Path | None = None,
    expected_tuning_lease_custody_receipt_sha256: str | None = None,
    expected_tuning_lease_custody_receipt_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate complete seed-17 candidate evidence and freeze deterministic choices."""
    destination = _resolve(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite tuning selection: {destination}")
    registry = load_parameter_registry(registry_path, expected_config=config)
    trajectory = load_trajectory_manifest(
        _resolve(data_root) / "trajectory_manifest.json", expected_config=config
    )
    design = tuning_design_identity(config, registry, trajectory)
    namespace: dict[str, Any] | None = None
    namespace_file_sha256: str | None = None
    custody: dict[str, Any] | None = None
    if is_formal(config):
        if not all(
            (
                tuning_namespace_path,
                expected_tuning_namespace_sha256,
                expected_tuning_namespace_file_sha256,
                tuning_lease_custody_receipt_path,
                expected_tuning_lease_custody_receipt_sha256,
                expected_tuning_lease_custody_receipt_file_sha256,
            )
        ):
            raise PermissionError(
                "Formal tuning freeze requires externally supplied namespace anchors"
            )
        namespace_file_sha256 = str(expected_tuning_namespace_file_sha256)
        namespace = validate_tuning_namespace(
            config,
            registry,
            trajectory,
            namespace_path=str(tuning_namespace_path),
            expected_namespace_sha256=str(expected_tuning_namespace_sha256),
            expected_namespace_file_sha256=namespace_file_sha256,
        )
        custody = validate_tuning_lease_custody_receipt(
            config,
            namespace,
            receipt_path=str(tuning_lease_custody_receipt_path),
            expected_receipt_sha256=str(
                expected_tuning_lease_custody_receipt_sha256
            ),
            expected_receipt_file_sha256=str(
                expected_tuning_lease_custody_receipt_file_sha256
            ),
            namespace_file_sha256=namespace_file_sha256,
        )
        validate_terminal_tuning_failure(namespace, custody)
        exact_paths = [
            str(_resolve(path)) for path in candidate_result_paths
        ]
        registered_paths = [
            str(slot["result_path"]) for slot in namespace["candidate_slots"]
        ]
        if exact_paths != registered_paths:
            raise ValueError(
                "Formal tuning freeze requires the exact ordered 28-slot namespace"
            )
    tuning = dict(config.get("tuning", {}))
    tuning_seed = int(tuning.get("seed", 17))
    learning_rates = [float(value) for value in tuning.get("learning_rates", [3e-4, 1e-3])]
    weight_decays = [float(value) for value in tuning.get("weight_decays", [0.0, 1e-4])]
    expected_pairs = {(lr, wd) for lr in learning_rates for wd in weight_decays}
    rows = []
    for result_index, result_path in enumerate(candidate_result_paths):
        row = _load_hashed(result_path, "tuning_candidate_sha256")
        if row.get("producer") != "liquid_revision.closed_loop.training.train_tuning_candidate":
            raise ValueError("Tuning candidate was not emitted by the registered runner")
        if row.get("tuning_design_sha256") != design["tuning_design_sha256"]:
            raise ValueError("Tuning candidate design identity mismatch")
        if not row.get("validation_only") or int(row.get("test_episodes_used", -1)) != 0:
            raise ValueError("Tuning candidate touched test data")
        if int(row.get("seed", -1)) != tuning_seed:
            raise ValueError(
                f"Tuning seed mismatch for {row.get('registered_name', 'unknown')}"
            )
        _validate_tuning_candidate_checkpoint(
            row,
            config=config,
            registry=registry,
            trajectory=trajectory,
            design=design,
            namespace=namespace,
            namespace_file_sha256=namespace_file_sha256,
        )
        row = dict(row)
        if namespace is not None:
            assert custody is not None
            slot = namespace["candidate_slots"][result_index]
            completion = validate_tuning_lease_completion(
                config,
                namespace,
                slot,
                row,
                custody_receipt_path=custody["_validated_path"],
                expected_custody_receipt_sha256=custody[
                    "tuning_lease_custody_receipt_sha256"
                ],
                expected_custody_receipt_file_sha256=custody[
                    "_validated_file_sha256"
                ],
                namespace_file_sha256=namespace_file_sha256,
            )
            row["tuning_lease_completion_receipt"] = {
                "path": completion["_validated_path"],
                "tuning_lease_completion_sha256": completion[
                    "tuning_lease_completion_sha256"
                ],
                "file_sha256": completion["_validated_file_sha256"],
                "authorization_sha256": completion[
                    "tuning_lease_authorization_sha256"
                ],
                "lease_session_id": completion["lease_session_id"],
                "global_active_event_sha256": completion[
                    "global_active_event_sha256"
                ],
            }
        row["selection_validation_receipt"] = _recompute_tuning_validation_loss(
            row,
            config=config,
            registry=registry,
            data_root=data_root,
        )
        rows.append(row)
    expected_models = tuple(config.get("matching", {}).get("registered_models", REGISTERED_MODELS))
    selected: dict[str, dict[str, Any]] = {}
    normalized_rows = []
    for model_name in expected_models:
        model_rows = [row for row in rows if row.get("registered_name") == model_name]
        pairs = {(float(row["learning_rate"]), float(row["weight_decay"])) for row in model_rows}
        if pairs != expected_pairs or len(model_rows) != len(expected_pairs):
            raise ValueError(f"Incomplete tuning grid for {model_name}: {sorted(pairs)}")
        for row in model_rows:
            if int(row.get("seed", -1)) != tuning_seed:
                raise ValueError(f"Tuning seed mismatch for {model_name}")
            if not isinstance(row.get("validation_loss"), (float, int)):
                raise ValueError(f"Missing validation_loss for {model_name}")
            if not row.get("checkpoint_sha256"):
                raise ValueError(f"Missing tuning checkpoint identity for {model_name}")
            normalized_rows.append(dict(row))
        winner = min(
            model_rows,
            key=lambda row: (
                float(
                    row["selection_validation_receipt"][
                        "recomputed_validation_loss"
                    ]
                ),
                float(row["weight_decay"]),
                float(row["learning_rate"]),
            ),
        )
        selected[model_name] = {
            "learning_rate": float(winner["learning_rate"]),
            "weight_decay": float(winner["weight_decay"]),
            "validation_loss": float(
                winner["selection_validation_receipt"][
                    "recomputed_validation_loss"
                ]
            ),
            "checkpoint_sha256": str(winner["checkpoint_sha256"]),
        }
    core = {
        "schema_version": 1,
        "tuning_seed": tuning_seed,
        "candidate_grid": {
            "learning_rates": learning_rates,
            "weight_decays": weight_decays,
        },
        "selection_rule": "minimum validation loss; then lower weight decay; then lower learning rate",
        "source_config_sha256": canonical_json_hash(config),
        "tuning_design": design,
        "tuning_design_sha256": design["tuning_design_sha256"],
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "validation_rows_sha256": canonical_json_hash(trajectory["splits"]["val"]),
        "parameter_registry_sha256": registry["registry_sha256"],
        "implementation_sha256": registry["implementation"]["aggregate_sha256"],
        "environment_sha256": design["environment_sha256"],
        "controller_selection_sha256": design["controller_selection_sha256"],
        "tuning_namespace_sha256": (
            namespace["tuning_namespace_sha256"] if namespace else None
        ),
        "tuning_namespace_file_sha256": namespace_file_sha256,
        "tuning_namespace_instance_id": (
            namespace["namespace_instance_id"] if namespace else None
        ),
        "tuning_lease_custody_receipt_path": (
            custody["_validated_path"] if custody else None
        ),
        "tuning_lease_custody_receipt_sha256": (
            custody["tuning_lease_custody_receipt_sha256"] if custody else None
        ),
        "tuning_lease_custody_receipt_file_sha256": (
            custody["_validated_file_sha256"] if custody else None
        ),
        "tuning_gpu_identity": design["tuning_gpu_identity"],
        "trusted_nvidia_smi": design["trusted_nvidia_smi"],
        "candidates": normalized_rows,
        "selected": selected,
        "validation_only": True,
        "test_episodes_used": 0,
    }
    artifact = {
        **core,
        "tuning_selection_sha256": canonical_json_hash(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(destination, artifact)
    return artifact


def validate_tuning_selection(
    config: dict[str, Any],
    registry: dict[str, Any],
    trajectory_manifest: dict[str, Any],
    *,
    data_root: str | Path | None = None,
    recompute_epoch_validation: bool = False,
) -> dict[str, Any] | None:
    path = config.get("tuning", {}).get("artifact")
    if not path:
        if is_formal(config):
            raise ValueError("Formal config lacks tuning.artifact")
        return None
    artifact = _load_hashed(path, "tuning_selection_sha256")
    expected = {
        "source_config_sha256": canonical_json_hash(config),
        "trajectory_manifest_sha256": trajectory_manifest["manifest_sha256"],
        "validation_rows_sha256": canonical_json_hash(trajectory_manifest["splits"]["val"]),
        "parameter_registry_sha256": registry["registry_sha256"],
        "implementation_sha256": registry["implementation"]["aggregate_sha256"],
        "tuning_gpu_identity": tuning_design_identity(
            config, registry, trajectory_manifest
        )["tuning_gpu_identity"],
        "trusted_nvidia_smi": tuning_design_identity(
            config, registry, trajectory_manifest
        )["trusted_nvidia_smi"],
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"Tuning selection identity mismatch for {key}")
    if is_formal(config):
        environment = validate_environment_freeze(config)
        controller = validate_controller_selection(config)
        if artifact.get("environment_sha256") != environment["environment_sha256"]:
            raise ValueError("Tuning selection environment identity mismatch")
        if artifact.get("controller_selection_sha256") != controller["controller_selection_sha256"]:
            raise ValueError("Tuning selection controller identity mismatch")
    expected_models = set(config.get("matching", {}).get("registered_models", REGISTERED_MODELS))
    if set(artifact.get("selected", {})) != expected_models:
        raise ValueError("Tuning selection does not contain every registered model")
    design = tuning_design_identity(config, registry, trajectory_manifest)
    if artifact.get("tuning_design_sha256") != design["tuning_design_sha256"]:
        raise ValueError("Tuning selection design identity mismatch")
    namespace: dict[str, Any] | None = None
    namespace_file_sha256: str | None = None
    custody: dict[str, Any] | None = None
    if is_formal(config):
        namespace_file_sha256 = str(
            artifact.get("tuning_namespace_file_sha256", "")
        )
        namespace = validate_tuning_namespace(
            config,
            registry,
            trajectory_manifest,
            namespace_path=config["tuning"]["namespace_artifact"],
            expected_namespace_sha256=str(
                artifact.get("tuning_namespace_sha256", "")
            ),
            expected_namespace_file_sha256=namespace_file_sha256,
        )
        if (
            artifact.get("tuning_namespace_instance_id")
            != namespace["namespace_instance_id"]
            or artifact.get("tuning_gpu_identity") != namespace["gpu_identity"]
        ):
            raise ValueError("Tuning selection namespace/GPU identity mismatch")
        custody = validate_tuning_lease_custody_receipt(
            config,
            namespace,
            receipt_path=str(
                artifact.get("tuning_lease_custody_receipt_path", "")
            ),
            expected_receipt_sha256=str(
                artifact.get("tuning_lease_custody_receipt_sha256", "")
            ),
            expected_receipt_file_sha256=str(
                artifact.get("tuning_lease_custody_receipt_file_sha256", "")
            ),
            namespace_file_sha256=namespace_file_sha256,
        )
        validate_terminal_tuning_failure(namespace, custody)
    grid_pairs = {
        (float(lr), float(wd))
        for lr in design["learning_rates"]
        for wd in design["weight_decays"]
    }
    rows = artifact.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("Tuning selection candidate evidence is missing")
    tuning_seed = int(design["tuning_seed"])
    recomputed_selected: dict[str, dict[str, Any]] = {}
    for model_name in sorted(expected_models):
        model_rows = [row for row in rows if row.get("registered_name") == model_name]
        pairs = {
            (float(row["learning_rate"]), float(row["weight_decay"]))
            for row in model_rows
        }
        if pairs != grid_pairs or len(model_rows) != len(grid_pairs):
            raise ValueError(f"Incomplete embedded tuning grid for {model_name}")
        for row in model_rows:
            row_core = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "tuning_candidate_sha256",
                    "created_at_utc",
                    "selection_validation_receipt",
                    "tuning_lease_completion_receipt",
                }
            }
            if canonical_json_hash(row_core) != row.get("tuning_candidate_sha256"):
                raise ValueError("Embedded tuning candidate hash mismatch")
            if int(row.get("seed", -1)) != tuning_seed:
                raise ValueError(f"Tuning seed mismatch for {model_name}")
            _validate_tuning_candidate_checkpoint(
                row,
                config=config,
                registry=registry,
                trajectory=trajectory_manifest,
                design=design,
                namespace=namespace,
                namespace_file_sha256=namespace_file_sha256,
            )
            if namespace is not None:
                assert custody is not None
                matching_slots = [
                    slot
                    for slot in namespace["candidate_slots"]
                    if slot["result_path"]
                    == str(_resolve(row["tuning_candidate_result_path"]))
                ]
                if len(matching_slots) != 1:
                    raise ValueError(
                        "Tuning selection row is not a unique lease-evidence slot"
                    )
                completion = validate_tuning_lease_completion(
                    config,
                    namespace,
                    matching_slots[0],
                    row,
                    custody_receipt_path=custody["_validated_path"],
                    expected_custody_receipt_sha256=custody[
                        "tuning_lease_custody_receipt_sha256"
                    ],
                    expected_custody_receipt_file_sha256=custody[
                        "_validated_file_sha256"
                    ],
                    namespace_file_sha256=namespace_file_sha256,
                )
                expected_completion = {
                    "path": completion["_validated_path"],
                    "tuning_lease_completion_sha256": completion[
                        "tuning_lease_completion_sha256"
                    ],
                    "file_sha256": completion["_validated_file_sha256"],
                    "authorization_sha256": completion[
                        "tuning_lease_authorization_sha256"
                    ],
                    "lease_session_id": completion["lease_session_id"],
                    "global_active_event_sha256": completion[
                        "global_active_event_sha256"
                    ],
                }
                if row.get("tuning_lease_completion_receipt") != expected_completion:
                    raise ValueError(
                        "Tuning selection completion evidence is stale or divergent"
                    )
            receipt = row.get("selection_validation_receipt")
            if (
                not isinstance(receipt, dict)
                or int(receipt.get("test_episodes_used", -1)) != 0
                or not math.isclose(
                    float(receipt.get("recomputed_validation_loss", math.nan)),
                    float(row["validation_loss"]),
                    rel_tol=1e-5,
                    abs_tol=1e-7,
                )
                or float(receipt.get("absolute_error", math.inf))
                > float(receipt.get("acceptance_tolerance", -1.0))
            ):
                raise ValueError("Tuning candidate validation-recompute receipt failed")
            if recompute_epoch_validation:
                if data_root is None:
                    raise ValueError(
                        "Tuning epoch replay requires the frozen train/validation data root"
                    )
                recomputed_receipt = _recompute_tuning_validation_loss(
                    row,
                    config=config,
                    registry=registry,
                    data_root=data_root,
                )
                if receipt != recomputed_receipt:
                    raise ValueError(
                        "Tuning selection epoch replay receipt is stale or divergent"
                    )
            elif (
                receipt.get("selection_rule")
                != "minimum_finite_validation_loss_then_earliest_epoch"
                or int(receipt.get("selected_epoch", -1)) < 0
                or len(receipt.get("per_epoch_replay", []))
                != int(design["epochs"])
                or receipt.get("per_epoch_replay_sha256")
                != canonical_json_hash(receipt.get("per_epoch_replay", []))
            ):
                raise ValueError("Tuning epoch replay receipt is incomplete")
        winner = min(
            model_rows,
            key=lambda row: (
                float(
                    row["selection_validation_receipt"][
                        "recomputed_validation_loss"
                    ]
                ),
                float(row["weight_decay"]),
                float(row["learning_rate"]),
            ),
        )
        recomputed_selected[model_name] = {
            "learning_rate": float(winner["learning_rate"]),
            "weight_decay": float(winner["weight_decay"]),
            "validation_loss": float(
                winner["selection_validation_receipt"][
                    "recomputed_validation_loss"
                ]
            ),
            "checkpoint_sha256": str(winner["checkpoint_sha256"]),
        }
    if artifact.get("selected") != recomputed_selected:
        raise ValueError("Tuning selected rows are not the deterministic grid winners")
    if namespace is not None:
        embedded_paths = [
            str(_resolve(row.get("tuning_candidate_result_path", "")))
            for row in rows
        ]
        registered_paths = [
            str(slot["result_path"]) for slot in namespace["candidate_slots"]
        ]
        if embedded_paths != registered_paths:
            raise ValueError("Tuning selection candidates differ from namespace order")
    for model_name, selected in artifact.get("selected", {}).items():
        if (float(selected["learning_rate"]), float(selected["weight_decay"])) not in grid_pairs:
            raise ValueError(f"Selected optimizer for {model_name} is outside frozen grid")
    if not artifact.get("validation_only") or artifact.get("test_episodes_used") != 0:
        raise ValueError("Tuning selection is not validation-only")
    return artifact


def _replay_controller_selection(
    config: dict[str, Any], artifact: dict[str, Any]
) -> dict[str, Any]:
    """Independently replay every validation-only controller candidate."""
    from .baselines import build_nonlearned_baseline
    from .benchmark import _rollout

    condition = {
        "condition_id": "controller_selection_nominal",
        "family": "controller_selection_nominal",
        "severity": 0.0,
        "spec": {},
    }
    validation_rows = build_trajectory_manifest(config)["splits"]["val"]
    recorded = {
        (
            float(row["joint_kp"]),
            float(row["joint_kd"]),
            str(row["trajectory_id"]),
        ): row
        for row in artifact["validation_episode_rows"]
    }
    replay_rows: list[dict[str, Any]] = []
    for candidate in artifact["candidates"]:
        kp = float(candidate["joint_kp"])
        kd = float(candidate["joint_kd"])
        candidate_config = copy.deepcopy(config)
        candidate_config.setdefault("controller", {})["joint_kp"] = kp
        candidate_config["controller"]["joint_kd"] = kd
        for validation_row in validation_rows:
            estimator = {
                "model_name": "full_state_pd",
                "training_seed": -1,
                "baseline": build_nonlearned_baseline("full_state_pd"),
            }
            result = _rollout(
                estimator,
                candidate_config,
                condition,
                episode_seed=int(validation_row["episode_seed"]),
                schedule_seed=int(validation_row["schedule_seed"]),
                device="cpu",
            )
            key = (kp, kd, str(validation_row["trajectory_id"]))
            expected = recorded.get(key)
            if expected is None:
                raise ValueError("Controller replay lacks recorded validation row")
            for metric in ("failure", "tracking_rmse", "control_effort"):
                if not math.isclose(
                    float(result[metric]),
                    float(expected[metric]),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                ):
                    raise ValueError(
                        f"Controller independent replay mismatch for {key}, {metric}"
                    )
            replay_rows.append(
                {
                    "joint_kp": kp,
                    "joint_kd": kd,
                    "trajectory_id": str(validation_row["trajectory_id"]),
                    "episode_seed": int(validation_row["episode_seed"]),
                    "schedule_seed": int(validation_row["schedule_seed"]),
                    "failure": float(result["failure"]),
                    "tracking_rmse": float(result["tracking_rmse"]),
                    "control_effort": float(result["control_effort"]),
                }
            )
    return {
        "schema_version": 1,
        "validation_only": True,
        "test_episodes_used": 0,
        "rows": replay_rows,
        "rows_sha256": canonical_json_hash(replay_rows),
        "all_recorded_rows_reproduced": True,
    }


def create_pretraining_master_release(
    config: dict[str, Any],
    data_root: str | Path,
    registry_path: str | Path,
    preflight_receipt_path: str | Path,
    run_registry_namespace_path: str | Path,
    run_registry_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze the full pre-confirmatory chain before any registered training."""
    if not is_formal(config) or is_development_only(config):
        raise ValueError("Pretraining master release requires a formal config")
    destination = _resolve(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite pretraining release: {destination}")
    from .manifests import verify_split_datasets

    environment = validate_environment_freeze(config)
    controller = validate_controller_selection(config)
    replay_verification = verify_split_datasets(
        _resolve(data_root), expected_config=config, full_replay=True
    )
    trajectory = load_trajectory_manifest(
        _resolve(data_root) / "trajectory_manifest.json", expected_config=config
    )
    replay_core = {
        "schema_version": 1,
        "source_config_sha256": canonical_json_hash(config),
        "data_root": str(_resolve(data_root)),
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "materialized_splits": replay_verification["materialized_splits"],
        "dataset_sha256": replay_verification["dataset_sha256"],
        "completion_sha256": replay_verification["completion_sha256"],
        "deterministic_full_replay_verified": replay_verification[
            "deterministic_full_replay_verified"
        ],
        "test_rows_committed_not_materialized": replay_verification[
            "test_rows_committed_not_materialized"
        ],
        "verification_implementation_sha256": source_tree_manifest(
            Path(__file__).resolve().parents[1]
        )["aggregate_sha256"],
    }
    replay_receipt = {
        **replay_core,
        "train_val_full_replay_receipt_sha256": canonical_json_hash(replay_core),
    }
    registry = load_parameter_registry(registry_path, expected_config=config)
    tuning = validate_tuning_selection(
        config,
        registry,
        trajectory,
        data_root=data_root,
        recompute_epoch_validation=True,
    )
    if any(
        row.get("selection_validation_receipt", {}).get("test_episodes_used") != 0
        for row in tuning.get("candidates", [])
    ):
        raise ValueError("Tuning validation recompute receipts are incomplete")
    controller_replay = _replay_controller_selection(config, controller)
    from .preflight import validate_preflight_receipt

    preflight = validate_preflight_receipt(
        config=config,
        data_root=data_root,
        registry_path=registry_path,
        receipt_path=preflight_receipt_path,
        device="cuda:0",
    )
    if (
        preflight.get("status") != "complete"
        or preflight.get("all_required_gates_passed") is not True
        or int(preflight.get("test_episodes_used", -1)) != 0
    ):
        raise ValueError("Preflight receipt has not passed every required gate")
    expected_preflight = {
        "source_config_sha256": canonical_json_hash(config),
        "environment_sha256": environment["environment_sha256"],
        "controller_selection_sha256": controller["controller_selection_sha256"],
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "tuning_selection_sha256": tuning["tuning_selection_sha256"],
    }
    for key, value in expected_preflight.items():
        if preflight.get(key) != value:
            raise ValueError(f"Preflight identity mismatch for {key}")
    from .runner import (
        create_pretraining_run_registry,
        create_run_registry_namespace,
    )

    run_namespace = create_run_registry_namespace(
        run_registry_namespace_path,
        run_registry_path=run_registry_path,
        config_sha256=canonical_json_hash(config),
        preflight_receipt_sha256=preflight["preflight_receipt_sha256"],
        models=config["matching"]["registered_models"],
        seeds=config["training"]["seeds"],
    )
    initial_run_registry = create_pretraining_run_registry(
        run_registry_path,
        namespace_path=run_registry_namespace_path,
        expected_namespace_sha256=run_namespace[
            "run_registry_namespace_sha256"
        ],
        config_sha256=canonical_json_hash(config),
        preflight_receipt_sha256=preflight["preflight_receipt_sha256"],
        models=config["matching"]["registered_models"],
        seeds=config["training"]["seeds"],
    )
    core = {
        "schema_version": 1,
        "phase": "pre_confirmatory_training",
        "source_config_sha256": canonical_json_hash(config),
        "environment_sha256": environment["environment_sha256"],
        "controller_selection_sha256": controller["controller_selection_sha256"],
        "controller_independent_replay": controller_replay,
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "train_val_full_replay_verified": True,
        "train_val_full_replay_receipt": replay_receipt,
        "train_val_full_replay_receipt_sha256": replay_receipt[
            "train_val_full_replay_receipt_sha256"
        ],
        "test_rows_committed_but_not_materialized": True,
        "parameter_registry_sha256": registry["registry_sha256"],
        "implementation_sha256": registry["implementation"]["aggregate_sha256"],
        "tuning_selection_sha256": tuning["tuning_selection_sha256"],
        "tuning_validation_receipts_sha256": canonical_json_hash(
            [
                row["selection_validation_receipt"]
                for row in tuning["candidates"]
            ]
        ),
        "preflight_receipt_sha256": preflight["preflight_receipt_sha256"],
        "run_registry_namespace_path": str(
            Path(run_registry_namespace_path).resolve()
        ),
        "run_registry_namespace_sha256": run_namespace[
            "run_registry_namespace_sha256"
        ],
        "run_registry_path": str(Path(run_registry_path).resolve()),
        "expected_run_identities": run_namespace["expected_run_identities"],
        "run_registry_instance_id": initial_run_registry[
            "run_registry_instance_id"
        ],
        "initial_registry_sha256": initial_run_registry[
            "initial_registry_sha256"
        ],
        "initial_registry_file_sha256": sha256_file(run_registry_path),
        "initial_registry_canonical_sha256": initial_run_registry[
            "registry_sha256"
        ],
        "selected_physical_gpu": preflight["identities"][
            "selected_physical_gpu"
        ],
        "precision_policy": preflight["identities"]["precision_policy"],
        "formal_artifact_identities": formal_artifact_identities(
            config,
            registry=registry,
            trajectory_manifest=trajectory,
            require_tuning=True,
        ),
        "test_episodes_used": 0,
        "approved_for_external_audit_anchor": True,
    }
    artifact = {
        **core,
        "pretraining_master_sha256": canonical_json_hash(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(destination, artifact)
    return artifact


def validate_pretraining_master_release(
    config: dict[str, Any],
    *,
    data_root: str | Path,
    expected_master_sha256: str | None,
    audit_receipt_path: str | Path | None,
    expected_audit_receipt_sha256: str | None,
    expected_audit_receipt_file_sha256: str | None,
    expected_identities: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require both the externally supplied root and independent audit receipt."""
    if not is_formal(config):
        raise ValueError("Pretraining master validation is formal-only")
    path = config.get("formal", {}).get("pretraining_master_artifact")
    if not path:
        raise ValueError("Formal config lacks pretraining_master_artifact")
    artifact = _load_hashed(path, "pretraining_master_sha256")
    if (
        not expected_master_sha256
        or artifact["pretraining_master_sha256"] != str(expected_master_sha256)
    ):
        raise ValueError("Pretraining master differs from external published anchor")
    preflight_path = config.get("formal", {}).get("preflight_artifact")
    if not preflight_path:
        raise ValueError("Formal config lacks formal.preflight_artifact")
    current_preflight = _load_hashed(
        preflight_path, "preflight_receipt_sha256"
    )
    from .manifests import verify_split_datasets

    replay_verification = verify_split_datasets(
        _resolve(data_root), expected_config=config, full_replay=True
    )
    current_trajectory = load_trajectory_manifest(
        _resolve(data_root) / "trajectory_manifest.json", expected_config=config
    )
    replay_core = {
        "schema_version": 1,
        "source_config_sha256": canonical_json_hash(config),
        "data_root": str(_resolve(data_root)),
        "trajectory_manifest_sha256": current_trajectory["manifest_sha256"],
        "materialized_splits": replay_verification["materialized_splits"],
        "dataset_sha256": replay_verification["dataset_sha256"],
        "completion_sha256": replay_verification["completion_sha256"],
        "deterministic_full_replay_verified": replay_verification[
            "deterministic_full_replay_verified"
        ],
        "test_rows_committed_not_materialized": replay_verification[
            "test_rows_committed_not_materialized"
        ],
        "verification_implementation_sha256": source_tree_manifest(
            Path(__file__).resolve().parents[1]
        )["aggregate_sha256"],
    }
    replay_receipt = {
        **replay_core,
        "train_val_full_replay_receipt_sha256": canonical_json_hash(replay_core),
    }
    if (
        artifact.get("train_val_full_replay_receipt") != replay_receipt
        or artifact.get("train_val_full_replay_receipt_sha256")
        != replay_receipt["train_val_full_replay_receipt_sha256"]
        or artifact.get("train_val_full_replay_verified") is not True
    ):
        raise ValueError(
            "Pretraining master deterministic train/validation replay receipt mismatch"
        )
    namespace_path_value = config.get("formal", {}).get(
        "run_registry_namespace_artifact"
    )
    current_run_registry_path = expected_identities.get("run_registry_path")
    if not namespace_path_value or not current_run_registry_path:
        raise ValueError(
            "Pretraining validation requires the frozen run-registry namespace/path"
        )
    from .runner import (
        validate_pretraining_run_registry_binding,
        validate_run_registry_namespace,
    )

    current_namespace = validate_run_registry_namespace(
        _resolve(namespace_path_value),
        run_registry_path=current_run_registry_path,
        config_sha256=canonical_json_hash(config),
        preflight_receipt_sha256=current_preflight[
            "preflight_receipt_sha256"
        ],
        models=config["matching"]["registered_models"],
        seeds=config["training"]["seeds"],
        expected_namespace_sha256=artifact.get(
            "run_registry_namespace_sha256"
        ),
    )
    current_run_registry = validate_pretraining_run_registry_binding(
        current_run_registry_path,
        pretraining_master=artifact,
    )
    current_expected = {
        "source_config_sha256": canonical_json_hash(config),
        "preflight_receipt_sha256": current_preflight[
            "preflight_receipt_sha256"
        ],
        "selected_physical_gpu": current_preflight["identities"][
            "selected_physical_gpu"
        ],
        "precision_policy": current_preflight["identities"][
            "precision_policy"
        ],
        "run_registry_namespace_path": str(
            _resolve(namespace_path_value)
        ),
        "run_registry_namespace_sha256": current_namespace[
            "run_registry_namespace_sha256"
        ],
        "run_registry_path": str(Path(current_run_registry_path).resolve()),
        "expected_run_identities": current_namespace[
            "expected_run_identities"
        ],
        "run_registry_instance_id": current_run_registry[
            "run_registry_instance_id"
        ],
        "initial_registry_sha256": current_run_registry[
            "initial_registry_sha256"
        ],
        **expected_identities,
    }
    for key, value in current_expected.items():
        if artifact.get(key) != value:
            raise ValueError(
                f"Pretraining master/current formal chain mismatch for {key}"
            )
    expected_formal = {
        key: current_expected[key]
        for key in (
            "environment_sha256",
            "controller_selection_sha256",
            "tuning_selection_sha256",
        )
    }
    if artifact.get("formal_artifact_identities") != expected_formal:
        raise ValueError("Pretraining master formal identities are stale or divergent")
    if (
        int(artifact.get("test_episodes_used", -1)) != 0
        or artifact.get("approved_for_external_audit_anchor") is not True
    ):
        raise ValueError("Pretraining master is not a pre-test approved artifact")
    if not audit_receipt_path:
        raise ValueError("Formal training requires an independent audit receipt")
    if (
        not expected_audit_receipt_file_sha256
        or len(str(expected_audit_receipt_file_sha256)) != 64
    ):
        raise PermissionError(
            "Formal training requires the externally published audit-receipt file SHA-256"
        )
    int(str(expected_audit_receipt_file_sha256), 16)
    if sha256_file(audit_receipt_path) != str(
        expected_audit_receipt_file_sha256
    ):
        raise PermissionError(
            "Independent pretraining audit receipt differs from its external file anchor"
        )
    receipt = _load_hashed(
        audit_receipt_path, "pretraining_audit_receipt_sha256"
    )
    if (
        not expected_audit_receipt_sha256
        or receipt.get("pretraining_audit_receipt_sha256")
        != str(expected_audit_receipt_sha256)
        or receipt.get("pretraining_master_sha256")
        != artifact["pretraining_master_sha256"]
        or receipt.get("decision") != "approved_for_confirmatory_training"
        or receipt.get("reviewer_role") != "independent_read_only_auditor"
        or int(receipt.get("test_episodes_used", -1)) != 0
    ):
        raise ValueError("Independent pretraining audit receipt is invalid")
    custody = receipt.get("run_registry_custody")
    if (
        not isinstance(custody, dict)
        or custody.get("mode") != "auditor_managed_append_only"
        or custody.get("run_registry_instance_id")
        != artifact["run_registry_instance_id"]
        or custody.get("initial_registry_sha256")
        != artifact["initial_registry_sha256"]
        or custody.get("acl_policy")
        != "create_only_no_modify_no_delete_existing_objects"
        or not Path(str(custody.get("journal_directory", ""))).is_absolute()
        or not str(custody.get("external_attestation_id", "")).strip()
    ):
        raise PermissionError(
            "Independent audit receipt lacks the externally managed append-only "
            "run-registry custody attestation"
        )
    return artifact, {
        **receipt,
        "_validated_file_sha256": str(expected_audit_receipt_file_sha256),
    }


def _validate_epoch_checkpoint_release(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    metadata: dict[str, Any],
    run_state: dict[str, Any],
    data_root: Path,
    registry: dict[str, Any],
    expected_identity: dict[str, Any],
    recompute_validation: bool,
) -> dict[str, Any]:
    """Verify every immutable validation checkpoint and the selection rule.

    The initial checkpoint release independently evaluates every epoch on the
    frozen validation dataset.  Later test-release/evaluation validation is
    anchored to that immutable release and rechecks every file hash and ledger
    relationship without repeating the expensive validation forward passes.
    """
    import torch

    from ..data.closed_loop import load_closed_loop_dataset
    from .training import (
        Normalizer,
        SequenceWindows,
        _epoch_checkpoint_ledger,
        _run_epoch,
    )

    ledger_path = checkpoint_path.parent / "checkpoint_ledger.json"
    ledger = _load_hashed(ledger_path, "checkpoint_ledger_sha256")
    rebuilt_ledger = _epoch_checkpoint_ledger(
        checkpoint_path.parent / "epoch_checkpoints",
        expected_identity=expected_identity,
    )
    if ledger != rebuilt_ledger:
        raise ValueError("Stored epoch ledger differs from immutable checkpoint chain")
    if (
        str(Path(run_state.get("checkpoint_ledger", "")).resolve())
        != str(ledger_path.resolve())
        or run_state.get("checkpoint_ledger_file_sha256")
        != sha256_file(ledger_path)
        or run_state.get("checkpoint_ledger_sha256")
        != ledger["checkpoint_ledger_sha256"]
        or metadata.get("checkpoint_ledger_sha256")
        != ledger["checkpoint_ledger_sha256"]
    ):
        raise ValueError("Run/checkpoint immutable epoch ledger mismatch")
    rows = ledger.get("checkpoints")
    expected_epochs = int(config.get("training", {}).get("epochs", 0))
    relative_tolerance = float(
        config.get("formal", {}).get(
            "validation_recompute_relative_tolerance", 1e-5
        )
    )
    absolute_tolerance = float(
        config.get("formal", {}).get(
            "validation_recompute_absolute_tolerance", 1e-7
        )
    )
    if (
        not isinstance(rows, list)
        or len(rows) != expected_epochs
        or int(ledger.get("checkpoint_count", -1)) != expected_epochs
        or ledger.get("identity") != expected_identity
        or ledger.get("write_once_epoch_checkpoints") is not True
        or not math.isfinite(relative_tolerance)
        or not math.isfinite(absolute_tolerance)
        or relative_tolerance < 0.0
        or absolute_tolerance < 0.0
    ):
        raise ValueError("Immutable epoch ledger does not contain the exact run")
    validation_arrays = None
    validation_manifest = None
    windows = None
    normalizer = None
    rows_payload_normalizer = None
    if recompute_validation:
        validation_arrays, validation_manifest = load_closed_loop_dataset(
            data_root / "val"
        )
    replay_rows: list[dict[str, Any]] = []
    selected_epoch = -1
    selected_loss = math.inf
    selected_payload: dict[str, Any] | None = None
    for epoch, row in enumerate(rows):
        expected_path = checkpoint_path.parent / "epoch_checkpoints" / f"epoch_{epoch:04d}.pt"
        if (
            int(row.get("epoch", -1)) != epoch
            or str(Path(row.get("checkpoint_path", "")).resolve())
            != str(expected_path.resolve())
            or not expected_path.is_file()
            or sha256_file(expected_path) != row.get("checkpoint_sha256")
        ):
            raise ValueError(f"Immutable epoch checkpoint mismatch at epoch {epoch}")
        payload = torch.load(expected_path, map_location="cpu", weights_only=False)
        for key, value in expected_identity.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Immutable epoch identity mismatch for {key} at epoch {epoch}"
                )
        history = payload.get("history")
        if (
            int(payload.get("epoch", -1)) != epoch
            or not isinstance(history, list)
            or len(history) != epoch + 1
            or int(history[-1].get("epoch", -1)) != epoch
            or canonical_json_hash(history) != row.get("history_sha256")
            or int(payload.get("optimizer_steps_completed", -1))
            != int(row.get("optimizer_steps_completed", -2))
            or not math.isclose(
                float(history[-1].get("val_loss", math.nan)),
                float(row.get("validation_loss", math.nan)),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError(f"Immutable epoch history mismatch at epoch {epoch}")
        recorded_loss = float(row["validation_loss"])
        if not math.isfinite(recorded_loss):
            raise ValueError(f"Non-finite validation loss at epoch {epoch}")
        recomputed_loss = recorded_loss
        if recompute_validation:
            if payload.get("data_sha256", {}).get("val") != validation_manifest.get(
                "dataset_sha256"
            ):
                raise ValueError("Epoch checkpoint validation data hash mismatch")
            if normalizer is None:
                normalizer = Normalizer.from_dict(payload["normalizer"])
                training = dict(config.get("training", {}))
                windows = SequenceWindows(
                    validation_arrays,
                    normalizer,
                    int(training.get("window_steps", 64)),
                    int(training.get("burn_in_steps", 0)),
                )
            elif payload["normalizer"] != rows_payload_normalizer:
                raise ValueError("Normalizer changed between epoch checkpoints")
            rows_payload_normalizer = payload["normalizer"]
            model = build_registered_model(
                str(metadata["registered_name"]), config, registry
            )
            load_graph_safe_state_dict(
                model, payload["model_state_dict"], strict=True
            )
            recomputed_loss, _ = _run_epoch(
                model,
                windows,
                batch_size=int(config.get("training", {}).get("batch_size", 32)),
                device=torch.device("cpu"),
                optimizer=None,
                gradient_clip=float(
                    config.get("training", {}).get("gradient_clip_norm", 1.0)
                ),
                order_seed=0,
            )
            if not math.isclose(
                recomputed_loss,
                recorded_loss,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                raise ValueError(
                    f"Epoch {epoch} independently recomputed validation loss "
                    f"differs: {recomputed_loss} vs {recorded_loss}"
                )
        if recorded_loss < selected_loss:
            selected_epoch = epoch
            selected_loss = recorded_loss
            selected_payload = payload
        replay_rows.append(
            {
                "epoch": epoch,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "recorded_validation_loss": recorded_loss,
                "recomputed_validation_loss": float(recomputed_loss),
            }
        )
    if (
        selected_payload is None
        or selected_epoch != int(run_state.get("best_epoch", -1))
        or not math.isclose(
            selected_loss,
            float(run_state.get("best_val_loss", math.nan)),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("Frozen validation rule does not select released best epoch")
    selected_epoch_path = str(Path(rows[selected_epoch]["checkpoint_path"]).resolve())
    selected_epoch_sha256 = str(rows[selected_epoch]["checkpoint_sha256"])
    for source_name, source in (("run_state", run_state), ("best", metadata)):
        if (
            source.get("selected_epoch_checkpoint_path") != selected_epoch_path
            or source.get("selected_epoch_checkpoint_sha256")
            != selected_epoch_sha256
        ):
            raise ValueError(
                f"{source_name} selected immutable epoch binding mismatch"
            )
    selected_state = selected_payload["model_state_dict"]
    released_state = metadata["model_state_dict"]
    if set(selected_state) != set(released_state) or any(
        not torch.equal(selected_state[key].cpu(), released_state[key].cpu())
        for key in selected_state
    ):
        raise ValueError("Released best weights differ from selected epoch checkpoint")
    receipt_core = {
        "schema_version": 1,
        "checkpoint_ledger_sha256": ledger["checkpoint_ledger_sha256"],
        "checkpoint_ledger_file_sha256": sha256_file(ledger_path),
        "checkpoint_count": len(rows),
        "selected_epoch": selected_epoch,
        "selected_validation_loss": selected_loss,
        "selection_rule": "minimum_finite_validation_loss_then_earliest_epoch",
        "selected_epoch_checkpoint_path": selected_epoch_path,
        "selected_epoch_checkpoint_sha256": selected_epoch_sha256,
        "validation_recomputed": bool(recompute_validation),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "validation_dataset_sha256": (
            validation_manifest["dataset_sha256"]
            if validation_manifest is not None
            else metadata.get("data_sha256", {}).get("val")
        ),
        "ledger_epoch_rows_sha256": canonical_json_hash(rows),
        "epoch_receipts_sha256": canonical_json_hash(replay_rows),
        "test_episodes_used": 0,
    }
    return {
        **receipt_core,
        "epoch_validation_receipt_sha256": canonical_json_hash(receipt_core),
    }


def _validate_released_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    trajectory_sha256: str,
    registry_sha256: str,
    implementation_sha256: str,
    formal_identities: dict[str, Any],
    expected_precision_sha256: str | None,
    expected_pretraining_master_sha256: str | None,
    expected_pretraining_audit_receipt_sha256: str | None,
    expected_pretraining_audit_receipt_file_sha256: str | None,
    data_root: Path,
    registry: dict[str, Any],
    recompute_epoch_validation: bool,
    expected_run_reservation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deeply bind one selected checkpoint, final run state, and latest state."""
    import torch

    from .training import load_trained_observer

    model, metadata = load_trained_observer(
        checkpoint_path,
        device="cpu",
        expected_config=config,
        expected_trajectory_manifest_sha256=trajectory_sha256,
    )
    del model
    identity = (str(metadata.get("registered_name")), int(metadata.get("seed", -1)))
    expected_metadata = {
        "trajectory_manifest_sha256": trajectory_sha256,
        "parameter_registry_sha256": registry_sha256,
        "parameter_registry_implementation_sha256": implementation_sha256,
        **formal_identities,
    }
    if expected_pretraining_master_sha256 is not None:
        expected_metadata["pretraining_master_sha256"] = str(
            expected_pretraining_master_sha256
        )
        if (
            not expected_pretraining_audit_receipt_sha256
            or metadata.get("pretraining_audit_receipt_sha256")
            != str(expected_pretraining_audit_receipt_sha256)
            or not expected_pretraining_audit_receipt_file_sha256
            or metadata.get("pretraining_audit_receipt_file_sha256")
            != str(expected_pretraining_audit_receipt_file_sha256)
        ):
            raise ValueError(f"Checkpoint lacks pretraining audit receipt: {identity}")
    if expected_run_reservation is not None:
        expected_metadata.update(
            {
                "formal_run_registry_path": str(
                    Path(expected_run_reservation["run_registry_path"]).resolve()
                ),
                "formal_run_registry_namespace_sha256": expected_run_reservation[
                    "run_registry_namespace_sha256"
                ],
                "formal_run_registry_instance_id": expected_run_reservation[
                    "run_registry_instance_id"
                ],
                "formal_initial_run_registry_sha256": expected_run_reservation[
                    "initial_registry_sha256"
                ],
                "formal_run_registry_custody_sha256": expected_run_reservation[
                    "run_registry_custody_sha256"
                ],
                "formal_run_attempt_id": expected_run_reservation[
                    "attempt_id"
                ],
                "formal_run_reservation_token_path": str(
                    Path(expected_run_reservation["reservation_token_path"]).resolve()
                ),
                "formal_run_reservation_token_sha256": expected_run_reservation[
                    "reservation_token_sha256"
                ],
                "formal_run_reservation_token_file_sha256": expected_run_reservation[
                    "reservation_token_file_sha256"
                ],
                "formal_run_reservation_nonce": expected_run_reservation[
                    "reservation_nonce"
                ],
            }
        )
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Released checkpoint identity mismatch for {key}: {identity}")
    run_state_path = checkpoint_path.parent / "run_state.json"
    if not run_state_path.is_file():
        raise ValueError(f"Checkpoint release lacks run_state: {checkpoint_path}")
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    for key, value in {
        "registered_name": identity[0],
        "seed": identity[1],
        **expected_metadata,
    }.items():
        if run_state.get(key) != value:
            raise ValueError(f"Run-state/checkpoint mismatch for {key}: {identity}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        run_state.get("status") != "complete"
        or run_state.get("best_checkpoint_sha256") != checkpoint_sha256
        or str(Path(run_state.get("best_checkpoint", "")).resolve())
        != str(checkpoint_path.resolve())
        or run_state.get("fixed_budget_gate_passed") is not True
        or int(run_state.get("optimizer_steps_completed", -1))
        != int(run_state.get("optimizer_steps_planned", -2))
    ):
        raise ValueError(f"Checkpoint run-state completion gate failed: {identity}")
    if (
        int(metadata.get("training_run_optimizer_steps_completed", -1))
        != int(run_state["optimizer_steps_completed"])
        or int(metadata.get("training_run_optimizer_steps_planned", -2))
        != int(run_state["optimizer_steps_planned"])
        or metadata.get("training_run_fixed_budget_gate_passed") is not True
        or not metadata.get("selection_finalized_after_training_run")
    ):
        raise ValueError(f"Selected checkpoint fixed-budget mismatch: {identity}")
    if (
        int(metadata.get("best_epoch", -1)) != int(run_state.get("best_epoch", -2))
        or not math.isclose(
            float(metadata.get("best_val_loss", math.nan)),
            float(run_state.get("best_val_loss", math.nan)),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError(f"Selected epoch/loss mismatch: {identity}")
    history = metadata.get("training_run_history")
    if (
        not isinstance(history, list)
        or len(history) != int(run_state.get("epochs_completed", -1))
        or len(history) != int(metadata.get("training_run_epochs_completed", -2))
        or not history
        or int(history[-1].get("optimizer_steps_completed", -1))
        != int(run_state["optimizer_steps_completed"])
    ):
        raise ValueError(f"Selected checkpoint history mismatch: {identity}")
    runtime_device = metadata.get("runtime_device")
    if runtime_device != run_state.get("runtime_device"):
        raise ValueError(f"Run-state/checkpoint runtime mismatch: {identity}")
    precision_policy = (runtime_device or {}).get("precision_policy")
    if not isinstance(precision_policy, dict):
        raise ValueError(f"Checkpoint lacks precision policy: {identity}")
    precision_sha256 = canonical_json_hash(precision_policy)
    if (
        expected_precision_sha256 is not None
        and precision_sha256 != expected_precision_sha256
    ):
        raise ValueError(f"Checkpoint precision differs from frozen policy: {identity}")
    physical_gpu = (runtime_device or {}).get("selected_physical_gpu")
    if not isinstance(physical_gpu, dict) or not {
        "uuid",
        "pci_bus_id",
    } <= set(physical_gpu):
        raise ValueError(f"Formal checkpoint lacks physical GPU identity: {identity}")
    latest_path = Path(run_state.get("latest_checkpoint", "")).resolve()
    if (
        not latest_path.is_file()
        or sha256_file(latest_path) != run_state.get("latest_checkpoint_sha256")
    ):
        raise ValueError(f"Latest checkpoint hash mismatch: {identity}")
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    if (
        latest.get("registered_name") != identity[0]
        or int(latest.get("seed", -1)) != identity[1]
        or int(latest.get("optimizer_steps_completed", -1))
        != int(run_state["optimizer_steps_completed"])
        or int(latest.get("best_epoch", -1)) != int(run_state["best_epoch"])
        or not math.isclose(
            float(latest.get("best_val_loss", math.nan)),
            float(run_state["best_val_loss"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or latest.get("runtime_device") != runtime_device
        or latest.get("optimizer_hyperparameters")
        != metadata.get("optimizer_hyperparameters")
    ):
        raise ValueError(f"Latest/selected checkpoint transaction mismatch: {identity}")
    ledger_identity = {
        "run_phase": "confirmatory",
        "registered_name": identity[0],
        "seed": identity[1],
        "trajectory_manifest_sha256": trajectory_sha256,
        "parameter_registry_sha256": registry_sha256,
        "config_sha256": canonical_json_hash(config),
        **formal_identities,
    }
    if expected_pretraining_master_sha256 is not None:
        ledger_identity.update(
            {
                "pretraining_master_sha256": str(
                    expected_pretraining_master_sha256
                ),
                "pretraining_audit_receipt_sha256": metadata[
                    "pretraining_audit_receipt_sha256"
                ],
                "pretraining_audit_receipt_file_sha256": metadata[
                    "pretraining_audit_receipt_file_sha256"
                ],
            }
        )
    if expected_run_reservation is not None:
        ledger_identity.update(
            {
                key: value
                for key, value in expected_metadata.items()
                if key.startswith("formal_run_")
            }
        )
    epoch_receipt = _validate_epoch_checkpoint_release(
        config=config,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        run_state=run_state,
        data_root=data_root,
        registry=registry,
        expected_identity=ledger_identity,
        recompute_validation=recompute_epoch_validation,
    )
    return {
        "registered_name": identity[0],
        "seed": identity[1],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "run_state_path": str(run_state_path.resolve()),
        "run_state_sha256": sha256_file(run_state_path),
        "latest_checkpoint_path": str(latest_path),
        "latest_checkpoint_sha256": run_state["latest_checkpoint_sha256"],
        "best_epoch": int(run_state["best_epoch"]),
        "best_val_loss": float(run_state["best_val_loss"]),
        "optimizer_steps_completed": int(run_state["optimizer_steps_completed"]),
        "optimizer_steps_planned": int(run_state["optimizer_steps_planned"]),
        "optimizer_hyperparameters": metadata.get("optimizer_hyperparameters"),
        "training_history_sha256": canonical_json_hash(history),
        "checkpoint_ledger_path": str(
            (checkpoint_path.parent / "checkpoint_ledger.json").resolve()
        ),
        "checkpoint_ledger_file_sha256": epoch_receipt[
            "checkpoint_ledger_file_sha256"
        ],
        "checkpoint_ledger_sha256": epoch_receipt[
            "checkpoint_ledger_sha256"
        ],
        "epoch_checkpoint_count": epoch_receipt["checkpoint_count"],
        "selected_epoch_checkpoint_path": epoch_receipt[
            "selected_epoch_checkpoint_path"
        ],
        "selected_epoch_checkpoint_sha256": epoch_receipt[
            "selected_epoch_checkpoint_sha256"
        ],
        "selection_rule": epoch_receipt["selection_rule"],
        "runtime_device_sha256": canonical_json_hash(runtime_device),
        "precision_policy_sha256": precision_sha256,
        "selected_physical_gpu": physical_gpu,
        "pretraining_master_sha256": metadata.get(
            "pretraining_master_sha256"
        ),
        "pretraining_audit_receipt_sha256": metadata.get(
            "pretraining_audit_receipt_sha256"
        ),
        "pretraining_audit_receipt_file_sha256": metadata.get(
            "pretraining_audit_receipt_file_sha256"
        ),
        "formal_run_attempt_id": metadata.get("formal_run_attempt_id"),
        "formal_run_reservation_token_sha256": metadata.get(
            "formal_run_reservation_token_sha256"
        ),
        "formal_run_reservation_token_file_sha256": metadata.get(
            "formal_run_reservation_token_file_sha256"
        ),
        "formal_run_reservation_nonce": metadata.get(
            "formal_run_reservation_nonce"
        ),
        "epoch_validation_receipt": epoch_receipt,
    }


def create_checkpoint_release(
    config: dict[str, Any],
    data_root: str | Path,
    registry_path: str | Path,
    checkpoint_paths: list[str | Path],
    output_path: str | Path,
    *,
    expected_pretraining_master_sha256: str,
    expected_pretraining_audit_receipt_sha256: str,
    expected_pretraining_audit_receipt_file_sha256: str,
    pretraining_audit_receipt_path: str | Path,
    run_registry_path: str | Path,
) -> dict[str, Any]:
    """Freeze all 35 completed learned checkpoints before test authorization."""
    if not is_formal(config) or is_development_only(config):
        raise ValueError("Checkpoint release requires a non-development formal config")
    destination = _resolve(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint release: {destination}")
    from .manifests import verify_split_datasets

    verify_split_datasets(_resolve(data_root), expected_config=config)
    trajectory = load_trajectory_manifest(
        _resolve(data_root) / "trajectory_manifest.json", expected_config=config
    )
    registry = load_parameter_registry(registry_path, expected_config=config)
    identities = formal_artifact_identities(
        config,
        registry=registry,
        trajectory_manifest=trajectory,
        require_tuning=True,
    )
    validated_master, validated_audit = validate_pretraining_master_release(
        config,
        data_root=data_root,
        expected_master_sha256=expected_pretraining_master_sha256,
        audit_receipt_path=pretraining_audit_receipt_path,
        expected_audit_receipt_sha256=(
            expected_pretraining_audit_receipt_sha256
        ),
        expected_audit_receipt_file_sha256=(
            expected_pretraining_audit_receipt_file_sha256
        ),
        expected_identities={
            "run_registry_path": str(Path(run_registry_path).resolve()),
            "environment_sha256": identities["environment_sha256"],
            "controller_selection_sha256": identities[
                "controller_selection_sha256"
            ],
            "trajectory_manifest_sha256": trajectory["manifest_sha256"],
            "parameter_registry_sha256": registry["registry_sha256"],
            "implementation_sha256": registry["implementation"][
                "aggregate_sha256"
            ],
            "tuning_selection_sha256": identities[
                "tuning_selection_sha256"
            ],
        },
    )
    if (
        validated_master["pretraining_master_sha256"]
        != str(expected_pretraining_master_sha256)
        or validated_audit["pretraining_audit_receipt_sha256"]
        != str(expected_pretraining_audit_receipt_sha256)
        or validated_audit["_validated_file_sha256"]
        != str(expected_pretraining_audit_receipt_file_sha256)
    ):
        raise ValueError("Checkpoint release pretraining authorization mismatch")
    environment = validate_environment_freeze(config)
    expected_precision = environment.get("torch_runtime", {}).get(
        "precision_policy"
    )
    if not isinstance(expected_precision, dict):
        raise ValueError("Environment freeze lacks a precision policy")
    expected_precision_sha256 = canonical_json_hash(
        {key: expected_precision[key] for key in _torch_precision_policy(__import__("torch"))}
    )
    expected = {
        (str(model_name), int(seed))
        for model_name in config.get("matching", {}).get(
            "registered_models", REGISTERED_MODELS
        )
        for seed in config.get("training", {}).get("seeds", [])
    }
    from .runner import freeze_complete_run_registry

    expected_run_identities = {
        f"{model}::seed_{seed}" for model, seed in expected
    }
    run_registry_receipt = freeze_complete_run_registry(
        run_registry_path,
        expected_config_sha256=canonical_json_hash(config),
        expected_pretraining_master_sha256=expected_pretraining_master_sha256,
        expected_pretraining_audit_receipt_sha256=(
            expected_pretraining_audit_receipt_sha256
        ),
        expected_run_registry_namespace_sha256=validated_master[
            "run_registry_namespace_sha256"
        ],
        expected_identities=expected_run_identities,
        expected_pretraining_audit_receipt_file_sha256=(
            expected_pretraining_audit_receipt_file_sha256
        ),
    )
    if (
        run_registry_receipt.get("run_registry_instance_id")
        != validated_master["run_registry_instance_id"]
        or run_registry_receipt.get("initial_registry_sha256")
        != validated_master["initial_registry_sha256"]
        or run_registry_receipt.get("run_registry_custody_sha256")
        != canonical_json_hash(validated_audit["run_registry_custody"])
    ):
        raise ValueError(
            "Completed run registry differs from the externally audited initial instance"
        )
    reservation_by_checkpoint = {
        str(Path(row["checkpoint_path"]).resolve()): {
            "run_registry_path": run_registry_receipt["run_registry_path"],
            "run_registry_namespace_sha256": run_registry_receipt[
                "run_registry_namespace_sha256"
            ],
            "run_registry_instance_id": run_registry_receipt[
                "run_registry_instance_id"
            ],
            "initial_registry_sha256": run_registry_receipt[
                "initial_registry_sha256"
            ],
            "run_registry_custody_sha256": run_registry_receipt[
                "run_registry_custody_sha256"
            ],
            "attempt_id": row["complete_attempt_id"],
            **next(
                attempt
                for attempt in row["attempts"]
                if attempt["attempt_id"] == row["complete_attempt_id"]
            ),
        }
        for row in run_registry_receipt["slots"]
    }
    supplied_checkpoint_set = {
        str(_resolve(value)) for value in checkpoint_paths
    }
    if supplied_checkpoint_set != set(reservation_by_checkpoint):
        raise ValueError(
            "Checkpoint paths differ from the unique completed run-registry slots"
        )
    if len(checkpoint_paths) != len(expected):
        raise ValueError("Checkpoint release does not contain the exact registered count")
    rows: list[dict[str, Any]] = []
    observed: set[tuple[str, int]] = set()
    precision_hashes: set[str] = set()
    for checkpoint_path_value in checkpoint_paths:
        checkpoint_path = _resolve(checkpoint_path_value)
        row = _validate_released_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            trajectory_sha256=trajectory["manifest_sha256"],
            registry_sha256=registry["registry_sha256"],
            implementation_sha256=registry["implementation"]["aggregate_sha256"],
            formal_identities=identities,
            expected_precision_sha256=expected_precision_sha256,
            expected_pretraining_master_sha256=expected_pretraining_master_sha256,
            expected_pretraining_audit_receipt_sha256=(
                expected_pretraining_audit_receipt_sha256
            ),
            expected_pretraining_audit_receipt_file_sha256=(
                expected_pretraining_audit_receipt_file_sha256
            ),
            data_root=_resolve(data_root),
            registry=registry,
            recompute_epoch_validation=True,
            expected_run_reservation=reservation_by_checkpoint[
                str(checkpoint_path.resolve())
            ],
        )
        identity = (str(row["registered_name"]), int(row["seed"]))
        if identity in observed:
            raise ValueError(f"Duplicate checkpoint release identity: {identity}")
        observed.add(identity)
        precision_hashes.add(str(row["precision_policy_sha256"]))
        rows.append(row)
    if observed != expected:
        raise ValueError(
            f"Checkpoint release identity mismatch; missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    if len(precision_hashes) != 1:
        raise ValueError("Confirmatory checkpoints use inconsistent precision policies")
    audit_receipt_hashes = {
        str(row["pretraining_audit_receipt_sha256"]) for row in rows
    }
    audit_receipt_file_hashes = {
        str(row["pretraining_audit_receipt_file_sha256"]) for row in rows
    }
    if len(audit_receipt_hashes) != 1:
        raise ValueError("Confirmatory checkpoints use inconsistent pretraining audit receipts")
    if audit_receipt_hashes != {str(expected_pretraining_audit_receipt_sha256)}:
        raise ValueError(
            "Confirmatory checkpoints differ from external pretraining audit anchor"
        )
    if audit_receipt_file_hashes != {
        str(expected_pretraining_audit_receipt_file_sha256)
    }:
        raise ValueError(
            "Confirmatory checkpoints differ from the external audit-receipt file anchor"
        )
    rows.sort(key=lambda row: (row["registered_name"], row["seed"]))
    core = {
        "schema_version": 1,
        "source_config_sha256": canonical_json_hash(config),
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "parameter_registry_path": str(_resolve(registry_path)),
        "data_root": str(_resolve(data_root)),
        "implementation_sha256": registry["implementation"]["aggregate_sha256"],
        "formal_artifact_identities": identities,
        "expected_checkpoint_count": len(expected),
        "checkpoints": rows,
        "precision_policy_sha256": next(iter(precision_hashes)),
        "pretraining_master_sha256": str(expected_pretraining_master_sha256),
        "pretraining_audit_receipt_sha256": next(iter(audit_receipt_hashes)),
        "pretraining_audit_receipt_file_sha256": next(
            iter(audit_receipt_file_hashes)
        ),
        "run_registry_namespace_sha256": validated_master[
            "run_registry_namespace_sha256"
        ],
        "run_registry_instance_id": validated_master[
            "run_registry_instance_id"
        ],
        "initial_registry_sha256": validated_master[
            "initial_registry_sha256"
        ],
        "run_registry_custody_sha256": run_registry_receipt[
            "run_registry_custody_sha256"
        ],
        "run_registry_receipt": run_registry_receipt,
        "run_registry_receipt_sha256": run_registry_receipt[
            "run_registry_receipt_sha256"
        ],
        "all_checkpoints_frozen_before_test": True,
        "test_episodes_used": 0,
        "test_schedules_materialized": False,
    }
    artifact = {
        **core,
        "checkpoint_release_sha256": canonical_json_hash(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(destination, artifact)
    return artifact


def validate_checkpoint_release(
    config: dict[str, Any],
    trajectory: dict[str, Any],
    checkpoint_paths: list[str | Path],
    *,
    expected_master_release_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not is_formal(config):
        return None
    path = config.get("evaluation", {}).get("checkpoint_release_artifact")
    if not path:
        raise ValueError("Formal config lacks evaluation.checkpoint_release_artifact")
    artifact = _load_hashed(path, "checkpoint_release_sha256")
    if (
        not expected_master_release_sha256
        or artifact.get("checkpoint_release_sha256")
        != str(expected_master_release_sha256)
    ):
        raise ValueError("Checkpoint release differs from the external master anchor")
    expected = {
        "source_config_sha256": canonical_json_hash(config),
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"Checkpoint release identity mismatch for {key}")
    supplied = {str(_resolve(path)) for path in checkpoint_paths}
    released = {row["checkpoint_path"] for row in artifact.get("checkpoints", [])}
    if supplied != released or len(supplied) != int(
        artifact.get("expected_checkpoint_count", -1)
    ):
        raise ValueError("Evaluation checkpoint paths differ from frozen release")
    precision_hashes: set[str] = set()
    registry = load_parameter_registry(
        artifact.get("parameter_registry_path", ""), expected_config=config
    )
    if registry.get("registry_sha256") != artifact.get("parameter_registry_sha256"):
        raise ValueError("Checkpoint release registry payload/hash mismatch")
    data_root = _resolve(artifact.get("data_root", ""))
    from .runner import validate_complete_run_registry

    expected_run_identities = {
        f"{model_name}::seed_{int(seed)}"
        for model_name in config.get("matching", {}).get(
            "registered_models", REGISTERED_MODELS
        )
        for seed in config.get("training", {}).get("seeds", [])
    }
    frozen_registry_receipt = artifact.get("run_registry_receipt")
    if not isinstance(frozen_registry_receipt, dict):
        raise ValueError("Checkpoint release lacks the complete run-registry receipt")
    checked_registry_receipt = validate_complete_run_registry(
        frozen_registry_receipt.get("run_registry_path", ""),
        expected_config_sha256=canonical_json_hash(config),
        expected_pretraining_master_sha256=artifact[
            "pretraining_master_sha256"
        ],
        expected_pretraining_audit_receipt_sha256=artifact[
            "pretraining_audit_receipt_sha256"
        ],
        expected_run_registry_namespace_sha256=artifact[
            "run_registry_namespace_sha256"
        ],
        expected_identities=expected_run_identities,
        expected_pretraining_audit_receipt_file_sha256=artifact[
            "pretraining_audit_receipt_file_sha256"
        ],
        require_frozen=True,
    )
    if (
        checked_registry_receipt != frozen_registry_receipt
        or checked_registry_receipt["run_registry_receipt_sha256"]
        != artifact.get("run_registry_receipt_sha256")
    ):
        raise ValueError("Checkpoint release run-registry receipt changed")
    reservation_by_checkpoint = {
        str(Path(row["checkpoint_path"]).resolve()): {
            "run_registry_path": checked_registry_receipt["run_registry_path"],
            "run_registry_namespace_sha256": checked_registry_receipt[
                "run_registry_namespace_sha256"
            ],
            "run_registry_instance_id": checked_registry_receipt[
                "run_registry_instance_id"
            ],
            "initial_registry_sha256": checked_registry_receipt[
                "initial_registry_sha256"
            ],
            "run_registry_custody_sha256": checked_registry_receipt[
                "run_registry_custody_sha256"
            ],
            "attempt_id": row["complete_attempt_id"],
            **next(
                attempt
                for attempt in row["attempts"]
                if attempt["attempt_id"] == row["complete_attempt_id"]
            ),
        }
        for row in checked_registry_receipt["slots"]
    }
    for row in artifact["checkpoints"]:
        checkpoint_path = Path(row["checkpoint_path"])
        checked = _validate_released_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            trajectory_sha256=artifact["trajectory_manifest_sha256"],
            registry_sha256=artifact["parameter_registry_sha256"],
            implementation_sha256=artifact["implementation_sha256"],
            formal_identities=artifact["formal_artifact_identities"],
            expected_precision_sha256=artifact["precision_policy_sha256"],
            expected_pretraining_master_sha256=artifact[
                "pretraining_master_sha256"
            ],
            expected_pretraining_audit_receipt_sha256=artifact[
                "pretraining_audit_receipt_sha256"
            ],
            expected_pretraining_audit_receipt_file_sha256=artifact[
                "pretraining_audit_receipt_file_sha256"
            ],
            data_root=data_root,
            registry=registry,
            recompute_epoch_validation=False,
            expected_run_reservation=reservation_by_checkpoint[
                str(checkpoint_path.resolve())
            ],
        )
        checked_receipt = checked.get("epoch_validation_receipt", {})
        released_receipt = row.get("epoch_validation_receipt", {})
        # The release-time receipt contains the expensive independent replay;
        # subsequent anchored validation rechecks all immutable epoch files and
        # selection relations while preserving that original replay receipt.
        checked["epoch_validation_receipt"] = released_receipt
        if (
            checked_receipt.get("checkpoint_ledger_sha256")
            != released_receipt.get("checkpoint_ledger_sha256")
            or checked_receipt.get("ledger_epoch_rows_sha256")
            != released_receipt.get("ledger_epoch_rows_sha256")
            or released_receipt.get("validation_recomputed") is not True
            or checked != row
        ):
            raise ValueError("Checkpoint release row differs from deep validation")
        precision_hashes.add(str(row["precision_policy_sha256"]))
        physical = row.get("selected_physical_gpu")
        if not isinstance(physical, dict) or not physical.get("uuid") or not physical.get(
            "pci_bus_id"
        ):
            raise ValueError("Checkpoint release lacks physical GPU mapping")
    if precision_hashes != {artifact.get("precision_policy_sha256")}:
        raise ValueError("Checkpoint release precision-policy mismatch")
    if not artifact.get("all_checkpoints_frozen_before_test") or int(
        artifact.get("test_episodes_used", -1)
    ) != 0:
        raise ValueError("Checkpoint release is not pre-test")
    return artifact


def _test_authorization_design(
    config: dict[str, Any],
    checkpoint_release: Mapping[str, Any],
    evaluation_output_dir: str | Path,
) -> dict[str, Any]:
    """Static test-evaluation identity available before schedules are opened."""
    core = {
        "schema_version": 1,
        "source_config_sha256": canonical_json_hash(config),
        "checkpoint_release_sha256": checkpoint_release[
            "checkpoint_release_sha256"
        ],
        "trajectory_manifest_sha256": checkpoint_release[
            "trajectory_manifest_sha256"
        ],
        "parameter_registry_sha256": checkpoint_release[
            "parameter_registry_sha256"
        ],
        "checkpoint_identities_sha256": canonical_json_hash(
            [
                {
                    "registered_name": row["registered_name"],
                    "seed": int(row["seed"]),
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
                for row in checkpoint_release["checkpoints"]
            ]
        ),
        "evaluation_spec_sha256": canonical_json_hash(
            config.get("evaluation", {})
        ),
        "authorized_evaluation_output_dir": str(
            Path(evaluation_output_dir).resolve()
        ),
    }
    return {**core, "test_authorization_design_sha256": canonical_json_hash(core)}


def _validate_test_claim_custody_receipt(
    *,
    receipt_path: str | Path,
    expected_receipt_file_sha256: str,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(receipt_path).resolve()
    if (
        len(str(expected_receipt_file_sha256)) != 64
        or sha256_file(source) != str(expected_receipt_file_sha256)
    ):
        raise PermissionError(
            "Test-claim custody receipt differs from its external file anchor"
        )
    receipt = _load_hashed(source, "test_claim_custody_receipt_sha256")
    expected = {
        "schema_version": 1,
        "action": "authorize_one_closed_loop_test_evaluation",
        "mode": "auditor_managed_append_only",
        "acl_policy": "create_only_no_modify_no_delete_existing_objects",
        "test_authorization_design_sha256": design[
            "test_authorization_design_sha256"
        ],
        "checkpoint_release_sha256": design["checkpoint_release_sha256"],
        "authorized_evaluation_output_dir": design[
            "authorized_evaluation_output_dir"
        ],
        "test_episodes_used_before_authorization": 0,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise PermissionError(
                f"Test-claim custody receipt mismatch for {key}"
            )
    claim_path = Path(str(receipt.get("claim_state_path", "")))
    if not claim_path.is_absolute() or not str(
        receipt.get("external_attestation_id", "")
    ).strip():
        raise PermissionError(
            "Test-claim custody receipt lacks an absolute append-only claim path/attestation"
        )
    output = Path(design["authorized_evaluation_output_dir"]).resolve()
    resolved_claim = claim_path.resolve()
    if resolved_claim == output or output in resolved_claim.parents:
        raise PermissionError(
            "Test claim state must be held outside the evaluation output tree"
        )
    if not resolved_claim.parent.is_dir():
        raise FileNotFoundError(
            "Auditor-managed test-claim custody directory is not provisioned"
        )
    return receipt


def create_test_release(
    config: dict[str, Any],
    output_path: str | Path,
    *,
    expected_master_release_sha256: str,
    authorized_evaluation_output_dir: str | Path,
    test_claim_custody_receipt_path: str | Path,
    expected_test_claim_custody_receipt_sha256: str,
) -> dict[str, Any]:
    """Create a separate, immutable authorization after checkpoint release."""
    destination = _resolve(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite test release: {destination}")
    checkpoint_path = config.get("evaluation", {}).get(
        "checkpoint_release_artifact"
    )
    if not checkpoint_path:
        raise ValueError("Test authorization lacks checkpoint release path")
    checkpoint_release = _load_hashed(
        checkpoint_path, "checkpoint_release_sha256"
    )
    checkpoint_release = validate_checkpoint_release(
        config,
        {"manifest_sha256": checkpoint_release["trajectory_manifest_sha256"]},
        [row["checkpoint_path"] for row in checkpoint_release["checkpoints"]],
        expected_master_release_sha256=str(expected_master_release_sha256),
    )
    if checkpoint_release is None:
        raise ValueError("Test authorization requires formal checkpoint release")
    if (
        not expected_master_release_sha256
        or checkpoint_release["checkpoint_release_sha256"]
        != str(expected_master_release_sha256)
    ):
        raise ValueError(
            "Explicit independently published master release anchor does not "
            "match the deeply revalidated checkpoint release"
        )
    if not checkpoint_release.get("all_checkpoints_frozen_before_test"):
        raise ValueError("Checkpoint release is not complete")
    design = _test_authorization_design(
        config, checkpoint_release, authorized_evaluation_output_dir
    )
    custody = _validate_test_claim_custody_receipt(
        receipt_path=test_claim_custody_receipt_path,
        expected_receipt_file_sha256=(
            expected_test_claim_custody_receipt_sha256
        ),
        design=design,
    )
    core = {
        "schema_version": 1,
        "source_config_sha256": canonical_json_hash(config),
        "checkpoint_release_sha256": checkpoint_release[
            "checkpoint_release_sha256"
        ],
        "expected_master_release_sha256": str(expected_master_release_sha256),
        "anchor_source": "explicit_external_value_not_derived_by_authorizer",
        "expected_checkpoint_count": checkpoint_release[
            "expected_checkpoint_count"
        ],
        "test_authorization_design": design,
        "test_authorization_design_sha256": design[
            "test_authorization_design_sha256"
        ],
        "authorized_evaluation_output_dir": design[
            "authorized_evaluation_output_dir"
        ],
        "test_claim_custody_receipt_path": str(
            Path(test_claim_custody_receipt_path).resolve()
        ),
        "test_claim_custody_receipt_sha256": custody[
            "test_claim_custody_receipt_sha256"
        ],
        "test_claim_custody_receipt_file_sha256": str(
            expected_test_claim_custody_receipt_sha256
        ),
        "test_claim_state_path": str(Path(custody["claim_state_path"]).resolve()),
        "authorization": "closed_loop_test_evaluation_unlocked",
        "test_episodes_used_before_authorization": 0,
    }
    artifact = {
        **core,
        "test_release_sha256": canonical_json_hash(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(destination, artifact)
    return artifact


def validate_test_release(
    config: dict[str, Any],
    checkpoint_release: dict[str, Any] | None,
    *,
    expected_master_release_sha256: str | None = None,
    expected_test_release_sha256: str | None = None,
    evaluation_output_dir: str | Path | None = None,
    test_claim_custody_receipt_path: str | Path | None = None,
    expected_test_claim_custody_receipt_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not is_formal(config):
        return None
    if checkpoint_release is None:
        raise ValueError("Formal test release requires checkpoint release")
    path = config.get("evaluation", {}).get("test_release_artifact")
    if not path:
        raise ValueError("Formal config lacks evaluation.test_release_artifact")
    artifact = _load_hashed(path, "test_release_sha256")
    if not evaluation_output_dir:
        raise PermissionError("Formal test validation requires the authorized output")
    design = _test_authorization_design(
        config, checkpoint_release, evaluation_output_dir
    )
    if not test_claim_custody_receipt_path or not expected_test_claim_custody_receipt_sha256:
        raise PermissionError(
            "Formal test validation requires an externally anchored claim-custody receipt"
        )
    custody = _validate_test_claim_custody_receipt(
        receipt_path=test_claim_custody_receipt_path,
        expected_receipt_file_sha256=(
            expected_test_claim_custody_receipt_sha256
        ),
        design=design,
    )
    if (
        not expected_master_release_sha256
        or not expected_test_release_sha256
        or artifact.get("test_release_sha256")
        != str(expected_test_release_sha256)
        or str(expected_master_release_sha256)
        != checkpoint_release["checkpoint_release_sha256"]
        or artifact.get("expected_master_release_sha256")
        != str(expected_master_release_sha256)
        or
        artifact.get("source_config_sha256") != canonical_json_hash(config)
        or artifact.get("checkpoint_release_sha256")
        != checkpoint_release["checkpoint_release_sha256"]
        or artifact.get("test_authorization_design") != design
        or artifact.get("test_authorization_design_sha256")
        != design["test_authorization_design_sha256"]
        or artifact.get("authorized_evaluation_output_dir")
        != design["authorized_evaluation_output_dir"]
        or artifact.get("test_claim_custody_receipt_path")
        != str(Path(test_claim_custody_receipt_path).resolve())
        or artifact.get("test_claim_custody_receipt_sha256")
        != custody["test_claim_custody_receipt_sha256"]
        or artifact.get("test_claim_custody_receipt_file_sha256")
        != str(expected_test_claim_custody_receipt_sha256)
        or artifact.get("test_claim_state_path")
        != str(Path(custody["claim_state_path"]).resolve())
        or artifact.get("authorization")
        != "closed_loop_test_evaluation_unlocked"
        or artifact.get("expected_master_release_sha256")
        != checkpoint_release["checkpoint_release_sha256"]
        or artifact.get("anchor_source")
        != "explicit_external_value_not_derived_by_authorizer"
        or int(artifact.get("test_episodes_used_before_authorization", -1)) != 0
    ):
        raise ValueError("Formal test-release authorization mismatch")
    return {**artifact, "_validated_test_claim_custody": custody}


def claim_test_release(
    config: dict[str, Any],
    test_release: dict[str, Any] | None,
    *,
    output_dir: str | Path,
    run_identity_sha256: str,
    resume: bool,
) -> dict[str, Any] | None:
    """Atomically bind a formal test authorization to exactly one run directory."""
    if not is_formal(config):
        return None
    if test_release is None:
        raise ValueError("Formal test claim requires a validated test release")
    custody = test_release.get("_validated_test_claim_custody")
    if not isinstance(custody, Mapping):
        raise PermissionError("Formal test claim lacks validated external custody")
    if str(Path(output_dir).resolve()) != test_release.get(
        "authorized_evaluation_output_dir"
    ):
        raise PermissionError("Formal test claim output differs from authorization")
    claim_path = Path(custody["claim_state_path"]).resolve()
    static_core = {
        "schema_version": 1,
        "source_config_sha256": canonical_json_hash(config),
        "test_release_sha256": test_release["test_release_sha256"],
        "checkpoint_release_sha256": test_release[
            "checkpoint_release_sha256"
        ],
        "claimed_output_dir": str(Path(output_dir).resolve()),
        "run_identity_sha256": str(run_identity_sha256),
        "test_authorization_design_sha256": test_release[
            "test_authorization_design_sha256"
        ],
        "test_claim_custody_receipt_sha256": custody[
            "test_claim_custody_receipt_sha256"
        ],
        "test_claim_custody_receipt_file_sha256": test_release[
            "test_claim_custody_receipt_file_sha256"
        ],
        "external_attestation_id": custody["external_attestation_id"],
        "authorization_state": "claimed_once",
    }
    if not claim_path.parent.is_dir():
        raise FileNotFoundError(
            "Auditor-managed test-claim custody directory is not provisioned"
        )
    if claim_path.exists():
        mutation_blocked = False
        try:
            descriptor = os.open(claim_path, os.O_WRONLY | os.O_APPEND)
        except PermissionError:
            mutation_blocked = True
        else:
            os.close(descriptor)
        if not mutation_blocked:
            raise PermissionError(
                "Auditor-managed test claim is mutable by the evaluation worker"
            )
        deletion_blocked = False
        try:
            claim_path.unlink()
        except PermissionError:
            deletion_blocked = True
        if not deletion_blocked:
            raise PermissionError(
                "Auditor-managed test claim was deletable by the evaluation worker"
            )
        existing = _load_hashed(claim_path, "test_release_state_sha256")
        if not resume:
            raise ValueError(
                "Formal test authorization was already claimed; only the bound "
                "output directory may resume"
            )
        for key, value in static_core.items():
            if existing.get(key) != value:
                raise ValueError(f"Formal test-release claim mismatch for {key}")
        evidence = existing.get("append_only_custody_evidence")
        if (
            not isinstance(evidence, dict)
            or evidence.get("journal_directory") != str(claim_path.parent)
            or evidence.get("mutation_blocked") is not True
            or evidence.get("deletion_blocked") is not True
            or not Path(str(evidence.get("probe_path", ""))).is_file()
            or sha256_file(evidence["probe_path"])
            != evidence.get("probe_file_sha256")
        ):
            raise PermissionError("Stored test-claim custody evidence is invalid")
        return existing

    from .runner import _verify_append_only_custody

    evidence = _verify_append_only_custody(
        claim_path.parent,
        run_registry_instance_id=(
            "test_" + test_release["test_authorization_design_sha256"][:32]
        ),
    )
    core = {**static_core, "append_only_custody_evidence": evidence}
    payload = {**core, "test_release_state_sha256": canonical_json_hash(core)}
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        )
    except FileExistsError as error:
        raise RuntimeError(
            "Test claim appeared concurrently; rerun only with explicit --resume"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    # The claim itself must immediately inherit the same no-modify/no-delete
    # policy.  A permissive folder fails before any test schedule is opened.
    mutation_blocked = False
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_APPEND)
    except PermissionError:
        mutation_blocked = True
    else:
        os.close(descriptor)
    if not mutation_blocked:
        raise PermissionError(
            "New test claim is mutable; external append-only custody is not active"
        )
    deletion_blocked = False
    try:
        claim_path.unlink()
    except PermissionError:
        deletion_blocked = True
    if not deletion_blocked:
        raise PermissionError(
            "New test claim is deletable; external append-only custody is not active"
        )
    return payload


def formal_artifact_identities(
    config: dict[str, Any],
    *,
    registry: dict[str, Any] | None = None,
    trajectory_manifest: dict[str, Any] | None = None,
    require_tuning: bool = False,
) -> dict[str, str]:
    if not is_formal(config):
        return {"mode": "development_only"}
    environment = validate_environment_freeze(config)
    controller = validate_controller_selection(config)
    result = {
        "environment_sha256": environment["environment_sha256"],
        "controller_selection_sha256": controller["controller_selection_sha256"],
    }
    if controller.get("environment_design_sha256") != environment["environment_sha256"]:
        raise ValueError(
            "Controller selection and final environment freeze use different design identities"
        )
    if require_tuning:
        if registry is None or trajectory_manifest is None:
            raise ValueError("Tuning validation requires registry and trajectory manifest")
        tuning = validate_tuning_selection(config, registry, trajectory_manifest)
        result["tuning_selection_sha256"] = tuning["tuning_selection_sha256"]
    return result
