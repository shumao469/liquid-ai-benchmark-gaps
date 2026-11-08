"""Two independent GPU workers, one writer, and a sealed train-only snapshot.

Each child owns one model/seed directory.  Only the supervisor writes the
aggregate and publishes the exact 40-checkpoint release.  No command here loads
test arrays or evaluates a test condition.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

from ..config import load_config
from ..io_utils import atomic_write_json, canonical_json_hash, sha256_file, source_tree_manifest
from .formal import (
    create_formal_freeze,
    expected_run_identities,
    publish_checkpoint_release,
    utc_now,
    verify_completed_run,
    verify_formal_freeze,
)
from .training import load_all_datasets, train_fixed_budget


SMI = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32" / "nvidia-smi.exe"


def gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [str(SMI), "--query-gpu=index,uuid,name,temperature.gpu,power.limit,utilization.gpu,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise ValueError("Unexpected physical GPU inventory")
        rows.append({
            "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "temperature_c": float(fields[3]), "power_limit_w": float(fields[4]),
            "utilization_percent": float(fields[5]), "memory_free_mib": float(fields[6]),
        })
    if len({row["uuid"] for row in rows}) != len(rows):
        raise ValueError("Duplicate GPU identity")
    return rows


def assign_longest_first(
    identities: list[dict[str, Any]],
    gpu_uuids: list[str],
    costs: dict[str, float],
    speed_factors: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic LPT assignment; each run retains its physical GPU on resume."""
    if not gpu_uuids or len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("GPU UUIDs must be nonempty and unique")
    run_ids = [str(item["run_id"]) for item in identities]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate run identity")
    queues: dict[str, list[dict[str, Any]]] = {uuid: [] for uuid in gpu_uuids}
    totals = {uuid: 0.0 for uuid in gpu_uuids}
    speeds = {uuid: float((speed_factors or {}).get(uuid, 1.0)) for uuid in gpu_uuids}
    if any(value <= 0 for value in speeds.values()):
        raise ValueError("GPU speed factors must be positive")
    ordered = sorted(
        identities,
        key=lambda row: (-float(costs.get(f"{row['dataset']}/{row['model']}", 1.0)), row["run_id"]),
    )
    for row in ordered:
        cost = float(costs.get(f"{row['dataset']}/{row['model']}", 1.0))
        selected = min(gpu_uuids, key=lambda uuid: (
            totals[uuid] + cost / speeds[uuid], gpu_uuids.index(uuid)))
        queues[selected].append(dict(row))
        totals[selected] += cost / speeds[selected]
    return queues


def science_identity(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for key in ("root", "manifest_dir"):
        result.get("data", {}).pop(key, None)
    result.get("formal", {}).pop("protocol_files", None)
    return result


def create_snapshot(config_path: Path, output_root: Path) -> dict[str, Any]:
    """Copy the numerical implementation and documents before any formal fitting."""
    if output_root.exists():
        raise FileExistsError(f"Snapshot output must be new: {output_root}")
    original, original_hash = load_config(config_path)
    source = Path(__file__).resolve().parents[1]
    source_before = source_tree_manifest(source)
    output_root.mkdir(parents=True, exist_ok=False)
    snapshot = output_root / "snapshot"
    package = snapshot / "src" / "liquid_revision"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_after = source_tree_manifest(source)
    if source_before != source_after:
        raise RuntimeError("Source changed during snapshot; no fitting has started")
    copied = source_tree_manifest(package)
    if copied["files"] != source_before["files"]:
        raise RuntimeError("Snapshot source copy differs")
    frozen_config = copy.deepcopy(original)
    for key in ("root", "manifest_dir"):
        path = Path(str(original["data"][key]))
        frozen_config["data"][key] = str(
            path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()
        )
    protocols = []
    for index, value in enumerate(original.get("formal", {}).get("protocol_files", [])):
        source_doc = Path(str(value))
        if not source_doc.is_absolute():
            source_doc = (config_path.parent / source_doc).resolve()
        destination = snapshot / "protocol" / f"{index}_{source_doc.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_doc, destination)
        if sha256_file(source_doc) != sha256_file(destination):
            raise RuntimeError("Protocol changed during snapshot")
        protocols.append(str(destination.resolve()))
    frozen_config["formal"]["protocol_files"] = protocols
    if science_identity(original) != science_identity(frozen_config):
        raise RuntimeError("Snapshot changed a scientific setting")
    frozen_path = snapshot / "configs" / config_path.name
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    with frozen_path.open("x", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(frozen_config, handle, sort_keys=False, allow_unicode=True)
    _, frozen_hash = load_config(frozen_path)
    receipt = {
        "created_at_utc": utc_now(), "source_config": str(config_path),
        "source_config_sha256": original_hash, "snapshot_config": str(frozen_path),
        "snapshot_config_sha256": frozen_hash,
        "source_tree": copied,
        "science_settings_unchanged": True,
        "changed_config_fields": ["data.root", "data.manifest_dir", "formal.protocol_files"],
        "test_metrics_computed": False,
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt)
    atomic_write_json(output_root / "snapshot_receipt.json", receipt)
    return receipt


def _costs(path: Path | None) -> dict[str, float]:
    if path is None:
        return {"har/ltc": 10.5, "har/ode-rnn": 3.2, "har/cfc": 2.1,
                "physionet2012/ltc": 2.8, "physionet2012/ode-rnn": 1.0,
                "physionet2012/cfc": 0.6, "har/lstm": 0.11, "physionet2012/lstm": 0.09}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"{row['dataset']}/{row['model']}": float(row["components_seconds"]["optimizer_steps"])
        for row in payload["runtime_forecast"]["per_model_seed"]
    }


def worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    config_path = Path(args.config).resolve()
    output = Path(args.output_dir).resolve()
    config, config_hash = load_config(config_path)
    freeze = verify_formal_freeze(
        config=config, config_path=config_path, config_hash=config_hash, output_root=output,
    )
    identity = next(
        (row for row in freeze["expected_runs"] if row["run_id"] == args.run_id), None
    )
    if identity is None:
        raise ValueError("Worker run is outside the frozen exact set")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("Worker physical GPU affinity differs")
    datasets = load_all_datasets(
        config, config_path, selected_datasets=[identity["dataset"]], include_test=False,
    )
    train_fixed_budget(
        dataset_name=identity["dataset"], model_name=identity["model"], seed=identity["seed"],
        data=datasets[identity["dataset"]], config=config, config_hash=config_hash,
        output_root=output / "runs", formal_context=freeze,
        stop_request_path=output / "control" / f"{args.gpu_uuid}.stop.json",
    )


def supervise(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    output = Path(args.output_dir).resolve()
    config, config_hash = load_config(config_path)
    if not config.get("formal", {}).get("enabled"):
        raise ValueError("Parallel supervisor requires the formal checkpoint gate")
    lock_path = output / "supervisor.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_fd, json.dumps({"pid": os.getpid(), "created_at_utc": utc_now()}).encode())
    os.close(lock_fd)
    state_path = output / "parallel_status.json"
    active: dict[str, dict[str, Any]] = {}
    workers: dict[str, dict[str, Any]] = {}
    changed_caps: dict[str, float] = {}
    state: dict[str, Any] = {
        "status": "preparing", "pid": os.getpid(), "started_at_utc": utc_now(),
        "gpu_uuids": args.gpu_uuids, "cpu_threads_per_worker": args.cpu_threads,
        "test_metrics_computed": False, "completed_runs": [], "active": {},
        "thermal_stop_c": args.thermal_stop_c, "thermal_resume_c": args.thermal_resume_c,
    }
    atomic_write_json(state_path, state)
    try:
        gpus = {row["uuid"]: row for row in gpu_inventory()}
        for uuid in args.gpu_uuids:
            if uuid not in gpus or gpus[uuid]["memory_free_mib"] < 4096:
                raise RuntimeError(f"GPU unavailable or insufficient free memory: {uuid}")
            if gpus[uuid]["temperature_c"] >= args.thermal_resume_c:
                raise RuntimeError(f"GPU must cool before admission: {uuid}")
            if gpus[uuid]["utilization_percent"] > args.admission_utilization:
                raise RuntimeError(f"GPU has another active workload: {uuid}")
            if "RTX PRO 5000" in gpus[uuid]["name"]:
                previous = float(gpus[uuid]["power_limit_w"])
                changed_caps[uuid] = float(getattr(args, "gpu1_restore_w", None) or previous)
                if previous > 250:
                    subprocess.run([str(SMI), "-i", uuid, "-pl", "250"], check=True,
                                   capture_output=True, text=True, timeout=15)
        datasets = load_all_datasets(config, config_path, include_test=False)
        freeze_path = output / "formal_freeze.json"
        if freeze_path.exists():
            freeze = verify_formal_freeze(
                config=config, config_path=config_path, config_hash=config_hash, output_root=output,
            )
        else:
            freeze = create_formal_freeze(
                config=config, config_path=config_path, config_hash=config_hash,
                datasets=datasets, output_root=output,
            )
        del datasets
        plan_path = output / "parallel_plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if plan["config_sha256"] != config_hash or plan["gpu_uuids"] != args.gpu_uuids:
                raise ValueError("Parallel plan config/GPU assignment changed")
        else:
            speeds = {uuid: 1.0 for uuid in args.gpu_uuids}
            profile_files = getattr(args, "speed_profile_files", []) or []
            if profile_files:
                profiles = {}
                for profile_file in profile_files:
                    profile = json.loads(Path(profile_file).read_text(encoding="utf-8"))
                    uuid = profile["accelerator"]["selected_physical_gpu"]["uuid"]
                    profiles[uuid] = {
                        row["model"]: float(row["median_step_seconds"])
                        for row in profile["rows"] if row["cpu_threads"] == args.cpu_threads
                    }
                anchor = profiles[args.gpu_uuids[0]]["ltc"]
                speeds = {uuid: anchor / profiles[uuid]["ltc"] for uuid in args.gpu_uuids}
            plan = {
                "config_sha256": config_hash, "gpu_uuids": args.gpu_uuids,
                "speed_factors": speeds,
                "queues": assign_longest_first(
                    expected_run_identities(config, config_hash), args.gpu_uuids,
                    _costs(Path(args.cost_profile) if args.cost_profile else None),
                    speeds,
                ),
                "assignment": "longest first, fixed physical UUID per run",
            }
            plan["plan_sha256"] = canonical_json_hash(plan)
            atomic_write_json(plan_path, plan)
        queues = {uuid: list(rows) for uuid, rows in plan["queues"].items()}
        for uuid, queue in queues.items():
            remaining = []
            for identity in queue:
                run_dir = output / "runs" / identity["run_id"]
                if (run_dir / "result.json").exists():
                    verify_completed_run(
                        run_dir=run_dir, identity=identity, config_hash=config_hash,
                        formal_freeze_sha256=freeze["formal_freeze_sha256"],
                    )
                    state["completed_runs"].append(identity["run_id"])
                else:
                    remaining.append(identity)
            queues[uuid] = remaining
        state["status"] = "training"
        state["formal_freeze_sha256"] = freeze["formal_freeze_sha256"]
        state["expected_runs"] = len(freeze["expected_runs"])
        (output / "control").mkdir(exist_ok=True)
        (output / "logs").mkdir(exist_ok=True)
        while any(queues.values()) or active:
            current = {row["uuid"]: row for row in gpu_inventory()}
            for uuid in args.gpu_uuids:
                telemetry = current[uuid]
                stop_path = output / "control" / f"{uuid}.stop.json"
                if uuid in active:
                    item = active[uuid]
                    process = item["process"]
                    exit_code = process.poll()
                    if exit_code is not None:
                        item["log"].close()
                        identity = item["identity"]
                        workers.pop(uuid, None)
                        del active[uuid]
                        if exit_code == 0:
                            verify_completed_run(
                                run_dir=output / "runs" / identity["run_id"],
                                identity=identity, config_hash=config_hash,
                                formal_freeze_sha256=freeze["formal_freeze_sha256"],
                            )
                            state["completed_runs"].append(identity["run_id"])
                        elif stop_path.is_file():
                            queues[uuid].insert(0, identity)
                        else:
                            raise RuntimeError(
                                f"Worker {identity['run_id']} failed with exit code {exit_code}; see logs"
                            )
                    elif telemetry["temperature_c"] >= args.thermal_stop_c:
                        if not stop_path.exists():
                            atomic_write_json(stop_path, {"reason": "thermal_pause",
                                "temperature_c": telemetry["temperature_c"], "at_utc": utc_now()})
                    continue
                if not queues[uuid] or telemetry["temperature_c"] > args.thermal_resume_c:
                    continue
                if telemetry["utilization_percent"] > args.admission_utilization:
                    continue
                stop_path.unlink(missing_ok=True)
                identity = queues[uuid].pop(0)
                environment = os.environ.copy()
                environment.update({"CUDA_VISIBLE_DEVICES": uuid,
                    "OMP_NUM_THREADS": str(args.cpu_threads), "MKL_NUM_THREADS": str(args.cpu_threads),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
                command = [sys.executable, "-m", "liquid_revision.offline.parallel", "worker",
                    "--config", str(config_path), "--output-dir", str(output),
                    "--run-id", identity["run_id"], "--gpu-uuid", uuid,
                    "--cpu-threads", str(args.cpu_threads)]
                log = (output / "logs" / f"{identity['run_id']}.log").open("a", encoding="utf-8")
                process = subprocess.Popen(command, env=environment, cwd=output,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                active[uuid] = {"process": process, "log": log, "identity": identity}
                workers[uuid] = {"pid": process.pid, **identity, "started_at_utc": utc_now()}
            state.update({"active": workers, "updated_at_utc": utc_now(),
                          "gpu_telemetry": list(current.values()),
                          "pending_count": sum(len(queue) for queue in queues.values())})
            atomic_write_json(state_path, state)
            time.sleep(args.poll_seconds)
        release = publish_checkpoint_release(
            config=config, config_hash=config_hash, output_root=output, formal_freeze=freeze,
        )
        state.update({"status": "checkpoint_release_frozen", "completed_at_utc": utc_now(),
                      "checkpoint_release_sha256": release["checkpoint_release_sha256"]})
    except BaseException as exc:
        state.update({"status": "failed_or_interrupted", "exception": str(exc),
                      "traceback": traceback.format_exc(), "ended_at_utc": utc_now()})
        for uuid in active:
            atomic_write_json(output / "control" / f"{uuid}.stop.json",
                              {"reason": "supervisor_stopping", "at_utc": utc_now()})
        for item in active.values():
            try:
                item["process"].wait(timeout=45)
            except subprocess.TimeoutExpired:
                item["process"].terminate()
                item["process"].wait(timeout=15)
            item["log"].close()
        raise
    finally:
        state["power_restore"] = []
        for uuid, previous in changed_caps.items():
            restored = subprocess.run([str(SMI), "-i", uuid, "-pl", str(int(previous))],
                capture_output=True, text=True, timeout=15)
            state["power_restore"].append({"uuid": uuid, "target_w": previous,
                "success": restored.returncode == 0, "stdout": restored.stdout,
                "stderr": restored.stderr})
        atomic_write_json(state_path, state)
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--config", required=True)
    snapshot.add_argument("--output-dir", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--gpu-uuids", nargs="+", required=True)
    run.add_argument("--cost-profile")
    run.add_argument("--speed-profile-files", nargs="*")
    run.add_argument("--gpu1-restore-w", type=float)
    run.add_argument("--cpu-threads", type=int, default=1)
    run.add_argument("--thermal-stop-c", type=float, default=82)
    run.add_argument("--thermal-resume-c", type=float, default=70)
    run.add_argument("--admission-utilization", type=float, default=20)
    run.add_argument("--poll-seconds", type=float, default=5)
    child = sub.add_parser("worker")
    child.add_argument("--config", required=True)
    child.add_argument("--output-dir", required=True)
    child.add_argument("--run-id", required=True)
    child.add_argument("--gpu-uuid", required=True)
    child.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.command == "snapshot":
        print(json.dumps(create_snapshot(Path(args.config).resolve(), Path(args.output_dir).resolve()), indent=2))
    elif args.command == "run":
        supervise(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
