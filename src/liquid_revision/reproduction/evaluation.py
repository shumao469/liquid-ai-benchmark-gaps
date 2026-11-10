"""Run and summarize the paired exploratory Reacher evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import tempfile
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from ..closed_loop import benchmark as bench
from ..config import load_config
from ..io_utils import atomic_write_json, canonical_json_hash, sha256_file, source_tree_manifest


MODELS = ("lstm", "odernn", "ctrnn", "ltc", "cfc", "ncp_cfc", "random_sparse_cfc")
REFERENCES = ("full_state_pd", "zoh_pd", "kalman_pd")
CORRECTED = ("cfc", "ncp_cfc", "random_sparse_cfc")
CONDITIONS = ("iid_dropout-0", "iid_dropout-0.3", "burst-5", "delay-4")
ANALYSIS_SEED = 20261108
SCOPE = "exploratory_paired_reacher"


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@contextmanager
def _lock_output(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".evaluation.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _episodes(config, count):
    if not 1 <= count <= 20:
        raise ValueError("The exploratory design supports 1–20 trajectories")
    settings = config.get("reproduction", {})
    episode_start = int(settings.get("episode_seed_start", 810000))
    schedule_start = int(settings.get("schedule_seed_start", 1910000))
    episodes = [
        {
            "trajectory_id": f"npj-mini-{index:04d}",
            "episode_seed": episode_start + index,
            "schedule_seed": schedule_start + index,
        }
        for index in range(count)
    ]
    for split in config["splits"].values():
        size = int(split["episodes"])
        for key in ("episode_seed", "schedule_seed"):
            start = int(split[f"{key}_start"])
            reserved = range(start, start + size)
            if any(episode[key] in reserved for episode in episodes):
                raise ValueError(f"Exploratory {key} overlaps a configured split")
        for episode in episodes:
            if any(
                bench._paired_schedule_seed(episode["schedule_seed"], condition) in reserved
                for condition in CONDITIONS
            ):
                raise ValueError("A paired perturbation schedule overlaps a configured split")
    return episodes


def _checkpoint(root, item):
    return root / item["readout"] / item["model_name"] / f"seed_{item['training_seed']}" / "best.pt"


def _runtime(device):
    metadata = {name: version(name) for name in ("gymnasium", "mujoco", "numpy", "torch")}
    metadata.update(python=platform.python_version(), system=platform.system(), machine=platform.machine())
    if torch.device(device).type == "cuda":
        gpu = torch.cuda.get_device_properties(device)
        metadata.update(cuda=torch.version.cuda, gpu_name=gpu.name, gpu_capability=[gpu.major, gpu.minor])
    return metadata


def _plan(config, config_hash, training_dir, registry, phase, episodes, replicates, device):
    if config.get("formal", {}).get("required", False):
        raise ValueError("Use the exploratory reproduction configuration")
    environment = config["environment"]
    if environment["kind"] != "reacher_v5":
        raise ValueError("The paired evaluation uses the moving-target Reacher-v5 environment")
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0 or replicates < 1:
        raise ValueError("Use distinct nonnegative training seeds and a positive bootstrap count")
    estimators = []
    for model in MODELS:
        readout = "scaled" if phase == "corrected" and model in CORRECTED else "unit"
        for seed in seeds:
            item = {"model_name": model, "training_seed": seed, "readout": readout}
            item["checkpoint_sha256"] = sha256_file(_checkpoint(training_dir, item))
            estimators.append(item)
    estimators.extend(
        {"model_name": name, "training_seed": -1, "readout": "reference", "checkpoint_sha256": ""}
        for name in REFERENCES
    )
    conditions = {item["condition_id"]: item for item in bench.build_response_conditions(config)}
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "phase": phase,
        "config_sha256": config_hash,
        "registry_sha256": sha256_file(registry),
        "package_source_sha256": source_tree_manifest(Path(__file__).resolve().parents[1])["aggregate_sha256"],
        "runtime": _runtime(device),
        "device": device,
        "episodes": _episodes(config, episodes),
        "conditions": [conditions[name] for name in CONDITIONS],
        "estimators": estimators,
        "training_seeds": seeds,
        "horizon_steps": int(environment["horizon_steps"]),
        "failure_penalty_m": float(config["evaluation"]["failure_tracking_penalty_m"]),
        "bootstrap_replicates": replicates,
    }


def _row_key(row):
    return row["model_name"], int(row["training_seed"]), row["condition_id"], row["trajectory_id"]


def _row_index(plan, rows):
    estimators = {
        (item["model_name"], item["training_seed"]): item for item in plan["estimators"]
    }
    conditions = {item["condition_id"]: item for item in plan["conditions"]}
    episodes = {item["trajectory_id"]: item for item in plan["episodes"]}
    cells = {}
    burst = {}
    for row in rows:
        key = _row_key(row)
        estimator, condition, episode = estimators.get(key[:2]), conditions.get(key[2]), episodes.get(key[3])
        if key in cells or not all((estimator, condition, episode)):
            raise ValueError(f"Unexpected or duplicate rollout: {key}")
        rmse, failure = float(row["tracking_rmse"]), float(row["failure"])
        if (
            row["checkpoint_sha256"] != estimator["checkpoint_sha256"]
            or row["evaluation_scope"] != SCOPE
            or int(row["episode_seed"]) != episode["episode_seed"]
            or int(row["schedule_seed"]) != bench._paired_schedule_seed(episode["schedule_seed"], key[2])
            or row["family"] != condition["family"]
            or float(row["severity"]) != condition["severity"]
            or int(row["tracking_scored_steps"]) != plan["horizon_steps"]
            or float(row["tracking_failure_penalty_m"]) != plan["failure_penalty_m"]
            or not math.isfinite(rmse)
            or rmse < 0
            or failure not in (0, 1)
        ):
            raise ValueError(f"Rollout differs from the paired endpoint or its frozen identity: {key}")
        cells[key] = row
        if key[2] == "burst-5":
            mask = tuple(int(row[name]) for name in ("burst_masked_steps", "burst_run_count", "burst_run_steps_total"))
            if mask != burst.setdefault(key[3], mask):
                raise ValueError(f"Paired burst schedules differ for {key[3]}")
    if len(cells) != len(estimators) * len(conditions) * len(episodes):
        raise ValueError("The paired rollout grid is incomplete")
    return cells, burst


def _original_rows(root, plan):
    original = _read_json(root / "mini_plan.json")
    complete = _read_json(root / "mini_complete.json")
    csv_path = root / "episode_metrics.csv"
    identity = sha256_file(csv_path)
    if (
        complete["plan_sha256"] != canonical_json_hash(original)
        or complete["episode_metrics_sha256"] != identity
        or complete["summary_sha256"] != sha256_file(root / "mini_summary.json")
    ):
        raise ValueError("The original table differs from its completion record")
    compatible_fields = (
        "scope", "config_sha256", "registry_sha256", "package_source_sha256",
        "episodes", "conditions", "training_seeds", "horizon_steps", "failure_penalty_m",
        "runtime", "device",
    )
    if original["phase"] != "original" or any(original[key] != plan[key] for key in compatible_fields):
        raise ValueError("The original table uses a different exploratory design, runtime or device")
    retained = [item for item in plan["estimators"] if item["model_name"] not in CORRECTED]
    if retained != [item for item in original["estimators"] if item["model_name"] not in CORRECTED]:
        raise ValueError("The unchanged observers differ from the original evaluation")
    cells, _ = _row_index(original, _read_rows(csv_path))
    if sha256_file(csv_path) != identity:
        raise ValueError("The original episode table changed during reading")
    return cells, {"plan_sha256": complete["plan_sha256"], "episode_metrics_sha256": identity}


def _load_estimator(item, config, training_dir, registry, device):
    if item["readout"] == "reference":
        settings = config["evaluation"].get("kalman") if item["model_name"] == "kalman_pd" else None
        return {**item, "baseline": bench.build_nonlearned_baseline(item["model_name"], settings)}
    from .training import load_observer

    path = _checkpoint(training_dir, item)
    observer, _ = load_observer(
        path, config=config, registry_path=registry, model=item["model_name"],
        seed=item["training_seed"], readout=item["readout"], device=device,
    )
    if sha256_file(path) != item["checkpoint_sha256"]:
        raise ValueError(f"Frozen observer changed: {path.name}")
    return {**item, "model": observer}


def _simulate_shard(root, estimator, condition, plan, config, plan_hash, row_source):
    rows, steps = [], []
    for episode in plan["episodes"]:
        schedule_seed = bench._paired_schedule_seed(episode["schedule_seed"], condition["condition_id"])
        result = bench._rollout(
            estimator, config, condition, episode_seed=episode["episode_seed"],
            schedule_seed=schedule_seed, device=plan["device"],
        )
        row_ids = {
            "trajectory_id": episode["trajectory_id"], "episode_seed": episode["episode_seed"],
            "schedule_seed": schedule_seed,
        }
        steps.extend({**step, **row_ids} for step in result.pop("_step_log"))
        rows.append({
            **result, **row_ids,
            **{key: estimator[key] for key in ("model_name", "training_seed", "checkpoint_sha256")},
            **{key: condition[key] for key in ("condition_id", "family", "severity")},
            "evaluation_scope": SCOPE, "row_provenance": row_source,
        })
    with tempfile.TemporaryDirectory(prefix=".rollouts-", dir=root.parent) as temporary:
        staging = Path(temporary)
        metrics = staging / "episode_metrics.csv"
        bench._write_csv(metrics, rows)
        log = bench._write_step_shard(staging / "step_log.npz", steps)
        atomic_write_json(staging / "manifest.json", {
            "plan_sha256": plan_hash, "episode_metrics_sha256": sha256_file(metrics),
            "step_log_sha256": log["sha256"],
        })
        staging.rename(root)


def _read_shard(root, plan_hash, expected):
    manifest = _read_json(root / "manifest.json")
    csv_path = root / "episode_metrics.csv"
    if (
        manifest["plan_sha256"] != plan_hash
        or manifest["episode_metrics_sha256"] != sha256_file(csv_path)
        or manifest["step_log_sha256"] != sha256_file(root / "step_log.npz")
    ):
        raise ValueError(f"Rollout shard digest differs: {root.name}")
    rows = _read_rows(csv_path)
    if [_row_key(row) for row in rows] != expected:
        raise ValueError(f"Rollout shard cell order differs: {root.name}")
    return rows


def _gather(root, plan, config, training_dir, registry, original):
    plan_hash = canonical_json_hash(plan)
    shard_dir = root / "shards"
    shard_dir.mkdir(exist_ok=True)
    rows = []
    for index, item in enumerate(plan["estimators"]):
        estimator = None
        for condition in plan["conditions"]:
            condition_name = condition["condition_id"]
            keys = [
                (item["model_name"], item["training_seed"], condition_name, episode["trajectory_id"])
                for episode in plan["episodes"]
            ]
            if original is not None and item["model_name"] not in CORRECTED:
                rows.extend(dict(original[key], row_provenance="reuse_original") for key in keys)
                continue
            shard = shard_dir / f"{index:02d}-{condition_name}"
            if not shard.exists():
                if estimator is None:
                    estimator = _load_estimator(item, config, training_dir, registry, plan["device"])
                source = "rerun_cfc_readout" if original is not None else "evaluated"
                _simulate_shard(shard, estimator, condition, plan, config, plan_hash, source)
            rows.extend(_read_shard(shard, plan_hash, keys))
    return rows


def _summarize(plan, rows, original):
    cells, burst = _row_index(plan, rows)
    episode_ids = [item["trajectory_id"] for item in plan["episodes"]]
    train_seeds = plan["training_seeds"]
    rng = np.random.default_rng(ANALYSIS_SEED)
    draws = plan["bootstrap_replicates"]
    seed_draws = rng.integers(len(train_seeds), size=(draws, len(train_seeds)))
    episode_draws = rng.integers(len(episode_ids), size=(draws, len(episode_ids)))

    def interval(values):
        seeds = seed_draws if len(values) > 1 else np.zeros((draws, 1), dtype=int)
        means = values[seeds[:, :, None], episode_draws[:, None, :]].mean(axis=(1, 2))
        return np.percentile(means, [2.5, 97.5]).tolist()

    def matrix(source, model, condition, seeds, field):
        return np.array([
            [float(source[model, seed, condition, episode][field]) for episode in episode_ids]
            for seed in seeds
        ])

    estimates, rmse, changes = [], {}, []
    for model in (*MODELS, *REFERENCES):
        seeds = train_seeds if model in MODELS else [-1]
        for condition in CONDITIONS:
            error = matrix(cells, model, condition, seeds, "tracking_rmse")
            failures = matrix(cells, model, condition, seeds, "failure")
            rmse[model, condition] = error
            estimates.append({
                "model": model, "condition": condition, "rollouts": int(error.size),
                "model_training_seeds": len(seeds), "unique_episodes": len(episode_ids),
                "mean_tracking_rmse_m": float(error.mean()), "ci95_m": interval(error),
                "failure_count": int(failures.sum()), "failure_rate": float(failures.mean()),
                "failure_count_by_seed": {str(seed): int(failures[i].sum()) for i, seed in enumerate(seeds)},
            })
            if original is not None and model in CORRECTED:
                previous = matrix(original, model, condition, seeds, "tracking_rmse")
                previous_failures = matrix(original, model, condition, seeds, "failure")
                delta = error - previous
                changes.append({
                    "model": model, "condition": condition, "paired_rollouts": int(delta.size),
                    "original_mean_tracking_rmse_m": float(previous.mean()),
                    "corrected_mean_tracking_rmse_m": float(error.mean()),
                    "corrected_minus_original_tracking_rmse_m": float(delta.mean()),
                    "ci95_m": interval(delta),
                    "original_failure_count": int(previous_failures.sum()),
                    "corrected_failure_count": int(failures.sum()),
                })
    contrasts = []
    for contrast, first, second, scope in bench.PRIMARY_CONTRASTS:
        for condition in CONDITIONS:
            difference = rmse[first, condition] - rmse[second, condition]
            contrasts.append({
                "contrast": contrast, "first_model": first, "second_model": second, "scope": scope,
                "condition": condition, "paired_training_seeds": len(train_seeds),
                "paired_trajectories": len(episode_ids),
                "mean_tracking_rmse_difference_m": float(difference.mean()),
                "ci95_m": interval(difference),
            })
    masked_steps, runs, run_steps = (sum(values) for values in zip(*burst.values()))
    return {
        "schema_version": 1, "scope": SCOPE, "phase": plan["phase"], "episode_rows": len(rows),
        "unique_trajectories": len(episode_ids), "conditions": list(CONDITIONS),
        "training_seeds": train_seeds, "horizon_steps": plan["horizon_steps"],
        "tracking_failure_penalty_m": plan["failure_penalty_m"], "analysis_seed": ANALYSIS_SEED,
        "bootstrap_replicates": draws, "reserved_holdout_opened": False, "test_episodes_used": 0,
        "independent_confirmation": False,
        "post_hoc_implementation_correction": plan["phase"] == "corrected",
        "tracking_endpoint": "Fixed-horizon end-effector RMSE per rollout with a post-failure step penalty.",
        "summary_estimand": "Mean penalized rollout RMSE with equal weight per seed and trajectory.",
        "contrast_direction": "First minus second; lower tracking RMSE favors the first model.",
        "interval_method": "Descriptive paired percentile bootstrap over training seeds and trajectory IDs.",
        "multiplicity_scope": "Unadjusted descriptive 95% intervals for the exploratory cohort.",
        "reused_rollouts": sum(row["row_provenance"] == "reuse_original" for row in rows),
        "rerun_rollouts": sum(row["row_provenance"] == "rerun_cfc_readout" for row in rows),
        "burst_realization": {
            "missing_fraction": masked_steps / (len(episode_ids) * plan["horizon_steps"]),
            "observed_mean_run_steps": run_steps / runs if runs else None,
            "unique_trajectories": len(episode_ids),
        },
        "model_condition_estimates": estimates, "mechanism_contrasts": contrasts,
        "original_vs_corrected_paired": changes,
    }


def evaluate(args):
    config, config_hash = load_config(args.config)
    count = args.episodes if args.episodes is not None else int(config.get("reproduction", {}).get("episodes", 20))
    replicates = args.bootstrap_replicates if args.bootstrap_replicates is not None else int(
        config["evaluation"].get("bootstrap_replicates", 10000)
    )
    plan = _plan(config, config_hash, args.training_dir, args.registry, args.phase, count, replicates, args.device)
    root = args.output_dir
    original = None
    if args.reuse_original is not None:
        if args.phase != "corrected" or args.reuse_original.resolve() == root.resolve():
            raise ValueError("Use a separate original run with the corrected phase")
        original, plan["original_evaluation"] = _original_rows(args.reuse_original, plan)
    plan_hash = canonical_json_hash(plan)
    with _lock_output(root):
        plan_path = root / "mini_plan.json"
        if plan_path.exists() and _read_json(plan_path) != plan:
            raise ValueError("The output directory already belongs to a different evaluation plan")
        atomic_write_json(plan_path, plan)
        csv_path = root / "episode_metrics.csv"
        summary_path = root / "mini_summary.json"
        completion_path = root / "mini_complete.json"
        if completion_path.exists():
            complete = _read_json(completion_path)
            if complete != {
                "plan_sha256": plan_hash, "episode_metrics_sha256": sha256_file(csv_path),
                "summary_sha256": sha256_file(summary_path),
            }:
                raise ValueError("Completed evaluation outputs differ from their published digests")
            return _read_json(summary_path)
        rows = _gather(root, plan, config, args.training_dir, args.registry, original)
        summary = _summarize(plan, rows, original)
        bench._write_csv(csv_path, rows)
        summary.update(
            plan_sha256=plan_hash, episode_metrics_sha256=sha256_file(csv_path),
            original_evaluation=plan.get("original_evaluation"),
        )
        atomic_write_json(summary_path, summary)
        atomic_write_json(completion_path, {
            "plan_sha256": plan_hash, "episode_metrics_sha256": summary["episode_metrics_sha256"],
            "summary_sha256": sha256_file(summary_path),
        })
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("original", "corrected"), default="corrected")
    parser.add_argument("--reuse-original", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args()
    summary = evaluate(args)
    print(json.dumps({key: summary[key] for key in ("episode_rows", "reused_rollouts", "rerun_rollouts")}))


if __name__ == "__main__":
    main()
