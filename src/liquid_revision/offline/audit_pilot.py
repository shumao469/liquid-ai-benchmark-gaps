"""Validate offline pilot artifacts and derive a transparent runtime forecast."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from liquid_revision.io_utils import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(
    *,
    pilot_root: Path,
    dt_smoke_path: Path,
    resume_probe_path: Path,
    parameter_table_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    pilot_manifest = _read(pilot_root / "offline_benchmark_manifest.json")
    dt_smoke = _read(dt_smoke_path)
    resume_probe = _read(resume_probe_path)
    parameter_table = _read(parameter_table_path)
    result_files = sorted((pilot_root / "runs").glob("*/result.json"))
    if pilot_manifest.get("status") != "completed" or len(result_files) != 8:
        raise AssertionError("expected eight completed dataset/model pilot runs")
    if dt_smoke.get("status") != "passed" or resume_probe.get("status") != "passed":
        raise AssertionError("delta-time smoke or resume probe did not pass")

    formal_steps = 20_000
    formal_validation_calls = formal_steps // 500
    formal_condition_count = 17
    formal_latency_calls = 1_000 + 10_000
    estimates: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for result_file in result_files:
        result = _read(result_file)
        dataset = str(result["dataset"])
        model = str(result["model"])
        expected = parameter_table["tasks"][dataset]["models"][model]
        evaluation = result["fixed_checkpoint_evaluation"]
        conditions = evaluation["conditions"]
        expected_conditions = [
            "mcar-0.0",
            "mcar-0.3",
            "temporal_block-0.0",
            "temporal_block-0.4",
        ]
        if [item["condition_id"] for item in conditions] != expected_conditions:
            raise AssertionError(f"pilot corruption grid mismatch for {dataset}/{model}")
        if int(result["trainable_parameters"]) != int(expected["trainable_parameters"]):
            raise AssertionError(f"parameter table mismatch for {dataset}/{model}")
        if evaluation["model_state_sha256_before"] != evaluation["model_state_sha256_after"]:
            raise AssertionError(f"evaluation mutated {dataset}/{model}")
        if evaluation["retrained_per_severity"] or int(
            evaluation["optimizer_steps_during_robustness_evaluation"]
        ) != 0:
            raise AssertionError(f"severity-specific optimization occurred for {dataset}/{model}")
        smoke_model = dt_smoke["datasets"][dataset]["models"][model]
        if model != "lstm" and not smoke_model["changed_with_elapsed_time"]:
            raise AssertionError(f"elapsed-time path inactive for {dataset}/{model}")

        validation_calls = len(result["validation_history"])
        if validation_calls <= 0:
            raise AssertionError("pilot must contain validation timing")
        step_seconds = float(result["optimizer_step_timing"]["median_seconds"])
        validation_seconds = float(
            result["validation_wall_time_seconds_this_attempt"] / validation_calls
        )
        condition_seconds = float(
            statistics.mean(
                float(item["evaluation_wall_time_seconds"]) for item in conditions
            )
        )
        total_sequence_ms = float(evaluation["latency"]["median_ms"])
        sequence_length = 128 if dataset == "har" else 48
        training_estimate = step_seconds * formal_steps
        validation_estimate = validation_seconds * formal_validation_calls
        robustness_estimate = condition_seconds * formal_condition_count
        latency_estimate = total_sequence_ms / 1000.0 * formal_latency_calls
        total_seconds = (
            training_estimate
            + validation_estimate
            + robustness_estimate
            + latency_estimate
        )
        gates.append(
            {
                "dataset": dataset,
                "model": model,
                "hidden_dim": int(result["hidden_dim"]),
                "trainable_parameters": int(result["trainable_parameters"]),
                "parameter_relative_error_percent": float(
                    expected["relative_error_percent"]
                ),
                "fixed_checkpoint_unchanged": True,
                "severity_specific_optimizer_steps": 0,
                "elapsed_time_active": smoke_model["changed_with_elapsed_time"],
                "full_sequence_latency": {
                    "measurement_scope": "full_padded_sequence_forward",
                    "sequence_length_updates": sequence_length,
                    "median_total_sequence_ms": total_sequence_ms,
                    "p95_total_sequence_ms": float(evaluation["latency"]["p95_ms"]),
                    "amortized_median_ms_per_update": total_sequence_ms
                    / sequence_length,
                    "per_update_value_is_amortized_not_independently_timed": True,
                    "batch_size": 1,
                    "host_to_device_transfer_included": False,
                    "gpu_synchronized": True,
                },
            }
        )
        estimates.append(
            {
                "dataset": dataset,
                "model": model,
                "pilot_median_optimizer_step_seconds": step_seconds,
                "pilot_mean_validation_call_seconds": validation_seconds,
                "pilot_mean_condition_evaluation_seconds": condition_seconds,
                "pilot_median_full_sequence_latency_ms": total_sequence_ms,
                "estimated_formal_seconds": total_seconds,
                "estimated_formal_hours_per_model_seed": total_seconds / 3600.0,
                "components_seconds": {
                    "optimizer_steps": training_estimate,
                    "validation_calls": validation_estimate,
                    "fixed_checkpoint_conditions": robustness_estimate,
                    "latency_protocol": latency_estimate,
                },
            }
        )

    one_seed_seconds = sum(item["estimated_formal_seconds"] for item in estimates)
    forty_run_seconds = one_seed_seconds * 5
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "confirmatory_evidence": False,
        "gpu_scope": {
            "cuda_visible_devices": "physical GPU 0 only",
            "pilot_device": "cuda:0",
            "physical_gpu": "NVIDIA GeForce RTX 5090",
            "gpu_1_used_by_this_pilot": False,
        },
        "gates": gates,
        "resume_probe": resume_probe,
        "runtime_forecast": {
            "method": (
                "median optimizer-step time from the four-step full-data GPU pilot, "
                "plus mean validation-condition timings and synchronized full-sequence "
                "batch-1 latency; total pilot wall time was not used for extrapolation"
            ),
            "formal_protocol": {
                "optimizer_steps_per_run": formal_steps,
                "validation_calls_per_run": formal_validation_calls,
                "fixed_checkpoint_conditions_per_run": formal_condition_count,
                "latency_warmup_plus_timed_calls_per_run": formal_latency_calls,
                "model_seed_runs": 40,
            },
            "per_model_seed": estimates,
            "estimated_hours_one_seed_all_eight_dataset_model_pairs": one_seed_seconds
            / 3600.0,
            "estimated_hours_40_runs_one_rtx5090": forty_run_seconds / 3600.0,
            "estimated_days_40_runs_one_rtx5090": forty_run_seconds / 86400.0,
            "conservative_hours_40_runs_one_rtx5090_with_15_percent_overhead": (
                forty_run_seconds * 1.15 / 3600.0
            ),
            "ideal_two_gpu_lower_bound_hours": forty_run_seconds / 7200.0,
            "two_gpu_lower_bound_not_currently_actionable": (
                "physical GPU 1 was occupied and thermally loaded during this pilot"
            ),
        },
        "limitations": [
            "Four optimization steps are sufficient for engineering timing only, not performance inference.",
            "Runtime scales can shift with GPU contention, thermal throttling, and later software changes.",
            "The latency call covers the entire padded sequence, not one online recurrent update.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--dt-smoke", required=True)
    parser.add_argument("--resume-probe", required=True)
    parser.add_argument("--parameter-table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(
        pilot_root=Path(args.pilot_root).resolve(),
        dt_smoke_path=Path(args.dt_smoke).resolve(),
        resume_probe_path=Path(args.resume_probe).resolve(),
        parameter_table_path=Path(args.parameter_table).resolve(),
        output_path=Path(args.output).resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
