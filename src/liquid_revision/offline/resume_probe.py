"""One-shot, non-confirmatory GPU probe for interrupted-run resumption.

This module is intentionally separate from the benchmark CLI.  It injects one
controlled exception immediately after the first validation checkpoint, then
restores the original batch scheduler and verifies continuation from that
checkpoint.  It must never be used to produce scientific outcome estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from liquid_revision.config import load_config
from liquid_revision.io_utils import atomic_write_json, canonical_json_hash, sha256_file

from . import training as training_module


def run_probe(config_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    base_config, _ = load_config(config_path)
    config = json.loads(json.dumps(base_config))
    config["experiment"]["datasets"] = ["har"]
    config["experiment"]["seeds"] = [101]
    config["experiment"]["confirmatory_evidence"] = False
    config["training"]["optimizer_steps"] = 3
    config["training"]["validation_interval"] = 1
    config["corruption"]["mcar_levels"] = [0.0]
    config["corruption"]["temporal_block_levels"] = [0.0]
    config["latency"]["warmup"] = 1
    config["latency"]["repeats"] = 2
    config_hash = canonical_json_hash(config)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    datasets = training_module.load_all_datasets(
        config, config_path, selected_datasets=["har"], include_test=False
    )
    original = training_module.deterministic_batch_indices

    def interrupt_after_first_checkpoint(
        sample_count: int, batch_size: int, seed: int, optimizer_step: int
    ):
        if optimizer_step == 1:
            raise RuntimeError("controlled resume probe interruption")
        return original(sample_count, batch_size, seed, optimizer_step)

    interrupted = False
    training_module.deterministic_batch_indices = interrupt_after_first_checkpoint
    try:
        training_module.train_fixed_budget(
            dataset_name="har",
            model_name="lstm",
            seed=101,
            data=datasets["har"],
            config=config,
            config_hash=config_hash,
            output_root=output / "runs",
        )
    except RuntimeError as exc:
        if str(exc) != "controlled resume probe interruption":
            raise
        interrupted = True
    finally:
        training_module.deterministic_batch_indices = original
    if not interrupted:
        raise AssertionError("controlled interruption did not occur")

    run_id = training_module._run_identifier("har", "lstm", 101, config_hash)
    run_dir = output / "runs" / run_id
    interrupted_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint_after_interrupt = torch.load(
        run_dir / "latest.pt", map_location="cpu", weights_only=False
    )
    checkpoint_hash_after_interrupt = sha256_file(run_dir / "latest.pt")
    if int(checkpoint_after_interrupt["optimizer_step"]) != 1:
        raise AssertionError("probe did not persist the first optimizer step")

    result = training_module.train_fixed_budget(
        dataset_name="har",
        model_name="lstm",
        seed=101,
        data=datasets["har"],
        config=config,
        config_hash=config_hash,
        output_root=output / "runs",
    )
    completed_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    history_steps = [
        int(item["optimizer_step"]) for item in result["validation_history"]
    ]
    if history_steps != [1, 2, 3]:
        raise AssertionError(f"validation history was not restored exactly: {history_steps}")
    report: dict[str, object] = {
        "status": "passed",
        "confirmatory_evidence": False,
        "device": str(config["experiment"]["device"]),
        "config_sha256": config_hash,
        "run_id": run_id,
        "controlled_interruption": {
            "status": interrupted_manifest["status"],
            "last_completed_step": interrupted_manifest["last_completed_step"],
            "validation_records_persisted": interrupted_manifest[
                "validation_records_persisted"
            ],
            "latest_checkpoint_sha256": checkpoint_hash_after_interrupt,
        },
        "resumed_run": {
            "status": result["status"],
            "attempt_number": completed_manifest["attempt_number"],
            "resumed_from_step": completed_manifest["resumed_from_step"],
            "restored_validation_records": completed_manifest[
                "restored_validation_records"
            ],
            "validation_history_steps": history_steps,
            "best_checkpoint_sha256": result["best_checkpoint_sha256"],
            "test_metrics_computed": result["test_metrics_computed"],
            "completion_artifact_present": (run_dir / "completion.json").is_file(),
        },
    }
    atomic_write_json(output / "resume_probe.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.config, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
