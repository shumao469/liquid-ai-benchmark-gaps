"""Command line entrypoints for preparation, capture, generation, and smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .prepare_data import prepare_revision_data, verify_prepared_data
from .repro import capture_environment


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="liquid-revision")
    subparsers = parser.add_subparsers(dest="command", required=True)
    from .offline.cli import register_subcommands
    from .closed_loop.cli import register_subcommands as register_closed_loop_subcommands

    prepare = subparsers.add_parser(
        "prepare-data", help="securely and atomically prepare HAR and PhysioNet"
    )
    prepare.add_argument("--har-zip", required=True)
    prepare.add_argument("--physionet-zip", required=True)
    prepare.add_argument("--physionet-outcomes", required=True)
    prepare.add_argument("--output", required=True)

    verify = subparsers.add_parser(
        "verify-data", help="rehash every prepared dataset file"
    )
    verify.add_argument("--root", required=True)

    capture = subparsers.add_parser(
        "capture-env", help="capture versions, GPU state, Git state, and config hashes"
    )
    capture.add_argument("--output", required=True)
    capture.add_argument("--config", action="append", default=[])
    capture.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))

    generate = subparsers.add_parser(
        "generate-demo", help="generate deterministic frozen-PD closed-loop demonstrations"
    )
    generate.add_argument("--config", required=True)
    generate.add_argument("--output-dir", required=True)

    smoke = subparsers.add_parser(
        "smoke", help="run environment/data/model smoke checks without training"
    )
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--output-dir", required=True)
    register_subcommands(subparsers)
    register_closed_loop_subcommands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare-data":
        manifest = prepare_revision_data(
            har_zip=args.har_zip,
            physionet_zip=args.physionet_zip,
            physionet_outcomes=args.physionet_outcomes,
            output_root=args.output,
        )
        _print(
            {
                "status": "prepared",
                "output": str(Path(args.output).resolve()),
                "har_files": manifest["har"]["member_count"],
                "physionet_records": manifest["physionet2012"]["validation"][
                    "record_count"
                ],
            }
        )
        return 0
    if args.command == "verify-data":
        _print({"status": "verified", **verify_prepared_data(args.root)})
        return 0
    if args.command == "capture-env":
        payload = capture_environment(
            args.output, config_paths=args.config, repo_path=args.repo
        )
        _print(
            {
                "status": "captured",
                "output": str(Path(args.output).resolve()),
                "torch_runtime": payload.get("torch_runtime"),
            }
        )
        return 0
    if args.command == "generate-demo":
        from .data.closed_loop import generate_closed_loop_dataset

        config, config_hash = load_config(args.config)
        manifest = generate_closed_loop_dataset(config, args.output_dir)
        _print(
            {
                "status": "generated",
                "training_performed": False,
                "config_sha256": config_hash,
                "output": str(Path(args.output_dir).resolve()),
                "episodes": manifest["episodes"],
                "transitions": manifest["transitions"],
                "dataset_sha256": manifest["dataset_sha256"],
            }
        )
        return 0
    if args.command == "smoke":
        from .smoke import run_smoke

        _print(run_smoke(args.config, args.output_dir))
        return 0
    from .offline.cli import run_command as run_offline_command

    offline_result = run_offline_command(args)
    if offline_result is not None:
        return offline_result
    from .closed_loop.cli import run_command as run_closed_loop_command

    closed_loop_result = run_closed_loop_command(args)
    if closed_loop_result is not None:
        return closed_loop_result
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

