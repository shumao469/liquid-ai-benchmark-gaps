"""Narrow CLI extension points; the root CLI owns only one import/callback."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .smoke import run_offline_smoke
from .training import run_offline_benchmark, run_offline_test_release


def register_subcommands(subparsers: Any) -> None:
    smoke = subparsers.add_parser(
        "offline-smoke", help="validate offline data, splits, corruption, and models without training"
    )
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--output-dir", required=True)

    run = subparsers.add_parser(
        "offline-run", help="train and freeze all validation-selected checkpoints; no test access"
    )
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)

    test_release = subparsers.add_parser(
        "offline-test-release",
        help="after the complete checkpoint release verifies, run the one unified test sweep",
    )
    test_release.add_argument("--config", required=True)
    test_release.add_argument("--training-output-dir", required=True)
    test_release.add_argument("--output-dir", required=True)


def run_command(args: argparse.Namespace) -> int | None:
    if args.command == "offline-smoke":
        payload = run_offline_smoke(args.config, args.output_dir)
    elif args.command == "offline-run":
        payload = run_offline_benchmark(args.config, args.output_dir)
    elif args.command == "offline-test-release":
        payload = run_offline_test_release(
            args.config, args.training_output_dir, args.output_dir
        )
    else:
        return None
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0

