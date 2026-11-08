"""CLI registration for the auditable closed-loop workflow."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import load_config
from .benchmark import evaluate_response_surface
from .formal import (
    create_checkpoint_release,
    create_controller_selection,
    create_environment_freeze,
    create_pretraining_master_release,
    create_test_release,
    create_tuning_namespace,
    create_tuning_selection,
)
from .manifests import prepare_split_datasets, verify_split_datasets
from .parameter_matching import create_parameter_registry
from .profiling import profile_optimizer_steps, profile_rollout_throughput
from .training import train_registered_model, train_tuning_candidate


def register_subcommands(subparsers) -> None:
    prepare = subparsers.add_parser(
        "closed-loop-prepare", help="freeze trajectory splits and generate Reacher data"
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output-dir", required=True)

    freeze_environment = subparsers.add_parser(
        "closed-loop-freeze-environment",
        help="freeze software/CUDA/MuJoCo XML/protocol/source provenance",
    )
    freeze_environment.add_argument("--config", required=True)
    freeze_environment.add_argument("--output", required=True)

    select_controller = subparsers.add_parser(
        "closed-loop-select-controller",
        help="run the validation-only 16-cell full-state PD grid",
    )
    select_controller.add_argument("--config", required=True)
    select_controller.add_argument("--output", required=True)
    select_controller.add_argument("--device", default="cpu")

    freeze_tuning = subparsers.add_parser(
        "closed-loop-freeze-tuning",
        help="validate all seed-17 optimizer candidates and freeze per-model choices",
    )
    freeze_tuning.add_argument("--config", required=True)
    freeze_tuning.add_argument("--registry", required=True)
    freeze_tuning.add_argument("--data-root", required=True)
    freeze_tuning.add_argument(
        "--candidate-result", action="append", required=True
    )
    freeze_tuning.add_argument("--output", required=True)
    freeze_tuning.add_argument("--tuning-namespace", required=True)
    freeze_tuning.add_argument(
        "--expected-tuning-namespace-sha256", required=True
    )
    freeze_tuning.add_argument(
        "--expected-tuning-namespace-file-sha256", required=True
    )
    freeze_tuning.add_argument(
        "--tuning-lease-custody-receipt", required=True
    )
    freeze_tuning.add_argument(
        "--expected-tuning-lease-custody-receipt-sha256", required=True
    )
    freeze_tuning.add_argument(
        "--expected-tuning-lease-custody-receipt-file-sha256", required=True
    )

    freeze_tuning_namespace = subparsers.add_parser(
        "closed-loop-freeze-tuning-namespace",
        help="freeze the exact 28 tuning paths/attempts before external anchoring",
    )
    freeze_tuning_namespace.add_argument("--config", required=True)
    freeze_tuning_namespace.add_argument("--registry", required=True)
    freeze_tuning_namespace.add_argument("--data-root", required=True)
    freeze_tuning_namespace.add_argument("--output", required=True)

    freeze_checkpoints = subparsers.add_parser(
        "closed-loop-freeze-checkpoints",
        help="freeze and rehash the exact completed 7x5 checkpoint set",
    )
    freeze_checkpoints.add_argument("--config", required=True)
    freeze_checkpoints.add_argument("--data-root", required=True)
    freeze_checkpoints.add_argument("--registry", required=True)
    freeze_checkpoints.add_argument(
        "--run-registry",
        required=True,
        help="frozen 7x5 attempt ledger that uniquely authorizes each checkpoint",
    )
    freeze_checkpoints.add_argument("--checkpoint", action="append", required=True)
    freeze_checkpoints.add_argument("--output", required=True)
    freeze_checkpoints.add_argument(
        "--expected-pretraining-master-sha256",
        required=True,
        help="externally published/audited pretraining master root",
    )
    freeze_checkpoints.add_argument(
        "--expected-pretraining-audit-receipt-sha256",
        required=True,
        help="externally published canonical pretraining-audit receipt hash",
    )
    freeze_checkpoints.add_argument(
        "--expected-pretraining-audit-receipt-file-sha256",
        required=True,
        help="externally published byte-level SHA-256 of the audit-receipt file",
    )
    freeze_checkpoints.add_argument(
        "--pretraining-audit-receipt",
        required=True,
        help="independent receipt whose hash is externally anchored",
    )

    freeze_pretraining = subparsers.add_parser(
        "closed-loop-freeze-pretraining",
        help="freeze the full pre-confirmatory chain for independent audit",
    )
    freeze_pretraining.add_argument("--config", required=True)
    freeze_pretraining.add_argument("--data-root", required=True)
    freeze_pretraining.add_argument("--registry", required=True)
    freeze_pretraining.add_argument("--preflight-receipt", required=True)
    freeze_pretraining.add_argument(
        "--run-registry-namespace",
        required=True,
        help="write-once namespace committed into the pretraining root",
    )
    freeze_pretraining.add_argument(
        "--run-registry",
        required=True,
        help="exact mutable ledger path committed by the namespace",
    )
    freeze_pretraining.add_argument("--output", required=True)

    authorize_test = subparsers.add_parser(
        "closed-loop-authorize-test",
        help="create the separate test unlock after checkpoint release",
    )
    authorize_test.add_argument("--config", required=True)
    authorize_test.add_argument("--output", required=True)
    authorize_test.add_argument("--evaluation-output-dir", required=True)
    authorize_test.add_argument("--test-claim-custody-receipt", required=True)
    authorize_test.add_argument(
        "--expected-test-claim-custody-receipt-sha256", required=True
    )
    authorize_test.add_argument(
        "--expected-master-release-sha256",
        required=True,
        help="externally published/audited checkpoint-release root hash",
    )

    tune_candidate = subparsers.add_parser(
        "closed-loop-tune-candidate",
        help="run/resume one registered validation-only seed-17 optimizer candidate",
    )
    tune_candidate.add_argument("--config", required=True)
    tune_candidate.add_argument("--data-root", required=True)
    tune_candidate.add_argument("--registry", required=True)
    tune_candidate.add_argument("--model", required=True)
    tune_candidate.add_argument("--learning-rate", required=True, type=float)
    tune_candidate.add_argument("--weight-decay", required=True, type=float)
    tune_candidate.add_argument("--output-dir", required=True)
    tune_candidate.add_argument("--device", default="cpu")
    tune_candidate.add_argument("--resume", action="store_true")
    tune_candidate.add_argument("--tuning-namespace", required=True)
    tune_candidate.add_argument(
        "--expected-tuning-namespace-sha256", required=True
    )
    tune_candidate.add_argument(
        "--expected-tuning-namespace-file-sha256", required=True
    )
    tune_candidate.add_argument("--tuning-attempt-id", required=True)
    tune_candidate.add_argument(
        "--tuning-lease-authorization", required=True
    )
    tune_candidate.add_argument(
        "--expected-tuning-lease-authorization-sha256", required=True
    )
    tune_candidate.add_argument(
        "--expected-tuning-lease-authorization-file-sha256", required=True
    )
    tune_candidate.add_argument(
        "--tuning-lease-custody-receipt", required=True
    )
    tune_candidate.add_argument(
        "--expected-tuning-lease-custody-receipt-sha256", required=True
    )
    tune_candidate.add_argument(
        "--expected-tuning-lease-custody-receipt-file-sha256", required=True
    )
    tune_candidate.add_argument(
        "--tuning-authorized-worker-pid", required=True, type=int
    )
    tune_candidate.add_argument(
        "--tuning-worker-capability-nonce", required=True
    )
    tune_candidate.add_argument("--tuning-output-reservation", required=True)
    tune_candidate.add_argument(
        "--expected-tuning-output-reservation-sha256", required=True
    )
    tune_candidate.add_argument(
        "--expected-tuning-output-reservation-file-sha256", required=True
    )

    verify = subparsers.add_parser(
        "closed-loop-verify", help="verify split manifest and every closed-loop dataset"
    )
    verify.add_argument("--data-root", required=True)

    match = subparsers.add_parser(
        "closed-loop-match", help="freeze the plus-or-minus five-percent parameter registry"
    )
    match.add_argument("--config", required=True)
    match.add_argument("--output", required=True)

    train = subparsers.add_parser(
        "closed-loop-train", help="train/resume one registered model seed"
    )
    train.add_argument("--config", required=True)
    train.add_argument("--data-root", required=True)
    train.add_argument("--registry", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--expected-pretraining-master-sha256",
        help="required for formal confirmatory training; externally published root",
    )
    train.add_argument(
        "--pretraining-audit-receipt",
        help="independent read-only audit receipt for the pretraining root",
    )
    train.add_argument(
        "--expected-pretraining-audit-receipt-sha256",
        help="external canonical anchor for the independent pretraining-audit receipt",
    )
    train.add_argument(
        "--expected-pretraining-audit-receipt-file-sha256",
        help="external byte-level SHA-256 anchor for the audit-receipt file",
    )
    train.add_argument(
        "--run-registry",
        help="formal write-once model/seed attempt registry",
    )
    train.add_argument(
        "--run-attempt-id",
        help="registry-issued unique attempt identity",
    )
    train.add_argument(
        "--run-reservation-token",
        help="registry-issued write-once reservation token",
    )
    train.add_argument(
        "--expected-run-reservation-token-sha256",
        help="external file SHA-256 supplied by the launcher for the reservation token",
    )

    evaluate = subparsers.add_parser(
        "closed-loop-evaluate", help="evaluate fixed checkpoints over paired perturbation surfaces"
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--checkpoint", action="append", default=[])
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument(
        "--expected-master-release-sha256",
        help="required for formal evaluation; externally published release root",
    )
    evaluate.add_argument(
        "--expected-test-release-sha256",
        help="required for formal evaluation; external test-release anchor",
    )
    evaluate.add_argument(
        "--test-claim-custody-receipt",
        help="externally anchored auditor-managed one-shot claim receipt",
    )
    evaluate.add_argument(
        "--expected-test-claim-custody-receipt-sha256",
        help="external file SHA-256 for the claim-custody receipt",
    )

    pilot = subparsers.add_parser(
        "closed-loop-pilot", help="prepare, match, train and evaluate a tiny configured pilot"
    )
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--device", default="cpu")

    profile_training = subparsers.add_parser(
        "closed-loop-profile-training",
        help="measure matched optimizer-step throughput before long training",
    )
    profile_training.add_argument("--config", required=True)
    profile_training.add_argument("--registry", required=True)
    profile_training.add_argument("--output", required=True)
    profile_training.add_argument("--device", default="cpu")
    profile_training.add_argument("--steps", type=int, default=100)
    profile_training.add_argument("--warmup-steps", type=int, default=5)

    profile_rollout = subparsers.add_parser(
        "closed-loop-profile-rollout",
        help="measure end-to-end Reacher plant-step throughput",
    )
    profile_rollout.add_argument("--config", required=True)
    profile_rollout.add_argument("--checkpoint", action="append", default=[])
    profile_rollout.add_argument("--output", required=True)
    profile_rollout.add_argument("--device", default="cpu")
    profile_rollout.add_argument("--episodes", type=int, default=2)


def _emit(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def run_command(args):
    if args.command == "closed-loop-verify":
        _emit(verify_split_datasets(args.data_root))
        return 0
    if not args.command.startswith("closed-loop-"):
        return None
    config, config_hash = load_config(args.config)
    if args.command == "closed-loop-freeze-environment":
        artifact = create_environment_freeze(config, args.output)
        _emit(
            {
                "status": "complete",
                "output": str(Path(args.output).resolve()),
                "environment_sha256": artifact["environment_sha256"],
            }
        )
        return 0
    if args.command == "closed-loop-select-controller":
        artifact = create_controller_selection(
            config, args.output, device=args.device
        )
        _emit(
            {
                "status": "complete",
                "output": str(Path(args.output).resolve()),
                "controller_selection_sha256": artifact[
                    "controller_selection_sha256"
                ],
                "selected": artifact["selected"],
            }
        )
        return 0
    if args.command == "closed-loop-freeze-tuning":
        artifact = create_tuning_selection(
            config,
            args.registry,
            args.data_root,
            args.candidate_result,
            args.output,
            tuning_namespace_path=args.tuning_namespace,
            expected_tuning_namespace_sha256=(
                args.expected_tuning_namespace_sha256
            ),
            expected_tuning_namespace_file_sha256=(
                args.expected_tuning_namespace_file_sha256
            ),
            tuning_lease_custody_receipt_path=(
                args.tuning_lease_custody_receipt
            ),
            expected_tuning_lease_custody_receipt_sha256=(
                args.expected_tuning_lease_custody_receipt_sha256
            ),
            expected_tuning_lease_custody_receipt_file_sha256=(
                args.expected_tuning_lease_custody_receipt_file_sha256
            ),
        )
        _emit(
            {
                "status": "complete",
                "output": str(Path(args.output).resolve()),
                "tuning_selection_sha256": artifact["tuning_selection_sha256"],
                "selected": artifact["selected"],
            }
        )
        return 0
    if args.command == "closed-loop-freeze-tuning-namespace":
        artifact = create_tuning_namespace(
            config,
            args.registry,
            args.data_root,
            args.output,
        )
        _emit(
            {
                "status": "complete",
                "output": str(Path(args.output).resolve()),
                "tuning_namespace_sha256": artifact[
                    "tuning_namespace_sha256"
                ],
                "candidate_count": artifact["candidate_count"],
            }
        )
        return 0
    if args.command == "closed-loop-freeze-pretraining":
        artifact = create_pretraining_master_release(
            config,
            args.data_root,
            args.registry,
            args.preflight_receipt,
            args.run_registry_namespace,
            args.run_registry,
            args.output,
        )
        _emit(
            {
                "status": "complete",
                "output": str(Path(args.output).resolve()),
                "pretraining_master_sha256": artifact[
                    "pretraining_master_sha256"
                ],
                "test_episodes_used": artifact["test_episodes_used"],
            }
        )
        return 0
    if args.command == "closed-loop-freeze-checkpoints":
        artifact = create_checkpoint_release(
            config,
            args.data_root,
            args.registry,
            args.checkpoint,
            args.output,
            expected_pretraining_master_sha256=(
                args.expected_pretraining_master_sha256
            ),
            expected_pretraining_audit_receipt_sha256=(
                args.expected_pretraining_audit_receipt_sha256
            ),
            expected_pretraining_audit_receipt_file_sha256=(
                args.expected_pretraining_audit_receipt_file_sha256
            ),
            pretraining_audit_receipt_path=args.pretraining_audit_receipt,
            run_registry_path=args.run_registry,
        )
        _emit(
            {
                "status": "complete",
                "output": str(Path(args.output).resolve()),
                "checkpoint_release_sha256": artifact[
                    "checkpoint_release_sha256"
                ],
                "checkpoint_count": artifact["expected_checkpoint_count"],
            }
        )
        return 0
    if args.command == "closed-loop-authorize-test":
        artifact = create_test_release(
            config,
            args.output,
            expected_master_release_sha256=args.expected_master_release_sha256,
            authorized_evaluation_output_dir=args.evaluation_output_dir,
            test_claim_custody_receipt_path=args.test_claim_custody_receipt,
            expected_test_claim_custody_receipt_sha256=(
                args.expected_test_claim_custody_receipt_sha256
            ),
        )
        _emit(
            {
                "status": "complete",
                "output": str(Path(args.output).resolve()),
                "test_release_sha256": artifact["test_release_sha256"],
            }
        )
        return 0
    if args.command == "closed-loop-tune-candidate":
        _emit(
            train_tuning_candidate(
                registered_name=args.model,
                config=config,
                data_root=args.data_root,
                registry_path=args.registry,
                output_dir=args.output_dir,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                device=args.device,
                resume=args.resume,
                tuning_namespace_path=args.tuning_namespace,
                expected_tuning_namespace_sha256=(
                    args.expected_tuning_namespace_sha256
                ),
                expected_tuning_namespace_file_sha256=(
                    args.expected_tuning_namespace_file_sha256
                ),
                tuning_attempt_id=args.tuning_attempt_id,
                tuning_lease_authorization_path=(
                    args.tuning_lease_authorization
                ),
                expected_tuning_lease_authorization_sha256=(
                    args.expected_tuning_lease_authorization_sha256
                ),
                expected_tuning_lease_authorization_file_sha256=(
                    args.expected_tuning_lease_authorization_file_sha256
                ),
                tuning_lease_custody_receipt_path=(
                    args.tuning_lease_custody_receipt
                ),
                expected_tuning_lease_custody_receipt_sha256=(
                    args.expected_tuning_lease_custody_receipt_sha256
                ),
                expected_tuning_lease_custody_receipt_file_sha256=(
                    args.expected_tuning_lease_custody_receipt_file_sha256
                ),
                tuning_authorized_worker_pid=args.tuning_authorized_worker_pid,
                tuning_worker_capability_nonce=(
                    args.tuning_worker_capability_nonce
                ),
                tuning_output_reservation_path=(
                    args.tuning_output_reservation
                ),
                expected_tuning_output_reservation_sha256=(
                    args.expected_tuning_output_reservation_sha256
                ),
                expected_tuning_output_reservation_file_sha256=(
                    args.expected_tuning_output_reservation_file_sha256
                ),
            )
        )
        return 0
    if args.command == "closed-loop-prepare":
        result = prepare_split_datasets(config, args.output_dir)
        _emit({"config_sha256": config_hash, **result})
        return 0
    if args.command == "closed-loop-match":
        registry = create_parameter_registry(config, args.output)
        _emit(
            {
                "status": "complete",
                "config_sha256": config_hash,
                "registry_sha256": registry["registry_sha256"],
                "output": str(Path(args.output).resolve()),
                "entries": registry["entries"],
            }
        )
        return 0
    if args.command == "closed-loop-train":
        _emit(
            train_registered_model(
                registered_name=args.model,
                seed=args.seed,
                config=config,
                data_root=args.data_root,
                registry_path=args.registry,
                output_dir=args.output_dir,
                device=args.device,
                resume=args.resume,
                expected_pretraining_master_sha256=(
                    args.expected_pretraining_master_sha256
                ),
                pretraining_audit_receipt_path=args.pretraining_audit_receipt,
                expected_pretraining_audit_receipt_sha256=(
                    args.expected_pretraining_audit_receipt_sha256
                ),
                expected_pretraining_audit_receipt_file_sha256=(
                    args.expected_pretraining_audit_receipt_file_sha256
                ),
                run_registry_path=args.run_registry,
                run_attempt_id=args.run_attempt_id,
                run_reservation_token_path=args.run_reservation_token,
                expected_run_reservation_token_sha256=(
                    args.expected_run_reservation_token_sha256
                ),
            )
        )
        return 0
    if args.command == "closed-loop-evaluate":
        _emit(
            evaluate_response_surface(
                config=config,
                data_root=args.data_root,
                checkpoints=args.checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                resume=args.resume,
                expected_master_release_sha256=args.expected_master_release_sha256,
                expected_test_release_sha256=args.expected_test_release_sha256,
                test_claim_custody_receipt_path=(
                    args.test_claim_custody_receipt
                ),
                expected_test_claim_custody_receipt_sha256=(
                    args.expected_test_claim_custody_receipt_sha256
                ),
            )
        )
        return 0
    if args.command == "closed-loop-profile-training":
        _emit(
            profile_optimizer_steps(
                config=config,
                registry_path=args.registry,
                output_path=args.output,
                device=args.device,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
            )
        )
        return 0
    if args.command == "closed-loop-profile-rollout":
        _emit(
            profile_rollout_throughput(
                config=config,
                checkpoints=args.checkpoint,
                output_path=args.output,
                device=args.device,
                episodes=args.episodes,
            )
        )
        return 0
    if args.command == "closed-loop-pilot":
        root = Path(args.output_dir).resolve()
        if root.exists():
            raise FileExistsError(f"Refusing to overwrite pilot: {root}")
        root.mkdir(parents=True, exist_ok=False)
        data_root = root / "data"
        registry_path = root / "parameter_registry.json"
        prepare_split_datasets(config, data_root)
        registry = create_parameter_registry(config, registry_path)
        checkpoints = []
        seeds = [int(value) for value in config.get("training", {}).get("seeds", [101])]
        for model_name in registry["entries"]:
            for seed in seeds:
                run_dir = root / "training" / model_name / f"seed_{seed}"
                result = train_registered_model(
                    registered_name=model_name,
                    seed=seed,
                    config=config,
                    data_root=data_root,
                    registry_path=registry_path,
                    output_dir=run_dir,
                    device=args.device,
                )
                checkpoints.append(result["best_checkpoint"])
        evaluation = evaluate_response_surface(
            config=config,
            data_root=data_root,
            checkpoints=checkpoints,
            output_dir=root / "evaluation",
            device=args.device,
        )
        _emit(
            {
                "status": "complete",
                "config_sha256": config_hash,
                "models": list(registry["entries"]),
                "seeds": seeds,
                "evaluation": evaluation,
            }
        )
        return 0
    return None
