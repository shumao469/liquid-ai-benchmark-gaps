from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from liquid_revision.reproduction import evaluation


def _config(seeds):
    path = Path(__file__).resolve().parents[1] / "configs" / "closed_loop_pilot.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["training"]["seeds"] = seeds
    config["environment"]["horizon_steps"] = 8
    config["evaluation"]["burst_lengths"] = [0, 5]
    config["evaluation"]["delay_steps"] = [0, 4]
    config["evaluation"]["failure_tracking_penalty_m"] = 1.0
    return config


def _checkpoint_grid(root, seeds):
    for variant, models in (("unit", evaluation.MODELS), ("scaled", evaluation.CORRECTED)):
        for model in models:
            for seed in seeds:
                checkpoint = root / variant / model / f"seed_{seed}" / "best.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(f"{variant}-{model}-{seed}", encoding="utf-8")


def test_full_reproduction_grid(tmp_path):
    seeds = [101, 202, 303, 404, 505]
    _checkpoint_grid(tmp_path, seeds)
    registry = tmp_path / "registry.json"
    registry.write_text("{}", encoding="utf-8")
    config = _config(seeds)
    plan = evaluation._plan(config, "config", tmp_path, registry, "corrected", 20, 10, "cpu")
    assert len(plan["estimators"]) * len(plan["episodes"]) * len(plan["conditions"]) == 3040
    assert plan["episodes"][0] == {
        "trajectory_id": "npj-mini-0000", "episode_seed": 810000, "schedule_seed": 1910000,
    }
    assert plan["episodes"][-1]["episode_seed"] == 810019
    assert sum(item["readout"] == "scaled" for item in plan["estimators"]) == 15
    config["splits"]["test"]["episode_seed_start"] = 810019
    with pytest.raises(ValueError, match="overlaps a configured split"):
        evaluation._episodes(config, 20)


def test_reuse_and_paired_summary_with_actual_reacher_rollouts(tmp_path, monkeypatch):
    _checkpoint_grid(tmp_path, [101])
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_config([101])), encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text("{}", encoding="utf-8")
    # Exercise the real simulator and archive path with a deterministic estimator fixture.
    monkeypatch.setattr(
        evaluation, "_load_estimator",
        lambda item, *_: {**item, "baseline": evaluation.bench.build_nonlearned_baseline("zoh_pd")},
    )
    args = Namespace(
        config=config, registry=registry, training_dir=tmp_path, output_dir=tmp_path / "original",
        phase="original", reuse_original=None, device="cpu", episodes=1, bootstrap_replicates=10,
    )
    original = evaluation.evaluate(args)
    assert original["episode_rows"] == 40
    args.phase, args.reuse_original = "corrected", args.output_dir
    args.output_dir = tmp_path / "corrected"
    corrected = evaluation.evaluate(args)
    assert (corrected["episode_rows"], corrected["rerun_rollouts"], corrected["reused_rollouts"]) == (40, 12, 28)
    assert len(corrected["original_vs_corrected_paired"]) == 12
    assert all(row["corrected_minus_original_tracking_rmse_m"] == 0 for row in corrected["original_vs_corrected_paired"])
    assert len(list((args.output_dir / "shards").glob("*/step_log.npz"))) == 12
    assert evaluation.evaluate(args) == corrected
    with (args.reuse_original / "episode_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("altered")
    with pytest.raises(ValueError, match="differs from its completion record"):
        evaluation.evaluate(args)
