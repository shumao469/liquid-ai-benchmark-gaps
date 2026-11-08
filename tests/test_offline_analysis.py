from __future__ import annotations

from pathlib import Path

import numpy as np

from liquid_revision.io_utils import sha256_file
from liquid_revision.offline.analysis import hierarchical_paired_analysis, holm_adjust


MODELS = ("lstm", "ode-rnn", "ltc", "cfc")


def _write_har_prediction(
    path: Path,
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    model: str,
    severity: float,
) -> None:
    logits = np.full((len(labels), 6), -3.0, dtype=np.float32)
    if severity == 0.0 or model in {"ltc", "cfc"}:
        predictions = labels
    else:
        predictions = (labels + 1) % 6
    logits[np.arange(len(labels)), predictions] = 3.0
    np.savez_compressed(
        path,
        ids=np.asarray([f"window-{index}" for index in range(len(labels))]),
        group_ids=groups,
        labels=labels,
        logits=logits,
    )


def test_hierarchical_subject_paired_bootstrap_and_holm(tmp_path: Path) -> None:
    labels = np.arange(6, dtype=np.int64)
    groups = np.asarray(["s1", "s1", "s2", "s2", "s3", "s3"])
    records = []
    for seed in (101, 202):
        for model in MODELS:
            for kind in ("mcar", "temporal_block"):
                for severity in (0.0, 1.0):
                    path = tmp_path / f"{seed}-{model}-{kind}-{severity}.npz"
                    _write_har_prediction(
                        path,
                        labels=labels,
                        groups=groups,
                        model=model,
                        severity=severity,
                    )
                    records.append(
                        {
                            "dataset": "har",
                            "model": model,
                            "seed": seed,
                            "condition_kind": kind,
                            "severity": severity,
                            "predictions": str(path),
                            "predictions_sha256": sha256_file(path),
                        }
                    )
    result = hierarchical_paired_analysis(
        records, bootstrap_replicates=40, analysis_seed=7
    )
    assert len(result["results"]) == 8
    assert result["har_resampling_unit"] == "subject_all_windows"
    assert result["multiplicity"].startswith("Holm step-down")
    for contrast in result["results"]:
        assert contrast["estimate"] > 0
        assert len(contrast["confidence_interval_95_percentile"]) == 2
        assert 0 <= contrast["raw_two_sided_bootstrap_p"] <= 1
        assert contrast["holm_adjusted_p"] >= contrast["raw_two_sided_bootstrap_p"]
        assert contrast["training_seed_count"] == 2


def test_physionet_bootstrap_resamples_patients_within_outcome_strata(tmp_path: Path) -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    groups = np.asarray([f"p{index}" for index in range(len(labels))])
    ids = groups.copy()
    records = []
    for seed in (101, 202):
        for model in MODELS:
            for severity in (0.0, 1.0):
                if severity == 0.0 or model in {"ltc", "cfc"}:
                    logits = np.where(labels == 1, 3.0, -3.0).astype(np.float32)
                else:
                    logits = np.where(labels == 1, -1.0, 1.0).astype(np.float32)
                path = tmp_path / f"phys-{seed}-{model}-{severity}.npz"
                np.savez_compressed(
                    path,
                    ids=ids,
                    group_ids=groups,
                    labels=labels,
                    logits=logits[:, None],
                )
                records.append(
                    {
                        "dataset": "physionet2012",
                        "model": model,
                        "seed": seed,
                        "condition_kind": "mcar",
                        "severity": severity,
                        "predictions": str(path),
                        "predictions_sha256": sha256_file(path),
                    }
                )
    result = hierarchical_paired_analysis(
        records, bootstrap_replicates=20, analysis_seed=11
    )
    assert len(result["results"]) == 4
    assert result["physionet_resampling_unit"] == "patient_stratified_by_outcome"
    assert all(value["estimate"] > 0 for value in result["results"])


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    raw = [0.04, 0.001, 0.02, 0.5]
    adjusted = holm_adjust(raw)
    order = np.argsort(raw)
    sorted_adjusted = np.asarray(adjusted)[order]
    assert np.all(np.diff(sorted_adjusted) >= 0)
    assert np.all(np.asarray(adjusted) >= np.asarray(raw))
