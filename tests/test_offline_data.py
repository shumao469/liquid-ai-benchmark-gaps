from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from liquid_revision.offline.data import (
    HAR_SIGNAL_NAMES,
    SequenceDatasetArrays,
    load_har_dataset,
    load_har_official,
    load_physionet_dataset,
    load_physionet_raw,
)
from liquid_revision.offline.splits import (
    SplitManifest,
    create_har_split_manifest,
    create_physionet_split_manifest,
    load_split_manifest,
    validate_har_manifest_identity,
    validate_physionet_manifest_identity,
    write_split_manifest,
)


def _write_har_fixture(root: Path) -> Path:
    dataset = root / "revision_data" / "har" / "UCI HAR Dataset"
    split_spec = {
        "train": (
            np.asarray([1, 1, 2, 2, 3, 3, 4, 4]),
            np.asarray([1, 2, 3, 4, 5, 6, 1, 2]),
        ),
        "test": (np.asarray([5, 5, 6, 6]), np.asarray([1, 2, 3, 4])),
    }
    for split, (subjects, labels) in split_spec.items():
        signal_dir = dataset / split / "Inertial Signals"
        signal_dir.mkdir(parents=True)
        np.savetxt(dataset / split / f"subject_{split}.txt", subjects, fmt="%d")
        np.savetxt(dataset / split / f"y_{split}.txt", labels, fmt="%d")
        n_samples = len(subjects)
        for channel, name in enumerate(HAR_SIGNAL_NAMES):
            base = np.arange(n_samples, dtype=np.float32)[:, None] + channel * 0.25
            values = np.repeat(base, 128, axis=1)
            if split == "test":
                values += 10_000.0
            np.savetxt(signal_dir / f"{name}_{split}.txt", values, fmt="%.6f")
    return root / "revision_data"


def _har_manifest(raw: dict[str, object]) -> SplitManifest:
    train = raw["official_train"]
    test = raw["official_test"]
    assert isinstance(train, SequenceDatasetArrays)
    assert isinstance(test, SequenceDatasetArrays)
    return create_har_split_manifest(
        train.ids,
        np.asarray(raw["train_subjects"]),
        test.ids,
        np.asarray(raw["test_subjects"]),
        validation_fraction=0.25,
        split_seed=17,
    )


def _write_physionet_fixture(root: Path, n_records: int = 12) -> Path:
    dataset = root / "revision_data" / "physionet2012"
    set_dir = dataset / "set-a"
    set_dir.mkdir(parents=True)
    outcome_lines = [
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death"
    ]
    for index in range(n_records):
        record_id = 100000 + index
        label = index % 2
        (set_dir / f"{record_id}.txt").write_text(
            "\n".join(
                [
                    "Time,Parameter,Value",
                    f"00:00,RecordID,{record_id}",
                    f"00:00,HR,{10 + index}",
                    f"00:30,HR,{12 + index}",
                    f"02:15,Temp,{30 + index}",
                    "03:00,Temp,-1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        outcome_lines.append(f"{record_id},1,1,3,-1,{label}")
    (dataset / "Outcomes-a.txt").write_text(
        "\n".join(outcome_lines) + "\n", encoding="utf-8"
    )
    return root / "revision_data"


def test_sequence_dataset_arrays_validates_and_exposes_mapping() -> None:
    values = np.zeros((2, 3, 4), dtype=np.float32)
    arrays = SequenceDatasetArrays(
        values=values,
        mask=np.ones_like(values),
        age=np.zeros_like(values),
        dt=np.ones((2, 3), dtype=np.float32),
        labels=np.asarray([0, 1]),
        ids=np.asarray(["a", "b"]),
    )
    assert arrays.x.shape == (2, 3, 4)
    assert arrays.as_mapping()["y"].tolist() == [0, 1]
    assert arrays.group_ids.tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="strictly positive"):
        SequenceDatasetArrays(
            values=values,
            mask=np.ones_like(values),
            age=np.zeros_like(values),
            dt=np.zeros((2, 3), dtype=np.float32),
            labels=np.asarray([0, 1]),
            ids=np.asarray(["a", "b"]),
        )


def test_har_uses_raw_nine_channel_windows_and_preserves_official_test(
    tmp_path: Path,
) -> None:
    root = _write_har_fixture(tmp_path)
    raw = load_har_official(root)
    train = raw["official_train"]
    assert isinstance(train, SequenceDatasetArrays)
    assert train.values.shape == (8, 128, 9)
    assert np.allclose(train.dt, 0.02)
    assert np.all(train.mask)
    assert train.values[3, 0, 4] == pytest.approx(4.0)

    manifest = _har_manifest(raw)
    repeated = _har_manifest(raw)
    assert manifest.digest == repeated.digest
    assert manifest.test_ids == tuple(raw["official_test"].ids.tolist())
    assert set(manifest.metadata["training_subjects"]).isdisjoint(
        manifest.metadata["validation_subjects"]
    )


def test_har_normalization_is_fitted_on_frozen_training_subjects_only(
    tmp_path: Path,
) -> None:
    root = _write_har_fixture(tmp_path)
    raw = load_har_official(root)
    manifest = _har_manifest(raw)
    loaded = load_har_dataset(root, manifest)
    official_train = raw["official_train"]
    train_positions = [
        official_train.ids.tolist().index(sample_id) for sample_id in manifest.train_ids
    ]
    expected_mean = official_train.values[train_positions].mean(axis=(0, 1))
    assert np.allclose(loaded["normalization"].mean, expected_mean)
    assert np.allclose(loaded["train"].values.mean(axis=(0, 1)), 0.0, atol=1e-6)
    assert float(np.abs(loaded["test"].values).mean()) > 1000.0
    assert loaded["subjects"]["train"].isdisjoint(loaded["subjects"]["val"])


def test_har_loader_rejects_subject_leakage_in_manual_manifest(tmp_path: Path) -> None:
    root = _write_har_fixture(tmp_path)
    raw = load_har_official(root)
    official_train = raw["official_train"]
    official_test = raw["official_test"]
    leaking = SplitManifest(
        dataset="uci_har_raw_inertial",
        train_ids=tuple(official_train.ids[[0, 2, 3, 4, 5, 6, 7]].tolist()),
        val_ids=(str(official_train.ids[1]),),
        test_ids=tuple(official_test.ids.tolist()),
    )
    with pytest.raises(ValueError, match="leaks subjects"):
        load_har_dataset(root, leaking)


def test_manifest_round_trip_is_lossless(tmp_path: Path) -> None:
    manifest = SplitManifest(
        dataset="fixture", train_ids=("a",), val_ids=("b",), test_ids=("c",)
    )
    path = tmp_path / "split.json"
    write_split_manifest(path, manifest)
    loaded = load_split_manifest(path)
    assert loaded == manifest
    assert loaded.digest == manifest.digest


def test_strict_manifest_identity_rejects_seed_fraction_and_id_drift(tmp_path: Path) -> None:
    root = _write_har_fixture(tmp_path)
    raw = load_har_official(root)
    manifest = _har_manifest(raw)
    validate_har_manifest_identity(
        manifest,
        train_ids=raw["official_train"].ids,
        train_subjects=raw["train_subjects"],
        test_ids=raw["official_test"].ids,
        test_subjects=raw["test_subjects"],
        validation_fraction=0.25,
        split_seed=17,
    )
    with pytest.raises(ValueError, match="registered seed"):
        validate_har_manifest_identity(
            manifest,
            train_ids=raw["official_train"].ids,
            train_subjects=raw["train_subjects"],
            test_ids=raw["official_test"].ids,
            test_subjects=raw["test_subjects"],
            validation_fraction=0.50,
            split_seed=18,
        )

    phys_root = _write_physionet_fixture(tmp_path / "phys")
    phys = load_physionet_raw(
        phys_root, expected_records=12, feature_names=("HR", "Temp"), n_hours=48
    )["all"]
    phys_manifest = create_physionet_split_manifest(
        phys.ids,
        phys.labels,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        split_seed=91,
    )
    validate_physionet_manifest_identity(
        phys_manifest,
        record_ids=phys.ids,
        outcomes=phys.labels,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        split_seed=91,
    )
    drifted_ids = phys.ids.copy()
    drifted_ids[0] = "999999"
    with pytest.raises(ValueError, match="registered seed"):
        validate_physionet_manifest_identity(
            phys_manifest,
            record_ids=drifted_ids,
            outcomes=phys.labels,
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
            split_seed=91,
        )


def test_physionet_parses_masks_age_dt_and_all_outcomes(tmp_path: Path) -> None:
    root = _write_physionet_fixture(tmp_path)
    loaded = load_physionet_raw(
        root, expected_records=12, feature_names=("HR", "Temp"), n_hours=48
    )
    arrays = loaded["all"]
    assert isinstance(arrays, SequenceDatasetArrays)
    assert arrays.values.shape == (12, 48, 2)
    assert arrays.labels.tolist() == [index % 2 for index in range(12)]
    assert arrays.values[0, 0, 0] == pytest.approx(11.0)
    assert arrays.mask[0, 0, 0]
    assert not arrays.mask[0, 1, 0]
    assert arrays.mask[0, 2, 1]
    assert arrays.age[0, 0].tolist() == [0.0, 1.0]
    assert arrays.age[0, 1].tolist() == [1.0, 2.0]
    assert arrays.age[0, 2].tolist() == [2.0, 0.0]
    assert np.all(arrays.dt == 1.0)
    semantics = loaded["temporal_semantics"]
    assert semantics["representation"] == "fixed_hourly_grid"
    assert semantics["horizon_hours"] == 48
    assert semantics["grid_interval_hours"] == 1.0
    assert semantics["dt_value"] == 1.0
    assert semantics["raw_irregular_event_times_preserved"] is False
    assert semantics["missingness_representation"] == [
        "observation_mask",
        "age_since_last_observation_hours",
    ]
    assert "not an irregular-event-time benchmark" in semantics["claim_boundary"]


def test_physionet_split_is_fixed_stratified_and_train_normalized(
    tmp_path: Path,
) -> None:
    root = _write_physionet_fixture(tmp_path)
    raw = load_physionet_raw(
        root, expected_records=12, feature_names=("HR", "Temp"), n_hours=48
    )["all"]
    manifest = create_physionet_split_manifest(
        raw.ids,
        raw.labels,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        split_seed=91,
    )
    repeated = create_physionet_split_manifest(
        list(reversed(raw.ids.tolist())),
        list(reversed(raw.labels.tolist())),
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        split_seed=91,
    )
    assert manifest.digest == repeated.digest
    label_by_id = dict(zip(raw.ids.tolist(), raw.labels.tolist(), strict=True))
    for ids in (manifest.train_ids, manifest.val_ids, manifest.test_ids):
        assert {label_by_id[sample_id] for sample_id in ids} == {0, 1}

    loaded = load_physionet_dataset(
        root,
        manifest,
        expected_records=12,
        feature_names=("HR", "Temp"),
        n_hours=48,
    )
    train_positions = [raw.ids.tolist().index(sample_id) for sample_id in manifest.train_ids]
    observed = raw.mask[train_positions]
    expected_hr_mean = raw.values[train_positions, :, 0][observed[:, :, 0]].mean()
    assert loaded["normalization"].mean[0] == pytest.approx(expected_hr_mean)
    train_hr = loaded["train"].values[:, :, 0][loaded["train"].mask[:, :, 0]]
    assert train_hr.mean() == pytest.approx(0.0, abs=1e-6)
    assert np.all(loaded["test"].values[~loaded["test"].mask] == 0.0)
    assert loaded["temporal_semantics"]["horizon_hours"] == 48


def test_physionet_rejects_incomplete_record_set(tmp_path: Path) -> None:
    root = _write_physionet_fixture(tmp_path, n_records=3)
    with pytest.raises(ValueError, match="Expected 4"):
        load_physionet_raw(
            root, expected_records=4, feature_names=("HR", "Temp"), n_hours=48
        )
