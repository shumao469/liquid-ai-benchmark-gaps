"""Leakage-safe data loading for the corrected offline benchmarks.

The loaders expose values, natural-observation masks, per-variable observation
ages, explicit elapsed time, labels, and stable sample IDs.  Missing values are
zero only *after* applying statistics fitted on the frozen training split.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .corruption import recompute_age
from .splits import SplitManifest, load_split_manifest, validate_manifest_coverage


HAR_SIGNAL_NAMES = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)
HAR_SAMPLE_PERIOD_SECONDS = 0.02

# Four admission descriptors plus the 37 Challenge time-series variables.  Weight
# appears in the official time-series list and is therefore included only once.
PHYSIONET_FEATURES = (
    "Age",
    "Gender",
    "Height",
    "ICUType",
    "Albumin",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "BUN",
    "Cholesterol",
    "Creatinine",
    "DiasABP",
    "FiO2",
    "GCS",
    "Glucose",
    "HCO3",
    "HCT",
    "HR",
    "K",
    "Lactate",
    "Mg",
    "MAP",
    "MechVent",
    "Na",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "PaCO2",
    "PaO2",
    "Platelets",
    "RespRate",
    "SaO2",
    "SysABP",
    "Temp",
    "TroponinI",
    "TroponinT",
    "Urine",
    "WBC",
    "Weight",
    "pH",
)


def physionet_temporal_semantics(n_hours: int = 48) -> dict[str, Any]:
    """Machine-readable boundary for the registered PhysioNet representation."""

    if int(n_hours) <= 0:
        raise ValueError("n_hours must be positive")
    return {
        "representation": "fixed_hourly_grid",
        "horizon_hours": int(n_hours),
        "sequence_length": int(n_hours),
        "grid_interval_hours": 1.0,
        "dt_units": "hours",
        "dt_value": 1.0,
        "within_bin_aggregation": "mean_of_observed_values",
        "raw_irregular_event_times_preserved": False,
        "missingness_representation": [
            "observation_mask",
            "age_since_last_observation_hours",
        ],
        "claim_boundary": (
            "48-hour hourly-grid benchmark with mask/age channels; "
            "not an irregular-event-time benchmark"
            if int(n_hours) == 48
            else "hourly-grid development representation; not an irregular-event-time benchmark"
        ),
    }


@dataclass(frozen=True)
class SequenceDatasetArrays:
    """Dense sequence arrays with explicit missingness and timing semantics."""

    values: np.ndarray
    mask: np.ndarray
    age: np.ndarray
    dt: np.ndarray
    labels: np.ndarray
    ids: np.ndarray
    group_ids: np.ndarray | None = None
    lengths: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        raw_mask = np.asarray(self.mask)
        if not np.all(np.logical_or(raw_mask == 0, raw_mask == 1)):
            raise ValueError("mask must contain only zero/one values")
        mask = raw_mask.astype(bool, copy=False)
        age = np.asarray(self.age, dtype=np.float32)
        dt = np.asarray(self.dt, dtype=np.float32)
        labels = np.asarray(self.labels).reshape(-1)
        ids = np.asarray([str(value) for value in np.asarray(self.ids).reshape(-1)])
        group_ids = (
            ids.copy()
            if self.group_ids is None
            else np.asarray(
                [str(value) for value in np.asarray(self.group_ids).reshape(-1)]
            )
        )
        lengths = None if self.lengths is None else np.asarray(self.lengths, dtype=np.int64).reshape(-1)

        if values.ndim != 3:
            raise ValueError("values must have shape [sample,time,feature]")
        if mask.shape != values.shape or age.shape != values.shape:
            raise ValueError("mask and age must have the same shape as values")
        if dt.shape != values.shape[:2]:
            raise ValueError("dt must have shape [sample,time]")
        n_samples, n_steps, _ = values.shape
        if (
            labels.shape != (n_samples,)
            or ids.shape != (n_samples,)
            or group_ids.shape != (n_samples,)
        ):
            raise ValueError("labels, ids, and group_ids must have one entry per sample")
        if lengths is not None:
            if lengths.shape != (n_samples,):
                raise ValueError("lengths must have one entry per sample")
            if np.any(lengths <= 0) or np.any(lengths > n_steps):
                raise ValueError("lengths must be within [1, sequence_length]")
        if len(set(ids.tolist())) != n_samples:
            raise ValueError("sample IDs must be unique")
        if not np.isfinite(values).all():
            raise ValueError("values must be finite; use mask for missingness")
        if not np.isfinite(age).all() or np.any(age < 0):
            raise ValueError("age must be finite and non-negative")
        if not np.isfinite(dt).all() or np.any(dt <= 0):
            raise ValueError("dt must be finite and strictly positive")

        object.__setattr__(self, "values", np.ascontiguousarray(values))
        object.__setattr__(self, "mask", np.ascontiguousarray(mask))
        object.__setattr__(self, "age", np.ascontiguousarray(age))
        object.__setattr__(self, "dt", np.ascontiguousarray(dt))
        object.__setattr__(self, "labels", np.ascontiguousarray(labels))
        object.__setattr__(self, "ids", np.ascontiguousarray(ids))
        object.__setattr__(self, "group_ids", np.ascontiguousarray(group_ids))
        object.__setattr__(
            self, "lengths", None if lengths is None else np.ascontiguousarray(lengths)
        )

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def x(self) -> np.ndarray:
        return self.values

    @property
    def y(self) -> np.ndarray:
        return self.labels

    def as_mapping(self) -> dict[str, np.ndarray | None]:
        return {
            "x": self.values,
            "values": self.values,
            "mask": self.mask,
            "age": self.age,
            "dt": self.dt,
            "y": self.labels,
            "labels": self.labels,
            "ids": self.ids,
            "group_ids": self.group_ids,
            "lengths": self.lengths,
        }

    def subset(self, indices: Sequence[int] | np.ndarray) -> "SequenceDatasetArrays":
        selection = np.asarray(indices)
        return SequenceDatasetArrays(
            values=self.values[selection],
            mask=self.mask[selection],
            age=self.age[selection],
            dt=self.dt[selection],
            labels=self.labels[selection],
            ids=self.ids[selection],
            group_ids=self.group_ids[selection],
            lengths=None if self.lengths is None else self.lengths[selection],
        )


@dataclass(frozen=True)
class FeatureNormalization:
    mean: np.ndarray
    std: np.ndarray
    observed_count: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        std = np.asarray(self.std, dtype=np.float32).reshape(-1)
        count = np.asarray(self.observed_count, dtype=np.int64).reshape(-1)
        if mean.shape != std.shape or mean.shape != count.shape:
            raise ValueError("normalization arrays must have equal one-dimensional shape")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("normalization mean/std must be finite and std positive")
        if np.any(count < 0):
            raise ValueError("observed counts must be non-negative")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "observed_count", count)

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "observed_count": self.observed_count.tolist(),
        }


def fit_feature_normalization(
    values: SequenceDatasetArrays | np.ndarray,
    mask: np.ndarray | None = None,
    *,
    minimum_std: float = 1e-6,
) -> FeatureNormalization:
    """Fit per-feature statistics from one training split only."""

    if isinstance(values, SequenceDatasetArrays):
        if mask is not None:
            raise ValueError("mask must be omitted when passing SequenceDatasetArrays")
        array = values.values
        observed = values.mask
    else:
        array = np.asarray(values, dtype=np.float32)
        if mask is None:
            raise ValueError("mask is required when values is an array")
        observed = np.asarray(mask, dtype=bool)
    if array.ndim != 3 or observed.shape != array.shape:
        raise ValueError("values and mask must have shape [sample,time,feature]")
    count = observed.sum(axis=(0, 1), dtype=np.int64)
    sums = np.where(observed, array, 0.0).sum(axis=(0, 1), dtype=np.float64)
    mean = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    centered = np.where(observed, array - mean[None, None, :], 0.0)
    variance = np.divide(
        np.square(centered, dtype=np.float64).sum(axis=(0, 1)),
        count,
        out=np.zeros_like(sums),
        where=count > 0,
    )
    std = np.sqrt(variance)
    std = np.where((count > 1) & (std >= minimum_std), std, 1.0)
    return FeatureNormalization(mean=mean, std=std, observed_count=count)


def apply_feature_normalization(
    dataset: SequenceDatasetArrays, stats: FeatureNormalization
) -> SequenceDatasetArrays:
    if dataset.values.shape[-1] != stats.mean.shape[0]:
        raise ValueError("Feature count does not match normalization statistics")
    normalized = np.zeros_like(dataset.values, dtype=np.float32)
    normalized_observed = (
        dataset.values - stats.mean[None, None, :]
    ) / stats.std[None, None, :]
    normalized[dataset.mask] = normalized_observed[dataset.mask]
    return SequenceDatasetArrays(
        values=normalized,
        mask=dataset.mask,
        age=dataset.age,
        dt=dataset.dt,
        labels=dataset.labels,
        ids=dataset.ids,
        group_ids=dataset.group_ids,
        lengths=dataset.lengths,
    )


def _resolve_har_root(root: str | Path) -> Path:
    root_path = Path(root)
    candidates = (
        root_path,
        root_path / "UCI HAR Dataset",
        root_path / "har" / "UCI HAR Dataset",
    )
    for candidate in candidates:
        if (candidate / "train" / "Inertial Signals").is_dir() and (
            candidate / "test" / "Inertial Signals"
        ).is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot locate UCI HAR Dataset under {root_path}")


def _load_vector(path: Path, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.loadtxt(path, dtype=dtype)
    return np.asarray(array).reshape(-1)


def _load_har_split(dataset_root: Path, split: str) -> tuple[SequenceDatasetArrays, np.ndarray]:
    signal_dir = dataset_root / split / "Inertial Signals"
    matrices: list[np.ndarray] = []
    expected_rows: int | None = None
    for signal in HAR_SIGNAL_NAMES:
        path = signal_dir / f"{signal}_{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        matrix = np.loadtxt(path, dtype=np.float32, ndmin=2)
        if matrix.shape[1] != 128:
            raise ValueError(f"{path} must contain 128 samples per window, got {matrix.shape}")
        if expected_rows is None:
            expected_rows = matrix.shape[0]
        elif matrix.shape[0] != expected_rows:
            raise ValueError(f"Raw HAR signal row mismatch at {path}")
        matrices.append(matrix)
    values = np.stack(matrices, axis=-1).astype(np.float32, copy=False)
    labels_raw = _load_vector(dataset_root / split / f"y_{split}.txt", dtype=np.int64)
    subjects = _load_vector(
        dataset_root / split / f"subject_{split}.txt", dtype=np.int64
    )
    if len(labels_raw) != len(values) or len(subjects) != len(values):
        raise ValueError(f"HAR {split} labels/subjects do not align with raw windows")
    if np.any(labels_raw < 1) or np.any(labels_raw > 6):
        raise ValueError("HAR activity labels must be in the official range 1..6")
    n_samples = len(values)
    ids = np.asarray([f"har-{split}-{index:06d}" for index in range(n_samples)])
    mask = np.ones_like(values, dtype=bool)
    dt = np.full((n_samples, 128), HAR_SAMPLE_PERIOD_SECONDS, dtype=np.float32)
    return (
        SequenceDatasetArrays(
            values=values,
            mask=mask,
            age=np.zeros_like(values, dtype=np.float32),
            dt=dt,
            labels=labels_raw.astype(np.int64) - 1,
            ids=ids,
            group_ids=subjects.astype(str),
            lengths=np.full(n_samples, 128, dtype=np.int64),
        ),
        subjects,
    )


def load_har_official(root: str | Path) -> dict[str, Any]:
    """Read the official raw 9 x 128 windows without engineered features."""

    dataset_root = _resolve_har_root(root)
    official_train, train_subjects = _load_har_split(dataset_root, "train")
    official_test, test_subjects = _load_har_split(dataset_root, "test")
    overlap = set(train_subjects.tolist()) & set(test_subjects.tolist())
    if overlap:
        raise ValueError(f"Official HAR train/test subject leakage: {sorted(overlap)}")
    return {
        "official_train": official_train,
        "official_test": official_test,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "feature_names": HAR_SIGNAL_NAMES,
        "dataset_root": dataset_root,
    }


def _coerce_manifest(manifest: SplitManifest | str | Path) -> SplitManifest:
    return manifest if isinstance(manifest, SplitManifest) else load_split_manifest(manifest)


def _indices_for_ids(dataset: SequenceDatasetArrays, ids: Sequence[str]) -> np.ndarray:
    positions = {sample_id: index for index, sample_id in enumerate(dataset.ids.tolist())}
    missing = [sample_id for sample_id in ids if sample_id not in positions]
    if missing:
        raise ValueError(f"Manifest references unknown IDs: {missing[:10]}")
    return np.asarray([positions[sample_id] for sample_id in ids], dtype=np.int64)


def load_har_dataset(
    root: str | Path,
    manifest: SplitManifest | str | Path,
    *,
    include_test: bool = True,
    preloaded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a frozen subject split and training-only normalization to HAR."""

    loaded = preloaded if preloaded is not None else load_har_official(root)
    split_manifest = _coerce_manifest(manifest)
    if split_manifest.dataset != "uci_har_raw_inertial":
        raise ValueError(f"Unexpected HAR manifest dataset: {split_manifest.dataset}")
    official_train: SequenceDatasetArrays = loaded["official_train"]
    official_test: SequenceDatasetArrays = loaded["official_test"]
    expected_ids = tuple(official_train.ids.tolist()) + tuple(official_test.ids.tolist())
    validate_manifest_coverage(
        split_manifest, expected_ids, official_test_ids=official_test.ids.tolist()
    )

    subject_by_id = {
        sample_id: int(subject)
        for sample_id, subject in zip(
            official_train.ids.tolist(), loaded["train_subjects"], strict=True
        )
    }
    subject_by_id.update(
        {
            sample_id: int(subject)
            for sample_id, subject in zip(
                official_test.ids.tolist(), loaded["test_subjects"], strict=True
            )
        }
    )
    groups = {
        name: {subject_by_id[sample_id] for sample_id in ids}
        for name, ids in (
            ("train", split_manifest.train_ids),
            ("val", split_manifest.val_ids),
            ("test", split_manifest.test_ids),
        )
    }
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups[
        "val"
    ] & groups["test"]:
        raise ValueError("HAR split manifest leaks subjects across splits")

    train_raw = official_train.subset(
        _indices_for_ids(official_train, split_manifest.train_ids)
    )
    val_raw = official_train.subset(_indices_for_ids(official_train, split_manifest.val_ids))
    stats = fit_feature_normalization(train_raw)
    result = {
        "task": "har",
        "train": apply_feature_normalization(train_raw, stats),
        "val": apply_feature_normalization(val_raw, stats),
        "normalization": stats,
        "feature_names": HAR_SIGNAL_NAMES,
        "n_classes": 6,
        "manifest": split_manifest,
        "subjects": groups,
    }
    if include_test:
        test_raw = official_test.subset(
            _indices_for_ids(official_test, split_manifest.test_ids)
        )
        result["test"] = apply_feature_normalization(test_raw, stats)
    return result


def _resolve_physionet_root(root: str | Path) -> Path:
    root_path = Path(root)
    candidates = (root_path, root_path / "physionet2012")
    for candidate in candidates:
        if (candidate / "set-a").is_dir() and (candidate / "Outcomes-a.txt").is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate PhysioNet 2012 set-a under {root_path}")


def _parse_outcomes(path: Path) -> dict[str, int]:
    outcomes: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"RecordID", "In-hospital_death"}
        if reader.fieldnames is None or not required.issubset(
            {name.strip() for name in reader.fieldnames}
        ):
            raise ValueError("Outcomes file must contain RecordID and In-hospital_death")
        for row in reader:
            normalized = {str(key).strip(): value for key, value in row.items()}
            record_id = str(int(float(normalized["RecordID"])))
            label = int(float(normalized["In-hospital_death"]))
            if label not in (0, 1):
                raise ValueError(f"Invalid mortality outcome for patient {record_id}: {label}")
            if record_id in outcomes:
                raise ValueError(f"Duplicate outcome row for patient {record_id}")
            outcomes[record_id] = label
    return outcomes


def _minutes_from_time(value: str) -> int:
    pieces = value.strip().split(":")
    if len(pieces) != 2:
        raise ValueError(f"Invalid PhysioNet time: {value!r}")
    hours, minutes = int(pieces[0]), int(pieces[1])
    if hours < 0 or minutes < 0 or minutes >= 60:
        raise ValueError(f"Invalid PhysioNet time: {value!r}")
    return hours * 60 + minutes


def _read_physionet_record(
    path: Path,
    *,
    feature_names: tuple[str, ...],
    n_hours: int,
    strict_features: bool,
) -> tuple[str, np.ndarray, np.ndarray]:
    feature_index = {name: index for index, name in enumerate(feature_names)}
    if len(feature_index) != len(feature_names):
        raise ValueError("PhysioNet feature names must be unique")
    sums = np.zeros((n_hours, len(feature_names)), dtype=np.float64)
    counts = np.zeros((n_hours, len(feature_names)), dtype=np.int32)
    declared_record_id: str | None = None
    unknown: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"Time", "Parameter", "Value"}.issubset(
            {name.strip() for name in reader.fieldnames}
        ):
            raise ValueError(f"Malformed PhysioNet record header: {path}")
        for row in reader:
            normalized = {str(key).strip(): value for key, value in row.items()}
            parameter = str(normalized["Parameter"]).strip()
            raw_value = str(normalized["Value"]).strip()
            if parameter == "RecordID":
                declared_record_id = str(int(float(raw_value)))
                continue
            if parameter not in feature_index:
                unknown.add(parameter)
                continue
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"Non-numeric value in {path}: {raw_value!r}") from exc
            # The Challenge files use -1 for unavailable admission descriptors.
            if not math.isfinite(value) or value == -1.0:
                continue
            minute = _minutes_from_time(str(normalized["Time"]))
            if minute >= n_hours * 60:
                continue
            hour = minute // 60
            column = feature_index[parameter]
            sums[hour, column] += value
            counts[hour, column] += 1
    stem_id = str(int(path.stem))
    if declared_record_id is None:
        raise ValueError(f"Missing RecordID row in {path}")
    if declared_record_id != stem_id:
        raise ValueError(
            f"RecordID/file mismatch in {path}: {declared_record_id} != {stem_id}"
        )
    if strict_features and unknown:
        raise ValueError(f"Unknown PhysioNet parameters in {path}: {sorted(unknown)}")
    mask = counts > 0
    values = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=mask,
    ).astype(np.float32)
    return stem_id, values, mask


def load_physionet_raw(
    root: str | Path,
    *,
    expected_records: int = 4000,
    feature_names: Sequence[str] = PHYSIONET_FEATURES,
    n_hours: int = 48,
    strict_features: bool = True,
) -> dict[str, Any]:
    """Parse every set-a patient onto a fixed hourly grid without fitting stats."""

    dataset_root = _resolve_physionet_root(root)
    set_dir = dataset_root / "set-a"
    records = sorted(set_dir.glob("*.txt"), key=lambda path: int(path.stem))
    if len(records) != int(expected_records):
        raise ValueError(
            f"Expected {expected_records} PhysioNet records, found {len(records)}"
        )
    if n_hours <= 0:
        raise ValueError("n_hours must be positive")
    feature_tuple = tuple(str(name) for name in feature_names)
    outcomes = _parse_outcomes(dataset_root / "Outcomes-a.txt")
    record_ids = [str(int(path.stem)) for path in records]
    if set(record_ids) != set(outcomes):
        missing_outcomes = sorted(set(record_ids) - set(outcomes))[:10]
        missing_records = sorted(set(outcomes) - set(record_ids))[:10]
        raise ValueError(
            "PhysioNet record/outcome mismatch; "
            f"without_outcomes={missing_outcomes}, without_records={missing_records}"
        )

    values = np.zeros((len(records), n_hours, len(feature_tuple)), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    for index, path in enumerate(records):
        record_id, record_values, record_mask = _read_physionet_record(
            path,
            feature_names=feature_tuple,
            n_hours=n_hours,
            strict_features=strict_features,
        )
        if record_id != record_ids[index]:
            raise ValueError(f"Unexpected record order at {path}")
        values[index] = record_values
        mask[index] = record_mask
    dt = np.ones((len(records), n_hours), dtype=np.float32)
    age = recompute_age(mask, dt)
    arrays = SequenceDatasetArrays(
        values=values,
        mask=mask,
        age=age,
        dt=dt,
        labels=np.asarray([outcomes[record_id] for record_id in record_ids], dtype=np.int64),
        ids=np.asarray(record_ids),
        group_ids=np.asarray(record_ids),
        lengths=np.full(len(records), n_hours, dtype=np.int64),
    )
    return {
        "all": arrays,
        "feature_names": feature_tuple,
        "dataset_root": dataset_root,
        "n_classes": 2,
        "temporal_semantics": physionet_temporal_semantics(n_hours),
    }


def load_physionet_dataset(
    root: str | Path,
    manifest: SplitManifest | str | Path,
    *,
    expected_records: int = 4000,
    feature_names: Sequence[str] = PHYSIONET_FEATURES,
    n_hours: int = 48,
    strict_features: bool = True,
    include_test: bool = True,
    preloaded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one frozen patient split and training-only preprocessing."""

    loaded = (
        preloaded
        if preloaded is not None
        else load_physionet_raw(
            root,
            expected_records=expected_records,
            feature_names=feature_names,
            n_hours=n_hours,
            strict_features=strict_features,
        )
    )
    split_manifest = _coerce_manifest(manifest)
    if split_manifest.dataset != "physionet2012_set_a":
        raise ValueError(f"Unexpected PhysioNet manifest dataset: {split_manifest.dataset}")
    all_data: SequenceDatasetArrays = loaded["all"]
    validate_manifest_coverage(split_manifest, all_data.ids.tolist())
    train_raw = all_data.subset(_indices_for_ids(all_data, split_manifest.train_ids))
    val_raw = all_data.subset(_indices_for_ids(all_data, split_manifest.val_ids))
    stats = fit_feature_normalization(train_raw)
    result = {
        "task": "physionet2012",
        "train": apply_feature_normalization(train_raw, stats),
        "val": apply_feature_normalization(val_raw, stats),
        "normalization": stats,
        "feature_names": loaded["feature_names"],
        "n_classes": 2,
        "manifest": split_manifest,
        "temporal_semantics": dict(loaded["temporal_semantics"]),
    }
    if include_test:
        test_raw = all_data.subset(
            _indices_for_ids(all_data, split_manifest.test_ids)
        )
        result["test"] = apply_feature_normalization(test_raw, stats)
    return result
