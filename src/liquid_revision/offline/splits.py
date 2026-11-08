"""Frozen, leakage-checked split manifests for the offline benchmarks.

Split construction is intentionally independent of model optimization seeds.  A
manifest is created once with a dedicated split seed, written to JSON, and then
shared by every architecture and training seed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..io_utils import atomic_write_json


DEFAULT_SPLIT_SEED = 20261102


def _string_ids(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _stable_rank(value: object, *, dataset: str, seed: int, stratum: object) -> bytes:
    payload = f"{dataset}|{int(seed)}|{stratum}|{value}".encode("utf-8")
    return hashlib.sha256(payload).digest()


@dataclass(frozen=True)
class SplitManifest:
    """Serializable train/validation/test membership with strict disjointness."""

    dataset: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    split_seed: int = DEFAULT_SPLIT_SEED
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", str(self.dataset))
        object.__setattr__(self, "train_ids", _string_ids(self.train_ids))
        object.__setattr__(self, "val_ids", _string_ids(self.val_ids))
        object.__setattr__(self, "test_ids", _string_ids(self.test_ids))
        object.__setattr__(self, "split_seed", int(self.split_seed))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        if not self.train_ids or not self.val_ids or not self.test_ids:
            raise ValueError("train, validation, and test splits must all be non-empty")
        split_sets = {
            "train": set(self.train_ids),
            "val": set(self.val_ids),
            "test": set(self.test_ids),
        }
        for name, ids in split_sets.items():
            source = getattr(self, f"{name}_ids")
            if len(ids) != len(source):
                raise ValueError(f"Duplicate sample IDs in {name} split")
        if split_sets["train"] & split_sets["val"]:
            raise ValueError("Train and validation IDs overlap")
        if split_sets["train"] & split_sets["test"]:
            raise ValueError("Train and test IDs overlap")
        if split_sets["val"] & split_sets["test"]:
            raise ValueError("Validation and test IDs overlap")

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.train_ids + self.val_ids + self.test_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "split_seed": self.split_seed,
            "splits": {
                "train": list(self.train_ids),
                "val": list(self.val_ids),
                "test": list(self.test_ids),
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitManifest":
        splits = payload.get("splits")
        if not isinstance(splits, Mapping):
            raise ValueError("Split manifest is missing the 'splits' mapping")
        return cls(
            dataset=str(payload["dataset"]),
            train_ids=tuple(splits["train"]),
            val_ids=tuple(splits["val"]),
            test_ids=tuple(splits["test"]),
            split_seed=int(payload.get("split_seed", DEFAULT_SPLIT_SEED)),
            metadata=dict(payload.get("metadata", {})),
            schema_version=int(payload.get("schema_version", 1)),
        )

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def write_split_manifest(path: str | Path, manifest: SplitManifest) -> None:
    """Atomically write a split manifest to an explicitly selected path."""

    atomic_write_json(Path(path), manifest.to_dict())


def load_split_manifest(path: str | Path) -> SplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitManifest.from_dict(payload)


def require_exact_manifest_identity(
    observed: SplitManifest, expected: SplitManifest, *, context: str
) -> None:
    """Reject a stale split even when its train/val/test sets are disjoint.

    Disjointness alone does not prove that the registered seed, fractions,
    algorithm, or memberships match the current formal configuration.  The
    canonical manifest digest binds all of those fields and every sample ID.
    """

    if observed.schema_version != expected.schema_version:
        raise ValueError(
            f"{context} split schema mismatch: "
            f"{observed.schema_version} != {expected.schema_version}"
        )
    if observed.digest != expected.digest:
        raise ValueError(
            f"{context} split manifest differs from the registered seed, "
            "fractions, algorithm, or exact ID membership; refusing reuse"
        )


def validate_har_manifest_identity(
    observed: SplitManifest,
    *,
    train_ids: Sequence[object],
    train_subjects: Sequence[int],
    test_ids: Sequence[object],
    test_subjects: Sequence[int],
    validation_fraction: float,
    split_seed: int,
) -> None:
    expected = create_har_split_manifest(
        train_ids,
        train_subjects,
        test_ids,
        test_subjects,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    require_exact_manifest_identity(observed, expected, context="HAR")


def validate_physionet_manifest_identity(
    observed: SplitManifest,
    *,
    record_ids: Sequence[object],
    outcomes: Sequence[int],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> None:
    expected = create_physionet_split_manifest(
        record_ids,
        outcomes,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )
    require_exact_manifest_identity(observed, expected, context="PhysioNet")


def validate_manifest_coverage(
    manifest: SplitManifest,
    expected_ids: Iterable[object],
    *,
    official_test_ids: Iterable[object] | None = None,
) -> None:
    """Reject missing, extra, duplicated, or redrawn test memberships."""

    expected_sequence = _string_ids(expected_ids)
    expected = set(expected_sequence)
    actual = set(manifest.all_ids)
    if len(expected) != len(expected_sequence):
        raise ValueError("Expected dataset IDs contain duplicates")
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise ValueError(f"Manifest coverage mismatch; missing={missing}, extra={extra}")
    if official_test_ids is not None:
        official = set(_string_ids(official_test_ids))
        if set(manifest.test_ids) != official:
            raise ValueError("Manifest test IDs do not preserve the official test split")


def create_har_split_manifest(
    train_ids: Sequence[object],
    train_subjects: Sequence[int],
    test_ids: Sequence[object],
    test_subjects: Sequence[int],
    *,
    validation_fraction: float = 0.2,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> SplitManifest:
    """Split official HAR training subjects while preserving official test data."""

    train_ids_s = _string_ids(train_ids)
    test_ids_s = _string_ids(test_ids)
    train_subjects_a = np.asarray(train_subjects, dtype=np.int64).reshape(-1)
    test_subjects_a = np.asarray(test_subjects, dtype=np.int64).reshape(-1)
    if len(train_ids_s) != len(train_subjects_a):
        raise ValueError("train_ids and train_subjects must have equal length")
    if len(test_ids_s) != len(test_subjects_a):
        raise ValueError("test_ids and test_subjects must have equal length")
    if len(set(train_ids_s + test_ids_s)) != len(train_ids_s) + len(test_ids_s):
        raise ValueError("HAR sample IDs must be globally unique")
    train_groups = set(int(value) for value in train_subjects_a)
    test_groups = set(int(value) for value in test_subjects_a)
    overlap = train_groups & test_groups
    if overlap:
        raise ValueError(f"Official HAR train/test subjects overlap: {sorted(overlap)}")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if len(train_groups) < 2:
        raise ValueError("At least two official training subjects are required")

    ordered_groups = sorted(
        train_groups,
        key=lambda subject: _stable_rank(
            subject, dataset="uci_har_raw_inertial", seed=split_seed, stratum="subject"
        ),
    )
    n_validation = max(1, int(round(len(ordered_groups) * validation_fraction)))
    n_validation = min(n_validation, len(ordered_groups) - 1)
    validation_groups = set(ordered_groups[:n_validation])
    manifest_train = tuple(
        sample_id
        for sample_id, subject in zip(train_ids_s, train_subjects_a, strict=True)
        if int(subject) not in validation_groups
    )
    manifest_val = tuple(
        sample_id
        for sample_id, subject in zip(train_ids_s, train_subjects_a, strict=True)
        if int(subject) in validation_groups
    )
    manifest = SplitManifest(
        dataset="uci_har_raw_inertial",
        train_ids=manifest_train,
        val_ids=manifest_val,
        test_ids=test_ids_s,
        split_seed=split_seed,
        metadata={
            "algorithm": "stable_sha256_subject_holdout_v1",
            "validation_fraction": float(validation_fraction),
            "official_train_subjects": sorted(train_groups),
            "validation_subjects": sorted(validation_groups),
            "training_subjects": sorted(train_groups - validation_groups),
            "official_test_subjects": sorted(test_groups),
            "official_train_count": len(train_ids_s),
            "official_test_count": len(test_ids_s),
        },
    )
    validate_manifest_coverage(
        manifest, train_ids_s + test_ids_s, official_test_ids=test_ids_s
    )
    return manifest


def _allocate_stratum(n_items: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.asarray(fractions, dtype=np.float64) * n_items
    counts = np.floor(raw).astype(np.int64)
    remainder = n_items - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    positive = [index for index, fraction in enumerate(fractions) if fraction > 0]
    if n_items >= len(positive):
        for index in positive:
            if counts[index] == 0:
                donor = max(positive, key=lambda candidate: int(counts[candidate]))
                if counts[donor] <= 1:
                    break
                counts[donor] -= 1
                counts[index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def create_physionet_split_manifest(
    record_ids: Sequence[object],
    outcomes: Sequence[int],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> SplitManifest:
    """Create one deterministic patient-level, outcome-stratified split."""

    ids = _string_ids(record_ids)
    labels = np.asarray(outcomes).reshape(-1)
    if len(ids) != len(labels):
        raise ValueError("record_ids and outcomes must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("PhysioNet patient IDs must be unique")
    fractions = (float(train_fraction), float(validation_fraction), float(test_fraction))
    if any(value <= 0.0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("train/validation/test fractions must be positive and sum to 1")
    label_values = sorted({int(value) for value in labels})
    if not label_values:
        raise ValueError("No outcomes supplied")

    split_members: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    class_counts: dict[str, dict[str, int]] = {}
    for label in label_values:
        stratum = [
            sample_id
            for sample_id, sample_label in zip(ids, labels, strict=True)
            if int(sample_label) == label
        ]
        stratum.sort(
            key=lambda sample_id: _stable_rank(
                sample_id,
                dataset="physionet2012_set_a",
                seed=split_seed,
                stratum=label,
            )
        )
        n_train, n_val, n_test = _allocate_stratum(len(stratum), fractions)
        boundaries = (n_train, n_train + n_val)
        portions = {
            "train": stratum[: boundaries[0]],
            "val": stratum[boundaries[0] : boundaries[1]],
            "test": stratum[boundaries[1] : boundaries[1] + n_test],
        }
        class_counts[str(label)] = {name: len(values) for name, values in portions.items()}
        for name, values in portions.items():
            split_members[name].extend(values)

    for name in split_members:
        split_members[name].sort(
            key=lambda sample_id: _stable_rank(
                sample_id,
                dataset="physionet2012_set_a",
                seed=split_seed,
                stratum=f"final-{name}",
            )
        )
    manifest = SplitManifest(
        dataset="physionet2012_set_a",
        train_ids=tuple(split_members["train"]),
        val_ids=tuple(split_members["val"]),
        test_ids=tuple(split_members["test"]),
        split_seed=split_seed,
        metadata={
            "algorithm": "stable_sha256_outcome_stratified_v1",
            "fractions": {
                "train": fractions[0],
                "val": fractions[1],
                "test": fractions[2],
            },
            "stratification": "In-hospital_death",
            "class_counts": class_counts,
            "patient_count": len(ids),
        },
    )
    validate_manifest_coverage(manifest, ids)
    return manifest
