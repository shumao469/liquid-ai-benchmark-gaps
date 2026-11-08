from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from liquid_revision.io_utils import sha256_file
from liquid_revision.prepare_data import (
    HAR_REQUIRED,
    HAR_SIGNAL_NAMES,
    extract_archive_with_manifest,
    inspect_archive,
    validate_har_members,
    validate_physionet_members,
    validate_physionet_outcomes,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, content in members.items():
            handle.writestr(name, content)


def test_safe_extract_records_sha256(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    _write_zip(archive, {"folder/a.txt": b"alpha", "folder/b.txt": b"beta"})
    destination = tmp_path / "out"
    manifest = extract_archive_with_manifest(archive, destination)
    assert manifest["member_count"] == 2
    assert (destination / "folder" / "a.txt").read_bytes() == b"alpha"
    member = next(item for item in manifest["members"] if item["path"] == "folder/a.txt")
    assert member["sha256"] == sha256_file(destination / "folder" / "a.txt")


@pytest.mark.parametrize(
    "unsafe_name", ["../escape.txt", "/absolute.txt", "C:/drive.txt", "a/../../b.txt"]
)
def test_archive_rejects_path_traversal(tmp_path: Path, unsafe_name: str) -> None:
    archive = tmp_path / "unsafe.zip"
    _write_zip(archive, {unsafe_name: b"bad"})
    with pytest.raises(ValueError):
        inspect_archive(archive)


def test_archive_rejects_case_collision(tmp_path: Path) -> None:
    archive = tmp_path / "collision.zip"
    _write_zip(archive, {"Data/A.txt": b"a", "data/a.txt": b"b"})
    with pytest.raises(ValueError, match="collision"):
        inspect_archive(archive)


def test_dataset_member_validators() -> None:
    har_names = set(HAR_REQUIRED)
    for signal in HAR_SIGNAL_NAMES:
        split = signal.rsplit("_", 1)[-1].removesuffix(".txt")
        har_names.add(f"UCI HAR Dataset/{split}/Inertial Signals/{signal}")
    assert validate_har_members(har_names)["raw_inertial_signal_files_present"] == 18
    physionet_names = {f"set-a/{130000 + index}.txt" for index in range(4000)}
    assert validate_physionet_members(physionet_names)["record_count"] == 4000


def test_extraction_refuses_existing_destination(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    _write_zip(archive, {"a.txt": b"a"})
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        extract_archive_with_manifest(archive, destination)


def test_outcomes_must_match_record_ids(tmp_path: Path) -> None:
    outcomes = tmp_path / "Outcomes-a.txt"
    outcomes.write_text(
        "RecordID,In-hospital_death\n130000,0\n130001,1\n",
        encoding="utf-8",
    )
    validation = validate_physionet_outcomes(outcomes, {130000, 130001})
    assert validation["row_count"] == 2
    with pytest.raises(ValueError, match="mismatch"):
        validate_physionet_outcomes(outcomes, {130000, 130002})

