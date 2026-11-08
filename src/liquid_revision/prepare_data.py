"""Secure, manifest-driven, atomic preparation of the public benchmark data."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .io_utils import CHUNK_SIZE, atomic_write_json, sha256_file, source_tree_manifest


MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 2000.0

HAR_REQUIRED = (
    "UCI HAR Dataset/train/X_train.txt",
    "UCI HAR Dataset/train/y_train.txt",
    "UCI HAR Dataset/train/subject_train.txt",
    "UCI HAR Dataset/test/X_test.txt",
    "UCI HAR Dataset/test/y_test.txt",
    "UCI HAR Dataset/test/subject_test.txt",
)
HAR_SIGNAL_NAMES = tuple(
    f"{prefix}_{axis}_{split}.txt"
    for split in ("train", "test")
    for prefix in ("body_acc", "body_gyro", "total_acc")
    for axis in ("x", "y", "z")
)


def _normalized_member_name(raw_name: str) -> str:
    if "\x00" in raw_name:
        raise ValueError("ZIP member contains a NUL byte")
    name = raw_name.replace("\\", "/")
    if name.startswith(("/", "//")) or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"Absolute ZIP member path is forbidden: {raw_name!r}")
    parts = PurePosixPath(name).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe ZIP member path: {raw_name!r}")
    return "/".join(parts)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(unix_mode)


def inspect_archive(
    archive_path: str | Path,
    *,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> list[tuple[zipfile.ZipInfo, str]]:
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    inspected: list[tuple[zipfile.ZipInfo, str]] = []
    seen: dict[str, str] = {}
    total = 0
    with zipfile.ZipFile(archive, "r") as handle:
        bad_crc = handle.testzip()
        if bad_crc is not None:
            raise ValueError(f"ZIP CRC validation failed at {bad_crc!r}")
        for info in handle.infolist():
            normalized = _normalized_member_name(info.filename)
            collision_key = normalized.casefold()
            previous = seen.get(collision_key)
            if previous is not None and previous != normalized:
                raise ValueError(
                    f"Case-insensitive ZIP path collision: {previous!r} and {normalized!r}"
                )
            if previous is not None:
                raise ValueError(f"Duplicate ZIP member: {normalized!r}")
            seen[collision_key] = normalized
            if _is_symlink(info):
                raise ValueError(f"ZIP symlink is forbidden: {normalized!r}")
            if info.is_dir():
                inspected.append((info, normalized))
                continue
            if info.file_size < 0 or info.file_size > max_member_bytes:
                raise ValueError(
                    f"ZIP member size outside limit: {normalized!r} ({info.file_size})"
                )
            total += info.file_size
            if total > max_total_bytes:
                raise ValueError(f"ZIP uncompressed total exceeds {max_total_bytes} bytes")
            compressed = max(1, info.compress_size)
            if info.file_size / compressed > MAX_COMPRESSION_RATIO:
                raise ValueError(f"Suspicious ZIP compression ratio: {normalized!r}")
            inspected.append((info, normalized))
    return inspected


def extract_archive_with_manifest(
    archive_path: str | Path,
    destination: str | Path,
    *,
    validator: Callable[[set[str]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract one archive into an empty staging directory and hash every file."""
    archive = Path(archive_path).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    inspected = inspect_archive(archive)
    member_names = {name for info, name in inspected if not info.is_dir()}
    validation = validator(member_names) if validator else {}
    manifest_members: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive, "r") as handle:
            for info, normalized in inspected:
                target = destination.joinpath(*PurePosixPath(normalized).parts)
                target.resolve().relative_to(destination)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = __import__("hashlib").sha256()
                written = 0
                with handle.open(info, "r") as source, target.open("xb") as sink:
                    while True:
                        chunk = source.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        sink.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                    sink.flush()
                    os.fsync(sink.fileno())
                if written != info.file_size:
                    raise IOError(
                        f"Extracted size mismatch for {normalized}: {written} != {info.file_size}"
                    )
                manifest_members.append(
                    {
                        "path": normalized,
                        "size_bytes": written,
                        "compressed_size_bytes": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                        "sha256": digest.hexdigest(),
                    }
                )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "member_count": len(manifest_members),
        "uncompressed_size_bytes": sum(item["size_bytes"] for item in manifest_members),
        "validation": validation,
        "members": manifest_members,
    }


def validate_har_members(names: set[str]) -> dict[str, Any]:
    missing = [path for path in HAR_REQUIRED if path not in names]
    for signal in HAR_SIGNAL_NAMES:
        split = signal.rsplit("_", 1)[-1].removesuffix(".txt")
        expected = f"UCI HAR Dataset/{split}/Inertial Signals/{signal}"
        if expected not in names:
            missing.append(expected)
    if missing:
        raise ValueError(f"HAR archive is missing {len(missing)} required files: {missing}")
    return {
        "dataset": "UCI HAR",
        "required_files_present": True,
        "raw_inertial_signal_files_present": len(HAR_SIGNAL_NAMES),
        "archive_file_count": len(names),
    }


def validate_physionet_members(names: set[str]) -> dict[str, Any]:
    record_paths = sorted(
        name for name in names if re.fullmatch(r"set-a/\d+\.txt", name)
    )
    if len(record_paths) != 4000:
        raise ValueError(
            f"PhysioNet set-a must contain 4000 record files, found {len(record_paths)}"
        )
    record_ids = [int(Path(name).stem) for name in record_paths]
    if len(set(record_ids)) != 4000:
        raise ValueError("PhysioNet set-a contains duplicate record IDs")
    return {
        "dataset": "PhysioNet Challenge 2012 set-a",
        "record_count": 4000,
        "minimum_record_id": min(record_ids),
        "maximum_record_id": max(record_ids),
    }


def validate_physionet_outcomes(
    source: str | Path, expected_record_ids: set[int]
) -> dict[str, Any]:
    path = Path(source)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"RecordID", "In-hospital_death"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"PhysioNet outcomes must contain columns {sorted(required)}"
            )
        record_ids: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                record_ids.append(int(row["RecordID"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid RecordID in outcomes at line {line_number}"
                ) from exc
    observed = set(record_ids)
    if len(record_ids) != len(observed):
        raise ValueError("PhysioNet outcomes contain duplicate RecordID values")
    missing = sorted(expected_record_ids - observed)
    extra = sorted(observed - expected_record_ids)
    if missing or extra:
        raise ValueError(
            "PhysioNet outcomes/records mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return {
        "row_count": len(record_ids),
        "record_ids_unique": True,
        "record_ids_match_extracted_records": True,
        "columns": reader.fieldnames,
    }


def _copy_with_manifest(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=CHUNK_SIZE)
        dst.flush()
        os.fsync(dst.fileno())
    return {
        "source": str(source.resolve()),
        "path": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def prepare_revision_data(
    *,
    har_zip: str | Path,
    physionet_zip: str | Path,
    physionet_outcomes: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing revision data: {output}. "
            "Use verify-data or select a new output path."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        har_manifest = extract_archive_with_manifest(
            har_zip, staging / "har", validator=validate_har_members
        )
        physionet_dir = staging / "physionet2012"
        physionet_manifest = extract_archive_with_manifest(
            physionet_zip, physionet_dir, validator=validate_physionet_members
        )
        physionet_record_ids = {
            int(Path(member["path"]).stem)
            for member in physionet_manifest["members"]
            if re.fullmatch(r"set-a/\d+\.txt", member["path"])
        }
        outcomes_validation = validate_physionet_outcomes(
            physionet_outcomes, physionet_record_ids
        )
        outcomes_manifest = _copy_with_manifest(
            Path(physionet_outcomes).resolve(), physionet_dir / "Outcomes-a.txt"
        )
        outcomes_manifest["validation"] = outcomes_validation
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "atomic_destination": str(output),
            "implementation": source_tree_manifest(Path(__file__).resolve().parent),
            "har": har_manifest,
            "physionet2012": {
                **physionet_manifest,
                "outcomes": outcomes_manifest,
            },
        }
        atomic_write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_prepared_data(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    manifest_path = root_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    errors: list[str] = []
    for dataset_key, relative_root in (("har", "har"), ("physionet2012", "physionet2012")):
        dataset = manifest[dataset_key]
        base = root_path / relative_root
        expected_relative: set[str] = set()
        for member in dataset["members"]:
            normalized = _normalized_member_name(member["path"])
            expected_relative.add(normalized)
            path = base.joinpath(*PurePosixPath(normalized).parts)
            path.resolve().relative_to(base.resolve())
            if not path.is_file():
                errors.append(f"missing:{path}")
                continue
            if path.stat().st_size != member["size_bytes"]:
                errors.append(f"size:{path}")
                continue
            if sha256_file(path) != member["sha256"]:
                errors.append(f"sha256:{path}")
            checked += 1
        if dataset_key == "physionet2012":
            expected_relative.add(dataset["outcomes"]["path"])
        actual_relative = {
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        }
        for extra in sorted(actual_relative - expected_relative):
            errors.append(f"unmanifested:{base / extra}")
        for absent in sorted(expected_relative - actual_relative):
            errors.append(f"not-listed-on-disk:{base / absent}")
    outcomes = manifest["physionet2012"]["outcomes"]
    outcomes_path = root_path / "physionet2012" / outcomes["path"]
    if (
        not outcomes_path.is_file()
        or outcomes_path.stat().st_size != outcomes["size_bytes"]
        or sha256_file(outcomes_path) != outcomes["sha256"]
    ):
        errors.append(f"outcomes:{outcomes_path}")
    result = {"root": str(root_path), "checked_files": checked + 1, "errors": errors}
    if errors:
        raise ValueError(f"Prepared-data verification failed: {errors[:10]}")
    return result

