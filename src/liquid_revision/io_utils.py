"""Small atomic I/O and hashing helpers used by every experiment stage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO


CHUNK_SIZE = 1024 * 1024


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(CHUNK_SIZE)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return sha256_stream(handle)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_tree_manifest(
    root: str | Path, *, suffixes: tuple[str, ...] = (".py", ".xml")
) -> dict[str, Any]:
    """Hash an auditable source tree, independent of Git tracking state."""
    source_root = Path(root).resolve()
    files = {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in suffixes
        and "__pycache__" not in path.parts
    }
    return {
        "root": str(source_root),
        "file_count": len(files),
        "aggregate_sha256": canonical_json_hash(files),
        "files": files,
    }

