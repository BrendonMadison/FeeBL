"""Strict data and provenance helpers shared by the offline harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


class Invalid(ValueError):
    """An input or scientific gate did not satisfy its declared contract."""


def require(condition, message):
    if not condition:
        raise Invalid(message)


def finite(value, name="number"):
    require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{name}: expected a number",
    )
    require(math.isfinite(value), f"{name}: must be finite")
    return float(value)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name}: expected integer >= {minimum}")
    return value


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with open(path, encoding="utf-8-sig") as stream:
        return json.load(
            stream,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(Invalid(f"Nonfinite JSON: {value}")),
        )


def write_json(path, value):
    """Never replace a previously reviewed plan or an existing result."""
    with open(path, "x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def keys(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(set(required) <= set(value), f"Missing fields: {sorted(set(required) - set(value))}")
    require(
        set(value) <= set(required) | set(optional),
        f"Unknown fields: {sorted(set(value) - set(required) - set(optional))}",
    )


def child(root, relative):
    require(
        isinstance(relative, str) and relative and "\\" not in relative,
        "Use a nonempty relative POSIX path",
    )
    item = Path(relative)
    require(
        not item.is_absolute() and ":" not in relative and ".." not in item.parts,
        "Path escapes the output root",
    )
    target = (Path(root) / item).resolve()
    require(target.is_relative_to(Path(root).resolve()), "Resolved path escapes root")
    return target


def authentic_artifacts(provenance):
    """Authenticate bytes, not the physical truth of a document or extraction."""
    required = ("source_reference", "extraction_status", "auditor", "raw_artifacts")
    keys(provenance, required, ("source_revision",))
    require(
        provenance["extraction_status"] == "VERIFIED", "Reference extraction has not been verified"
    )
    require(
        bool(provenance["source_reference"]) and bool(provenance["auditor"]),
        "Missing source/auditor",
    )
    require(
        isinstance(provenance["raw_artifacts"], list) and provenance["raw_artifacts"],
        "Missing raw evidence",
    )
    for artifact in provenance["raw_artifacts"]:
        keys(artifact, ("path", "sha256"))
        path = Path(artifact["path"])
        require(path.is_absolute(), f"Raw evidence requires an absolute, unambiguous path: {path}")
        require(path.is_file(), f"Missing raw artifact: {path}")
        require(sha256(path) == artifact["sha256"], f"Raw artifact hash mismatch: {path}")


def publication_scope():
    return "A numeric equivalence result is scoped to the supplied observable, seed ensemble, uncertainty budget and authenticated bytes; it is not a universal physics validation or a publication authorization."
