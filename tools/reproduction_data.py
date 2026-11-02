"""Verify or restore the optional FeeBL reproduction data, using Python 3.10+.

The repository-root data_manifest.json identifies every allowed data file and
the complete ZIP archive. Restoration checks the entire archive and existing
files before creating anything. It never replaces an existing file. Keep the
repository and archive unchanged while restoration runs. An interrupted restore
may leave complete files; rerunning safely retains files with matching hashes.
No network access, external commands or simulation are involved.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import zipfile


BUFFER_BYTES = 1024 * 1024
SCHEMA = "feebl.optional-data/1"
REPOSITORY = Path(__file__).resolve().parents[1]
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


class DataError(ValueError):
    """The archive, manifest or destination fails the restoration contract."""


@dataclass(frozen=True)
class DataFile:
    """A regular repository-relative file, with exact byte count and SHA256."""

    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Manifest:
    """The immutable archive identity and complete allowed file list."""

    archive_name: str
    archive_sha256: str
    files: tuple[DataFile, ...]


def safe_relative_path(value: str) -> str:
    """Require a canonical path that is safe on Windows as well as POSIX."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise DataError(f"Invalid repository-relative path: {value!r}")
    for part in value.split("/"):
        if (part in ("", ".", "..") or part.endswith((".", " "))
                or any(ord(character) < 32 or character in ':*?"<>|' for character in part)
                or part.split(".", 1)[0].upper() in RESERVED_NAMES):
            raise DataError(f"Unsafe repository-relative path: {value!r}")
    return value


def checked_digest(value: str) -> str:
    """Reject malformed digests rather than weakening an identity check."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise DataError("Expected a 64-character SHA256 digest")
    return value.lower()


def load_manifest(root: Path) -> Manifest:
    """Validate the schema and reject duplicate, aliased or nested file names."""
    raw = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise DataError(f"Expected manifest schema {SCHEMA}")
    archive_name = safe_relative_path(raw.get("archive_name"))
    if "/" in archive_name:
        raise DataError("archive_name must be a file name, not a path")
    rows = raw.get("files")
    if not isinstance(rows, list) or not rows:
        raise DataError("The manifest must contain a nonempty files list")
    files = []
    names = set()
    for row in rows:
        if not isinstance(row, dict):
            raise DataError("Each manifest file must be an object")
        path = safe_relative_path(row.get("path"))
        size = row.get("bytes")
        if type(size) is not int or size < 0:
            raise DataError(f"Invalid byte count: {path}")
        if path.casefold() in names:
            raise DataError(f"Duplicate or case-aliased manifest path: {path}")
        names.add(path.casefold())
        files.append(DataFile(path, checked_digest(row.get("sha256")), size))
    for name in names:
        parents = name.split("/")[:-1]
        if any("/".join(parents[:index]) in names for index in range(1, len(parents) + 1)):
            raise DataError(f"Manifest file is also a parent directory: {name}")
    return Manifest(archive_name, checked_digest(raw.get("archive_sha256")), tuple(files))


def stream_digest(stream, expected_bytes: int | None = None, destination=None) -> str:
    """Hash in bounded buffers, optionally copying an exactly sized data file."""
    digest = hashlib.sha256()
    total = 0
    while True:
        limit = BUFFER_BYTES if expected_bytes is None else min(BUFFER_BYTES, expected_bytes - total + 1)
        block = stream.read(limit)
        if not block:
            break
        total += len(block)
        if expected_bytes is not None and total > expected_bytes:
            raise DataError("Data exceeds its declared byte count")
        digest.update(block)
        if destination is not None:
            destination.write(block)
    if expected_bytes is not None and total != expected_bytes:
        raise DataError(f"Expected {expected_bytes} bytes, read {total}")
    return digest.hexdigest()


def destination_path(root: Path, name: str) -> Path:
    """Reject links, junctions, reparse points and non-directory ancestors."""
    paths = [root]
    for component in safe_relative_path(name).split("/"):
        paths.append(paths[-1] / component)
    target = paths[-1]
    for path in paths:
        try:
            attributes = path.lstat()
        except FileNotFoundError:
            continue
        reparse = getattr(attributes, "st_file_attributes", 0) & 0x400
        if stat.S_ISLNK(attributes.st_mode) or reparse:
            raise DataError(f"Destination contains a link or reparse point: {path}")
        if path != target and not stat.S_ISDIR(attributes.st_mode):
            raise DataError(f"Destination parent is not a directory: {path}")
        if path == target and not stat.S_ISREG(attributes.st_mode):
            raise DataError(f"Destination is not a regular file: {path}")
    return target


def verify_data(root: Path, manifest: Manifest) -> dict:
    """Report missing and mismatched data without modifying the repository."""
    missing, conflicting = [], []
    for entry in manifest.files:
        target = destination_path(root, entry.path)
        if not target.exists():
            missing.append(entry.path)
        elif target.stat().st_size != entry.bytes:
            conflicting.append(entry.path)
        else:
            with target.open("rb") as stream:
                if stream_digest(stream, entry.bytes) != entry.sha256:
                    conflicting.append(entry.path)
    return {"files": len(manifest.files), "missing": missing, "conflicting": conflicting}


def validate_members(archive: zipfile.ZipFile, manifest: Manifest) -> dict:
    """Accept exactly the regular files in the manifest, including no ZIP extras."""
    entries = {entry.path: entry for entry in manifest.files}
    members = {}
    names = set()
    for member in archive.infolist():
        name = safe_relative_path(member.orig_filename)
        file_type = stat.S_IFMT(member.external_attr >> 16)
        if (name != member.filename or name.casefold() in names
                or member.is_dir() or file_type not in (0, stat.S_IFREG)
                or member.external_attr & (0x10 | 0x400)
                or member.flag_bits & 1):
            raise DataError(f"Duplicate, linked, encrypted or non-regular ZIP member: {name}")
        if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise DataError(f"Unsupported ZIP compression: {name}")
        if name not in entries or member.file_size != entries[name].bytes:
            raise DataError(f"Unexpected ZIP member or byte count: {name}")
        names.add(name.casefold())
        members[name] = member
    if set(members) != set(entries):
        raise DataError("The ZIP archive is missing manifest files")
    return members


def restore_data(root: Path, archive_path: Path, manifest: Manifest, dry_run: bool = False) -> dict:
    """Check everything first, then create only missing files with exclusive writes."""
    with archive_path.open("rb") as archive_stream:
        if stream_digest(archive_stream) != manifest.archive_sha256:
            raise DataError("Archive SHA256 does not match data_manifest.json")
        archive_stream.seek(0)
        with zipfile.ZipFile(archive_stream) as archive:
            members = validate_members(archive, manifest)
            for entry in manifest.files:
                with archive.open(members[entry.path]) as stream:
                    if stream_digest(stream, entry.bytes) != entry.sha256:
                        raise DataError(f"ZIP member SHA256 mismatch: {entry.path}")
            state = verify_data(root, manifest)
            if state["conflicting"]:
                raise DataError("Existing files differ; nothing restored: " + ", ".join(state["conflicting"]))
            if dry_run:
                return {**state, "restored": 0, "dry_run": True}

            # A second bounded pass writes only fully admitted members. Exclusive
            # creation protects an existing file even if it appeared after preflight.
            missing = set(state["missing"])
            for entry in manifest.files:
                if entry.path not in missing:
                    continue
                target = destination_path(root, entry.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                destination_path(root, entry.path)
                destination = target.open("xb")
                complete = False
                try:
                    with destination:
                        with archive.open(members[entry.path]) as stream:
                            digest = stream_digest(stream, entry.bytes, destination)
                        if digest != entry.sha256:
                            raise DataError(f"ZIP member changed during restoration: {entry.path}")
                    complete = True
                finally:
                    # Remove only the file exclusively created by this invocation.
                    # This also handles Ctrl-C; completed files remain reusable.
                    if not complete:
                        target.unlink()
    return {**state, "restored": len(state["missing"]), "dry_run": False}


def main(argv=None, root: Path = REPOSITORY) -> int:
    """Return 0 on success, 2 for missing optional data, and 1 for invalid data."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check installed optional data without writing")
    restore = commands.add_parser("restore", help="Verify an archive, then install missing files")
    restore.add_argument("archive", type=Path, help="Local data ZIP; an absolute path is accepted")
    restore.add_argument("--dry-run", action="store_true", help="Check everything without writing")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(root)
        if args.command == "restore":
            state = restore_data(root, args.archive, manifest, args.dry_run)
            action = "Would restore" if args.dry_run else "Restored"
            count = len(state["missing"]) if args.dry_run else state["restored"]
            print(f"{action} {count} files; {state['files'] - len(state['missing'])} already match.")
            return 0
        state = verify_data(root, manifest)
        for category in ("conflicting", "missing"):
            if state[category]:
                print(f"{category.capitalize()} optional data: {len(state[category])} files")
                for name in state[category][:10]:
                    print(f"  {name}")
                if len(state[category]) > 10:
                    print("  See data_manifest.json for the complete allowed file list.")
        if state["conflicting"]:
            return 1
        if state["missing"]:
            print(f'Restore with: python tools/reproduction_data.py restore "{manifest.archive_name}"')
            return 2
        print(f"Verified all {state['files']} optional data files.")
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Optional data check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
