#!/usr/bin/env python3
"""Write or verify an explicit native build-input identity for a source export.

The identity covers native sources, build configuration and test inputs. It
does not identify the paper, analysis outputs or entire distribution; those
are bound separately by the release file manifest and archive checksum.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIRECTORIES = ("src", "cmake", "examples")
ROOT_FILES = (
    "CMakeLists.txt",
    "CMakePresets.json",
    "CTestConfig.cmake",
    "configure.ac",
    "configure",
    "aclocal.m4",
    "Makefile.in",
    "compile",
    "config.guess",
    "config.sub",
    "install-sh",
    "missing",
)


def inventory(root):
    """Hash the explicitly named build-input trees in deterministic path order."""
    paths = [root / name for name in ROOT_FILES]
    for name in DIRECTORIES:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    if any(not path.is_file() for path in paths):
        raise ValueError("A declared native build input is missing")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def identity(files):
    """Hash UTF-8 canonical JSON of the path-to-file-SHA256 mapping."""
    payload = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def stamp(root, write=False):
    files = inventory(root)
    value = identity(files)
    document = {
        "schema": "feebl.native-source-export/1",
        "identity": value,
        "scope": "Native build configuration, sources and test inputs; not the full distribution",
        "algorithm": "sha256 of UTF-8 JSON files mapping, sorted keys, separators comma/colon, ASCII escaping",
        "directories": list(DIRECTORIES),
        "root_files": list(ROOT_FILES),
        "excluded": ["__pycache__"],
        "files": files,
    }
    marker, manifest = root / "SOURCE_REVISION", root / "SOURCE_CONTENT.json"
    if write:
        manifest.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        marker.write_text(value + "\n", encoding="ascii")
    else:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != document or marker.read_text(encoding="ascii").strip() != value:
            raise ValueError(
                "Native build inputs changed after the source export was stamped"
            )
    return value, len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value, count = stamp(args.root.resolve(), args.write)
    print(f"{value} ({count} native build-input files)")
