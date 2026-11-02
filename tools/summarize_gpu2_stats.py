#!/usr/bin/env python3
"""Extract FeeBL GPU2 provenance and performance records into CSV and JSON.

The reader accepts regular files or directories.  It understands JSON objects,
CSV rows, and the key=value records emitted on FeeBL's stderr/result streams.
Unknown fields are retained and absent fields are represented as empty CSV
cells/JSON nulls; this lets the tool span implementation revisions without
silently inventing zeroes for counters that were not reported.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


BASE_COLUMNS = ["source", "line_number", "record_type"]
GPU2_CUE = re.compile(r"(?:\bgpu2\b|\bgpu2[_-]|\bGPU2\b|\bgpu_mode\s*=\s*2\b)", re.IGNORECASE)
CONTEXT_WORD = re.compile(
    r"^(?:compute_backend|cpu_threads|cpu_variant|reduction_mode|rng_backend|"
    r"gpu_variant|gpu_device|gpu_min_particles|gpu_mode|gpu2_[A-Za-z0-9_.-]+)$"
)
KEY_VALUE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*"
    r"(?P<value>\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[^\s,;]+)"
)
TOKEN = re.compile(
    r"(?P<assignment>[A-Za-z_][A-Za-z0-9_.-]*\s*=\s*"
    r"(?:\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[^\s,;]+))"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_.-]*)"
)


def scalar(value: Any) -> Any:
    """Convert textual scalars while preserving non-scalar JSON values."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        try:
            if text[0] == '"':
                text = json.loads(text)
            else:
                text = shlex.split(text)[0]
        except (ValueError, json.JSONDecodeError):
            text = text[1:-1]
    lower = text.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none", "nan"}:
        return None if lower != "nan" else text
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            pass
    return text


def flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flatten(name, child, output)
    elif isinstance(value, list):
        output[prefix] = json.dumps(value, separators=(",", ":"), sort_keys=True)
    else:
        output[prefix] = scalar(value)


def is_gpu2_mapping(mapping: dict[str, Any]) -> bool:
    for key, value in mapping.items():
        key_lower = key.lower()
        if key_lower.startswith("gpu2") or key_lower == "gpu_mode":
            return True
        if isinstance(value, str) and GPU2_CUE.search(value):
            return True
    return False


def record_type_for(line: str, mapping: dict[str, Any]) -> str:
    lower = line.lower()
    explicit = mapping.get("record_type") or mapping.get("type")
    if isinstance(explicit, str) and "gpu2" in explicit.lower():
        return explicit
    if "provenance" in lower:
        return "provenance"
    if "compute_backend" in lower and "requested=" in lower:
        return "provenance"
    if (
        "stat" in lower
        or "summary" in lower
        or any(key.endswith(("_bytes", "_ms", "_calls", "_operations")) for key in mapping)
    ):
        return "statistics"
    if "error" in lower or "failed" in lower:
        return "error"
    return "gpu2_record"


def parse_key_value_line(line: str) -> dict[str, Any] | None:
    if not GPU2_CUE.search(line):
        return None
    output: dict[str, Any] = {}
    context: str | None = None
    for match in TOKEN.finditer(line):
        assignment = match.group("assignment")
        if assignment:
            kv = KEY_VALUE.fullmatch(assignment)
            if not kv:
                continue
            key = kv.group("key")
            value = scalar(kv.group("value"))
            if key in {"requested", "actual"} and context:
                output[f"{context}_{key}"] = value
            else:
                output[key] = value
                if key.startswith("gpu2_") or key == "gpu_mode":
                    context = key
        else:
            word = match.group("word")
            if word and CONTEXT_WORD.fullmatch(word):
                context = word
    if not output:
        return None
    return output


def iter_text_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        stream = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return
    with stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            mapping: dict[str, Any] | None = None
            if line.lstrip().startswith("{"):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    candidate = None
                if isinstance(candidate, dict):
                    flattened: dict[str, Any] = {}
                    flatten("", candidate, flattened)
                    if is_gpu2_mapping(flattened):
                        mapping = flattened
            if mapping is None:
                mapping = parse_key_value_line(line)
            if mapping is None:
                continue
            mapping = dict(mapping)
            mapping["source"] = str(path)
            mapping["line_number"] = line_number
            mapping["record_type"] = record_type_for(line, mapping)
            yield mapping


def iter_csv_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not any(
                name and (name.lower().startswith("gpu2") or name.lower() == "gpu_mode")
                for name in reader.fieldnames
            ):
                return
            for row_number, row in enumerate(reader, 2):
                converted = {key: scalar(value) for key, value in row.items() if key}
                if not is_gpu2_mapping(converted):
                    continue
                converted["source"] = str(path)
                converted["line_number"] = row_number
                converted["record_type"] = "csv"
                yield converted
    except (OSError, csv.Error) as exc:
        print(f"warning: cannot parse CSV {path}: {exc}", file=sys.stderr)


def files_from_inputs(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files.extend(path for path in item.rglob("*") if path.is_file())
        elif item.is_file():
            files.append(item)
        else:
            raise FileNotFoundError(str(item))
    return sorted(set(path.resolve() for path in files))


def collect(inputs: Iterable[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    files = files_from_inputs(inputs)
    records: list[dict[str, Any]] = []
    for path in files:
        before = len(records)
        records.extend(iter_text_records(path))
        if len(records) == before and path.suffix.lower() == ".csv":
            records.extend(iter_csv_records(path))
    return files, records


def columns_for(records: list[dict[str, Any]]) -> list[str]:
    dynamic = sorted({key for row in records for key in row} - set(BASE_COLUMNS))
    return BASE_COLUMNS + dynamic


def write_csv(records: list[dict[str, Any]], columns: list[str], target: io.TextIOBase) -> None:
    writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow(
            {key: "" if record.get(key) is None else record.get(key, "") for key in columns}
        )


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="log/result file or directory")
    parser.add_argument("--csv", type=Path, help="write a union-schema CSV")
    parser.add_argument("--json", type=Path, help="write a self-describing JSON document")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.csv and args.json and args.csv.resolve() == args.json.resolve():
        raise SystemExit("--csv and --json must name different files")
    try:
        files, records = collect(args.inputs)
    except FileNotFoundError as exc:
        print(f"input does not exist: {exc}", file=sys.stderr)
        return 2
    columns = columns_for(records)
    document = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": [str(path) for path in files],
        "record_count": len(records),
        "columns": columns,
        "records": records,
    }
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            write_csv(records, columns, stream)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not args.csv and not args.json:
        write_csv(records, columns, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
