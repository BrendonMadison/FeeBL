#!/usr/bin/env python3
"""Plan, validate, and optionally execute conservative FeeBL root-count replays.

This is an external complete-run driver.  It does not split particles, insert
roots into an evolved bunch, implement RQMC, or certify physics convergence.
It changes exactly two integer literals in a selected FeeBL PARAMETERS block:
``n_m`` and ``rndm_seed``.  Every resulting bunch is regenerated at the initial
state and is evolved by a fresh FeeBL process.

Only the Python standard library is used so the audit path does not depend on a
scientific Python installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence


REQUEST_SCHEMA = "feebl.adaptive-root-replay-request.v1"
RUNTIME_SIGNAL_SCHEMA = "feebl.adaptive-root-replay-signal.v2"
RUNTIME_SIGNAL_PREFIX = "FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2"
POLICY_SCHEMA = "feebl.adaptive-root-replay-policy.v1"
MANIFEST_SCHEMA = "feebl.adaptive-root-replay-execution-manifest.v1"
RESULT_SCHEMA = "feebl.adaptive-root-replay-execution-results.v1"
GENERATOR_ID = "feebl_legacy_mc_complete_run_v1"
NORMALIZATION_ID = "feebl_n_m_attempted_legacy_markers_v1"
ROUNDING_ID = "ceil_then_ceil_to_positive_multiple_v1"
ESTIMATOR_ID = "independent_complete_run_sample_se_v1"
COUNT_FORMULA_ID = "n_now_times_se_over_tau_squared_v1"
INT32_MAX = 2_147_483_647
MINIMUM_INDEPENDENT_RUNS = 4
REPLAY_REQUIRED_EXIT_CODE = 75
FORBIDDEN_POLICY_KEY_PARTS = (
    "seed",
    "scramble",
    "realization_id",
    "rng_state",
    "particle_id",
    "representative_id",
)


class ReplayError(RuntimeError):
    """A fail-closed replay-contract violation."""


@dataclass(frozen=True)
class Block:
    kind: str
    name: str
    body_start: int
    body_end: int


@dataclass(frozen=True)
class Assignment:
    key: str
    value: int
    value_start: int
    value_end: int


@dataclass(frozen=True)
class RawAssignment:
    """One quote-aware top-level FeeBL assignment with exact RHS bytes."""

    key: str
    value: str
    value_start: int
    value_end: int


def canonical_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON for the deliberately float-free artifact subset."""
    _reject_float(value, "$artifact")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_float(value: Any, where: str) -> None:
    if isinstance(value, float):
        raise ReplayError(
            f"{where}: binary floating-point JSON values are forbidden; use exact decimal strings"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReplayError(f"{where}: every JSON object key must be a string")
            _reject_float(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_float(item, f"{where}[{index}]")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_hash(text: Any, where: str) -> str:
    if not isinstance(text, str) or re.fullmatch(r"[0-9a-fA-F]{64}", text) is None:
        raise ReplayError(f"{where}: expected a 64-digit SHA-256 hexadecimal string")
    return text.lower()


def checked_file(path: Path, expected: Any, where: str, *, nonempty: bool = True) -> str:
    expected_hash = checked_hash(expected, f"{where}.sha256")
    if not path.is_file():
        raise ReplayError(f"{where}: bound file does not exist: {path}")
    if nonempty and path.stat().st_size == 0:
        raise ReplayError(f"{where}: bound file is empty: {path}")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ReplayError(
            f"{where}: SHA-256 mismatch for {path}; expected {expected_hash}, got {actual}"
        )
    return actual


def resolve_path(base: Path, raw: Any, where: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ReplayError(f"{where}: expected a nonempty path string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def checked_relative_path(raw: Any, where: str, *, leaf: bool = False) -> Path:
    """Return a canonical POSIX relative path with no traversal semantics.

    Execution manifests are portable JSON artifacts, so their relative paths
    always use POSIX separators even when validation is performed on Windows.
    Rejecting both POSIX and Windows absolute/drive forms avoids interpreting a
    re-signed manifest differently on another host.
    """
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ReplayError(f"{where}: expected a nonempty safe relative path")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        "\\" in raw
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.as_posix() != raw
        or not posix.parts
        or any(part in ("", ".", "..") for part in posix.parts)
    ):
        raise ReplayError(f"{where}: expected a canonical safe relative POSIX path")
    if leaf and len(posix.parts) != 1:
        raise ReplayError(f"{where}: expected a single safe filename")
    return Path(*posix.parts)


def resolve_within(base: Path, relative: Path, where: str) -> Path:
    """Resolve a manifest path and prove it remains under its declared root."""
    resolved_base = base.resolve()
    resolved = (resolved_base / relative).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ReplayError(f"{where}: path escapes its declared directory") from exc
    return resolved


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read JSON {path}: {exc}") from exc


def write_canonical(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def with_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in value:
        raise ReplayError("internal error: self_sha256 already exists")
    result = dict(value)
    result["self_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def validate_self_hash(value: Any, schema: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReplayError(f"{path}: top-level JSON value must be an object")
    if value.get("schema") != schema:
        raise ReplayError(f"{path}: expected schema {schema!r}")
    declared = checked_hash(value.get("self_sha256"), f"{path}.self_sha256")
    unsigned = dict(value)
    del unsigned["self_sha256"]
    actual = sha256_bytes(canonical_bytes(unsigned))
    if declared != actual:
        raise ReplayError(
            f"{path}: self SHA-256 mismatch; expected {declared}, recomputed {actual}"
        )
    if path.read_bytes() != canonical_bytes(value):
        raise ReplayError(f"{path}: artifact is not in canonical JSON encoding")
    return value


def decimal_value(value: Any, where: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, str):
        raise ReplayError(f"{where}: exact scalar data must be a decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ReplayError(f"{where}: invalid decimal string {value!r}") from exc
    if not result.is_finite():
        raise ReplayError(f"{where}: scalar must be finite")
    if positive and result <= 0:
        raise ReplayError(f"{where}: scalar must be positive")
    return result


def decimal_text(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    # Avoid the process-global Decimal context silently rounding policy data.
    with localcontext() as context:
        context.prec = max(60, len(value.as_tuple().digits) + 10)
        return str(value.normalize(context=context))


def checked_int(value: Any, where: str, low: int, high: int = INT32_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayError(f"{where}: expected an integer")
    if value < low or value > high:
        raise ReplayError(f"{where}: expected {low} <= value <= {high}, got {value}")
    return value


def parse_runtime_replay_signal(text: str) -> dict[str, str]:
    """Validate FeeBL's atomic process-level replay handoff record.

    A strict record precedes blocked final-summary publication and exit 75. An
    exploratory record is emitted only after those outputs complete and the
    process is committed to exit 0. The record is intentionally not a replay
    request: endpoint values and the hashes required by ``REQUEST_SCHEMA``
    still come from an independently assembled, auditable campaign request.
    This parser connects the runtime controller to that workflow without
    treating an in-process split as a new source root or silently launching
    another process.
    """
    if not isinstance(text, str):
        raise ReplayError("runtime replay signal input must be text")
    candidates = [
        line.strip() for line in text.splitlines() if line.strip().startswith(RUNTIME_SIGNAL_PREFIX)
    ]
    if len(candidates) != 1:
        raise ReplayError("runtime log must contain exactly one atomic adaptive-root replay signal")
    tokens = candidates[0].split()
    if not tokens or tokens[0] != RUNTIME_SIGNAL_PREFIX:
        raise ReplayError("runtime replay signal prefix is malformed")
    fields: dict[str, str] = {}
    for token in tokens[1:]:
        if token.count("=") != 1:
            raise ReplayError(f"runtime replay signal token is malformed: {token!r}")
        key, value = token.split("=", 1)
        if not key or not value or key in fields:
            raise ReplayError(f"runtime replay signal has an empty or duplicate field: {key!r}")
        fields[key] = value

    required = {
        "schema",
        "request_schema",
        "exit_code",
        "publication_blocked",
        "exploratory_outputs_authorized",
        "final_physics_outputs_published",
        "publication_scope",
        "pre_gate_diagnostic_artifacts_may_exist",
        "pre_gate_artifacts_physics_qualification_claim",
        "preexisting_artifacts_not_removed",
        "fresh_run_directory_required",
        "final_polarization_sample_complete",
        "bpm_finalization_complete",
        "global_luminosity_publication_complete",
        "complete_collision",
        "physics_qualification_claim",
        "recommended_action",
        "reason",
        "generator_id",
        "normalization_contract",
        "rndm_seed",
        "n_m_beam1",
        "n_m_beam2",
        "common_n_m",
        "independent_complete_runs",
        "minimum_complete_runs",
        "controller_observations",
        "controller_trace_hash",
        "roots_added_online",
        "fresh_complete_process_required",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise ReplayError("runtime replay signal is missing fields: " + ", ".join(missing))
    expected_tokens = {
        "schema": RUNTIME_SIGNAL_SCHEMA,
        "request_schema": REQUEST_SCHEMA,
        "generator_id": GENERATOR_ID,
        "normalization_contract": NORMALIZATION_ID,
        "publication_scope": "final_summary_outputs_only_v1",
        "pre_gate_diagnostic_artifacts_may_exist": "1",
        "pre_gate_artifacts_physics_qualification_claim": "0",
        "preexisting_artifacts_not_removed": "1",
        "fresh_run_directory_required": "1",
        "final_polarization_sample_complete": "1",
        "bpm_finalization_complete": "1",
        "global_luminosity_publication_complete": "1",
        "complete_collision": "1",
        "physics_qualification_claim": "0",
        "independent_complete_runs": "1",
        "minimum_complete_runs": str(MINIMUM_INDEPENDENT_RUNS),
        "roots_added_online": "0",
        "fresh_complete_process_required": "1",
    }
    for key, expected in expected_tokens.items():
        if fields[key] != expected:
            raise ReplayError(
                f"runtime replay signal {key} must be {expected!r}, got {fields[key]!r}"
            )
    if fields["recommended_action"] not in (
        "request_initial_state_replay",
        "joint_replay_and_mesh_refinement",
    ):
        raise ReplayError("runtime replay signal has no fresh-root replay action")

    def signal_integer(key: str, low: int = 0) -> int:
        value = fields[key]
        if re.fullmatch(r"[0-9]+", value) is None:
            raise ReplayError(f"runtime replay signal {key} must be an integer")
        parsed = int(value)
        if parsed < low:
            raise ReplayError(f"runtime replay signal {key} must be at least {low}")
        return parsed

    blocked = signal_integer("publication_blocked")
    exploratory = signal_integer("exploratory_outputs_authorized")
    exit_code = signal_integer("exit_code")
    common_n_m = signal_integer("common_n_m")
    n_m1 = signal_integer("n_m_beam1", 1)
    n_m2 = signal_integer("n_m_beam2", 1)
    signal_integer("rndm_seed")
    signal_integer("controller_observations", 1)
    signal_integer("controller_trace_hash")
    if blocked not in (0, 1) or exploratory != 1 - blocked:
        raise ReplayError("runtime replay signal publication disposition is inconsistent")
    expected_exit = REPLAY_REQUIRED_EXIT_CODE if blocked else 0
    if exit_code != expected_exit:
        raise ReplayError("runtime replay signal exit code does not match its publication gate")
    final_outputs = signal_integer("final_physics_outputs_published")
    if final_outputs not in (0, 1) or final_outputs != 1 - blocked:
        raise ReplayError("runtime replay signal final-summary publication is inconsistent")
    if common_n_m not in (0, 1) or common_n_m != int(n_m1 == n_m2):
        raise ReplayError("runtime replay signal common_n_m is inconsistent")
    return fields


_HEADER_RE = re.compile(r"(?m)^[ \t]*\$(ACCELERATOR|PARAMETERS)::[ \t]*([A-Za-z_][A-Za-z0-9_.-]*)")


def parse_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    for match in _HEADER_RE.finditer(text):
        brace = text.find("{", match.end())
        if brace < 0:
            raise ReplayError(f"card block {match.group(1)}::{match.group(2)} has no opening brace")
        next_header = _HEADER_RE.search(text, match.end())
        if next_header is not None and next_header.start() < brace:
            raise ReplayError(f"card block {match.group(1)}::{match.group(2)} has no body")
        depth = 0
        quote: str | None = None
        escaped = False
        end = -1
        for index in range(brace, len(text)):
            char = text[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in ('"', "'"):
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
                if depth < 0:
                    break
        if end < 0:
            raise ReplayError(f"card block {match.group(1)}::{match.group(2)} is unterminated")
        blocks.append(Block(match.group(1), match.group(2), brace + 1, end))
    if not blocks:
        raise ReplayError("card has no $ACCELERATOR or $PARAMETERS blocks")
    return blocks


def selected_block(text: str, kind: str, name: str) -> Block:
    matches = [item for item in parse_blocks(text) if item.kind == kind and item.name == name]
    if len(matches) != 1:
        raise ReplayError(
            f"card must contain exactly one {kind}::{name} block; found {len(matches)}"
        )
    return matches[0]


_RAW_ASSIGNMENT_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def raw_assignments(text: str, block: Block) -> list[RawAssignment]:
    """Lex top-level ``key=raw RHS;`` statements in one selected block.

    Delimiters inside single- or double-quoted FeeBL strings are data.  A
    backslash escapes the following character only while quoted.  Bare text,
    nested braces, a second unquoted equals sign, missing semicolons, and
    unterminated strings are rejected instead of being partially ignored.
    """
    body = text[block.body_start : block.body_end]
    size = len(body)
    cursor = 0
    result: list[RawAssignment] = []
    observed: set[str] = set()

    def skip_space(index: int) -> int:
        while index < size and body[index].isspace():
            index += 1
        return index

    while True:
        cursor = skip_space(cursor)
        if cursor == size:
            break
        key_match = _RAW_ASSIGNMENT_KEY_RE.match(body, cursor)
        if key_match is None:
            raise ReplayError(
                f"selected {block.kind}::{block.name} contains malformed trailing "
                f"content at body offset {cursor}"
            )
        key = key_match.group(0)
        cursor = skip_space(key_match.end())
        if cursor >= size or body[cursor] != "=":
            raise ReplayError(
                f"selected {block.kind}::{block.name} assignment {key!r} has no equals sign"
            )
        cursor += 1
        raw_start = cursor
        quote: str | None = None
        escaped = False
        semicolon = -1
        while cursor < size:
            char = body[cursor]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
            elif char == ";":
                semicolon = cursor
                break
            elif char == "=":
                raise ReplayError(
                    f"selected {block.kind}::{block.name} assignment {key!r} "
                    "contains an unquoted second equals sign"
                )
            elif char in "{}":
                raise ReplayError(
                    f"selected {block.kind}::{block.name} assignment {key!r} "
                    "contains a nested brace"
                )
            cursor += 1
        if quote is not None:
            raise ReplayError(
                f"selected {block.kind}::{block.name} assignment {key!r} "
                "contains an unterminated quoted string"
            )
        if semicolon < 0:
            raise ReplayError(
                f"selected {block.kind}::{block.name} assignment {key!r} "
                "has no terminating semicolon"
            )
        value_start = raw_start
        while value_start < semicolon and body[value_start].isspace():
            value_start += 1
        value_end = semicolon
        while value_end > value_start and body[value_end - 1].isspace():
            value_end -= 1
        if value_start == value_end:
            raise ReplayError(
                f"selected {block.kind}::{block.name} assignment {key!r} has an empty RHS"
            )
        if key in observed:
            raise ReplayError(
                f"selected {block.kind}::{block.name} contains duplicate assignment {key!r}"
            )
        observed.add(key)
        result.append(
            RawAssignment(
                key=key,
                value=body[value_start:value_end],
                value_start=block.body_start + value_start,
                value_end=block.body_start + value_end,
            )
        )
        cursor = semicolon + 1
    if not result:
        raise ReplayError(f"selected {block.kind}::{block.name} contains no FeeBL assignments")
    return result


def _unquoted_view(text: str) -> str:
    """Keep offsets while hiding quoted content from assignment recognition."""
    visible = list(text)
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            visible[index] = " "
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
            visible[index] = " "
    if quote is not None:
        raise ReplayError("unterminated quoted string in selected card block")
    return "".join(visible)


def assignments(text: str, block: Block, key: str) -> list[Assignment]:
    # FeeBL accepts dotted mirror selectors; callers name those explicitly.
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_.]){re.escape(key)}[ \t\r\n]*=[ \t\r\n]*([+-]?[0-9]+)(?=[ \t\r\n]*;)"
    )
    body = text[block.body_start : block.body_end]
    visible_body = _unquoted_view(body)
    result: list[Assignment] = []
    for match in pattern.finditer(visible_body):
        start = block.body_start + match.start(1)
        end = block.body_start + match.end(1)
        result.append(Assignment(key, int(match.group(1)), start, end))
    any_assignment = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(key)}[ \t\r\n]*=")
    if len(list(any_assignment.finditer(visible_body))) != len(result):
        raise ReplayError(
            f"selected {block.kind}::{block.name} contains a nonliteral or malformed {key} assignment"
        )
    return result


def one_assignment(text: str, block: Block, key: str) -> Assignment:
    found = assignments(text, block, key)
    if len(found) != 1:
        raise ReplayError(
            f"selected {block.kind}::{block.name} must contain exactly one literal {key}= integer assignment; found {len(found)}"
        )
    return found[0]


def optional_int(text: str, block: Block, key: str, default: int) -> int:
    found = assignments(text, block, key)
    if len(found) > 1:
        raise ReplayError(
            f"selected {block.kind}::{block.name} contains duplicate {key} assignments"
        )
    return default if not found else found[0].value


def mirrored_ints(text: str, block: Block, key: str, default: int) -> tuple[int, int]:
    base = optional_int(text, block, key, default)
    one = optional_int(text, block, f"{key}.1", base)
    two = optional_int(text, block, f"{key}.2", base)
    return one, two


def patch_card(
    raw: bytes,
    accelerator_name: str,
    parameters_name: str,
    n_m: int,
    seed: int,
) -> bytes:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplayError("FeeBL template card must be UTF-8") from exc
    # Validate both names even though only the parameter block is edited.
    selected_block(text, "ACCELERATOR", accelerator_name)
    parameter_block = selected_block(text, "PARAMETERS", parameters_name)
    targets = [
        (one_assignment(text, parameter_block, "n_m"), str(n_m)),
        (one_assignment(text, parameter_block, "rndm_seed"), str(seed)),
    ]
    for assignment, replacement in sorted(
        targets, key=lambda item: item[0].value_start, reverse=True
    ):
        text = text[: assignment.value_start] + replacement + text[assignment.value_end :]
    return text.encode("utf-8")


def masked_card_bytes(raw: bytes, accelerator_name: str, parameters_name: str) -> bytes:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplayError("FeeBL card must be UTF-8") from exc
    selected_block(text, "ACCELERATOR", accelerator_name)
    parameter_block = selected_block(text, "PARAMETERS", parameters_name)
    targets = [
        (one_assignment(text, parameter_block, "n_m"), "<FEEBL_REPLAY_N_M>"),
        (one_assignment(text, parameter_block, "rndm_seed"), "<FEEBL_REPLAY_RNDM_SEED>"),
    ]
    for assignment, replacement in sorted(
        targets, key=lambda item: item[0].value_start, reverse=True
    ):
        text = text[: assignment.value_start] + replacement + text[assignment.value_end :]
    return text.encode("utf-8")


def inspect_card(raw: bytes, accelerator_name: str, parameters_name: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplayError("FeeBL template card must be UTF-8") from exc
    accelerator = selected_block(text, "ACCELERATOR", accelerator_name)
    parameters = selected_block(text, "PARAMETERS", parameters_name)
    n_m = one_assignment(text, parameters, "n_m").value
    seed = one_assignment(text, parameters, "rndm_seed").value
    if assignments(text, parameters, "n_m.1") or assignments(text, parameters, "n_m.2"):
        raise ReplayError(
            "beam-specific n_m.1/n_m.2 overrides are unsupported by this first replay driver; use one unambiguous n_m assignment"
        )
    charge_symmetric = optional_int(text, parameters, "charge_symmetric", 0)
    force_symmetric = optional_int(text, parameters, "force_symmetric", 0)
    load_beam = optional_int(text, parameters, "load_beam", 0)
    rndm_load = optional_int(text, parameters, "rndm_load", 1)
    dist_z = mirrored_ints(text, accelerator, "dist_z", 0)
    if n_m <= 0 or n_m > INT32_MAX:
        raise ReplayError(f"template n_m={n_m} is outside FeeBL's supported positive int range")
    if seed <= 0 or seed > INT32_MAX:
        raise ReplayError(f"template rndm_seed={seed} must be in [1,{INT32_MAX}]")
    if load_beam != 0:
        raise ReplayError(
            "external beam loading is unsupported: changing n_m cannot invent independent roots; provide a larger validated ensemble instead"
        )
    if rndm_load != 0:
        raise ReplayError(
            "rndm_load must be explicitly 0: otherwise rndm_seed does not uniquely bind a fresh complete-run realization"
        )
    if charge_symmetric != 0:
        detail = (
            "; the symmetric constant-z path also has the audited factor-four attempt-count ambiguity"
            if 1 in dist_z
            else ""
        )
        raise ReplayError(
            "charge_symmetric is unsupported by this independent-root estimator: four mirror lanes are one correlated family"
            + detail
        )
    if any(value not in (0, 1) for value in dist_z):
        raise ReplayError(
            f"unsupported dist_z values {dist_z}; only audited legacy values 0 and 1 are accepted"
        )
    warnings: list[str] = []
    if 1 in dist_z:
        warnings.append(
            "constant_z_unsymmetric: n_m is treated as attempted legacy markers; no conditional renormalization or symmetry-family claim is made"
        )
    if force_symmetric != 0:
        warnings.append(
            "force_symmetric_non_sampling_switch: retained verbatim; independent uncertainty is still measured across complete runs"
        )
    return {
        "n_m": n_m,
        "rndm_seed": seed,
        "charge_symmetric": charge_symmetric,
        "force_symmetric": force_symmetric,
        "load_beam": load_beam,
        "rndm_load": rndm_load,
        "dist_z": dist_z,
        "warnings": warnings,
        "card_family_sha256": sha256_bytes(
            masked_card_bytes(raw, accelerator_name, parameters_name)
        ),
    }


def forbid_realization_fields(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            folded = key.casefold()
            if any(part in folded for part in FORBIDDEN_POLICY_KEY_PARTS):
                raise ReplayError(
                    f"{where}.{key}: physics/generator policy metadata must be realization-independent"
                )
            forbid_realization_fields(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            forbid_realization_fields(item, f"{where}[{index}]")


def canonical_object_hash(value: Any, where: str) -> str:
    if not isinstance(value, dict) or not value:
        raise ReplayError(f"{where}: expected a nonempty realization-independent object")
    forbid_realization_fields(value, where)
    return sha256_bytes(canonical_bytes(value))


def policy_has_forbidden_fields(value: Any, where: str = "$policy") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            folded = key.casefold()
            if any(part in folded for part in FORBIDDEN_POLICY_KEY_PARTS):
                raise ReplayError(
                    f"{where}.{key}: realization-specific field is forbidden in policy"
                )
            policy_has_forbidden_fields(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            policy_has_forbidden_fields(item, f"{where}[{index}]")


def sample_statistics(values: Sequence[Decimal]) -> tuple[Decimal, Decimal]:
    if len(values) < 2:
        raise ReplayError("at least two values are needed for a sample standard error")
    with localcontext() as context:
        context.prec = 60
        count = Decimal(len(values))
        mean = sum(values, Decimal(0)) / count
        variance = sum((value - mean) ** 2 for value in values) / Decimal(len(values) - 1)
        standard_error = (variance / count).sqrt()
    return mean, standard_error


def ceil_decimal(value: Decimal) -> int:
    if not value.is_finite() or value < 0:
        raise ReplayError("cannot round a negative or nonfinite count")
    integral = int(value)
    return integral if value == Decimal(integral) else integral + 1


def ceil_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def required_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReplayError(f"{where}: expected an object")
    return value


def required_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReplayError(f"{where}: expected an array")
    return value


def validate_binding_set(
    bindings: Mapping[str, Any],
    expected: Mapping[str, str],
    where: str,
) -> None:
    for key, expected_value in expected.items():
        actual = bindings.get(key)
        if actual != expected_value:
            raise ReplayError(
                f"{where}.{key}: compatibility mismatch; expected {expected_value!r}, got {actual!r}"
            )


def plan_request(request_path: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    request_path = request_path.resolve()
    base = request_path.parent
    request = load_json(request_path)
    request = required_mapping(request, "$request")
    _reject_float(request, "$request")
    if request.get("schema") != REQUEST_SCHEMA:
        raise ReplayError(f"request schema must be {REQUEST_SCHEMA!r}")

    template_spec = required_mapping(request.get("template_card"), "template_card")
    template_path = resolve_path(base, template_spec.get("path"), "template_card.path")
    template_hash = checked_file(template_path, template_spec.get("sha256"), "template_card")
    accelerator_name = template_spec.get("accelerator")
    parameters_name = template_spec.get("parameters")
    if not isinstance(accelerator_name, str) or not accelerator_name:
        raise ReplayError("template_card.accelerator must be a nonempty name")
    if not isinstance(parameters_name, str) or not parameters_name:
        raise ReplayError("template_card.parameters must be a nonempty name")
    template_raw = template_path.read_bytes()
    card_info = inspect_card(template_raw, accelerator_name, parameters_name)

    binding_spec = required_mapping(request.get("bindings"), "bindings")
    binary_path = resolve_path(
        base, binding_spec.get("feebl_binary_path"), "bindings.feebl_binary_path"
    )
    binary_hash = checked_file(
        binary_path,
        binding_spec.get("feebl_binary_sha256"),
        "bindings.feebl_binary",
    )
    source_state_path = resolve_path(
        base, binding_spec.get("source_state_path"), "bindings.source_state_path"
    )
    source_state_hash = checked_file(
        source_state_path,
        binding_spec.get("source_state_sha256"),
        "bindings.source_state",
    )
    generator = required_mapping(
        binding_spec.get("generator_transform"), "bindings.generator_transform"
    )
    if generator.get("id") != GENERATOR_ID:
        raise ReplayError(
            f"only generator id {GENERATOR_ID!r} is implemented; this driver does not implement or claim RQMC"
        )
    generator_hash = canonical_object_hash(generator, "bindings.generator_transform")
    physics_family = required_mapping(binding_spec.get("physics_family"), "bindings.physics_family")
    physics_hash = canonical_object_hash(physics_family, "bindings.physics_family")

    sampling = required_mapping(request.get("sampling"), "sampling")
    n_now = checked_int(sampling.get("n_now"), "sampling.n_now", 1)
    if n_now != card_info["n_m"]:
        raise ReplayError(
            f"sampling.n_now={n_now} does not match selected template n_m={card_info['n_m']}"
        )
    if sampling.get("normalization_contract") != NORMALIZATION_ID:
        raise ReplayError(f"sampling.normalization_contract must be {NORMALIZATION_ID!r}")
    round_to = checked_int(sampling.get("round_up_to_multiple"), "sampling.round_up_to_multiple", 1)
    minimum_runs = checked_int(
        sampling.get("minimum_complete_runs", MINIMUM_INDEPENDENT_RUNS),
        "sampling.minimum_complete_runs",
        MINIMUM_INDEPENDENT_RUNS,
        10_000,
    )
    safety_factor = decimal_value(
        sampling.get("standard_error_safety_factor"),
        "sampling.standard_error_safety_factor",
        positive=True,
    )
    if safety_factor < 1:
        raise ReplayError("sampling.standard_error_safety_factor must be >= 1")

    endpoints_raw = required_list(request.get("endpoints"), "endpoints")
    if not endpoints_raw:
        raise ReplayError("endpoints must contain at least one scalar observable endpoint")
    endpoint_specs: dict[str, dict[str, Any]] = {}
    for index, raw_endpoint in enumerate(endpoints_raw):
        endpoint = required_mapping(raw_endpoint, f"endpoints[{index}]")
        endpoint_id = endpoint.get("id")
        if not isinstance(endpoint_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", endpoint_id):
            raise ReplayError(f"endpoints[{index}].id is invalid")
        if endpoint_id in endpoint_specs:
            raise ReplayError(f"duplicate endpoint id {endpoint_id!r}")
        scale = decimal_value(endpoint.get("scale"), f"endpoints[{index}].scale", positive=True)
        target_fraction = decimal_value(
            endpoint.get("target_relative_standard_error"),
            f"endpoints[{index}].target_relative_standard_error",
            positive=True,
        )
        endpoint_specs[endpoint_id] = {
            "scale": scale,
            "target_fraction": target_fraction,
            "units": endpoint.get("units", "dimensionless"),
        }
        if not isinstance(endpoint_specs[endpoint_id]["units"], str):
            raise ReplayError(f"endpoints[{index}].units must be a string")

    expected_bindings = {
        "template_card_family_sha256": card_info["card_family_sha256"],
        "feebl_binary_sha256": binary_hash,
        "source_state_sha256": source_state_hash,
        "generator_transform_sha256": generator_hash,
        "physics_family_sha256": physics_hash,
    }

    pilots = required_list(request.get("pilot_runs"), "pilot_runs")
    if len(pilots) < minimum_runs:
        raise ReplayError(
            f"inadequate independent repetitions: need at least {minimum_runs} complete pilot runs, got {len(pilots)}"
        )
    seen_pilot_seeds: set[int] = set()
    values_by_endpoint: dict[str, list[Decimal]] = {name: [] for name in endpoint_specs}
    pilot_provenance: list[dict[str, str]] = []
    for index, raw_pilot in enumerate(pilots):
        where = f"pilot_runs[{index}]"
        pilot = required_mapping(raw_pilot, where)
        seed = checked_int(pilot.get("seed"), f"{where}.seed", 1)
        if seed in seen_pilot_seeds:
            raise ReplayError(f"{where}.seed={seed} duplicates another pilot realization")
        seen_pilot_seeds.add(seed)
        if pilot.get("complete") is not True or pilot.get("exit_code") != 0:
            raise ReplayError(f"{where}: pilot must assert complete=true and exit_code=0")
        if checked_int(pilot.get("n_m"), f"{where}.n_m", 1) != n_now:
            raise ReplayError(f"{where}.n_m must equal sampling.n_now={n_now}")
        pilot_card_path = resolve_path(base, pilot.get("card_path"), f"{where}.card_path")
        pilot_card_hash = checked_file(pilot_card_path, pilot.get("card_sha256"), f"{where}.card")
        pilot_card_raw = pilot_card_path.read_bytes()
        pilot_info = inspect_card(pilot_card_raw, accelerator_name, parameters_name)
        if pilot_info["n_m"] != n_now or pilot_info["rndm_seed"] != seed:
            raise ReplayError(
                f"{where}: pilot card n_m/rndm_seed does not match declared {n_now}/{seed}"
            )
        if pilot_info["card_family_sha256"] != card_info["card_family_sha256"]:
            raise ReplayError(f"{where}: pilot card differs from template outside n_m/rndm_seed")
        evidence_path = resolve_path(base, pilot.get("evidence_path"), f"{where}.evidence_path")
        evidence_hash = checked_file(
            evidence_path, pilot.get("evidence_sha256"), f"{where}.evidence"
        )
        pilot_bindings = required_mapping(pilot.get("bindings"), f"{where}.bindings")
        validate_binding_set(pilot_bindings, expected_bindings, f"{where}.bindings")
        pilot_values = required_mapping(pilot.get("values"), f"{where}.values")
        if set(pilot_values) != set(endpoint_specs):
            raise ReplayError(
                f"{where}.values must contain exactly endpoint ids {sorted(endpoint_specs)}"
            )
        for endpoint_id in endpoint_specs:
            values_by_endpoint[endpoint_id].append(
                decimal_value(pilot_values[endpoint_id], f"{where}.values.{endpoint_id}")
            )
        pilot_provenance.append(
            {
                "card_sha256": pilot_card_hash,
                "completion_evidence_sha256": evidence_hash,
            }
        )

    production_seeds_raw = required_list(request.get("production_seeds"), "production_seeds")
    if not production_seeds_raw:
        raise ReplayError("production_seeds must contain at least one holdout complete-run seed")
    production_seeds: list[int] = []
    production_seen: set[int] = set()
    for index, raw_seed in enumerate(production_seeds_raw):
        seed = checked_int(raw_seed, f"production_seeds[{index}]", 1)
        if seed in production_seen:
            raise ReplayError(f"duplicate production seed {seed}")
        if seed in seen_pilot_seeds:
            raise ReplayError(f"production seed {seed} was used to select the pilot policy")
        production_seen.add(seed)
        production_seeds.append(seed)
    if len(production_seeds) < minimum_runs:
        raise ReplayError(
            "inadequate independent production repetitions: "
            f"need at least {minimum_runs} complete holdout runs, "
            f"got {len(production_seeds)}"
        )
    # COUNT_FORMULA_ID uses the standard error of the pilot-run mean and has
    # no replicate-count ratio.  It is therefore valid only when the planned
    # production mean has the same number of independent complete runs.  A
    # future schema may include the explicit R_pilot/R_production factor; v1
    # fails closed instead of silently changing its statistical estimand.
    if len(production_seeds) != len(pilots):
        raise ReplayError(
            "production complete-run count must equal pilot complete-run count "
            f"for {COUNT_FORMULA_ID}: pilots={len(pilots)}, "
            f"production={len(production_seeds)}"
        )

    limits = required_mapping(request.get("replay_limits"), "replay_limits")
    maximum_attempts = checked_int(
        limits.get("maximum_attempts"), "replay_limits.maximum_attempts", 1, 100
    )
    attempt_index = checked_int(
        limits.get("attempt_index"), "replay_limits.attempt_index", 1, maximum_attempts
    )
    maximum_n_m = checked_int(limits.get("maximum_n_m"), "replay_limits.maximum_n_m", n_now + 1)
    minimum_progress = decimal_value(
        limits.get("minimum_progress_fraction"),
        "replay_limits.minimum_progress_fraction",
        positive=True,
    )
    if minimum_progress >= 1:
        raise ReplayError("replay_limits.minimum_progress_fraction must be < 1")

    endpoint_results: list[dict[str, str]] = []
    raw_requirements: list[Decimal] = []
    current_ratios: dict[str, Decimal] = {}
    with localcontext() as context:
        context.prec = 60
        for endpoint_id in sorted(endpoint_specs):
            spec = endpoint_specs[endpoint_id]
            mean, observed_se = sample_statistics(values_by_endpoint[endpoint_id])
            conservative_se = observed_se * safety_factor
            tau = spec["scale"] * spec["target_fraction"]
            ratio = conservative_se / tau
            required = Decimal(n_now) * ratio * ratio
            current_ratios[endpoint_id] = ratio
            raw_requirements.append(required)
            endpoint_results.append(
                {
                    "id": endpoint_id,
                    "units": spec["units"],
                    "scale": decimal_text(spec["scale"]),
                    "target_relative_standard_error": decimal_text(spec["target_fraction"]),
                    "target_absolute_standard_error": decimal_text(tau),
                    "pilot_complete_run_count": str(len(values_by_endpoint[endpoint_id])),
                    "pilot_mean": decimal_text(mean),
                    "pilot_sample_standard_error": decimal_text(observed_se),
                    "conservative_standard_error": decimal_text(conservative_se),
                    "conservative_se_over_target": decimal_text(ratio),
                    "required_n_m_unrounded": decimal_text(required),
                }
            )

    raw_target = max(raw_requirements)
    target_n_m = ceil_multiple(ceil_decimal(raw_target), round_to)
    if target_n_m <= n_now:
        raise ReplayError(
            "pilot standard errors do not justify an n_m increase under the frozen formula; no replay plan was emitted"
        )
    if target_n_m > maximum_n_m:
        raise ReplayError(
            f"required n_m={target_n_m} exceeds the explicit resource limit maximum_n_m={maximum_n_m}; target was not clipped"
        )
    if target_n_m > INT32_MAX:
        raise ReplayError(f"required n_m={target_n_m} exceeds FeeBL's int card capacity")

    history = required_list(request.get("replay_history", []), "replay_history")
    if attempt_index == 1 and history:
        raise ReplayError("replay_history must be empty for attempt_index=1")
    if attempt_index > 1:
        if len(history) != attempt_index - 1:
            raise ReplayError(
                "replay_history must contain exactly one frozen summary for every preceding attempt"
            )
        previous_count = 0
        for history_index, raw_history_item in enumerate(history, start=1):
            history_item = required_mapping(
                raw_history_item, f"replay_history[{history_index - 1}]"
            )
            if history_item.get("attempt_index") != history_index:
                raise ReplayError("replay_history attempt indices must be contiguous and canonical")
            history_count = checked_int(
                history_item.get("n_m"),
                f"replay_history[{history_index - 1}].n_m",
                1,
            )
            if history_count <= previous_count:
                raise ReplayError("replay_history n_m counts must increase strictly")
            previous_count = history_count
        previous = required_mapping(history[-1], "replay_history[-1]")
        if previous.get("attempt_index") != attempt_index - 1:
            raise ReplayError("last replay_history attempt_index is not the immediate predecessor")
        previous_n = checked_int(previous.get("n_m"), "replay_history[-1].n_m", 1)
        if n_now <= previous_n:
            raise ReplayError("current n_now must exceed the previous replay attempt's n_m")
        previous_errors = required_mapping(
            previous.get("endpoint_conservative_standard_errors"),
            "replay_history[-1].endpoint_conservative_standard_errors",
        )
        if set(previous_errors) != set(endpoint_specs):
            raise ReplayError("previous endpoint errors do not match the current frozen endpoints")
        previous_ratio = max(
            decimal_value(previous_errors[name], f"replay_history[-1].{name}", positive=True)
            / (endpoint_specs[name]["scale"] * endpoint_specs[name]["target_fraction"])
            for name in endpoint_specs
        )
        current_ratio = max(current_ratios.values())
        progress = (previous_ratio - current_ratio) / previous_ratio
        if progress < minimum_progress:
            raise ReplayError(
                f"replay non-progress: controlling normalized SE improved by {decimal_text(progress)}, below required {decimal_text(minimum_progress)}"
            )

    compatibility = {
        "template_card_family_sha256": card_info["card_family_sha256"],
        "feebl_binary_sha256": binary_hash,
        "source_state_sha256": source_state_hash,
        "generator_transform": generator,
        "generator_transform_sha256": generator_hash,
        "physics_family": physics_family,
        "physics_family_sha256": physics_hash,
    }
    unsigned_policy: dict[str, Any] = {
        "schema": POLICY_SCHEMA,
        "schema_version": "1",
        "artifact_semantics": "transferable_allocation_policy_without_realization_seeds",
        "qualification_status": "planning_only_not_physics_qualified",
        "scientific_claims": [
            "ordinary_MC_uncertainty_is_estimated_across_independent_complete_runs",
            "n_m_is_the_attempted_legacy_physical_marker_count_for_supported_generated_cards",
            "new_roots_are_created_only_by_restart_from_the_initial_state",
            "pilot_realizations_do_not_certify_production_realizations",
            "this_driver_does_not_implement_or_claim_RQMC",
        ],
        "source_distribution_contract": {
            "kind": "feebl_generated_legacy_mc",
            "normalization_contract": NORMALIZATION_ID,
            "charge_symmetric": "0",
            "load_beam": "0",
            "dist_z_beam_1": str(card_info["dist_z"][0]),
            "dist_z_beam_2": str(card_info["dist_z"][1]),
            "warnings": card_info["warnings"],
        },
        "compatibility": compatibility,
        "allocation": {
            "estimator": ESTIMATOR_ID,
            "count_formula": COUNT_FORMULA_ID,
            "pilot_complete_run_count": str(len(pilots)),
            "production_complete_run_count": str(len(production_seeds)),
            "rounding": ROUNDING_ID,
            "n_now": str(n_now),
            "target_n_m": str(target_n_m),
            "round_up_to_multiple": str(round_to),
            "standard_error_safety_factor": decimal_text(safety_factor),
            "selection_rule": "maximum_required_n_m_across_all_frozen_scalar_endpoints",
            "endpoints": endpoint_results,
        },
        "replay_limits": {
            "attempt_index": str(attempt_index),
            "maximum_attempts": str(maximum_attempts),
            "maximum_n_m": str(maximum_n_m),
            "minimum_progress_fraction": decimal_text(minimum_progress),
        },
        "pilot_provenance": {
            "complete_run_count": str(len(pilots)),
            "minimum_complete_run_count": str(minimum_runs),
            "selection_data_not_valid_for_certification": True,
            "opaque_card_and_completion_evidence_hashes": pilot_provenance,
        },
    }
    policy_id = sha256_bytes(canonical_bytes(unsigned_policy))
    unsigned_policy["policy_id"] = f"sha256:{policy_id}"
    policy = with_self_hash(unsigned_policy)
    policy_has_forbidden_fields(policy)

    if output_dir.exists():
        raise ReplayError(f"output directory already exists; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir()
    policy_path = output_dir / "adaptive_root_policy.json"
    write_canonical(policy_path, policy)

    realizations: list[dict[str, Any]] = []
    for ordinal, seed in enumerate(production_seeds, start=1):
        relative_directory = Path("runs") / f"{ordinal:04d}_seed_{seed}"
        run_directory = output_dir / relative_directory
        run_directory.mkdir()
        card = patch_card(template_raw, accelerator_name, parameters_name, target_n_m, seed)
        if (
            sha256_bytes(masked_card_bytes(card, accelerator_name, parameters_name))
            != card_info["card_family_sha256"]
        ):
            raise ReplayError(
                "internal error: generated card changed outside the two permitted literals"
            )
        acc_path = run_directory / "acc.dat"
        acc_path.write_bytes(card)
        generated_info = inspect_card(card, accelerator_name, parameters_name)
        if generated_info["n_m"] != target_n_m or generated_info["rndm_seed"] != seed:
            raise ReplayError(
                "internal error: generated card did not retain requested n_m/rndm_seed"
            )
        realizations.append(
            {
                "ordinal": str(ordinal),
                "seed": str(seed),
                "run_directory": relative_directory.as_posix(),
                "card": (relative_directory / "acc.dat").as_posix(),
                "card_sha256": sha256_bytes(card),
                "card_family_sha256": card_info["card_family_sha256"],
                "n_m": str(target_n_m),
                "expected_output": "out.dat",
                "combined_log": "log.txt",
                "argv": [str(binary_path), accelerator_name, parameters_name, "out.dat"],
            }
        )

    unsigned_manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": "1",
        "execution_semantics": "strictly_sequential_fresh_complete_processes",
        "qualification_status": "unexecuted_or_execution_evidence_only_not_physics_qualified",
        "request_file_sha256": sha256_file(request_path),
        "policy_file": "adaptive_root_policy.json",
        "policy_file_sha256": sha256_file(policy_path),
        "policy_self_sha256": policy["self_sha256"],
        "bindings": {
            "template_card_path": str(template_path),
            "template_card_sha256": template_hash,
            "template_card_family_sha256": card_info["card_family_sha256"],
            "feebl_binary_path": str(binary_path),
            "feebl_binary_sha256": binary_hash,
            "source_state_path": str(source_state_path),
            "source_state_sha256": source_state_hash,
            "generator_transform_sha256": generator_hash,
            "physics_family_sha256": physics_hash,
        },
        "selection": {
            "accelerator": accelerator_name,
            "parameters": parameters_name,
            "n_now": str(n_now),
            "target_n_m": str(target_n_m),
            "pilot_complete_run_count": str(len(pilots)),
            "production_complete_run_count": str(len(production_seeds)),
            "attempt_index": str(attempt_index),
            "maximum_attempts": str(maximum_attempts),
        },
        "realizations": realizations,
    }
    manifest = with_self_hash(unsigned_manifest)
    write_canonical(output_dir / "execution_manifest.json", manifest)
    return policy, manifest


def _int_text(value: Any, where: str, low: int = 1) -> int:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise ReplayError(f"{where}: expected an unsigned decimal string")
    return checked_int(int(value), where, low)


def validate_artifacts(
    policy_path: Path, manifest_path: Path, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_path = policy_path.resolve()
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    policy = validate_self_hash(load_json(policy_path), POLICY_SCHEMA, policy_path)
    manifest = validate_self_hash(load_json(manifest_path), MANIFEST_SCHEMA, manifest_path)
    policy_has_forbidden_fields(policy)
    declared_policy_id = policy.get("policy_id")
    policy_id_input = dict(policy)
    del policy_id_input["self_sha256"]
    policy_id_input.pop("policy_id", None)
    expected_policy_id = f"sha256:{sha256_bytes(canonical_bytes(policy_id_input))}"
    if declared_policy_id != expected_policy_id:
        raise ReplayError(
            "policy_id does not match the canonical realization-independent policy content"
        )
    if manifest.get("policy_file_sha256") != sha256_file(policy_path):
        raise ReplayError("execution manifest does not bind the supplied policy file")
    if manifest.get("policy_self_sha256") != policy.get("self_sha256"):
        raise ReplayError("execution manifest policy self-hash mismatch")

    compatibility = required_mapping(policy.get("compatibility"), "policy.compatibility")
    bindings = required_mapping(manifest.get("bindings"), "manifest.bindings")
    binding_pairs = {
        "template_card_family_sha256": "template_card_family_sha256",
        "feebl_binary_sha256": "feebl_binary_sha256",
        "source_state_sha256": "source_state_sha256",
        "generator_transform_sha256": "generator_transform_sha256",
        "physics_family_sha256": "physics_family_sha256",
    }
    for policy_key, manifest_key in binding_pairs.items():
        if compatibility.get(policy_key) != bindings.get(manifest_key):
            raise ReplayError(f"policy/manifest compatibility mismatch for {policy_key}")
    if (
        required_mapping(
            compatibility.get("generator_transform"), "policy.generator_transform"
        ).get("id")
        != GENERATOR_ID
    ):
        raise ReplayError("policy generator is not the implemented legacy complete-run generator")
    if canonical_object_hash(
        compatibility["generator_transform"], "policy.generator_transform"
    ) != compatibility.get("generator_transform_sha256"):
        raise ReplayError("policy generator transform hash mismatch")
    if canonical_object_hash(
        required_mapping(compatibility.get("physics_family"), "policy.physics_family"),
        "policy.physics_family",
    ) != compatibility.get("physics_family_sha256"):
        raise ReplayError("policy physics-family hash mismatch")

    template_path = Path(str(bindings.get("template_card_path")))
    binary_path = Path(str(bindings.get("feebl_binary_path")))
    source_path = Path(str(bindings.get("source_state_path")))
    checked_file(template_path, bindings.get("template_card_sha256"), "manifest.template_card")
    checked_file(binary_path, bindings.get("feebl_binary_sha256"), "manifest.feebl_binary")
    checked_file(source_path, bindings.get("source_state_sha256"), "manifest.source_state")

    selection = required_mapping(manifest.get("selection"), "manifest.selection")
    accelerator = selection.get("accelerator")
    parameters = selection.get("parameters")
    if not isinstance(accelerator, str) or not isinstance(parameters, str):
        raise ReplayError("manifest selection names are invalid")
    target_n_m = _int_text(selection.get("target_n_m"), "manifest.selection.target_n_m")
    n_now = _int_text(selection.get("n_now"), "manifest.selection.n_now")
    allocation = required_mapping(policy.get("allocation"), "policy.allocation")
    if str(target_n_m) != allocation.get("target_n_m"):
        raise ReplayError("policy/manifest target n_m mismatch")
    pilot_count = _int_text(
        allocation.get("pilot_complete_run_count"),
        "policy.allocation.pilot_complete_run_count",
        MINIMUM_INDEPENDENT_RUNS,
    )
    production_count = _int_text(
        allocation.get("production_complete_run_count"),
        "policy.allocation.production_complete_run_count",
        MINIMUM_INDEPENDENT_RUNS,
    )
    if pilot_count != production_count:
        raise ReplayError(
            "policy pilot/production complete-run counts are inconsistent with "
            f"{COUNT_FORMULA_ID}"
        )
    if (
        _int_text(
            selection.get("pilot_complete_run_count"),
            "manifest.selection.pilot_complete_run_count",
            MINIMUM_INDEPENDENT_RUNS,
        )
        != pilot_count
    ):
        raise ReplayError("policy/manifest pilot complete-run count mismatch")
    if (
        _int_text(
            selection.get("production_complete_run_count"),
            "manifest.selection.production_complete_run_count",
            MINIMUM_INDEPENDENT_RUNS,
        )
        != production_count
    ):
        raise ReplayError("policy/manifest production complete-run count mismatch")
    expected_family = checked_hash(
        bindings.get("template_card_family_sha256"), "manifest.card_family"
    )
    template_info = inspect_card(template_path.read_bytes(), accelerator, parameters)
    if template_info["card_family_sha256"] != expected_family:
        raise ReplayError("bound template does not match the policy card family")
    if template_info["n_m"] != n_now:
        raise ReplayError("bound template n_m does not match manifest n_now")
    realizations = required_list(manifest.get("realizations"), "manifest.realizations")
    if len(realizations) != production_count:
        raise ReplayError(
            "execution manifest realization count does not match the frozen "
            f"production complete-run count {production_count}"
        )
    seen_seeds: set[int] = set()
    for index, raw_realization in enumerate(realizations):
        where = f"manifest.realizations[{index}]"
        realization = required_mapping(raw_realization, where)
        if realization.get("ordinal") != str(index + 1):
            raise ReplayError(f"{where}: noncanonical execution order")
        seed = _int_text(realization.get("seed"), f"{where}.seed")
        if seed in seen_seeds:
            raise ReplayError(f"{where}: duplicate production seed")
        seen_seeds.add(seed)
        if _int_text(realization.get("n_m"), f"{where}.n_m") != target_n_m:
            raise ReplayError(f"{where}: n_m does not match target")
        expected_run_text = f"runs/{index + 1:04d}_seed_{seed}"
        run_relative = checked_relative_path(
            realization.get("run_directory"), f"{where}.run_directory"
        )
        if run_relative.as_posix() != expected_run_text:
            raise ReplayError(
                f"{where}.run_directory: expected canonical path {expected_run_text!r}"
            )
        run_directory = resolve_within(output_dir, run_relative, f"{where}.run_directory")
        if not run_directory.is_dir():
            raise ReplayError(f"{where}.run_directory: directory does not exist")
        card_relative = checked_relative_path(realization.get("card"), f"{where}.card")
        expected_card_text = f"{expected_run_text}/acc.dat"
        if card_relative.as_posix() != expected_card_text:
            raise ReplayError(f"{where}.card: expected canonical path {expected_card_text!r}")
        card_path = resolve_within(output_dir, card_relative, f"{where}.card")
        if card_path.parent != run_directory:
            raise ReplayError(f"{where}.card: card is not inside its run directory")
        expected_output = checked_relative_path(
            realization.get("expected_output"), f"{where}.expected_output", leaf=True
        )
        if expected_output.as_posix() != "out.dat":
            raise ReplayError(f"{where}.expected_output: expected canonical filename 'out.dat'")
        combined_log = checked_relative_path(
            realization.get("combined_log"), f"{where}.combined_log", leaf=True
        )
        if combined_log.as_posix() != "log.txt":
            raise ReplayError(f"{where}.combined_log: expected canonical filename 'log.txt'")
        resolve_within(run_directory, expected_output, f"{where}.expected_output")
        resolve_within(run_directory, combined_log, f"{where}.combined_log")
        checked_file(card_path, realization.get("card_sha256"), f"{where}.card")
        raw = card_path.read_bytes()
        info = inspect_card(raw, accelerator, parameters)
        if info["n_m"] != target_n_m or info["rndm_seed"] != seed:
            raise ReplayError(f"{where}: generated card n_m/rndm_seed mismatch")
        if info["card_family_sha256"] != expected_family:
            raise ReplayError(f"{where}: generated card family mismatch")
        if realization.get("card_family_sha256") != expected_family:
            raise ReplayError(f"{where}: manifest card-family mismatch")
        argv = realization.get("argv")
        expected_argv = [str(binary_path), accelerator, parameters, "out.dat"]
        if argv != expected_argv:
            raise ReplayError(f"{where}: executable argv binding mismatch")
    return policy, manifest


def execute_artifacts(policy_path: Path, manifest_path: Path, output_dir: Path) -> int:
    _, manifest = validate_artifacts(policy_path, manifest_path, output_dir)
    bindings = required_mapping(manifest["bindings"], "manifest.bindings")
    binary = Path(str(bindings["feebl_binary_path"]))
    if os.name != "nt" and not os.access(binary, os.X_OK):
        raise ReplayError(f"bound FeeBL binary is not executable: {binary}")
    results_path = output_dir / "execution_results.json"
    if results_path.exists():
        raise ReplayError(f"refusing to overwrite existing execution evidence: {results_path}")
    results: list[dict[str, Any]] = []
    overall = 0
    for index, raw_realization in enumerate(manifest["realizations"]):
        where = f"manifest.realizations[{index}]"
        realization = required_mapping(raw_realization, where)
        run_directory = resolve_within(
            output_dir,
            checked_relative_path(realization["run_directory"], f"{where}.run_directory"),
            f"{where}.run_directory",
        )
        log_path = resolve_within(
            run_directory,
            checked_relative_path(realization["combined_log"], f"{where}.combined_log", leaf=True),
            f"{where}.combined_log",
        )
        out_path = resolve_within(
            run_directory,
            checked_relative_path(
                realization["expected_output"], f"{where}.expected_output", leaf=True
            ),
            f"{where}.expected_output",
        )
        if log_path.exists() or out_path.exists():
            raise ReplayError(f"refusing to overwrite run artifacts in {run_directory}")
        argv = list(realization["argv"])
        with log_path.open("wb") as log_handle:
            completed = subprocess.run(
                argv,
                cwd=run_directory,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                shell=False,
            )
        record: dict[str, Any] = {
            "ordinal": realization["ordinal"],
            "seed": realization["seed"],
            "exit_code": str(completed.returncode),
            "log_sha256": sha256_file(log_path),
            "complete": completed.returncode == 0
            and out_path.is_file()
            and out_path.stat().st_size > 0,
        }
        if out_path.is_file():
            record["out_dat_sha256"] = sha256_file(out_path)
            record["out_dat_nonempty"] = out_path.stat().st_size > 0
        results.append(record)
        if not record["complete"]:
            overall = 3
            break
    unsigned_results = {
        "schema": RESULT_SCHEMA,
        "schema_version": "1",
        "manifest_file_sha256": sha256_file(manifest_path),
        "execution_order": "sequential_stop_on_first_failure",
        "physics_qualification_claim": False,
        "completed_all_planned_runs": overall == 0
        and len(results) == len(manifest["realizations"]),
        "results": results,
    }
    write_canonical(results_path, with_self_hash(unsigned_results))
    return overall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "plan", help="validate pilot evidence and emit a frozen policy/manifest/cards"
    )
    plan.add_argument("request", type=Path)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument(
        "--run",
        action="store_true",
        help="after full validation, execute production cards strictly sequentially",
    )
    validate = subparsers.add_parser(
        "validate", help="revalidate policy, manifest, bindings, and generated cards"
    )
    validate.add_argument("--policy", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    execute = subparsers.add_parser(
        "execute", help="validate, then execute every planned run sequentially"
    )
    execute.add_argument("--policy", type=Path, required=True)
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            policy, _ = plan_request(args.request, args.output_dir.resolve())
            print(
                "adaptive root replay plan emitted: "
                f"target_n_m={policy['allocation']['target_n_m']} "
                "status=planning_only_not_physics_qualified"
            )
            if args.run:
                return execute_artifacts(
                    args.output_dir / "adaptive_root_policy.json",
                    args.output_dir / "execution_manifest.json",
                    args.output_dir,
                )
            return 0
        if args.command == "validate":
            validate_artifacts(args.policy, args.manifest, args.output_dir)
            print("adaptive root replay artifacts validate")
            return 0
        if args.command == "execute":
            return execute_artifacts(args.policy, args.manifest, args.output_dir)
        raise ReplayError(f"unknown command {args.command}")
    except ReplayError as exc:
        print(f"ADAPTIVE_ROOT_REPLAY_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
