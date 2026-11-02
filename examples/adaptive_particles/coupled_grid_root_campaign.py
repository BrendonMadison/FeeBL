#!/usr/bin/env python3
"""Freeze, run, and analyze a bounded FeeBL grid-by-root pilot campaign.

This external driver obtains new statistical roots only through fresh complete
FeeBL processes. It never inserts unevolved particles into a running collision
and never treats grid points or repeated measurements from one process as
independent runs. The v1 design is ordinary Monte Carlo with common seeds
across a full grid x n_m factorial, enabling paired numerical contrasts while
retaining complete runs as the uncertainty unit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence


REPLAY_DIRECTORY = Path(__file__).resolve().parent
if not (REPLAY_DIRECTORY / "adaptive_root_replay.py").is_file():
    # Frozen campaigns retain their self-contained campaign/replay layout.
    REPLAY_DIRECTORY = REPLAY_DIRECTORY.parent / "replay"
if str(REPLAY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPLAY_DIRECTORY))
import adaptive_root_replay as replay  # noqa: E402


REQUEST_SCHEMA = "feebl.coupled-grid-root-campaign-request.v1"
POLICY_SCHEMA = "feebl.coupled-grid-root-campaign-policy.v1"
MANIFEST_SCHEMA = "feebl.coupled-grid-root-pilot-manifest.v1"
STATE_SCHEMA = "feebl.coupled-grid-root-pilot-state.v1"
ANALYSIS_SCHEMA = "feebl.coupled-grid-root-pilot-analysis.v1"
NEXT_MANIFEST_SCHEMA = "feebl.coupled-grid-root-next-stage-manifest.v1"
NEXT_STATE_SCHEMA = "feebl.coupled-grid-root-next-stage-state.v1"
FINAL_ANALYSIS_SCHEMA = "feebl.coupled-grid-root-final-analysis.v1"
GENERATOR_ID = replay.GENERATOR_ID
NORMALIZATION_ID = replay.NORMALIZATION_ID
MINIMUM_PILOT_SEEDS = 4
NEXT_STAGE_ACTIONS = {
    "grid",
    "root",
    "joint",
    "replicate",
    "grid_replicate",
    "root_replicate",
    "joint_replicate",
    "hold",
    "diagnostic_grid",
    "diagnostic_root",
    "diagnostic_joint",
    "insufficient_preregistered_seeds",
}
REPLICATION_ACTIONS = {
    "replicate",
    "grid_replicate",
    "root_replicate",
    "joint_replicate",
}
INT32_MAX = replay.INT32_MAX
NUMBER_PATTERN = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
UNSIGNED_NUMBER_PATTERN = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
SUPPORTED_ENDPOINT_FORMATS = {"assignment", "hash_comment", "json_scalar"}
GAUSSIAN_CORE_TRANSFORM_ID = "feebl_gaussian_core_spectrum_v1"
GAUSSIAN_CORE_SCHEMA = "feebl.gaussian-core-spectrum.v1"
GAUSSIAN_CORE_INPUT = "lumi.ee.out.dLdW.fne"
GAUSSIAN_CORE_JSON = "gaussian_core.json"
GAUSSIAN_CORE_CSV = "gaussian_core.csv"
FROZEN_INPUT_ROOT = Path("frozen_inputs")
FROZEN_REQUEST_PATH = FROZEN_INPUT_ROOT / "request" / "request.json"
FROZEN_SOURCE_STATE_PATH = FROZEN_INPUT_ROOT / "source" / "source_state.json"
FROZEN_DRIVER_PATH = FROZEN_INPUT_ROOT / "scripts" / "campaign" / "coupled_grid_root_campaign.py"
FROZEN_REPLAY_HELPER_PATH = FROZEN_INPUT_ROOT / "scripts" / "replay" / "adaptive_root_replay.py"
FROZEN_BINARY_PATH = FROZEN_INPUT_ROOT / "runtime" / "feebl"
FROZEN_PYTHON_PATH = FROZEN_INPUT_ROOT / "runtime" / "python"
FROZEN_GAUSSIAN_PATH = FROZEN_INPUT_ROOT / "transforms" / "gaussian_core_spectrum.py"
GAUSSIAN_CORE_ENDPOINT_KEYS = {
    "core_fit.mu_core_GeV",
    "core_fit.sigma_core_GeV",
    "core_fit.gaussian_core_area",
    "core_fit.peak_dL_dW",
    "core_fit.mu_fit_uncertainty_GeV_diagnostic",
    "core_fit.sigma_fit_uncertainty_GeV_diagnostic",
    "stability.maximum_window_mu_shift_GeV",
    "stability.maximum_window_sigma_relative_shift",
    "stability.pair_rebin_mu_shift_GeV",
    "stability.pair_rebin_sigma_relative_shift",
}
BASE_GRID_ONLY_CONTRACT = "base_grid_only_v1"
COUPLED_ADAPTIVE_MAXIMA_CONTRACT = "base_grid_and_adaptive_maxima_v1"
GRID_PATCH_CONTRACTS = {
    BASE_GRID_ONLY_CONTRACT,
    COUPLED_ADAPTIVE_MAXIMA_CONTRACT,
}
GRID_CERTIFICATE_PREFIX = "FEEBL_ADAPTIVE_GRID_CERTIFICATE_V1 "
GRID_CERTIFICATE_SCHEMA = "feebl.adaptive-grid-certificate.v1"
GRID_CERTIFICATE_FIELDS = (
    "schema",
    "requested_mode",
    "actual_mode",
    "error_norm_policy",
    "source_model",
    "error_target",
    "error_quantile",
    "pilot_diagnostics",
    "fail_on_unmet_error",
    "encounters",
    "certified_encounters",
    "unmet_encounters",
    "noise_stop_encounters",
    "resource_unmet_encounters",
    "vector_catastrophe_failure_encounters",
    "maximum_vector_difference",
    "catastrophe_limit",
    "terminal_patch_enabled",
    "terminal_patch_attempts",
    "terminal_patch_accepts",
    "terminal_patch_rejects",
    "terminal_patch_rejection_mask_or",
    "external_replay_recommended",
    "all_encounters_certified",
    "disposition",
)


class CampaignError(RuntimeError):
    """A fail-closed campaign contract violation."""


def required_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError(f"{where}: expected an object")
    return value


def required_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise CampaignError(f"{where}: expected an array")
    return value


def checked_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise CampaignError(f"{where}: invalid identifier")
    return value


def checked_int(value: Any, where: str, low: int, high: int = INT32_MAX) -> int:
    try:
        return replay.checked_int(value, where, low, high)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def decimal_value(value: Any, where: str, *, positive: bool = False) -> Decimal:
    try:
        return replay.decimal_value(value, where, positive=positive)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def decimal_text(value: Decimal) -> str:
    return replay.decimal_text(value)


def _one_literal_decimal_card_value(text: str, block: replay.Block, key: str) -> Decimal:
    """Read one finite literal decimal from the selected PARAMETERS block.

    Qualification settings are deliberately stricter than FeeBL's general
    expression language: the campaign must be able to bind the value in the
    final runtime certificate to an unambiguous frozen scalar in the card.
    """
    try:
        found = [item for item in replay.raw_assignments(text, block) if item.key == key]
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc
    if len(found) != 1 or re.fullmatch(NUMBER_PATTERN, found[0].value) is None:
        raise CampaignError(
            f"selected {block.kind}::{block.name} must contain exactly one "
            f"literal {key}= decimal assignment"
        )
    return decimal_value(found[0].value, f"card.{key}")


def _one_literal_string_card_value(text: str, block: replay.Block, key: str) -> str:
    """Read one simple quoted identifier from the selected PARAMETERS block."""
    try:
        found = [item for item in replay.raw_assignments(text, block) if item.key == key]
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc
    if len(found) != 1:
        raise CampaignError(
            f"selected {block.kind}::{block.name} must contain exactly one "
            f"simple quoted {key}= string assignment"
        )
    match = re.fullmatch(r'"([A-Za-z][A-Za-z0-9_.-]{0,63})"', found[0].value)
    if match is None:
        raise CampaignError(
            f"selected {block.kind}::{block.name} must contain exactly one "
            f"simple quoted {key}= string assignment"
        )
    return match.group(1)


def _one_literal_csv_integer_card_value(
    text: str,
    block: replay.Block,
    key: str,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    """Read one canonical quoted, strictly increasing integer CSV.

    FeeBL accepts a somewhat wider whitespace-tolerant grammar.  Frozen
    campaigns deliberately use a canonical subset so the class bank has one
    byte representation and can be checked before any simulation process is
    started.
    """
    try:
        found = [item for item in replay.raw_assignments(text, block) if item.key == key]
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc
    if len(found) != 1:
        raise CampaignError(
            f"selected {block.kind}::{block.name} must contain exactly one "
            f"literal {key}= quoted integer-list assignment"
        )
    raw_value = found[0].value
    if raw_value == '""':
        if allow_empty:
            return ()
        raise CampaignError(f"card.{key} must not be empty")
    match = re.fullmatch(r'"([0-9]+(?:,[0-9]+)*)"', raw_value)
    if match is None:
        raise CampaignError(f"card.{key} must be a canonical quoted comma-separated integer list")
    tokens = match.group(1).split(",")
    values = tuple(int(token) for token in tokens)
    if any(token != str(value) for token, value in zip(tokens, values)):
        raise CampaignError(f"card.{key} entries must use canonical decimal spelling")
    if any(value <= 0 for value in values):
        raise CampaignError(f"card.{key} entries must be positive")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise CampaignError(f"card.{key} entries must be unique and strictly increasing")
    return values


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Evaluate I_x(a,b) with a fail-closed continued fraction.

    This standard-library implementation is used only to obtain small-sample
    Student-t critical values.  It follows the stable symmetry transform and
    modified-Lentz continued fraction from the usual Numerical Recipes
    formulation; invalid or nonconvergent inputs are campaign errors.
    """
    if not (math.isfinite(x) and math.isfinite(a) and math.isfinite(b)):
        raise CampaignError("non-finite incomplete-beta input")
    if not (0.0 <= x <= 1.0 and a > 0.0 and b > 0.0):
        raise CampaignError("invalid incomplete-beta input")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    def continued_fraction(aa: float, bb: float, xx: float) -> float:
        qab = aa + bb
        qap = aa + 1.0
        qam = aa - 1.0
        tiny = 1.0e-300
        c = 1.0
        d = 1.0 - qab * xx / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        result = d
        for m in range(1, 10001):
            m2 = 2 * m
            term = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + term * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + term / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            result *= d * c
            term = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + term * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + term / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            result *= delta
            if abs(delta - 1.0) <= 3.0e-14:
                if not math.isfinite(result) or result <= 0.0:
                    raise CampaignError("invalid incomplete-beta fraction")
                return result
        raise CampaignError("incomplete-beta continued fraction did not converge")

    log_prefactor = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    prefactor = math.exp(log_prefactor)
    if x < (a + 1.0) / (a + b + 2.0):
        value = prefactor * continued_fraction(a, b, x) / a
    else:
        value = 1.0 - prefactor * continued_fraction(b, a, 1.0 - x) / b
    if not math.isfinite(value) or value < -1.0e-13 or value > 1.0 + 1.0e-13:
        raise CampaignError("regularized incomplete-beta result is invalid")
    return min(1.0, max(0.0, value))


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1 or not math.isfinite(value):
        raise CampaignError("invalid Student-t CDF input")
    if value == 0.0:
        return 0.5
    nu = float(degrees_of_freedom)
    beta = _regularized_incomplete_beta(nu / (nu + value * value), nu / 2.0, 0.5)
    return 1.0 - 0.5 * beta if value > 0.0 else 0.5 * beta


def _student_t_quantile(probability: Decimal, degrees_of_freedom: int) -> Decimal:
    """Positive Student-t quantile for a probability strictly above one half."""
    p = float(probability)
    if not (0.5 < p < 1.0) or degrees_of_freedom < 1:
        raise CampaignError("invalid Student-t quantile input")
    low = 0.0
    high = 1.0
    while _student_t_cdf(high, degrees_of_freedom) < p:
        high *= 2.0
        if high > 1.0e12:
            raise CampaignError("Student-t quantile could not be bracketed")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _student_t_cdf(middle, degrees_of_freedom) < p:
            low = middle
        else:
            high = middle
    result = 0.5 * (low + high)
    if not math.isfinite(result) or result <= 0.0:
        raise CampaignError("Student-t quantile is invalid")
    return Decimal(str(result))


def checked_hash(value: Any, where: str) -> str:
    try:
        return replay.checked_hash(value, where)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def checked_file(path: Path, expected: Any, where: str, *, nonempty: bool = True) -> str:
    try:
        return replay.checked_file(path, expected, where, nonempty=nonempty)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def resolve_path(base: Path, value: Any, where: str) -> Path:
    try:
        return replay.resolve_path(base, value, where)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def relative_path(value: Any, where: str, *, leaf: bool = False) -> Path:
    try:
        return replay.checked_relative_path(value, where, leaf=leaf)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def resolve_within(base: Path, value: Path, where: str) -> Path:
    try:
        return replay.resolve_within(base, value, where)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def load_json(path: Path) -> Any:
    try:
        return replay.load_json(path)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def write_canonical(path: Path, value: Any) -> None:
    try:
        replay.write_canonical(path, value)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def with_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return replay.with_self_hash(value)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def validate_self_hash(path: Path, schema: str) -> dict[str, Any]:
    try:
        return replay.validate_self_hash(load_json(path), schema, path)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def sha256_file(path: Path) -> str:
    return replay.sha256_file(path)


def sha256_object(value: Any) -> str:
    try:
        return replay.sha256_bytes(replay.canonical_bytes(value))
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc


def _frozen_template_path(index: int, template_id: str) -> Path:
    return FROZEN_INPUT_ROOT / "templates" / f"{index:03d}_{template_id}.dat"


def _snapshot_record(relative: Path, sha256: str) -> dict[str, str]:
    return {"path": relative.as_posix(), "sha256": checked_hash(sha256, "snapshot")}


def _copy_exact_snapshot(
    source: Path,
    output_dir: Path,
    relative: Path,
    where: str,
    *,
    executable: bool = False,
) -> dict[str, str]:
    """Copy one authenticated input into a new campaign without following it later."""

    target = resolve_within(output_dir, relative, where)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise CampaignError(f"{where}: frozen snapshot target already exists")
    try:
        with source.open("rb") as source_handle, target.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        source_mode = stat.S_IMODE(source.stat().st_mode)
        if executable:
            source_mode |= stat.S_IXUSR
        os.chmod(target, source_mode)
    except OSError as exc:
        raise CampaignError(f"{where}: cannot create exact frozen snapshot: {exc}") from exc
    source_hash = sha256_file(source)
    target_hash = sha256_file(target)
    if target_hash != source_hash:
        raise CampaignError(f"{where}: snapshot bytes differ from their source")
    return _snapshot_record(relative, target_hash)


def _bound_snapshot_path(
    output_dir: Path,
    raw: Any,
    expected_relative: Path,
    where: str,
) -> Path:
    """Authenticate one canonical, campaign-local regular-file snapshot."""

    binding = required_mapping(raw, where)
    if set(binding) != {"path", "sha256"}:
        raise CampaignError(f"{where}: expected exactly path and sha256")
    relative = relative_path(binding.get("path"), f"{where}.path")
    if relative != expected_relative:
        raise CampaignError(
            f"{where}.path must be the canonical snapshot path " f"{expected_relative.as_posix()!r}"
        )
    # resolve_within catches traversal and symlink escape.  Reject in-tree
    # symlinks too: otherwise later mutation of their target would defeat the
    # campaign-local immutable-input contract.
    cursor = output_dir.resolve()
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise CampaignError(f"{where}: frozen snapshot path contains a symlink")
    path = resolve_within(output_dir, relative, where)
    checked_file(path, binding.get("sha256"), where)
    return path


def _replace_assignments(
    raw: bytes,
    accelerator: str,
    parameters: str,
    replacements: Mapping[str, str],
) -> bytes:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CampaignError("FeeBL campaign template must be UTF-8") from exc
    try:
        replay.selected_block(text, "ACCELERATOR", accelerator)
        block = replay.selected_block(text, "PARAMETERS", parameters)
        targets = [
            (replay.one_assignment(text, block, key), replacement)
            for key, replacement in replacements.items()
        ]
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc
    for assignment, replacement in sorted(
        targets, key=lambda item: item[0].value_start, reverse=True
    ):
        text = text[: assignment.value_start] + replacement + text[assignment.value_end :]
    return text.encode("utf-8")


def patch_card(
    raw: bytes,
    accelerator: str,
    parameters: str,
    n_x: int,
    n_y: int,
    n_m: int,
    seed: int,
    grid_patch_contract: str = BASE_GRID_ONLY_CONTRACT,
) -> bytes:
    if grid_patch_contract not in GRID_PATCH_CONTRACTS:
        raise CampaignError(f"unsupported grid patch contract {grid_patch_contract!r}")
    replacements = {
        "n_x": str(n_x),
        "n_y": str(n_y),
        "n_m": str(n_m),
        "rndm_seed": str(seed),
    }
    if grid_patch_contract == COUPLED_ADAPTIVE_MAXIMA_CONTRACT:
        replacements.update({"adaptive_grid_max_nx": str(n_x), "adaptive_grid_max_ny": str(n_y)})
    return _replace_assignments(
        raw,
        accelerator,
        parameters,
        replacements,
    )


def masked_card(
    raw: bytes,
    accelerator: str,
    parameters: str,
    grid_patch_contract: str = BASE_GRID_ONLY_CONTRACT,
) -> bytes:
    if grid_patch_contract not in GRID_PATCH_CONTRACTS:
        raise CampaignError(f"unsupported grid patch contract {grid_patch_contract!r}")
    replacements = {
        "n_x": "<FEEBL_CAMPAIGN_N_X>",
        "n_y": "<FEEBL_CAMPAIGN_N_Y>",
        "n_m": "<FEEBL_CAMPAIGN_N_M>",
        "rndm_seed": "<FEEBL_CAMPAIGN_RNDM_SEED>",
    }
    if grid_patch_contract == COUPLED_ADAPTIVE_MAXIMA_CONTRACT:
        replacements.update(
            {
                "adaptive_grid_max_nx": "<FEEBL_CAMPAIGN_N_X>",
                "adaptive_grid_max_ny": "<FEEBL_CAMPAIGN_N_Y>",
            }
        )
    return _replace_assignments(
        raw,
        accelerator,
        parameters,
        replacements,
    )


def inspect_campaign_card(
    raw: bytes,
    accelerator: str,
    parameters: str,
    grid_patch_contract: str = BASE_GRID_ONLY_CONTRACT,
) -> dict[str, Any]:
    if grid_patch_contract not in GRID_PATCH_CONTRACTS:
        raise CampaignError(f"unsupported grid patch contract {grid_patch_contract!r}")
    try:
        base = replay.inspect_card(raw, accelerator, parameters)
        text = raw.decode("utf-8")
        block = replay.selected_block(text, "PARAMETERS", parameters)
        if replay.assignments(text, block, "n_x.1") or replay.assignments(text, block, "n_x.2"):
            raise CampaignError("beam-specific n_x overrides are unsupported")
        if replay.assignments(text, block, "n_y.1") or replay.assignments(text, block, "n_y.2"):
            raise CampaignError("beam-specific n_y overrides are unsupported")
        n_x = replay.one_assignment(text, block, "n_x").value
        n_y = replay.one_assignment(text, block, "n_y").value
        spectrum_w_min = _one_literal_decimal_card_value(text, block, "spectrum_w_min")
        spectrum_w_max = _one_literal_decimal_card_value(text, block, "spectrum_w_max")
        spectrum_w_bins = replay.one_assignment(text, block, "spectrum_w_bins").value
        if not spectrum_w_max > spectrum_w_min:
            raise CampaignError("spectrum_w_max must exceed spectrum_w_min")
        if not (26 <= spectrum_w_bins <= 10_000_000):
            raise CampaignError("spectrum_w_bins must be in [26,10000000]")
        adaptive_mode_assignments = replay.assignments(text, block, "adaptive_macroparticle_mode")
        if len(adaptive_mode_assignments) > 1:
            raise CampaignError("adaptive_macroparticle_mode must occur at most once")
        adaptive_mode = adaptive_mode_assignments[0].value if adaptive_mode_assignments else 0
        if adaptive_mode not in (0, 2, 3):
            raise CampaignError(
                "campaign adaptive_macroparticle_mode must be 0, 2, or 3; "
                "diagnostic Mode 1 has no runtime replay handoff"
            )
        adaptive_grid_mode = replay.optional_int(text, block, "adaptive_grid_mode", 0)
        if adaptive_grid_mode not in (0, 2, 3):
            raise CampaignError(
                "campaign adaptive_grid_mode must be 0, 2, or 3; "
                "diagnostic Mode 1 cannot provide a qualification certificate"
            )
        adaptive_grid_pilot_diagnostics = replay.optional_int(
            text, block, "adaptive_grid_pilot_diagnostics", 0
        )
        adaptive_grid_fail_on_unmet_error = replay.optional_int(
            text, block, "adaptive_grid_fail_on_unmet_error", 1
        )
        adaptive_grid_terminal_patch_certification = replay.optional_int(
            text, block, "adaptive_grid_terminal_patch_certification", 0
        )
        if adaptive_grid_mode in (2, 3):
            adaptive_grid_error_norm_policy = _one_literal_string_card_value(
                text, block, "adaptive_grid_error_norm_policy"
            )
            adaptive_grid_source_model = _one_literal_string_card_value(
                text, block, "adaptive_grid_source_model"
            )
            adaptive_grid_field_spacing_classes = _one_literal_csv_integer_card_value(
                text, block, "adaptive_grid_field_spacing_classes"
            )
            if (
                len(adaptive_grid_field_spacing_classes) < 3
                or adaptive_grid_field_spacing_classes[0] != 1
                or any(value & (value - 1) for value in adaptive_grid_field_spacing_classes)
                or any(
                    right != 2 * left
                    for left, right in zip(
                        adaptive_grid_field_spacing_classes,
                        adaptive_grid_field_spacing_classes[1:],
                    )
                )
            ):
                raise CampaignError(
                    "adaptive_grid_field_spacing_classes must begin at 1 and "
                    "contain at least three consecutive ratio-two powers of two"
                )
            adaptive_grid_error_target = _one_literal_decimal_card_value(
                text, block, "adaptive_grid_error_target"
            )
            adaptive_grid_error_quantile = _one_literal_decimal_card_value(
                text, block, "adaptive_grid_error_quantile"
            )
            adaptive_grid_error_catastrophe_limit = _one_literal_decimal_card_value(
                text, block, "adaptive_grid_error_catastrophe_limit"
            )
            if not (Decimal(0) < adaptive_grid_error_target < Decimal(1)):
                raise CampaignError("adaptive_grid_error_target must be in (0,1)")
            if not (Decimal("0.95") <= adaptive_grid_error_quantile < Decimal(1)):
                raise CampaignError("adaptive_grid_error_quantile must be in [0.95,1)")
            if not (Decimal(0) < adaptive_grid_error_catastrophe_limit <= Decimal(2)):
                raise CampaignError("adaptive_grid_error_catastrophe_limit must be in (0,2]")
            if adaptive_grid_source_model == "reference_cic_v1":
                adaptive_grid_source_reference_nx = replay.one_assignment(
                    text, block, "adaptive_grid_source_reference_nx"
                ).value
                adaptive_grid_source_reference_ny = replay.one_assignment(
                    text, block, "adaptive_grid_source_reference_ny"
                ).value
                if (
                    adaptive_grid_source_reference_nx <= 0
                    or adaptive_grid_source_reference_ny <= 0
                    or adaptive_grid_source_reference_nx % n_x != 0
                    or adaptive_grid_source_reference_ny % n_y != 0
                ):
                    raise CampaignError(
                        "reference_cic_v1 source dimensions must be positive "
                        "integer multiples of the campaign grid"
                    )
                factor_x = adaptive_grid_source_reference_nx // n_x
                factor_y = adaptive_grid_source_reference_ny // n_y
                if factor_x != factor_y or factor_x <= 0 or factor_x & (factor_x - 1):
                    raise CampaignError(
                        "reference_cic_v1 requires equal power-of-two source "
                        "factors on both campaign axes"
                    )
                if factor_x not in adaptive_grid_field_spacing_classes:
                    raise CampaignError(
                        "reference_cic_v1 field classes must contain the "
                        "source-factor identity level"
                    )
                if any(
                    (spacing < factor_x and factor_x % spacing != 0)
                    or (spacing > factor_x and spacing % factor_x != 0)
                    for spacing in adaptive_grid_field_spacing_classes
                ):
                    raise CampaignError(
                        "reference_cic_v1 field classes are not registered "
                        "with the source lattice"
                    )
            else:
                adaptive_grid_source_reference_nx = None
                adaptive_grid_source_reference_ny = None
        else:
            adaptive_grid_error_norm_policy = None
            adaptive_grid_source_model = None
            adaptive_grid_field_spacing_classes = None
            adaptive_grid_source_reference_nx = None
            adaptive_grid_source_reference_ny = None
            adaptive_grid_error_target = None
            adaptive_grid_error_quantile = None
            adaptive_grid_error_catastrophe_limit = None
        for key, value in (
            ("adaptive_grid_pilot_diagnostics", adaptive_grid_pilot_diagnostics),
            ("adaptive_grid_fail_on_unmet_error", adaptive_grid_fail_on_unmet_error),
            (
                "adaptive_grid_terminal_patch_certification",
                adaptive_grid_terminal_patch_certification,
            ),
        ):
            if value not in (0, 1):
                raise CampaignError(f"{key} must be 0 or 1")
        if adaptive_grid_mode == 0 and adaptive_grid_terminal_patch_certification:
            raise CampaignError("terminal patch certification requires adaptive_grid_mode=3")
        if adaptive_grid_terminal_patch_certification and adaptive_grid_mode != 3:
            raise CampaignError("adaptive_grid_terminal_patch_certification=1 requires Mode 3")
        if adaptive_grid_mode in (2, 3) and not adaptive_grid_pilot_diagnostics:
            raise CampaignError(
                "qualification campaign Mode 2/3 requires " "adaptive_grid_pilot_diagnostics=1"
            )
        max_nx_assignments = replay.assignments(text, block, "adaptive_grid_max_nx")
        max_ny_assignments = replay.assignments(text, block, "adaptive_grid_max_ny")
        if grid_patch_contract == BASE_GRID_ONLY_CONTRACT:
            if max_nx_assignments or max_ny_assignments:
                raise CampaignError(
                    "base_grid_only_v1 forbids adaptive_grid_max_nx/max_ny; "
                    "declare the coupled-maxima contract instead"
                )
            max_nx = None
            max_ny = None
        else:
            max_nx = replay.one_assignment(text, block, "adaptive_grid_max_nx").value
            max_ny = replay.one_assignment(text, block, "adaptive_grid_max_ny").value
            if max_nx != n_x or max_ny != n_y:
                raise CampaignError("coupled adaptive_grid_max_nx/max_ny must equal n_x/n_y")
        raw_parameter_assignments = {
            item.key: item.value for item in replay.raw_assignments(text, block)
        }
        has_size_classes_x = "adaptive_grid_size_classes_x" in raw_parameter_assignments
        has_size_classes_y = "adaptive_grid_size_classes_y" in raw_parameter_assignments
        if has_size_classes_x != has_size_classes_y:
            raise CampaignError(
                "adaptive_grid_size_classes_x/y must either both be absent or both be present"
            )
        if has_size_classes_x:
            adaptive_grid_size_classes_x = _one_literal_csv_integer_card_value(
                text, block, "adaptive_grid_size_classes_x", allow_empty=True
            )
            adaptive_grid_size_classes_y = _one_literal_csv_integer_card_value(
                text, block, "adaptive_grid_size_classes_y", allow_empty=True
            )
        else:
            adaptive_grid_size_classes_x = ()
            adaptive_grid_size_classes_y = ()
        if adaptive_grid_mode in (2, 3):
            min_nx = replay.optional_int(text, block, "adaptive_grid_min_nx", 0)
            min_ny = replay.optional_int(text, block, "adaptive_grid_min_ny", 0)
            effective_min_nx = 4 if min_nx == 0 else min_nx
            effective_min_ny = 4 if min_ny == 0 else min_ny
            effective_max_nx = n_x if max_nx is None else max_nx
            effective_max_ny = n_y if max_ny is None else max_ny
            for axis, classes, minimum, maximum in (
                (
                    "x",
                    adaptive_grid_size_classes_x,
                    effective_min_nx,
                    effective_max_nx,
                ),
                (
                    "y",
                    adaptive_grid_size_classes_y,
                    effective_min_ny,
                    effective_max_ny,
                ),
            ):
                if minimum < 4 or minimum > maximum:
                    raise CampaignError(f"adaptive grid {axis} minimum/maximum are inconsistent")
                if any(value < minimum or value > maximum or value % 2 != 0 for value in classes):
                    raise CampaignError(
                        f"adaptive_grid_size_classes_{axis} entries must be even "
                        f"and lie inside [{minimum},{maximum}] for this campaign grid"
                    )
    except (replay.ReplayError, UnicodeDecodeError) as exc:
        raise CampaignError(str(exc)) from exc
    if n_x <= 0 or n_y <= 0:
        raise CampaignError("template n_x and n_y must be positive integers")
    base.update(
        {
            "n_x": n_x,
            "n_y": n_y,
            "spectrum_w_min": decimal_text(spectrum_w_min),
            "spectrum_w_max": decimal_text(spectrum_w_max),
            "spectrum_w_bins": spectrum_w_bins,
            "adaptive_grid_max_nx": max_nx,
            "adaptive_grid_max_ny": max_ny,
            "adaptive_grid_mode": adaptive_grid_mode,
            "adaptive_grid_pilot_diagnostics": adaptive_grid_pilot_diagnostics,
            "adaptive_grid_fail_on_unmet_error": adaptive_grid_fail_on_unmet_error,
            "adaptive_grid_terminal_patch_certification": (
                adaptive_grid_terminal_patch_certification
            ),
            "adaptive_grid_error_norm_policy": adaptive_grid_error_norm_policy,
            "adaptive_grid_source_model": adaptive_grid_source_model,
            "adaptive_grid_size_classes_x": list(adaptive_grid_size_classes_x),
            "adaptive_grid_size_classes_y": list(adaptive_grid_size_classes_y),
            "adaptive_grid_field_spacing_classes": (
                None
                if adaptive_grid_field_spacing_classes is None
                else list(adaptive_grid_field_spacing_classes)
            ),
            "adaptive_grid_source_reference_nx": (adaptive_grid_source_reference_nx),
            "adaptive_grid_source_reference_ny": (adaptive_grid_source_reference_ny),
            "adaptive_grid_error_target": (
                None
                if adaptive_grid_error_target is None
                else decimal_text(adaptive_grid_error_target)
            ),
            "adaptive_grid_error_quantile": (
                None
                if adaptive_grid_error_quantile is None
                else decimal_text(adaptive_grid_error_quantile)
            ),
            "adaptive_grid_error_catastrophe_limit": (
                None
                if adaptive_grid_error_catastrophe_limit is None
                else decimal_text(adaptive_grid_error_catastrophe_limit)
            ),
            "adaptive_macroparticle_mode": adaptive_mode,
            "grid_patch_contract": grid_patch_contract,
            "campaign_family_sha256": replay.sha256_bytes(
                masked_card(raw, accelerator, parameters, grid_patch_contract)
            ),
        }
    )
    return base


def parse_endpoint(run_directory: Path, spec: Mapping[str, Any]) -> tuple[str, str]:
    endpoint_id = checked_id(spec.get("id"), "endpoint.id")
    filename = relative_path(spec.get("file"), f"endpoint[{endpoint_id}].file", leaf=True)
    path = resolve_within(run_directory, filename, f"endpoint[{endpoint_id}].file")
    if not path.is_file() or path.stat().st_size == 0:
        raise CampaignError(f"endpoint {endpoint_id}: missing or empty output file {filename}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CampaignError(f"endpoint {endpoint_id}: cannot read {filename}: {exc}") from exc
    key = spec.get("key")
    endpoint_format = spec.get("format")
    if endpoint_format == "json_scalar":
        if filename.as_posix() != GAUSSIAN_CORE_JSON:
            raise CampaignError(
                f"endpoint {endpoint_id}: json_scalar is restricted to {GAUSSIAN_CORE_JSON}"
            )
        if key not in GAUSSIAN_CORE_ENDPOINT_KEYS:
            raise CampaignError(f"endpoint {endpoint_id}: unsupported Gaussian-core key")
        document = required_mapping(load_json(path), f"endpoint[{endpoint_id}].json")
        try:
            canonical = replay.canonical_bytes(document) + b"\n"
        except replay.ReplayError as exc:
            raise CampaignError(str(exc)) from exc
        if path.read_bytes() != canonical:
            raise CampaignError(f"endpoint {endpoint_id}: Gaussian-core JSON is not canonical")
        if document.get("schema") != GAUSSIAN_CORE_SCHEMA or document.get("schema_version") != "1":
            raise CampaignError(f"endpoint {endpoint_id}: Gaussian-core schema mismatch")
        if document.get("status") != "accepted_core_fit_not_physics_qualification":
            raise CampaignError(f"endpoint {endpoint_id}: Gaussian-core fit was not accepted")
        value: Any = document
        for part in str(key).split("."):
            value = required_mapping(value, f"endpoint[{endpoint_id}].{part}").get(part)
        if not isinstance(value, str):
            raise CampaignError(
                f"endpoint {endpoint_id}: JSON scalar must be an exact decimal string"
            )
        parsed = decimal_value(value, f"endpoint[{endpoint_id}].value")
        return decimal_text(parsed), sha256_file(path)
    if not isinstance(key, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key) is None:
        raise CampaignError(f"endpoint {endpoint_id}: invalid extraction key")
    if endpoint_format == "assignment":
        pattern = re.compile(
            rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*({NUMBER_PATTERN})(?=[ \t;]|$)"
        )
    elif endpoint_format == "hash_comment":
        # FeeBL spectrum summaries use ``#key value +- uncertainty``. Accept
        # exactly that optional numeric suffix, while refusing arbitrary text
        # after the preregistered leading scalar.
        pattern = re.compile(
            rf"(?m)^[ \t]*#{re.escape(key)}[ \t]+({NUMBER_PATTERN})"
            rf"(?:[ \t]+\+\-[ \t]*{UNSIGNED_NUMBER_PATTERN})?[ \t]*$"
        )
    else:
        raise CampaignError(f"endpoint {endpoint_id}: unsupported format {endpoint_format!r}")
    matches = pattern.findall(text)
    if not matches:
        raise CampaignError(f"endpoint {endpoint_id}: key {key!r} was not found in {filename}")
    values = [decimal_value(item, f"endpoint[{endpoint_id}].value") for item in matches]
    if any(item != values[0] for item in values[1:]):
        raise CampaignError(f"endpoint {endpoint_id}: ambiguous repeated values in {filename}")
    return decimal_text(values[0]), sha256_file(path)


def _adaptive_grid_certificate_evidence(
    log_path: Path,
    realization: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse and bind the final Mode-2/3 all-encounter certificate.

    Endpoint stability is not a substitute for the internal discretization
    proof.  Diagnostic runs may finish successfully with an unmet target, so
    execution records the certificate either way and analysis independently
    forbids a qualification pass unless its strict disposition is accepted.
    """
    if not log_path.is_file():
        raise CampaignError(f"adaptive-grid certificate log is missing: {log_path}")
    try:
        text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CampaignError(f"adaptive-grid certificate log is not readable UTF-8: {exc}") from exc
    template_id = str(realization.get("template_id"))
    try:
        template = next(item for item in config["templates"] if item["id"] == template_id)
    except StopIteration as exc:
        raise CampaignError(
            f"adaptive-grid certificate references unknown template {template_id!r}"
        ) from exc
    card_info = required_mapping(template.get("card_info"), f"template[{template_id}].card_info")
    adaptive_grid_mode = int(card_info["adaptive_grid_mode"])
    all_certificate_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("FEEBL_ADAPTIVE_GRID_CERTIFICATE_")
    ]
    current_lines = [
        line for line in all_certificate_lines if line.startswith(GRID_CERTIFICATE_PREFIX)
    ]
    if adaptive_grid_mode == 0:
        if all_certificate_lines:
            raise CampaignError(
                "fixed-grid template emitted an unexpected adaptive-grid certificate"
            )
        return {
            "required": False,
            "present": False,
            "qualification_passed": True,
            "combined_log_sha256": sha256_file(log_path),
        }
    if adaptive_grid_mode not in (2, 3):
        raise CampaignError("unsupported adaptive-grid certificate mode")
    if len(current_lines) != 1 or len(all_certificate_lines) != 1:
        raise CampaignError(
            "Mode-2/3 log must contain exactly one current adaptive-grid certificate"
        )
    tokens = current_lines[0][len(GRID_CERTIFICATE_PREFIX) :].split()
    if len(tokens) != len(GRID_CERTIFICATE_FIELDS):
        raise CampaignError("adaptive-grid certificate field count is noncanonical")
    parsed: dict[str, str] = {}
    ordered_keys: list[str] = []
    for token in tokens:
        if token.count("=") != 1:
            raise CampaignError("adaptive-grid certificate token is malformed")
        key, value = token.split("=", 1)
        if not key or not value or key in parsed:
            raise CampaignError("adaptive-grid certificate has an empty or duplicate field")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) is None:
            raise CampaignError("adaptive-grid certificate field name is invalid")
        parsed[key] = value
        ordered_keys.append(key)
    if tuple(ordered_keys) != GRID_CERTIFICATE_FIELDS:
        raise CampaignError("adaptive-grid certificate field order/schema is noncanonical")
    if parsed["schema"] != GRID_CERTIFICATE_SCHEMA:
        raise CampaignError("adaptive-grid certificate schema mismatch")
    integer_fields = {
        "requested_mode": (2, 3),
        "actual_mode": (2, 3),
        "pilot_diagnostics": (0, 1),
        "fail_on_unmet_error": (0, 1),
        "encounters": (1, INT32_MAX),
        "certified_encounters": (0, INT32_MAX),
        "unmet_encounters": (0, INT32_MAX),
        "noise_stop_encounters": (0, INT32_MAX),
        "resource_unmet_encounters": (0, INT32_MAX),
        "vector_catastrophe_failure_encounters": (0, INT32_MAX),
        "terminal_patch_enabled": (0, 1),
        "terminal_patch_attempts": (0, INT32_MAX),
        "terminal_patch_accepts": (0, INT32_MAX),
        "terminal_patch_rejects": (0, INT32_MAX),
        "terminal_patch_rejection_mask_or": (0, 2**32 - 1),
        "external_replay_recommended": (0, 1),
        "all_encounters_certified": (0, 1),
    }
    integers: dict[str, int] = {}
    for key, (low, high) in integer_fields.items():
        value = parsed[key]
        if re.fullmatch(r"[0-9]+", value) is None:
            raise CampaignError(f"adaptive-grid certificate {key} must be an integer")
        integers[key] = int(value)
        if integers[key] < low or integers[key] > high:
            raise CampaignError(f"adaptive-grid certificate {key} is out of range")
    error_target = decimal_value(parsed["error_target"], "grid_certificate.error_target")
    error_quantile = decimal_value(parsed["error_quantile"], "grid_certificate.error_quantile")
    maximum_vector_difference = decimal_value(
        parsed["maximum_vector_difference"],
        "grid_certificate.maximum_vector_difference",
    )
    catastrophe_limit = decimal_value(
        parsed["catastrophe_limit"], "grid_certificate.catastrophe_limit"
    )
    if not (Decimal(0) < error_target < Decimal(1)):
        raise CampaignError("adaptive-grid certificate error_target must be in (0,1)")
    if not (Decimal("0.95") <= error_quantile < Decimal(1)):
        raise CampaignError("adaptive-grid certificate error_quantile must be in [0.95,1)")
    if not (Decimal(0) <= maximum_vector_difference <= Decimal(2)):
        raise CampaignError("adaptive-grid certificate maximum vector difference must be in [0,2]")
    if not (Decimal(0) < catastrophe_limit <= Decimal(2)):
        raise CampaignError("adaptive-grid certificate catastrophe limit must be in (0,2]")
    for key in ("error_norm_policy", "source_model", "disposition"):
        if ID_PATTERN.fullmatch(parsed[key]) is None:
            raise CampaignError(f"adaptive-grid certificate {key} is invalid")
    expected_mode = adaptive_grid_mode
    expected_pilot = int(card_info["adaptive_grid_pilot_diagnostics"])
    expected_fail = int(card_info["adaptive_grid_fail_on_unmet_error"])
    expected_terminal = int(card_info["adaptive_grid_terminal_patch_certification"])
    if integers["requested_mode"] != expected_mode or integers["actual_mode"] != expected_mode:
        raise CampaignError("adaptive-grid certificate mode differs from the card")
    if integers["pilot_diagnostics"] != expected_pilot:
        raise CampaignError(
            "adaptive-grid certificate pilot-diagnostics flag differs from the card"
        )
    if integers["fail_on_unmet_error"] != expected_fail:
        raise CampaignError("adaptive-grid certificate fail-on-unmet flag differs from the card")
    if integers["terminal_patch_enabled"] != expected_terminal:
        raise CampaignError("adaptive-grid certificate terminal-patch flag differs from the card")
    if parsed["error_norm_policy"] != card_info["adaptive_grid_error_norm_policy"]:
        raise CampaignError("adaptive-grid certificate error-norm policy differs from the card")
    if parsed["source_model"] != card_info["adaptive_grid_source_model"]:
        raise CampaignError("adaptive-grid certificate source model differs from the card")
    if error_target != decimal_value(
        card_info["adaptive_grid_error_target"], "card_info.adaptive_grid_error_target"
    ):
        raise CampaignError("adaptive-grid certificate error target differs from the card")
    if error_quantile != decimal_value(
        card_info["adaptive_grid_error_quantile"],
        "card_info.adaptive_grid_error_quantile",
    ):
        raise CampaignError("adaptive-grid certificate error quantile differs from the card")
    if catastrophe_limit != decimal_value(
        card_info["adaptive_grid_error_catastrophe_limit"],
        "card_info.adaptive_grid_error_catastrophe_limit",
    ):
        raise CampaignError("adaptive-grid certificate catastrophe limit differs from the card")
    if integers["certified_encounters"] + integers["unmet_encounters"] != integers["encounters"]:
        raise CampaignError("adaptive-grid certificate encounter partition is inconsistent")
    if (
        integers["terminal_patch_accepts"] + integers["terminal_patch_rejects"]
        != integers["terminal_patch_attempts"]
    ):
        raise CampaignError("adaptive-grid certificate terminal-patch partition is inconsistent")
    if expected_terminal == 0 and any(
        integers[key] != 0
        for key in (
            "terminal_patch_attempts",
            "terminal_patch_accepts",
            "terminal_patch_rejects",
            "terminal_patch_rejection_mask_or",
        )
    ):
        raise CampaignError("disabled terminal patch has nonzero certificate counters")
    strict_pass = (
        integers["certified_encounters"] == integers["encounters"]
        and integers["unmet_encounters"] == 0
        and integers["noise_stop_encounters"] == 0
        and integers["resource_unmet_encounters"] == 0
        and integers["vector_catastrophe_failure_encounters"] == 0
        and maximum_vector_difference <= catastrophe_limit
        and integers["terminal_patch_attempts"] == integers["terminal_patch_accepts"]
        and integers["terminal_patch_rejects"] == 0
        and integers["terminal_patch_rejection_mask_or"] == 0
        and integers["external_replay_recommended"] == 0
    )
    declared_pass = integers["all_encounters_certified"] == 1
    if declared_pass != strict_pass:
        raise CampaignError("adaptive-grid all-encounters disposition is internally inconsistent")
    expected_disposition = "accepted_all_encounters" if strict_pass else "unqualified_diagnostics"
    if parsed["disposition"] != expected_disposition:
        raise CampaignError("adaptive-grid certificate disposition is inconsistent")
    terminal_patch_qualification_exercised = (
        expected_terminal == 0 or integers["terminal_patch_attempts"] > 0
    )
    canonical_fields = dict(sorted(parsed.items()))
    return {
        "required": True,
        "present": True,
        "qualification_passed": (strict_pass and terminal_patch_qualification_exercised),
        "certificate_internal_passed": strict_pass,
        "terminal_patch_qualification_exercised": (terminal_patch_qualification_exercised),
        "schema": GRID_CERTIFICATE_SCHEMA,
        "parsed_fields": canonical_fields,
        "parsed_fields_sha256": sha256_object(canonical_fields),
        "certificate_line_sha256": replay.sha256_bytes(current_lines[0].encode("utf-8")),
        "combined_log_sha256": sha256_file(log_path),
    }


def _unique_seeds(raw: Any, where: str, minimum: int) -> list[int]:
    values = required_list(raw, where)
    if len(values) < minimum:
        raise CampaignError(f"{where}: need at least {minimum} unique seeds")
    result = [checked_int(item, f"{where}[{index}]", 1) for index, item in enumerate(values)]
    if len(set(result)) != len(result):
        raise CampaignError(f"{where}: duplicate seed")
    return result


def _endpoint_specs(raw: Any) -> list[dict[str, Any]]:
    values = required_list(raw, "endpoints")
    if not values:
        raise CampaignError("endpoints must not be empty")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(values):
        spec = required_mapping(item, f"endpoints[{index}]")
        endpoint_id = checked_id(spec.get("id"), f"endpoints[{index}].id")
        if endpoint_id in seen:
            raise CampaignError(f"duplicate endpoint id {endpoint_id!r}")
        seen.add(endpoint_id)
        filename = relative_path(spec.get("file"), f"endpoints[{index}].file", leaf=True).as_posix()
        endpoint_format = spec.get("format")
        if endpoint_format not in SUPPORTED_ENDPOINT_FORMATS:
            raise CampaignError(
                f"endpoints[{index}].format must be assignment, hash_comment, or json_scalar"
            )
        key = spec.get("key")
        if endpoint_format == "json_scalar":
            if filename != GAUSSIAN_CORE_JSON:
                raise CampaignError(
                    f"endpoints[{index}]: json_scalar file must be {GAUSSIAN_CORE_JSON}"
                )
            if key not in GAUSSIAN_CORE_ENDPOINT_KEYS:
                raise CampaignError(f"endpoints[{index}]: unsupported Gaussian-core JSON key")
        elif not isinstance(key, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key) is None:
            raise CampaignError(f"endpoints[{index}].key is invalid")
        scale = decimal_value(spec.get("scale"), f"endpoints[{index}].scale", positive=True)
        target = decimal_value(
            spec.get("target_relative_error"),
            f"endpoints[{index}].target_relative_error",
            positive=True,
        )
        margin = decimal_value(
            spec.get("equivalence_margin_relative"),
            f"endpoints[{index}].equivalence_margin_relative",
            positive=True,
        )
        expected = decimal_value(
            spec.get("expected_true_difference_relative"),
            f"endpoints[{index}].expected_true_difference_relative",
        )
        if target >= 1 or margin > Decimal("0.01") or abs(expected) >= margin:
            raise CampaignError(
                f"endpoints[{index}]: require target < 1, equivalence margin <= 0.01, "
                "and |expected difference| < margin"
            )
        units = spec.get("units", "dimensionless")
        if not isinstance(units, str):
            raise CampaignError(f"endpoints[{index}].units must be a string")
        scale_contract_raw = spec.get("scale_contract")
        scale_contract: dict[str, str] | None = None
        if scale_contract_raw is not None:
            contract = required_mapping(scale_contract_raw, f"endpoints[{index}].scale_contract")
            reference_value = decimal_value(
                contract.get("reference_value"),
                f"endpoints[{index}].scale_contract.reference_value",
            )
            near_zero_floor = decimal_value(
                contract.get("near_zero_floor"),
                f"endpoints[{index}].scale_contract.near_zero_floor",
                positive=True,
            )
            scale_basis = decimal_value(
                contract.get("scale_basis"),
                f"endpoints[{index}].scale_contract.scale_basis",
                positive=True,
            )
            absolute_target = decimal_value(
                contract.get("absolute_target"),
                f"endpoints[{index}].scale_contract.absolute_target",
                positive=True,
            )
            provenance = contract.get("reference_value_provenance")
            if not isinstance(provenance, str) or not provenance.strip() or len(provenance) > 512:
                raise CampaignError(
                    f"endpoints[{index}].scale_contract.reference_value_provenance "
                    "must be a nonempty string of at most 512 characters"
                )
            floor_provenance = contract.get("near_zero_floor_provenance")
            if (
                not isinstance(floor_provenance, str)
                or not floor_provenance.strip()
                or len(floor_provenance) > 512
            ):
                raise CampaignError(
                    f"endpoints[{index}].scale_contract.near_zero_floor_provenance "
                    "must be a nonempty string of at most 512 characters"
                )
            expected_scale_basis = max(abs(reference_value), near_zero_floor)
            if scale_basis != expected_scale_basis:
                raise CampaignError(
                    f"endpoints[{index}].scale_contract.scale_basis must equal "
                    "max(abs(reference_value), near_zero_floor)"
                )
            if scale != scale_basis:
                raise CampaignError(
                    f"endpoints[{index}].scale must equal scale_contract.scale_basis"
                )
            if absolute_target != scale_basis * target:
                raise CampaignError(
                    f"endpoints[{index}].scale_contract.absolute_target must equal "
                    "scale_basis times target_relative_error"
                )
            scale_contract = {
                "reference_value": decimal_text(reference_value),
                "near_zero_floor": decimal_text(near_zero_floor),
                "scale_basis": decimal_text(scale_basis),
                "absolute_target": decimal_text(absolute_target),
                "reference_value_provenance": provenance.strip(),
                "near_zero_floor_provenance": floor_provenance.strip(),
                "scale_basis_formula": ("scale_basis=max(abs(reference_value),near_zero_floor)"),
                "absolute_target_formula": ("absolute_target=scale_basis*target_relative_error"),
            }
        result.append(
            {
                "id": endpoint_id,
                "file": filename,
                "format": endpoint_format,
                "key": key,
                "scale": decimal_text(scale),
                "target_relative_error": decimal_text(target),
                "equivalence_margin_relative": decimal_text(margin),
                "expected_true_difference_relative": decimal_text(expected),
                "units": units,
                "scale_contract": scale_contract,
            }
        )
    return result


def _grid_levels(raw: Any) -> list[dict[str, Any]]:
    values = required_list(raw, "grid_levels")
    if len(values) < 2:
        raise CampaignError("grid_levels must contain at least two levels")
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_shapes: set[tuple[int, int]] = set()
    for index, item in enumerate(values):
        level = required_mapping(item, f"grid_levels[{index}]")
        level_id = checked_id(level.get("id"), f"grid_levels[{index}].id")
        n_x = checked_int(level.get("n_x"), f"grid_levels[{index}].n_x", 1)
        n_y = checked_int(level.get("n_y"), f"grid_levels[{index}].n_y", 1)
        if level_id in seen_ids or (n_x, n_y) in seen_shapes:
            raise CampaignError("grid level ids and shapes must be unique")
        seen_ids.add(level_id)
        seen_shapes.add((n_x, n_y))
        result.append({"id": level_id, "n_x": n_x, "n_y": n_y})
    result.sort(key=lambda level: (level["n_x"] * level["n_y"], level["n_x"], level["n_y"]))
    for lower, upper in zip(result, result[1:]):
        if upper["n_x"] <= lower["n_x"] or upper["n_y"] <= lower["n_y"]:
            raise CampaignError("grid levels must increase strictly in both n_x and n_y")
    return result


def _n_m_levels(raw: Any) -> list[int]:
    values = required_list(raw, "n_m_levels")
    if len(values) < 2:
        raise CampaignError("n_m_levels must contain at least two values")
    result = sorted(checked_int(item, f"n_m_levels[{i}]", 1) for i, item in enumerate(values))
    if len(set(result)) != len(result):
        raise CampaignError("n_m_levels must be unique")
    return result


def _atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise CampaignError(f"refusing to overwrite campaign artifact: {path}")
    payload = replay.canonical_bytes(value)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise CampaignError(f"cannot atomically write {path}: {exc}") from exc


def _atomic_replace(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise CampaignError(f"stale state temporary blocks safe resume: {temporary}")
    payload = replay.canonical_bytes(value)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise CampaignError(f"cannot atomically update {path}: {exc}") from exc


def _configuration(request: Mapping[str, Any], base: Path) -> dict[str, Any]:
    """Validate a request and resolve every externally bound input."""
    if request.get("schema") != REQUEST_SCHEMA:
        raise CampaignError(f"request.schema must be {REQUEST_SCHEMA!r}")
    # This both rejects JSON floats and proves the request lies in the same
    # canonical-data subset as all emitted artifacts.
    try:
        replay.canonical_bytes(request)
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc

    raw_bindings = required_mapping(request.get("bindings"), "bindings")
    binary_path = resolve_path(
        base, raw_bindings.get("feebl_binary_path"), "bindings.feebl_binary_path"
    )
    binary_sha = checked_file(
        binary_path, raw_bindings.get("feebl_binary_sha256"), "bindings.feebl_binary"
    )
    source_path = resolve_path(
        base, raw_bindings.get("source_state_path"), "bindings.source_state_path"
    )
    source_sha = checked_file(
        source_path, raw_bindings.get("source_state_sha256"), "bindings.source_state"
    )
    generator = required_mapping(
        raw_bindings.get("generator_transform"), "bindings.generator_transform"
    )
    if generator.get("id") != GENERATOR_ID:
        raise CampaignError(f"bindings.generator_transform.id must be {GENERATOR_ID!r}")
    physics_family = required_mapping(raw_bindings.get("physics_family"), "bindings.physics_family")
    generator_sha = sha256_object(generator)
    physics_sha = sha256_object(physics_family)

    endpoints = _endpoint_specs(request.get("endpoints"))
    needs_gaussian_core = any(endpoint["format"] == "json_scalar" for endpoint in endpoints)
    raw_core_transform = raw_bindings.get("gaussian_core_transform")
    gaussian_core_transform: dict[str, Any] | None = None
    if needs_gaussian_core:
        transform = required_mapping(raw_core_transform, "bindings.gaussian_core_transform")
        if transform.get("id") != GAUSSIAN_CORE_TRANSFORM_ID:
            raise CampaignError(
                f"bindings.gaussian_core_transform.id must be {GAUSSIAN_CORE_TRANSFORM_ID!r}"
            )
        transform_path = resolve_path(
            base, transform.get("path"), "bindings.gaussian_core_transform.path"
        )
        transform_sha = checked_file(
            transform_path,
            transform.get("sha256"),
            "bindings.gaussian_core_transform",
        )
        python_path = resolve_path(
            base,
            transform.get("python_executable_path"),
            "bindings.gaussian_core_transform.python_executable_path",
        )
        python_sha = checked_file(
            python_path,
            transform.get("python_executable_sha256"),
            "bindings.gaussian_core_transform.python_executable",
        )
        gaussian_core_transform = {
            "id": GAUSSIAN_CORE_TRANSFORM_ID,
            "path": str(transform_path),
            "sha256": transform_sha,
            "python_executable_path": str(python_path),
            "python_executable_sha256": python_sha,
            "input": GAUSSIAN_CORE_INPUT,
            "json_output": GAUSSIAN_CORE_JSON,
            "csv_output": GAUSSIAN_CORE_CSV,
            "output_schema": GAUSSIAN_CORE_SCHEMA,
            "execution": "fresh_subprocess_shell_false_after_successful_feebl_v1",
        }
    elif raw_core_transform is not None:
        raise CampaignError(
            "bindings.gaussian_core_transform is forbidden when no json_scalar endpoint uses it"
        )
    grids = _grid_levels(request.get("grid_levels"))
    roots = _n_m_levels(request.get("n_m_levels"))
    pilot_seeds = _unique_seeds(request.get("pilot_seeds"), "pilot_seeds", MINIMUM_PILOT_SEEDS)
    extension_seeds = _unique_seeds(
        request.get("extension_seeds"), "extension_seeds", MINIMUM_PILOT_SEEDS
    )
    holdout_seeds = _unique_seeds(
        request.get("holdout_seeds"), "holdout_seeds", MINIMUM_PILOT_SEEDS
    )
    named_seed_sets = {
        "pilot": set(pilot_seeds),
        "extension": set(extension_seeds),
        "holdout": set(holdout_seeds),
    }
    for left, right in (("pilot", "extension"), ("pilot", "holdout"), ("extension", "holdout")):
        overlap = sorted(named_seed_sets[left].intersection(named_seed_sets[right]))
        if overlap:
            raise CampaignError(f"{left} and {right} seeds must be disjoint; overlap={overlap}")

    raw_templates = required_list(request.get("templates"), "templates")
    if not raw_templates:
        raise CampaignError("templates must not be empty")
    templates: list[dict[str, Any]] = []
    template_ids: set[str] = set()
    for index, raw_template in enumerate(raw_templates):
        where = f"templates[{index}]"
        template = required_mapping(raw_template, where)
        template_id = checked_id(template.get("id"), f"{where}.id")
        if template_id in template_ids:
            raise CampaignError(f"duplicate template id {template_id!r}")
        template_ids.add(template_id)
        accelerator = checked_id(template.get("accelerator"), f"{where}.accelerator")
        parameters = checked_id(template.get("parameters"), f"{where}.parameters")
        grid_patch_contract = template.get("grid_patch_contract")
        if grid_patch_contract not in GRID_PATCH_CONTRACTS:
            raise CampaignError(
                f"{where}.grid_patch_contract must be one of {sorted(GRID_PATCH_CONTRACTS)}"
            )
        path = resolve_path(base, template.get("path"), f"{where}.path")
        exact_sha = checked_file(path, template.get("sha256"), f"{where}.card")
        raw = path.read_bytes()
        info = inspect_campaign_card(raw, accelerator, parameters, grid_patch_contract)
        templates.append(
            {
                "id": template_id,
                "path": path,
                "path_text": str(path),
                "sha256": exact_sha,
                "family_sha256": info["campaign_family_sha256"],
                "accelerator": accelerator,
                "parameters": parameters,
                "grid_patch_contract": grid_patch_contract,
                "raw": raw,
                "card_info": info,
            }
        )
    reference_template_id = checked_id(
        request.get("reference_template_id"), "reference_template_id"
    )
    if reference_template_id not in template_ids:
        raise CampaignError("reference_template_id must name one frozen template")

    statistics = required_mapping(request.get("statistics"), "statistics")
    safety = decimal_value(
        statistics.get("standard_error_safety_factor"),
        "statistics.standard_error_safety_factor",
        positive=True,
    )
    confidence = decimal_value(
        statistics.get("confidence_level"), "statistics.confidence_level", positive=True
    )
    alpha = decimal_value(
        statistics.get("equivalence_alpha"), "statistics.equivalence_alpha", positive=True
    )
    power = decimal_value(
        statistics.get("equivalence_power"), "statistics.equivalence_power", positive=True
    )
    total_budget_raw = statistics.get("conservative_total_budget_multiplier")
    if not isinstance(total_budget_raw, str):
        raise CampaignError(
            "statistics.conservative_total_budget_multiplier must be an exact decimal string"
        )
    total_budget_multiplier = decimal_value(
        total_budget_raw,
        "statistics.conservative_total_budget_multiplier",
        positive=True,
    )
    if safety < 1 or safety > 10:
        raise CampaignError("statistics.standard_error_safety_factor must be in [1, 10]")
    if confidence <= Decimal("0.5") or confidence >= Decimal("0.999999"):
        raise CampaignError("statistics.confidence_level must be in (0.5, 0.999999)")
    if alpha >= Decimal("0.5"):
        raise CampaignError("statistics.equivalence_alpha must be in (0, 0.5)")
    if power <= Decimal("0.5") or power >= Decimal("0.999999"):
        raise CampaignError("statistics.equivalence_power must be in (0.5, 0.999999)")
    if total_budget_multiplier < 1 or total_budget_multiplier > 4:
        raise CampaignError("statistics.conservative_total_budget_multiplier must be in [1, 4]")
    z_conf = Decimal(str(NormalDist().inv_cdf(float((Decimal(1) + confidence) / 2))))
    z_alpha = Decimal(str(NormalDist().inv_cdf(float(Decimal(1) - alpha))))
    # Conservative closed-form normal planning for a symmetric TOST.  Using
    # z_power directly would deliver only 2*power-1 at the centered null in
    # the known-sigma limit.  Splitting beta across both rejection tails
    # guarantees at least the requested power (and is conservative away from
    # the center); the executed gate still uses familywise Student-t bounds.
    z_power = Decimal(str(NormalDist().inv_cdf(float((Decimal(1) + power) / Decimal(2)))))
    pilot_degrees_of_freedom = len(pilot_seeds) - 1
    numerical_family_statement_count = len(templates) * len(endpoints) * 4
    method_equivalence_statements = (
        max(0, len(templates) - 1) * len(grids) * len(roots) * len(endpoints)
    )
    numerical_equivalence_statements = len(templates) * len(endpoints) * 3
    equivalence_family_statement_count = max(
        1, method_equivalence_statements + numerical_equivalence_statements
    )
    with localcontext() as context:
        context.prec = 60
        confidence_tail = Decimal(1) - confidence
        family_confidence_probability = Decimal(1) - confidence_tail / (
            Decimal(2) * Decimal(numerical_family_statement_count)
        )
        family_equivalence_probability = Decimal(1) - alpha / Decimal(
            equivalence_family_statement_count
        )
    t_conf_family = _student_t_quantile(family_confidence_probability, pilot_degrees_of_freedom)
    t_alpha_family = _student_t_quantile(family_equivalence_probability, pilot_degrees_of_freedom)
    z_alpha_family = Decimal(str(NormalDist().inv_cdf(float(family_equivalence_probability))))

    limits = required_mapping(request.get("limits"), "limits")
    maximum_grid_cells = checked_int(
        limits.get("maximum_grid_cells"), "limits.maximum_grid_cells", 1, INT32_MAX
    )
    maximum_n_m = checked_int(limits.get("maximum_n_m"), "limits.maximum_n_m", 1, INT32_MAX)
    maximum_pilot_runs = checked_int(
        limits.get("maximum_pilot_runs"), "limits.maximum_pilot_runs", 1, 1_000_000
    )
    maximum_wall_seconds = checked_int(
        limits.get("maximum_wall_seconds_per_run"),
        "limits.maximum_wall_seconds_per_run",
        1,
        604_800,
    )
    run_count = len(templates) * len(grids) * len(roots) * len(pilot_seeds)
    if run_count > maximum_pilot_runs:
        raise CampaignError(
            f"full factorial requires {run_count} pilot runs, exceeding maximum_pilot_runs={maximum_pilot_runs}"
        )
    for grid in grids:
        cells = grid["n_x"] * grid["n_y"]
        if cells > maximum_grid_cells:
            raise CampaignError(
                f"grid {grid['id']} has {cells} cells, exceeding maximum_grid_cells={maximum_grid_cells}"
            )
    if roots[-1] > maximum_n_m:
        raise CampaignError(f"pilot n_m={roots[-1]} exceeds maximum_n_m={maximum_n_m}")

    raw_next = required_mapping(request.get("next_stage"), "next_stage")
    next_grid_raw = required_mapping(raw_next.get("grid"), "next_stage.grid")
    next_grid = {
        "id": checked_id(next_grid_raw.get("id"), "next_stage.grid.id"),
        "n_x": checked_int(next_grid_raw.get("n_x"), "next_stage.grid.n_x", 1),
        "n_y": checked_int(next_grid_raw.get("n_y"), "next_stage.grid.n_y", 1),
    }
    next_n_m = checked_int(raw_next.get("n_m"), "next_stage.n_m", 1)
    if next_grid["n_x"] <= grids[-1]["n_x"] or next_grid["n_y"] <= grids[-1]["n_y"]:
        raise CampaignError("next_stage.grid must be strictly finer in both dimensions")
    if next_n_m <= roots[-1]:
        raise CampaignError("next_stage.n_m must be strictly greater than the largest pilot n_m")
    if next_grid["n_x"] * next_grid["n_y"] > maximum_grid_cells:
        raise CampaignError("next_stage.grid exceeds maximum_grid_cells")
    if next_n_m > maximum_n_m:
        raise CampaignError("next_stage.n_m exceeds maximum_n_m")
    # Preflight the exact card semantics for every declared pilot grid and the
    # next-stage grid before materializing a campaign.  The coupled-maxima
    # contract changes a runtime validation bound, so checking only the source
    # template can miss a class-bank or reference-CIC lattice incompatibility.
    for template in templates:
        for grid in [*grids, next_grid]:
            probe = patch_card(
                template["raw"],
                template["accelerator"],
                template["parameters"],
                grid["n_x"],
                grid["n_y"],
                roots[0],
                pilot_seeds[0],
                template["grid_patch_contract"],
            )
            probe_info = inspect_campaign_card(
                probe,
                template["accelerator"],
                template["parameters"],
                template["grid_patch_contract"],
            )
            if probe_info["campaign_family_sha256"] != template["family_sha256"]:
                raise CampaignError(
                    "declared-grid preflight changed outside the grid/root/seed patch contract"
                )
    # A replication follow-up is a complete paired factorial for each fresh
    # extension seed.  Bound that actual worst case rather than only one
    # selected-template terminal cell.
    largest_followup = max(
        len(templates)
        * (
            len(grids) * len(roots) * len(extension_seeds)
            + 3 * (len(pilot_seeds) + len(extension_seeds))
        ),
        len(templates) * len(holdout_seeds),
        3 * len(templates) * len(pilot_seeds),
    )
    if largest_followup > maximum_pilot_runs:
        raise CampaignError(
            "a bounded next stage would exceed maximum_pilot_runs; increase the explicit limit"
        )

    return {
        "binary_path": binary_path,
        "binary_sha256": binary_sha,
        "source_state_path": source_path,
        "source_state_sha256": source_sha,
        "generator_transform": generator,
        "generator_transform_sha256": generator_sha,
        "physics_family": physics_family,
        "physics_family_sha256": physics_sha,
        "gaussian_core_transform": gaussian_core_transform,
        "templates": templates,
        "reference_template_id": reference_template_id,
        "endpoints": endpoints,
        "grids": grids,
        "roots": roots,
        "pilot_seeds": pilot_seeds,
        "extension_seeds": extension_seeds,
        "holdout_seeds": holdout_seeds,
        "statistics": {
            "standard_error_safety_factor": decimal_text(safety),
            "confidence_level": decimal_text(confidence),
            "equivalence_alpha": decimal_text(alpha),
            "equivalence_power": decimal_text(power),
            "conservative_total_budget_multiplier": decimal_text(total_budget_multiplier),
            "two_sided_confidence_normal_quantile": decimal_text(z_conf),
            "one_sided_equivalence_normal_quantile": decimal_text(z_alpha),
            "power_normal_quantile": decimal_text(z_power),
            "power_normal_quantile_contract": (
                "z_((1+requested_power)/2)_conservative_symmetric_TOST"
            ),
            "pilot_degrees_of_freedom": str(pilot_degrees_of_freedom),
            "numerical_family_statement_count": str(numerical_family_statement_count),
            "equivalence_family_statement_count": str(equivalence_family_statement_count),
            "familywise_error_control": "bonferroni_v1",
            "two_sided_confidence_familywise_probability": decimal_text(
                family_confidence_probability
            ),
            "two_sided_confidence_familywise_student_t_quantile": decimal_text(t_conf_family),
            "one_sided_equivalence_familywise_probability": decimal_text(
                family_equivalence_probability
            ),
            "one_sided_equivalence_familywise_student_t_quantile": decimal_text(t_alpha_family),
            "one_sided_equivalence_familywise_normal_quantile_for_power": decimal_text(
                z_alpha_family
            ),
        },
        "limits": {
            "maximum_grid_cells": str(maximum_grid_cells),
            "maximum_n_m": str(maximum_n_m),
            "maximum_pilot_runs": str(maximum_pilot_runs),
            "maximum_wall_seconds_per_run": str(maximum_wall_seconds),
        },
        "next_grid": next_grid,
        "next_n_m": next_n_m,
        "run_count": run_count,
    }


def _template_public(template: Mapping[str, Any]) -> dict[str, Any]:
    info = required_mapping(template.get("card_info"), "internal.template.card_info")
    return {
        "id": template["id"],
        "path": template.get("campaign_path", template["path_text"]),
        "sha256": template["sha256"],
        "campaign_family_sha256": template["family_sha256"],
        "accelerator": template["accelerator"],
        "parameters": template["parameters"],
        "grid_patch_contract": template["grid_patch_contract"],
        "adaptive_grid_certificate_contract": {
            "adaptive_grid_mode": str(info["adaptive_grid_mode"]),
            "pilot_diagnostics": str(info["adaptive_grid_pilot_diagnostics"]),
            "fail_on_unmet_error": str(info["adaptive_grid_fail_on_unmet_error"]),
            "terminal_patch_certification": str(info["adaptive_grid_terminal_patch_certification"]),
            "runtime_schema": (
                GRID_CERTIFICATE_SCHEMA
                if info["adaptive_grid_mode"] in (2, 3)
                else "not_required_fixed_grid"
            ),
        },
        "source_distribution_contract": {
            "kind": "feebl_generated_legacy_mc",
            "normalization_contract": NORMALIZATION_ID,
            "charge_symmetric": "0",
            "load_beam": "0",
            "dist_z_beam_1": str(info["dist_z"][0]),
            "dist_z_beam_2": str(info["dist_z"][1]),
            "warnings": info["warnings"],
            "adaptive_macroparticle_mode": str(info["adaptive_macroparticle_mode"]),
        },
    }


def _public_gaussian_transform(config: Mapping[str, Any]) -> dict[str, Any] | None:
    transform = config.get("gaussian_core_transform")
    if transform is None:
        return None
    transform = required_mapping(transform, "internal.gaussian_core_transform")
    result = dict(transform)
    result["path"] = transform["campaign_path"]
    result["python_executable_path"] = transform["python_campaign_path"]
    result.pop("campaign_path", None)
    result.pop("python_campaign_path", None)
    return result


def _public_external_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feebl_binary_path": config["binary_campaign_path"],
        "feebl_binary_sha256": config["binary_sha256"],
        "source_state_path": config["source_state_campaign_path"],
        "source_state_sha256": config["source_state_sha256"],
        "generator_transform": config["generator_transform"],
        "generator_transform_sha256": config["generator_transform_sha256"],
        "physics_family": config["physics_family"],
        "physics_family_sha256": config["physics_family_sha256"],
        "gaussian_core_transform": _public_gaussian_transform(config),
    }


def _rewrite_request_to_snapshots(request: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Replace provenance-only source paths with canonical campaign snapshots."""

    rewritten = copy.deepcopy(dict(request))
    bindings = required_mapping(rewritten.get("bindings"), "bindings")
    bindings["feebl_binary_path"] = str(
        resolve_within(output_dir, FROZEN_BINARY_PATH, "frozen.feebl_binary")
    )
    bindings["source_state_path"] = str(
        resolve_within(output_dir, FROZEN_SOURCE_STATE_PATH, "frozen.source_state")
    )
    raw_templates = required_list(rewritten.get("templates"), "templates")
    for index, raw_template in enumerate(raw_templates):
        template = required_mapping(raw_template, f"templates[{index}]")
        template_id = checked_id(template.get("id"), f"templates[{index}].id")
        template["path"] = str(
            resolve_within(
                output_dir,
                _frozen_template_path(index, template_id),
                f"frozen.templates[{index}]",
            )
        )
    transform = bindings.get("gaussian_core_transform")
    if transform is not None:
        transform = required_mapping(transform, "bindings.gaussian_core_transform")
        transform["path"] = str(resolve_within(output_dir, FROZEN_GAUSSIAN_PATH, "frozen.gaussian"))
        transform["python_executable_path"] = str(
            resolve_within(output_dir, FROZEN_PYTHON_PATH, "frozen.python")
        )
    return rewritten


def _attach_campaign_paths(config: dict[str, Any]) -> dict[str, Any]:
    config["binary_campaign_path"] = FROZEN_BINARY_PATH.as_posix()
    config["source_state_campaign_path"] = FROZEN_SOURCE_STATE_PATH.as_posix()
    for index, template in enumerate(config["templates"]):
        template["campaign_path"] = _frozen_template_path(index, str(template["id"])).as_posix()
    transform = config.get("gaussian_core_transform")
    if transform is not None:
        transform["campaign_path"] = FROZEN_GAUSSIAN_PATH.as_posix()
        transform["python_campaign_path"] = FROZEN_PYTHON_PATH.as_posix()
    return config


def _input_provenance(request_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    transform = config.get("gaussian_core_transform")
    return {
        "status": "provenance_only_never_dereferenced_after_freeze",
        "request_path": str(request_path),
        "feebl_binary_path": str(config["binary_path"]),
        "source_state_path": str(config["source_state_path"]),
        "template_paths": [
            {"id": item["id"], "path": item["path_text"]} for item in config["templates"]
        ],
        "campaign_driver_path": str(Path(__file__).resolve()),
        "replay_helper_path": str(Path(replay.__file__).resolve()),
        "gaussian_core_transform_path": (None if transform is None else str(transform["path"])),
        "python_executable_path": (
            str(Path(sys.executable).resolve())
            if transform is None
            else str(transform["python_executable_path"])
        ),
    }


def _freeze_input_snapshots(
    request_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the complete immutable execution/input closure."""

    script_path = Path(__file__).resolve()
    helper_path = Path(replay.__file__).resolve()
    transform = config.get("gaussian_core_transform")
    python_source = (
        Path(sys.executable).resolve()
        if transform is None
        else Path(str(transform["python_executable_path"]))
    )
    records: dict[str, Any] = {
        "request": _copy_exact_snapshot(
            request_path, output_dir, FROZEN_REQUEST_PATH, "frozen.request"
        ),
        "source_state": _copy_exact_snapshot(
            Path(str(config["source_state_path"])),
            output_dir,
            FROZEN_SOURCE_STATE_PATH,
            "frozen.source_state",
        ),
        "campaign_driver": _copy_exact_snapshot(
            script_path, output_dir, FROZEN_DRIVER_PATH, "frozen.campaign_driver"
        ),
        "replay_helper": _copy_exact_snapshot(
            helper_path,
            output_dir,
            FROZEN_REPLAY_HELPER_PATH,
            "frozen.replay_helper",
        ),
        "feebl_binary": _copy_exact_snapshot(
            Path(str(config["binary_path"])),
            output_dir,
            FROZEN_BINARY_PATH,
            "frozen.feebl_binary",
            executable=True,
        ),
        "python_executable": _copy_exact_snapshot(
            python_source,
            output_dir,
            FROZEN_PYTHON_PATH,
            "frozen.python_executable",
            executable=True,
        ),
        "templates": [],
        "gaussian_core_transform": None,
    }
    for index, template in enumerate(config["templates"]):
        records["templates"].append(
            {
                "id": template["id"],
                **_copy_exact_snapshot(
                    Path(str(template["path"])),
                    output_dir,
                    _frozen_template_path(index, str(template["id"])),
                    f"frozen.templates[{index}]",
                ),
            }
        )
    if transform is not None:
        records["gaussian_core_transform"] = _copy_exact_snapshot(
            Path(str(transform["path"])),
            output_dir,
            FROZEN_GAUSSIAN_PATH,
            "frozen.gaussian_core_transform",
        )
    return records


def _validate_frozen_inputs(
    policy: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate snapshots and reconstruct config without external I/O."""

    frozen = required_mapping(policy.get("frozen_inputs"), "policy.frozen_inputs")
    request_path = _bound_snapshot_path(
        output_dir, frozen.get("request"), FROZEN_REQUEST_PATH, "frozen.request"
    )
    _bound_snapshot_path(
        output_dir,
        frozen.get("source_state"),
        FROZEN_SOURCE_STATE_PATH,
        "frozen.source_state",
    )
    frozen_driver = _bound_snapshot_path(
        output_dir,
        frozen.get("campaign_driver"),
        FROZEN_DRIVER_PATH,
        "frozen.campaign_driver",
    )
    frozen_helper = _bound_snapshot_path(
        output_dir,
        frozen.get("replay_helper"),
        FROZEN_REPLAY_HELPER_PATH,
        "frozen.replay_helper",
    )
    _bound_snapshot_path(
        output_dir,
        frozen.get("feebl_binary"),
        FROZEN_BINARY_PATH,
        "frozen.feebl_binary",
    )
    frozen_python = _bound_snapshot_path(
        output_dir,
        frozen.get("python_executable"),
        FROZEN_PYTHON_PATH,
        "frozen.python_executable",
    )
    # Paths legitimately differ when the campaign is moved or when the
    # caller uses the source-tree entry point.  Bytes may not differ: all
    # campaign logic, parsing logic, and interpreter identity must equal the
    # immutable execution closure before it can validate, execute, or analyze.
    running_identities = (
        (
            Path(__file__).resolve(),
            sha256_file(frozen_driver),
            "running.campaign_driver",
        ),
        (
            Path(replay.__file__).resolve(),
            sha256_file(frozen_helper),
            "running.replay_helper",
        ),
        (
            Path(sys.executable).resolve(),
            sha256_file(frozen_python),
            "running.python_executable",
        ),
    )
    for running_path, frozen_sha, where in running_identities:
        checked_file(running_path, frozen_sha, where)
    request = required_mapping(load_json(request_path), "frozen.request")
    raw_templates = required_list(request.get("templates"), "templates")
    template_bindings = required_list(frozen.get("templates"), "policy.frozen_inputs.templates")
    if len(template_bindings) != len(raw_templates):
        raise CampaignError("frozen template binding count differs from request")
    for index, (raw_template, binding) in enumerate(zip(raw_templates, template_bindings)):
        template = required_mapping(raw_template, f"templates[{index}]")
        template_id = checked_id(template.get("id"), f"templates[{index}].id")
        bound = required_mapping(binding, f"policy.frozen_inputs.templates[{index}]")
        if bound.get("id") != template_id:
            raise CampaignError(f"frozen template binding {index} id mismatch")
        _bound_snapshot_path(
            output_dir,
            {"path": bound.get("path"), "sha256": bound.get("sha256")},
            _frozen_template_path(index, template_id),
            f"frozen.templates[{index}]",
        )
    needs_gaussian = any(
        item.get("format") == "json_scalar"
        for item in required_list(request.get("endpoints"), "endpoints")
        if isinstance(item, dict)
    )
    gaussian_binding = frozen.get("gaussian_core_transform")
    if needs_gaussian:
        _bound_snapshot_path(
            output_dir,
            gaussian_binding,
            FROZEN_GAUSSIAN_PATH,
            "frozen.gaussian_core_transform",
        )
    elif gaussian_binding is not None:
        raise CampaignError("unexpected frozen Gaussian-core transform")
    rewritten = _rewrite_request_to_snapshots(request, output_dir)
    config = _attach_campaign_paths(_configuration(rewritten, output_dir))
    return request, config


def freeze_campaign(request_path: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze a full factorial pilot and materialize every card before execution."""
    request_path = request_path.resolve()
    output_dir = output_dir.resolve()
    request = required_mapping(load_json(request_path), "request")
    original_config = _configuration(request, request_path.parent)
    provenance = _input_provenance(request_path, original_config)
    if output_dir.exists():
        raise CampaignError(f"output directory already exists; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    frozen_inputs = _freeze_input_snapshots(request_path, output_dir, original_config)
    frozen_request = _rewrite_request_to_snapshots(request, output_dir)
    config = _attach_campaign_paths(_configuration(frozen_request, output_dir))

    unsigned_policy: dict[str, Any] = {
        "schema": POLICY_SCHEMA,
        "schema_version": "1",
        "qualification_status": "preregistered_pilot_not_physics_qualified",
        "execution_semantics": "strictly_sequential_fresh_complete_processes",
        "statistical_unit": "independent_complete_FeeBL_run",
        "request_file": FROZEN_REQUEST_PATH.as_posix(),
        "request_file_sha256": frozen_inputs["request"]["sha256"],
        "frozen_inputs": frozen_inputs,
        "input_path_provenance": provenance,
        "implementation_bindings": {
            "campaign_driver": frozen_inputs["campaign_driver"],
            "replay_helper": frozen_inputs["replay_helper"],
            "python_executable": frozen_inputs["python_executable"],
        },
        "external_bindings": _public_external_bindings(config),
        "templates": [_template_public(item) for item in config["templates"]],
        "design": {
            "grid_levels": config["grids"],
            "n_m_levels": [str(item) for item in config["roots"]],
            "pilot_seed_count": str(len(config["pilot_seeds"])),
            "extension_seed_count": str(len(config["extension_seeds"])),
            "holdout_seed_count": str(len(config["holdout_seeds"])),
            "pilot_run_count": str(config["run_count"]),
            "pairing": "same_seed_across_every_grid_and_n_m_cell",
            "reference_template_id": config["reference_template_id"],
        },
        "endpoints": config["endpoints"],
        "statistics": config["statistics"],
        "limits": config["limits"],
        "next_stage_candidates": {
            "grid": config["next_grid"],
            "n_m": str(config["next_n_m"]),
        },
        "scientific_claims": [
            "pilot_values_are_parsed_from_bound_FeeBL_output_files",
            "shared_seeds_support_paired_numerical_contrasts",
            "complete_runs_are_the_only_independent_replicates",
            "extension_seeds_are_not_executed_by_the_pilot_command",
            "holdout_seeds_are_not_executed_by_the_pilot_command",
            "normal_approximation_power_is_planning_only",
        ],
    }
    policy_identity_input = dict(unsigned_policy)
    policy_id = sha256_object(policy_identity_input)
    unsigned_policy["policy_id"] = f"sha256:{policy_id}"
    policy = with_self_hash(unsigned_policy)

    realizations: list[dict[str, Any]] = []
    cards: list[tuple[Path, bytes]] = []
    ordinal = 0
    for template in config["templates"]:
        for grid_index, grid in enumerate(config["grids"]):
            for root_index, n_m in enumerate(config["roots"]):
                for seed_index, seed in enumerate(config["pilot_seeds"]):
                    ordinal += 1
                    directory_text = f"pilot/runs/{ordinal:05d}_{template['id']}_{grid['id']}_nm_{n_m}_seed_{seed}"
                    relative_directory = relative_path(directory_text, "generated.run_directory")
                    card = patch_card(
                        template["raw"],
                        template["accelerator"],
                        template["parameters"],
                        grid["n_x"],
                        grid["n_y"],
                        n_m,
                        seed,
                        template["grid_patch_contract"],
                    )
                    info = inspect_campaign_card(
                        card,
                        template["accelerator"],
                        template["parameters"],
                        template["grid_patch_contract"],
                    )
                    if info["campaign_family_sha256"] != template["family_sha256"]:
                        raise CampaignError(
                            "internal error: generated card changed outside the declared grid/root/seed patch contract"
                        )
                    if (info["n_x"], info["n_y"], info["n_m"], info["rndm_seed"]) != (
                        grid["n_x"],
                        grid["n_y"],
                        n_m,
                        seed,
                    ):
                        raise CampaignError(
                            "internal error: generated card did not retain its design cell"
                        )
                    card_relative = relative_directory / "acc.dat"
                    cards.append((card_relative, card))
                    realizations.append(
                        {
                            "ordinal": str(ordinal),
                            "template_id": template["id"],
                            "grid_index": str(grid_index),
                            "grid_id": grid["id"],
                            "n_x": str(grid["n_x"]),
                            "n_y": str(grid["n_y"]),
                            "n_m_index": str(root_index),
                            "n_m": str(n_m),
                            "seed_index": str(seed_index),
                            "seed": str(seed),
                            "run_directory": relative_directory.as_posix(),
                            "card": card_relative.as_posix(),
                            "card_sha256": replay.sha256_bytes(card),
                            "card_family_sha256": template["family_sha256"],
                            "expected_output": "out.dat",
                            "combined_log": "log.txt",
                            "argv": [
                                config["binary_campaign_path"],
                                template["accelerator"],
                                template["parameters"],
                                "out.dat",
                            ],
                        }
                    )

    policy_path = output_dir / "campaign_policy.json"
    write_canonical(policy_path, policy)
    for relative_card, payload in cards:
        target = resolve_within(output_dir, relative_card, "generated.card")
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)

    unsigned_manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": "1",
        "qualification_status": "unexecuted_pilot",
        "policy_file": "campaign_policy.json",
        "policy_file_sha256": sha256_file(policy_path),
        "policy_self_sha256": policy["self_sha256"],
        "request_file_sha256": frozen_inputs["request"]["sha256"],
        "pilot_seeds": [str(item) for item in config["pilot_seeds"]],
        "preregistered_extension_seeds": [str(item) for item in config["extension_seeds"]],
        "reserved_holdout_seeds": [str(item) for item in config["holdout_seeds"]],
        "endpoints_sha256": sha256_object(config["endpoints"]),
        "realizations": realizations,
    }
    manifest = with_self_hash(unsigned_manifest)
    write_canonical(output_dir / "pilot_manifest.json", manifest)
    return policy, manifest


def _int_text(value: Any, where: str, low: int = 0, high: int = INT32_MAX) -> int:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise CampaignError(f"{where}: expected an unsigned decimal string")
    parsed = checked_int(int(value), where, low, high)
    if value != str(parsed):
        raise CampaignError(f"{where}: expected a canonical unsigned decimal string")
    return parsed


def _exit_code_text(value: Any, where: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"-?[0-9]+", value) is None:
        raise CampaignError(f"{where}: expected a signed decimal exit-code string")
    parsed = int(value)
    if value != str(parsed) or parsed < -INT32_MAX - 1 or parsed > INT32_MAX:
        raise CampaignError(f"{where}: exit code is noncanonical or out of range")
    return parsed


def _policy_identity(policy: Mapping[str, Any]) -> str:
    identity = dict(policy)
    identity.pop("self_sha256", None)
    identity.pop("policy_id", None)
    return f"sha256:{sha256_object(identity)}"


def _validate_policy_against_request(policy: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected_external = _public_external_bindings(config)
    if policy.get("external_bindings") != expected_external:
        raise CampaignError("policy external bindings do not match the SHA-bound request")
    if policy.get("templates") != [_template_public(item) for item in config["templates"]]:
        raise CampaignError("policy templates do not match the SHA-bound request")
    expected_design = {
        "grid_levels": config["grids"],
        "n_m_levels": [str(item) for item in config["roots"]],
        "pilot_seed_count": str(len(config["pilot_seeds"])),
        "extension_seed_count": str(len(config["extension_seeds"])),
        "holdout_seed_count": str(len(config["holdout_seeds"])),
        "pilot_run_count": str(config["run_count"]),
        "pairing": "same_seed_across_every_grid_and_n_m_cell",
        "reference_template_id": config["reference_template_id"],
    }
    if policy.get("design") != expected_design:
        raise CampaignError("policy design does not match the SHA-bound request")
    if policy.get("endpoints") != config["endpoints"]:
        raise CampaignError("policy endpoints do not match the SHA-bound request")
    if policy.get("statistics") != config["statistics"]:
        raise CampaignError("policy statistics do not match the SHA-bound request")
    if policy.get("limits") != config["limits"]:
        raise CampaignError("policy limits do not match the SHA-bound request")
    expected_next = {"grid": config["next_grid"], "n_m": str(config["next_n_m"])}
    if policy.get("next_stage_candidates") != expected_next:
        raise CampaignError("policy next-stage candidates do not match the request")


def validate_campaign(
    policy_path: Path, manifest_path: Path, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Revalidate every frozen byte, design cell, path, and generated card."""
    output_dir = output_dir.resolve()
    policy_path = policy_path.resolve()
    manifest_path = manifest_path.resolve()
    if policy_path != output_dir / "campaign_policy.json":
        raise CampaignError("policy must be the canonical campaign_policy.json inside output_dir")
    if manifest_path != output_dir / "pilot_manifest.json":
        raise CampaignError("manifest must be the canonical pilot_manifest.json inside output_dir")
    policy = validate_self_hash(policy_path, POLICY_SCHEMA)
    manifest = validate_self_hash(manifest_path, MANIFEST_SCHEMA)
    if policy.get("policy_id") != _policy_identity(policy):
        raise CampaignError("policy_id does not match canonical policy content")
    if manifest.get("policy_file") != "campaign_policy.json":
        raise CampaignError("manifest policy filename is noncanonical")
    if manifest.get("policy_file_sha256") != sha256_file(policy_path):
        raise CampaignError("manifest does not bind the supplied policy bytes")
    if manifest.get("policy_self_sha256") != policy.get("self_sha256"):
        raise CampaignError("manifest/policy self-hash mismatch")

    if policy.get("request_file") != FROZEN_REQUEST_PATH.as_posix():
        raise CampaignError("policy request snapshot filename is noncanonical")
    request_sha = checked_hash(policy.get("request_file_sha256"), "policy.request_file_sha256")
    if manifest.get("request_file_sha256") != request_sha:
        raise CampaignError("manifest/request hash mismatch")
    request, config = _validate_frozen_inputs(policy, output_dir)
    if (
        sha256_file(resolve_within(output_dir, FROZEN_REQUEST_PATH, "frozen.request"))
        != request_sha
    ):
        raise CampaignError("policy request hash differs from frozen request binding")
    _validate_policy_against_request(policy, config)

    implementation = required_mapping(
        policy.get("implementation_bindings"), "policy.implementation_bindings"
    )
    frozen = required_mapping(policy.get("frozen_inputs"), "policy.frozen_inputs")
    expected_implementation = {
        "campaign_driver": frozen.get("campaign_driver"),
        "replay_helper": frozen.get("replay_helper"),
        "python_executable": frozen.get("python_executable"),
    }
    if implementation != expected_implementation:
        raise CampaignError("implementation bindings differ from frozen inputs")
    provenance = required_mapping(
        policy.get("input_path_provenance"), "policy.input_path_provenance"
    )
    if provenance.get("status") != "provenance_only_never_dereferenced_after_freeze":
        raise CampaignError("input path provenance is not explicitly non-operative")

    if manifest.get("pilot_seeds") != [str(item) for item in config["pilot_seeds"]]:
        raise CampaignError("manifest pilot seeds do not match the frozen request")
    if manifest.get("preregistered_extension_seeds") != [
        str(item) for item in config["extension_seeds"]
    ]:
        raise CampaignError("manifest extension seeds do not match the frozen request")
    if manifest.get("reserved_holdout_seeds") != [str(item) for item in config["holdout_seeds"]]:
        raise CampaignError("manifest holdout seeds do not match the frozen request")
    if manifest.get("endpoints_sha256") != sha256_object(config["endpoints"]):
        raise CampaignError("manifest endpoint contract hash mismatch")
    realizations = required_list(manifest.get("realizations"), "manifest.realizations")
    if len(realizations) != config["run_count"]:
        raise CampaignError("manifest does not contain the complete frozen factorial")

    ordinal = 0
    for template in config["templates"]:
        for grid_index, grid in enumerate(config["grids"]):
            for root_index, n_m in enumerate(config["roots"]):
                for seed_index, seed in enumerate(config["pilot_seeds"]):
                    ordinal += 1
                    where = f"manifest.realizations[{ordinal - 1}]"
                    realization = required_mapping(realizations[ordinal - 1], where)
                    directory_text = f"pilot/runs/{ordinal:05d}_{template['id']}_{grid['id']}_nm_{n_m}_seed_{seed}"
                    expected_scalars = {
                        "ordinal": str(ordinal),
                        "template_id": template["id"],
                        "grid_index": str(grid_index),
                        "grid_id": grid["id"],
                        "n_x": str(grid["n_x"]),
                        "n_y": str(grid["n_y"]),
                        "n_m_index": str(root_index),
                        "n_m": str(n_m),
                        "seed_index": str(seed_index),
                        "seed": str(seed),
                        "run_directory": directory_text,
                        "card": f"{directory_text}/acc.dat",
                        "card_family_sha256": template["family_sha256"],
                        "expected_output": "out.dat",
                        "combined_log": "log.txt",
                    }
                    for key, expected in expected_scalars.items():
                        if realization.get(key) != expected:
                            raise CampaignError(
                                f"{where}.{key}: expected canonical value {expected!r}"
                            )
                    run_relative = relative_path(
                        realization.get("run_directory"), f"{where}.run_directory"
                    )
                    run_directory = resolve_within(
                        output_dir, run_relative, f"{where}.run_directory"
                    )
                    if not run_directory.is_dir():
                        raise CampaignError(f"{where}: run directory does not exist")
                    card_relative = relative_path(realization.get("card"), f"{where}.card")
                    card_path = resolve_within(output_dir, card_relative, f"{where}.card")
                    if card_path.parent != run_directory:
                        raise CampaignError(f"{where}: card is outside its run directory")
                    checked_file(card_path, realization.get("card_sha256"), f"{where}.card")
                    card = card_path.read_bytes()
                    info = inspect_campaign_card(
                        card,
                        template["accelerator"],
                        template["parameters"],
                        template["grid_patch_contract"],
                    )
                    if (info["n_x"], info["n_y"], info["n_m"], info["rndm_seed"]) != (
                        grid["n_x"],
                        grid["n_y"],
                        n_m,
                        seed,
                    ):
                        raise CampaignError(f"{where}: generated card/design mismatch")
                    if info["campaign_family_sha256"] != template["family_sha256"]:
                        raise CampaignError(
                            f"{where}: card changed outside its declared patch contract"
                        )
                    expected_argv = [
                        config["binary_campaign_path"],
                        template["accelerator"],
                        template["parameters"],
                        "out.dat",
                    ]
                    if realization.get("argv") != expected_argv:
                        raise CampaignError(f"{where}: executable argv binding mismatch")
                    relative_path(
                        realization.get("expected_output"), f"{where}.expected_output", leaf=True
                    )
                    relative_path(
                        realization.get("combined_log"), f"{where}.combined_log", leaf=True
                    )
    return policy, manifest, config


def _run_paths(
    output_dir: Path, realization: Mapping[str, Any], where: str
) -> tuple[Path, Path, Path]:
    run_directory = resolve_within(
        output_dir,
        relative_path(realization.get("run_directory"), f"{where}.run_directory"),
        f"{where}.run_directory",
    )
    card_path = resolve_within(
        output_dir,
        relative_path(realization.get("card"), f"{where}.card"),
        f"{where}.card",
    )
    log_path = resolve_within(
        run_directory,
        relative_path(realization.get("combined_log"), f"{where}.combined_log", leaf=True),
        f"{where}.combined_log",
    )
    if card_path.parent != run_directory:
        raise CampaignError(f"{where}: card is outside its run directory")
    return run_directory, card_path, log_path


def _runtime_signal_evidence(
    log_path: Path,
    realization: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    expected_exit_code: int,
) -> dict[str, Any]:
    """Parse and bind the V2 runtime handoff when adaptive roots are active."""
    if not log_path.is_file():
        raise CampaignError(f"runtime signal log is missing: {log_path}")
    try:
        text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CampaignError(f"runtime signal log is not readable UTF-8: {exc}") from exc
    template_id = str(realization.get("template_id"))
    try:
        template = next(item for item in config["templates"] if item["id"] == template_id)
    except StopIteration as exc:
        raise CampaignError(f"runtime signal references unknown template {template_id!r}") from exc
    adaptive_mode = int(template["card_info"]["adaptive_macroparticle_mode"])
    signal_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_")
    ]
    if adaptive_mode == 0:
        if signal_lines:
            raise CampaignError("nonadaptive template emitted an unexpected adaptive-root signal")
        return {
            "required": False,
            "present": False,
            "combined_log_sha256": sha256_file(log_path),
        }
    try:
        fields = replay.parse_runtime_replay_signal(text)
    except replay.ReplayError as exc:
        raise CampaignError(f"invalid or missing adaptive-root runtime signal: {exc}") from exc
    expected_seed = str(realization.get("seed"))
    expected_n_m = str(realization.get("n_m"))
    expected = {
        "exit_code": str(expected_exit_code),
        "rndm_seed": expected_seed,
        "n_m_beam1": expected_n_m,
        "n_m_beam2": expected_n_m,
        "common_n_m": "1",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise CampaignError(
                f"runtime signal {key}={fields.get(key)!r} does not match realization {value!r}"
            )
    if expected_exit_code == 0:
        disposition = {
            "publication_blocked": "0",
            "exploratory_outputs_authorized": "1",
            "final_physics_outputs_published": "1",
        }
    elif expected_exit_code == replay.REPLAY_REQUIRED_EXIT_CODE:
        disposition = {
            "publication_blocked": "1",
            "exploratory_outputs_authorized": "0",
            "final_physics_outputs_published": "0",
        }
    else:
        raise CampaignError(
            "runtime signal evidence is supported only for exploratory success or strict replay-required exit"
        )
    for key, value in disposition.items():
        if fields.get(key) != value:
            raise CampaignError(f"runtime signal has inconsistent {key}")
    canonical_fields = dict(sorted(fields.items()))
    current_prefix_lines = [
        line for line in signal_lines if line.startswith(replay.RUNTIME_SIGNAL_PREFIX)
    ]
    if len(current_prefix_lines) != 1:
        # The parser also checks this; retaining the exact line here makes its
        # byte hash unambiguous and rejects a mixed legacy/current log.
        raise CampaignError("runtime log does not contain exactly one current V2 signal line")
    if len(signal_lines) != 1:
        raise CampaignError("runtime log contains an additional legacy or duplicate replay signal")
    return {
        "required": True,
        "present": True,
        "schema": fields["schema"],
        "parsed_fields": canonical_fields,
        "parsed_fields_sha256": sha256_object(canonical_fields),
        "signal_line_sha256": replay.sha256_bytes(current_prefix_lines[0].encode("utf-8")),
        "combined_log_sha256": sha256_file(log_path),
    }


def _execution_failure_return_code(
    process_exit_code: int,
    failure_kind: str,
    runtime_signal: Mapping[str, Any] | None,
) -> int:
    """Reserve 75 for a successfully authenticated strict replay request."""
    if (
        process_exit_code == replay.REPLAY_REQUIRED_EXIT_CODE
        and failure_kind == "runtime_replay_required"
        and runtime_signal is not None
        and runtime_signal.get("required") is True
        and runtime_signal.get("present") is True
    ):
        return replay.REPLAY_REQUIRED_EXIT_CODE
    if process_exit_code in (0, replay.REPLAY_REQUIRED_EXIT_CODE):
        return 4
    return process_exit_code


def _validate_failed_execution_record(
    failure: Mapping[str, Any],
    index: int,
    realization: Mapping[str, Any],
    log_path: Path,
    config: Mapping[str, Any],
    where: str,
) -> None:
    """Authenticate one terminal failed/request record against current bytes."""
    required_keys = {
        "ordinal",
        "kind",
        "detail",
        "realization_sha256",
        "exit_code",
        "elapsed_nanoseconds",
        "combined_log_sha256",
    }
    optional_keys = {"runtime_replay_signal"}
    keys = set(failure)
    if not required_keys.issubset(keys) or not keys.issubset(required_keys | optional_keys):
        raise CampaignError(f"{where}: failed-record fields are incomplete or noncanonical")
    if failure.get("ordinal") != str(index + 1):
        raise CampaignError(f"{where}: failure ordinal is inconsistent")
    if failure.get("realization_sha256") != sha256_object(realization):
        raise CampaignError(f"{where}: failed realization hash mismatch")
    kind = failure.get("kind")
    allowed_kinds = {
        "runtime_replay_required",
        "invalid_runtime_signal",
        "invalid_grid_certificate",
        "invalid_output",
        "nonzero_exit",
        "timeout",
        "launch_error",
    }
    if kind not in allowed_kinds:
        raise CampaignError(f"{where}: invalid failure kind")
    detail = failure.get("detail")
    if not isinstance(detail, str) or not detail:
        raise CampaignError(f"{where}: failure detail must be nonempty text")
    exit_code = _exit_code_text(failure.get("exit_code"), f"{where}.exit_code")
    _int_text(
        failure.get("elapsed_nanoseconds"),
        f"{where}.elapsed_nanoseconds",
        1,
        10**18,
    )
    checked_file(
        log_path,
        failure.get("combined_log_sha256"),
        f"{where}.combined_log",
        nonempty=False,
    )

    expected_exit_by_kind = {
        "runtime_replay_required": replay.REPLAY_REQUIRED_EXIT_CODE,
        "invalid_grid_certificate": 0,
        "invalid_output": 0,
        "timeout": 124,
        "launch_error": 126,
    }
    if kind in expected_exit_by_kind and exit_code != expected_exit_by_kind[kind]:
        raise CampaignError(f"{where}: failure kind and exit code are inconsistent")
    if kind in ("invalid_runtime_signal", "invalid_grid_certificate") and exit_code not in (
        0,
        replay.REPLAY_REQUIRED_EXIT_CODE,
    ):
        raise CampaignError(f"{where}: invalid-evidence failure has an impossible exit code")
    if kind == "nonzero_exit" and exit_code in (0, replay.REPLAY_REQUIRED_EXIT_CODE):
        raise CampaignError(f"{where}: nonzero-exit failure has an inconsistent exit code")

    stored_signal = failure.get("runtime_replay_signal")
    if kind == "runtime_replay_required":
        if stored_signal is None:
            raise CampaignError(f"{where}: replay-required failure lacks signal evidence")
        current_signal = _runtime_signal_evidence(
            log_path,
            realization,
            config,
            expected_exit_code=replay.REPLAY_REQUIRED_EXIT_CODE,
        )
        if current_signal.get("required") is not True or stored_signal != current_signal:
            raise CampaignError(f"{where}: replay-required signal evidence changed")
        return

    if kind == "invalid_runtime_signal":
        # The log hash always binds rejected evidence.  A signal object may be
        # present only for the narrow case where parsing succeeded but the
        # surrounding process/card disposition was inconsistent; it never
        # upgrades this record into a valid exit-75 replay request.
        try:
            current_signal = _runtime_signal_evidence(
                log_path,
                realization,
                config,
                expected_exit_code=exit_code,
            )
        except CampaignError:
            if stored_signal is not None:
                raise CampaignError(
                    f"{where}: rejected signal object cannot be reproduced from the log"
                )
        else:
            if stored_signal != current_signal:
                raise CampaignError(f"{where}: rejected signal evidence changed")
            if (
                exit_code == replay.REPLAY_REQUIRED_EXIT_CODE
                and current_signal.get("required") is True
            ):
                raise CampaignError(
                    f"{where}: authenticated exit-75 evidence is misclassified as invalid"
                )
        return

    if kind == "invalid_grid_certificate":
        current_signal = _runtime_signal_evidence(
            log_path,
            realization,
            config,
            expected_exit_code=0,
        )
        if stored_signal != current_signal:
            raise CampaignError(
                f"{where}: replay-signal evidence preceding the rejected grid certificate changed"
            )
        try:
            _adaptive_grid_certificate_evidence(log_path, realization, config)
        except CampaignError:
            return
        raise CampaignError(f"{where}: valid adaptive-grid certificate is misclassified as invalid")

    if stored_signal is not None:
        raise CampaignError(f"{where}: unrelated failure contains replay signal evidence")


def _endpoint_evidence(
    run_directory: Path, endpoints: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    hashes: dict[str, str] = {}
    out_path = resolve_within(run_directory, Path("out.dat"), "run.out.dat")
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise CampaignError("successful FeeBL process did not produce a nonempty out.dat")
    hashes["out.dat"] = sha256_file(out_path)
    for spec in endpoints:
        endpoint_id = str(spec["id"])
        value, file_hash = parse_endpoint(run_directory, spec)
        filename = str(spec["file"])
        values[endpoint_id] = value
        previous = hashes.setdefault(filename, file_hash)
        if previous != file_hash:
            raise CampaignError(f"endpoint output {filename} changed while it was parsed")
    return values, dict(sorted(hashes.items()))


def _gaussian_core_expected_binding(
    run_directory: Path,
    realization: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Derive the fitter identity and lattice from the executed acc.dat."""
    template_id = str(realization.get("template_id"))
    try:
        template = next(item for item in config["templates"] if item["id"] == template_id)
    except StopIteration as exc:
        raise CampaignError(
            f"Gaussian-core realization references unknown template {template_id!r}"
        ) from exc
    card_path = resolve_within(run_directory, Path("acc.dat"), "gaussian_core.acc.dat")
    if not card_path.is_file() or card_path.stat().st_size == 0:
        raise CampaignError("Gaussian-core executed acc.dat is missing or empty")
    info = inspect_campaign_card(
        card_path.read_bytes(),
        template["accelerator"],
        template["parameters"],
        template["grid_patch_contract"],
    )
    expected_seed = str(realization.get("seed"))
    if str(info["rndm_seed"]) != expected_seed:
        raise CampaignError("Gaussian-core executed-card seed disagrees with realization")
    run_id = Path(str(realization.get("run_directory"))).name
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) is None:
        raise CampaignError("Gaussian-core run directory does not yield a safe run id")

    def fitter_decimal(value: Any, label: str) -> str:
        converted = float(decimal_value(value, f"Gaussian-core card.{label}"))
        if not math.isfinite(converted):
            raise CampaignError(f"Gaussian-core {label} is outside finite binary64")
        return format(converted, ".17g")

    lumi_specs = [
        endpoint
        for endpoint in config["endpoints"]
        if endpoint.get("file") == "out.dat"
        and endpoint.get("format") == "assignment"
        and endpoint.get("key") == "lumi_ee"
    ]
    if len(lumi_specs) != 1:
        raise CampaignError(
            "Gaussian-core qualification requires exactly one out.dat lumi_ee endpoint"
        )
    expected_total_luminosity, _ = parse_endpoint(run_directory, lumi_specs[0])

    return {
        "run_id": run_id,
        "rndm_seed": expected_seed,
        "spectrum_w_min": fitter_decimal(info["spectrum_w_min"], "spectrum_w_min"),
        "spectrum_w_max": fitter_decimal(info["spectrum_w_max"], "spectrum_w_max"),
        "spectrum_w_bins": str(info["spectrum_w_bins"]),
        "total_luminosity": fitter_decimal(expected_total_luminosity, "out.dat.lumi_ee"),
    }


def _gaussian_core_output_evidence(
    run_directory: Path,
    realization: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    transform = config.get("gaussian_core_transform")
    if transform is None:
        return {}
    transform = required_mapping(transform, "gaussian_core_transform")
    input_path = resolve_within(run_directory, Path(GAUSSIAN_CORE_INPUT), "gaussian_core.input")
    json_path = resolve_within(run_directory, Path(GAUSSIAN_CORE_JSON), "gaussian_core.json")
    csv_path = resolve_within(run_directory, Path(GAUSSIAN_CORE_CSV), "gaussian_core.csv")
    for path, where in (
        (input_path, "gaussian_core.input"),
        (json_path, "gaussian_core.json"),
        (csv_path, "gaussian_core.csv"),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise CampaignError(f"{where}: missing or empty transform artifact")
    document = required_mapping(load_json(json_path), "gaussian_core.json")
    try:
        expected_json = replay.canonical_bytes(document) + b"\n"
    except replay.ReplayError as exc:
        raise CampaignError(str(exc)) from exc
    if json_path.read_bytes() != expected_json:
        raise CampaignError("gaussian_core.json is not canonical compact UTF-8")
    if document.get("schema") != GAUSSIAN_CORE_SCHEMA or document.get("schema_version") != "1":
        raise CampaignError("gaussian_core.json schema mismatch")
    if document.get("status") != "accepted_core_fit_not_physics_qualification":
        raise CampaignError("Gaussian-core transform did not publish an accepted fit")
    input_evidence = required_mapping(document.get("input"), "gaussian_core.input_evidence")
    if input_evidence.get("file_name") != GAUSSIAN_CORE_INPUT:
        raise CampaignError("Gaussian-core JSON names the wrong input file")
    if input_evidence.get("sha256") != sha256_file(input_path):
        raise CampaignError("Gaussian-core JSON/input spectrum hash mismatch")
    expected_binding = _gaussian_core_expected_binding(run_directory, realization, config)
    binding = required_mapping(
        document.get("qualification_binding"), "gaussian_core.qualification_binding"
    )
    expected_binding_fields: dict[str, Any] = {
        "status": "qualification_input_bound",
        "native_format": "FEEBL-NATIVE 1.0",
        "rndm_seed": expected_binding["rndm_seed"],
        "run_id": expected_binding["run_id"],
        "spectrum_w_min_GeV": expected_binding["spectrum_w_min"],
        "spectrum_w_max_GeV": expected_binding["spectrum_w_max"],
        "spectrum_w_bins": expected_binding["spectrum_w_bins"],
        "expected_total_luminosity": expected_binding["total_luminosity"],
        "lattice_phase_proven": True,
    }
    for key, expected_value in expected_binding_fields.items():
        if binding.get(key) != expected_value:
            raise CampaignError(
                f"Gaussian-core qualification binding {key} disagrees with executed acc.dat"
            )
    for key in sorted(GAUSSIAN_CORE_ENDPOINT_KEYS):
        value: Any = document
        for part in key.split("."):
            value = required_mapping(value, f"gaussian_core.{part}").get(part)
        if not isinstance(value, str):
            raise CampaignError(f"Gaussian-core field {key} is not an exact decimal string")
        decimal_value(value, f"gaussian_core.{key}")

    # The human-readable CSV is an authenticated rendition of the same fit,
    # not an unaudited side product.  Bind its provenance columns and every
    # scalar that can be consumed from the JSON contract.  The exact bytes are
    # still retained below, so extra diagnostic columns remain deterministic
    # evidence without expanding the endpoint schema.
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CampaignError(f"cannot parse Gaussian-core CSV: {exc}") from exc
    if len(rows) != 1 or reader.fieldnames is None:
        raise CampaignError("Gaussian-core CSV must contain exactly one data row")
    row = rows[0]
    required_csv = {
        "schema": GAUSSIAN_CORE_SCHEMA,
        "input_file_name": GAUSSIAN_CORE_INPUT,
        "input_sha256": str(input_evidence["sha256"]),
        "status": "accepted_core_fit_not_physics_qualification",
        "binding_status": "qualification_input_bound",
        "run_id": expected_binding["run_id"],
        "rndm_seed": expected_binding["rndm_seed"],
        "spectrum_w_min_GeV": expected_binding["spectrum_w_min"],
        "spectrum_w_max_GeV": expected_binding["spectrum_w_max"],
        "spectrum_w_bins": expected_binding["spectrum_w_bins"],
        "expected_total_luminosity": expected_binding["total_luminosity"],
        "lattice_phase_proven": "true",
    }
    for key, expected in required_csv.items():
        if row.get(key) != expected:
            raise CampaignError(f"Gaussian-core CSV field {key} disagrees with JSON evidence")
    for key in sorted(GAUSSIAN_CORE_ENDPOINT_KEYS):
        section, field = key.split(".", 1)
        expected = required_mapping(document.get(section), f"gaussian_core.{section}").get(field)
        if row.get(field) != expected:
            raise CampaignError(f"Gaussian-core CSV field {field} disagrees with JSON evidence")
    return {
        GAUSSIAN_CORE_INPUT: sha256_file(input_path),
        GAUSSIAN_CORE_JSON: sha256_file(json_path),
        GAUSSIAN_CORE_CSV: sha256_file(csv_path),
    }


def _run_gaussian_core_transform(
    run_directory: Path,
    log_path: Path,
    realization: Mapping[str, Any],
    config: Mapping[str, Any],
    deadline_ns: int,
) -> None:
    transform = config.get("gaussian_core_transform")
    if transform is None:
        return
    transform = required_mapping(transform, "gaussian_core_transform")
    script_path = Path(str(transform["path"]))
    python_path = Path(str(transform["python_executable_path"]))
    checked_file(script_path, transform.get("sha256"), "gaussian_core_transform.script")
    checked_file(
        python_path,
        transform.get("python_executable_sha256"),
        "gaussian_core_transform.python",
    )
    input_path = resolve_within(run_directory, Path(GAUSSIAN_CORE_INPUT), "gaussian_core.input")
    json_path = resolve_within(run_directory, Path(GAUSSIAN_CORE_JSON), "gaussian_core.json")
    csv_path = resolve_within(run_directory, Path(GAUSSIAN_CORE_CSV), "gaussian_core.csv")
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise CampaignError("Gaussian-core transform input spectrum is missing or empty")
    if json_path.exists() or csv_path.exists():
        raise CampaignError("refusing to overwrite Gaussian-core transform outputs")
    remaining_ns = deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise CampaignError("no frozen per-run wall time remains for Gaussian-core transform")
    expected_binding = _gaussian_core_expected_binding(run_directory, realization, config)
    argv = [
        str(python_path),
        "-B",
        str(script_path),
        str(input_path),
        "--json",
        str(json_path),
        "--csv",
        str(csv_path),
        "--expected-seed",
        expected_binding["rndm_seed"],
        "--expected-w-min",
        expected_binding["spectrum_w_min"],
        "--expected-w-max",
        expected_binding["spectrum_w_max"],
        "--expected-bins",
        expected_binding["spectrum_w_bins"],
        "--expected-total-luminosity",
        expected_binding["total_luminosity"],
        "--expected-run-id",
        expected_binding["run_id"],
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        with log_path.open("ab") as log_handle:
            process = subprocess.run(
                argv,
                cwd=run_directory,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=max(0.001, remaining_ns / 1_000_000_000),
                check=False,
                shell=False,
                env=environment,
            )
    except subprocess.TimeoutExpired as exc:
        raise CampaignError(
            "Gaussian-core transform exceeded the frozen per-run wall time"
        ) from exc
    except OSError as exc:
        raise CampaignError(f"cannot launch Gaussian-core transform: {exc}") from exc
    if process.returncode != 0:
        raise CampaignError(f"Gaussian-core transform returned {process.returncode}")
    _gaussian_core_output_evidence(run_directory, realization, config)


def _complete_output_evidence(
    run_directory: Path,
    realization: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    values, hashes = _endpoint_evidence(run_directory, endpoints)
    for filename, file_hash in _gaussian_core_output_evidence(
        run_directory, realization, config
    ).items():
        previous = hashes.setdefault(filename, file_hash)
        if previous != file_hash:
            raise CampaignError(f"transform artifact {filename} changed during validation")
    return values, dict(sorted(hashes.items()))


def _state_unsigned(
    policy_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    status: str,
    completed: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "schema_version": "1",
        "qualification_status": "execution_evidence_only_not_physics_qualified",
        "execution_semantics": "strictly_sequential_fresh_complete_processes",
        "policy_file_sha256": sha256_file(policy_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_self_sha256": manifest["self_sha256"],
        "status": status,
        "next_ordinal": str(len(completed) + 1),
        "completed": list(completed),
    }
    if failure is not None:
        result["failure"] = dict(failure)
    return result


def validate_state(
    state_path: Path,
    policy_path: Path,
    manifest_path: Path,
    output_dir: Path,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the completed prefix and detect ambiguous partial work."""
    del policy  # Bound by hashes below; retained in the signature for callers.
    state = validate_self_hash(state_path, STATE_SCHEMA)
    if state.get("policy_file_sha256") != sha256_file(policy_path):
        raise CampaignError("pilot state/policy hash mismatch")
    if state.get("manifest_file_sha256") != sha256_file(manifest_path):
        raise CampaignError("pilot state/manifest hash mismatch")
    if state.get("manifest_self_sha256") != manifest.get("self_sha256"):
        raise CampaignError("pilot state/manifest self-hash mismatch")
    status_value = state.get("status")
    if status_value not in ("running", "complete", "failed"):
        raise CampaignError("pilot state has an invalid status")
    realizations = required_list(manifest.get("realizations"), "manifest.realizations")
    completed = required_list(state.get("completed"), "state.completed")
    if len(completed) > len(realizations):
        raise CampaignError("pilot state completed prefix exceeds the manifest")
    if _int_text(state.get("next_ordinal"), "state.next_ordinal", 1) != len(completed) + 1:
        raise CampaignError("pilot state next_ordinal is inconsistent")
    if status_value == "complete" and len(completed) != len(realizations):
        raise CampaignError("complete pilot state does not cover every realization")
    if status_value == "running" and len(completed) == len(realizations):
        raise CampaignError("running pilot state already covers every realization")
    if status_value == "failed" and not isinstance(state.get("failure"), dict):
        raise CampaignError("failed pilot state lacks failure evidence")
    if status_value != "failed" and "failure" in state:
        raise CampaignError("nonfailed pilot state contains failure evidence")

    if status_value == "failed":
        failed_index = len(completed)
        if failed_index >= len(realizations):
            raise CampaignError("failed pilot state has no corresponding realization")
        failed_realization = required_mapping(
            realizations[failed_index], f"manifest.realizations[{failed_index}]"
        )
        _, _, failed_log_path = _run_paths(
            output_dir,
            failed_realization,
            f"manifest.realizations[{failed_index}]",
        )
        _validate_failed_execution_record(
            required_mapping(state.get("failure"), "state.failure"),
            failed_index,
            failed_realization,
            failed_log_path,
            config,
            "state.failure",
        )

    endpoints = config["endpoints"]
    for index, realization_raw in enumerate(realizations):
        where = f"manifest.realizations[{index}]"
        realization = required_mapping(realization_raw, where)
        run_directory, card_path, log_path = _run_paths(output_dir, realization, where)
        allowed_before_run = {card_path.name}
        if index >= len(completed):
            extras = sorted(
                item.name for item in run_directory.iterdir() if item.name not in allowed_before_run
            )
            # A failed process is deliberately non-resumable, and its partial
            # files are authenticated by the failure record instead.
            if extras and not (status_value == "failed" and index == len(completed)):
                raise CampaignError(
                    f"ambiguous unrecorded artifacts block safe resume in {run_directory}: {extras}"
                )
            continue

        record = required_mapping(completed[index], f"state.completed[{index}]")
        if record.get("ordinal") != str(index + 1):
            raise CampaignError(f"state.completed[{index}]: noncanonical ordinal")
        if record.get("realization_sha256") != sha256_object(realization):
            raise CampaignError(f"state.completed[{index}]: realization hash mismatch")
        if record.get("exit_code") != "0":
            raise CampaignError(
                f"state.completed[{index}]: successful prefix must have exit_code=0"
            )
        _int_text(
            record.get("elapsed_nanoseconds"),
            f"state.completed[{index}].elapsed_nanoseconds",
            1,
            10**18,
        )
        checked_file(
            log_path,
            record.get("combined_log_sha256"),
            f"state.completed[{index}].log",
            nonempty=False,
        )
        signal_evidence = _runtime_signal_evidence(
            log_path, realization, config, expected_exit_code=0
        )
        if record.get("runtime_replay_signal") != signal_evidence:
            raise CampaignError(f"state.completed[{index}]: runtime replay signal evidence changed")
        grid_certificate = _adaptive_grid_certificate_evidence(log_path, realization, config)
        if record.get("adaptive_grid_certificate") != grid_certificate:
            raise CampaignError(
                f"state.completed[{index}]: adaptive-grid certificate evidence changed"
            )
        values, hashes = _complete_output_evidence(run_directory, realization, endpoints, config)
        if record.get("endpoint_values") != values:
            raise CampaignError(f"state.completed[{index}]: parsed endpoint values changed")
        if record.get("output_file_sha256") != hashes:
            raise CampaignError(f"state.completed[{index}]: output file hashes changed")
    return state


def execute_pilot(policy_path: Path, manifest_path: Path, output_dir: Path) -> int:
    """Execute or safely resume only the preregistered pilot, one process at a time."""
    output_dir = output_dir.resolve()
    policy_path = policy_path.resolve()
    manifest_path = manifest_path.resolve()
    policy, manifest, config = validate_campaign(policy_path, manifest_path, output_dir)
    binary = Path(str(config["binary_path"]))
    if os.name != "nt" and not os.access(binary, os.X_OK):
        raise CampaignError(f"bound FeeBL binary is not executable: {binary}")
    state_path = output_dir / "pilot_state.json"
    if state_path.exists():
        state = validate_state(
            state_path, policy_path, manifest_path, output_dir, policy, manifest, config
        )
        if state["status"] == "complete":
            return 0
        if state["status"] == "failed":
            raise CampaignError("pilot state is failed; automatic retry is forbidden")
        completed: list[dict[str, Any]] = list(state["completed"])
    else:
        # A missing state is safe only immediately after freeze. Any process
        # artifact makes its provenance ambiguous and blocks execution.
        for index, raw_realization in enumerate(manifest["realizations"]):
            run_directory, card_path, _ = _run_paths(
                output_dir,
                required_mapping(raw_realization, f"manifest.realizations[{index}]"),
                f"manifest.realizations[{index}]",
            )
            extras = sorted(
                item.name for item in run_directory.iterdir() if item.name != card_path.name
            )
            if extras:
                raise CampaignError(
                    f"unrecorded artifacts block initial execution in {run_directory}: {extras}"
                )
        completed = []
        initial = with_self_hash(
            _state_unsigned(policy_path, manifest_path, manifest, "running", completed)
        )
        _atomic_write(state_path, initial)

    realizations = manifest["realizations"]
    maximum_seconds = int(config["limits"]["maximum_wall_seconds_per_run"])
    for index in range(len(completed), len(realizations)):
        where = f"manifest.realizations[{index}]"
        realization = required_mapping(realizations[index], where)
        run_directory, _, log_path = _run_paths(output_dir, realization, where)
        if log_path.exists():
            raise CampaignError(f"refusing to overwrite log for {where}")
        argv = realization.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise CampaignError(f"{where}: invalid argv")
        launch_argv = [str(binary), *argv[1:]]
        start_ns = time.monotonic_ns()
        exit_code: int
        failure_kind: str | None = None
        failure_detail: str | None = None
        try:
            with log_path.open("xb") as log_handle:
                try:
                    process = subprocess.run(
                        launch_argv,
                        cwd=run_directory,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        timeout=maximum_seconds,
                        check=False,
                        shell=False,
                    )
                    exit_code = process.returncode
                    if exit_code != 0:
                        if exit_code == replay.REPLAY_REQUIRED_EXIT_CODE:
                            failure_kind = "runtime_replay_required"
                            failure_detail = (
                                "strict adaptive-root gate requested fresh complete-process replay"
                            )
                        else:
                            failure_kind = "nonzero_exit"
                            failure_detail = f"FeeBL returned {exit_code}"
                except subprocess.TimeoutExpired:
                    exit_code = 124
                    failure_kind = "timeout"
                    failure_detail = f"exceeded {maximum_seconds} seconds"
        except OSError as exc:
            exit_code = 126
            failure_kind = "launch_error"
            failure_detail = str(exc)
        output_error: CampaignError | None = None
        if failure_kind is None and exit_code == 0:
            try:
                _run_gaussian_core_transform(
                    run_directory,
                    log_path,
                    realization,
                    config,
                    start_ns + maximum_seconds * 1_000_000_000,
                )
            except CampaignError as exc:
                output_error = exc
        elapsed_ns = max(1, time.monotonic_ns() - start_ns)

        runtime_signal: dict[str, Any] | None = None
        grid_certificate: dict[str, Any] | None = None
        if exit_code in (0, replay.REPLAY_REQUIRED_EXIT_CODE) and log_path.is_file():
            try:
                runtime_signal = _runtime_signal_evidence(
                    log_path,
                    realization,
                    config,
                    expected_exit_code=exit_code,
                )
                if (
                    exit_code == replay.REPLAY_REQUIRED_EXIT_CODE
                    and runtime_signal.get("required") is not True
                ):
                    raise CampaignError("strict replay-required exit came from a nonadaptive card")
            except CampaignError as exc:
                failure_kind = "invalid_runtime_signal"
                failure_detail = str(exc)
        if failure_kind is None and exit_code == 0 and log_path.is_file():
            try:
                grid_certificate = _adaptive_grid_certificate_evidence(
                    log_path, realization, config
                )
            except CampaignError as exc:
                failure_kind = "invalid_grid_certificate"
                failure_detail = str(exc)

        if failure_kind is not None:
            if not log_path.is_file():
                try:
                    with log_path.open("xb") as log_handle:
                        log_handle.write(
                            ("campaign launch failure: " + str(failure_detail) + "\n").encode(
                                "utf-8"
                            )
                        )
                except OSError as exc:
                    raise CampaignError(
                        f"cannot materialize authenticated failure log {log_path}: {exc}"
                    ) from exc
            failure: dict[str, Any] = {
                "ordinal": str(index + 1),
                "kind": failure_kind,
                "detail": failure_detail,
                "realization_sha256": sha256_object(realization),
                "exit_code": str(exit_code),
                "elapsed_nanoseconds": str(elapsed_ns),
            }
            if log_path.is_file():
                failure["combined_log_sha256"] = sha256_file(log_path)
            if runtime_signal is not None:
                failure["runtime_replay_signal"] = runtime_signal
            failed_state = with_self_hash(
                _state_unsigned(policy_path, manifest_path, manifest, "failed", completed, failure)
            )
            _atomic_replace(state_path, failed_state)
            return _execution_failure_return_code(exit_code, failure_kind, runtime_signal)

        try:
            if output_error is not None:
                raise output_error
            values, output_hashes = _complete_output_evidence(
                run_directory, realization, config["endpoints"], config
            )
        except CampaignError as exc:
            failure = {
                "ordinal": str(index + 1),
                "kind": "invalid_output",
                "detail": str(exc),
                "realization_sha256": sha256_object(realization),
                "exit_code": "0",
                "elapsed_nanoseconds": str(elapsed_ns),
                "combined_log_sha256": sha256_file(log_path),
            }
            failed_state = with_self_hash(
                _state_unsigned(policy_path, manifest_path, manifest, "failed", completed, failure)
            )
            _atomic_replace(state_path, failed_state)
            raise
        completed.append(
            {
                "ordinal": str(index + 1),
                "realization_sha256": sha256_object(realization),
                "exit_code": "0",
                "elapsed_nanoseconds": str(elapsed_ns),
                "combined_log_sha256": sha256_file(log_path),
                "runtime_replay_signal": runtime_signal,
                "adaptive_grid_certificate": grid_certificate,
                "output_file_sha256": output_hashes,
                "endpoint_values": values,
            }
        )
        status_value = "complete" if len(completed) == len(realizations) else "running"
        updated_state = with_self_hash(
            _state_unsigned(policy_path, manifest_path, manifest, status_value, completed)
        )
        _atomic_replace(state_path, updated_state)
    return 0


def _sample_summary(values: Sequence[Decimal]) -> tuple[Decimal, Decimal, Decimal]:
    if len(values) < 2:
        raise CampaignError("at least two independent complete runs are required")
    with localcontext() as context:
        context.prec = 60
        count = Decimal(len(values))
        mean = sum(values, Decimal(0)) / count
        variance = sum((item - mean) ** 2 for item in values) / Decimal(len(values) - 1)
        standard_deviation = variance.sqrt()
        standard_error = standard_deviation / count.sqrt()
    return mean, standard_error, standard_deviation


def _paired_timing_summaries(
    elapsed_by_template_and_case: Mapping[str, Mapping[tuple[Any, ...], int]],
    reference_template_id: str,
    confidence_level: Decimal,
) -> dict[str, dict[str, Any]]:
    """Compare timing only through matched process pairs and a two-sided CI."""
    if reference_template_id not in elapsed_by_template_and_case:
        raise CampaignError("timing comparison has no reference template")
    reference = elapsed_by_template_and_case[reference_template_id]
    summaries: dict[str, dict[str, Any]] = {}
    for template_id, timings in elapsed_by_template_and_case.items():
        if set(timings) != set(reference):
            raise CampaignError("timing comparison is not a complete matched-pair design")
        if template_id == reference_template_id:
            summaries[template_id] = {
                "comparison_to_reference": "reference",
                "paired_process_count": str(len(reference)),
                "candidate_minus_reference_mean_seconds": "0",
                "candidate_minus_reference_ci_low_seconds": "0",
                "candidate_minus_reference_ci_high_seconds": "0",
                "confidence_level": decimal_text(confidence_level),
            }
            continue
        differences = [
            Decimal(timings[key] - reference[key]) / Decimal(1_000_000_000)
            for key in sorted(reference, key=repr)
        ]
        mean, standard_error, _ = _sample_summary(differences)
        probability = (Decimal(1) + confidence_level) / Decimal(2)
        critical = _student_t_quantile(probability, len(differences) - 1)
        low = mean - critical * standard_error
        high = mean + critical * standard_error
        comparison = (
            "significantly_faster"
            if high < 0
            else ("significantly_slower" if low > 0 else "unresolved_timing_tie")
        )
        summaries[template_id] = {
            "comparison_to_reference": comparison,
            "paired_process_count": str(len(differences)),
            "candidate_minus_reference_mean_seconds": decimal_text(mean),
            "candidate_minus_reference_ci_low_seconds": decimal_text(low),
            "candidate_minus_reference_ci_high_seconds": decimal_text(high),
            "confidence_level": decimal_text(confidence_level),
        }
    return summaries


def _required_complete_runs_for_bound(
    standard_deviation: Decimal,
    available_gap: Decimal,
    multiplier: Decimal,
) -> int:
    """Conservative fixed-critical-value replication planning count.

    ``multiplier`` is the already familywise, small-sample Student-t critical
    value times the frozen safety factor.  Holding that pilot critical value
    fixed as ``n`` grows is deliberately conservative; importantly, the
    returned number counts fresh complete processes, never particles inside a
    process or a larger ``n_m``.
    """
    if available_gap <= 0:
        return INT32_MAX
    with localcontext() as context:
        context.prec = 60
        raw_required = (multiplier * standard_deviation / available_gap) ** 2
    return max(MINIMUM_PILOT_SEEDS, replay.ceil_decimal(raw_required))


def _contrast_summary(
    values: Sequence[Decimal],
    scale: Decimal,
    target_fraction: Decimal,
    multiplier: Decimal,
) -> dict[str, str]:
    mean, standard_error, standard_deviation = _sample_summary(values)
    target = scale * target_fraction
    bound = abs(mean) + multiplier * standard_error
    gap = target - abs(mean)
    precision_required = _required_complete_runs_for_bound(standard_deviation, gap, multiplier)
    return {
        "complete_run_count": str(len(values)),
        "paired_mean": decimal_text(mean),
        "paired_sample_standard_deviation": decimal_text(standard_deviation),
        "paired_sample_standard_error": decimal_text(standard_error),
        "conservative_absolute_bound": decimal_text(bound),
        "absolute_paired_mean_over_target": decimal_text(abs(mean) / target),
        "familywise_uncertainty_over_target": decimal_text(multiplier * standard_error / target),
        "approximate_precision_required_complete_runs": str(precision_required),
        "target_absolute": decimal_text(target),
        "conservative_bound_over_target": decimal_text(bound / target),
    }


def _sampling_summary(
    values: Sequence[Decimal],
    scale: Decimal,
    target_fraction: Decimal,
    multiplier: Decimal,
) -> dict[str, str]:
    mean, standard_error, standard_deviation = _sample_summary(values)
    target = scale * target_fraction
    bound = multiplier * standard_error
    precision_required = _required_complete_runs_for_bound(standard_deviation, target, multiplier)
    return {
        "complete_run_count": str(len(values)),
        "mean": decimal_text(mean),
        "independent_run_sample_standard_deviation": decimal_text(standard_deviation),
        "independent_run_sample_standard_error": decimal_text(standard_error),
        "conservative_absolute_bound": decimal_text(bound),
        "familywise_uncertainty_over_target": decimal_text(bound / target),
        "approximate_precision_required_complete_runs": str(precision_required),
        "target_absolute": decimal_text(target),
        "conservative_bound_over_target": decimal_text(bound / target),
    }


def _equivalence_summary(
    values: Sequence[Decimal],
    scale: Decimal,
    margin_fraction: Decimal,
    expected_fraction: Decimal,
    equivalence_multiplier: Decimal,
    z_alpha: Decimal,
    z_power: Decimal,
) -> dict[str, Any]:
    mean, standard_error, standard_deviation = _sample_summary(values)
    margin = scale * margin_fraction
    conservative_bound = abs(mean) + equivalence_multiplier * standard_error
    gap = scale * (margin_fraction - abs(expected_fraction))
    if gap <= 0:
        raise CampaignError("equivalence power has a nonpositive margin gap")
    with localcontext() as context:
        context.prec = 60
        raw_required = ((z_alpha + z_power) * standard_deviation / gap) ** 2
    required = max(MINIMUM_PILOT_SEEDS, replay.ceil_decimal(raw_required))
    precision_required = _required_complete_runs_for_bound(
        standard_deviation, margin - abs(mean), equivalence_multiplier
    )
    combined_required = max(required, precision_required)
    return {
        "complete_run_count": str(len(values)),
        "paired_mean_shift": decimal_text(mean),
        "paired_sample_standard_deviation": decimal_text(standard_deviation),
        "paired_sample_standard_error": decimal_text(standard_error),
        "equivalence_margin_absolute": decimal_text(margin),
        "conservative_equivalence_bound": decimal_text(conservative_bound),
        "absolute_paired_mean_over_margin": decimal_text(abs(mean) / margin),
        "conservative_bound_over_margin": decimal_text(conservative_bound / margin),
        "equivalent_at_frozen_margin": conservative_bound <= margin,
        "approximate_power_required_complete_runs": str(required),
        "approximate_precision_required_complete_runs": str(precision_required),
        "approximate_combined_required_complete_runs": str(combined_required),
        "power_formula": (
            "ceil(((z_(1-alpha)+z_((1+power)/2))*paired_sd/"
            "(scale*(margin-|expected|)))^2); conservative_normal_TOST"
        ),
    }


def _action_axis(action: str) -> str | None:
    """Return the numerical axis without confusing diagnostic/seed semantics."""
    normalized = action.removeprefix("diagnostic_").removesuffix("_replicate")
    return normalized if normalized in {"grid", "root", "joint"} else None


def _solution_verification_design_status(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """State why v1 evidence is diagnostic rather than absolute convergence.

    V1 estimates only one adjacent difference on each axis and its holdout
    stage samples only the terminal cell.  Neither a third level nor a fresh
    holdout bracket can be inferred after looking at results.  Until an
    explicitly preregistered observed-order/Richardson/GCI design and
    bracketed holdout are implemented, final qualification is fail closed.
    """
    blockers: list[str] = []
    if len(config["grids"]) < 3:
        blockers.append("fewer_than_three_preregistered_grid_levels")
    if len(config["roots"]) < 3:
        blockers.append("fewer_than_three_preregistered_n_m_levels")
    for endpoint in config["endpoints"]:
        if endpoint.get("scale_contract") is None:
            blockers.append(f"endpoint_scale_reference_provenance_missing:{endpoint['id']}")
    blockers.extend(
        [
            "observed_order_richardson_gci_not_implemented_in_v1_schema",
            "sequestered_holdout_lacks_independent_numerical_bracketing",
        ]
    )
    return {
        "absolute_solution_convergence_qualification_ready": False,
        "status": "hard_blocked_diagnostic_only",
        "grid_level_count": str(len(config["grids"])),
        "n_m_level_count": str(len(config["roots"])),
        "minimum_levels_per_axis_for_future_gci": "3",
        "blockers": blockers,
        "manual_override_permitted": False,
    }


def _choose_action(
    grid_ratio: Decimal,
    root_ratio: Decimal,
    interaction_ratio: Decimal,
    sampling_ratio: Decimal,
    total_ratio: Decimal,
    grid_point_ratio: Decimal,
    root_point_ratio: Decimal,
    interaction_point_ratio: Decimal,
    total_point_ratio: Decimal,
    total_budget_multiplier: Decimal,
    method_pass: bool,
    method_point_pass: bool,
    required_power_runs: int,
    pilot_count: int,
    extension_count: int,
    holdout_count: int,
) -> tuple[str, bool, list[str]]:
    reasons: list[str] = []
    if not method_point_pass:
        return (
            "rejected_method_bias",
            False,
            ["paired_method_shift_exceeds_frozen_equivalence_margin"],
        )
    if not method_pass:
        reasons.append("method_equivalence_precision_insufficient")
    if grid_ratio > 1:
        reasons.append("mesh_contrast_above_target")
    if root_ratio > 1:
        reasons.append("root_contrast_above_target")
    if interaction_ratio > 1:
        reasons.append("grid_root_interaction_above_target")
    if sampling_ratio > 1:
        reasons.append("independent_run_sampling_error_above_target")
    if required_power_runs > holdout_count:
        reasons.append("reserved_holdout_has_insufficient_approximate_equivalence_power")
    if total_ratio > total_budget_multiplier:
        reasons.append("summed_conservative_error_budget_above_frozen_multiplier")
    component_gate = max(grid_ratio, root_ratio, interaction_ratio, sampling_ratio) <= 1
    gate = (
        component_gate
        and total_ratio <= total_budget_multiplier
        and method_pass
        and required_power_runs <= holdout_count
    )
    if gate:
        return "hold", True, ["pilot_gate_passed_for_frozen_error_and_power_criteria"]
    # A point-estimate numerical deficit calls for refinement of that axis.
    # A confidence-bound, sampling, or method-equivalence precision deficit
    # calls for fresh complete-process seeds.  Increasing n_m is never treated
    # as a substitute for independent replication.
    grid_need = grid_point_ratio > 1
    root_need = root_point_ratio > 1
    joint_need = interaction_point_ratio > 1 or total_point_ratio > total_budget_multiplier
    replication_need = (
        not method_pass
        or (grid_ratio > 1 and not grid_need)
        or (root_ratio > 1 and not root_need)
        or (interaction_ratio > 1 and interaction_point_ratio <= 1)
        or sampling_ratio > 1
        or (total_ratio > total_budget_multiplier and not joint_need)
    )
    axis_action = (
        "joint"
        if joint_need or (grid_need and root_need)
        else ("grid" if grid_need else ("root" if root_need else None))
    )
    required_extension = max(0, required_power_runs - pilot_count)
    if replication_need and required_extension == 0:
        # The fixed-critical-value planning calculation should normally make
        # this unreachable, but fail safe by requiring one fresh replicate.
        required_extension = 1
    if replication_need and required_extension > extension_count:
        reasons.append("insufficient_preregistered_extension_seeds")
        if axis_action is not None:
            reasons.append("extension_seed_pool_exhausted_diagnostic_axis_only_not_qualification")
            return f"diagnostic_{axis_action}", False, reasons
        return "insufficient_preregistered_seeds", False, reasons
    if required_power_runs > holdout_count and not replication_need:
        reasons.append("insufficient_preregistered_holdout_seeds")
        if axis_action is not None:
            reasons.append("holdout_seed_pool_exhausted_diagnostic_axis_only_not_qualification")
            return f"diagnostic_{axis_action}", False, reasons
        return "insufficient_preregistered_seeds", False, reasons
    if replication_need:
        if axis_action is None:
            return "replicate", False, reasons
        return f"{axis_action}_replicate", False, reasons
    if axis_action is not None:
        return axis_action, False, reasons
    # Multiple sub-threshold terms can still exhaust the summed budget. Their
    # attribution is not identifiable from this pilot, so refine both axes.
    return "joint", False, reasons or ["combined_budget_requires_joint_followup"]


def _apply_runtime_action_constraint(
    action: str,
    gate: bool,
    reasons: Sequence[str],
    action_counts: Mapping[str, int],
) -> tuple[str, bool, list[str]]:
    """Never let scalar endpoints erase the C++ controller's replay request."""
    result_reasons = list(reasons)
    if action in ("rejected_method_bias", "insufficient_preregistered_seeds"):
        return action, False, result_reasons
    has_replication = action == "replicate" or action.endswith("_replicate")
    base_action = _action_axis(action) or (
        action.removesuffix("_replicate") if has_replication else action
    )

    def with_axis(axis: str, *, force_replication: bool = False) -> str:
        current_axes = {
            "grid": {"grid"},
            "root": {"root"},
            "joint": {"grid", "root"},
            "hold": set(),
            "replicate": set(),
        }.get(base_action, set())
        required_axes = {"grid", "root"} if axis == "joint" else {axis}
        axes = current_axes.union(required_axes)
        resulting_base = "joint" if axes == {"grid", "root"} else next(iter(axes))
        return (
            f"{resulting_base}_replicate"
            if has_replication or force_replication
            else resulting_base
        )

    if action_counts.get("joint_replay_and_mesh_refinement", 0) > 0:
        result_reasons.append("runtime_controller_requested_joint_replay_and_mesh_refinement")
        return with_axis("joint", force_replication=True), False, result_reasons
    if action_counts.get("request_initial_state_replay", 0) > 0:
        result_reasons.append("runtime_controller_requested_fresh_initial_state_roots")
        return with_axis("root", force_replication=True), False, result_reasons
    return action, gate, result_reasons


def _apply_replication_seed_capacity(
    action: str,
    gate: bool,
    reasons: Sequence[str],
    required_total_complete_runs: int,
    pilot_count: int,
    extension_count: int,
) -> tuple[str, bool, list[str]]:
    """Fail closed when a post-controller replay cannot obtain fresh seeds."""
    result_reasons = list(reasons)
    if action not in REPLICATION_ACTIONS:
        return action, gate, result_reasons
    required_extension = max(1, required_total_complete_runs - pilot_count)
    if required_extension <= extension_count:
        return action, gate, result_reasons
    result_reasons.append("insufficient_preregistered_extension_seeds_after_runtime_constraint")
    axis = _action_axis(action)
    if axis is not None:
        result_reasons.append(
            "extension_seed_pool_exhausted_diagnostic_axis_only_not_qualification"
        )
        return f"diagnostic_{axis}", False, result_reasons
    return "insufficient_preregistered_seeds", False, result_reasons


def _runtime_action_summaries(
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Summarize authenticated V2 actions, with terminal-cell gating scope."""
    realizations = required_list(manifest.get("realizations"), "manifest.realizations")
    completed = required_list(state.get("completed"), "state.completed")
    if len(completed) != len(realizations):
        raise CampaignError("runtime action summary requires a complete pilot")
    final_grid_index = len(config["grids"]) - 1
    final_root_index = len(config["roots"]) - 1
    allowed_actions = {
        "request_initial_state_replay",
        "joint_replay_and_mesh_refinement",
    }
    accumulators: dict[str, dict[str, Any]] = {
        str(template["id"]): {
            "complete_run_count": 0,
            "all_counts": {},
            "terminal_complete_run_count": 0,
            "terminal_counts": {},
        }
        for template in config["templates"]
    }
    modes = {
        str(template["id"]): int(template["card_info"]["adaptive_macroparticle_mode"])
        for template in config["templates"]
    }

    for index, (raw_realization, raw_record) in enumerate(zip(realizations, completed)):
        realization = required_mapping(raw_realization, f"manifest.realizations[{index}]")
        record = required_mapping(raw_record, f"state.completed[{index}]")
        template_id = str(realization.get("template_id"))
        if template_id not in accumulators:
            raise CampaignError("runtime action summary references an unknown template")
        accumulator = accumulators[template_id]
        accumulator["complete_run_count"] += 1
        terminal = (
            _int_text(realization.get("grid_index"), "realization.grid_index") == final_grid_index
            and _int_text(realization.get("n_m_index"), "realization.n_m_index") == final_root_index
        )
        if terminal:
            accumulator["terminal_complete_run_count"] += 1
        evidence = required_mapping(
            record.get("runtime_replay_signal"),
            f"state.completed[{index}].runtime_replay_signal",
        )
        required = evidence.get("required")
        mode = modes[template_id]
        if mode == 0:
            if required is not False or evidence.get("present") is not False:
                raise CampaignError(
                    "nonadaptive completed record has adaptive runtime signal evidence"
                )
            continue
        if required is not True or evidence.get("present") is not True:
            raise CampaignError(
                "adaptive completed record lacks authenticated runtime signal evidence"
            )
        fields = required_mapping(
            evidence.get("parsed_fields"),
            f"state.completed[{index}].runtime_replay_signal.parsed_fields",
        )
        action = fields.get("recommended_action")
        if action not in allowed_actions:
            raise CampaignError("adaptive completed record has an invalid runtime action")
        accumulator["all_counts"][action] = accumulator["all_counts"].get(action, 0) + 1
        if terminal:
            accumulator["terminal_counts"][action] = (
                accumulator["terminal_counts"].get(action, 0) + 1
            )

    summaries: dict[str, dict[str, Any]] = {}
    expected_terminal_runs = len(config["pilot_seeds"])
    for template in config["templates"]:
        template_id = str(template["id"])
        mode = modes[template_id]
        accumulator = accumulators[template_id]
        all_signal_count = sum(accumulator["all_counts"].values())
        terminal_signal_count = sum(accumulator["terminal_counts"].values())
        if mode == 0:
            if all_signal_count != 0 or terminal_signal_count != 0:
                raise CampaignError("nonadaptive template accumulated runtime replay actions")
        else:
            if all_signal_count != accumulator["complete_run_count"]:
                raise CampaignError("adaptive template runtime signal count is incomplete")
            if (
                accumulator["terminal_complete_run_count"] != expected_terminal_runs
                or terminal_signal_count != expected_terminal_runs
            ):
                raise CampaignError(
                    "adaptive template terminal-cell runtime signal count is incomplete"
                )
        summaries[template_id] = {
            "adaptive_macroparticle_mode": str(mode),
            "complete_run_count": str(accumulator["complete_run_count"]),
            "all_cell_signal_count": str(all_signal_count),
            "all_cell_recommended_action_counts": {
                key: str(value) for key, value in sorted(accumulator["all_counts"].items())
            },
            "constraint_scope": "finest_frozen_grid_x_highest_frozen_n_m_across_pilot_seeds_v1",
            "terminal_grid_id": str(config["grids"][-1]["id"]),
            "terminal_n_m": str(config["roots"][-1]),
            "terminal_complete_run_count": str(accumulator["terminal_complete_run_count"]),
            "terminal_signal_count": str(terminal_signal_count),
            "terminal_recommended_action_counts": {
                key: str(value) for key, value in sorted(accumulator["terminal_counts"].items())
            },
        }
    return summaries


def _adaptive_grid_certificate_summaries(
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Aggregate hash-bound grid certificates without hiding coarse cells.

    All-cell counts remain visible, while only the finest grid at the highest
    root count is allowed to authorize a holdout.  Coarser factorial cells are
    deliberately present to measure convergence and need not themselves be
    terminal production choices.
    """
    realizations = required_list(manifest.get("realizations"), "manifest.realizations")
    completed = required_list(state.get("completed"), "state.completed")
    if len(realizations) != len(completed):
        raise CampaignError("grid-certificate summary requires a complete pilot")
    final_grid_index = len(config["grids"]) - 1
    final_root_index = len(config["roots"]) - 1
    summaries: dict[str, dict[str, Any]] = {}
    for template in config["templates"]:
        template_id = str(template["id"])
        grid_mode = int(template["card_info"]["adaptive_grid_mode"])
        summaries[template_id] = {
            "adaptive_grid_mode": str(grid_mode),
            "all_cell_complete_run_count": "0",
            "all_cell_certificate_pass_count": "0",
            "all_cell_certificate_fail_count": "0",
            "terminal_complete_run_count": "0",
            "terminal_certificate_pass_count": "0",
            "terminal_certificate_fail_count": "0",
            "terminal_patch_certification_requested": bool(
                template["card_info"]["adaptive_grid_terminal_patch_certification"]
            ),
            "terminal_patch_exercised_run_count": "0",
            "terminal_patch_unexercised_run_count": "0",
            "terminal_all_certified": True,
            "constraint_scope": ("finest_frozen_grid_x_highest_frozen_n_m_across_pilot_seeds_v1"),
            "terminal_grid_id": str(config["grids"][-1]["id"]),
            "terminal_n_m": str(config["roots"][-1]),
        }
    integer_keys = (
        "all_cell_complete_run_count",
        "all_cell_certificate_pass_count",
        "all_cell_certificate_fail_count",
        "terminal_complete_run_count",
        "terminal_certificate_pass_count",
        "terminal_certificate_fail_count",
        "terminal_patch_exercised_run_count",
        "terminal_patch_unexercised_run_count",
    )
    for index, (raw_realization, raw_record) in enumerate(zip(realizations, completed)):
        realization = required_mapping(raw_realization, f"manifest.realizations[{index}]")
        record = required_mapping(raw_record, f"state.completed[{index}]")
        template_id = str(realization.get("template_id"))
        if template_id not in summaries:
            raise CampaignError("grid certificate references an unknown template")
        summary = summaries[template_id]
        grid_mode = int(summary["adaptive_grid_mode"])
        evidence = required_mapping(
            record.get("adaptive_grid_certificate"),
            f"state.completed[{index}].adaptive_grid_certificate",
        )
        required = evidence.get("required")
        present = evidence.get("present")
        if grid_mode == 0:
            if required is not False or present is not False:
                raise CampaignError("fixed-grid record has adaptive-grid certificate evidence")
            passed = True
        else:
            if required is not True or present is not True:
                raise CampaignError("Mode-2/3 record lacks authenticated adaptive-grid certificate")
            passed = evidence.get("qualification_passed") is True
        terminal = (
            _int_text(realization.get("grid_index"), "realization.grid_index") == final_grid_index
            and _int_text(realization.get("n_m_index"), "realization.n_m_index") == final_root_index
        )
        counts = {key: int(summary[key]) for key in integer_keys}
        counts["all_cell_complete_run_count"] += 1
        counts[
            "all_cell_certificate_pass_count" if passed else "all_cell_certificate_fail_count"
        ] += 1
        if terminal:
            counts["terminal_complete_run_count"] += 1
            counts[
                "terminal_certificate_pass_count" if passed else "terminal_certificate_fail_count"
            ] += 1
            if summary["terminal_patch_certification_requested"]:
                if evidence.get("terminal_patch_qualification_exercised") is True:
                    counts["terminal_patch_exercised_run_count"] += 1
                else:
                    counts["terminal_patch_unexercised_run_count"] += 1
        for key, value in counts.items():
            summary[key] = str(value)
    expected_terminal = len(config["pilot_seeds"])
    for template in config["templates"]:
        summary = summaries[str(template["id"])]
        if int(summary["terminal_complete_run_count"]) != expected_terminal:
            raise CampaignError("adaptive-grid certificate terminal-cell run count is incomplete")
        summary["terminal_all_certified"] = (
            int(summary["terminal_certificate_fail_count"]) == 0
            and int(summary["terminal_certificate_pass_count"]) == expected_terminal
            and (
                not summary["terminal_patch_certification_requested"]
                or int(summary["terminal_patch_exercised_run_count"]) == expected_terminal
            )
        )
    return summaries


def _next_stage_run_specs(
    config: Mapping[str, Any],
    action: str,
    required_total_complete_runs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the canonical next-stage design before card materialization.

    Replication is a full paired frozen factorial on genuinely new seeds.  An
    axis extension uses the original common pilot seeds, plus the newly added
    seeds in a combined action, at every new cell needed to identify that
    axis.  This prevents either a larger ``n_m`` or extra cells from being
    counted as independent observations.
    """
    if action not in NEXT_STAGE_ACTIONS:
        raise CampaignError(f"cannot generate cards for recommendation {action!r}")
    required_total = max(MINIMUM_PILOT_SEEDS, required_total_complete_runs)
    pilot_seeds = list(config["pilot_seeds"])
    extension_seeds: list[int] = []
    if action in REPLICATION_ACTIONS:
        needed = max(1, required_total - len(pilot_seeds))
        if needed > len(config["extension_seeds"]):
            raise CampaignError("replication action exceeds the preregistered extension-seed pool")
        extension_seeds = list(config["extension_seeds"][:needed])

    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int, int]] = set()

    def add(
        template: Mapping[str, Any],
        grid: Mapping[str, Any],
        n_m: int,
        seed: int,
        population: str,
        role: str,
    ) -> None:
        identity = (
            str(template["id"]),
            int(grid["n_x"]),
            int(grid["n_y"]),
            int(n_m),
            int(seed),
        )
        if identity in seen:
            return
        seen.add(identity)
        specs.append(
            {
                "template": template,
                "grid": grid,
                "n_m": n_m,
                "seed": seed,
                "seed_population": population,
                "design_role": role,
            }
        )

    if action == "hold":
        needed = required_total
        if needed > len(config["holdout_seeds"]):
            raise CampaignError("hold action exceeds the preregistered holdout-seed pool")
        for template in config["templates"]:
            for seed in config["holdout_seeds"][:needed]:
                add(
                    template,
                    config["grids"][-1],
                    config["roots"][-1],
                    seed,
                    "reserved_holdout",
                    "confirmatory_terminal_pair",
                )
    elif action == "insufficient_preregistered_seeds":
        pass
    else:
        if action in REPLICATION_ACTIONS:
            for template in config["templates"]:
                for grid in config["grids"]:
                    for n_m in config["roots"]:
                        for seed in extension_seeds:
                            add(
                                template,
                                grid,
                                n_m,
                                seed,
                                "preregistered_extension",
                                "complete_paired_factorial_replication",
                            )
        axis = _action_axis(action)
        if axis in ("grid", "root", "joint"):
            fine_grid = config["grids"][-1]
            high_n_m = config["roots"][-1]
            cells: list[tuple[Mapping[str, Any], int, str]] = []
            if axis in ("grid", "joint"):
                cells.append((config["next_grid"], high_n_m, "new_grid_at_high_n_m"))
            if axis in ("root", "joint"):
                cells.append((fine_grid, config["next_n_m"], "fine_grid_at_new_n_m"))
            if axis == "joint":
                cells.append((config["next_grid"], config["next_n_m"], "new_grid_at_new_n_m"))
            axis_seeds = pilot_seeds + extension_seeds
            for template in config["templates"]:
                for grid, n_m, role in cells:
                    for seed in axis_seeds:
                        add(
                            template,
                            grid,
                            n_m,
                            seed,
                            (
                                "pilot_followup"
                                if seed in config["pilot_seeds"]
                                else "preregistered_extension"
                            ),
                            (
                                f"nonqualifying_diagnostic_{role}"
                                if action.startswith("diagnostic_")
                                else role
                            ),
                        )

    populations = sorted({str(item["seed_population"]) for item in specs})
    return specs, {
        "required_total_complete_runs": str(required_total),
        "required_total_complete_seed_blocks": str(required_total),
        "materialized_extension_seed_count": str(len(extension_seeds)),
        "materialized_extension_seed_block_count": str(len(extension_seeds)),
        "materialized_holdout_seed_count": str(required_total if action == "hold" else 0),
        "materialized_holdout_seed_block_count": str(required_total if action == "hold" else 0),
        "materialized_distinct_seed_block_count": str(len({int(item["seed"]) for item in specs})),
        "materialized_feebl_process_count": str(len(specs)),
        "counting_contract": (
            "one_seed_block_is_one_independent_common_seed_realized_by_one_fresh_"
            "FeeBL_process_per_template_cell; process_count_is_not_sample_size"
        ),
        "qualification_eligible": not action.startswith("diagnostic_"),
        "seed_populations": populations,
    }


def _build_next_stage(
    output_dir: Path,
    policy_path: Path,
    manifest_path: Path,
    analysis_path: Path,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    selected_template_id: str,
    action: str,
    pilot_gate_passed: bool,
    required_total_complete_runs: int,
) -> dict[str, Any]:
    if action == "hold":
        if not pilot_gate_passed:
            raise CampaignError("holdout cards cannot be generated before the pilot gate passes")
    elif pilot_gate_passed:
        raise CampaignError("a passed pilot gate must recommend the reserved holdout")

    specs, stage_design = _next_stage_run_specs(config, action, required_total_complete_runs)
    if len(specs) > int(config["limits"]["maximum_pilot_runs"]):
        raise CampaignError("materialized next stage exceeds maximum_pilot_runs")

    cards: list[tuple[Path, bytes]] = []
    runs: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        template = spec["template"]
        grid = spec["grid"]
        n_m = int(spec["n_m"])
        seed = int(spec["seed"])
        role = str(spec["design_role"])
        directory_text = (
            f"next_stage/runs/{index:05d}_{template['id']}_{action}_{role}_"
            f"nx_{grid['n_x']}_ny_{grid['n_y']}_nm_{n_m}_seed_{seed}"
        )
        relative_directory = relative_path(directory_text, "next_stage.run_directory")
        card = patch_card(
            template["raw"],
            template["accelerator"],
            template["parameters"],
            grid["n_x"],
            grid["n_y"],
            n_m,
            seed,
            template["grid_patch_contract"],
        )
        info = inspect_campaign_card(
            card,
            template["accelerator"],
            template["parameters"],
            template["grid_patch_contract"],
        )
        if info["campaign_family_sha256"] != template["family_sha256"]:
            raise CampaignError("next-stage card changed outside its declared patch contract")
        card_relative = relative_directory / "acc.dat"
        cards.append((card_relative, card))
        runs.append(
            {
                "ordinal": str(index),
                "seed_population": spec["seed_population"],
                "design_role": role,
                "seed": str(seed),
                "template_id": template["id"],
                "grid_id": str(grid["id"]),
                "n_x": str(grid["n_x"]),
                "n_y": str(grid["n_y"]),
                "n_m": str(n_m),
                "run_directory": relative_directory.as_posix(),
                "card": card_relative.as_posix(),
                "card_sha256": replay.sha256_bytes(card),
                "card_family_sha256": template["family_sha256"],
                "expected_output": "out.dat",
                "combined_log": "log.txt",
                "argv": [
                    config["binary_campaign_path"],
                    template["accelerator"],
                    template["parameters"],
                    "out.dat",
                ],
            }
        )

    next_root = output_dir / "next_stage"
    if next_root.exists():
        raise CampaignError(f"refusing to overwrite next-stage directory: {next_root}")
    next_root.mkdir()
    for relative_card, payload in cards:
        target = resolve_within(output_dir, relative_card, "next_stage.card")
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
    unsigned: dict[str, Any] = {
        "schema": NEXT_MANIFEST_SCHEMA,
        "schema_version": "1",
        "qualification_status": "unexecuted_next_stage_not_physics_qualified",
        "recommendation": action,
        "pilot_gate_passed": pilot_gate_passed,
        "holdout_execution_authorized": action == "hold" and pilot_gate_passed,
        "execution_authorized": action != "insufficient_preregistered_seeds",
        "execution_semantics": "strictly_sequential_fresh_complete_processes",
        "policy_file_sha256": sha256_file(policy_path),
        "policy_self_sha256": policy["self_sha256"],
        "pilot_manifest_file_sha256": sha256_file(manifest_path),
        "pilot_manifest_self_sha256": manifest["self_sha256"],
        "analysis_file_sha256": sha256_file(analysis_path),
        "selected_template_id": selected_template_id,
        "stage_design": stage_design,
        "runs": runs,
    }
    result = with_self_hash(unsigned)
    write_canonical(next_root / "next_stage_manifest.json", result)
    return result


def analyze_pilot(
    policy_path: Path, manifest_path: Path, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze the complete pilot and emit a bounded, executable next stage."""
    output_dir = output_dir.resolve()
    policy_path = policy_path.resolve()
    manifest_path = manifest_path.resolve()
    policy, manifest, config = validate_campaign(policy_path, manifest_path, output_dir)
    state_path = output_dir / "pilot_state.json"
    if not state_path.is_file():
        raise CampaignError("pilot_state.json is missing; no pilot analysis is allowed")
    state = validate_state(
        state_path, policy_path, manifest_path, output_dir, policy, manifest, config
    )
    if state.get("status") != "complete":
        raise CampaignError("pilot must be complete before analysis or holdout materialization")
    analysis_path = output_dir / "pilot_analysis.json"
    if analysis_path.exists() or (output_dir / "next_stage").exists():
        raise CampaignError("refusing to overwrite existing analysis or next-stage artifacts")

    data: dict[tuple[str, int, int, int, str], Decimal] = {}
    elapsed_by_template: dict[str, list[int]] = {item["id"]: [] for item in config["templates"]}
    elapsed_by_template_and_case: dict[str, dict[tuple[Any, ...], int]] = {
        item["id"]: {} for item in config["templates"]
    }
    for raw_realization, raw_record in zip(manifest["realizations"], state["completed"]):
        realization = required_mapping(raw_realization, "manifest.realization")
        record = required_mapping(raw_record, "state.completed")
        template_id = str(realization["template_id"])
        grid_index = _int_text(realization["grid_index"], "realization.grid_index")
        root_index = _int_text(realization["n_m_index"], "realization.n_m_index")
        seed_index = _int_text(realization["seed_index"], "realization.seed_index")
        values = required_mapping(record.get("endpoint_values"), "record.endpoint_values")
        for endpoint in config["endpoints"]:
            endpoint_id = str(endpoint["id"])
            data[(template_id, grid_index, root_index, seed_index, endpoint_id)] = decimal_value(
                values.get(endpoint_id), f"record.endpoint_values.{endpoint_id}"
            )
        elapsed = _int_text(
            record.get("elapsed_nanoseconds"), "record.elapsed_nanoseconds", 1, 10**18
        )
        elapsed_by_template[template_id].append(elapsed)
        elapsed_by_template_and_case[template_id][(grid_index, root_index, seed_index)] = elapsed

    runtime_action_summaries = _runtime_action_summaries(manifest, state, config)
    grid_certificate_summaries = _adaptive_grid_certificate_summaries(manifest, state, config)

    safety = decimal_value(
        config["statistics"]["standard_error_safety_factor"], "statistics.safety", positive=True
    )
    t_conf = decimal_value(
        config["statistics"]["two_sided_confidence_familywise_student_t_quantile"],
        "statistics.t_conf_familywise",
        positive=True,
    )
    t_alpha = decimal_value(
        config["statistics"]["one_sided_equivalence_familywise_student_t_quantile"],
        "statistics.t_alpha_familywise",
        positive=True,
    )
    z_alpha = decimal_value(
        config["statistics"]["one_sided_equivalence_familywise_normal_quantile_for_power"],
        "statistics.z_alpha_familywise_for_power",
        positive=True,
    )
    z_power = decimal_value(
        config["statistics"]["power_normal_quantile"], "statistics.z_power", positive=True
    )
    total_budget_multiplier = decimal_value(
        config["statistics"]["conservative_total_budget_multiplier"],
        "statistics.conservative_total_budget_multiplier",
        positive=True,
    )
    target_multiplier = safety * t_conf
    equivalence_multiplier = safety * t_alpha
    final_grid = len(config["grids"]) - 1
    previous_grid = final_grid - 1
    final_root = len(config["roots"]) - 1
    previous_root = final_root - 1
    seed_count = len(config["pilot_seeds"])
    reference_id = config["reference_template_id"]
    timing_summaries = _paired_timing_summaries(
        elapsed_by_template_and_case,
        str(reference_id),
        decimal_value(config["statistics"]["confidence_level"], "statistics.confidence"),
    )

    method_results: dict[str, dict[str, Any]] = {}
    for template in config["templates"]:
        template_id = template["id"]
        if template_id == reference_id:
            method_results[template_id] = {
                "reference_template_id": reference_id,
                "is_reference": True,
                "equal_cell_paired_shifts": [],
                "maximum_conservative_bound_over_margin": "0",
                "maximum_approximate_power_required_complete_runs": str(MINIMUM_PILOT_SEEDS),
                "maximum_approximate_combined_required_complete_runs": str(MINIMUM_PILOT_SEEDS),
                "maximum_absolute_paired_mean_over_margin": "0",
                "point_estimates_inside_all_frozen_equivalence_margins": True,
                "passes_all_frozen_equivalence_margins": True,
            }
            continue
        cells: list[dict[str, Any]] = []
        ratios: list[Decimal] = []
        point_ratios: list[Decimal] = []
        required_counts: list[int] = []
        for grid_index, grid in enumerate(config["grids"]):
            for root_index, n_m in enumerate(config["roots"]):
                for endpoint in config["endpoints"]:
                    endpoint_id = endpoint["id"]
                    differences = [
                        data[(template_id, grid_index, root_index, seed_index, endpoint_id)]
                        - data[(reference_id, grid_index, root_index, seed_index, endpoint_id)]
                        for seed_index in range(seed_count)
                    ]
                    summary = _equivalence_summary(
                        differences,
                        decimal_value(
                            endpoint["scale"], f"endpoint[{endpoint_id}].scale", positive=True
                        ),
                        decimal_value(
                            endpoint["equivalence_margin_relative"],
                            f"endpoint[{endpoint_id}].margin",
                            positive=True,
                        ),
                        decimal_value(
                            endpoint["expected_true_difference_relative"],
                            f"endpoint[{endpoint_id}].expected",
                        ),
                        equivalence_multiplier,
                        z_alpha,
                        z_power,
                    )
                    ratio = decimal_value(
                        summary["conservative_bound_over_margin"], "method.ratio", positive=False
                    )
                    ratios.append(ratio)
                    point_ratios.append(
                        decimal_value(
                            summary["absolute_paired_mean_over_margin"],
                            "method.point_ratio",
                        )
                    )
                    required_counts.append(
                        int(summary["approximate_combined_required_complete_runs"])
                    )
                    cells.append(
                        {
                            "grid_id": grid["id"],
                            "n_x": str(grid["n_x"]),
                            "n_y": str(grid["n_y"]),
                            "n_m": str(n_m),
                            "endpoint_id": endpoint_id,
                            "summary": summary,
                        }
                    )
        maximum_ratio = max(ratios)
        method_results[template_id] = {
            "reference_template_id": reference_id,
            "is_reference": False,
            "equal_cell_paired_shifts": cells,
            "maximum_conservative_bound_over_margin": decimal_text(maximum_ratio),
            "maximum_approximate_power_required_complete_runs": str(
                max(
                    int(cell["summary"]["approximate_power_required_complete_runs"])
                    for cell in cells
                )
            ),
            "maximum_approximate_combined_required_complete_runs": str(max(required_counts)),
            "maximum_absolute_paired_mean_over_margin": decimal_text(max(point_ratios)),
            "point_estimates_inside_all_frozen_equivalence_margins": max(point_ratios) <= 1,
            "passes_all_frozen_equivalence_margins": maximum_ratio <= 1,
        }

    template_results: list[dict[str, Any]] = []
    sortable: list[tuple[int, Decimal, int, int, str, dict[str, Any]]] = []
    holdout_count = len(config["holdout_seeds"])
    for template in config["templates"]:
        template_id = template["id"]
        endpoint_results: list[dict[str, Any]] = []
        grid_ratios: list[Decimal] = []
        root_ratios: list[Decimal] = []
        interaction_ratios: list[Decimal] = []
        sampling_ratios: list[Decimal] = []
        grid_point_ratios: list[Decimal] = []
        root_point_ratios: list[Decimal] = []
        interaction_point_ratios: list[Decimal] = []
        total_point_ratios: list[Decimal] = []
        power_counts: list[int] = [
            int(method_results[template_id]["maximum_approximate_combined_required_complete_runs"])
        ]
        total_ratios: list[Decimal] = []
        for endpoint in config["endpoints"]:
            endpoint_id = endpoint["id"]
            scale = decimal_value(
                endpoint["scale"], f"endpoint[{endpoint_id}].scale", positive=True
            )
            target_fraction = decimal_value(
                endpoint["target_relative_error"], f"endpoint[{endpoint_id}].target", positive=True
            )
            margin_fraction = decimal_value(
                endpoint["equivalence_margin_relative"],
                f"endpoint[{endpoint_id}].margin",
                positive=True,
            )
            expected_fraction = decimal_value(
                endpoint["expected_true_difference_relative"], f"endpoint[{endpoint_id}].expected"
            )
            grid_differences: list[Decimal] = []
            root_differences: list[Decimal] = []
            interactions: list[Decimal] = []
            terminal_values: list[Decimal] = []
            for seed_index in range(seed_count):
                fine_high = data[(template_id, final_grid, final_root, seed_index, endpoint_id)]
                coarse_high = data[
                    (template_id, previous_grid, final_root, seed_index, endpoint_id)
                ]
                fine_low = data[(template_id, final_grid, previous_root, seed_index, endpoint_id)]
                coarse_low = data[
                    (template_id, previous_grid, previous_root, seed_index, endpoint_id)
                ]
                grid_differences.append(fine_high - coarse_high)
                root_differences.append(fine_high - fine_low)
                interactions.append((fine_high - fine_low) - (coarse_high - coarse_low))
                terminal_values.append(fine_high)
            grid_summary = _contrast_summary(
                grid_differences, scale, target_fraction, target_multiplier
            )
            root_summary = _contrast_summary(
                root_differences, scale, target_fraction, target_multiplier
            )
            interaction_summary = _contrast_summary(
                interactions, scale, target_fraction, target_multiplier
            )
            sampling_summary = _sampling_summary(
                terminal_values, scale, target_fraction, target_multiplier
            )
            grid_equivalence = _equivalence_summary(
                grid_differences,
                scale,
                margin_fraction,
                expected_fraction,
                equivalence_multiplier,
                z_alpha,
                z_power,
            )
            root_equivalence = _equivalence_summary(
                root_differences,
                scale,
                margin_fraction,
                expected_fraction,
                equivalence_multiplier,
                z_alpha,
                z_power,
            )
            interaction_equivalence = _equivalence_summary(
                interactions,
                scale,
                margin_fraction,
                expected_fraction,
                equivalence_multiplier,
                z_alpha,
                z_power,
            )
            grid_summary["approximate_equivalence"] = grid_equivalence
            root_summary["approximate_equivalence"] = root_equivalence
            interaction_summary["approximate_equivalence"] = interaction_equivalence
            g_ratio = decimal_value(grid_summary["conservative_bound_over_target"], "grid.ratio")
            r_ratio = decimal_value(root_summary["conservative_bound_over_target"], "root.ratio")
            i_ratio = decimal_value(
                interaction_summary["conservative_bound_over_target"], "interaction.ratio"
            )
            s_ratio = decimal_value(
                sampling_summary["conservative_bound_over_target"], "sampling.ratio"
            )
            g_point_ratio = decimal_value(
                grid_summary["absolute_paired_mean_over_target"], "grid.point_ratio"
            )
            r_point_ratio = decimal_value(
                root_summary["absolute_paired_mean_over_target"], "root.point_ratio"
            )
            i_point_ratio = decimal_value(
                interaction_summary["absolute_paired_mean_over_target"],
                "interaction.point_ratio",
            )
            for summary, point_ratio in (
                (grid_summary, g_point_ratio),
                (root_summary, r_point_ratio),
                (interaction_summary, i_point_ratio),
            ):
                if point_ratio < 1:
                    power_counts.append(
                        int(summary["approximate_precision_required_complete_runs"])
                    )
            power_counts.append(
                int(sampling_summary["approximate_precision_required_complete_runs"])
            )
            for summary in (
                grid_equivalence,
                root_equivalence,
                interaction_equivalence,
            ):
                if (
                    decimal_value(
                        summary["absolute_paired_mean_over_margin"],
                        "numerical_equivalence.point_ratio",
                    )
                    < 1
                ):
                    power_counts.append(int(summary["approximate_combined_required_complete_runs"]))
            total_bound = sum(
                (
                    decimal_value(grid_summary["conservative_absolute_bound"], "grid.bound"),
                    decimal_value(root_summary["conservative_absolute_bound"], "root.bound"),
                    decimal_value(
                        interaction_summary["conservative_absolute_bound"], "interaction.bound"
                    ),
                    decimal_value(
                        sampling_summary["conservative_absolute_bound"], "sampling.bound"
                    ),
                ),
                Decimal(0),
            )
            target_absolute = scale * target_fraction
            total_ratio = total_bound / target_absolute
            total_point_ratio = (
                abs(decimal_value(grid_summary["paired_mean"], "grid.mean"))
                + abs(decimal_value(root_summary["paired_mean"], "root.mean"))
                + abs(decimal_value(interaction_summary["paired_mean"], "interaction.mean"))
            ) / target_absolute
            grid_ratios.append(g_ratio)
            root_ratios.append(r_ratio)
            interaction_ratios.append(i_ratio)
            sampling_ratios.append(s_ratio)
            grid_point_ratios.append(g_point_ratio)
            root_point_ratios.append(r_point_ratio)
            interaction_point_ratios.append(i_point_ratio)
            total_point_ratios.append(total_point_ratio)
            total_ratios.append(total_ratio)
            endpoint_results.append(
                {
                    "id": endpoint_id,
                    "units": endpoint["units"],
                    "terminal_grid_contrast": grid_summary,
                    "terminal_n_m_contrast": root_summary,
                    "grid_n_m_interaction": interaction_summary,
                    "terminal_independent_run_sampling": sampling_summary,
                    "summed_conservative_absolute_error": decimal_text(total_bound),
                    "summed_conservative_error_over_target": decimal_text(total_ratio),
                    "summed_absolute_point_estimates_over_target": decimal_text(total_point_ratio),
                    "allowed_summed_conservative_absolute_error": decimal_text(
                        target_absolute * total_budget_multiplier
                    ),
                    "summed_conservative_error_over_allowed_total_budget": decimal_text(
                        total_ratio / total_budget_multiplier
                    ),
                }
            )
        method_pass = bool(method_results[template_id]["passes_all_frozen_equivalence_margins"])
        method_point_pass = bool(
            method_results[template_id]["point_estimates_inside_all_frozen_equivalence_margins"]
        )
        required_total_complete_runs = max(power_counts)
        action, gate, reasons = _choose_action(
            max(grid_ratios),
            max(root_ratios),
            max(interaction_ratios),
            max(sampling_ratios),
            max(total_ratios),
            max(grid_point_ratios),
            max(root_point_ratios),
            max(interaction_point_ratios),
            max(total_point_ratios),
            total_budget_multiplier,
            method_pass,
            method_point_pass,
            required_total_complete_runs,
            seed_count,
            len(config["extension_seeds"]),
            holdout_count,
        )
        runtime_summary = runtime_action_summaries[template_id]
        terminal_action_counts = {
            key: _int_text(value, f"runtime_action_counts.{key}")
            for key, value in required_mapping(
                runtime_summary.get("terminal_recommended_action_counts"),
                "runtime_summary.terminal_recommended_action_counts",
            ).items()
        }
        action, gate, reasons = _apply_runtime_action_constraint(
            action, gate, reasons, terminal_action_counts
        )
        grid_certificate_summary = grid_certificate_summaries[template_id]
        if (
            int(grid_certificate_summary["adaptive_grid_mode"]) in (2, 3)
            and grid_certificate_summary["terminal_all_certified"] is not True
        ):
            reasons = list(reasons)
            if int(grid_certificate_summary["terminal_patch_unexercised_run_count"]) > 0:
                reasons.append("terminal_patch_certification_enabled_but_unexercised")
            reasons.append("terminal_adaptive_grid_internal_certificate_not_accepted")
            if action != "rejected_method_bias":
                if action != "insufficient_preregistered_seeds":
                    action = (
                        "diagnostic_joint"
                        if action.startswith("diagnostic_")
                        else (
                            "joint_replicate"
                            if action == "replicate" or action.endswith("_replicate")
                            else "joint"
                        )
                    )
            gate = False
        action, gate, reasons = _apply_replication_seed_capacity(
            action,
            gate,
            reasons,
            required_total_complete_runs,
            seed_count,
            len(config["extension_seeds"]),
        )
        elapsed_total = sum(elapsed_by_template[template_id])
        elapsed_mean = Decimal(elapsed_total) / Decimal(len(elapsed_by_template[template_id]))
        result: dict[str, Any] = {
            "template_id": template_id,
            "method_equivalence": method_results[template_id],
            "runtime_controller_signals": runtime_summary,
            "adaptive_grid_certificates": grid_certificate_summary,
            "endpoints": endpoint_results,
            "maximum_ratios": {
                "grid": decimal_text(max(grid_ratios)),
                "n_m": decimal_text(max(root_ratios)),
                "interaction": decimal_text(max(interaction_ratios)),
                "independent_sampling": decimal_text(max(sampling_ratios)),
                "summed_budget": decimal_text(max(total_ratios)),
                "summed_budget_allowed_multiplier": decimal_text(total_budget_multiplier),
                "grid_point_estimate": decimal_text(max(grid_point_ratios)),
                "n_m_point_estimate": decimal_text(max(root_point_ratios)),
                "interaction_point_estimate": decimal_text(max(interaction_point_ratios)),
                "summed_point_estimates": decimal_text(max(total_point_ratios)),
            },
            "approximate_required_complete_runs": str(required_total_complete_runs),
            "approximate_equivalence_power_required_complete_runs": str(
                required_total_complete_runs
            ),
            "required_extension_complete_runs": str(
                max(0, required_total_complete_runs - seed_count)
            ),
            "approximate_required_complete_seed_blocks": str(required_total_complete_runs),
            "pilot_complete_seed_block_count": str(seed_count),
            "required_extension_seed_blocks": str(
                max(0, required_total_complete_runs - seed_count)
            ),
            "seed_block_counting_contract": (
                "one independent common seed across all required template cells; "
                "not one FeeBL process"
            ),
            "reserved_holdout_complete_runs": str(holdout_count),
            "pilot_gate_passed": gate,
            "recommendation": action,
            "recommendation_reasons": reasons,
            "timing": {
                "complete_process_count": str(len(elapsed_by_template[template_id])),
                "total_elapsed_nanoseconds": str(elapsed_total),
                "mean_elapsed_seconds": decimal_text(elapsed_mean / Decimal(1_000_000_000)),
                "paired_reference_comparison": timing_summaries[template_id],
            },
        }
        template_results.append(result)
        tier = 0 if gate else (1 if method_pass else 2)
        convergence_score = max(
            max(grid_ratios),
            max(root_ratios),
            max(interaction_ratios),
            max(sampling_ratios),
            max(total_ratios) / total_budget_multiplier,
        )
        timing_status = timing_summaries[template_id]["comparison_to_reference"]
        timing_tier = (
            0
            if timing_status == "significantly_faster"
            else (2 if timing_status == "significantly_slower" else 1)
        )
        reference_tie_break = 0 if template_id == reference_id else 1
        sortable.append(
            (tier, convergence_score, timing_tier, reference_tie_break, template_id, result)
        )

    sortable.sort(key=lambda item: item[:-1])
    ranking = [
        {
            "rank": str(index),
            "template_id": item[4],
            "pilot_gate_passed": item[5]["pilot_gate_passed"],
            "method_equivalence_passed": item[5]["method_equivalence"][
                "passes_all_frozen_equivalence_margins"
            ],
            "recommendation": item[5]["recommendation"],
            "convergence_score": decimal_text(item[1]),
            "paired_timing_status": item[5]["timing"]["paired_reference_comparison"][
                "comparison_to_reference"
            ],
            "mean_elapsed_seconds_diagnostic_only": item[5]["timing"]["mean_elapsed_seconds"],
        }
        for index, item in enumerate(sortable, start=1)
    ]
    selected = sortable[0][5]
    if selected["recommendation"] == "rejected_method_bias":
        raise CampaignError("no method-equivalent reference or candidate remained for a next stage")
    unsigned_analysis: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": "1",
        "qualification_status": "pilot_decision_not_holdout_certification",
        "policy_file_sha256": sha256_file(policy_path),
        "policy_self_sha256": policy["self_sha256"],
        "pilot_manifest_file_sha256": sha256_file(manifest_path),
        "pilot_manifest_self_sha256": manifest["self_sha256"],
        "pilot_state_file_sha256": sha256_file(state_path),
        "pilot_state_self_sha256": state["self_sha256"],
        "statistics": {
            "target_bound_multiplier": decimal_text(target_multiplier),
            "equivalence_bound_multiplier": decimal_text(equivalence_multiplier),
            "conservative_total_budget_multiplier": decimal_text(total_budget_multiplier),
            "component_conservative_bounds_each_must_not_exceed_endpoint_target": True,
            "summed_conservative_bound_must_not_exceed_multiplier_times_endpoint_target": True,
            "paired_contrasts_use_common_complete_run_seeds": True,
            "independent_run_standard_error_uses_complete_runs": True,
            "equivalence_power_is_conservative_normal_TOST_planning_only": True,
            "small_sample_bounds_use_student_t": True,
            "familywise_error_control": config["statistics"]["familywise_error_control"],
            "numerical_family_statement_count": config["statistics"][
                "numerical_family_statement_count"
            ],
            "equivalence_family_statement_count": config["statistics"][
                "equivalence_family_statement_count"
            ],
            "pilot_degrees_of_freedom": config["statistics"]["pilot_degrees_of_freedom"],
        },
        "solution_verification_design": _solution_verification_design_status(config),
        "terminal_comparison": {
            "coarse_grid": config["grids"][previous_grid],
            "fine_grid": config["grids"][final_grid],
            "lower_n_m": str(config["roots"][previous_root]),
            "higher_n_m": str(config["roots"][final_root]),
        },
        "reference_template_id": reference_id,
        "template_results": template_results,
        "candidate_ranking_rule": (
            "physics_gate_then_method_equivalence_then_conservative_convergence_score; "
            "timing_breaks_a_convergence_tie_only_when_the_paired_confidence_interval_"
            "proves_a_difference; otherwise_the_reference_wins_the_stable_tie"
        ),
        "candidate_ranking": ranking,
        "selected_template_id": selected["template_id"],
        "recommendation": selected["recommendation"],
        "pilot_gate_passed": selected["pilot_gate_passed"],
    }
    analysis = with_self_hash(unsigned_analysis)
    write_canonical(analysis_path, analysis)
    next_manifest = _build_next_stage(
        output_dir,
        policy_path,
        manifest_path,
        analysis_path,
        policy,
        manifest,
        config,
        selected["template_id"],
        selected["recommendation"],
        bool(selected["pilot_gate_passed"]),
        int(selected["approximate_required_complete_runs"]),
    )
    validated_analysis, validated_next = validate_analysis_and_next(
        policy_path, manifest_path, output_dir, policy, manifest, config
    )
    if validated_analysis != analysis or validated_next != next_manifest:
        raise CampaignError(
            "internal error: emitted decision artifacts failed round-trip validation"
        )
    return analysis, next_manifest


def validate_analysis_and_next(
    policy_path: Path,
    manifest_path: Path,
    output_dir: Path,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the decision linkage and every materialized next-stage card."""
    analysis_path = output_dir / "pilot_analysis.json"
    next_path = output_dir / "next_stage" / "next_stage_manifest.json"
    analysis = validate_self_hash(analysis_path, ANALYSIS_SCHEMA)
    next_manifest = validate_self_hash(next_path, NEXT_MANIFEST_SCHEMA)
    state_path = output_dir / "pilot_state.json"
    state = validate_state(
        state_path,
        policy_path,
        manifest_path,
        output_dir,
        policy,
        manifest,
        config,
    )
    if state.get("status") != "complete":
        raise CampaignError("analysis requires a complete authenticated pilot state")
    expected_links = {
        "policy_file_sha256": sha256_file(policy_path),
        "policy_self_sha256": policy["self_sha256"],
        "pilot_manifest_file_sha256": sha256_file(manifest_path),
        "pilot_manifest_self_sha256": manifest["self_sha256"],
    }
    for key, expected in expected_links.items():
        if analysis.get(key) != expected or next_manifest.get(key) != expected:
            raise CampaignError(f"analysis/next-stage linkage mismatch for {key}")
    if analysis.get("pilot_state_file_sha256") != sha256_file(state_path):
        raise CampaignError("analysis does not bind the completed pilot state")
    if analysis.get("pilot_state_self_sha256") != state.get("self_sha256"):
        raise CampaignError("analysis/pilot-state self-hash mismatch")
    if next_manifest.get("analysis_file_sha256") != sha256_file(analysis_path):
        raise CampaignError("next-stage manifest does not bind the analysis bytes")
    action = analysis.get("recommendation")
    selected_id = analysis.get("selected_template_id")
    gate = analysis.get("pilot_gate_passed")
    if action not in NEXT_STAGE_ACTIONS:
        raise CampaignError("analysis has no recognized next-stage recommendation")
    if not isinstance(selected_id, str) or selected_id not in {
        item["id"] for item in config["templates"]
    }:
        raise CampaignError("analysis selected template is not frozen")
    if not isinstance(gate, bool):
        raise CampaignError("analysis pilot gate is not Boolean")
    if next_manifest.get("recommendation") != action:
        raise CampaignError("analysis/next-stage recommendation mismatch")
    if next_manifest.get("selected_template_id") != selected_id:
        raise CampaignError("analysis/next-stage selected-template mismatch")
    if next_manifest.get("pilot_gate_passed") is not gate:
        raise CampaignError("analysis/next-stage pilot-gate mismatch")
    if (action == "hold") != gate:
        raise CampaignError("hold recommendation and pilot gate are inconsistent")
    if next_manifest.get("holdout_execution_authorized") is not (action == "hold" and gate):
        raise CampaignError("holdout execution authorization is inconsistent")
    if next_manifest.get("execution_authorized") is not (
        action != "insufficient_preregistered_seeds"
    ):
        raise CampaignError("next-stage execution authorization is inconsistent")

    expected_runtime_summaries = _runtime_action_summaries(manifest, state, config)
    expected_grid_summaries = _adaptive_grid_certificate_summaries(manifest, state, config)
    selected_result = None
    seen_result_ids: set[str] = set()
    for raw_result in required_list(analysis.get("template_results"), "analysis.template_results"):
        result = required_mapping(raw_result, "analysis.template_result")
        result_id = result.get("template_id")
        if not isinstance(result_id, str) or result_id not in expected_runtime_summaries:
            raise CampaignError("analysis contains a result for an unknown template")
        if result_id in seen_result_ids:
            raise CampaignError("analysis duplicates a template result")
        seen_result_ids.add(result_id)
        if result.get("runtime_controller_signals") != expected_runtime_summaries[result_id]:
            raise CampaignError("analysis runtime-controller summary changed")
        if result.get("adaptive_grid_certificates") != expected_grid_summaries[result_id]:
            raise CampaignError("analysis adaptive-grid certificate summary changed")
        if result_id == selected_id:
            if selected_result is not None:
                raise CampaignError("analysis duplicates its selected template")
            selected_result = result
    if seen_result_ids != set(expected_runtime_summaries):
        raise CampaignError("analysis does not contain exactly one result per frozen template")
    if selected_result is None:
        raise CampaignError("analysis omits its selected template result")
    method = required_mapping(
        selected_result.get("method_equivalence"), "selected.method_equivalence"
    )
    if method.get("passes_all_frozen_equivalence_margins") is not True:
        raise CampaignError("a method-biased template was selected")
    if (
        selected_result.get("recommendation") != action
        or selected_result.get("pilot_gate_passed") is not gate
    ):
        raise CampaignError("selected result and global recommendation are inconsistent")

    ratios = required_mapping(selected_result.get("maximum_ratios"), "selected.maximum_ratios")
    total_budget_multiplier = decimal_value(
        config["statistics"]["conservative_total_budget_multiplier"],
        "statistics.conservative_total_budget_multiplier",
        positive=True,
    )
    if ratios.get("summed_budget_allowed_multiplier") != decimal_text(total_budget_multiplier):
        raise CampaignError("analysis total-budget multiplier differs from the frozen request")
    expected_action, expected_gate, expected_reasons = _choose_action(
        decimal_value(ratios.get("grid"), "selected.maximum_ratios.grid"),
        decimal_value(ratios.get("n_m"), "selected.maximum_ratios.n_m"),
        decimal_value(ratios.get("interaction"), "selected.maximum_ratios.interaction"),
        decimal_value(
            ratios.get("independent_sampling"),
            "selected.maximum_ratios.independent_sampling",
        ),
        decimal_value(ratios.get("summed_budget"), "selected.maximum_ratios.summed_budget"),
        decimal_value(
            ratios.get("grid_point_estimate"),
            "selected.maximum_ratios.grid_point_estimate",
        ),
        decimal_value(
            ratios.get("n_m_point_estimate"),
            "selected.maximum_ratios.n_m_point_estimate",
        ),
        decimal_value(
            ratios.get("interaction_point_estimate"),
            "selected.maximum_ratios.interaction_point_estimate",
        ),
        decimal_value(
            ratios.get("summed_point_estimates"),
            "selected.maximum_ratios.summed_point_estimates",
        ),
        total_budget_multiplier,
        True,
        True,
        _int_text(
            selected_result.get("approximate_required_complete_runs"),
            "selected.approximate_required_complete_runs",
            MINIMUM_PILOT_SEEDS,
        ),
        len(config["pilot_seeds"]),
        len(config["extension_seeds"]),
        len(config["holdout_seeds"]),
    )
    terminal_counts = {
        key: _int_text(value, f"selected.runtime_terminal_action_counts.{key}")
        for key, value in required_mapping(
            expected_runtime_summaries[selected_id].get("terminal_recommended_action_counts"),
            "selected.runtime_terminal_action_counts",
        ).items()
    }
    expected_action, expected_gate, expected_reasons = _apply_runtime_action_constraint(
        expected_action, expected_gate, expected_reasons, terminal_counts
    )
    selected_grid_summary = expected_grid_summaries[selected_id]
    if (
        int(selected_grid_summary["adaptive_grid_mode"]) in (2, 3)
        and selected_grid_summary["terminal_all_certified"] is not True
    ):
        expected_reasons = list(expected_reasons)
        if int(selected_grid_summary["terminal_patch_unexercised_run_count"]) > 0:
            expected_reasons.append("terminal_patch_certification_enabled_but_unexercised")
        expected_reasons.append("terminal_adaptive_grid_internal_certificate_not_accepted")
        if expected_action not in (
            "rejected_method_bias",
            "insufficient_preregistered_seeds",
        ):
            expected_action = (
                "diagnostic_joint"
                if expected_action.startswith("diagnostic_")
                else (
                    "joint_replicate"
                    if expected_action == "replicate" or expected_action.endswith("_replicate")
                    else "joint"
                )
            )
        expected_gate = False
    expected_action, expected_gate, expected_reasons = _apply_replication_seed_capacity(
        expected_action,
        expected_gate,
        expected_reasons,
        _int_text(
            selected_result.get("approximate_required_complete_runs"),
            "selected.approximate_required_complete_runs",
            MINIMUM_PILOT_SEEDS,
        ),
        len(config["pilot_seeds"]),
        len(config["extension_seeds"]),
    )
    if (
        action != expected_action
        or gate is not expected_gate
        or selected_result.get("recommendation_reasons") != expected_reasons
    ):
        raise CampaignError("selected action does not honor frozen scalar and runtime gates")

    required_total = _int_text(
        selected_result.get("approximate_required_complete_runs"),
        "selected.approximate_required_complete_runs",
        MINIMUM_PILOT_SEEDS,
    )
    specs, expected_stage_design = _next_stage_run_specs(config, action, required_total)
    if next_manifest.get("stage_design") != expected_stage_design:
        raise CampaignError("next-stage design metadata is inconsistent")
    runs = required_list(next_manifest.get("runs"), "next_stage.runs")
    if len(runs) != len(specs):
        raise CampaignError("next-stage run count does not match its frozen design")
    for index, (raw_run, spec) in enumerate(zip(runs, specs), start=1):
        where = f"next_stage.runs[{index - 1}]"
        run = required_mapping(raw_run, where)
        template = spec["template"]
        grid = spec["grid"]
        n_m = int(spec["n_m"])
        seed = int(spec["seed"])
        role = str(spec["design_role"])
        directory_text = (
            f"next_stage/runs/{index:05d}_{template['id']}_{action}_{role}_"
            f"nx_{grid['n_x']}_ny_{grid['n_y']}_nm_{n_m}_seed_{seed}"
        )
        expected = {
            "ordinal": str(index),
            "seed_population": spec["seed_population"],
            "design_role": role,
            "seed": str(seed),
            "template_id": template["id"],
            "grid_id": str(grid["id"]),
            "n_x": str(grid["n_x"]),
            "n_y": str(grid["n_y"]),
            "n_m": str(n_m),
            "run_directory": directory_text,
            "card": f"{directory_text}/acc.dat",
            "card_family_sha256": template["family_sha256"],
            "expected_output": "out.dat",
            "combined_log": "log.txt",
        }
        for key, value in expected.items():
            if run.get(key) != value:
                raise CampaignError(f"{where}.{key}: expected canonical value {value!r}")
        run_directory = resolve_within(
            output_dir,
            relative_path(run.get("run_directory"), f"{where}.run_directory"),
            f"{where}.run_directory",
        )
        card_path = resolve_within(
            output_dir,
            relative_path(run.get("card"), f"{where}.card"),
            f"{where}.card",
        )
        if card_path.parent != run_directory:
            raise CampaignError(f"{where}: card is outside its run directory")
        checked_file(card_path, run.get("card_sha256"), f"{where}.card")
        info = inspect_campaign_card(
            card_path.read_bytes(),
            template["accelerator"],
            template["parameters"],
            template["grid_patch_contract"],
        )
        if (info["n_x"], info["n_y"], info["n_m"], info["rndm_seed"]) != (
            grid["n_x"],
            grid["n_y"],
            n_m,
            seed,
        ):
            raise CampaignError(f"{where}: card/design mismatch")
        if info["campaign_family_sha256"] != template["family_sha256"]:
            raise CampaignError(f"{where}: card family mismatch")
        expected_argv = [
            config["binary_campaign_path"],
            template["accelerator"],
            template["parameters"],
            "out.dat",
        ]
        if run.get("argv") != expected_argv:
            raise CampaignError(f"{where}: executable argv mismatch")
    return analysis, next_manifest


def _next_state_unsigned(
    analysis_path: Path,
    next_manifest_path: Path,
    next_manifest: Mapping[str, Any],
    status: str,
    completed: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": NEXT_STATE_SCHEMA,
        "schema_version": "1",
        "qualification_status": "next_stage_execution_evidence_not_physics_qualified",
        "execution_semantics": "strictly_sequential_fresh_complete_processes",
        "analysis_file_sha256": sha256_file(analysis_path),
        "next_manifest_file_sha256": sha256_file(next_manifest_path),
        "next_manifest_self_sha256": next_manifest["self_sha256"],
        "status": status,
        "next_ordinal": str(len(completed) + 1),
        "completed": list(completed),
    }
    if failure is not None:
        result["failure"] = dict(failure)
    return result


def validate_next_state(
    state_path: Path,
    analysis_path: Path,
    next_manifest_path: Path,
    output_dir: Path,
    next_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    state = validate_self_hash(state_path, NEXT_STATE_SCHEMA)
    if state.get("analysis_file_sha256") != sha256_file(analysis_path):
        raise CampaignError("next-stage state/analysis hash mismatch")
    if state.get("next_manifest_file_sha256") != sha256_file(next_manifest_path):
        raise CampaignError("next-stage state/manifest hash mismatch")
    if state.get("next_manifest_self_sha256") != next_manifest.get("self_sha256"):
        raise CampaignError("next-stage state/manifest self-hash mismatch")
    status_value = state.get("status")
    if status_value not in ("running", "complete", "failed"):
        raise CampaignError("next-stage state has invalid status")
    runs = required_list(next_manifest.get("runs"), "next_manifest.runs")
    completed = required_list(state.get("completed"), "next_state.completed")
    if len(completed) > len(runs):
        raise CampaignError("next-stage completed prefix exceeds its manifest")
    if _int_text(state.get("next_ordinal"), "next_state.next_ordinal", 1) != len(completed) + 1:
        raise CampaignError("next-stage next_ordinal is inconsistent")
    if status_value == "complete" and len(completed) != len(runs):
        raise CampaignError("complete next-stage state does not cover every run")
    if status_value == "running" and len(completed) == len(runs):
        raise CampaignError("running next-stage state already covers every run")
    if status_value == "failed" and not isinstance(state.get("failure"), dict):
        raise CampaignError("failed next-stage state lacks failure evidence")
    if status_value != "failed" and "failure" in state:
        raise CampaignError("nonfailed next-stage state contains failure evidence")
    if status_value == "failed":
        failed_index = len(completed)
        if failed_index >= len(runs):
            raise CampaignError("failed next-stage state has no corresponding run")
        failed_run = required_mapping(runs[failed_index], f"next_manifest.runs[{failed_index}]")
        _, _, failed_log_path = _run_paths(
            output_dir, failed_run, f"next_manifest.runs[{failed_index}]"
        )
        _validate_failed_execution_record(
            required_mapping(state.get("failure"), "next_state.failure"),
            failed_index,
            failed_run,
            failed_log_path,
            config,
            "next_state.failure",
        )
    for index, raw_run in enumerate(runs):
        where = f"next_manifest.runs[{index}]"
        run = required_mapping(raw_run, where)
        run_directory, card_path, log_path = _run_paths(output_dir, run, where)
        if index >= len(completed):
            extras = sorted(
                item.name for item in run_directory.iterdir() if item.name != card_path.name
            )
            if extras and not (status_value == "failed" and index == len(completed)):
                raise CampaignError(
                    f"ambiguous unrecorded next-stage artifacts block safe resume in {run_directory}: {extras}"
                )
            continue
        record = required_mapping(completed[index], f"next_state.completed[{index}]")
        if record.get("ordinal") != str(index + 1):
            raise CampaignError(f"next_state.completed[{index}]: noncanonical ordinal")
        if record.get("realization_sha256") != sha256_object(run):
            raise CampaignError(f"next_state.completed[{index}]: run hash mismatch")
        if record.get("exit_code") != "0":
            raise CampaignError(
                f"next_state.completed[{index}]: successful prefix has nonzero exit"
            )
        _int_text(record.get("elapsed_nanoseconds"), "next_state.elapsed_nanoseconds", 1, 10**18)
        checked_file(
            log_path,
            record.get("combined_log_sha256"),
            f"next_state.completed[{index}].log",
            nonempty=False,
        )
        signal_evidence = _runtime_signal_evidence(log_path, run, config, expected_exit_code=0)
        if record.get("runtime_replay_signal") != signal_evidence:
            raise CampaignError(
                f"next_state.completed[{index}]: runtime replay signal evidence changed"
            )
        grid_certificate = _adaptive_grid_certificate_evidence(log_path, run, config)
        if record.get("adaptive_grid_certificate") != grid_certificate:
            raise CampaignError(
                f"next_state.completed[{index}]: adaptive-grid certificate evidence changed"
            )
        values, hashes = _complete_output_evidence(run_directory, run, config["endpoints"], config)
        if record.get("endpoint_values") != values:
            raise CampaignError(f"next_state.completed[{index}]: endpoint values changed")
        if record.get("output_file_sha256") != hashes:
            raise CampaignError(f"next_state.completed[{index}]: output hashes changed")
    return state


def execute_next_stage(
    policy_path: Path,
    manifest_path: Path,
    output_dir: Path,
    confirm_analysis_sha256: str,
    *,
    allow_holdout: bool,
) -> int:
    """Deliberately execute an analyzed next stage; holdout needs extra consent."""
    output_dir = output_dir.resolve()
    policy_path = policy_path.resolve()
    manifest_path = manifest_path.resolve()
    policy, manifest, config = validate_campaign(policy_path, manifest_path, output_dir)
    pilot_state_path = output_dir / "pilot_state.json"
    pilot_state = validate_state(
        pilot_state_path,
        policy_path,
        manifest_path,
        output_dir,
        policy,
        manifest,
        config,
    )
    if pilot_state.get("status") != "complete":
        raise CampaignError("next-stage execution requires a complete validated pilot")
    analysis, next_manifest = validate_analysis_and_next(
        policy_path, manifest_path, output_dir, policy, manifest, config
    )
    analysis_path = output_dir / "pilot_analysis.json"
    declared_confirmation = checked_hash(confirm_analysis_sha256, "confirm_analysis_sha256")
    if declared_confirmation != sha256_file(analysis_path):
        raise CampaignError("explicit analysis SHA-256 confirmation does not match")
    if next_manifest.get("execution_authorized") is not True:
        raise CampaignError("next-stage execution is blocked: insufficient preregistered seeds")
    stage_design = required_mapping(next_manifest.get("stage_design"), "next_manifest.stage_design")
    populations = required_list(
        stage_design.get("seed_populations"), "next_manifest.stage_design.seed_populations"
    )
    is_holdout = populations == ["reserved_holdout"]
    if is_holdout:
        if (
            analysis.get("pilot_gate_passed") is not True
            or next_manifest.get("holdout_execution_authorized") is not True
        ):
            raise CampaignError("holdout execution is blocked because the pilot gate did not pass")
        if not allow_holdout:
            raise CampaignError("holdout execution requires the explicit --allow-holdout flag")
    elif allow_holdout:
        raise CampaignError("--allow-holdout was supplied for a nonholdout follow-up")

    binary = Path(str(config["binary_path"]))
    if os.name != "nt" and not os.access(binary, os.X_OK):
        raise CampaignError(f"bound FeeBL binary is not executable: {binary}")
    next_manifest_path = output_dir / "next_stage" / "next_stage_manifest.json"
    state_path = output_dir / "next_stage" / "next_stage_state.json"
    if state_path.exists():
        state = validate_next_state(
            state_path,
            analysis_path,
            next_manifest_path,
            output_dir,
            next_manifest,
            config,
        )
        if state["status"] == "complete":
            return 0
        if state["status"] == "failed":
            raise CampaignError("next-stage state is failed; automatic retry is forbidden")
        completed: list[dict[str, Any]] = list(state["completed"])
    else:
        for index, raw_run in enumerate(next_manifest["runs"]):
            run = required_mapping(raw_run, f"next_manifest.runs[{index}]")
            run_directory, card_path, _ = _run_paths(
                output_dir, run, f"next_manifest.runs[{index}]"
            )
            extras = sorted(
                item.name for item in run_directory.iterdir() if item.name != card_path.name
            )
            if extras:
                raise CampaignError(
                    f"unrecorded artifacts block next-stage execution in {run_directory}: {extras}"
                )
        completed = []
        initial = with_self_hash(
            _next_state_unsigned(
                analysis_path,
                next_manifest_path,
                next_manifest,
                "running",
                completed,
            )
        )
        _atomic_write(state_path, initial)

    maximum_seconds = int(config["limits"]["maximum_wall_seconds_per_run"])
    runs = next_manifest["runs"]
    for index in range(len(completed), len(runs)):
        where = f"next_manifest.runs[{index}]"
        run = required_mapping(runs[index], where)
        run_directory, _, log_path = _run_paths(output_dir, run, where)
        if log_path.exists():
            raise CampaignError(f"refusing to overwrite next-stage log for {where}")
        argv = run.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise CampaignError(f"{where}: invalid argv")
        launch_argv = [str(binary), *argv[1:]]
        start_ns = time.monotonic_ns()
        exit_code: int
        failure_kind: str | None = None
        failure_detail: str | None = None
        try:
            with log_path.open("xb") as log_handle:
                try:
                    process = subprocess.run(
                        launch_argv,
                        cwd=run_directory,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        timeout=maximum_seconds,
                        check=False,
                        shell=False,
                    )
                    exit_code = process.returncode
                    if exit_code != 0:
                        if exit_code == replay.REPLAY_REQUIRED_EXIT_CODE:
                            failure_kind = "runtime_replay_required"
                            failure_detail = (
                                "strict adaptive-root gate requested fresh complete-process replay"
                            )
                        else:
                            failure_kind = "nonzero_exit"
                            failure_detail = f"FeeBL returned {exit_code}"
                except subprocess.TimeoutExpired:
                    exit_code = 124
                    failure_kind = "timeout"
                    failure_detail = f"exceeded {maximum_seconds} seconds"
        except OSError as exc:
            exit_code = 126
            failure_kind = "launch_error"
            failure_detail = str(exc)
        output_error: CampaignError | None = None
        if failure_kind is None and exit_code == 0:
            try:
                _run_gaussian_core_transform(
                    run_directory,
                    log_path,
                    run,
                    config,
                    start_ns + maximum_seconds * 1_000_000_000,
                )
            except CampaignError as exc:
                output_error = exc
        elapsed_ns = max(1, time.monotonic_ns() - start_ns)
        runtime_signal: dict[str, Any] | None = None
        grid_certificate: dict[str, Any] | None = None
        if exit_code in (0, replay.REPLAY_REQUIRED_EXIT_CODE) and log_path.is_file():
            try:
                runtime_signal = _runtime_signal_evidence(
                    log_path,
                    run,
                    config,
                    expected_exit_code=exit_code,
                )
                if (
                    exit_code == replay.REPLAY_REQUIRED_EXIT_CODE
                    and runtime_signal.get("required") is not True
                ):
                    raise CampaignError("strict replay-required exit came from a nonadaptive card")
            except CampaignError as exc:
                failure_kind = "invalid_runtime_signal"
                failure_detail = str(exc)
        if failure_kind is None and exit_code == 0 and log_path.is_file():
            try:
                grid_certificate = _adaptive_grid_certificate_evidence(log_path, run, config)
            except CampaignError as exc:
                failure_kind = "invalid_grid_certificate"
                failure_detail = str(exc)
        if failure_kind is not None:
            if not log_path.is_file():
                try:
                    with log_path.open("xb") as log_handle:
                        log_handle.write(
                            ("campaign launch failure: " + str(failure_detail) + "\n").encode(
                                "utf-8"
                            )
                        )
                except OSError as exc:
                    raise CampaignError(
                        f"cannot materialize authenticated failure log {log_path}: {exc}"
                    ) from exc
            failure: dict[str, Any] = {
                "ordinal": str(index + 1),
                "kind": failure_kind,
                "detail": failure_detail,
                "realization_sha256": sha256_object(run),
                "exit_code": str(exit_code),
                "elapsed_nanoseconds": str(elapsed_ns),
            }
            if log_path.is_file():
                failure["combined_log_sha256"] = sha256_file(log_path)
            if runtime_signal is not None:
                failure["runtime_replay_signal"] = runtime_signal
            failed = with_self_hash(
                _next_state_unsigned(
                    analysis_path,
                    next_manifest_path,
                    next_manifest,
                    "failed",
                    completed,
                    failure,
                )
            )
            _atomic_replace(state_path, failed)
            return _execution_failure_return_code(exit_code, failure_kind, runtime_signal)
        try:
            if output_error is not None:
                raise output_error
            values, output_hashes = _complete_output_evidence(
                run_directory, run, config["endpoints"], config
            )
        except CampaignError as exc:
            failure = {
                "ordinal": str(index + 1),
                "kind": "invalid_output",
                "detail": str(exc),
                "realization_sha256": sha256_object(run),
                "exit_code": "0",
                "elapsed_nanoseconds": str(elapsed_ns),
                "combined_log_sha256": sha256_file(log_path),
            }
            failed = with_self_hash(
                _next_state_unsigned(
                    analysis_path,
                    next_manifest_path,
                    next_manifest,
                    "failed",
                    completed,
                    failure,
                )
            )
            _atomic_replace(state_path, failed)
            raise
        completed.append(
            {
                "ordinal": str(index + 1),
                "realization_sha256": sha256_object(run),
                "exit_code": "0",
                "elapsed_nanoseconds": str(elapsed_ns),
                "combined_log_sha256": sha256_file(log_path),
                "runtime_replay_signal": runtime_signal,
                "adaptive_grid_certificate": grid_certificate,
                "output_file_sha256": output_hashes,
                "endpoint_values": values,
            }
        )
        status_value = "complete" if len(completed) == len(runs) else "running"
        updated = with_self_hash(
            _next_state_unsigned(
                analysis_path,
                next_manifest_path,
                next_manifest,
                status_value,
                completed,
            )
        )
        _atomic_replace(state_path, updated)
    return 0


def _completed_observations(
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    next_manifest: Mapping[str, Any],
    next_state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return authenticated pilot and next-stage complete-process records.

    The caller must first run the existing state validators.  This routine
    additionally rejects duplicate physical identities across stages; a
    repeated output can therefore never masquerade as a fresh statistical
    root.
    """

    pilot: list[dict[str, Any]] = []
    followup: list[dict[str, Any]] = []
    identities: set[tuple[str, int, int, int, int]] = set()

    def add(
        target: list[dict[str, Any]],
        run: Mapping[str, Any],
        record: Mapping[str, Any],
        *,
        stage: str,
        grid_id: str,
        n_x: int,
        n_y: int,
        n_m: int,
        seed: int,
        seed_population: str,
    ) -> None:
        template_id = str(run.get("template_id"))
        identity = (template_id, n_x, n_y, n_m, seed)
        if identity in identities:
            raise CampaignError(
                "pilot and next stage contain a duplicate complete-process identity"
            )
        identities.add(identity)
        values = required_mapping(record.get("endpoint_values"), "record.endpoint_values")
        target.append(
            {
                "stage": stage,
                "template_id": template_id,
                "grid_id": grid_id,
                "n_x": n_x,
                "n_y": n_y,
                "n_m": n_m,
                "seed": seed,
                "seed_population": seed_population,
                "run_directory": str(run.get("run_directory")),
                "run": run,
                "record": record,
                "endpoint_values": values,
                "elapsed_nanoseconds": _int_text(
                    record.get("elapsed_nanoseconds"),
                    "record.elapsed_nanoseconds",
                    1,
                    10**18,
                ),
            }
        )

    pilot_runs = required_list(manifest.get("realizations"), "manifest.realizations")
    pilot_records = required_list(state.get("completed"), "state.completed")
    if len(pilot_runs) != len(pilot_records):
        raise CampaignError("pilot observations are incomplete")
    for index, (raw_run, raw_record) in enumerate(zip(pilot_runs, pilot_records)):
        run = required_mapping(raw_run, f"manifest.realizations[{index}]")
        record = required_mapping(raw_record, f"state.completed[{index}]")
        grid_index = _int_text(run.get("grid_index"), "realization.grid_index")
        root_index = _int_text(run.get("n_m_index"), "realization.n_m_index")
        seed_index = _int_text(run.get("seed_index"), "realization.seed_index")
        grid = config["grids"][grid_index]
        add(
            pilot,
            run,
            record,
            stage="pilot",
            grid_id=str(grid["id"]),
            n_x=int(grid["n_x"]),
            n_y=int(grid["n_y"]),
            n_m=int(config["roots"][root_index]),
            seed=int(config["pilot_seeds"][seed_index]),
            seed_population="pilot",
        )

    next_runs = required_list(next_manifest.get("runs"), "next_manifest.runs")
    next_records = required_list(next_state.get("completed"), "next_state.completed")
    if len(next_runs) != len(next_records):
        raise CampaignError("next-stage observations are incomplete")
    for index, (raw_run, raw_record) in enumerate(zip(next_runs, next_records)):
        run = required_mapping(raw_run, f"next_manifest.runs[{index}]")
        record = required_mapping(raw_record, f"next_state.completed[{index}]")
        add(
            followup,
            run,
            record,
            stage="next_stage",
            grid_id=str(run.get("grid_id")),
            n_x=_int_text(run.get("n_x"), "next_run.n_x", 1),
            n_y=_int_text(run.get("n_y"), "next_run.n_y", 1),
            n_m=_int_text(run.get("n_m"), "next_run.n_m", 1),
            seed=_int_text(run.get("seed"), "next_run.seed", 0),
            seed_population=str(run.get("seed_population")),
        )
    return pilot, followup


def _observation_cell(item: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(item["n_x"]), int(item["n_y"]), int(item["n_m"])


def _endpoint_lookup(
    observations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[tuple[str, tuple[int, int, int], int, str], Decimal]:
    result: dict[tuple[str, tuple[int, int, int], int, str], Decimal] = {}
    endpoint_ids = [str(item["id"]) for item in config["endpoints"]]
    for observation in observations:
        values = required_mapping(observation.get("endpoint_values"), "observation.endpoint_values")
        for endpoint_id in endpoint_ids:
            key = (
                str(observation["template_id"]),
                _observation_cell(observation),
                int(observation["seed"]),
                endpoint_id,
            )
            if key in result:
                raise CampaignError("duplicate endpoint observation identity")
            result[key] = decimal_value(values.get(endpoint_id), f"endpoint_values.{endpoint_id}")
    return result


def _paired_values(
    data: Mapping[tuple[str, tuple[int, int, int], int, str], Decimal],
    template_id: str,
    positive_cell: tuple[int, int, int],
    negative_cell: tuple[int, int, int],
    endpoint_id: str,
) -> tuple[list[Decimal], list[int]]:
    positive = {
        key[2]: value
        for key, value in data.items()
        if key[0] == template_id and key[1] == positive_cell and key[3] == endpoint_id
    }
    negative = {
        key[2]: value
        for key, value in data.items()
        if key[0] == template_id and key[1] == negative_cell and key[3] == endpoint_id
    }
    seeds = sorted(set(positive).intersection(negative))
    if len(seeds) < 2:
        raise CampaignError("a predeclared paired contrast has fewer than two common seeds")
    return [positive[seed] - negative[seed] for seed in seeds], seeds


def _interaction_values(
    data: Mapping[tuple[str, tuple[int, int, int], int, str], Decimal],
    template_id: str,
    fine_high: tuple[int, int, int],
    fine_low: tuple[int, int, int],
    coarse_high: tuple[int, int, int],
    coarse_low: tuple[int, int, int],
    endpoint_id: str,
) -> tuple[list[Decimal], list[int]]:
    cells = (fine_high, fine_low, coarse_high, coarse_low)
    by_cell = [
        {
            key[2]: value
            for key, value in data.items()
            if key[0] == template_id and key[1] == cell and key[3] == endpoint_id
        }
        for cell in cells
    ]
    seeds = sorted(set.intersection(*(set(values) for values in by_cell)))
    if len(seeds) < 2:
        raise CampaignError("the predeclared interaction has fewer than two common seeds")
    return [
        (by_cell[0][seed] - by_cell[1][seed]) - (by_cell[2][seed] - by_cell[3][seed])
        for seed in seeds
    ], seeds


def _cell_values(
    data: Mapping[tuple[str, tuple[int, int, int], int, str], Decimal],
    template_id: str,
    cell: tuple[int, int, int],
    endpoint_id: str,
) -> tuple[list[Decimal], list[int]]:
    values = {
        key[2]: value
        for key, value in data.items()
        if key[0] == template_id and key[1] == cell and key[3] == endpoint_id
    }
    seeds = sorted(values)
    if len(seeds) < 2:
        raise CampaignError("a terminal sampling estimate has fewer than two complete runs")
    return [values[seed] for seed in seeds], seeds


def _expanded_comparison_cells(
    action: str, config: Mapping[str, Any]
) -> dict[str, tuple[int, int, int]]:
    """Choose the adjacent, preregistered cells appropriate to the action."""

    base_coarse = config["grids"][-2]
    base_fine = config["grids"][-1]
    next_grid = config["next_grid"]
    low_root = int(config["roots"][-2])
    high_root = int(config["roots"][-1])
    next_root = int(config["next_n_m"])
    coarse_low = (int(base_coarse["n_x"]), int(base_coarse["n_y"]), low_root)
    coarse_high = (int(base_coarse["n_x"]), int(base_coarse["n_y"]), high_root)
    fine_low = (int(base_fine["n_x"]), int(base_fine["n_y"]), low_root)
    fine_high = (int(base_fine["n_x"]), int(base_fine["n_y"]), high_root)
    axis = _action_axis(action)
    if axis is None and action == "replicate":
        axis = "base"
    if axis == "grid":
        terminal = (int(next_grid["n_x"]), int(next_grid["n_y"]), high_root)
        return {
            "terminal": terminal,
            "grid_positive": terminal,
            "grid_negative": fine_high,
            "root_negative": fine_low,
            "root_positive": fine_high,
            "interaction_fine_high": fine_high,
            "interaction_fine_low": fine_low,
            "interaction_coarse_high": coarse_high,
            "interaction_coarse_low": coarse_low,
        }
    if axis == "root":
        terminal = (int(base_fine["n_x"]), int(base_fine["n_y"]), next_root)
        return {
            "terminal": terminal,
            "grid_positive": fine_high,
            "grid_negative": coarse_high,
            "root_negative": fine_high,
            "root_positive": terminal,
            "interaction_fine_high": fine_high,
            "interaction_fine_low": fine_low,
            "interaction_coarse_high": coarse_high,
            "interaction_coarse_low": coarse_low,
        }
    if axis == "joint":
        next_fine_high = (
            int(next_grid["n_x"]),
            int(next_grid["n_y"]),
            next_root,
        )
        next_fine_low = (
            int(next_grid["n_x"]),
            int(next_grid["n_y"]),
            high_root,
        )
        next_coarse_high = (
            int(base_fine["n_x"]),
            int(base_fine["n_y"]),
            next_root,
        )
        return {
            "terminal": next_fine_high,
            "grid_positive": next_fine_high,
            "grid_negative": next_coarse_high,
            "root_negative": next_fine_low,
            "root_positive": next_fine_high,
            "interaction_fine_high": next_fine_high,
            "interaction_fine_low": next_fine_low,
            "interaction_coarse_high": next_coarse_high,
            "interaction_coarse_low": fine_high,
        }
    if axis == "base":
        return {
            "terminal": fine_high,
            "grid_positive": fine_high,
            "grid_negative": coarse_high,
            "root_negative": fine_low,
            "root_positive": fine_high,
            "interaction_fine_high": fine_high,
            "interaction_fine_low": fine_low,
            "interaction_coarse_high": coarse_high,
            "interaction_coarse_low": coarse_low,
        }
    raise CampaignError(f"unsupported completed next-stage action {action!r}")


def _expanded_statistics(
    config: Mapping[str, Any], sample_count: int, observed_cell_count: int
) -> tuple[dict[str, Any], Decimal, Decimal, Decimal, Decimal, Decimal]:
    if sample_count < 2:
        raise CampaignError("expanded analysis requires at least two common seeds")
    confidence = decimal_value(
        config["statistics"]["confidence_level"], "statistics.confidence_level"
    )
    alpha = decimal_value(config["statistics"]["equivalence_alpha"], "statistics.equivalence_alpha")
    safety = decimal_value(
        config["statistics"]["standard_error_safety_factor"],
        "statistics.standard_error_safety_factor",
    )
    endpoint_count = len(config["endpoints"])
    template_count = len(config["templates"])
    numerical_count = template_count * endpoint_count * 4
    method_count = max(0, template_count - 1) * observed_cell_count * endpoint_count
    equivalence_count = max(1, method_count + template_count * endpoint_count * 3)
    with localcontext() as context:
        context.prec = 60
        confidence_probability = Decimal(1) - (Decimal(1) - confidence) / (
            Decimal(2) * Decimal(numerical_count)
        )
        equivalence_probability = Decimal(1) - alpha / Decimal(equivalence_count)
    t_conf = _student_t_quantile(confidence_probability, sample_count - 1)
    t_alpha = _student_t_quantile(equivalence_probability, sample_count - 1)
    z_alpha = Decimal(str(NormalDist().inv_cdf(float(equivalence_probability))))
    z_power = decimal_value(config["statistics"]["power_normal_quantile"], "statistics.z_power")
    total_multiplier = decimal_value(
        config["statistics"]["conservative_total_budget_multiplier"],
        "statistics.conservative_total_budget_multiplier",
    )
    public = {
        "method": "same_frozen_bonferroni_student_t_and_normal_TOST_formulas_recomputed_for_authenticated_expanded_design_v1",
        "minimum_common_complete_run_count": str(sample_count),
        "degrees_of_freedom": str(sample_count - 1),
        "observed_equal_cell_count": str(observed_cell_count),
        "numerical_family_statement_count": str(numerical_count),
        "equivalence_family_statement_count": str(equivalence_count),
        "two_sided_confidence_familywise_probability": decimal_text(confidence_probability),
        "one_sided_equivalence_familywise_probability": decimal_text(equivalence_probability),
        "target_bound_multiplier": decimal_text(safety * t_conf),
        "equivalence_bound_multiplier": decimal_text(safety * t_alpha),
        "conservative_total_budget_multiplier": decimal_text(total_multiplier),
    }
    return (
        public,
        safety * t_conf,
        safety * t_alpha,
        z_alpha,
        z_power,
        total_multiplier,
    )


def _method_equivalence_from_observations(
    data: Mapping[tuple[str, tuple[int, int, int], int, str], Decimal],
    observations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    equivalence_multiplier: Decimal,
    z_alpha: Decimal,
    z_power: Decimal,
) -> dict[str, dict[str, Any]]:
    reference_id = str(config["reference_template_id"])
    template_ids = [str(item["id"]) for item in config["templates"]]
    cells = sorted({_observation_cell(item) for item in observations})
    results: dict[str, dict[str, Any]] = {}
    for template_id in template_ids:
        if template_id == reference_id:
            results[template_id] = {
                "reference_template_id": reference_id,
                "is_reference": True,
                "equal_cell_paired_shifts": [],
                "maximum_conservative_bound_over_margin": "0",
                "maximum_approximate_combined_required_complete_runs": str(MINIMUM_PILOT_SEEDS),
                "maximum_absolute_paired_mean_over_margin": "0",
                "point_estimates_inside_all_frozen_equivalence_margins": True,
                "passes_all_frozen_equivalence_margins": True,
            }
            continue
        summaries: list[dict[str, Any]] = []
        ratios: list[Decimal] = []
        point_ratios: list[Decimal] = []
        required: list[int] = []
        for cell in cells:
            for endpoint in config["endpoints"]:
                endpoint_id = str(endpoint["id"])
                left = {
                    key[2]: value
                    for key, value in data.items()
                    if key[0] == template_id and key[1] == cell and key[3] == endpoint_id
                }
                right = {
                    key[2]: value
                    for key, value in data.items()
                    if key[0] == reference_id and key[1] == cell and key[3] == endpoint_id
                }
                seeds = sorted(set(left).intersection(right))
                if not seeds:
                    continue
                if len(seeds) < 2:
                    raise CampaignError(
                        "equal-cell method comparison has fewer than two common seeds"
                    )
                summary = _equivalence_summary(
                    [left[seed] - right[seed] for seed in seeds],
                    decimal_value(endpoint["scale"], "endpoint.scale", positive=True),
                    decimal_value(
                        endpoint["equivalence_margin_relative"],
                        "endpoint.equivalence_margin_relative",
                        positive=True,
                    ),
                    decimal_value(
                        endpoint["expected_true_difference_relative"],
                        "endpoint.expected_true_difference_relative",
                    ),
                    equivalence_multiplier,
                    z_alpha,
                    z_power,
                )
                ratio = decimal_value(summary["conservative_bound_over_margin"], "method.ratio")
                point_ratio = decimal_value(
                    summary["absolute_paired_mean_over_margin"], "method.point_ratio"
                )
                ratios.append(ratio)
                point_ratios.append(point_ratio)
                required.append(int(summary["approximate_combined_required_complete_runs"]))
                summaries.append(
                    {
                        "n_x": str(cell[0]),
                        "n_y": str(cell[1]),
                        "n_m": str(cell[2]),
                        "endpoint_id": endpoint_id,
                        "common_seeds": [str(seed) for seed in seeds],
                        "summary": summary,
                    }
                )
        if not summaries:
            raise CampaignError("candidate has no authenticated equal-cell comparison")
        results[template_id] = {
            "reference_template_id": reference_id,
            "is_reference": False,
            "equal_cell_paired_shifts": summaries,
            "maximum_conservative_bound_over_margin": decimal_text(max(ratios)),
            "maximum_approximate_combined_required_complete_runs": str(max(required)),
            "maximum_absolute_paired_mean_over_margin": decimal_text(max(point_ratios)),
            "point_estimates_inside_all_frozen_equivalence_margins": max(point_ratios) <= 1,
            "passes_all_frozen_equivalence_margins": max(ratios) <= 1,
        }
    return results


def _terminal_runtime_and_grid_summary(
    observations: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    terminal: tuple[int, int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [
        item
        for item in observations
        if item["template_id"] == template["id"] and _observation_cell(item) == terminal
    ]
    if len(selected) < 2:
        raise CampaignError("terminal controller/certificate gate has fewer than two runs")
    action_counts: dict[str, int] = {}
    certificate_failures = 0
    terminal_patch_unexercised = 0
    macro_mode = int(template["card_info"]["adaptive_macroparticle_mode"])
    grid_mode = int(template["card_info"]["adaptive_grid_mode"])
    terminal_patch_requested = bool(
        template["card_info"]["adaptive_grid_terminal_patch_certification"]
    )
    for item in selected:
        record = required_mapping(item["record"], "observation.record")
        signal = required_mapping(
            record.get("runtime_replay_signal"), "record.runtime_replay_signal"
        )
        if macro_mode == 0:
            if signal.get("required") is not False or signal.get("present") is not False:
                raise CampaignError("nonadaptive terminal run has a replay signal")
        else:
            if signal.get("required") is not True or signal.get("present") is not True:
                raise CampaignError("adaptive terminal run lacks a replay signal")
            fields = required_mapping(signal.get("parsed_fields"), "signal.parsed_fields")
            action = str(fields.get("recommended_action"))
            if action not in (
                "request_initial_state_replay",
                "joint_replay_and_mesh_refinement",
            ):
                raise CampaignError("adaptive terminal run has an invalid replay action")
            action_counts[action] = action_counts.get(action, 0) + 1
        certificate = required_mapping(
            record.get("adaptive_grid_certificate"),
            "record.adaptive_grid_certificate",
        )
        if grid_mode == 0:
            if certificate.get("required") is not False:
                raise CampaignError("fixed terminal run has adaptive-grid evidence")
        else:
            if certificate.get("required") is not True or certificate.get("present") is not True:
                raise CampaignError("adaptive-grid terminal run lacks a certificate")
            if certificate.get("qualification_passed") is not True:
                certificate_failures += 1
            if (
                terminal_patch_requested
                and certificate.get("terminal_patch_qualification_exercised") is not True
            ):
                terminal_patch_unexercised += 1
    runtime = {
        "scope": "authenticated_final_terminal_cell_only_v1",
        "complete_run_count": str(len(selected)),
        "recommended_action_counts": {
            key: str(value) for key, value in sorted(action_counts.items())
        },
        "no_further_replay_requested": not action_counts,
    }
    grid = {
        "scope": "authenticated_final_terminal_cell_only_v1",
        "adaptive_grid_mode": str(grid_mode),
        "complete_run_count": str(len(selected)),
        "certificate_failure_count": str(certificate_failures),
        "terminal_patch_certification_requested": terminal_patch_requested,
        "terminal_patch_unexercised_run_count": str(terminal_patch_unexercised),
        "all_terminal_runs_certified": (
            certificate_failures == 0 and terminal_patch_unexercised == 0
        ),
    }
    return runtime, grid


def _gaussian_fit_diagnostic_summary(
    observations: Sequence[Mapping[str, Any]],
    template_id: str,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    gaussian_endpoints = [
        endpoint for endpoint in config["endpoints"] if endpoint["format"] == "json_scalar"
    ]
    if not gaussian_endpoints:
        return {
            "present": False,
            "direct_diagnostic_guard_passed": True,
            "absolute_fit_model_validity_established": False,
            "endpoint_diagnostics": [],
        }
    selected = [item for item in observations if item["template_id"] == template_id]
    if not selected:
        raise CampaignError("Gaussian diagnostic summary has no observations")
    documents: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for item in selected:
        run_directory = resolve_within(
            output_dir,
            relative_path(item["run_directory"], "observation.run_directory"),
            "observation.run_directory",
        )
        document = required_mapping(
            load_json(run_directory / GAUSSIAN_CORE_JSON), "gaussian_core.json"
        )
        if document.get("status") != "accepted_core_fit_not_physics_qualification":
            raise CampaignError("Gaussian estimator is not a bound accepted fit")
        documents.append((item, document))
    direct_keys = {
        "core_fit.mu_core_GeV": "mu_fit_uncertainty_GeV_diagnostic",
        "core_fit.sigma_core_GeV": "sigma_fit_uncertainty_GeV_diagnostic",
    }
    endpoint_summaries: list[dict[str, Any]] = []
    overall_guard = True
    for endpoint in gaussian_endpoints:
        endpoint_id = str(endpoint["id"])
        estimator_key = str(endpoint["key"])
        uncertainty_key = direct_keys.get(estimator_key)
        scale = decimal_value(endpoint["scale"], "endpoint.scale", positive=True)
        target = decimal_value(endpoint["target_relative_error"], "endpoint.target_relative_error")
        ratios: list[Decimal] = []
        maximum_window_sigma: list[Decimal] = []
        maximum_rebin_sigma: list[Decimal] = []
        for item, document in documents:
            stability = required_mapping(document.get("stability"), "gaussian.stability")
            maximum_window_sigma.append(
                decimal_value(
                    stability.get("maximum_window_sigma_relative_shift"),
                    "gaussian.maximum_window_sigma_relative_shift",
                )
            )
            rebin = stability.get("pair_rebin_sigma_relative_shift")
            if rebin is not None:
                maximum_rebin_sigma.append(
                    decimal_value(rebin, "gaussian.pair_rebin_sigma_relative_shift")
                )
            if uncertainty_key is not None:
                fit = required_mapping(document.get("core_fit"), "gaussian.core_fit")
                uncertainty = decimal_value(fit.get(uncertainty_key), f"gaussian.{uncertainty_key}")
                if uncertainty < 0:
                    raise CampaignError("Gaussian fit diagnostic uncertainty is negative")
                ratios.append(uncertainty / scale)
        direct_available = uncertainty_key is not None
        passed: bool | None = None
        maximum_ratio: str | None = None
        if direct_available:
            if len(ratios) != len(documents):
                raise CampaignError("Gaussian direct diagnostic coverage is incomplete")
            maximum = max(ratios)
            maximum_ratio = decimal_text(maximum)
            passed = maximum <= target
            overall_guard = overall_guard and passed
        endpoint_summaries.append(
            {
                "endpoint_id": endpoint_id,
                "estimator_key": estimator_key,
                "complete_fit_count": str(len(documents)),
                "direct_fit_diagnostic_key": uncertainty_key,
                "direct_relative_diagnostic_maximum": maximum_ratio,
                "direct_diagnostic_target_relative": decimal_text(target),
                "direct_diagnostic_guard_applied": direct_available,
                "direct_diagnostic_guard_passed": passed,
                "maximum_window_sigma_relative_shift": decimal_text(max(maximum_window_sigma)),
                "maximum_pair_rebin_sigma_relative_shift": (
                    None if not maximum_rebin_sigma else decimal_text(max(maximum_rebin_sigma))
                ),
                "absolute_fit_model_validity_established": False,
            }
        )
    return {
        "present": True,
        "scope": "all_authenticated_observations_used_by_this_decision_v1",
        "direct_diagnostic_guard_passed": overall_guard,
        "endpoint_diagnostics": endpoint_summaries,
        "deterministic_estimator_equivalence_is_distinct_from_absolute_fit_model_validity": True,
        "absolute_fit_model_validity_established": False,
        "interpretation": (
            "reported per-fit diagnostics guard directly supported mu/sigma endpoints; "
            "accepted fitter stability and deterministic equivalence do not validate the "
            "Gaussian model against an external truth distribution"
        ),
    }


def _final_bindings(
    policy_path: Path,
    manifest_path: Path,
    state_path: Path,
    analysis_path: Path,
    next_manifest_path: Path,
    next_state_path: Path,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    pilot_analysis: Mapping[str, Any],
    next_manifest: Mapping[str, Any],
    next_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "policy_file_sha256": sha256_file(policy_path),
        "policy_self_sha256": policy["self_sha256"],
        "pilot_manifest_file_sha256": sha256_file(manifest_path),
        "pilot_manifest_self_sha256": manifest["self_sha256"],
        "pilot_state_file_sha256": sha256_file(state_path),
        "pilot_state_self_sha256": state["self_sha256"],
        "pilot_analysis_file_sha256": sha256_file(analysis_path),
        "pilot_analysis_self_sha256": pilot_analysis["self_sha256"],
        "next_manifest_file_sha256": sha256_file(next_manifest_path),
        "next_manifest_self_sha256": next_manifest["self_sha256"],
        "next_state_file_sha256": sha256_file(next_state_path),
        "next_state_self_sha256": next_state["self_sha256"],
    }


def _compute_extension_final_analysis(
    policy_path: Path,
    manifest_path: Path,
    state_path: Path,
    pilot_analysis_path: Path,
    next_manifest_path: Path,
    next_state_path: Path,
    output_dir: Path,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    pilot_analysis: Mapping[str, Any],
    next_manifest: Mapping[str, Any],
    next_state: Mapping[str, Any],
    config: Mapping[str, Any],
    pilot: Sequence[Mapping[str, Any]],
    followup: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    action = str(next_manifest["recommendation"])
    comparison = _expanded_comparison_cells(action, config)
    observations = [*pilot, *followup]
    data = _endpoint_lookup(observations, config)
    template_ids = [str(item["id"]) for item in config["templates"]]
    elapsed_by_template_and_case: dict[str, dict[tuple[Any, ...], int]] = {
        template_id: {} for template_id in template_ids
    }
    for item in observations:
        case = (*_observation_cell(item), int(item["seed"]))
        if case in elapsed_by_template_and_case[str(item["template_id"])]:
            raise CampaignError("expanded timing evidence duplicates a template/cell/seed")
        elapsed_by_template_and_case[str(item["template_id"])][case] = int(
            item["elapsed_nanoseconds"]
        )
    timing_summaries = _paired_timing_summaries(
        elapsed_by_template_and_case,
        str(config["reference_template_id"]),
        decimal_value(config["statistics"]["confidence_level"], "statistics.confidence"),
    )
    observed_cells = sorted({_observation_cell(item) for item in observations})
    common_counts: list[int] = []
    for cell in observed_cells:
        seed_sets = [
            {
                int(item["seed"])
                for item in observations
                if item["template_id"] == template_id and _observation_cell(item) == cell
            }
            for template_id in template_ids
        ]
        if seed_sets:
            count = len(set.intersection(*seed_sets))
            if count:
                common_counts.append(count)
    if not common_counts:
        raise CampaignError("expanded design has no common complete-process seeds")
    statistics, target_multiplier, equivalence_multiplier, z_alpha, z_power, total_multiplier = (
        _expanded_statistics(config, min(common_counts), len(observed_cells))
    )
    methods = _method_equivalence_from_observations(
        data,
        observations,
        config,
        equivalence_multiplier,
        z_alpha,
        z_power,
    )
    templates_by_id = {str(item["id"]): item for item in config["templates"]}
    used_extension = {
        int(item["seed"])
        for item in followup
        if item["seed_population"] == "preregistered_extension"
    }
    results: list[dict[str, Any]] = []
    sortable: list[tuple[int, Decimal, int, int, str, dict[str, Any]]] = []
    for template_id in template_ids:
        endpoints: list[dict[str, Any]] = []
        grid_ratios: list[Decimal] = []
        root_ratios: list[Decimal] = []
        interaction_ratios: list[Decimal] = []
        sampling_ratios: list[Decimal] = []
        grid_points: list[Decimal] = []
        root_points: list[Decimal] = []
        interaction_points: list[Decimal] = []
        total_points: list[Decimal] = []
        total_ratios: list[Decimal] = []
        power_counts = [
            int(methods[template_id]["maximum_approximate_combined_required_complete_runs"])
        ]
        for endpoint in config["endpoints"]:
            endpoint_id = str(endpoint["id"])
            scale = decimal_value(endpoint["scale"], "endpoint.scale", positive=True)
            target = decimal_value(
                endpoint["target_relative_error"], "endpoint.target", positive=True
            )
            margin = decimal_value(
                endpoint["equivalence_margin_relative"],
                "endpoint.margin",
                positive=True,
            )
            expected = decimal_value(
                endpoint["expected_true_difference_relative"], "endpoint.expected"
            )
            grid_values, grid_seeds = _paired_values(
                data,
                template_id,
                comparison["grid_positive"],
                comparison["grid_negative"],
                endpoint_id,
            )
            root_values, root_seeds = _paired_values(
                data,
                template_id,
                comparison["root_positive"],
                comparison["root_negative"],
                endpoint_id,
            )
            interaction_values, interaction_seeds = _interaction_values(
                data,
                template_id,
                comparison["interaction_fine_high"],
                comparison["interaction_fine_low"],
                comparison["interaction_coarse_high"],
                comparison["interaction_coarse_low"],
                endpoint_id,
            )
            terminal_values, terminal_seeds = _cell_values(
                data, template_id, comparison["terminal"], endpoint_id
            )
            grid_summary = _contrast_summary(grid_values, scale, target, target_multiplier)
            root_summary = _contrast_summary(root_values, scale, target, target_multiplier)
            interaction_summary = _contrast_summary(
                interaction_values, scale, target, target_multiplier
            )
            sampling_summary = _sampling_summary(terminal_values, scale, target, target_multiplier)
            for summary, values in (
                (grid_summary, grid_values),
                (root_summary, root_values),
                (interaction_summary, interaction_values),
            ):
                summary["approximate_equivalence"] = _equivalence_summary(
                    values,
                    scale,
                    margin,
                    expected,
                    equivalence_multiplier,
                    z_alpha,
                    z_power,
                )
            g_ratio = decimal_value(grid_summary["conservative_bound_over_target"], "grid.ratio")
            r_ratio = decimal_value(root_summary["conservative_bound_over_target"], "root.ratio")
            i_ratio = decimal_value(
                interaction_summary["conservative_bound_over_target"], "interaction.ratio"
            )
            s_ratio = decimal_value(
                sampling_summary["conservative_bound_over_target"], "sampling.ratio"
            )
            g_point = decimal_value(grid_summary["absolute_paired_mean_over_target"], "grid.point")
            r_point = decimal_value(root_summary["absolute_paired_mean_over_target"], "root.point")
            i_point = decimal_value(
                interaction_summary["absolute_paired_mean_over_target"], "interaction.point"
            )
            summaries = (grid_summary, root_summary, interaction_summary)
            for summary, point in zip(summaries, (g_point, r_point, i_point)):
                if point < 1:
                    power_counts.append(
                        int(summary["approximate_precision_required_complete_runs"])
                    )
                equivalence = required_mapping(
                    summary["approximate_equivalence"], "summary.approximate_equivalence"
                )
                if (
                    decimal_value(
                        equivalence["absolute_paired_mean_over_margin"],
                        "equivalence.point",
                    )
                    < 1
                ):
                    power_counts.append(
                        int(equivalence["approximate_combined_required_complete_runs"])
                    )
            power_counts.append(
                int(sampling_summary["approximate_precision_required_complete_runs"])
            )
            total_bound = sum(
                (
                    decimal_value(grid_summary["conservative_absolute_bound"], "grid.bound"),
                    decimal_value(root_summary["conservative_absolute_bound"], "root.bound"),
                    decimal_value(
                        interaction_summary["conservative_absolute_bound"],
                        "interaction.bound",
                    ),
                    decimal_value(
                        sampling_summary["conservative_absolute_bound"],
                        "sampling.bound",
                    ),
                ),
                Decimal(0),
            )
            target_absolute = scale * target
            total_ratio = total_bound / target_absolute
            total_point = (
                abs(decimal_value(grid_summary["paired_mean"], "grid.mean"))
                + abs(decimal_value(root_summary["paired_mean"], "root.mean"))
                + abs(decimal_value(interaction_summary["paired_mean"], "interaction.mean"))
            ) / target_absolute
            grid_ratios.append(g_ratio)
            root_ratios.append(r_ratio)
            interaction_ratios.append(i_ratio)
            sampling_ratios.append(s_ratio)
            grid_points.append(g_point)
            root_points.append(r_point)
            interaction_points.append(i_point)
            total_points.append(total_point)
            total_ratios.append(total_ratio)
            endpoints.append(
                {
                    "id": endpoint_id,
                    "units": endpoint["units"],
                    "grid_common_seeds": [str(seed) for seed in grid_seeds],
                    "root_common_seeds": [str(seed) for seed in root_seeds],
                    "interaction_common_seeds": [str(seed) for seed in interaction_seeds],
                    "terminal_sampling_seeds": [str(seed) for seed in terminal_seeds],
                    "terminal_grid_contrast": grid_summary,
                    "terminal_n_m_contrast": root_summary,
                    "grid_n_m_interaction": interaction_summary,
                    "terminal_independent_run_sampling": sampling_summary,
                    "summed_conservative_absolute_error": decimal_text(total_bound),
                    "summed_conservative_error_over_target": decimal_text(total_ratio),
                    "summed_absolute_point_estimates_over_target": decimal_text(total_point),
                }
            )
        method = methods[template_id]
        required_total = max(power_counts)
        action_value, gate, reasons = _choose_action(
            max(grid_ratios),
            max(root_ratios),
            max(interaction_ratios),
            max(sampling_ratios),
            max(total_ratios),
            max(grid_points),
            max(root_points),
            max(interaction_points),
            max(total_points),
            total_multiplier,
            bool(method["passes_all_frozen_equivalence_margins"]),
            bool(method["point_estimates_inside_all_frozen_equivalence_margins"]),
            required_total,
            min(common_counts),
            len(config["extension_seeds"]) - len(used_extension),
            len(config["holdout_seeds"]),
        )
        runtime, grid = _terminal_runtime_and_grid_summary(
            observations, templates_by_id[template_id], comparison["terminal"]
        )
        action_value, gate, reasons = _apply_runtime_action_constraint(
            action_value,
            gate,
            reasons,
            {key: int(value) for key, value in runtime["recommended_action_counts"].items()},
        )
        if grid["all_terminal_runs_certified"] is not True:
            gate = False
            reasons = [*reasons, "terminal_adaptive_grid_internal_certificate_not_accepted"]
        fit = _gaussian_fit_diagnostic_summary(observations, template_id, config, output_dir)
        if fit["direct_diagnostic_guard_passed"] is not True:
            gate = False
            reasons = [*reasons, "gaussian_direct_fit_diagnostic_exceeds_endpoint_target"]
        action_value, gate, reasons = _apply_replication_seed_capacity(
            action_value,
            gate,
            reasons,
            required_total,
            min(common_counts),
            len(config["extension_seeds"]) - len(used_extension),
        )
        if action_value != "hold":
            gate = False
        elapsed = [
            int(item["elapsed_nanoseconds"])
            for item in observations
            if item["template_id"] == template_id
        ]
        mean_elapsed = Decimal(sum(elapsed)) / Decimal(len(elapsed))
        result = {
            "template_id": template_id,
            "method_equivalence": method,
            "runtime_controller_signals": runtime,
            "adaptive_grid_certificates": grid,
            "gaussian_fit_diagnostics": fit,
            "endpoints": endpoints,
            "maximum_ratios": {
                "grid": decimal_text(max(grid_ratios)),
                "n_m": decimal_text(max(root_ratios)),
                "interaction": decimal_text(max(interaction_ratios)),
                "independent_sampling": decimal_text(max(sampling_ratios)),
                "summed_budget": decimal_text(max(total_ratios)),
                "grid_point_estimate": decimal_text(max(grid_points)),
                "n_m_point_estimate": decimal_text(max(root_points)),
                "interaction_point_estimate": decimal_text(max(interaction_points)),
                "summed_point_estimates": decimal_text(max(total_points)),
            },
            "approximate_required_complete_runs": str(required_total),
            "approximate_required_complete_seed_blocks": str(required_total),
            "seed_block_counting_contract": (
                "one independent common seed across all required template cells; "
                "not one FeeBL process"
            ),
            "statistical_next_action": action_value,
            "expanded_gate_passed": gate,
            "decision_reasons": reasons,
            "mean_elapsed_seconds": decimal_text(mean_elapsed / Decimal(1_000_000_000)),
            "paired_reference_timing": timing_summaries[template_id],
        }
        results.append(result)
        tier = 0 if gate else (1 if method["passes_all_frozen_equivalence_margins"] else 2)
        convergence_score = max(
            max(grid_ratios),
            max(root_ratios),
            max(interaction_ratios),
            max(sampling_ratios),
            max(total_ratios) / total_multiplier,
        )
        timing_status = timing_summaries[template_id]["comparison_to_reference"]
        timing_tier = (
            0
            if timing_status == "significantly_faster"
            else (2 if timing_status == "significantly_slower" else 1)
        )
        reference_tie_break = 0 if template_id == str(config["reference_template_id"]) else 1
        sortable.append(
            (tier, convergence_score, timing_tier, reference_tie_break, template_id, result)
        )
    sortable.sort(key=lambda item: item[:-1])
    selected = sortable[0][5]
    adjacent_estimator_gate_passed = bool(selected["expanded_gate_passed"])
    design_validity = _solution_verification_design_status(config)
    passed = bool(
        adjacent_estimator_gate_passed
        and design_validity["absolute_solution_convergence_qualification_ready"]
    )
    unsigned = {
        "schema": FINAL_ANALYSIS_SCHEMA,
        "schema_version": "1",
        "qualification_status": (
            "expanded_training_evidence_ready_for_sequestered_confirmation"
            if passed
            else "expanded_training_evidence_diagnostic_failure"
        ),
        "analysis_kind": "extension_or_axis_followup",
        "bindings": _final_bindings(
            policy_path,
            manifest_path,
            state_path,
            pilot_analysis_path,
            next_manifest_path,
            next_state_path,
            policy,
            manifest,
            state,
            pilot_analysis,
            next_manifest,
            next_state,
        ),
        "completed_next_stage_action": action,
        "statistics": statistics,
        "comparison_cells": {
            key: {"n_x": str(value[0]), "n_y": str(value[1]), "n_m": str(value[2])}
            for key, value in comparison.items()
        },
        "seed_accounting": {
            "pilot_seeds": [str(seed) for seed in config["pilot_seeds"]],
            "used_extension_seeds": [str(seed) for seed in sorted(used_extension)],
            "holdout_seeds_used_for_selection": [],
        },
        "template_results": results,
        "candidate_ranking_rule": (
            "expanded_gate_then_method_equivalence_then_conservative_convergence_score; "
            "timing_breaks_a_convergence_tie_only_with_a_sign_definite_paired_CI"
        ),
        "candidate_ranking": [
            {
                "rank": str(index),
                "template_id": item[4],
                "expanded_gate_passed": item[5]["expanded_gate_passed"],
                "method_equivalence_passed": item[5]["method_equivalence"][
                    "passes_all_frozen_equivalence_margins"
                ],
                "convergence_score": decimal_text(item[1]),
                "paired_timing_status": item[5]["paired_reference_timing"][
                    "comparison_to_reference"
                ],
            }
            for index, item in enumerate(sortable, start=1)
        ],
        "selected_template_id": selected["template_id"],
        "qualified_template_id": selected["template_id"] if passed else None,
        "expanded_gate_passed": adjacent_estimator_gate_passed,
        "absolute_solution_convergence_gate_passed": passed,
        "solution_verification_design": design_validity,
        "recommended_disposition": (
            "freeze_selected_template_then_use_only_sequestered_unseen_confirmation_seeds"
            if passed
            else "diagnostic_failure_no_manual_promotion"
        ),
        "selection_firewall": {
            "holdout_observations_used_for_selection": False,
            "manual_promotion_permitted": False,
        },
        "qualification_scope": (
            "declared_endpoint_and_deterministic_gaussian_estimator_convergence_only; "
            "not_absolute_Gaussian_fit_model_validity"
        ),
        "absolute_gaussian_fit_model_validity_established": False,
    }
    return with_self_hash(unsigned)


def _compute_holdout_final_analysis(
    policy_path: Path,
    manifest_path: Path,
    state_path: Path,
    pilot_analysis_path: Path,
    next_manifest_path: Path,
    next_state_path: Path,
    output_dir: Path,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    pilot_analysis: Mapping[str, Any],
    next_manifest: Mapping[str, Any],
    next_state: Mapping[str, Any],
    config: Mapping[str, Any],
    followup: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        pilot_analysis.get("recommendation") != "hold"
        or pilot_analysis.get("pilot_gate_passed") is not True
    ):
        raise CampaignError("holdout analysis requires a passed, pilot-frozen hold decision")
    if any(item["seed_population"] != "reserved_holdout" for item in followup):
        raise CampaignError("holdout analysis contains a nonholdout seed population")
    selected_id = str(pilot_analysis["selected_template_id"])
    reference_id = str(config["reference_template_id"])
    terminal_cells = {_observation_cell(item) for item in followup}
    if len(terminal_cells) != 1:
        raise CampaignError("holdout stage must contain exactly one terminal cell")
    terminal = next(iter(terminal_cells))
    data = _endpoint_lookup(followup, config)
    selected_seeds = {int(item["seed"]) for item in followup if item["template_id"] == selected_id}
    if len(selected_seeds) < MINIMUM_PILOT_SEEDS:
        raise CampaignError("holdout confirmation has too few selected-template seeds")
    statistics, target_multiplier, equivalence_multiplier, z_alpha, z_power, _ = (
        _expanded_statistics(config, len(selected_seeds), 1)
    )
    endpoint_results: list[dict[str, Any]] = []
    all_equivalent = True
    all_sampling = True
    required_counts: list[int] = []
    for endpoint in config["endpoints"]:
        endpoint_id = str(endpoint["id"])
        scale = decimal_value(endpoint["scale"], "endpoint.scale", positive=True)
        target = decimal_value(endpoint["target_relative_error"], "endpoint.target", positive=True)
        margin = decimal_value(
            endpoint["equivalence_margin_relative"], "endpoint.margin", positive=True
        )
        expected = decimal_value(endpoint["expected_true_difference_relative"], "endpoint.expected")
        selected_values, sampling_seeds = _cell_values(data, selected_id, terminal, endpoint_id)
        sampling = _sampling_summary(selected_values, scale, target, target_multiplier)
        if selected_id == reference_id:
            differences = [Decimal(0) for _ in sampling_seeds]
            common_seeds = sampling_seeds
        else:
            selected_map = {
                key[2]: value
                for key, value in data.items()
                if key[0] == selected_id and key[1] == terminal and key[3] == endpoint_id
            }
            reference_map = {
                key[2]: value
                for key, value in data.items()
                if key[0] == reference_id and key[1] == terminal and key[3] == endpoint_id
            }
            common_seeds = sorted(set(selected_map).intersection(reference_map))
            if len(common_seeds) != len(selected_seeds):
                raise CampaignError("holdout method comparison lacks a complete paired seed set")
            differences = [selected_map[seed] - reference_map[seed] for seed in common_seeds]
        equivalence = _equivalence_summary(
            differences,
            scale,
            margin,
            expected,
            equivalence_multiplier,
            z_alpha,
            z_power,
        )
        equivalent = bool(equivalence["equivalent_at_frozen_margin"])
        sampling_pass = (
            decimal_value(sampling["conservative_bound_over_target"], "sampling.ratio") <= 1
        )
        all_equivalent = all_equivalent and equivalent
        all_sampling = all_sampling and sampling_pass
        required_counts.append(int(equivalence["approximate_combined_required_complete_runs"]))
        required_counts.append(int(sampling["approximate_precision_required_complete_runs"]))
        endpoint_results.append(
            {
                "id": endpoint_id,
                "units": endpoint["units"],
                "holdout_seeds": [str(seed) for seed in common_seeds],
                "selected_vs_reference_equal_cell_equivalence": equivalence,
                "selected_terminal_independent_run_sampling": sampling,
                "equivalence_passed": equivalent,
                "sampling_precision_passed": sampling_pass,
            }
        )
    template = next(item for item in config["templates"] if item["id"] == selected_id)
    runtime, grid = _terminal_runtime_and_grid_summary(followup, template, terminal)
    fit = _gaussian_fit_diagnostic_summary(followup, selected_id, config, output_dir)
    pilot_selected = next(
        required_mapping(item, "pilot_analysis.template_result")
        for item in required_list(
            pilot_analysis.get("template_results"), "pilot_analysis.template_results"
        )
        if required_mapping(item, "pilot_analysis.template_result").get("template_id")
        == selected_id
    )
    planned_required = _int_text(
        pilot_selected.get("approximate_required_complete_runs"),
        "pilot_selected.approximate_required_complete_runs",
        MINIMUM_PILOT_SEEDS,
    )
    achieved_power_count = len(selected_seeds) >= planned_required
    holdout_estimator_gate_passed = (
        all_equivalent
        and all_sampling
        and runtime["no_further_replay_requested"] is True
        and grid["all_terminal_runs_certified"] is True
        and fit["direct_diagnostic_guard_passed"] is True
        and achieved_power_count
    )
    reasons: list[str] = []
    if not all_equivalent:
        reasons.append("holdout_equal_cell_equivalence_failed")
    if not all_sampling:
        reasons.append("holdout_independent_sampling_precision_failed")
    if runtime["no_further_replay_requested"] is not True:
        reasons.append("holdout_runtime_controller_requested_further_replay")
    if grid["all_terminal_runs_certified"] is not True:
        reasons.append("holdout_adaptive_grid_internal_certificate_failed")
    if fit["direct_diagnostic_guard_passed"] is not True:
        reasons.append("holdout_gaussian_direct_fit_diagnostic_failed")
    if not achieved_power_count:
        reasons.append("holdout_did_not_reach_pilot_planned_complete_run_count")
    design_validity = _solution_verification_design_status(config)
    passed = bool(
        holdout_estimator_gate_passed
        and design_validity["absolute_solution_convergence_qualification_ready"]
    )
    if not design_validity["absolute_solution_convergence_qualification_ready"]:
        reasons.append("absolute_solution_verification_design_is_hard_blocked")
    unsigned = {
        "schema": FINAL_ANALYSIS_SCHEMA,
        "schema_version": "1",
        "qualification_status": (
            "sequestered_holdout_confirmed_declared_estimator_scope"
            if passed
            else "sequestered_holdout_rejected_declared_estimator_scope"
        ),
        "analysis_kind": "sequestered_holdout_confirmation",
        "bindings": _final_bindings(
            policy_path,
            manifest_path,
            state_path,
            pilot_analysis_path,
            next_manifest_path,
            next_state_path,
            policy,
            manifest,
            state,
            pilot_analysis,
            next_manifest,
            next_state,
        ),
        "selection_firewall": {
            "selected_template_source": "pilot_analysis.json only",
            "holdout_observations_used_to_rerank_or_replace_selection": False,
            "alternative_template_promotion_after_holdout": "forbidden",
            "manual_promotion_permitted": False,
        },
        "selected_template_id": selected_id,
        "reference_template_id": reference_id,
        "qualified_template_id": selected_id if passed else None,
        "holdout_gate_passed": passed,
        "holdout_declared_estimator_gate_passed": holdout_estimator_gate_passed,
        "solution_verification_design": design_validity,
        "decision_reasons": reasons,
        "terminal_cell": {
            "n_x": str(terminal[0]),
            "n_y": str(terminal[1]),
            "n_m": str(terminal[2]),
        },
        "statistics": statistics,
        "power_accounting": {
            "pilot_planned_required_complete_runs": str(planned_required),
            "authenticated_holdout_complete_runs": str(len(selected_seeds)),
            "planned_count_achieved": achieved_power_count,
        },
        "endpoint_results": endpoint_results,
        "runtime_controller_signals": runtime,
        "adaptive_grid_certificates": grid,
        "gaussian_fit_diagnostics": fit,
        "recommended_disposition": (
            "accept_pilot_frozen_template_for_declared_estimator_scope"
            if passed
            else "diagnostic_failure_no_alternative_or_manual_promotion"
        ),
        "qualification_scope": (
            "declared_endpoint_and_deterministic_gaussian_estimator_confirmation_only; "
            "not_absolute_Gaussian_fit_model_validity"
        ),
        "absolute_gaussian_fit_model_validity_established": False,
    }
    return with_self_hash(unsigned)


def _compute_final_analysis(
    policy_path: Path,
    manifest_path: Path,
    output_dir: Path,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    state_path = output_dir / "pilot_state.json"
    state = validate_state(
        state_path,
        policy_path,
        manifest_path,
        output_dir,
        policy,
        manifest,
        config,
    )
    if state.get("status") != "complete":
        raise CampaignError("final analysis requires a complete authenticated pilot")
    pilot_analysis, next_manifest = validate_analysis_and_next(
        policy_path, manifest_path, output_dir, policy, manifest, config
    )
    pilot_analysis_path = output_dir / "pilot_analysis.json"
    next_manifest_path = output_dir / "next_stage" / "next_stage_manifest.json"
    next_state_path = output_dir / "next_stage" / "next_stage_state.json"
    if not next_state_path.is_file():
        raise CampaignError("next-stage state is missing; final analysis is forbidden")
    next_state = validate_next_state(
        next_state_path,
        pilot_analysis_path,
        next_manifest_path,
        output_dir,
        next_manifest,
        config,
    )
    if next_state.get("status") != "complete":
        raise CampaignError("next-stage state must be complete before final analysis")
    pilot, followup = _completed_observations(manifest, state, next_manifest, next_state, config)
    stage_design = required_mapping(next_manifest.get("stage_design"), "next_manifest.stage_design")
    populations = required_list(
        stage_design.get("seed_populations"), "next_manifest.stage_design.seed_populations"
    )
    common = (
        policy_path,
        manifest_path,
        state_path,
        pilot_analysis_path,
        next_manifest_path,
        next_state_path,
        output_dir,
        policy,
        manifest,
        state,
        pilot_analysis,
        next_manifest,
        next_state,
        config,
    )
    if populations == ["reserved_holdout"]:
        return _compute_holdout_final_analysis(*common, followup)
    if "reserved_holdout" in populations:
        raise CampaignError("holdout population is mixed with training observations")
    if not followup:
        raise CampaignError("empty next stage cannot be analyzed")
    return _compute_extension_final_analysis(*common, pilot, followup)


def analyze_next_stage(policy_path: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    """Authenticate and analyze a complete next stage exactly once."""

    output_dir = output_dir.resolve()
    policy_path = policy_path.resolve()
    manifest_path = manifest_path.resolve()
    final_path = output_dir / "final_analysis.json"
    if final_path.exists():
        raise CampaignError("refusing to overwrite existing final_analysis.json")
    policy, manifest, config = validate_campaign(policy_path, manifest_path, output_dir)
    result = _compute_final_analysis(
        policy_path, manifest_path, output_dir, policy, manifest, config
    )
    write_canonical(final_path, result)
    validated = validate_final_analysis(
        final_path, policy_path, manifest_path, output_dir, policy, manifest, config
    )
    if validated != result:
        raise CampaignError("internal error: final analysis failed round-trip validation")
    return result


def validate_final_analysis(
    final_path: Path,
    policy_path: Path,
    manifest_path: Path,
    output_dir: Path,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the final decision from hash-bound evidence and compare bytes."""

    if final_path.resolve() != (output_dir.resolve() / "final_analysis.json"):
        raise CampaignError("final analysis must use the canonical campaign path")
    actual = validate_self_hash(final_path, FINAL_ANALYSIS_SCHEMA)
    expected = _compute_final_analysis(
        policy_path, manifest_path, output_dir, policy, manifest, config
    )
    if actual != expected:
        raise CampaignError(
            "final analysis differs from the deterministic authenticated recomputation"
        )
    return actual


def _artifact_paths(output_dir: Path) -> tuple[Path, Path]:
    root = output_dir.resolve()
    return root / "campaign_policy.json", root / "pilot_manifest.json"


def _print_result(value: Mapping[str, Any]) -> None:
    sys.stdout.buffer.write(replay.canonical_bytes(value) + b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded external FeeBL grid-by-n_m pilot actuator and analyzer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze", help="freeze policy, pilot manifest, and cards")
    freeze_parser.add_argument("--request", required=True, type=Path)
    freeze_parser.add_argument("--output-dir", required=True, type=Path)
    for name in ("validate", "execute", "analyze", "analyze-next"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-dir", required=True, type=Path)
    next_parser = subparsers.add_parser(
        "execute-next", help="deliberately execute the validated analyzed next stage"
    )
    next_parser.add_argument("--output-dir", required=True, type=Path)
    next_parser.add_argument("--confirm-analysis-sha256", required=True)
    next_parser.add_argument("--allow-holdout", action="store_true")
    all_parser = subparsers.add_parser("all", help="freeze, execute pilot, then analyze")
    all_parser.add_argument("--request", required=True, type=Path)
    all_parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            policy, manifest = freeze_campaign(args.request, args.output_dir)
            _print_result(
                {
                    "policy_self_sha256": policy["self_sha256"],
                    "pilot_manifest_self_sha256": manifest["self_sha256"],
                    "pilot_run_count": str(len(manifest["realizations"])),
                    "status": "frozen",
                }
            )
            return 0
        if args.command == "all":
            freeze_campaign(args.request, args.output_dir)
            policy_path, manifest_path = _artifact_paths(args.output_dir)
            result = execute_pilot(policy_path, manifest_path, args.output_dir)
            if result != 0:
                return result
            analysis, next_manifest = analyze_pilot(policy_path, manifest_path, args.output_dir)
            _print_result(
                {
                    "analysis_self_sha256": analysis["self_sha256"],
                    "next_manifest_self_sha256": next_manifest["self_sha256"],
                    "recommendation": analysis["recommendation"],
                    "selected_template_id": analysis["selected_template_id"],
                    "status": "analyzed",
                }
            )
            return 0

        if args.command == "execute-next":
            policy_path, manifest_path = _artifact_paths(args.output_dir)
            return execute_next_stage(
                policy_path,
                manifest_path,
                args.output_dir,
                args.confirm_analysis_sha256,
                allow_holdout=args.allow_holdout,
            )

        if args.command == "analyze-next":
            policy_path, manifest_path = _artifact_paths(args.output_dir)
            final_analysis = analyze_next_stage(policy_path, manifest_path, args.output_dir)
            _print_result(
                {
                    "final_analysis_self_sha256": final_analysis["self_sha256"],
                    "analysis_kind": final_analysis["analysis_kind"],
                    "qualified_template_id": final_analysis["qualified_template_id"],
                    "status": "final_analyzed",
                }
            )
            return 0

        policy_path, manifest_path = _artifact_paths(args.output_dir)
        if args.command == "validate":
            policy, manifest, config = validate_campaign(
                policy_path, manifest_path, args.output_dir
            )
            state_path = args.output_dir.resolve() / "pilot_state.json"
            status_value = "frozen"
            if state_path.exists():
                state = validate_state(
                    state_path,
                    policy_path,
                    manifest_path,
                    args.output_dir.resolve(),
                    policy,
                    manifest,
                    config,
                )
                status_value = str(state["status"])
            analysis_path = args.output_dir.resolve() / "pilot_analysis.json"
            next_root = args.output_dir.resolve() / "next_stage"
            next_path = next_root / "next_stage_manifest.json"
            final_path = args.output_dir.resolve() / "final_analysis.json"
            if analysis_path.exists() != next_path.exists() or (
                next_root.exists() and not next_path.exists()
            ):
                raise CampaignError("analysis and next-stage manifest must exist together")
            if final_path.exists() and not analysis_path.exists():
                raise CampaignError("final analysis exists without pilot decision artifacts")
            if analysis_path.exists():
                _, next_manifest = validate_analysis_and_next(
                    policy_path,
                    manifest_path,
                    args.output_dir.resolve(),
                    policy,
                    manifest,
                    config,
                )
                status_value = "analyzed"
                next_state_path = next_root / "next_stage_state.json"
                if next_state_path.exists():
                    next_state = validate_next_state(
                        next_state_path,
                        analysis_path,
                        next_path,
                        args.output_dir.resolve(),
                        next_manifest,
                        config,
                    )
                    status_value = "next_stage_" + str(next_state["status"])
                if final_path.exists():
                    if not next_state_path.exists():
                        raise CampaignError("final analysis exists without a next-stage state")
                    validate_final_analysis(
                        final_path,
                        policy_path,
                        manifest_path,
                        args.output_dir.resolve(),
                        policy,
                        manifest,
                        config,
                    )
                    status_value = "final_analyzed"
            _print_result({"status": status_value, "validation": "passed"})
            return 0
        if args.command == "execute":
            return execute_pilot(policy_path, manifest_path, args.output_dir)
        if args.command == "analyze":
            analysis, next_manifest = analyze_pilot(policy_path, manifest_path, args.output_dir)
            _print_result(
                {
                    "analysis_self_sha256": analysis["self_sha256"],
                    "next_manifest_self_sha256": next_manifest["self_sha256"],
                    "recommendation": analysis["recommendation"],
                    "selected_template_id": analysis["selected_template_id"],
                    "status": "analyzed",
                }
            )
            return 0
        raise CampaignError(f"unsupported command {args.command!r}")
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
