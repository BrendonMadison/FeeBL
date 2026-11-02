#!/usr/bin/env python3
"""Fail-closed, serial harness for the frozen FCC-Z adaptive holdout.

The harness deliberately separates materialization (``freeze``), execution
(``run-next``), and analysis (``analyze``).  ``freeze`` never starts FeeBL.
Every later command requires the externally retained campaign SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPLAY_DIRECTORY = Path(__file__).resolve().parent
if not (REPLAY_DIRECTORY / "adaptive_root_replay.py").is_file():
    # Frozen campaigns retain their self-contained campaign/replay layout.
    REPLAY_DIRECTORY = REPLAY_DIRECTORY.parent / "replay"
if str(REPLAY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPLAY_DIRECTORY))
import adaptive_root_replay as replay  # noqa: E402


CAMPAIGN_SCHEMA = "feebl.fccz-frozen-eight-campaign.v1"
CONFIG_SCHEMA = "feebl.fccz-frozen-eight-config.v1"
UNSEEN_SCHEMA = "feebl.fccz-preregistered-unseen-seeds.v1"
# Kept in lockstep with generate_coupled_pilot_request.py.  These 64 IDs are
# reserved exclusively for frozen-harness unseen complete-process evidence.
PREREGISTERED_UNSEEN_SEED_MIN = 5001
PREREGISTERED_UNSEEN_SEED_MAX = 5064
SUPPORTED_SIGMA_Z_MM = (Decimal("0.2265625"), Decimal("14.5"))
RUN_MANIFEST_SCHEMA = "feebl.fccz-frozen-eight-run-files.v1"
GAUSSIAN_SCHEMA = "feebl.gaussian-core-spectrum.v1"
GRID_CERTIFICATE_RECORD = "FEEBL_ADAPTIVE_GRID_CERTIFICATE_V1"
GRID_CERTIFICATE_SCHEMA = "feebl.adaptive-grid-certificate.v1"
MACROPARTICLE_SUMMARY_RECORD = "FEEBL_ADAPTIVE_MACROPARTICLE_SUMMARY_V1"
GRID_CERTIFICATE_FIELD_ORDER = (
    "schema",
    "requested_mode",
    "actual_mode",
    "error_norm_policy",
    "source_model",
    "error_budget_sha256",
    "card_hash",
    "error_target",
    "error_quantile",
    "error_reference_field",
    "error_absolute_floor_fraction",
    "error_catastrophe_absolute_fraction",
    "error_exposure_limit",
    "maximum_beam_above_target_exposure",
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
GRID_CERTIFICATE_FIELDS = frozenset(GRID_CERTIFICATE_FIELD_ORDER)

REQUIRED_SUCCESS_FILES = (
    "original_acc.dat",
    "acc.dat",
    "out.dat",
    "log.txt",
    "time.txt",
    "time.json",
    "exit_code",
    "lumi.ee.out.dLdW.fne",
    "gaussian_core.json",
    "gaussian_core.csv",
    "gaussian_core_fit.log",
    "adaptive_telemetry.json",
    "environment.json",
    "run_request.json",
)
ADAPTIVE_REQUIRED_FILES = ("frozen_adaptive_policy.json",)
RESERVED_RUN_PATHS = frozenset(
    REQUIRED_SUCCESS_FILES
    + ADAPTIVE_REQUIRED_FILES
    + ("sha256_manifest.json", "run_receipt.json", "harness_error.txt")
)

CERTIFICATE_BOUND_ADAPTIVE_KEYS = frozenset(
    {
        "adaptive_grid_mode",
        "adaptive_grid_error_norm_policy",
        "adaptive_grid_error_budget_sha256",
        "adaptive_grid_source_model",
        "adaptive_grid_error_target",
        "adaptive_grid_error_quantile",
        "adaptive_grid_error_catastrophe_limit",
        "adaptive_grid_error_reference_field",
        "adaptive_grid_error_absolute_floor_fraction",
        "adaptive_grid_error_catastrophe_absolute_fraction",
        "adaptive_grid_error_exposure_limit",
        "adaptive_grid_pilot_diagnostics",
        "adaptive_grid_fail_on_unmet_error",
        "adaptive_grid_terminal_patch_certification",
    }
)

PROTECTED_PHYSICS_KEYS = frozenset(
    {
        "energy",
        "particles",
        "beta_x",
        "beta_y",
        "emitt_x",
        "emitt_y",
        "sigma_z",
        "espread.1",
        "espread.2",
        "which_espread.1",
        "which_espread.2",
        "dist_z",
        "f_rep",
        "n_b",
        "waist_y",
        "angle_x.1",
        "angle_x.2",
        "angle_phi.1",
        "angle_phi.2",
        "cut_x",
        "cut_y",
        "cut_z",
        "force_symmetric",
        "electron_ratio",
        "do_photons",
        "store_photons",
        "photon_ratio",
        "ecm_min",
        "do_hadrons",
        "store_hadrons",
        "do_coherent",
        "do_trident",
        "do_pairs",
        "track_pairs",
        "store_pairs",
        "do_compt",
        "do_muons",
        "grids",
        "integration_method",
        "electron_distribution_rho",
        "rndm_load",
        "rndm_save",
        "rndm_seed",
        "load_beam",
        "charge_sign",
        "do_espread",
        "do_eloss",
        "waist_mode",
        "waist_fraction",
        "crab_full_crossing_angle",
        "crab_longitudinal_sign.1",
        "crab_longitudinal_sign.2",
        "waist_symplecticity_tolerance",
        "store_native",
        "do_lumi",
        "num_lumi",
        "ip_dispersion.1",
        "ip_dispersion.2",
        "spectrum_w_min",
        "spectrum_w_max",
        "spectrum_w_bins",
        "hist_ee_bins",
        "hist_ee_max",
    }
)

SHARED_NUMERICAL_KEYS = frozenset({"n_x", "n_y", "n_z", "n_m", "n_t"})
BACKEND_KEY_RE = re.compile(
    r"^(?:compute_backend|allow_backend_fallback|reduction_mode|rng_backend|"
    r"cpu_.+|gpu_.+|electron_lumi_.+|photon_lumi_.+|wavefront_.+|"
    r"collision_schedule|sparse_cells|sparse_direct_interaction_limit)$"
)
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
HEX256_RE = re.compile(r"^[0-9a-f]{64}$")

TELEMETRY_FIELD_GROUPS: Mapping[str, frozenset[str]] = {
    "roots": frozenset(
        {
            "adaptive_macroparticle_roots1",
            "live_distinct_roots1",
            "distinct_roots",
            "root_visits1",
        }
    ),
    "representatives": frozenset(
        {
            "adaptive_macroparticle_representatives1",
            "representative_visits1",
            "peak_representatives1",
        }
    ),
    "ancestry": frozenset(
        {
            "adaptive_macroparticle_many_root1",
            "many_root_representative_visits1",
            "minimum_clone_support_proxy1",
            "clone_support_proxy",
            "ancestry_exact",
            "ancestry_hash",
        }
    ),
    "ess": frozenset(
        {
            "adaptive_macroparticle_weight_ess1",
            "maximum_weight_ess1",
            "weight_ess",
        }
    ),
    "error_budget": frozenset(
        {
            "controller",
            "controller_total_bound",
            "total_bound_met",
            "error_target_met",
        }
    ),
    "resources": frozenset(
        {
            "max_bytes",
            "capacity_max_bytes",
            "memory_budget_bytes",
            "peak_tracked_device_bytes",
            "measured_device_high_water_bytes",
        }
    ),
}

TELEMETRY_ROLE_CARD_KEYS = {
    "policy": "adaptive_macroparticle_policy_save",
    "execution_trace": "adaptive_macroparticle_trace_save",
    "decision_schedule": "adaptive_grid_schedule_save",
}


class CampaignError(RuntimeError):
    """A fail-closed campaign contract violation."""


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Standard-library incomplete beta for audited Student-t quantiles."""
    if not (math.isfinite(x) and math.isfinite(a) and math.isfinite(b)):
        raise CampaignError("non-finite incomplete-beta input")
    if not (0.0 <= x <= 1.0 and a > 0.0 and b > 0.0):
        raise CampaignError("invalid incomplete-beta input")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    def fraction(aa: float, bb: float, xx: float) -> float:
        qab, qap, qam = aa + bb, aa + 1.0, aa - 1.0
        tiny = 1.0e-300
        c = 1.0
        d = 1.0 - qab * xx / qap
        d = 1.0 / (tiny if abs(d) < tiny else d)
        result = d
        for m in range(1, 10001):
            m2 = 2 * m
            term = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + term * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + term / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            result *= d * c
            term = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + term * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + term / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            delta = d * c
            result *= delta
            if abs(delta - 1.0) <= 3.0e-14:
                if not math.isfinite(result) or result <= 0.0:
                    raise CampaignError("invalid incomplete-beta fraction")
                return result
        raise CampaignError("incomplete-beta continued fraction did not converge")

    prefactor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        value = prefactor * fraction(a, b, x) / a
    else:
        value = 1.0 - prefactor * fraction(b, a, 1.0 - x) / b
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


def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    if not (0.5 < probability < 1.0) or degrees_of_freedom < 1:
        raise CampaignError("invalid Student-t quantile input")
    low, high = 0.0, 1.0
    while _student_t_cdf(high, degrees_of_freedom) < probability:
        high *= 2.0
        if high > 1.0e12:
            raise CampaignError("Student-t quantile could not be bracketed")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _student_t_cdf(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass(frozen=True)
class Assignment:
    key: str
    value: str
    value_start: int
    value_end: int


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise CampaignError(f"non-standard/nonfinite JSON number {token!r} is forbidden")


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except CampaignError:
        raise
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise CampaignError(f"cannot read JSON {path}: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_exclusive_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError as exc:
        raise CampaignError(f"refusing to overwrite preexisting file: {path}") from exc


def write_exclusive_text(path: Path, text: str) -> None:
    write_exclusive_bytes(path, text.encode("utf-8"))


def write_exclusive_json(path: Path, value: Any) -> None:
    write_exclusive_bytes(path, canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CampaignError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_file(path: Path, label: str, *, nonempty: bool = True) -> Path:
    if not path.is_file() or path.is_symlink():
        raise CampaignError(f"{label} is missing or is not a regular file: {path}")
    if nonempty and path.stat().st_size == 0:
        raise CampaignError(f"{label} is empty: {path}")
    return path


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX256_RE.fullmatch(value) is None:
        raise CampaignError(f"{label} must be a lowercase SHA-256")
    return value


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CampaignError(f"{label} must be an array")
    return value


def require_exact_keys(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise CampaignError(f"{label} has unsupported keys: {', '.join(unknown)}")


def safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        raise CampaignError(f"{label} is not a safe identifier")
    return value


def safe_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CampaignError(f"{label} must be a nonempty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or candidate == Path("."):
        raise CampaignError(f"{label} must be a contained relative path")
    if any(part in ("", ".") for part in candidate.parts):
        raise CampaignError(f"{label} contains an unsafe path component")
    return candidate


def contained(root: Path, relative: Path, label: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CampaignError(f"{label} escapes {root}") from exc
    return candidate


def selected_card_block(text: str, kind: str, name: str, label: str) -> replay.Block:
    try:
        return replay.selected_block(text, kind, name)
    except replay.ReplayError as exc:
        raise CampaignError(f"{label}: {exc}") from exc


def parse_assignments(
    text: str,
    label: str,
    *,
    kind: str = "PARAMETERS",
    name: str = "p_fccz",
) -> dict[str, Assignment]:
    block = selected_card_block(text, kind, name, label)
    try:
        parsed = replay.raw_assignments(text, block)
    except replay.ReplayError as exc:
        raise CampaignError(f"{label}: {exc}") from exc
    return {
        item.key: Assignment(
            key=item.key,
            value=item.value,
            value_start=item.value_start,
            value_end=item.value_end,
        )
        for item in parsed
    }


def decimal_assignment(assignments: Mapping[str, Assignment], key: str, label: str) -> Decimal:
    try:
        raw = assignments[key].value
    except KeyError as exc:
        raise CampaignError(f"{label} is missing required assignment {key!r}") from exc
    try:
        return Decimal(raw)
    except InvalidOperation as exc:
        raise CampaignError(f"{label} assignment {key!r} is not a decimal literal") from exc


def require_decimal_equal(
    assignments: Mapping[str, Assignment], key: str, expected: Any, label: str
) -> None:
    actual = decimal_assignment(assignments, key, label)
    try:
        wanted = Decimal(str(expected))
    except InvalidOperation as exc:
        raise CampaignError(f"{label} has invalid expected value for {key}") from exc
    if actual != wanted:
        raise CampaignError(f"{label} {key}: expected {wanted}, found {actual}")


def manifest_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CampaignError(f"{label} must be a finite decimal value")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise CampaignError(f"{label} must be a finite decimal value") from exc
    if not result.is_finite():
        raise CampaignError(f"{label} must be a finite decimal value")
    return result


def validate_run_physics_identity(
    run: Mapping[str, Any], common: Mapping[str, Any], label: str
) -> tuple[Decimal, int, str]:
    sigma_mm = manifest_decimal(run.get("sigma_z_mm"), f"{label}.sigma_z_mm")
    if sigma_mm not in SUPPORTED_SIGMA_Z_MM:
        raise CampaignError(f"{label}.sigma_z_mm must be exactly 0.2265625 or 14.5")
    sigma_card_um = manifest_decimal(
        run.get("sigma_z_card_micrometres"),
        f"{label}.sigma_z_card_micrometres",
    )
    if sigma_card_um != Decimal(1000) * sigma_mm:
        raise CampaignError(f"{label} sigma_z_card_micrometres must equal 1000*sigma_z_mm")
    seed = run.get("rndm_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 1 or seed > 2_147_483_647:
        raise CampaignError(f"{label}.rndm_seed must be an integer in [1, 2147483647]")
    variant = run.get("variant")
    if variant not in ("opp", "same"):
        raise CampaignError(f"{label}.variant must be exactly 'opp' or 'same'")
    dispersion_1 = manifest_decimal(
        common.get("ip_dispersion_1_m"), "common_card_contract.ip_dispersion_1_m"
    )
    dispersion_2 = manifest_decimal(run.get("ip_dispersion_2_m"), f"{label}.ip_dispersion_2_m")
    expected_dispersion_2 = -dispersion_1 if variant == "opp" else dispersion_1
    if dispersion_2 != expected_dispersion_2:
        raise CampaignError(
            f"{label}.ip_dispersion_2_m must equal "
            f"{'-' if variant == 'opp' else '+'}common_card_contract.ip_dispersion_1_m"
        )
    return sigma_mm, seed, variant


def validate_preregistered_unseen_seed(seed: int) -> None:
    if not (PREREGISTERED_UNSEEN_SEED_MIN <= seed <= PREREGISTERED_UNSEEN_SEED_MAX):
        raise CampaignError(
            "additional manifest seed is outside the reserved preregistered-"
            f"unseen namespace [{PREREGISTERED_UNSEEN_SEED_MIN}, "
            f"{PREREGISTERED_UNSEEN_SEED_MAX}]"
        )


def overlay_card(
    source_text: str, overrides: Mapping[str, str], label: str
) -> tuple[str, list[dict[str, Any]]]:
    source = parse_assignments(source_text, f"{label} source card")
    edits: list[tuple[int, int, str]] = []
    inserted: list[tuple[str, str]] = []
    changes: list[dict[str, Any]] = []
    for key, value in overrides.items():
        if not isinstance(value, str) or not value or ";" in value or "\n" in value:
            raise CampaignError(f"{label} override {key!r} has an unsafe value")
        if key in source:
            item = source[key]
            edits.append((item.value_start, item.value_end, value))
            changes.append(
                {
                    "key": key,
                    "operation": "replace",
                    "source_value": item.value,
                    "effective_value": value,
                }
            )
        else:
            inserted.append((key, value))
            changes.append(
                {
                    "key": key,
                    "operation": "insert",
                    "source_value": None,
                    "effective_value": value,
                }
            )

    effective = source_text
    for start, end, value in sorted(edits, reverse=True):
        effective = effective[:start] + value + effective[end:]

    if inserted:
        parameter_block = selected_card_block(
            effective, "PARAMETERS", "p_fccz", f"{label} source card"
        )
        payload = "\n" + "".join(f"{key}={value};\n" for key, value in inserted)
        effective = (
            effective[: parameter_block.body_end] + payload + effective[parameter_block.body_end :]
        )

    result = parse_assignments(effective, f"{label} effective card")
    if set(result) != set(source) | set(overrides):
        raise CampaignError(f"{label} effective-card assignment set drifted")
    for key, assignment in source.items():
        wanted = overrides.get(key, assignment.value)
        if result[key].value != wanted:
            raise CampaignError(f"{label} failed to preserve assignment {key!r}")
    for key, value in overrides.items():
        if result[key].value != value:
            raise CampaignError(f"{label} failed to apply override {key!r}")
    return effective, changes


def load_config(
    path: Path,
    manifest: Path,
    template: Path,
    binary: Path,
    fitter: Path,
) -> dict[str, Any]:
    config = require_mapping(load_json(path), "config")
    require_exact_keys(
        config,
        {
            "schema",
            "schema_version",
            "campaign_id",
            "finalized",
            "adaptive_arm_id",
            "input_sha256",
            "card_overlay",
            "execution",
            "telemetry",
            "statistics",
            "additional_manifest_sha256",
            "notes",
        },
        "config",
    )
    if config.get("schema") != CONFIG_SCHEMA or config.get("schema_version") != 1:
        raise CampaignError("unsupported config schema")
    safe_id(config.get("campaign_id"), "config.campaign_id")
    if config.get("finalized") is not True:
        raise CampaignError("config.finalized must be true")
    arm_id = safe_id(config.get("adaptive_arm_id"), "config.adaptive_arm_id")
    if arm_id == "control":
        raise CampaignError("adaptive_arm_id may not be 'control'")

    expected = require_mapping(config.get("input_sha256"), "config.input_sha256")
    require_exact_keys(
        expected,
        {"frozen_manifest", "adaptive_template", "binary", "gaussian_core_fitter"},
        "config.input_sha256",
    )
    for key, item in (
        ("frozen_manifest", manifest),
        ("adaptive_template", template),
        ("binary", binary),
        ("gaussian_core_fitter", fitter),
    ):
        wanted = require_hash(expected.get(key), f"config.input_sha256.{key}")
        actual = sha256_file(require_file(item, key))
        if actual != wanted:
            raise CampaignError(f"{key} SHA-256 mismatch: expected {wanted}, found {actual}")

    overlay = require_mapping(config.get("card_overlay"), "config.card_overlay")
    require_exact_keys(
        overlay,
        {
            "backend_keys",
            "shared_numerical_keys",
            "adaptive_keys",
            "control_values",
        },
        "config.card_overlay",
    )
    categories: dict[str, list[str]] = {}
    for name in ("backend_keys", "shared_numerical_keys", "adaptive_keys"):
        raw = require_list(overlay.get(name), f"config.card_overlay.{name}")
        if not raw or not all(isinstance(item, str) for item in raw):
            raise CampaignError(f"config.card_overlay.{name} must be a nonempty string array")
        if len(set(raw)) != len(raw):
            raise CampaignError(f"config.card_overlay.{name} contains duplicates")
        categories[name] = raw
    all_keys = [item for values in categories.values() for item in values]
    if len(set(all_keys)) != len(all_keys):
        raise CampaignError("card overlay key categories overlap")
    forbidden = sorted(set(all_keys) & PROTECTED_PHYSICS_KEYS)
    if forbidden:
        raise CampaignError("protected physics keys may not be overridden: " + ", ".join(forbidden))
    for key in categories["backend_keys"]:
        if BACKEND_KEY_RE.fullmatch(key) is None:
            raise CampaignError(f"backend key is not narrowly classified: {key}")
    for key in categories["shared_numerical_keys"]:
        if key not in SHARED_NUMERICAL_KEYS:
            raise CampaignError(f"unsupported shared numerical key: {key}")
    for key in categories["adaptive_keys"]:
        if not key.startswith("adaptive_"):
            raise CampaignError(f"adaptive key does not begin with 'adaptive_': {key}")
    missing_certificate_keys = sorted(
        CERTIFICATE_BOUND_ADAPTIVE_KEYS - set(categories["adaptive_keys"])
    )
    if missing_certificate_keys:
        raise CampaignError(
            "adaptive_keys must explicitly bind every terminal certificate setting; "
            f"missing={missing_certificate_keys}"
        )

    template_text = require_file(template, "adaptive template").read_text(encoding="utf-8")
    template_assignments = parse_assignments(template_text, "adaptive template")
    if set(template_assignments) != set(all_keys):
        missing = sorted(set(all_keys) - set(template_assignments))
        extra = sorted(set(template_assignments) - set(all_keys))
        raise CampaignError(
            f"adaptive template must contain exactly classified keys; missing={missing}, extra={extra}"
        )
    for mode_key in ("adaptive_grid_mode", "adaptive_macroparticle_mode"):
        if mode_key not in template_assignments:
            raise CampaignError(f"adaptive template must explicitly set {mode_key}")
        try:
            mode = int(template_assignments[mode_key].value)
        except ValueError as exc:
            raise CampaignError(f"{mode_key} must be an integer literal") from exc
        valid_modes = (2, 3) if mode_key == "adaptive_grid_mode" else (1, 2, 3)
        if mode not in valid_modes:
            raise CampaignError(f"adaptive template {mode_key} must be one of {valid_modes}")

    error_policy = _card_string_assignment(
        template_assignments,
        "adaptive_grid_error_norm_policy",
        "adaptive template",
    )
    if error_policy not in {
        "component_rms_linf_v1",
        "vector_rms_quantile_v1",
        "vector_mixed_rms_quantile_v2",
    }:
        raise CampaignError("adaptive template has an unknown grid error policy")
    budget_sha256 = _card_quoted_string_assignment_allow_empty(
        template_assignments,
        "adaptive_grid_error_budget_sha256",
        "adaptive template",
    )
    mixed_values = {
        key: decimal_assignment(template_assignments, key, "adaptive template")
        for key in (
            "adaptive_grid_error_reference_field",
            "adaptive_grid_error_absolute_floor_fraction",
            "adaptive_grid_error_catastrophe_absolute_fraction",
            "adaptive_grid_error_exposure_limit",
        )
    }
    if any(not value.is_finite() for value in mixed_values.values()):
        raise CampaignError("adaptive template has a nonfinite mixed-V2 control")
    if error_policy == "vector_mixed_rms_quantile_v2":
        if HEX256_RE.fullmatch(budget_sha256) is None:
            raise CampaignError("adaptive template mixed-V2 budget must be a lowercase SHA-256")
        if not (
            mixed_values["adaptive_grid_error_reference_field"] > 0
            and 0 < mixed_values["adaptive_grid_error_absolute_floor_fraction"] <= 1
            and 0 < mixed_values["adaptive_grid_error_catastrophe_absolute_fraction"] <= 2
            and 0 < mixed_values["adaptive_grid_error_exposure_limit"] < 1
        ):
            raise CampaignError("adaptive template has invalid mixed-V2 controls")
    elif budget_sha256 != "" or any(value != 0 for value in mixed_values.values()):
        raise CampaignError(
            "adaptive template non-mixed policy must leave the mixed-V2 "
            "budget and controls empty/zero"
        )

    controls = require_mapping(overlay.get("control_values"), "control_values")
    if set(controls) != {"adaptive_grid_mode", "adaptive_macroparticle_mode"}:
        raise CampaignError("control_values must set exactly both adaptive mode selectors")
    for key, value in controls.items():
        if value != "0":
            raise CampaignError(f"control_values.{key} must be the exact string '0'")

    execution = require_mapping(config.get("execution"), "config.execution")
    require_exact_keys(
        execution,
        {"argv", "timeout_seconds", "inherit_environment", "environment"},
        "config.execution",
    )
    argv = require_list(execution.get("argv"), "config.execution.argv")
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise CampaignError("execution.argv must be a nonempty string array")
    if sum(token.count("{binary}") for token in argv) != 1:
        raise CampaignError("execution.argv must contain {binary} exactly once")
    allowed_placeholders = {"{binary}", "{python}", "{run_id}", "{arm}"}
    for token in argv:
        for placeholder in re.findall(r"\{[^{}]+\}", token):
            if placeholder not in allowed_placeholders:
                raise CampaignError(f"unsupported execution placeholder: {placeholder}")
    timeout = execution.get("timeout_seconds")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise CampaignError("execution.timeout_seconds must be positive")
    inherited = require_list(
        execution.get("inherit_environment"), "config.execution.inherit_environment"
    )
    if not all(isinstance(item, str) and item and "=" not in item for item in inherited):
        raise CampaignError("inherit_environment contains an invalid name")
    if len(set(inherited)) != len(inherited):
        raise CampaignError("inherit_environment contains duplicates")
    environment = require_mapping(execution.get("environment"), "config.execution.environment")
    for key, value in environment.items():
        if not isinstance(key, str) or not key or "=" in key or not isinstance(value, str):
            raise CampaignError("execution.environment must map valid names to strings")
        if "\x00" in key or "\x00" in value:
            raise CampaignError("execution.environment contains NUL")

    telemetry = require_mapping(config.get("telemetry"), "config.telemetry")
    require_exact_keys(
        telemetry,
        {"artifacts", "required_record_prefixes", "required_fields"},
        "config.telemetry",
    )
    artifacts = require_list(telemetry.get("artifacts"), "config.telemetry.artifacts")
    roles: set[str] = set()
    paths: set[str] = set()
    for index, raw in enumerate(artifacts):
        item = require_mapping(raw, f"telemetry.artifacts[{index}]")
        require_exact_keys(item, {"role", "path"}, f"telemetry.artifacts[{index}]")
        role = item.get("role")
        if role not in TELEMETRY_ROLE_CARD_KEYS:
            raise CampaignError(f"unsupported telemetry artifact role: {role!r}")
        rel = safe_relative_path(item.get("path"), f"telemetry.artifacts[{index}].path")
        if rel.as_posix() in RESERVED_RUN_PATHS:
            raise CampaignError(
                f"telemetry artifact path is reserved by the harness: {rel.as_posix()}"
            )
        roles.add(role)
        paths.add(rel.as_posix())
        configured_path = _card_string_assignment(
            template_assignments,
            TELEMETRY_ROLE_CARD_KEYS[role],
            "adaptive template",
        )
        if rel.as_posix() != Path(configured_path).as_posix():
            raise CampaignError(
                f"telemetry {role} artifact does not match "
                f"{TELEMETRY_ROLE_CARD_KEYS[role]} in the adaptive template"
            )
    if roles != {"policy", "execution_trace", "decision_schedule"} or len(paths) != 3:
        raise CampaignError("telemetry must declare three distinct policy/trace/schedule artifacts")
    prefixes = require_list(
        telemetry.get("required_record_prefixes"),
        "config.telemetry.required_record_prefixes",
    )
    if not prefixes or not all(
        isinstance(item, str) and item.startswith("FEEBL_ADAPTIVE_") for item in prefixes
    ):
        raise CampaignError("required_record_prefixes must name adaptive FeeBL records")
    if len(set(prefixes)) != len(prefixes):
        raise CampaignError("required_record_prefixes contains duplicates")
    if GRID_CERTIFICATE_RECORD not in prefixes:
        raise CampaignError(f"required_record_prefixes must include {GRID_CERTIFICATE_RECORD}")
    fields = require_list(telemetry.get("required_fields"), "config.telemetry.required_fields")
    if not all(isinstance(item, str) and item for item in fields):
        raise CampaignError("telemetry.required_fields must be a string array")

    statistical = require_mapping(config.get("statistics"), "config.statistics")
    require_exact_keys(
        statistical,
        {
            "minimum_unseen_seeds_per_configuration",
            "relative_equivalence_margin",
            "expected_relative_difference",
            "equivalence_alpha",
            "equivalence_power",
            "standard_error_safety_factor",
            "maximum_supported_unseen_seeds_per_configuration",
        },
        "config.statistics",
    )
    minimum_unseen = statistical.get("minimum_unseen_seeds_per_configuration")
    maximum_unseen = statistical.get("maximum_supported_unseen_seeds_per_configuration")
    if (
        not isinstance(minimum_unseen, int)
        or isinstance(minimum_unseen, bool)
        or minimum_unseen < 4
    ):
        raise CampaignError("minimum_unseen_seeds_per_configuration must be an integer >= 4")
    if (
        not isinstance(maximum_unseen, int)
        or isinstance(maximum_unseen, bool)
        or maximum_unseen < minimum_unseen
    ):
        raise CampaignError(
            "maximum_supported_unseen_seeds_per_configuration must be >= the minimum"
        )
    decimals: dict[str, Decimal] = {}
    for key in (
        "relative_equivalence_margin",
        "expected_relative_difference",
        "equivalence_alpha",
        "equivalence_power",
        "standard_error_safety_factor",
    ):
        raw = statistical.get(key)
        if not isinstance(raw, str):
            raise CampaignError(f"config.statistics.{key} must be an exact decimal string")
        try:
            value = Decimal(raw)
        except InvalidOperation as exc:
            raise CampaignError(f"config.statistics.{key} is invalid") from exc
        if not value.is_finite():
            raise CampaignError(f"config.statistics.{key} must be finite")
        decimals[key] = value
    margin = decimals["relative_equivalence_margin"]
    expected_difference = decimals["expected_relative_difference"]
    alpha = decimals["equivalence_alpha"]
    power = decimals["equivalence_power"]
    safety = decimals["standard_error_safety_factor"]
    if margin <= 0 or margin > Decimal("0.01") or abs(expected_difference) >= margin:
        raise CampaignError(
            "statistics require 0 < margin <= 0.01 and |expected difference| < margin"
        )
    if alpha <= 0 or alpha >= Decimal("0.5"):
        raise CampaignError("statistics.equivalence_alpha must be in (0, 0.5)")
    if power <= Decimal("0.5") or power >= 1:
        raise CampaignError("statistics.equivalence_power must be in (0.5, 1)")
    if safety < 1 or safety > 10:
        raise CampaignError("statistics.standard_error_safety_factor must be in [1, 10]")

    additional_hash = config.get("additional_manifest_sha256")
    if additional_hash is not None:
        require_hash(additional_hash, "config.additional_manifest_sha256")
    return config


def _validate_run_card(
    run: Mapping[str, Any], card: Path, common: Mapping[str, Any], label: str
) -> tuple[Decimal, int, str]:
    identity = validate_run_physics_identity(run, common, label)
    wanted_hash = require_hash(run.get("card_sha256"), f"{label}.card_sha256")
    actual_hash = sha256_file(require_file(card, f"{label} source card"))
    if wanted_hash != actual_hash:
        raise CampaignError(
            f"{label} source-card SHA-256 mismatch: expected {wanted_hash}, found {actual_hash}"
        )
    card_text = card.read_text(encoding="utf-8")
    accelerator_assignments = parse_assignments(
        card_text,
        f"{label} source card",
        kind="ACCELERATOR",
        name="fccz",
    )
    parameter_assignments = parse_assignments(
        card_text, f"{label} source card", kind="PARAMETERS", name="p_fccz"
    )
    require_decimal_equal(accelerator_assignments, "energy", common["beam_energy_gev"], label)
    require_decimal_equal(
        accelerator_assignments,
        "sigma_z",
        run["sigma_z_card_micrometres"],
        label,
    )
    expected_parameters = {
        "n_x": common["n_x"],
        "n_y": common["n_y"],
        "n_z": common["n_z"],
        "n_m": common["n_m"],
        "num_lumi": common["num_lumi"],
        "crab_full_crossing_angle": common["crab_full_crossing_angle_rad"],
        "crab_longitudinal_sign.1": common["crab_longitudinal_sign_1"],
        "crab_longitudinal_sign.2": common["crab_longitudinal_sign_2"],
        "ip_dispersion.1": common["ip_dispersion_1_m"],
        "spectrum_w_min": common["spectrum_w_min_gev"],
        "spectrum_w_max": common["spectrum_w_max_gev"],
        "spectrum_w_bins": common["spectrum_w_bins"],
    }
    for key, expected in expected_parameters.items():
        require_decimal_equal(parameter_assignments, key, expected, label)
    require_decimal_equal(parameter_assignments, "rndm_seed", run["rndm_seed"], label)
    require_decimal_equal(parameter_assignments, "ip_dispersion.2", run["ip_dispersion_2_m"], label)
    if any(key.startswith("adaptive_") for key in parameter_assignments):
        raise CampaignError(f"{label} source card already contains adaptive assignments")
    return identity


def load_run_manifests(
    manifest_path: Path,
    source_root: Path,
    additional_path: Path | None,
    additional_expected_hash: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    main = require_mapping(load_json(manifest_path), "frozen manifest")
    if main.get("schema_version") != 1:
        raise CampaignError("unsupported frozen manifest schema_version")
    if main.get("pristine_unseen_qualification_holdout") is not False:
        raise CampaignError("frozen manifest must retain its non-pristine caveat")
    common = require_mapping(main.get("common_card_contract"), "common_card_contract")
    raw_runs = require_list(main.get("runs"), "frozen manifest runs")
    if len(raw_runs) != 8:
        raise CampaignError("frozen manifest must contain exactly eight runs")
    source_root = source_root.resolve()
    all_runs: list[dict[str, Any]] = []
    identities: set[tuple[Decimal, int, str]] = set()
    physical_card_paths: set[Path] = set()
    for index, raw in enumerate(raw_runs):
        run = require_mapping(raw, f"runs[{index}]")
        run_id = safe_id(run.get("run_id"), f"runs[{index}].run_id")
        rel = safe_relative_path(run.get("card_relative_path"), f"{run_id}.card_relative_path")
        card = contained(source_root, rel, f"{run_id}.card_relative_path")
        if card in physical_card_paths:
            raise CampaignError(f"duplicate physical card path: {card}")
        physical_card_paths.add(card)
        identity = _validate_run_card(run, card, common, run_id)
        if identity in identities:
            raise CampaignError(f"duplicate physics identity: {identity}")
        identities.add(identity)
        normalized = dict(run)
        normalized["origin"] = "frozen"
        normalized["source_card_path"] = str(card)
        all_runs.append(normalized)
    expected = {
        (sigma, seed, variant)
        for sigma in SUPPORTED_SIGMA_Z_MM
        for seed in (1, 2)
        for variant in ("opp", "same")
    }
    if identities != expected:
        raise CampaignError("frozen 2 x 2 x 2 control matrix is incomplete")

    additional: dict[str, Any] | None = None
    if additional_expected_hash is None and additional_path is not None:
        raise CampaignError("additional cards are forbidden without a preregistered config hash")
    if additional_expected_hash is not None and additional_path is None:
        raise CampaignError("config preregisters an additional manifest but none was supplied")
    if additional_path is not None:
        actual = sha256_file(require_file(additional_path, "additional manifest"))
        if actual != additional_expected_hash:
            raise CampaignError(
                f"additional manifest SHA-256 mismatch: expected {additional_expected_hash}, found {actual}"
            )
        additional = require_mapping(load_json(additional_path), "additional manifest")
        if (
            additional.get("schema") != UNSEEN_SCHEMA
            or additional.get("schema_version") != 1
            or additional.get("preregistered") is not True
            or not isinstance(additional.get("preregistered_at_utc"), str)
        ):
            raise CampaignError(
                "additional manifest is not an explicit preregistered unseen-seed input"
            )
        extra_runs = require_list(additional.get("runs"), "additional manifest runs")
        if not extra_runs:
            raise CampaignError("additional manifest has no runs")
        extra_by_seed: dict[int, set[tuple[Decimal, str]]] = {}
        for index, raw in enumerate(extra_runs):
            run = require_mapping(raw, f"additional.runs[{index}]")
            run_id = safe_id(run.get("run_id"), f"additional.runs[{index}].run_id")
            seed_value = run.get("rndm_seed")
            if (
                not isinstance(seed_value, int)
                or isinstance(seed_value, bool)
                or seed_value < 1
                or seed_value > 2_147_483_647
            ):
                raise CampaignError(
                    f"additional {run_id} rndm_seed must be an integer in [1, 2147483647]"
                )
            seed = seed_value
            validate_preregistered_unseen_seed(seed)
            rel = safe_relative_path(
                run.get("card_relative_path"), f"additional {run_id}.card_relative_path"
            )
            card = contained(source_root, rel, f"additional {run_id}.card_relative_path")
            if card in physical_card_paths:
                raise CampaignError(f"duplicate physical card path: {card}")
            physical_card_paths.add(card)
            identity = _validate_run_card(run, card, common, f"additional {run_id}")
            if identity in identities:
                raise CampaignError(f"duplicate additional physics identity: {identity}")
            identities.add(identity)
            extra_by_seed.setdefault(seed, set()).add((identity[0], identity[2]))
            normalized = dict(run)
            normalized["origin"] = "preregistered_unseen"
            normalized["source_card_path"] = str(card)
            all_runs.append(normalized)
        expected_per_seed = {
            (sigma, variant) for sigma in SUPPORTED_SIGMA_Z_MM for variant in ("opp", "same")
        }
        for seed, observed in extra_by_seed.items():
            if observed != expected_per_seed:
                raise CampaignError(
                    f"additional seed {seed} does not provide the complete 2 x 2 matrix"
                )
    run_ids = [str(run["run_id"]) for run in all_runs]
    if len(set(run_ids)) != len(run_ids):
        raise CampaignError("run_id values collide across frozen/additional manifests")
    return main, all_runs, additional


def _historical_entry(
    category: str, label: str, run: Mapping[str, Any], directory: Path
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "category": category,
        "label": label,
        "run_id": run["run_id"],
        "directory": str(directory.resolve()),
        "available": False,
    }
    if not directory.exists():
        result["reason"] = "directory_absent_at_freeze"
        return result
    if not directory.is_dir() or directory.is_symlink():
        result["reason"] = "candidate_is_not_a_regular_directory"
        return result
    required = ("acc.dat", "out.dat", "lumi.ee.out.dLdW.fne")
    missing = [
        name
        for name in required
        if not (directory / name).is_file()
        or (directory / name).is_symlink()
        or (directory / name).stat().st_size == 0
    ]
    if missing:
        result["reason"] = "partial_evidence_missing:" + ",".join(missing)
        return result
    try:
        card_text = (directory / "acc.dat").read_text(encoding="utf-8")
        accelerator_assignments = parse_assignments(
            card_text,
            f"historical {label} card",
            kind="ACCELERATOR",
            name="fccz",
        )
        parameter_assignments = parse_assignments(
            card_text,
            f"historical {label} card",
            kind="PARAMETERS",
            name="p_fccz",
        )
        require_decimal_equal(
            accelerator_assignments,
            "sigma_z",
            run["sigma_z_card_micrometres"],
            label,
        )
        require_decimal_equal(parameter_assignments, "rndm_seed", run["rndm_seed"], label)
        require_decimal_equal(
            parameter_assignments, "ip_dispersion.2", run["ip_dispersion_2_m"], label
        )
    except (CampaignError, OSError, UnicodeError) as exc:
        result["reason"] = f"identity_validation_failed:{exc}"
        return result
    files: dict[str, Any] = {}
    for name in ("acc.dat", "out.dat", "lumi.ee.out.dLdW.fne", "log.txt"):
        path = directory / name
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0:
            files[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
    result["available"] = True
    result["files"] = files
    return result


def discover_historical(run: Mapping[str, Any], source_root: Path) -> list[dict[str, Any]]:
    if run.get("origin") != "frozen":
        return []
    source_card = Path(str(run["source_card_path"]))
    entries = [
        _historical_entry("fixed512", "fixed512_cpu", run, source_card.parent / "cpu"),
        _historical_entry("fixed512", "fixed512_gpu", run, source_card.parent / "gpu"),
    ]
    convergence = source_root / "benchmark_results" / "FeeBL_FCCZ_convergence_8_20260830"
    entries.append(
        _historical_entry("xgrid1024", "xgrid1024", run, convergence / f"xgrid_{run['run_id']}")
    )
    entries.append(
        _historical_entry("xstat1m", "xstat1m", run, convergence / f"xstat_{run['run_id']}")
    )
    return entries


def freeze_campaign(args: argparse.Namespace) -> str:
    manifest_path = require_file(args.manifest.resolve(), "frozen manifest")
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise CampaignError(f"source root is not a directory: {source_root}")
    template = require_file(args.adaptive_template.resolve(), "adaptive template")
    binary = require_file(args.binary.resolve(), "FeeBL binary")
    fitter = require_file(args.fitter.resolve(), "Gaussian-core fitter")
    config_path = require_file(args.config.resolve(), "finalized config")
    config = load_config(config_path, manifest_path, template, binary, fitter)
    main, source_runs, additional = load_run_manifests(
        manifest_path,
        source_root,
        args.additional_manifest.resolve() if args.additional_manifest else None,
        config.get("additional_manifest_sha256"),
    )
    template_text = template.read_text(encoding="utf-8")
    template_assignments = parse_assignments(template_text, "adaptive template")
    overlay = config["card_overlay"]
    backend_and_shared = list(overlay["backend_keys"]) + list(overlay["shared_numerical_keys"])
    adaptive_keys = list(overlay["adaptive_keys"])
    adaptive_overrides = {
        key: template_assignments[key].value for key in backend_and_shared + adaptive_keys
    }
    control_overrides = {key: template_assignments[key].value for key in backend_and_shared}
    control_overrides.update(overlay["control_values"])

    prepared: list[dict[str, Any]] = []
    for source_run in source_runs:
        source_path = Path(str(source_run["source_card_path"]))
        source_text = source_path.read_text(encoding="utf-8")
        control_text, control_changes = overlay_card(
            source_text, control_overrides, f"{source_run['run_id']} control"
        )
        adaptive_text, adaptive_changes = overlay_card(
            source_text, adaptive_overrides, f"{source_run['run_id']} adaptive"
        )
        prepared.append(
            {
                "source": source_run,
                "source_bytes": source_path.read_bytes(),
                "control_bytes": control_text.encode("utf-8"),
                "control_changes": control_changes,
                "adaptive_bytes": adaptive_text.encode("utf-8"),
                "adaptive_changes": adaptive_changes,
            }
        )

    output = args.output.resolve()
    if output.exists():
        raise CampaignError(f"refusing preexisting campaign output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    (output / "inputs").mkdir()
    (output / "cards").mkdir()
    (output / "runs").mkdir()
    shutil.copyfile(manifest_path, output / "inputs" / "frozen_holdout_manifest.json")
    shutil.copyfile(config_path, output / "inputs" / "finalized_config.json")
    shutil.copyfile(template, output / "inputs" / "adaptive_template.dat")
    shutil.copyfile(fitter, output / "inputs" / "gaussian_core_spectrum.py")
    if args.additional_manifest:
        shutil.copyfile(
            args.additional_manifest.resolve(), output / "inputs" / "additional_manifest.json"
        )
    frozen_policy = {
        "schema": "feebl.fccz-realization-independent-adaptive-policy.v1",
        "campaign_id": config["campaign_id"],
        "adaptive_arm_id": config["adaptive_arm_id"],
        "realization_scope": "independent_of_run_identity_seed_and_outputs",
        "adaptive_template_sha256": sha256_file(template),
        "binary_sha256": sha256_file(binary),
        "classified_template_values": {
            key: template_assignments[key].value
            for key in list(overlay["backend_keys"])
            + list(overlay["shared_numerical_keys"])
            + list(overlay["adaptive_keys"])
        },
        "control_values": overlay["control_values"],
        "telemetry_contract": config["telemetry"],
    }
    write_exclusive_json(output / "inputs" / "frozen_adaptive_policy.json", frozen_policy)

    plan_runs: list[dict[str, Any]] = []
    historical: list[dict[str, Any]] = []
    sequence = 0
    for item in prepared:
        source_run = item["source"]
        card_dir = output / "cards" / str(source_run["run_id"])
        card_dir.mkdir()
        write_exclusive_bytes(card_dir / "original_acc.dat", item["source_bytes"])
        write_exclusive_bytes(card_dir / "control_acc.dat", item["control_bytes"])
        adaptive_name = f"{config['adaptive_arm_id']}_acc.dat"
        write_exclusive_bytes(card_dir / adaptive_name, item["adaptive_bytes"])
        preservation = {
            "schema": "feebl.fccz-card-preservation.v1",
            "run_id": source_run["run_id"],
            "source_card_sha256": sha256_bytes(item["source_bytes"]),
            "source_assignment_count": len(
                parse_assignments(item["source_bytes"].decode("utf-8"), "source card")
            ),
            "control": {
                "effective_card_sha256": sha256_bytes(item["control_bytes"]),
                "changes": item["control_changes"],
            },
            config["adaptive_arm_id"]: {
                "effective_card_sha256": sha256_bytes(item["adaptive_bytes"]),
                "changes": item["adaptive_changes"],
            },
            "physics_assignment_policy": "all_source_assignments_exact_except_classified_overlay_keys",
        }
        write_exclusive_json(card_dir / "preservation_receipt.json", preservation)
        card_files = {
            path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted(card_dir.iterdir())
        }
        for arm, filename in (
            ("control", "control_acc.dat"),
            (config["adaptive_arm_id"], adaptive_name),
        ):
            sequence += 1
            run_key = f"{sequence:04d}_{source_run['run_id']}_{arm}"
            plan_runs.append(
                {
                    "sequence": sequence,
                    "run_key": run_key,
                    "run_id": source_run["run_id"],
                    "origin": source_run["origin"],
                    "sigma_z_mm": source_run["sigma_z_mm"],
                    "rndm_seed": source_run["rndm_seed"],
                    "variant": source_run["variant"],
                    "arm": arm,
                    "source_card": f"cards/{source_run['run_id']}/original_acc.dat",
                    "source_card_sha256": source_run["card_sha256"],
                    "effective_card": f"cards/{source_run['run_id']}/{filename}",
                    "effective_card_sha256": card_files[filename]["sha256"],
                    "binary_sha256": sha256_file(binary),
                }
            )
        historical.extend(discover_historical(source_run, source_root))

    python_path = Path(sys.executable).resolve()
    static_files: dict[str, Any] = {}
    for static_directory in (output / "inputs", output / "cards"):
        for relative, path in _iter_regular_files(static_directory).items():
            campaign_relative = f"{static_directory.name}/{relative}"
            static_files[campaign_relative] = {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }

    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "frozen_at_utc": utc_now(),
        "qualification_status": "diagnostic_confirmatory_not_pristine_unseen",
        "inputs": {
            "source_root": str(source_root),
            "manifest_external_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_copy": "inputs/frozen_holdout_manifest.json",
            "config_external_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config_copy": "inputs/finalized_config.json",
            "adaptive_template_external_path": str(template),
            "adaptive_template_sha256": sha256_file(template),
            "adaptive_template_copy": "inputs/adaptive_template.dat",
            "binary_path": str(binary),
            "binary_sha256": sha256_file(binary),
            "gaussian_core_fitter_external_path": str(fitter),
            "gaussian_core_fitter_sha256": sha256_file(fitter),
            "gaussian_core_fitter_copy": "inputs/gaussian_core_spectrum.py",
            "frozen_adaptive_policy_copy": "inputs/frozen_adaptive_policy.json",
            "frozen_adaptive_policy_sha256": sha256_file(
                output / "inputs" / "frozen_adaptive_policy.json"
            ),
            "python_executable_path": str(python_path),
            "python_executable_sha256": sha256_file(python_path),
            "additional_manifest_sha256": (
                sha256_file(args.additional_manifest.resolve())
                if args.additional_manifest
                else None
            ),
            "additional_manifest_copy": (
                "inputs/additional_manifest.json" if additional is not None else None
            ),
        },
        "adaptive_arm_id": config["adaptive_arm_id"],
        "execution": config["execution"],
        "telemetry": config["telemetry"],
        "statistics": config["statistics"],
        "card_overlay": config["card_overlay"],
        "runs": plan_runs,
        "historical_evidence": historical,
        "static_files": static_files,
        "statistical_contract": main["statistical_contract"],
        "frozen_run_count": 8,
        "additional_run_count": len(source_runs) - 8,
        "process_contract": "exactly_one_FeeBL_process_per_run-next_invocation",
    }
    campaign_bytes = canonical_json_bytes(campaign)
    campaign_hash = sha256_bytes(campaign_bytes)
    write_exclusive_bytes(output / "campaign.json", campaign_bytes)
    write_exclusive_text(output / "campaign.sha256", campaign_hash + "  campaign.json\n")
    return campaign_hash


def _verify_hash_record(path: Path, record: Mapping[str, Any], label: str) -> None:
    require_file(path, label)
    expected_hash = require_hash(record.get("sha256"), f"{label}.sha256")
    if sha256_file(path) != expected_hash:
        raise CampaignError(f"tampered {label}: SHA-256 mismatch")
    if path.stat().st_size != record.get("size"):
        raise CampaignError(f"tampered {label}: size mismatch")


def load_campaign(
    root: Path, expected_hash: str, *, verify_historical: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    require_hash(expected_hash, "--campaign-sha256")
    campaign_path = require_file(root / "campaign.json", "campaign.json")
    actual_hash = sha256_file(campaign_path)
    if actual_hash != expected_hash:
        raise CampaignError(
            f"campaign binding mismatch: expected {expected_hash}, found {actual_hash}"
        )
    marker = require_file(root / "campaign.sha256", "campaign.sha256")
    if marker.read_text(encoding="utf-8") != expected_hash + "  campaign.json\n":
        raise CampaignError("campaign.sha256 is tampered or noncanonical")
    campaign = require_mapping(load_json(campaign_path), "campaign")
    if campaign.get("schema") != CAMPAIGN_SCHEMA or campaign.get("schema_version") != 1:
        raise CampaignError("unsupported campaign schema")
    static_records = require_mapping(campaign.get("static_files"), "campaign.static_files")
    actual_static: dict[str, Path] = {}
    for directory_name in ("inputs", "cards"):
        directory = root / directory_name
        if not directory.is_dir() or directory.is_symlink():
            raise CampaignError(f"campaign static directory is missing: {directory}")
        for relative, path in _iter_regular_files(directory).items():
            actual_static[f"{directory_name}/{relative}"] = path
    if set(actual_static) != set(static_records):
        raise CampaignError("campaign static file set is missing, extra, or tampered")
    for name, raw_record in static_records.items():
        _verify_hash_record(
            actual_static[name], require_mapping(raw_record, name), f"static file {name}"
        )
    inputs = require_mapping(campaign.get("inputs"), "campaign.inputs")
    for copy_key, hash_key, label in (
        ("manifest_copy", "manifest_sha256", "manifest copy"),
        ("config_copy", "config_sha256", "config copy"),
        ("adaptive_template_copy", "adaptive_template_sha256", "template copy"),
        ("gaussian_core_fitter_copy", "gaussian_core_fitter_sha256", "fitter copy"),
        (
            "frozen_adaptive_policy_copy",
            "frozen_adaptive_policy_sha256",
            "frozen adaptive policy",
        ),
    ):
        path = contained(root, safe_relative_path(inputs[copy_key], copy_key), copy_key)
        if sha256_file(require_file(path, label)) != require_hash(inputs[hash_key], hash_key):
            raise CampaignError(f"tampered {label}")
    if inputs.get("additional_manifest_copy") is not None:
        path = contained(
            root,
            safe_relative_path(inputs["additional_manifest_copy"], "additional_manifest_copy"),
            "additional_manifest_copy",
        )
        if sha256_file(require_file(path, "additional manifest copy")) != require_hash(
            inputs.get("additional_manifest_sha256"), "additional_manifest_sha256"
        ):
            raise CampaignError("tampered additional manifest copy")
    for path_key, hash_key, label in (
        ("binary_path", "binary_sha256", "FeeBL binary"),
        ("python_executable_path", "python_executable_sha256", "Python executable"),
    ):
        path = Path(str(inputs[path_key]))
        if sha256_file(require_file(path, label)) != require_hash(inputs[hash_key], hash_key):
            raise CampaignError(f"{label} drifted after freeze")
    runs = require_list(campaign.get("runs"), "campaign.runs")
    if not runs:
        raise CampaignError("campaign has no runs")
    keys: set[str] = set()
    for index, raw in enumerate(runs):
        run = require_mapping(raw, f"campaign.runs[{index}]")
        run_key = safe_id(run.get("run_key"), f"campaign.runs[{index}].run_key")
        if run_key in keys:
            raise CampaignError("duplicate campaign run_key")
        keys.add(run_key)
        for path_key, hash_key in (
            ("source_card", "source_card_sha256"),
            ("effective_card", "effective_card_sha256"),
        ):
            path = contained(root, safe_relative_path(run[path_key], path_key), path_key)
            if sha256_file(require_file(path, path_key)) != require_hash(run[hash_key], hash_key):
                raise CampaignError(f"tampered planned card for {run_key}")
        if run.get("binary_sha256") != inputs["binary_sha256"]:
            raise CampaignError(f"{run_key} is not bound to the campaign binary")
    if verify_historical:
        for raw in require_list(campaign.get("historical_evidence"), "historical_evidence"):
            item = require_mapping(raw, "historical evidence")
            if item.get("available") is True:
                for name, record in require_mapping(item.get("files"), "historical files").items():
                    _verify_hash_record(Path(str(record["path"])), record, f"historical {name}")
    return campaign


def _iter_regular_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CampaignError(f"symlink is forbidden in run evidence: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            result[relative] = path
        elif not path.is_dir():
            raise CampaignError(f"special filesystem object in run evidence: {path}")
    return result


def seal_run(run_dir: Path, campaign_hash: str, run: Mapping[str, Any], status: str) -> None:
    if status not in {"success", "failed"}:
        raise CampaignError("invalid run seal status")
    files = _iter_regular_files(run_dir)
    if "sha256_manifest.json" in files or "run_receipt.json" in files:
        raise CampaignError("refusing preexisting run seal files")
    manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "run_key": run["run_key"],
        "status": status,
        "files": {
            name: {"sha256": sha256_file(path), "size": path.stat().st_size}
            for name, path in sorted(files.items())
        },
    }
    write_exclusive_json(run_dir / "sha256_manifest.json", manifest)
    manifest_hash = sha256_file(run_dir / "sha256_manifest.json")
    receipt = {
        "schema": "feebl.fccz-frozen-eight-run-receipt.v1",
        "run_key": run["run_key"],
        "run_id": run["run_id"],
        "arm": run["arm"],
        "status": status,
        "campaign_sha256": campaign_hash,
        "sha256_manifest_sha256": manifest_hash,
        "sealed_at_utc": utc_now(),
    }
    write_exclusive_json(run_dir / "run_receipt.json", receipt)


def verify_sealed_run(
    run_dir: Path, campaign_hash: str, run: Mapping[str, Any], *, require_success: bool
) -> str:
    manifest_path = require_file(run_dir / "sha256_manifest.json", "run SHA manifest")
    receipt_path = require_file(run_dir / "run_receipt.json", "run receipt")
    receipt = require_mapping(load_json(receipt_path), "run receipt")
    if (
        receipt.get("schema") != "feebl.fccz-frozen-eight-run-receipt.v1"
        or receipt.get("run_key") != run["run_key"]
        or receipt.get("run_id") != run["run_id"]
        or receipt.get("arm") != run["arm"]
        or receipt.get("campaign_sha256") != campaign_hash
        or receipt.get("sha256_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise CampaignError(f"tampered run receipt: {run_dir}")
    status = receipt.get("status")
    if status not in {"success", "failed"}:
        raise CampaignError(f"invalid sealed run status: {run_dir}")
    manifest = require_mapping(load_json(manifest_path), "run SHA manifest")
    if (
        manifest.get("schema") != RUN_MANIFEST_SCHEMA
        or manifest.get("run_key") != run["run_key"]
        or manifest.get("status") != status
    ):
        raise CampaignError(f"tampered run SHA manifest: {run_dir}")
    records = require_mapping(manifest.get("files"), "run manifest files")
    actual = _iter_regular_files(run_dir)
    expected_names = set(records) | {"sha256_manifest.json", "run_receipt.json"}
    if set(actual) != expected_names:
        raise CampaignError(f"preexisting, missing, or unmanifested output in {run_dir}")
    for name, record in records.items():
        path = contained(run_dir, safe_relative_path(name, f"run file {name}"), name)
        _verify_hash_record(path, require_mapping(record, name), f"run file {name}")
    if status == "success":
        missing = sorted(set(REQUIRED_SUCCESS_FILES) - set(records))
        if missing:
            raise CampaignError(f"successful run is missing retained outputs: {missing}")
        for name in REQUIRED_SUCCESS_FILES:
            require_file(run_dir / name, f"retained {name}")
        adaptive = run["arm"] != "control"
        for name in ADAPTIVE_REQUIRED_FILES:
            present = name in records
            if adaptive and not present:
                raise CampaignError(f"adaptive run is missing retained {name}")
            if not adaptive and present:
                raise CampaignError(f"nonadaptive control unexpectedly contains {name}")
    if require_success and status != "success":
        raise CampaignError(f"campaign contains sealed failed run: {run['run_key']}")
    return str(status)


class CampaignLock:
    def __init__(self, root: Path):
        self.path = root / ".campaign.lock"
        self.acquired = False

    def __enter__(self) -> "CampaignLock":
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise CampaignError(
                f"campaign lock already exists; inspect before removing it: {self.path}"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\ncreated_utc={utc_now()}\n")
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except OSError as unlink_error:
                if exc is None:
                    raise CampaignError(f"cannot release campaign lock: {unlink_error}")


def scan_runs(
    root: Path, campaign: Mapping[str, Any], campaign_hash: str, *, require_success: bool
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        raise CampaignError("campaign runs directory is missing")
    planned = [require_mapping(item, "planned run") for item in campaign["runs"]]
    expected_names = {str(run["run_key"]) for run in planned}
    actual_names = {path.name for path in runs_dir.iterdir()}
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        raise CampaignError(f"unexpected preexisting run outputs: {unexpected}")
    completed: list[dict[str, Any]] = []
    first_missing: dict[str, Any] | None = None
    seen_gap = False
    for run in planned:
        run_dir = runs_dir / str(run["run_key"])
        if run_dir.exists():
            if seen_gap:
                raise CampaignError("preexisting run output appears after a sequence gap")
            if not run_dir.is_dir() or run_dir.is_symlink():
                raise CampaignError(f"run output is not a directory: {run_dir}")
            verify_sealed_run(run_dir, campaign_hash, run, require_success=require_success)
            completed.append(run)
        else:
            seen_gap = True
            if first_missing is None:
                first_missing = run
    return completed, first_missing


def _resource_snapshot() -> dict[str, Any]:
    result = {"os_times": list(os.times())}
    try:
        import resource  # POSIX standard library; intentionally optional on Windows.

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result["rusage_children"] = {
            "user_seconds": usage.ru_utime,
            "system_seconds": usage.ru_stime,
            "max_rss_native_units_process_lifetime": usage.ru_maxrss,
            "minor_faults": usage.ru_minflt,
            "major_faults": usage.ru_majflt,
            "voluntary_context_switches": usage.ru_nvcsw,
            "involuntary_context_switches": usage.ru_nivcsw,
        }
    except (ImportError, AttributeError):
        result["rusage_children"] = None
    return result


def _resource_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_times = before["os_times"]
    after_times = after["os_times"]
    result: dict[str, Any] = {
        "children_user_seconds": after_times[2] - before_times[2],
        "children_system_seconds": after_times[3] - before_times[3],
        "resource_method": "os.times child deltas",
    }
    end_usage = after.get("rusage_children")
    start_usage = before.get("rusage_children")
    if isinstance(end_usage, dict) and isinstance(start_usage, dict):
        result.update(
            {
                "rusage_user_seconds_delta": end_usage["user_seconds"]
                - start_usage["user_seconds"],
                "rusage_system_seconds_delta": end_usage["system_seconds"]
                - start_usage["system_seconds"],
                "maximum_resident_set_native_units_process_lifetime": end_usage[
                    "max_rss_native_units_process_lifetime"
                ],
                "minor_faults_delta": end_usage["minor_faults"] - start_usage["minor_faults"],
                "major_faults_delta": end_usage["major_faults"] - start_usage["major_faults"],
                "voluntary_context_switches_delta": end_usage["voluntary_context_switches"]
                - start_usage["voluntary_context_switches"],
                "involuntary_context_switches_delta": end_usage["involuntary_context_switches"]
                - start_usage["involuntary_context_switches"],
                "rusage_maxrss_units": "KiB on Linux; bytes on macOS; unavailable on Windows",
            }
        )
    else:
        result["maximum_resident_set_native_units_process_lifetime"] = None
    return result


def parse_telemetry(log_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        line = raw.strip()
        if not line.startswith("FEEBL_ADAPTIVE_"):
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            tokens = line.split()
        fields: dict[str, str] = {}
        for token in tokens[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                if key in fields:
                    raise CampaignError(f"duplicate telemetry field {key!r} on line {line_number}")
                fields[key] = value
        records.append(
            {"line_number": line_number, "record": tokens[0], "fields": fields, "raw": line}
        )
    return records


def _telemetry_integer(
    fields: Mapping[str, str], key: str, label: str = "adaptive-grid certificate"
) -> int:
    try:
        raw = fields[key]
        if re.fullmatch(r"[+-]?[0-9]+", raw) is None:
            raise ValueError
        return int(raw)
    except (KeyError, ValueError) as exc:
        raise CampaignError(f"{label} has invalid integer {key}") from exc


def _telemetry_decimal(
    fields: Mapping[str, str], key: str, label: str = "adaptive-grid certificate"
) -> Decimal:
    try:
        value = Decimal(fields[key])
    except (KeyError, InvalidOperation) as exc:
        raise CampaignError(f"{label} has invalid decimal {key}") from exc
    if not value.is_finite():
        raise CampaignError(f"{label} has nonfinite decimal {key}")
    return value


def _card_string_assignment(assignments: Mapping[str, Assignment], key: str, label: str) -> str:
    try:
        raw = assignments[key].value
    except KeyError as exc:
        raise CampaignError(f"{label} is missing required assignment {key!r}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CampaignError(f"{label} assignment {key!r} is not a quoted string") from exc
    if not isinstance(parsed, str) or not parsed:
        raise CampaignError(f"{label} assignment {key!r} is not a nonempty string")
    return parsed


def _card_quoted_string_assignment_allow_empty(
    assignments: Mapping[str, Assignment], key: str, label: str
) -> str:
    """Read an explicitly classified quoted string, preserving empty."""
    try:
        raw = assignments[key].value
    except KeyError as exc:
        raise CampaignError(f"{label} is missing required assignment {key!r}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CampaignError(f"{label} assignment {key!r} is not a quoted string") from exc
    if not isinstance(parsed, str):
        raise CampaignError(f"{label} assignment {key!r} is not a quoted string")
    return parsed


def _bind_certificate_to_effective_card(
    fields: Mapping[str, str], assignments: Mapping[str, Assignment]
) -> None:
    label = "executed adaptive card"
    expected_mode = decimal_assignment(assignments, "adaptive_grid_mode", label)
    if expected_mode not in (Decimal(2), Decimal(3)):
        raise CampaignError("executed adaptive card is not in grid Mode 2 or 3")
    for certificate_key in ("requested_mode", "actual_mode"):
        if Decimal(_telemetry_integer(fields, certificate_key)) != expected_mode:
            raise CampaignError(
                f"adaptive-grid certificate {certificate_key} does not match acc.dat"
            )

    for certificate_key, card_key in (
        ("pilot_diagnostics", "adaptive_grid_pilot_diagnostics"),
        ("fail_on_unmet_error", "adaptive_grid_fail_on_unmet_error"),
        ("terminal_patch_enabled", "adaptive_grid_terminal_patch_certification"),
    ):
        if Decimal(_telemetry_integer(fields, certificate_key)) != decimal_assignment(
            assignments, card_key, label
        ):
            raise CampaignError(
                f"adaptive-grid certificate {certificate_key} does not match acc.dat"
            )

    for certificate_key, card_key in (
        ("error_target", "adaptive_grid_error_target"),
        ("error_quantile", "adaptive_grid_error_quantile"),
        ("catastrophe_limit", "adaptive_grid_error_catastrophe_limit"),
        ("error_reference_field", "adaptive_grid_error_reference_field"),
        (
            "error_absolute_floor_fraction",
            "adaptive_grid_error_absolute_floor_fraction",
        ),
        (
            "error_catastrophe_absolute_fraction",
            "adaptive_grid_error_catastrophe_absolute_fraction",
        ),
        ("error_exposure_limit", "adaptive_grid_error_exposure_limit"),
    ):
        certificate_value = _telemetry_decimal(fields, certificate_key)
        if certificate_value != decimal_assignment(assignments, card_key, label):
            raise CampaignError(
                f"adaptive-grid certificate {certificate_key} does not match acc.dat"
            )

    for certificate_key, card_key in (
        ("error_norm_policy", "adaptive_grid_error_norm_policy"),
        ("source_model", "adaptive_grid_source_model"),
    ):
        if fields[certificate_key] != _card_string_assignment(assignments, card_key, label):
            raise CampaignError(
                f"adaptive-grid certificate {certificate_key} does not match acc.dat"
            )

    policy = fields["error_norm_policy"]
    card_budget = _card_quoted_string_assignment_allow_empty(
        assignments, "adaptive_grid_error_budget_sha256", label
    )
    certificate_budget = fields["error_budget_sha256"]
    if policy == "vector_mixed_rms_quantile_v2":
        if HEX256_RE.fullmatch(card_budget) is None:
            raise CampaignError(
                "executed adaptive card mixed-V2 error budget is not a "
                "canonical lowercase SHA-256"
            )
        if certificate_budget != card_budget:
            raise CampaignError(
                "adaptive-grid certificate error_budget_sha256 does not " "match acc.dat"
            )
    else:
        if card_budget != "":
            raise CampaignError(
                "executed non-mixed adaptive card has a nonempty "
                "adaptive_grid_error_budget_sha256"
            )
        if certificate_budget != "not_applicable":
            raise CampaignError(
                "non-mixed adaptive-grid certificate must report "
                "error_budget_sha256=not_applicable"
            )

    if HEX256_RE.fullmatch(fields["card_hash"]) is None:
        raise CampaignError(
            "adaptive-grid certificate card_hash is not a canonical " "lowercase SHA-256"
        )


def validate_macroparticle_summary(
    records: Sequence[Mapping[str, Any]],
    effective_assignments: Mapping[str, Assignment],
) -> dict[str, Any]:
    summaries = [item for item in records if item.get("record") == MACROPARTICLE_SUMMARY_RECORD]
    if len(summaries) != 1:
        raise CampaignError(
            f"adaptive run must emit exactly one {MACROPARTICLE_SUMMARY_RECORD}; "
            f"found {len(summaries)}"
        )
    summary = summaries[0]
    fields = require_mapping(summary.get("fields"), "adaptive macroparticle summary")

    expected_mode = decimal_assignment(
        effective_assignments, "adaptive_macroparticle_mode", "executed adaptive card"
    )
    if expected_mode not in (Decimal(1), Decimal(2), Decimal(3)):
        raise CampaignError("executed adaptive card has no active macroparticle mode")
    for key in ("requested_mode", "actual_mode"):
        observed_mode = _telemetry_integer(fields, key, "adaptive macroparticle summary")
        if Decimal(observed_mode) != expected_mode:
            raise CampaignError(f"adaptive macroparticle summary {key} does not match acc.dat")
    for key in (
        "controller_replay_required",
        "controller_action_unmet",
        "physical_primary_population_mutations",
        "source_enters_lumi_ee",
        "source_enters_stochastic_physics",
    ):
        if _telemetry_integer(fields, key, "adaptive macroparticle summary") != 0:
            raise CampaignError(f"adaptive macroparticle qualification gate failed: {key}")
    if _telemetry_integer(fields, "fail_on_unmet_error", "adaptive macroparticle summary") != 1:
        raise CampaignError("adaptive macroparticle fail-on-unmet policy was disabled")

    raw = str(summary.get("raw", ""))
    return {
        "record": MACROPARTICLE_SUMMARY_RECORD,
        "line_number": summary["line_number"],
        "raw_sha256": sha256_bytes(raw.encode("utf-8")),
        "canonical_fields": {key: fields[key] for key in sorted(fields)},
        "field_set_sha256": sha256_bytes(
            canonical_json_bytes({key: fields[key] for key in sorted(fields)})
        ),
        "qualification_gate": "active_mode_bound_no_unmet_or_replay",
    }


def validate_grid_certificate(
    records: Sequence[Mapping[str, Any]],
    effective_assignments: Mapping[str, Assignment],
) -> dict[str, Any]:
    certificates = [item for item in records if item.get("record") == GRID_CERTIFICATE_RECORD]
    if len(certificates) != 1:
        raise CampaignError(
            f"adaptive run must emit exactly one {GRID_CERTIFICATE_RECORD}; found {len(certificates)}"
        )
    certificate = certificates[0]
    fields = require_mapping(certificate.get("fields"), "adaptive-grid certificate fields")
    raw = str(certificate.get("raw", ""))
    try:
        raw_tokens = shlex.split(raw, posix=True)
    except ValueError as exc:
        raise CampaignError("adaptive-grid certificate has invalid quoting") from exc
    if (
        len(raw_tokens) != len(GRID_CERTIFICATE_FIELD_ORDER) + 1
        or not raw_tokens
        or raw_tokens[0] != GRID_CERTIFICATE_RECORD
        or any("=" not in token for token in raw_tokens[1:])
    ):
        raise CampaignError("adaptive-grid certificate contains noncanonical tokens")
    if tuple(fields) != GRID_CERTIFICATE_FIELD_ORDER:
        missing = sorted(GRID_CERTIFICATE_FIELDS - set(fields))
        extra = sorted(set(fields) - GRID_CERTIFICATE_FIELDS)
        raise CampaignError(
            "adaptive-grid certificate field set/order is noncanonical; "
            f"missing={missing}, extra={extra}"
        )
    if fields["schema"] != GRID_CERTIFICATE_SCHEMA:
        raise CampaignError("adaptive-grid certificate schema mismatch")
    _bind_certificate_to_effective_card(fields, effective_assignments)
    requested = _telemetry_integer(fields, "requested_mode")
    actual = _telemetry_integer(fields, "actual_mode")
    if requested not in (2, 3) or actual != requested:
        raise CampaignError("adaptive-grid certificate does not authenticate active Mode 2/3")
    encounters = _telemetry_integer(fields, "encounters")
    certified = _telemetry_integer(fields, "certified_encounters")
    if encounters <= 0 or certified != encounters:
        raise CampaignError("adaptive-grid certificate does not cover every encounter")
    for key in (
        "unmet_encounters",
        "noise_stop_encounters",
        "resource_unmet_encounters",
        "vector_catastrophe_failure_encounters",
        "terminal_patch_rejects",
        "terminal_patch_rejection_mask_or",
        "external_replay_recommended",
    ):
        if _telemetry_integer(fields, key) != 0:
            raise CampaignError(f"adaptive-grid qualification gate failed: {key} is nonzero")
    if _telemetry_integer(fields, "pilot_diagnostics") != 0:
        raise CampaignError("adaptive-grid certificate is only unqualified pilot diagnostics")
    if _telemetry_integer(fields, "fail_on_unmet_error") != 1:
        raise CampaignError("adaptive-grid fail-on-unmet policy was disabled")
    if _telemetry_integer(fields, "all_encounters_certified") != 1:
        raise CampaignError("adaptive-grid certificate rejects at least one encounter")
    if fields["disposition"] != "accepted_all_encounters":
        raise CampaignError("adaptive-grid terminal disposition is not accepted_all_encounters")
    enabled = _telemetry_integer(fields, "terminal_patch_enabled")
    attempts = _telemetry_integer(fields, "terminal_patch_attempts")
    accepts = _telemetry_integer(fields, "terminal_patch_accepts")
    if enabled not in (0, 1) or attempts < 0 or accepts < 0:
        raise CampaignError("adaptive-grid terminal patch counts are invalid")
    if (enabled == 0 and (attempts != 0 or accepts != 0)) or (enabled == 1 and attempts != accepts):
        raise CampaignError("adaptive-grid terminal patch attempts are not all accepted")
    if enabled == 1 and attempts == 0:
        raise CampaignError(
            "terminal-patch certification was enabled but no terminal patch was exercised"
        )
    maximum_vector_difference = _telemetry_decimal(fields, "maximum_vector_difference")
    catastrophe_limit = _telemetry_decimal(fields, "catastrophe_limit")
    error_target = _telemetry_decimal(fields, "error_target")
    error_quantile = _telemetry_decimal(fields, "error_quantile")
    error_reference_field = _telemetry_decimal(fields, "error_reference_field")
    error_absolute_floor_fraction = _telemetry_decimal(fields, "error_absolute_floor_fraction")
    error_catastrophe_absolute_fraction = _telemetry_decimal(
        fields, "error_catastrophe_absolute_fraction"
    )
    error_exposure_limit = _telemetry_decimal(fields, "error_exposure_limit")
    maximum_beam_exposure = _telemetry_decimal(fields, "maximum_beam_above_target_exposure")
    if (
        maximum_vector_difference < 0
        or catastrophe_limit <= 0
        or maximum_vector_difference > catastrophe_limit
        or catastrophe_limit > 2
        or error_target <= 0
        or error_target >= 1
        or error_quantile < Decimal("0.95")
        or error_quantile >= 1
        or maximum_beam_exposure < 0
        or maximum_beam_exposure > 1
    ):
        raise CampaignError("adaptive-grid certificate has nonfinite/invalid error evidence")

    policy = fields["error_norm_policy"]
    if policy not in {
        "component_rms_linf_v1",
        "vector_rms_quantile_v1",
        "vector_mixed_rms_quantile_v2",
    }:
        raise CampaignError("adaptive-grid certificate has an unknown error policy")
    if policy == "vector_mixed_rms_quantile_v2":
        if (
            error_reference_field <= 0
            or error_absolute_floor_fraction <= 0
            or error_absolute_floor_fraction > 1
            or error_catastrophe_absolute_fraction <= 0
            or error_catastrophe_absolute_fraction > 2
            or error_exposure_limit <= 0
            or error_exposure_limit >= 1
            or maximum_beam_exposure > error_exposure_limit
        ):
            raise CampaignError(
                "mixed-V2 adaptive-grid certificate has invalid scale or "
                "per-beam exposure evidence"
            )
    elif (
        error_reference_field != 0
        or error_absolute_floor_fraction != 0
        or error_catastrophe_absolute_fraction != 0
        or error_exposure_limit != 0
        or maximum_beam_exposure != 0
    ):
        raise CampaignError("non-mixed adaptive-grid certificate has live mixed-V2 fields")

    # Detail telemetry cannot contradict the terminal certificate.  This is
    # deliberately independent of endpoint agreement and of fail_on_unmet=0.
    for record in records:
        detail = require_mapping(record.get("fields"), "adaptive telemetry fields")
        for key, value in detail.items():
            if key == "error_target_met" and value != "1":
                raise CampaignError("an adaptive encounter reports error_target_met=0")
            if key == "vector_catastrophe_guard_passed" and value != "1":
                raise CampaignError("an adaptive encounter failed its catastrophe guard")
            if key.endswith("unmet_error_encounters") or key.endswith("unmet_encounters"):
                try:
                    if int(value) != 0:
                        raise CampaignError(f"adaptive telemetry reports nonzero {key}")
                except ValueError as exc:
                    raise CampaignError(f"adaptive telemetry has invalid {key}") from exc
            if "catastrophe" in key and (
                key.endswith("failures") or key.endswith("failure_encounters")
            ):
                try:
                    if int(value) != 0:
                        raise CampaignError(f"adaptive telemetry reports nonzero {key}")
                except ValueError as exc:
                    raise CampaignError(f"adaptive telemetry has invalid {key}") from exc
            if "terminal" in key and ("reject" in key or "rejection" in key):
                try:
                    if int(value) != 0:
                        raise CampaignError(f"adaptive telemetry reports terminal rejection {key}")
                except ValueError as exc:
                    raise CampaignError(f"adaptive telemetry has invalid {key}") from exc
    return {
        "record": GRID_CERTIFICATE_RECORD,
        "line_number": certificate["line_number"],
        "raw_sha256": sha256_bytes(raw.encode("utf-8")),
        "canonical_fields": {key: fields[key] for key in sorted(fields)},
        "field_set_sha256": sha256_bytes(
            canonical_json_bytes({key: fields[key] for key in sorted(fields)})
        ),
        "qualification_gate": "accepted_all_encounters",
    }


def validate_and_write_telemetry(
    run_dir: Path, campaign: Mapping[str, Any], run: Mapping[str, Any]
) -> None:
    records = parse_telemetry(run_dir / "log.txt")
    adaptive = run["arm"] == campaign["adaptive_arm_id"]
    document: dict[str, Any] = {
        "schema": "feebl.fccz-adaptive-telemetry-capture.v1",
        "run_key": run["run_key"],
        "arm": run["arm"],
        "adaptive": adaptive,
        "log_sha256": sha256_file(run_dir / "log.txt"),
        "records": records,
    }
    if adaptive:
        telemetry = campaign["telemetry"]
        observed_records = {str(record["record"]) for record in records}
        missing_prefixes = [
            prefix
            for prefix in telemetry["required_record_prefixes"]
            if not any(value.startswith(prefix) for value in observed_records)
        ]
        if missing_prefixes:
            raise CampaignError(f"adaptive telemetry records missing: {missing_prefixes}")
        field_names = {
            key for record in records for key in require_mapping(record["fields"], "fields")
        }
        missing_fields = sorted(set(telemetry["required_fields"]) - field_names)
        if missing_fields:
            raise CampaignError(f"adaptive telemetry fields missing: {missing_fields}")
        missing_groups = [
            group
            for group, alternatives in TELEMETRY_FIELD_GROUPS.items()
            if not (alternatives & field_names)
        ]
        if missing_groups:
            raise CampaignError(f"adaptive telemetry semantic groups missing: {missing_groups}")
        effective_assignments = parse_assignments(
            require_file(run_dir / "acc.dat", "executed adaptive card").read_text(encoding="utf-8"),
            "executed adaptive card",
        )
        document["macroparticle_summary"] = validate_macroparticle_summary(
            records, effective_assignments
        )
        document["grid_certificate"] = validate_grid_certificate(records, effective_assignments)
        frozen_policy = require_file(
            run_dir / "frozen_adaptive_policy.json", "frozen adaptive policy"
        )
        expected_policy_hash = campaign["inputs"]["frozen_adaptive_policy_sha256"]
        if sha256_file(frozen_policy) != expected_policy_hash:
            raise CampaignError("frozen realization-independent policy was modified")
        document["realization_independent_policy"] = {
            "path": "frozen_adaptive_policy.json",
            "sha256": expected_policy_hash,
        }
        artifact_records: list[dict[str, Any]] = []
        for raw in telemetry["artifacts"]:
            rel = safe_relative_path(raw["path"], "telemetry artifact path")
            path = contained(run_dir, rel, "telemetry artifact")
            require_file(path, f"adaptive {raw['role']} artifact")
            artifact_records.append(
                {
                    "role": raw["role"],
                    "path": rel.as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
        document["artifacts"] = artifact_records
        document["semantic_groups"] = {
            group: sorted(alternatives & field_names)
            for group, alternatives in TELEMETRY_FIELD_GROUPS.items()
        }
        document["status"] = "complete"
    else:
        active_particle_records = [
            item
            for item in records
            if str(item["record"]).startswith("FEEBL_ADAPTIVE_MACROPARTICLE_")
        ]
        if active_particle_records:
            raise CampaignError("nonadaptive control emitted active macroparticle telemetry")
        if any(item["record"] == GRID_CERTIFICATE_RECORD for item in records):
            raise CampaignError("nonadaptive control emitted an adaptive-grid certificate")
        document["artifacts"] = []
        document["status"] = "not_applicable_nonadaptive_control"
    write_exclusive_json(run_dir / "adaptive_telemetry.json", document)


def _gaussian_expected_binding(run_dir: Path, run_id: str) -> dict[str, str]:
    safe_id(run_id, "Gaussian-core run id")
    card = require_file(run_dir / "acc.dat", "Gaussian-core executed acc.dat")
    assignments = parse_assignments(
        card.read_text(encoding="utf-8"), "Gaussian-core executed acc.dat"
    )
    values = {
        key: decimal_assignment(assignments, key, "Gaussian-core executed acc.dat")
        for key in (
            "rndm_seed",
            "spectrum_w_min",
            "spectrum_w_max",
            "spectrum_w_bins",
        )
    }
    if values["rndm_seed"] != values["rndm_seed"].to_integral_value() or not (
        Decimal(1) <= values["rndm_seed"] <= Decimal(2_147_483_647)
    ):
        raise CampaignError("Gaussian-core executed-card rndm_seed is not a supported integer")
    if values["spectrum_w_bins"] != values["spectrum_w_bins"].to_integral_value() or not (
        Decimal(26) <= values["spectrum_w_bins"] <= Decimal(10_000_000)
    ):
        raise CampaignError("Gaussian-core executed-card spectrum_w_bins is invalid")
    if not values["spectrum_w_max"] > values["spectrum_w_min"]:
        raise CampaignError("Gaussian-core executed-card spectrum range is invalid")

    def fitter_decimal(value: Decimal, label: str) -> str:
        converted = float(value)
        if not math.isfinite(converted):
            raise CampaignError(f"Gaussian-core {label} is outside finite binary64")
        return format(converted, ".17g")

    return {
        "run_id": run_id,
        "rndm_seed": str(int(values["rndm_seed"])),
        "spectrum_w_min": fitter_decimal(values["spectrum_w_min"], "spectrum_w_min"),
        "spectrum_w_max": fitter_decimal(values["spectrum_w_max"], "spectrum_w_max"),
        "spectrum_w_bins": str(int(values["spectrum_w_bins"])),
    }


def validate_gaussian_output(run_dir: Path, run_id: str) -> dict[str, Any]:
    spectrum = require_file(run_dir / "lumi.ee.out.dLdW.fne", "dL/dW spectrum")
    output_path = require_file(run_dir / "gaussian_core.json", "Gaussian-core JSON")
    output = require_mapping(load_json(output_path), "Gaussian-core JSON")
    if output_path.read_bytes() != canonical_json_bytes(output):
        raise CampaignError("Gaussian-core JSON is not canonical compact UTF-8")
    if (
        output.get("schema") != GAUSSIAN_SCHEMA
        or output.get("status") != "accepted_core_fit_not_physics_qualification"
    ):
        raise CampaignError("Gaussian-core fitter did not publish an accepted schema")
    evidence = require_mapping(output.get("input"), "Gaussian-core input evidence")
    if evidence.get("file_name") != "lumi.ee.out.dLdW.fne":
        raise CampaignError("Gaussian-core result names the wrong spectrum")
    if evidence.get("sha256") != sha256_file(spectrum):
        raise CampaignError("Gaussian-core result is not bound to the run spectrum")
    expected = _gaussian_expected_binding(run_dir, run_id)
    binding = require_mapping(
        output.get("qualification_binding"), "Gaussian-core qualification binding"
    )
    expected_fields: dict[str, Any] = {
        "status": "qualification_input_bound",
        "native_format": "FEEBL-NATIVE 1.0",
        "rndm_seed": expected["rndm_seed"],
        "run_id": expected["run_id"],
        "spectrum_w_min_GeV": expected["spectrum_w_min"],
        "spectrum_w_max_GeV": expected["spectrum_w_max"],
        "spectrum_w_bins": expected["spectrum_w_bins"],
        "lattice_phase_proven": True,
    }
    for key, wanted in expected_fields.items():
        if binding.get(key) != wanted:
            raise CampaignError(
                f"Gaussian-core qualification binding {key} disagrees with executed acc.dat"
            )
    core = require_mapping(output.get("core_fit"), "Gaussian-core fit")
    for key in ("mu_core_GeV", "sigma_core_GeV", "gaussian_core_area", "peak_dL_dW"):
        try:
            value = float(core[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignError(f"invalid Gaussian-core field {key}") from exc
        if not math.isfinite(value) or (key == "sigma_core_GeV" and value <= 0):
            raise CampaignError(f"nonfinite or invalid Gaussian-core field {key}")
    csv_path = require_file(run_dir / "gaussian_core.csv", "Gaussian-core CSV")
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CampaignError(f"cannot parse Gaussian-core CSV: {exc}") from exc
    if len(rows) != 1 or reader.fieldnames is None:
        raise CampaignError("Gaussian-core CSV must contain exactly one data row")
    row = rows[0]
    expected_csv = {
        "schema": GAUSSIAN_SCHEMA,
        "input_file_name": "lumi.ee.out.dLdW.fne",
        "input_sha256": str(evidence["sha256"]),
        "status": "accepted_core_fit_not_physics_qualification",
        "binding_status": "qualification_input_bound",
        "run_id": expected["run_id"],
        "rndm_seed": expected["rndm_seed"],
        "spectrum_w_min_GeV": expected["spectrum_w_min"],
        "spectrum_w_max_GeV": expected["spectrum_w_max"],
        "spectrum_w_bins": expected["spectrum_w_bins"],
        "lattice_phase_proven": "true",
        "mu_core_GeV": str(core["mu_core_GeV"]),
        "sigma_core_GeV": str(core["sigma_core_GeV"]),
        "gaussian_core_area": str(core["gaussian_core_area"]),
        "peak_dL_dW": str(core["peak_dL_dW"]),
    }
    for key, wanted in expected_csv.items():
        if row.get(key) != wanted:
            raise CampaignError(f"Gaussian-core CSV field {key} disagrees with JSON/card")
    return output


def _expanded_argv(
    campaign: Mapping[str, Any], run: Mapping[str, Any], binary: Path, python: Path
) -> list[str]:
    replacements = {
        "{binary}": str(binary),
        "{python}": str(python),
        "{run_id}": str(run["run_id"]),
        "{arm}": str(run["arm"]),
    }
    result: list[str] = []
    for token in campaign["execution"]["argv"]:
        expanded = token
        for old, new in replacements.items():
            expanded = expanded.replace(old, new)
        result.append(expanded)
    return result


def _effective_environment(campaign: Mapping[str, Any]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in campaign["execution"]["inherit_environment"]:
        if name in os.environ:
            environment[name] = os.environ[name]
    environment.update(campaign["execution"]["environment"])
    return environment


def _run_fitter(root: Path, campaign: Mapping[str, Any], directory: Path, run_id: str) -> None:
    inputs = campaign["inputs"]
    python = Path(str(inputs["python_executable_path"]))
    fitter = contained(
        root,
        safe_relative_path(inputs["gaussian_core_fitter_copy"], "fitter copy"),
        "fitter copy",
    )
    expected_python_hash = require_hash(
        inputs["python_executable_sha256"], "python_executable_sha256"
    )
    expected_fitter_hash = require_hash(
        inputs["gaussian_core_fitter_sha256"], "gaussian_core_fitter_sha256"
    )
    if sha256_file(require_file(python, "Python executable")) != expected_python_hash:
        raise CampaignError("Python executable drifted before Gaussian-core fitting")
    if sha256_file(require_file(fitter, "fitter copy")) != expected_fitter_hash:
        raise CampaignError("Gaussian-core fitter copy drifted before execution")
    prior_files = _iter_regular_files(directory)
    generated_names = {
        "gaussian_core.json",
        "gaussian_core.csv",
        "gaussian_core_fit.log",
    }
    collisions = sorted(generated_names & set(prior_files))
    if collisions:
        raise CampaignError(f"refusing preexisting Gaussian-core outputs: {collisions}")
    prior_hashes = {name: sha256_file(path) for name, path in prior_files.items()}
    expected = _gaussian_expected_binding(directory, run_id)
    command = [
        str(python),
        "-B",
        str(fitter),
        str(directory / "lumi.ee.out.dLdW.fne"),
        "--json",
        str(directory / "gaussian_core.json"),
        "--csv",
        str(directory / "gaussian_core.csv"),
        "--expected-seed",
        expected["rndm_seed"],
        "--expected-w-min",
        expected["spectrum_w_min"],
        "--expected-w-max",
        expected["spectrum_w_max"],
        "--expected-bins",
        expected["spectrum_w_bins"],
        "--expected-run-id",
        expected["run_id"],
    ]
    with (directory / "gaussian_core_fit.log").open("xb") as log:
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=float(campaign["execution"]["timeout_seconds"]),
                env=_effective_environment(campaign),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CampaignError(f"cannot complete Gaussian-core fitter: {exc}") from exc
    if result.returncode != 0:
        raise CampaignError(f"Gaussian-core fitter returned {result.returncode}")
    if sha256_file(python) != expected_python_hash:
        raise CampaignError("Python executable drifted during Gaussian-core fitting")
    if sha256_file(fitter) != expected_fitter_hash:
        raise CampaignError("Gaussian-core fitter copy drifted during execution")
    after_files = _iter_regular_files(directory)
    if set(after_files) != set(prior_files) | generated_names:
        raise CampaignError("Gaussian-core fitter produced a missing or unexpected file set")
    for name, expected_hash in prior_hashes.items():
        if sha256_file(after_files[name]) != expected_hash:
            raise CampaignError(f"Gaussian-core fitter modified preexisting evidence: {name}")
    validate_gaussian_output(directory, run_id)


def execute_one(
    root: Path, campaign: Mapping[str, Any], campaign_hash: str, run: Mapping[str, Any]
) -> None:
    run_dir = root / "runs" / str(run["run_key"])
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise CampaignError(f"refusing preexisting run directory: {run_dir}") from exc
    source = contained(root, safe_relative_path(run["source_card"], "source card"), "source card")
    effective = contained(
        root, safe_relative_path(run["effective_card"], "effective card"), "effective card"
    )
    shutil.copyfile(source, run_dir / "original_acc.dat")
    shutil.copyfile(effective, run_dir / "acc.dat")
    inputs = campaign["inputs"]
    if run["arm"] == campaign["adaptive_arm_id"]:
        policy_source = contained(
            root,
            safe_relative_path(
                inputs["frozen_adaptive_policy_copy"], "frozen adaptive policy copy"
            ),
            "frozen adaptive policy copy",
        )
        shutil.copyfile(policy_source, run_dir / "frozen_adaptive_policy.json")
    binary = Path(str(inputs["binary_path"]))
    python = Path(str(inputs["python_executable_path"]))
    command = _expanded_argv(campaign, run, binary, python)
    environment = _effective_environment(campaign)
    environment_document = {
        "schema": "feebl.fccz-frozen-eight-environment.v1",
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(run_dir),
        "inherited_names": campaign["execution"]["inherit_environment"],
        "effective_environment": environment,
    }
    write_exclusive_json(run_dir / "environment.json", environment_document)
    environment_hash = sha256_file(run_dir / "environment.json")
    request = {
        "schema": "feebl.fccz-frozen-eight-run-request.v1",
        "campaign_sha256": campaign_hash,
        "run": run,
        "command": command,
        "binary_sha256_verified": sha256_file(binary),
        "original_acc_sha256": sha256_file(run_dir / "original_acc.dat"),
        "acc_sha256": sha256_file(run_dir / "acc.dat"),
        "frozen_adaptive_policy_sha256": (
            inputs["frozen_adaptive_policy_sha256"]
            if run["arm"] == campaign["adaptive_arm_id"]
            else None
        ),
    }
    write_exclusive_json(run_dir / "run_request.json", request)
    request_hash = sha256_file(run_dir / "run_request.json")
    started_utc = utc_now()
    start_wall = time.monotonic()
    before = _resource_snapshot()
    timed_out = False
    return_code: int | None = None
    launch_error: str | None = None
    try:
        with (run_dir / "log.txt").open("xb") as log:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=run_dir,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name != "nt"),
                )
                try:
                    return_code = process.wait(
                        timeout=float(campaign["execution"]["timeout_seconds"])
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    return_code = process.wait()
            except OSError as exc:
                launch_error = str(exc)
                return_code = 127
    finally:
        finished_utc = utc_now()
        elapsed = time.monotonic() - start_wall
        after = _resource_snapshot()
        time_document = {
            "schema": "feebl.fccz-frozen-eight-time-resource.v1",
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "elapsed_seconds": elapsed,
            "timed_out": timed_out,
            "timeout_seconds": campaign["execution"]["timeout_seconds"],
            "return_code": return_code,
            "launch_error": launch_error,
            "command": command,
            "resources": _resource_delta(before, after),
        }
        write_exclusive_json(run_dir / "time.json", time_document)
        resource_lines = [
            "schema=FEEBL_FCCZ_FROZEN_EIGHT_TIME_RESOURCE_V1",
            f"started_utc={started_utc}",
            f"finished_utc={finished_utc}",
            f"elapsed_seconds={elapsed:.9f}",
            f"timed_out={int(timed_out)}",
            f"return_code={return_code}",
        ]
        for key, value in time_document["resources"].items():
            resource_lines.append(f"{key}={value}")
        write_exclusive_text(run_dir / "time.txt", "\n".join(resource_lines) + "\n")
        write_exclusive_text(run_dir / "exit_code", f"{return_code}\n")

    failure: CampaignError | None = None
    try:
        if launch_error is not None:
            raise CampaignError(f"cannot launch FeeBL binary: {launch_error}")
        if timed_out:
            raise CampaignError("FeeBL process exceeded the frozen timeout")
        if return_code != 0:
            raise CampaignError(f"FeeBL process returned {return_code}")
        for name in ("original_acc.dat", "acc.dat", "out.dat", "log.txt", "lumi.ee.out.dLdW.fne"):
            require_file(run_dir / name, f"run output {name}")
        if sha256_file(run_dir / "original_acc.dat") != run["source_card_sha256"]:
            raise CampaignError("FeeBL modified original_acc.dat")
        if sha256_file(run_dir / "acc.dat") != run["effective_card_sha256"]:
            raise CampaignError("FeeBL modified acc.dat")
        if sha256_file(run_dir / "environment.json") != environment_hash:
            raise CampaignError("FeeBL modified environment.json")
        if sha256_file(run_dir / "run_request.json") != request_hash:
            raise CampaignError("FeeBL modified run_request.json")
        if sha256_file(binary) != inputs["binary_sha256"]:
            raise CampaignError("FeeBL binary drifted during execution")
        if sha256_file(python) != inputs["python_executable_sha256"]:
            raise CampaignError("Python executable drifted during FeeBL execution")
        if (
            run["arm"] == campaign["adaptive_arm_id"]
            and sha256_file(run_dir / "frozen_adaptive_policy.json")
            != inputs["frozen_adaptive_policy_sha256"]
        ):
            raise CampaignError("FeeBL modified the frozen adaptive policy")
        validate_and_write_telemetry(run_dir, campaign, run)
        _run_fitter(root, campaign, run_dir, str(run["run_key"]))
        if sha256_file(run_dir / "original_acc.dat") != run["source_card_sha256"]:
            raise CampaignError("postprocessor modified original_acc.dat")
        if sha256_file(run_dir / "acc.dat") != run["effective_card_sha256"]:
            raise CampaignError("postprocessor modified acc.dat")
        for name in REQUIRED_SUCCESS_FILES:
            require_file(run_dir / name, f"retained output {name}")
    except CampaignError as exc:
        failure = exc

    if failure is not None:
        if not (run_dir / "harness_error.txt").exists():
            write_exclusive_text(run_dir / "harness_error.txt", str(failure) + "\n")
        seal_run(run_dir, campaign_hash, run, "failed")
        raise failure
    seal_run(run_dir, campaign_hash, run, "success")
    verify_sealed_run(run_dir, campaign_hash, run, require_success=True)


def run_next(args: argparse.Namespace) -> str:
    root = args.campaign.resolve()
    with CampaignLock(root):
        campaign = load_campaign(root, args.campaign_sha256, verify_historical=False)
        completed, next_run = scan_runs(root, campaign, args.campaign_sha256, require_success=True)
        if next_run is None:
            return f"COMPLETE {len(completed)}/{len(campaign['runs'])}"
        execute_one(root, campaign, args.campaign_sha256, next_run)
        return f"PASS {next_run['run_key']} ({len(completed) + 1}/{len(campaign['runs'])})"


def parse_out_metrics(path: Path) -> dict[str, float]:
    text = require_file(path, "out.dat").read_text(encoding="utf-8", errors="replace")
    result: dict[str, float] = {}
    patterns = {
        "lumi_ee": r"(?m)^\s*lumi_ee\s*=\s*([^\s;]+)",
        "lumi_ee_high": r"(?m)^\s*lumi_ee_high\s*=\s*([^\s;]+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise CampaignError(f"out.dat is missing {key}")
        values: list[float] = []
        for raw in matches:
            try:
                value = float(raw)
            except ValueError as exc:
                raise CampaignError(f"out.dat has invalid {key}") from exc
            if not math.isfinite(value):
                raise CampaignError(f"out.dat has nonfinite {key}")
            values.append(value)
        if any(value != values[0] for value in values[1:]):
            raise CampaignError(f"out.dat has conflicting duplicate {key} values")
        result[key] = values[0]
    return result


def load_metrics(directory: Path, run_id: str) -> dict[str, float]:
    metrics = parse_out_metrics(directory / "out.dat")
    gaussian = validate_gaussian_output(directory, run_id)
    core = gaussian["core_fit"]
    metrics.update(
        {
            "gaussian_core_mu_GeV": float(core["mu_core_GeV"]),
            "gaussian_core_sigma_GeV": float(core["sigma_core_GeV"]),
            "gaussian_core_area": float(core["gaussian_core_area"]),
            "gaussian_core_peak_dL_dW": float(core["peak_dL_dW"]),
        }
    )
    return metrics


def relative_delta(candidate: float, reference: float) -> float | None:
    return (candidate - reference) / reference if reference != 0.0 else None


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", encoding="utf-8", newline="")
    except FileExistsError as exc:
        raise CampaignError(f"refusing preexisting CSV: {path}") from exc
    with stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fit_historical(
    root: Path,
    campaign: Mapping[str, Any],
    analysis_dir: Path,
    entry: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    destination = analysis_dir / "historical_fits" / str(entry["label"]) / str(entry["run_id"])
    destination.mkdir(parents=True)
    files = entry["files"]
    for name in ("acc.dat", "lumi.ee.out.dLdW.fne", "out.dat"):
        source_record = require_mapping(files[name], f"historical {name}")
        copied = destination / name
        shutil.copyfile(source_record["path"], copied)
        _verify_hash_record(copied, source_record, f"copied historical {name}")
    status: dict[str, Any] = {
        "category": entry["category"],
        "label": entry["label"],
        "run_id": entry["run_id"],
        "status": "available",
    }
    fit_run_id = safe_id(f"historical-{entry['label']}-{entry['run_id']}", "historical fit run id")
    try:
        _run_fitter(root, campaign, destination, fit_run_id)
        metrics = load_metrics(destination, fit_run_id)
        status["fit_status"] = "accepted"
        return metrics, status
    except CampaignError as exc:
        status["fit_status"] = "rejected"
        status["fit_error"] = str(exc)
        return parse_out_metrics(destination / "out.dat"), status


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 2:
        raise CampaignError("the frozen two-seed summary requires exactly two values")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    t_975_df1 = 12.706204736432095
    half_width = t_975_df1 * standard_deviation / math.sqrt(2.0)
    return {
        "n": 2,
        "degrees_of_freedom": 1,
        "values": list(values),
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "t_0.975_df1": t_975_df1,
        "two_sided_95_percent_mean_interval_half_width": half_width,
        "relative_half_width_to_abs_mean": (half_width / abs(mean) if mean != 0.0 else None),
        "status": "diagnostic_only_not_equivalence_qualification",
    }


def _independent_seed_power_plan(
    per_identity: Mapping[str, Mapping[str, Any]],
    adaptive_arm: str,
    statistical_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Plan replication from paired complete-process differences.

    The four sigma-z/sign configurations are evaluated separately with the
    same seed shared only for pairing.  A larger macroparticle count, a second
    arm, or another observable from the same process is never counted as an
    independent replicate.
    """
    minimum = int(statistical_contract["minimum_unseen_seeds_per_configuration"])
    maximum = int(statistical_contract["maximum_supported_unseen_seeds_per_configuration"])
    margin = float(statistical_contract["relative_equivalence_margin"])
    expected = float(statistical_contract["expected_relative_difference"])
    alpha = float(statistical_contract["equivalence_alpha"])
    power = float(statistical_contract["equivalence_power"])
    safety = float(statistical_contract["standard_error_safety_factor"])
    if not per_identity:
        raise CampaignError("cannot plan independent seeds without paired results")
    first = next(iter(per_identity.values()))
    metrics = sorted(first["arms"]["control"])
    statement_count = 4 * len(metrics)
    alpha_family = alpha / statement_count
    z_alpha = statistics.NormalDist().inv_cdf(1.0 - alpha_family)
    z_power = statistics.NormalDist().inv_cdf(power)
    gap = margin - abs(expected)

    populations = {
        "all_available_diagnostic": {"frozen", "preregistered_unseen"},
        "preregistered_unseen_only": {"preregistered_unseen"},
    }
    documents: dict[str, Any] = {}
    maximum_required_by_population: dict[str, int | None] = {}
    for population, allowed_origins in populations.items():
        grouped: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
        for identity in per_identity.values():
            if identity["origin"] not in allowed_origins:
                continue
            arms = identity["arms"]
            for metric in metrics:
                relative = relative_delta(arms[adaptive_arm][metric], arms["control"][metric])
                if relative is None:
                    raise CampaignError(
                        f"zero control value prevents paired relative analysis for {metric}"
                    )
                key = (str(identity["sigma_z_mm"]), str(identity["variant"]), metric)
                grouped.setdefault(key, []).append((int(identity["rndm_seed"]), relative))
        summaries: list[dict[str, Any]] = []
        required_counts: list[int] = []
        point_failures = 0
        bound_failures = 0
        precision_method_revisions = 0
        for (sigma_z, variant, metric), seeded in sorted(grouped.items()):
            seeded.sort()
            seed_ids = [seed for seed, _ in seeded]
            if len(seed_ids) != len(set(seed_ids)):
                raise CampaignError("duplicate seed in independent-process analysis")
            values = [value for _, value in seeded]
            record: dict[str, Any] = {
                "sigma_z_mm": sigma_z,
                "variant": variant,
                "metric": metric,
                "seed_ids": seed_ids,
                "complete_process_count": len(values),
                "equivalence_margin_relative": margin,
            }
            if len(values) < 2:
                record["status"] = "insufficient_complete_processes_for_variance"
                summaries.append(record)
                continue
            mean = statistics.fmean(values)
            sd = statistics.stdev(values)
            se = sd / math.sqrt(len(values))
            family_probability = 1.0 - alpha_family
            critical = _student_t_quantile(family_probability, len(values) - 1)
            bound = abs(mean) + safety * critical * se
            raw_power_required = ((z_alpha + z_power) * safety * sd / gap) ** 2
            power_required = max(minimum, math.ceil(raw_power_required))
            observed_margin_gap = margin - abs(mean)
            if observed_margin_gap <= 0.0:
                raw_precision_required: float | None = None
                precision_required: int | None = None
                required: int | None = None
                precision_status = "nonpositive_observed_margin_gap_method_revision_required"
                precision_method_revisions += 1
            else:
                raw_precision_required = (safety * critical * sd / observed_margin_gap) ** 2
                precision_required = max(minimum, math.ceil(raw_precision_required))
                required = max(minimum, power_required, precision_required)
                required_counts.append(required)
                precision_status = "fixed_current_student_t_critical_planning_value"
            point_inside = abs(mean) < margin
            equivalent = point_inside and bound <= margin
            point_failures += int(not point_inside)
            bound_failures += int(not equivalent)
            record.update(
                {
                    "status": "analyzed",
                    "degrees_of_freedom": len(values) - 1,
                    "paired_relative_mean": mean,
                    "paired_relative_sample_standard_deviation": sd,
                    "paired_relative_standard_error": se,
                    "familywise_one_sided_student_t_critical": critical,
                    "standard_error_safety_factor": safety,
                    "simultaneous_conservative_absolute_bound": bound,
                    "point_estimate_inside_margin": point_inside,
                    "equivalent_at_margin": equivalent,
                    "observed_margin_gap": observed_margin_gap,
                    "normal_power_raw_required_complete_processes": raw_power_required,
                    "normal_power_required_complete_processes": power_required,
                    "fixed_critical_precision_status": precision_status,
                    "fixed_critical_precision_raw_required_complete_processes": raw_precision_required,
                    "fixed_critical_precision_required_complete_processes": precision_required,
                    "recommended_required_complete_processes": required,
                    # Retained for v1 consumers; now conservatively means the
                    # max of minimum, normal-power, and fixed-critical t precision.
                    "approximate_power_required_complete_processes": required,
                }
            )
            summaries.append(record)
        documents[population] = {
            "allowed_origins": sorted(allowed_origins),
            "comparisons": summaries,
            "maximum_approximate_power_required_complete_processes": (
                max(required_counts) if required_counts else None
            ),
            "point_estimate_failure_count": point_failures,
            "simultaneous_bound_failure_count": bound_failures,
            "nonpositive_observed_margin_gap_method_revision_count": (precision_method_revisions),
        }
        maximum_required_by_population[population] = (
            max(required_counts) if required_counts else None
        )

    unseen_counts = {
        int(identity["rndm_seed"])
        for identity in per_identity.values()
        if identity["origin"] == "preregistered_unseen"
    }
    unseen_count = len(unseen_counts)
    unseen_required = maximum_required_by_population["preregistered_unseen_only"]
    planning_source = "preregistered_unseen_only"
    required = unseen_required
    if required is None:
        planning_source = "all_available_diagnostic"
        required = maximum_required_by_population[planning_source]
    if required is None:
        required = minimum
    required = max(minimum, required)
    unseen_document = documents["preregistered_unseen_only"]
    if unseen_document["nonpositive_observed_margin_gap_method_revision_count"] > 0:
        disposition = "point_estimate_outside_margin_method_revision_required"
    elif required > maximum:
        disposition = "insufficient_preregistered_seed_capacity"
    elif unseen_count < required:
        disposition = "add_preregistered_unseen_complete_process_seeds"
    elif unseen_document["simultaneous_bound_failure_count"] > 0:
        disposition = "additional_replication_or_method_revision_required"
    else:
        disposition = "statistical_equivalence_gate_passed_not_overall_qualification"
    return {
        "schema": "feebl.fccz-independent-seed-power-plan.v1",
        "qualification_status": "planning_and_statistical_gate_not_overall_physics_qualification",
        "independent_unit": "one_complete_FeeBL_process_per_seed_configuration_and_arm_pair",
        "larger_n_m_is_not_independent_replication": True,
        "multiplicity": {
            "method": "bonferroni",
            "configuration_count": 4,
            "metric_count": len(metrics),
            "one_sided_statement_count": statement_count,
            "nominal_alpha": alpha,
            "familywise_per_statement_alpha": alpha_family,
        },
        "power_planning": {
            "requested_power": power,
            "normal_quantile_one_minus_familywise_alpha": z_alpha,
            "normal_quantile_power": z_power,
            "expected_relative_difference": expected,
            "relative_equivalence_margin": margin,
            "standard_deviation_safety_factor": safety,
            "normal_power_formula": "ceil(((z_(1-alpha/M)+z_power)*safety*paired_sd/(margin-|expected|))^2)",
            "fixed_critical_student_t_precision_formula": "ceil((safety*t_(1-alpha/M,current_df)*paired_sd/(margin-|observed_mean|))^2)",
            "required_count_rule": "max(minimum,normal_power_count,fixed_critical_student_t_precision_count)",
            "nonpositive_observed_margin_gap_rule": "method_revision_required_not_more_replication",
            "minimum_unseen_seeds_per_configuration": minimum,
            "maximum_supported_unseen_seeds_per_configuration": maximum,
        },
        "populations": documents,
        "current_preregistered_unseen_seed_count_per_configuration": unseen_count,
        "planning_variance_source": planning_source,
        "recommended_total_preregistered_unseen_seed_count_per_configuration": required,
        "additional_unseen_seeds_required": (
            0
            if disposition == "point_estimate_outside_margin_method_revision_required"
            else max(0, required - unseen_count)
        ),
        "disposition": disposition,
    }


def analyze(args: argparse.Namespace) -> str:
    root = args.campaign.resolve()
    with CampaignLock(root):
        campaign = load_campaign(root, args.campaign_sha256, verify_historical=True)
        completed, next_run = scan_runs(root, campaign, args.campaign_sha256, require_success=True)
        if next_run is not None:
            raise CampaignError(
                f"analysis requires every planned run; next missing run is {next_run['run_key']}"
            )
        analysis_dir = root / "analysis"
        if analysis_dir.exists():
            raise CampaignError(f"refusing preexisting analysis output: {analysis_dir}")
        analysis_dir.mkdir()

        per_identity: dict[str, dict[str, Any]] = {}
        run_by_id: dict[str, Mapping[str, Any]] = {}
        for run in completed:
            run_dir = root / "runs" / str(run["run_key"])
            metrics = load_metrics(run_dir, str(run["run_key"]))
            slot = per_identity.setdefault(
                str(run["run_id"]),
                {
                    "run_id": run["run_id"],
                    "origin": run["origin"],
                    "sigma_z_mm": run["sigma_z_mm"],
                    "rndm_seed": run["rndm_seed"],
                    "variant": run["variant"],
                    "arms": {},
                },
            )
            slot["arms"][run["arm"]] = metrics
            run_by_id[str(run["run_id"])] = run

        paired_rows: list[dict[str, Any]] = []
        paired_documents: list[dict[str, Any]] = []
        adaptive_arm = str(campaign["adaptive_arm_id"])
        for run_id, identity in sorted(per_identity.items()):
            arms = identity["arms"]
            if set(arms) != {"control", adaptive_arm}:
                raise CampaignError(f"{run_id} does not have exactly one control/adaptive pair")
            comparison: dict[str, Any] = {
                key: identity[key]
                for key in ("run_id", "origin", "sigma_z_mm", "rndm_seed", "variant")
            }
            comparison["metrics"] = {}
            for metric, control_value in arms["control"].items():
                adaptive_value = arms[adaptive_arm][metric]
                row = {
                    "run_id": run_id,
                    "origin": identity["origin"],
                    "sigma_z_mm": identity["sigma_z_mm"],
                    "rndm_seed": identity["rndm_seed"],
                    "variant": identity["variant"],
                    "metric": metric,
                    "control": control_value,
                    "adaptive": adaptive_value,
                    "adaptive_minus_control": adaptive_value - control_value,
                    "relative_to_control": relative_delta(adaptive_value, control_value),
                }
                paired_rows.append(row)
                comparison["metrics"][metric] = {
                    "control": control_value,
                    "adaptive": adaptive_value,
                    "difference": adaptive_value - control_value,
                    "relative_to_control": relative_delta(adaptive_value, control_value),
                }
            paired_documents.append(comparison)
        paired_payload = {
            "schema": "feebl.fccz-adaptive-control-pairs.v1",
            "adaptive_arm": adaptive_arm,
            "energy_spread_definition": "gaussian_core_sigma_GeV",
            "global_spectrum_rms_used": False,
            "pairs": paired_documents,
        }
        write_exclusive_json(analysis_dir / "paired_adaptive_vs_control.json", paired_payload)
        _write_csv(
            analysis_dir / "paired_adaptive_vs_control.csv",
            (
                "run_id",
                "origin",
                "sigma_z_mm",
                "rndm_seed",
                "variant",
                "metric",
                "control",
                "adaptive",
                "adaptive_minus_control",
                "relative_to_control",
            ),
            paired_rows,
        )

        historical_rows: list[dict[str, Any]] = []
        historical_status: list[dict[str, Any]] = []
        for entry in campaign["historical_evidence"]:
            if entry.get("available") is not True:
                historical_status.append(dict(entry))
                continue
            historical_metrics, status = _fit_historical(root, campaign, analysis_dir, entry)
            historical_status.append(status)
            identity = per_identity[str(entry["run_id"])]
            for metric, historical_value in historical_metrics.items():
                for arm in ("control", adaptive_arm):
                    if metric not in identity["arms"][arm]:
                        continue
                    current = identity["arms"][arm][metric]
                    historical_rows.append(
                        {
                            "run_id": entry["run_id"],
                            "historical_category": entry["category"],
                            "historical_label": entry["label"],
                            "current_arm": arm,
                            "metric": metric,
                            "historical": historical_value,
                            "current": current,
                            "current_minus_historical": current - historical_value,
                            "relative_to_historical": relative_delta(current, historical_value),
                        }
                    )
        write_exclusive_json(
            analysis_dir / "historical_comparisons.json",
            {
                "schema": "feebl.fccz-historical-comparisons.v1",
                "evidence_status": historical_status,
                "comparisons": historical_rows,
            },
        )
        _write_csv(
            analysis_dir / "historical_comparisons.csv",
            (
                "run_id",
                "historical_category",
                "historical_label",
                "current_arm",
                "metric",
                "historical",
                "current",
                "current_minus_historical",
                "relative_to_historical",
            ),
            historical_rows,
        )

        grouped: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
        for identity in per_identity.values():
            if identity["origin"] != "frozen":
                continue
            for metric in identity["arms"]["control"]:
                control_value = identity["arms"]["control"][metric]
                adaptive_value = identity["arms"][adaptive_arm][metric]
                rel = relative_delta(adaptive_value, control_value)
                if rel is None:
                    continue
                key = (str(identity["sigma_z_mm"]), str(identity["variant"]), metric)
                grouped.setdefault(key, []).append((int(identity["rndm_seed"]), rel))
        limits: list[dict[str, Any]] = []
        for (sigma_z, variant, metric), values in sorted(grouped.items()):
            values.sort()
            if [seed for seed, _ in values] != [1, 2]:
                raise CampaignError("frozen two-seed analysis lost seed 1 or 2")
            summary = _sample_summary([value for _, value in values])
            summary.update(
                {
                    "sigma_z_mm": sigma_z,
                    "variant": variant,
                    "metric": metric,
                    "quantity": "paired_relative_adaptive_minus_control",
                }
            )
            limits.append(summary)
        limits_payload = {
            "schema": "feebl.fccz-two-seed-limits.v1",
            "independent_unit": "one independently generated complete bunch-pair run",
            "frozen_seed_count_per_sigma_z_and_sign": 2,
            "variance_degrees_of_freedom": 1,
            "qualification_allowed": False,
            "reason": "two frozen seeds are diagnostic only and the identities were previously inspected",
            "additional_unseen_seed_count": campaign["additional_run_count"] // 4,
            "additional_seed_policy": "accepted only through the SHA-bound preregistered input; no automatic qualification claim",
            "energy_spread_definition": "Gaussian-core sigma_core_GeV; global spectrum RMS is forbidden",
            "paired_limits": limits,
        }
        write_exclusive_json(analysis_dir / "two_seed_limits.json", limits_payload)

        power_plan = _independent_seed_power_plan(
            per_identity,
            adaptive_arm,
            require_mapping(campaign.get("statistics"), "campaign.statistics"),
        )
        write_exclusive_json(analysis_dir / "independent_seed_power_plan.json", power_plan)

        categories = {"fixed512": 0, "xgrid1024": 0, "xstat1m": 0}
        for item in historical_status:
            if item.get("available") is True or item.get("status") == "available":
                categories[str(item["category"])] += 1
        core_limits = [item for item in limits if item["metric"] == "gaussian_core_sigma_GeV"]
        report_lines = [
            "# Frozen FCC-Z adaptive/control analysis",
            "",
            "This is a regression/confirmatory analysis, not a pristine unseen qualification.",
            "Every identity has one same-final-binary nonadaptive control and one selected adaptive arm.",
            "Energy spread is the fitted Gaussian-core `sigma_core_GeV`; global spectrum RMS is not used.",
            "",
            "## Two-seed limit",
            "",
            "The frozen sample has two independent complete runs per sigma-z/sign configuration, hence one variance degree of freedom. The reported two-sided 95% paired-mean intervals use t(0.975, 1)=12.7062 and are diagnostic only.",
            "",
            "| sigma_z [mm] | sign | mean relative core-sigma change | 95% half-width |",
            "|---:|---|---:|---:|",
        ]
        for item in core_limits:
            report_lines.append(
                f"| {item['sigma_z_mm']} | {item['variant']} | {item['mean']:.8g} | {item['two_sided_95_percent_mean_interval_half_width']:.8g} |"
            )
        report_lines.extend(
            [
                "",
                "## Independent-seed extension",
                "",
                (
                    "The multiplicity-adjusted power plan recommends "
                    f"{power_plan['recommended_total_preregistered_unseen_seed_count_per_configuration']} "
                    "preregistered unseen complete-process seeds per sigma-z/sign configuration; "
                    f"{power_plan['additional_unseen_seeds_required']} remain to be added. "
                    f"Disposition: `{power_plan['disposition']}`. A larger `n_m` is not counted as replication."
                ),
                "",
                "## Historical anchors",
                "",
                f"Hash-bound available evidence entries: fixed512={categories['fixed512']}, xgrid1024={categories['xgrid1024']}, xstat1m={categories['xstat1m']}. Missing seed-2 convergence evidence is reported as unavailable and is not imputed.",
                "",
                "See `paired_adaptive_vs_control.csv`, `historical_comparisons.csv`, `two_seed_limits.json`, and `independent_seed_power_plan.json` for machine-readable values.",
                "",
            ]
        )
        write_exclusive_text(analysis_dir / "REPORT.md", "\n".join(report_lines))

        files = _iter_regular_files(analysis_dir)
        analysis_manifest = {
            "schema": "feebl.fccz-frozen-eight-analysis-files.v1",
            "campaign_sha256": args.campaign_sha256,
            "created_at_utc": utc_now(),
            "files": {
                name: {"sha256": sha256_file(path), "size": path.stat().st_size}
                for name, path in sorted(files.items())
            },
        }
        write_exclusive_json(analysis_dir / "sha256_manifest.json", analysis_manifest)
        analysis_hash = sha256_file(analysis_dir / "sha256_manifest.json")
        write_exclusive_text(
            analysis_dir / "analysis.sha256", analysis_hash + "  sha256_manifest.json\n"
        )
        return analysis_hash


def status(args: argparse.Namespace) -> str:
    root = args.campaign.resolve()
    campaign = load_campaign(root, args.campaign_sha256, verify_historical=False)
    completed, next_run = scan_runs(root, campaign, args.campaign_sha256, require_success=True)
    suffix = "complete" if next_run is None else f"next={next_run['run_key']}"
    return f"{len(completed)}/{len(campaign['runs'])} verified; {suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="validate and materialize; never execute FeeBL")
    freeze.add_argument("--manifest", required=True, type=Path)
    freeze.add_argument("--source-root", required=True, type=Path)
    freeze.add_argument("--adaptive-template", required=True, type=Path)
    freeze.add_argument("--config", required=True, type=Path)
    freeze.add_argument("--binary", required=True, type=Path)
    freeze.add_argument("--fitter", required=True, type=Path)
    freeze.add_argument("--additional-manifest", type=Path)
    freeze.add_argument("--output", required=True, type=Path)

    for name, help_text in (
        ("run-next", "run exactly one next control/adaptive process"),
        ("status", "verify all existing outputs without executing"),
        ("analyze", "analyze a complete campaign without executing FeeBL"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--campaign", required=True, type=Path)
        command.add_argument("--campaign-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            digest = freeze_campaign(args)
            print(f"FROZEN campaign_sha256={digest}")
            print("No FeeBL process was executed.")
        elif args.command == "run-next":
            print(run_next(args))
        elif args.command == "status":
            print(status(args))
        elif args.command == "analyze":
            print(f"ANALYSIS analysis_manifest_sha256={analyze(args)}")
        else:  # pragma: no cover - argparse prevents this.
            raise CampaignError(f"unknown command {args.command}")
    except (CampaignError, KeyError, OSError, UnicodeError, ValueError) as exc:
        print(f"frozen_eight_campaign: ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
