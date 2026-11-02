#!/usr/bin/env python3
"""Pure, fail-closed v2 convergence-campaign foundation for FeeBL.

This module deliberately does not execute FeeBL.  It defines and checks the
versioned statistical design that a future actuator can consume.  In
particular, analysis and next-stage planning are pure functions of a sealed
request, the exactly materialized manifest, and a sealed complete state.  The
same functions are used when artifacts are emitted and when they are later
validated byte-for-byte.

The v2 foundation remains *not physics qualified*.  That hard block is
intentional until the new collision quadrature, composite AMR solver, runtime
receipts, and external validation are integrated by the surrounding project.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence


REQUEST_SCHEMA_V2 = "feebl.coupled-grid-root-campaign-request.v2"
MANIFEST_SCHEMA_V2 = "feebl.coupled-grid-root-pilot-manifest.v2"
STATE_SCHEMA_V2 = "feebl.coupled-grid-root-pilot-state.v2"
ANALYSIS_SCHEMA_V2 = "feebl.coupled-grid-root-pilot-analysis.v2"
NEXT_STAGE_SCHEMA_V2 = "feebl.coupled-grid-root-next-stage-plan.v2"
SCHEMA_VERSION_V2 = "2"

FOUNDATION_QUALIFICATION_BLOCKER = "v2_foundation_not_runtime_integrated_or_physics_qualified"
SELF_HASH_KEY = "self_sha256"
SHA256_HEX_LENGTH = 64
MINIMUM_PILOT_LEVELS = 3
DEFAULT_HELDOUT_SEED_FLOOR = 100

SUPPORTED_ENDPOINT_KINDS = {
    "continuous",
    "poisson_count",
    "poisson_rate",
    "binomial_proportion",
    "paired_bernoulli",
    "weighted_compound_poisson",
    "exact_invariant",
}

# One global ledger is required for each endpoint.  The first four components
# are recomputed from campaign observations; all others need prospectively
# bound evidence.  A missing or unresolved item is represented explicitly and
# makes the endpoint gate fail.
REQUIRED_UNCERTAINTY_COMPONENTS = (
    "mesh_gci",
    "collision_lattice_shape",
    "source_root",
    "grid_root_interaction",
    "outer_sampling",
    "poisson_solver",
    "longitudinal_time_slices",
    "domain_and_tails",
    "amr_interface",
    "resampling",
    "stochastic_pair_physics",
    "backend_reproducibility",
    "fit_model_binning",
    "external_reference",
)
COMPUTED_UNCERTAINTY_COMPONENTS = {
    "mesh_gci",
    "source_root",
    "grid_root_interaction",
    "outer_sampling",
}

ACTION_MESH_REFINEMENT = "mesh_refinement"
ACTION_WITHIN_RUN_ROOT_SUPPORT = "within_run_root_support"
ACTION_OUTER_REPLICATE_PRECISION = "outer_replicate_precision"
ACTION_HOLDOUT_BRACKET = "holdout_bracket"
ACTION_DIAGNOSTIC_FAILURE = "diagnostic_failure"


class CampaignV2Error(RuntimeError):
    """A fail-closed v2 contract violation."""


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignV2Error(f"{where}: expected an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise CampaignV2Error(f"{where}: expected an array")
    return value


def _text(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise CampaignV2Error(f"{where}: expected a nonempty string")
    return value


def _identifier(value: Any, where: str) -> str:
    result = _text(value, where)
    if (
        len(result) > 96
        or not result[0].isalpha()
        or any(not (character.isalnum() or character in "_.-") for character in result)
    ):
        raise CampaignV2Error(f"{where}: invalid identifier")
    return result


def _sha256_hex(value: Any, where: str) -> str:
    result = _text(value, where)
    if len(result) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise CampaignV2Error(f"{where}: expected a canonical lowercase 64-hex SHA-256 digest")
    return result


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise CampaignV2Error(f"{where}: Boolean is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CampaignV2Error(f"{where}: invalid integer") from exc
    if str(result) != str(value) and not isinstance(value, int):
        raise CampaignV2Error(f"{where}: integer must be canonical")
    if result < minimum:
        raise CampaignV2Error(f"{where}: must be >= {minimum}")
    return result


def _decimal(value: Any, where: str, *, nonnegative: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise CampaignV2Error(f"{where}: expected an exact decimal string")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise CampaignV2Error(f"{where}: invalid decimal") from exc
    if not result.is_finite():
        raise CampaignV2Error(f"{where}: decimal must be finite")
    if nonnegative and result < 0:
        raise CampaignV2Error(f"{where}: decimal must be nonnegative")
    return result


def _positive_decimal(value: Any, where: str) -> Decimal:
    result = _decimal(value, where)
    if result <= 0:
        raise CampaignV2Error(f"{where}: decimal must be positive")
    return result


def _number_text(value: float | Decimal) -> str:
    """Return a stable finite decimal representation for canonical evidence."""

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise CampaignV2Error("cannot serialize a nonfinite Decimal")
        if value == 0:
            return "0"
        return format(value.normalize(), "f")
    if not math.isfinite(value):
        raise CampaignV2Error("cannot serialize a nonfinite float")
    if value == 0.0:
        return "0"
    return format(value, ".17g")


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by regularized incomplete beta.

    This is a deterministic Lentz evaluation of the standard continued
    fraction.  It is kept local so the campaign foundation has no optional
    SciPy dependency and can authenticate the exact numerical implementation.
    """

    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, 401):
        m = float(iteration)
        m2 = 2.0 * m
        coefficient = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= 8.0e-15:
            return result
    raise CampaignV2Error("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b) and a > 0.0 and b > 0.0):
        raise CampaignV2Error("invalid incomplete-beta shape")
    if not math.isfinite(x) or x < 0.0 or x > 1.0:
        raise CampaignV2Error("invalid incomplete-beta argument")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf_v2(value: float, degrees_of_freedom: float) -> float:
    if (
        not math.isfinite(degrees_of_freedom)
        or degrees_of_freedom < 1.0
        or not math.isfinite(value)
    ):
        raise CampaignV2Error("invalid Student-t CDF argument")
    if value == 0.0:
        return 0.5
    nu = float(degrees_of_freedom)
    beta = _regularized_incomplete_beta(0.5 * nu, 0.5, nu / (nu + value * value))
    return 1.0 - 0.5 * beta if value > 0.0 else 0.5 * beta


def student_t_quantile_v2(probability: float, degrees_of_freedom: float) -> float:
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise CampaignV2Error("Student-t probability must lie in (0,1)")
    if not math.isfinite(degrees_of_freedom) or degrees_of_freedom < 1.0:
        raise CampaignV2Error("Student-t degrees of freedom must be positive")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -student_t_quantile_v2(1.0 - probability, degrees_of_freedom)
    low = 0.0
    high = 1.0
    while student_t_cdf_v2(high, degrees_of_freedom) < probability:
        high *= 2.0
        if high > 1.0e12:
            raise CampaignV2Error("Student-t quantile could not be bracketed")
    for _ in range(100):
        middle = 0.5 * (low + high)
        if student_t_cdf_v2(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _regularized_gamma_p(shape: float, x: float) -> float:
    if not (math.isfinite(shape) and shape > 0.0 and math.isfinite(x) and x >= 0.0):
        raise CampaignV2Error("invalid regularized-gamma argument")
    if x == 0.0:
        return 0.0
    log_prefactor = -x + shape * math.log(x) - math.lgamma(shape)
    if x < shape + 1.0:
        term = 1.0 / shape
        total = term
        denominator = shape
        for _ in range(1, 10001):
            denominator += 1.0
            term *= x / denominator
            total += term
            if abs(term) <= abs(total) * 8.0e-15:
                return total * math.exp(log_prefactor)
        raise CampaignV2Error("regularized-gamma series did not converge")
    tiny = 1.0e-300
    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / max(b, tiny)
    fraction = d
    for iteration in range(1, 10001):
        coefficient = -float(iteration) * (float(iteration) - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= 8.0e-15:
            return 1.0 - math.exp(log_prefactor) * fraction
    raise CampaignV2Error("regularized-gamma continued fraction did not converge")


def chi_square_quantile_v2(probability: float, degrees_of_freedom: int) -> float:
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise CampaignV2Error("chi-square probability must lie in (0,1)")
    if degrees_of_freedom < 1:
        raise CampaignV2Error("chi-square degrees of freedom must be positive")
    shape = 0.5 * float(degrees_of_freedom)
    low = 0.0
    high = max(1.0, float(degrees_of_freedom))
    while _regularized_gamma_p(shape, 0.5 * high) < probability:
        high *= 2.0
        if high > 1.0e12:
            raise CampaignV2Error("chi-square quantile could not be bracketed")
    for _ in range(120):
        middle = 0.5 * (low + high)
        if _regularized_gamma_p(shape, 0.5 * middle) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def normal_sigma_upper_confidence_v2(
    sample_sd: float, sample_count: int, one_sided_alpha: float
) -> float:
    """Normal-model one-sided upper confidence limit for unknown sigma."""

    if (
        not math.isfinite(sample_sd)
        or sample_sd < 0.0
        or sample_count < 2
        or not math.isfinite(one_sided_alpha)
        or not 0.0 < one_sided_alpha < 0.5
    ):
        raise CampaignV2Error("invalid variance-envelope argument")
    lower_chi_square = chi_square_quantile_v2(one_sided_alpha, sample_count - 1)
    if not lower_chi_square > 0.0:
        raise CampaignV2Error("variance-envelope chi-square quantile is nonpositive")
    return sample_sd * math.sqrt((sample_count - 1) / lower_chi_square)


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_object(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(value))


def seal_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop(SELF_HASH_KEY, None)
    result[SELF_HASH_KEY] = sha256_object(result)
    return result


def validate_sealed_artifact(value: Mapping[str, Any], schema: str, where: str) -> dict[str, Any]:
    document = dict(_mapping(value, where))
    if document.get("schema") != schema:
        raise CampaignV2Error(f"{where}: schema mismatch")
    if document.get("schema_version") != SCHEMA_VERSION_V2:
        raise CampaignV2Error(f"{where}: schema_version mismatch")
    supplied = document.pop(SELF_HASH_KEY, None)
    if not isinstance(supplied, str) or len(supplied) != SHA256_HEX_LENGTH:
        raise CampaignV2Error(f"{where}: missing or malformed self hash")
    expected = sha256_object(document)
    if supplied != expected:
        raise CampaignV2Error(f"{where}: self hash mismatch")
    document[SELF_HASH_KEY] = supplied
    return document


def _unique_integers(value: Any, where: str, minimum_count: int) -> list[int]:
    result = [
        _integer(item, f"{where}[{index}]", 0) for index, item in enumerate(_list(value, where))
    ]
    if len(result) < minimum_count:
        raise CampaignV2Error(f"{where}: need at least {minimum_count} values")
    if len(set(result)) != len(result):
        raise CampaignV2Error(f"{where}: duplicate value")
    return result


def _validate_level_ids(levels: Sequence[Mapping[str, Any]], where: str) -> list[str]:
    ids = [
        _identifier(level.get("id"), f"{where}[{index}].id") for index, level in enumerate(levels)
    ]
    if len(set(ids)) != len(ids):
        raise CampaignV2Error(f"{where}: duplicate level id")
    return ids


def _geometric_ratio(values: Sequence[Decimal], where: str, *, decreasing: bool) -> Decimal:
    if len(values) < MINIMUM_PILOT_LEVELS + 1:
        raise CampaignV2Error(
            f"{where}: require three pilot values plus a separate confirmation value"
        )
    ratios: list[Decimal] = []
    for left, right in zip(values, values[1:]):
        if decreasing:
            if not left > right:
                raise CampaignV2Error(f"{where}: values must decrease strictly")
            ratios.append(left / right)
        else:
            if not right > left:
                raise CampaignV2Error(f"{where}: values must increase strictly")
            ratios.append(right / left)
    first = ratios[0]
    if first <= 1 or any(item != first for item in ratios[1:]):
        raise CampaignV2Error(f"{where}: values must have one exact geometric ratio")
    return first


def _validate_scale_contract(endpoint: Mapping[str, Any], where: str) -> dict[str, str]:
    scale = _mapping(endpoint.get("scale_contract"), f"{where}.scale_contract")
    reference = _decimal(scale.get("reference_value"), f"{where}.reference_value")
    floor = _positive_decimal(scale.get("near_zero_floor"), f"{where}.near_zero_floor")
    basis = _positive_decimal(scale.get("scale_basis"), f"{where}.scale_basis")
    relative = _positive_decimal(scale.get("target_relative"), f"{where}.target_relative")
    absolute = _positive_decimal(scale.get("absolute_target"), f"{where}.absolute_target")
    if relative > Decimal("0.01"):
        raise CampaignV2Error(f"{where}: target_relative must not exceed 0.01")
    expected_basis = max(abs(reference), floor)
    if basis != expected_basis:
        raise CampaignV2Error(
            f"{where}: scale_basis must equal max(abs(reference_value), near_zero_floor)"
        )
    if absolute != basis * relative:
        raise CampaignV2Error(f"{where}: absolute_target must equal scale_basis*target_relative")
    reference_provenance = _text(
        scale.get("reference_value_provenance"), f"{where}.reference_value_provenance"
    )
    floor_provenance = _text(
        scale.get("near_zero_floor_provenance"), f"{where}.near_zero_floor_provenance"
    )
    return {
        "reference_value": _number_text(reference),
        "near_zero_floor": _number_text(floor),
        "scale_basis": _number_text(basis),
        "target_relative": _number_text(relative),
        "absolute_target": _number_text(absolute),
        "reference_value_provenance": reference_provenance,
        "near_zero_floor_provenance": floor_provenance,
    }


def _validate_uncertainty_contract(
    endpoint: Mapping[str, Any], where: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    entries = _list(endpoint.get("uncertainty_budget", []), f"{where}.uncertainty_budget")
    for index, raw in enumerate(entries):
        entry = _mapping(raw, f"{where}.uncertainty_budget[{index}]")
        component = _identifier(entry.get("component"), f"{where}.component")
        if component not in REQUIRED_UNCERTAINTY_COMPONENTS:
            raise CampaignV2Error(f"{where}: unknown uncertainty component {component!r}")
        if component in result:
            raise CampaignV2Error(f"{where}: duplicate uncertainty component {component!r}")
        allocation = _decimal(
            entry.get("allocation", "0"), f"{where}.{component}.allocation", nonnegative=True
        )
        if component in COMPUTED_UNCERTAINTY_COMPONENTS:
            result[component] = {
                "component": component,
                "allocation": _number_text(allocation),
                "status": "computed_from_authenticated_observations",
            }
            continue
        status = entry.get("status", "unresolved")
        if status not in {"resolved", "unresolved"}:
            raise CampaignV2Error(f"{where}.{component}.status: invalid status")
        upper_raw = entry.get("upper_bound")
        upper = (
            None
            if upper_raw is None
            else _decimal(upper_raw, f"{where}.{component}.upper_bound", nonnegative=True)
        )
        method = entry.get("method")
        evidence = entry.get("evidence")
        result[component] = {
            "component": component,
            "allocation": _number_text(allocation),
            "status": status,
            "upper_bound": None if upper is None else _number_text(upper),
            "method": method if isinstance(method, str) and method else None,
            "evidence": evidence if isinstance(evidence, str) and evidence else None,
        }
    return result


def validate_request_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate and normalize the prospectively frozen v2 request."""

    request = validate_sealed_artifact(value, REQUEST_SCHEMA_V2, "request")
    campaign_id = _identifier(request.get("campaign_id"), "request.campaign_id")
    collider_family = _identifier(request.get("collider_family"), "request.collider_family")

    methods_raw = _list(request.get("methods"), "request.methods")
    if not methods_raw:
        raise CampaignV2Error("request.methods must not be empty")
    methods: list[dict[str, str]] = []
    method_ids: set[str] = set()
    reference_ids: list[str] = []
    for index, raw in enumerate(methods_raw):
        method = _mapping(raw, f"request.methods[{index}]")
        method_id = _identifier(method.get("id"), f"request.methods[{index}].id")
        role = method.get("role")
        if role not in {"reference", "candidate"}:
            raise CampaignV2Error(f"request.methods[{index}].role: invalid role")
        if method_id in method_ids:
            raise CampaignV2Error(f"request.methods: duplicate id {method_id!r}")
        method_ids.add(method_id)
        if role == "reference":
            reference_ids.append(method_id)
        methods.append({"id": method_id, "role": role})
    if len(reference_ids) != 1:
        raise CampaignV2Error("request.methods requires exactly one reference")

    design = _mapping(request.get("design"), "request.design")
    pilot_grids_raw = [
        _mapping(item, f"request.design.pilot_grid_levels[{index}]")
        for index, item in enumerate(
            _list(design.get("pilot_grid_levels"), "request.design.pilot_grid_levels")
        )
    ]
    if len(pilot_grids_raw) < MINIMUM_PILOT_LEVELS:
        raise CampaignV2Error("design requires at least three pilot grid levels")
    confirmation_grid_raw = _mapping(
        design.get("confirmation_grid_level"), "request.design.confirmation_grid_level"
    )
    all_grids_raw = [*pilot_grids_raw, confirmation_grid_raw]
    _validate_level_ids(all_grids_raw, "request.design.grid_levels")
    grids: list[dict[str, Any]] = []
    for index, level in enumerate(all_grids_raw):
        n_x = _integer(level.get("n_x"), f"grid[{index}].n_x", 2)
        n_y = _integer(level.get("n_y"), f"grid[{index}].n_y", 2)
        h = _positive_decimal(level.get("h"), f"grid[{index}].h")
        grids.append(
            {
                "id": _identifier(level.get("id"), f"grid[{index}].id"),
                "n_x": n_x,
                "n_y": n_y,
                "h": _number_text(h),
            }
        )
    grid_ratio = _geometric_ratio(
        [Decimal(level["h"]) for level in grids], "grid spacing", decreasing=True
    )
    if grid_ratio != Decimal("2"):
        raise CampaignV2Error("grid spacing ladder must use exact ratio-two refinement")
    extent_x = Decimal(grids[0]["n_x"]) * Decimal(grids[0]["h"])
    extent_y = Decimal(grids[0]["n_y"]) * Decimal(grids[0]["h"])
    for level in grids[1:]:
        if (
            Decimal(level["n_x"]) * Decimal(level["h"]) != extent_x
            or Decimal(level["n_y"]) * Decimal(level["h"]) != extent_y
        ):
            raise CampaignV2Error(
                "grid ladder must preserve both frozen physical extents and aspect ratio"
            )

    pilot_roots_raw = [
        _mapping(item, f"request.design.pilot_root_levels[{index}]")
        for index, item in enumerate(
            _list(design.get("pilot_root_levels"), "request.design.pilot_root_levels")
        )
    ]
    if len(pilot_roots_raw) < MINIMUM_PILOT_LEVELS:
        raise CampaignV2Error("design requires at least three pilot root levels")
    confirmation_root_raw = _mapping(
        design.get("confirmation_root_level"), "request.design.confirmation_root_level"
    )
    all_roots_raw = [*pilot_roots_raw, confirmation_root_raw]
    _validate_level_ids(all_roots_raw, "request.design.root_levels")
    roots: list[dict[str, Any]] = []
    for index, level in enumerate(all_roots_raw):
        roots.append(
            {
                "id": _identifier(level.get("id"), f"root[{index}].id"),
                "n_m": _integer(level.get("n_m"), f"root[{index}].n_m", 1),
            }
        )
    root_ratio = _geometric_ratio(
        [Decimal(level["n_m"]) for level in roots], "root population", decreasing=False
    )
    if design.get("root_prefix_contract") != "nested_initial_root_prefix_v2":
        raise CampaignV2Error("design.root_prefix_contract must be nested_initial_root_prefix_v2")
    physical_contract = _mapping(
        design.get("fixed_physical_contract"), "request.design.fixed_physical_contract"
    )
    required_contract_keys = {
        "domain_sha256",
        "source_shape_sha256",
        "collision_lattice_sha256",
        "longitudinal_time_sha256",
        "fit_contract_sha256",
    }
    if set(physical_contract) != required_contract_keys:
        raise CampaignV2Error("fixed_physical_contract must contain five exact SHA-256 bindings")
    physical_contract = {
        key: _sha256_hex(physical_contract[key], f"fixed_physical_contract.{key}")
        for key in sorted(required_contract_keys)
    }

    seeds = _mapping(request.get("seed_pools"), "request.seed_pools")
    seed_pools = {
        "tuning": _unique_integers(seeds.get("tuning"), "seed_pools.tuning", 2),
        "confirmation": _unique_integers(seeds.get("confirmation"), "seed_pools.confirmation", 2),
        "extension": _unique_integers(seeds.get("extension", []), "seed_pools.extension", 0),
        "holdout": _unique_integers(seeds.get("holdout"), "seed_pools.holdout", 1),
    }
    named = list(seed_pools)
    for left_index, left in enumerate(named):
        for right in named[left_index + 1 :]:
            overlap = set(seed_pools[left]).intersection(seed_pools[right])
            if overlap:
                raise CampaignV2Error(
                    f"seed pools {left!r} and {right!r} overlap: {sorted(overlap)}"
                )

    endpoints_raw = _list(request.get("endpoints"), "request.endpoints")
    if not endpoints_raw:
        raise CampaignV2Error("request.endpoints must not be empty")
    endpoints: list[dict[str, Any]] = []
    endpoint_ids: set[str] = set()
    for index, raw in enumerate(endpoints_raw):
        endpoint = _mapping(raw, f"request.endpoints[{index}]")
        endpoint_id = _identifier(endpoint.get("id"), f"request.endpoints[{index}].id")
        if endpoint_id in endpoint_ids:
            raise CampaignV2Error(f"request.endpoints: duplicate id {endpoint_id!r}")
        endpoint_ids.add(endpoint_id)
        kind = endpoint.get("distribution_kind")
        if kind not in SUPPORTED_ENDPOINT_KINDS:
            raise CampaignV2Error(f"endpoint {endpoint_id}: unsupported distribution kind")
        margin = _positive_decimal(
            endpoint.get("equivalence_margin_relative"),
            f"endpoint[{endpoint_id}].equivalence_margin_relative",
        )
        expected = _decimal(
            endpoint.get("expected_true_difference_relative", "0"),
            f"endpoint[{endpoint_id}].expected_true_difference_relative",
        )
        if abs(expected) >= margin:
            raise CampaignV2Error(
                f"endpoint {endpoint_id}: expected difference must lie inside the margin"
            )
        endpoints.append(
            {
                "id": endpoint_id,
                "units": _text(endpoint.get("units"), f"endpoint[{endpoint_id}].units"),
                "distribution_kind": kind,
                "scale_contract": _validate_scale_contract(endpoint, f"endpoint[{endpoint_id}]"),
                "equivalence_margin_relative": _number_text(margin),
                "expected_true_difference_relative": _number_text(expected),
                "uncertainty_budget": _validate_uncertainty_contract(
                    endpoint, f"endpoint[{endpoint_id}]"
                ),
            }
        )

    claims_raw = _list(request.get("claims"), "request.claims")
    claims: list[dict[str, str]] = []
    claim_ids: set[str] = set()
    claimed_endpoints: set[str] = set()
    for index, raw in enumerate(claims_raw):
        claim = _mapping(raw, f"request.claims[{index}]")
        claim_id = _identifier(claim.get("id"), f"request.claims[{index}].id")
        if claim_id in claim_ids:
            raise CampaignV2Error(f"request.claims: duplicate id {claim_id!r}")
        claim_ids.add(claim_id)
        endpoint_id = _identifier(claim.get("endpoint_id"), f"claim[{claim_id}].endpoint_id")
        if endpoint_id not in endpoint_ids:
            raise CampaignV2Error(f"claim {claim_id}: unknown endpoint")
        family_id = _identifier(claim.get("family_id"), f"claim[{claim_id}].family_id")
        kind = claim.get("kind")
        if kind not in {"method_equivalence", "absolute_convergence", "exact_invariant"}:
            raise CampaignV2Error(f"claim {claim_id}: invalid kind")
        endpoint_kind = next(
            endpoint["distribution_kind"] for endpoint in endpoints if endpoint["id"] == endpoint_id
        )
        if endpoint_kind == "exact_invariant" and kind != "exact_invariant":
            raise CampaignV2Error(
                f"claim {claim_id}: exact_invariant endpoint requires an exact_invariant claim"
            )
        if endpoint_kind != "exact_invariant" and kind == "exact_invariant":
            raise CampaignV2Error(
                f"claim {claim_id}: exact_invariant claim requires an exact_invariant endpoint"
            )
        candidate = claim.get("candidate_method_id")
        reference = claim.get("reference_method_id", reference_ids[0])
        if reference != reference_ids[0]:
            raise CampaignV2Error(f"claim {claim_id}: reference differs from frozen reference")
        if kind == "method_equivalence":
            if candidate not in method_ids or candidate == reference:
                raise CampaignV2Error(f"claim {claim_id}: invalid candidate method")
        elif candidate is not None:
            raise CampaignV2Error(f"claim {claim_id}: candidate is forbidden for {kind}")
        claims.append(
            {
                "id": claim_id,
                "endpoint_id": endpoint_id,
                "family_id": family_id,
                "kind": kind,
                "candidate_method_id": candidate,
                "reference_method_id": reference,
            }
        )
        claimed_endpoints.add(endpoint_id)

    # A method-equivalence claim alone cannot establish that both methods are
    # close to the frozen physical reference: they may share an arbitrarily
    # large common bias.  Require one absolute reference claim for every
    # stochastic endpoint and one equivalence claim for every candidate.  An
    # exact invariant instead has exactly its zero-tolerance identity claim.
    candidate_ids = [method["id"] for method in methods if method["role"] == "candidate"]
    for endpoint in endpoints:
        endpoint_claims = [claim for claim in claims if claim["endpoint_id"] == endpoint["id"]]
        kinds = [claim["kind"] for claim in endpoint_claims]
        if endpoint["distribution_kind"] == "exact_invariant":
            if kinds != ["exact_invariant"]:
                raise CampaignV2Error(
                    f"endpoint {endpoint['id']}: require exactly one exact_invariant claim"
                )
            continue
        if kinds.count("absolute_convergence") != 1:
            raise CampaignV2Error(
                f"endpoint {endpoint['id']}: require exactly one absolute_convergence claim"
            )
        equivalence_candidates = sorted(
            claim["candidate_method_id"]
            for claim in endpoint_claims
            if claim["kind"] == "method_equivalence"
        )
        if equivalence_candidates != sorted(candidate_ids):
            raise CampaignV2Error(
                f"endpoint {endpoint['id']}: require one method_equivalence claim per candidate"
            )

    statistics = _mapping(request.get("statistics"), "request.statistics")
    familywise_alpha = _positive_decimal(
        statistics.get("familywise_alpha"), "statistics.familywise_alpha"
    )
    desired_power = _positive_decimal(statistics.get("desired_power"), "statistics.desired_power")
    if familywise_alpha >= Decimal("0.5"):
        raise CampaignV2Error("statistics.familywise_alpha must be below 0.5")
    if desired_power < Decimal("0.9") or desired_power >= 1:
        raise CampaignV2Error("statistics.desired_power must be in [0.9, 1)")
    if statistics.get("multiplicity_method") != "bonferroni_v2":
        raise CampaignV2Error("statistics.multiplicity_method must be bonferroni_v2")
    heldout_floor = _integer(
        statistics.get("minimum_heldout_seed_blocks", DEFAULT_HELDOUT_SEED_FLOOR),
        "statistics.minimum_heldout_seed_blocks",
        1,
    )
    if heldout_floor < DEFAULT_HELDOUT_SEED_FLOOR:
        raise CampaignV2Error("statistics.minimum_heldout_seed_blocks cannot lower the v2 floor")
    safety_factor = _positive_decimal(
        statistics.get("gci_safety_factor", "1.25"), "statistics.gci_safety_factor"
    )
    p_min = _positive_decimal(statistics.get("minimum_order", "0.25"), "minimum_order")
    p_max = _positive_decimal(statistics.get("maximum_order", "8"), "maximum_order")
    if p_min >= p_max:
        raise CampaignV2Error("minimum_order must be below maximum_order")
    order_tolerance = _positive_decimal(
        statistics.get("confirmation_order_tolerance", "1"),
        "statistics.confirmation_order_tolerance",
    )
    asymptotic_ratio_min = _positive_decimal(
        statistics.get("asymptotic_ratio_min", "0.5"), "statistics.asymptotic_ratio_min"
    )
    asymptotic_ratio_max = _positive_decimal(
        statistics.get("asymptotic_ratio_max", "2"), "statistics.asymptotic_ratio_max"
    )
    if asymptotic_ratio_min >= asymptotic_ratio_max:
        raise CampaignV2Error("asymptotic ratio limits are reversed")

    blind_raw = statistics.get("blinded_power_contract")
    blinded_power_contract = None
    if blind_raw is not None:
        blind = _mapping(blind_raw, "statistics.blinded_power_contract")
        blinded_power_contract = {
            "status": blind.get("status"),
            "candidate_ids_blinded": blind.get("candidate_ids_blinded"),
            "prospectively_frozen": blind.get("prospectively_frozen"),
            "sigma_upper_bound_relative": _number_text(
                _positive_decimal(
                    blind.get("sigma_upper_bound_relative"),
                    "blinded_power_contract.sigma_upper_bound_relative",
                )
            ),
            "requested_seed_blocks": str(
                _integer(
                    blind.get("requested_seed_blocks"),
                    "blinded_power_contract.requested_seed_blocks",
                    2,
                )
            ),
            "method": _text(blind.get("method"), "blinded_power_contract.method"),
            "provenance": _text(blind.get("provenance"), "blinded_power_contract.provenance"),
        }

    missing_claim_endpoints = sorted(endpoint_ids - claimed_endpoints)
    return {
        "campaign_id": campaign_id,
        "collider_family": collider_family,
        "request_self_sha256": request[SELF_HASH_KEY],
        "methods": methods,
        "reference_method_id": reference_ids[0],
        "pilot_grids": grids[:-1],
        "confirmation_grid": grids[-1],
        "all_grids": grids,
        "pilot_roots": roots[:-1],
        "confirmation_root": roots[-1],
        "all_roots": roots,
        "grid_ratio": _number_text(grid_ratio),
        "grid_extent_x_in_h_units": _number_text(extent_x),
        "grid_extent_y_in_h_units": _number_text(extent_y),
        "root_ratio": _number_text(root_ratio),
        "root_prefix_contract": design["root_prefix_contract"],
        "fixed_physical_contract": dict(physical_contract),
        "seed_pools": seed_pools,
        "endpoints": endpoints,
        "claims": claims,
        "missing_claim_endpoints": missing_claim_endpoints,
        "statistics": {
            "familywise_alpha": _number_text(familywise_alpha),
            "desired_power": _number_text(desired_power),
            "multiplicity_method": "bonferroni_v2",
            "minimum_heldout_seed_blocks": str(heldout_floor),
            "gci_safety_factor": _number_text(safety_factor),
            "minimum_order": _number_text(p_min),
            "maximum_order": _number_text(p_max),
            "confirmation_order_tolerance": _number_text(order_tolerance),
            "asymptotic_ratio_min": _number_text(asymptotic_ratio_min),
            "asymptotic_ratio_max": _number_text(asymptotic_ratio_max),
            "blinded_power_contract": blinded_power_contract,
        },
    }


def _stage_cells(
    config: Mapping[str, Any], stage: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if stage == "tuning_pilot":
        grids = list(config["pilot_grids"])
        roots = list(config["pilot_roots"])
    elif stage == "finer_confirmation":
        grids = list(config["all_grids"][-3:])
        roots = list(config["all_roots"][-3:])
    elif stage == "holdout_bracket":
        grids = list(config["all_grids"][-2:])
        roots = list(config["all_roots"][-2:])
    else:
        raise CampaignV2Error(f"unknown stage {stage!r}")
    return [(grid, root) for grid in grids for root in roots]


def counterbalanced_execution_order_v2(
    method_ids: Sequence[str],
    cells: Sequence[tuple[str, str]],
    seeds: Sequence[int],
    stage: str,
) -> list[dict[str, Any]]:
    """Create deterministic AB/BA or cyclic-Latin execution blocks."""

    if not method_ids:
        raise CampaignV2Error("counterbalanced schedule has no methods")
    schedule: list[dict[str, Any]] = []
    block_index = 0
    for seed in seeds:
        for grid_id, root_id in cells:
            offset = block_index % len(method_ids)
            order = list(method_ids[offset:]) + list(method_ids[:offset])
            # Alternating reversal suppresses monotonic warm-up bias for more
            # than two methods while preserving equal first-position counts.
            if len(method_ids) > 2 and (block_index // len(method_ids)) % 2:
                order = list(reversed(order))
            schedule.append(
                {
                    "block_index": str(block_index),
                    "stage": stage,
                    "seed": str(seed),
                    "grid_id": grid_id,
                    "root_id": root_id,
                    "method_order": order,
                }
            )
            block_index += 1
    return schedule


def materialize_pilot_manifest_v2(request_value: Mapping[str, Any]) -> dict[str, Any]:
    config = validate_request_v2(request_value)
    method_ids = [method["id"] for method in config["methods"]]
    realizations: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    ordinal = 0
    for stage, seed_pool in (
        ("tuning_pilot", "tuning"),
        ("finer_confirmation", "confirmation"),
    ):
        cells = _stage_cells(config, stage)
        cell_ids = [(grid["id"], root["id"]) for grid, root in cells]
        stage_schedule = counterbalanced_execution_order_v2(
            method_ids, cell_ids, config["seed_pools"][seed_pool], stage
        )
        schedules.extend(stage_schedule)
        by_grid = {grid["id"]: grid for grid in config["all_grids"]}
        by_root = {root["id"]: root for root in config["all_roots"]}
        for block in stage_schedule:
            for position, method_id in enumerate(block["method_order"]):
                grid = by_grid[block["grid_id"]]
                root = by_root[block["root_id"]]
                realization_id = (
                    f"{stage}.{int(block['block_index']):05d}."
                    f"{method_id}.{grid['id']}.{root['id']}.seed{block['seed']}"
                )
                realizations.append(
                    {
                        "ordinal": str(ordinal),
                        "realization_id": realization_id,
                        "stage": stage,
                        "timing_block_index": block["block_index"],
                        "timing_position": str(position),
                        "method_id": method_id,
                        "grid_id": grid["id"],
                        "root_id": root["id"],
                        "n_x": str(grid["n_x"]),
                        "n_y": str(grid["n_y"]),
                        "h": grid["h"],
                        "n_m": str(root["n_m"]),
                        "seed": block["seed"],
                        "seed_population": seed_pool,
                        "independent_seed_block": block["seed"],
                    }
                )
                ordinal += 1
    unsigned = {
        "schema": MANIFEST_SCHEMA_V2,
        "schema_version": SCHEMA_VERSION_V2,
        "qualification_status": "unexecuted_v2_foundation_not_physics_qualified",
        "qualification_blocker": FOUNDATION_QUALIFICATION_BLOCKER,
        "campaign_id": config["campaign_id"],
        "request_self_sha256": config["request_self_sha256"],
        "fixed_physical_contract": config["fixed_physical_contract"],
        "counterbalance_contract": "cyclic_latin_with_alternating_reversal_v2",
        "schedules": schedules,
        "realizations": realizations,
        "independent_seed_blocks": str(
            len(config["seed_pools"]["tuning"]) + len(config["seed_pools"]["confirmation"])
        ),
        "materialized_feebl_processes": str(len(realizations)),
    }
    return seal_artifact(unsigned)


def validate_manifest_v2(
    request_value: Mapping[str, Any], manifest_value: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = validate_sealed_artifact(manifest_value, MANIFEST_SCHEMA_V2, "pilot manifest")
    expected = materialize_pilot_manifest_v2(request_value)
    if canonical_bytes(manifest) != canonical_bytes(expected):
        raise CampaignV2Error("pilot manifest differs from deterministic request materialization")
    return manifest


def _validate_endpoint_observation(value: Any, endpoint: Mapping[str, Any], where: str) -> str:
    parsed = _decimal(value, where)
    kind = endpoint["distribution_kind"]
    if kind in {
        "poisson_count",
        "poisson_rate",
        "binomial_proportion",
        "paired_bernoulli",
        "weighted_compound_poisson",
    }:
        if parsed < 0:
            raise CampaignV2Error(f"{where}: count/proportion cannot be negative")
    if kind == "poisson_count" and parsed != parsed.to_integral_value():
        raise CampaignV2Error(f"{where}: Poisson count must be integral")
    if kind in {"binomial_proportion", "paired_bernoulli"} and parsed > 1:
        raise CampaignV2Error(f"{where}: proportion must be in [0,1]")
    return _number_text(parsed)


def _authenticated_pilot_observations_v2(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    state_value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Return authenticated, complete observations in manifest order."""

    config = validate_request_v2(request_value)
    manifest = validate_manifest_v2(request_value, manifest_value)
    state = validate_sealed_artifact(state_value, STATE_SCHEMA_V2, "pilot state")
    if state.get("campaign_id") != config["campaign_id"]:
        raise CampaignV2Error("pilot state campaign id mismatch")
    if state.get("request_self_sha256") != config["request_self_sha256"]:
        raise CampaignV2Error("pilot state request binding mismatch")
    if state.get("manifest_self_sha256") != manifest[SELF_HASH_KEY]:
        raise CampaignV2Error("pilot state manifest binding mismatch")
    if state.get("status") != "complete":
        raise CampaignV2Error("pilot state is not complete")
    completed = _list(state.get("completed"), "pilot state.completed")
    realizations = _list(manifest.get("realizations"), "pilot manifest.realizations")
    if len(completed) != len(realizations):
        raise CampaignV2Error("pilot state does not cover every materialized process")
    endpoints = {endpoint["id"]: endpoint for endpoint in config["endpoints"]}
    observations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (raw_record, realization) in enumerate(zip(completed, realizations)):
        record = _mapping(raw_record, f"pilot state.completed[{index}]")
        realization_id = realization["realization_id"]
        if record.get("realization_id") != realization_id:
            raise CampaignV2Error(
                f"pilot state.completed[{index}]: realization order/identity mismatch"
            )
        if realization_id in seen:
            raise CampaignV2Error("pilot state contains a duplicate realization")
        seen.add(realization_id)
        elapsed_ns = _integer(record.get("elapsed_ns"), f"record[{index}].elapsed_ns", 1)
        raw_values = _mapping(record.get("endpoint_values"), f"record[{index}].endpoint_values")
        if set(raw_values) != set(endpoints):
            raise CampaignV2Error(f"record[{index}]: endpoint set differs from frozen request")
        values = {
            endpoint_id: _validate_endpoint_observation(
                raw_values[endpoint_id], endpoint, f"record[{index}].{endpoint_id}"
            )
            for endpoint_id, endpoint in endpoints.items()
        }
        evidence_hash = _sha256_hex(
            record.get("execution_evidence_sha256"),
            f"record[{index}].execution_evidence_sha256",
        )
        observations.append(
            {
                **dict(realization),
                "elapsed_ns": str(elapsed_ns),
                "endpoint_values": values,
                "execution_evidence_sha256": evidence_hash,
            }
        )
    return config, manifest, observations


def authenticated_pilot_observations_v2(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    state_value: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return _authenticated_pilot_observations_v2(request_value, manifest_value, state_value)[2]


def normal_tost_power_v2(
    sample_size: int,
    sigma: float,
    margin: float,
    true_difference: float,
    one_sided_alpha: float,
) -> float:
    """Exact known-variance normal-theory TOST acceptance probability."""

    if sample_size < 2:
        raise CampaignV2Error("TOST sample size must be at least two")
    if sigma < 0 or margin <= 0 or not (0 < one_sided_alpha < 0.5):
        raise CampaignV2Error("invalid exact-normal TOST parameters")
    if abs(true_difference) >= margin:
        return 0.0
    if sigma == 0:
        return 1.0
    standard_error = sigma / math.sqrt(sample_size)
    critical = NormalDist().inv_cdf(1.0 - one_sided_alpha)
    lower = -margin + critical * standard_error
    upper = margin - critical * standard_error
    if lower >= upper:
        return 0.0
    normal = NormalDist(mu=true_difference, sigma=standard_error)
    return max(0.0, min(1.0, normal.cdf(upper) - normal.cdf(lower)))


def exact_normal_tost_required_n_v2(
    sigma: float,
    margin: float,
    true_difference: float,
    one_sided_alpha: float,
    desired_power: float,
    *,
    maximum_n: int = 10_000_000,
) -> int:
    """Solve the normal TOST power equation by monotone integer search."""

    if not (0 < desired_power < 1):
        raise CampaignV2Error("desired TOST power must be in (0,1)")
    if abs(true_difference) >= margin:
        raise CampaignV2Error("true difference lies outside the equivalence margin")
    if sigma == 0:
        return 2
    low = 2
    if normal_tost_power_v2(low, sigma, margin, true_difference, one_sided_alpha) >= desired_power:
        return low
    high = low
    while high < maximum_n:
        high = min(maximum_n, high * 2)
        if (
            normal_tost_power_v2(high, sigma, margin, true_difference, one_sided_alpha)
            >= desired_power
        ):
            break
    else:
        raise CampaignV2Error("exact-normal TOST power target exceeds maximum_n")
    if normal_tost_power_v2(high, sigma, margin, true_difference, one_sided_alpha) < desired_power:
        raise CampaignV2Error("exact-normal TOST power target exceeds maximum_n")
    while low + 1 < high:
        middle = (low + high) // 2
        if (
            normal_tost_power_v2(middle, sigma, margin, true_difference, one_sided_alpha)
            >= desired_power
        ):
            high = middle
        else:
            low = middle
    return high


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise CampaignV2Error("empty sample")
    return math.fsum(values) / len(values)


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise CampaignV2Error("sample standard deviation requires two values")
    center = _mean(values)
    return math.sqrt(math.fsum((item - center) ** 2 for item in values) / (len(values) - 1))


def _paired_summary(values: Sequence[float], critical: float) -> dict[str, Any]:
    center = _mean(values)
    sd = _sample_sd(values)
    se = sd / math.sqrt(len(values))
    one_sided_alpha = 1.0 - NormalDist().cdf(critical)
    student_critical = student_t_quantile_v2(1.0 - one_sided_alpha, len(values) - 1)
    half_width = student_critical * se
    return {
        "independent_seed_blocks": str(len(values)),
        "mean": _number_text(center),
        "sample_sd": _number_text(sd),
        "standard_error": _number_text(se),
        "normal_critical_diagnostic": _number_text(critical),
        "unknown_variance_student_t_critical": _number_text(student_critical),
        "simultaneous_half_width": _number_text(half_width),
        "simultaneous_upper_absolute": _number_text(abs(center) + half_width),
    }


def _convergence_triplet(
    q1: float,
    q2: float,
    q3: float,
    ratio: float,
    safety_factor: float,
    p_min: float,
    p_max: float,
) -> dict[str, Any]:
    d12 = q1 - q2
    d23 = q2 - q3
    reasons: list[str] = []
    sign_contracting = d12 * d23 > 0 and abs(d23) < abs(d12)
    if not sign_contracting:
        reasons.append("differences_not_monotone_sign_contracting")
    order: float | None = None
    gci: float | None = None
    if d12 != 0 and d23 != 0:
        order = math.log(abs(d12 / d23)) / math.log(ratio)
        if not math.isfinite(order) or not (p_min <= order <= p_max):
            reasons.append("observed_order_outside_plausible_range")
        elif sign_contracting:
            denominator = ratio**order - 1.0
            if denominator <= 0:
                reasons.append("nonpositive_richardson_denominator")
            else:
                gci = safety_factor * abs(d23) / denominator
    else:
        reasons.append("zero_difference_prevents_order_estimate")
    passed = not reasons and order is not None and gci is not None
    return {
        "values": [_number_text(q1), _number_text(q2), _number_text(q3)],
        "differences": [_number_text(d12), _number_text(d23)],
        "sign_contracting": sign_contracting,
        "observed_order": None if order is None else _number_text(order),
        "gci_finest_absolute": None if gci is None else _number_text(gci),
        "passed": passed,
        "reasons": reasons,
    }


def _closed_interval_absolute_extrema(low: float, high: float) -> tuple[float, float]:
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise CampaignV2Error("invalid confidence interval")
    minimum = 0.0 if low <= 0.0 <= high else min(abs(low), abs(high))
    return minimum, max(abs(low), abs(high))


def _stochastic_convergence_triplet(
    cell_values: Sequence[Mapping[int, float]],
    ratio: float,
    safety_factor: float,
    p_min: float,
    p_max: float,
    critical: float,
) -> dict[str, Any]:
    """Richardson/GCI evidence with paired seed-block contrast uncertainty.

    The point triplet is retained for interpretability, but certification uses
    simultaneous Student-t intervals for both adjacent paired differences.
    The lower contraction order and upper finest difference yield the GCI
    upper bound.  If either contrast can change sign, contraction is not
    statistically established and the result fails closed.
    """

    if len(cell_values) != 3:
        raise CampaignV2Error("stochastic convergence requires three cells")
    means = [_mean(list(values.values())) for values in cell_values]
    point = _convergence_triplet(means[0], means[1], means[2], ratio, safety_factor, p_min, p_max)
    first = _paired_summary(_paired_differences(cell_values[0], cell_values[1]), critical)
    second = _paired_summary(_paired_differences(cell_values[1], cell_values[2]), critical)
    intervals: list[tuple[float, float]] = []
    for summary in (first, second):
        center = float(summary["mean"])
        half = float(summary["simultaneous_half_width"])
        intervals.append((center - half, center + half))
    low1, high1 = intervals[0]
    low2, high2 = intervals[1]
    same_sign = (low1 > 0.0 and low2 > 0.0) or (high1 < 0.0 and high2 < 0.0)
    min1, max1 = _closed_interval_absolute_extrema(low1, high1)
    min2, max2 = _closed_interval_absolute_extrema(low2, high2)
    robust_contraction = same_sign and min1 > max2 and max2 > 0.0
    reasons: list[str] = []
    if not point["passed"]:
        reasons.extend(f"point_{reason}" for reason in point["reasons"])
    if not same_sign:
        reasons.append("paired_contrast_confidence_intervals_do_not_fix_common_sign")
    if same_sign and not robust_contraction:
        reasons.append("paired_contrast_confidence_intervals_do_not_prove_contraction")
    conservative_order: float | None = None
    conservative_order_upper: float | None = None
    conservative_gci: float | None = None
    if robust_contraction:
        conservative_order = math.log(min1 / max2) / math.log(ratio)
        if min2 > 0.0:
            conservative_order_upper = math.log(max1 / min2) / math.log(ratio)
        if not math.isfinite(conservative_order) or conservative_order < p_min:
            reasons.append("confidence_bounded_order_below_plausible_minimum")
        elif (
            conservative_order_upper is None
            or not math.isfinite(conservative_order_upper)
            or conservative_order_upper > p_max
        ):
            reasons.append("confidence_bounded_order_above_plausible_maximum")
        else:
            denominator = ratio**conservative_order - 1.0
            if not math.isfinite(denominator) or denominator <= 0.0:
                reasons.append("confidence_bounded_richardson_denominator_invalid")
            else:
                conservative_gci = safety_factor * max2 / denominator
                if not math.isfinite(conservative_gci):
                    reasons.append("confidence_bounded_GCI_nonfinite")
    passed = not reasons and conservative_gci is not None
    return {
        **point,
        "point_gci_finest_absolute": point["gci_finest_absolute"],
        "gci_finest_absolute": (
            None if conservative_gci is None else _number_text(conservative_gci)
        ),
        "paired_contrast_summaries": [first, second],
        "paired_contrast_intervals": [
            [_number_text(low1), _number_text(high1)],
            [_number_text(low2), _number_text(high2)],
        ],
        "confidence_intervals_fix_common_sign": same_sign,
        "confidence_bounded_contraction": robust_contraction,
        "confidence_bounded_order": (
            None if conservative_order is None else _number_text(conservative_order)
        ),
        "confidence_bounded_order_upper": (
            None if conservative_order_upper is None else _number_text(conservative_order_upper)
        ),
        "passed": passed,
        "reasons": reasons,
    }


def _independent_cell_consistency_summary(
    left: Sequence[float], right: Sequence[float], critical: float
) -> dict[str, Any]:
    """Welch Student-t bound for disjoint tuning/confirmation seed pools."""

    if len(left) < 2 or len(right) < 2:
        raise CampaignV2Error("independent consistency comparison needs two samples")
    mean_difference = _mean(left) - _mean(right)
    left_variance_term = _sample_sd(left) ** 2 / len(left)
    right_variance_term = _sample_sd(right) ** 2 / len(right)
    standard_error = math.sqrt(left_variance_term + right_variance_term)
    one_sided_alpha = 1.0 - NormalDist().cdf(critical)
    if standard_error == 0.0:
        degrees_of_freedom: float | None = None
        student_critical = 0.0
        half_width = 0.0
    else:
        denominator = left_variance_term * left_variance_term / (
            len(left) - 1
        ) + right_variance_term * right_variance_term / (len(right) - 1)
        if not denominator > 0.0:
            raise CampaignV2Error("Welch degrees-of-freedom denominator vanished")
        degrees_of_freedom = (left_variance_term + right_variance_term) ** 2 / denominator
        student_critical = student_t_quantile_v2(1.0 - one_sided_alpha, degrees_of_freedom)
        half_width = student_critical * standard_error
    return {
        "left_independent_seed_blocks": str(len(left)),
        "right_independent_seed_blocks": str(len(right)),
        "mean_difference": _number_text(mean_difference),
        "standard_error": _number_text(standard_error),
        "welch_degrees_of_freedom": (
            None if degrees_of_freedom is None else _number_text(degrees_of_freedom)
        ),
        "unknown_variance_student_t_critical": _number_text(student_critical),
        "simultaneous_half_width": _number_text(half_width),
        "simultaneous_upper_absolute": _number_text(abs(mean_difference) + half_width),
    }


def _confirmed_convergence(
    tuning: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    ratio: float,
    order_tolerance: float,
    asymptotic_min: float,
    asymptotic_max: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not tuning["passed"]:
        reasons.append("tuning_triplet_failed")
    if not confirmation["passed"]:
        reasons.append("finer_confirmation_triplet_failed")
    asymptotic_ratio: float | None = None
    order_delta: float | None = None
    if tuning["observed_order"] is not None and confirmation["observed_order"] is not None:
        p_tuning = float(tuning["observed_order"])
        p_confirmation = float(confirmation["observed_order"])
        order_delta = abs(p_tuning - p_confirmation)
        if order_delta > order_tolerance:
            reasons.append("observed_order_not_stable_on_finer_confirmation")
        tuning_ratio = ratio**p_tuning
        confirmation_ratio = ratio**p_confirmation
        asymptotic_ratio = tuning_ratio / confirmation_ratio
        if not (asymptotic_min <= asymptotic_ratio <= asymptotic_max):
            reasons.append("asymptotic_ratio_outside_frozen_interval")
    else:
        reasons.append("confirmation_order_comparison_unavailable")
    return {
        "tuning_triplet": dict(tuning),
        "finer_confirmation_triplet": dict(confirmation),
        "order_absolute_change": None if order_delta is None else _number_text(order_delta),
        "asymptotic_contraction_ratio": (
            None if asymptotic_ratio is None else _number_text(asymptotic_ratio)
        ),
        "passed": not reasons,
        "reasons": reasons,
    }


def _solve_linear_system(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    size = len(rhs)
    augmented = [list(matrix[row]) + [rhs[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-30:
            raise CampaignV2Error("crossed convergence fit is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [item / scale for item in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _crossed_model_fit(
    cells: Sequence[tuple[float, float, float]], p: float, alpha: float
) -> dict[str, Any]:
    if (
        len(cells) < 4
        or not math.isfinite(p)
        or not math.isfinite(alpha)
        or p <= 0.0
        or alpha <= 0.0
    ):
        raise CampaignV2Error("invalid crossed convergence model input")
    rows = [[1.0, h**p, n_m ** (-alpha), h**p * n_m ** (-alpha)] for h, n_m, _ in cells]
    if any(not math.isfinite(value) for row in rows for value in row) or any(
        not math.isfinite(cell[2]) for cell in cells
    ):
        raise CampaignV2Error("crossed convergence design is non-finite")

    # Column-scaled, twice-reorthogonalized modified Gram-Schmidt QR avoids
    # squaring the condition number as the former normal equations did.
    # The diagonal ratio is a conservative rank/conditioning diagnostic; an
    # ill-conditioned design is unresolved rather than promoted to a bound.
    column_count = 4
    column_scales = [
        math.sqrt(math.fsum(row[column] ** 2 for row in rows)) for column in range(column_count)
    ]
    if any(not scale > 0.0 or not math.isfinite(scale) for scale in column_scales):
        raise CampaignV2Error("crossed convergence design has a zero column")
    columns = [
        [row[column] / column_scales[column] for row in rows] for column in range(column_count)
    ]
    q_columns: list[list[float]] = []
    upper = [[0.0] * column_count for _ in range(column_count)]
    for column in range(column_count):
        work = list(columns[column])
        for _ in range(2):
            for prior, q_column in enumerate(q_columns):
                projection = math.fsum(q_value * value for q_value, value in zip(q_column, work))
                upper[prior][column] += projection
                work = [value - projection * q_value for value, q_value in zip(work, q_column)]
        norm = math.sqrt(math.fsum(value * value for value in work))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            raise CampaignV2Error("crossed convergence QR is rank deficient")
        upper[column][column] = norm
        q_columns.append([value / norm for value in work])
    diagonal = [abs(upper[index][index]) for index in range(column_count)]
    condition_proxy = max(diagonal) / min(diagonal)
    if not math.isfinite(condition_proxy) or condition_proxy > 1.0e10:
        raise CampaignV2Error("crossed convergence QR is ill conditioned")
    values = [cell[2] for cell in cells]
    projected = [
        math.fsum(q_value * value for q_value, value in zip(q_column, values))
        for q_column in q_columns
    ]
    scaled_coefficients = [0.0] * column_count
    for row in range(column_count - 1, -1, -1):
        remainder = projected[row] - math.fsum(
            upper[row][column] * scaled_coefficients[column]
            for column in range(row + 1, column_count)
        )
        scaled_coefficients[row] = remainder / upper[row][row]
    coefficients = [
        scaled_coefficients[column] / column_scales[column] for column in range(column_count)
    ]
    residuals = [
        value - math.fsum(coefficient * basis for coefficient, basis in zip(coefficients, row))
        for row, (_, _, value) in zip(rows, cells)
    ]
    rms = math.sqrt(math.fsum(item * item for item in residuals) / len(residuals))
    maximum_residual = max(abs(item) for item in residuals)
    h_finest = min(item[0] for item in cells)
    n_largest = max(item[1] for item in cells)
    interaction = coefficients[3] * h_finest**p * n_largest ** (-alpha)
    return {
        "model": "Q=Q_inf+a*h^p+b*N^-alpha+c*h^p*N^-alpha",
        "fit_algorithm": "column_scaled_reorthogonalized_MGS_QR_v2",
        "QR_diagonal_condition_proxy": _number_text(condition_proxy),
        "Q_inf": _number_text(coefficients[0]),
        "a": _number_text(coefficients[1]),
        "b": _number_text(coefficients[2]),
        "c": _number_text(coefficients[3]),
        "rms_residual": _number_text(rms),
        "maximum_absolute_residual": _number_text(maximum_residual),
        "terminal_interaction_signed": _number_text(interaction),
        "terminal_interaction_absolute": _number_text(abs(interaction)),
    }


def _observation_index(
    observations: Sequence[Mapping[str, Any]], endpoint_id: str
) -> dict[tuple[str, str, str, str], dict[int, tuple[float, int]]]:
    result: dict[tuple[str, str, str, str], dict[int, tuple[float, int]]] = {}
    for item in observations:
        key = (
            str(item["stage"]),
            str(item["method_id"]),
            str(item["grid_id"]),
            str(item["root_id"]),
        )
        seed = int(item["seed"])
        bucket = result.setdefault(key, {})
        if seed in bucket:
            raise CampaignV2Error("duplicate observation cell/seed identity")
        bucket[seed] = (
            float(Decimal(item["endpoint_values"][endpoint_id])),
            int(item["elapsed_ns"]),
        )
    return result


def _cell_values(
    index: Mapping[tuple[str, str, str, str], Mapping[int, tuple[float, int]]],
    stage: str,
    method_id: str,
    grid_id: str,
    root_id: str,
) -> dict[int, float]:
    key = (stage, method_id, grid_id, root_id)
    if key not in index:
        raise CampaignV2Error(f"missing observation cell {key}")
    return {seed: value[0] for seed, value in index[key].items()}


def _paired_differences(left: Mapping[int, float], right: Mapping[int, float]) -> list[float]:
    seeds = sorted(set(left).intersection(right))
    if len(seeds) < 2 or len(seeds) != len(left) or len(seeds) != len(right):
        raise CampaignV2Error("paired contrast lacks a complete common-seed block")
    return [left[seed] - right[seed] for seed in seeds]


def _crossed_stage_evidence(
    index: Mapping[tuple[str, str, str, str], Mapping[int, tuple[float, int]]],
    stage: str,
    method_id: str,
    grids: Sequence[Mapping[str, Any]],
    roots: Sequence[Mapping[str, Any]],
    grid_convergence: Mapping[str, Any],
    root_convergence: Mapping[str, Any],
    critical: float,
) -> tuple[dict[str, Any], float | None]:
    """Fit and bound one complete three-by-three crossed stage.

    The fitted interaction alone is not a model-adequacy bound.  Certification
    therefore includes the maximum absolute residual over every materialized
    cell, evaluated for every corner of the confidence-bounded order rectangle
    and every independent seed block.
    """

    if not grid_convergence["passed"] or not root_convergence["passed"]:
        return {
            "status": "unresolved",
            "stage": stage,
            "reason": "crossed_stage_axis_convergence_failed",
        }, None

    crossed_cells: list[tuple[float, float, float]] = []
    crossed_value_maps: list[tuple[Mapping[str, Any], Mapping[str, Any], dict[int, float]]] = []
    for grid in grids:
        for root in roots:
            values = _cell_values(index, stage, method_id, str(grid["id"]), str(root["id"]))
            crossed_cells.append(
                (
                    float(grid["h"]),
                    float(root["n_m"]),
                    _mean(list(values.values())),
                )
            )
            crossed_value_maps.append((grid, root, values))

    try:
        model = _crossed_model_fit(
            crossed_cells,
            float(grid_convergence["observed_order"]),
            float(root_convergence["observed_order"]),
        )
        grid_orders = sorted(
            {
                float(grid_convergence["confidence_bounded_order"]),
                float(grid_convergence["observed_order"]),
                float(grid_convergence["confidence_bounded_order_upper"]),
            }
        )
        root_orders = sorted(
            {
                float(root_convergence["confidence_bounded_order"]),
                float(root_convergence["observed_order"]),
                float(root_convergence["confidence_bounded_order_upper"]),
            }
        )
        seed_sets = [set(values) for _, _, values in crossed_value_maps]
        common_seeds = sorted(set.intersection(*seed_sets))
        if len(common_seeds) < 2 or any(seeds != set(common_seeds) for seeds in seed_sets):
            raise CampaignV2Error("crossed model lacks independent complete seed blocks")

        per_seed_interaction_envelopes: list[float] = []
        per_seed_lack_of_fit_envelopes: list[float] = []
        for seed in common_seeds:
            seed_cells = [
                (
                    float(grid["h"]),
                    float(root["n_m"]),
                    values[seed],
                )
                for grid, root, values in crossed_value_maps
            ]
            seed_fits = [
                _crossed_model_fit(seed_cells, p, alpha)
                for p in grid_orders
                for alpha in root_orders
            ]
            per_seed_interaction_envelopes.append(
                max(float(item["terminal_interaction_absolute"]) for item in seed_fits)
            )
            per_seed_lack_of_fit_envelopes.append(
                max(float(item["maximum_absolute_residual"]) for item in seed_fits)
            )

        interaction_summary = _paired_summary(per_seed_interaction_envelopes, critical)
        lack_of_fit_summary = _paired_summary(per_seed_lack_of_fit_envelopes, critical)
        interaction_term_upper = float(interaction_summary["simultaneous_upper_absolute"])
        lack_of_fit_upper = float(lack_of_fit_summary["simultaneous_upper_absolute"])
        combined_upper = interaction_term_upper + lack_of_fit_upper
        model.update(
            {
                "status": "resolved",
                "stage": stage,
                "order_uncertainty_envelope": {
                    "grid_orders": [_number_text(value) for value in grid_orders],
                    "root_orders": [_number_text(value) for value in root_orders],
                },
                "per_seed_terminal_interaction_envelopes": [
                    _number_text(value) for value in per_seed_interaction_envelopes
                ],
                "terminal_interaction_seed_block_summary": interaction_summary,
                "per_seed_lack_of_fit_envelopes": [
                    _number_text(value) for value in per_seed_lack_of_fit_envelopes
                ],
                "lack_of_fit_seed_block_summary": lack_of_fit_summary,
                "terminal_interaction_term_upper_bound": _number_text(interaction_term_upper),
                "lack_of_fit_upper_bound": _number_text(lack_of_fit_upper),
                "terminal_interaction_upper_confidence_bound": _number_text(combined_upper),
            }
        )
        return model, combined_upper
    except CampaignV2Error as exc:
        return {
            "status": "unresolved",
            "stage": stage,
            "reason": str(exc),
        }, None


def _endpoint_distribution_status(kind: str) -> dict[str, Any]:
    if kind == "continuous":
        return {
            "engine": "exact_known_variance_normal_TOST_power_with_paired_normal_bound_v2",
            "qualification_ready": True,
        }
    if kind == "exact_invariant":
        return {
            "engine": "zero_tolerance_exact_identity_v2",
            "qualification_ready": True,
        }
    engines = {
        "poisson_count": "exact_or_profile_poisson_equivalence_required",
        "poisson_rate": "exact_or_profile_poisson_rate_equivalence_required",
        "binomial_proportion": "exact_binomial_equivalence_required",
        "paired_bernoulli": "conditional_paired_binomial_equivalence_required",
        "weighted_compound_poisson": "compound_poisson_sumw_sumw2_ess_equivalence_required",
    }
    return {
        "engine": engines[kind],
        "qualification_ready": False,
        "blocker": "distribution_specific_power_engine_not_yet_integrated",
    }


def _global_uncertainty_ledger(
    endpoint: Mapping[str, Any],
    mesh_bound: float | None,
    root_bound: float | None,
    interaction_bound: float | None,
    sampling_bound: float | None,
) -> dict[str, Any]:
    computed = {
        "mesh_gci": (mesh_bound, "four_level_confirmed_GCI_v2"),
        "source_root": (root_bound, "four_level_confirmed_root_extrapolation_v2"),
        "grid_root_interaction": (interaction_bound, "crossed_h_power_N_power_fit_v2"),
        "outer_sampling": (sampling_bound, "familywise_paired_complete_seed_bound_v2"),
    }
    contract = endpoint["uncertainty_budget"]
    entries: list[dict[str, Any]] = []
    upper_sum = 0.0
    allocation_sum = 0.0
    all_resolved = True
    for component in REQUIRED_UNCERTAINTY_COMPONENTS:
        supplied = contract.get(component)
        if supplied is None:
            entries.append(
                {
                    "component": component,
                    "allocation": "0",
                    "estimate": None,
                    "upper_bound": None,
                    "method": None,
                    "evidence": None,
                    "status": "missing",
                }
            )
            all_resolved = False
            continue
        allocation = float(Decimal(supplied["allocation"]))
        allocation_sum += allocation
        if component in computed:
            bound, method = computed[component]
            status = "resolved" if bound is not None and math.isfinite(bound) else "unresolved"
            entries.append(
                {
                    "component": component,
                    "allocation": supplied["allocation"],
                    "estimate": None if bound is None else _number_text(bound),
                    "upper_bound": None if bound is None else _number_text(bound),
                    "method": method,
                    "evidence": "authenticated_pilot_and_finer_confirmation_state",
                    "status": status,
                }
            )
            if status == "resolved":
                upper_sum += float(bound)
                if bound > allocation:
                    all_resolved = False
            else:
                all_resolved = False
        else:
            upper = supplied.get("upper_bound")
            status = supplied.get("status")
            resolved = (
                status == "resolved"
                and upper is not None
                and supplied.get("method") is not None
                and supplied.get("evidence") is not None
            )
            entries.append(
                {
                    "component": component,
                    "allocation": supplied["allocation"],
                    "estimate": upper,
                    "upper_bound": upper,
                    "method": supplied.get("method"),
                    "evidence": supplied.get("evidence"),
                    "status": "resolved" if resolved else "unresolved",
                }
            )
            if resolved:
                upper_value = float(Decimal(upper))
                upper_sum += upper_value
                if upper_value > allocation:
                    all_resolved = False
            else:
                all_resolved = False
    target = float(Decimal(endpoint["scale_contract"]["absolute_target"]))
    if allocation_sum > target + max(1e-30, 1e-14 * target):
        all_resolved = False
    if upper_sum > target:
        all_resolved = False
    return {
        "endpoint_id": endpoint["id"],
        "target_absolute": endpoint["scale_contract"]["absolute_target"],
        "allocation_sum": _number_text(allocation_sum),
        "conservative_upper_sum": _number_text(upper_sum),
        "all_required_components_present_and_resolved": all(
            entry["status"] == "resolved" for entry in entries
        ),
        "every_component_within_allocation": all(
            entry["status"] == "resolved"
            and float(Decimal(entry["upper_bound"])) <= float(Decimal(entry["allocation"]))
            for entry in entries
        ),
        "allocation_sum_within_global_target": allocation_sum <= target,
        "upper_sum_within_global_target": upper_sum <= target,
        "passed": all_resolved,
        "components": entries,
    }


def _method_timing_summary(
    observations: Sequence[Mapping[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    by_identity: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for item in observations:
        identity = (
            str(item["stage"]),
            str(item["grid_id"]),
            str(item["root_id"]),
            str(item["seed"]),
        )
        by_identity.setdefault(identity, {})[str(item["method_id"])] = float(item["elapsed_ns"])
    differences = [
        methods[candidate] - methods[reference]
        for methods in by_identity.values()
        if candidate in methods and reference in methods
    ]
    if len(differences) < 2:
        raise CampaignV2Error("timing comparison lacks matched counterbalanced blocks")
    ordered = sorted(differences)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    # A distribution-free, deliberately conservative interval: the full
    # central 95% empirical order-statistic span.  It is timing-only and can
    # never rescue a physics failure.
    low_index = max(0, math.floor(0.025 * len(ordered)))
    high_index = min(len(ordered) - 1, math.ceil(0.975 * len(ordered)) - 1)
    low = ordered[low_index]
    high = ordered[high_index]
    status = "tie_or_unresolved"
    if high < 0:
        status = "candidate_faster"
    elif low > 0:
        status = "candidate_slower"
    return {
        "contract": "matched_counterbalanced_robust_order_statistic_interval_v2",
        "matched_blocks": str(len(differences)),
        "median_candidate_minus_reference_ns": _number_text(median),
        "interval_low_ns": _number_text(low),
        "interval_high_ns": _number_text(high),
        "status": status,
        "physics_gate_precedes_timing": True,
    }


def _compute_pilot_analysis_v2(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    state_value: Mapping[str, Any],
) -> dict[str, Any]:
    config, manifest, observations = _authenticated_pilot_observations_v2(
        request_value, manifest_value, state_value
    )
    state = validate_sealed_artifact(state_value, STATE_SCHEMA_V2, "pilot state")
    claim_count = max(1, len(config["claims"]))
    familywise_alpha = float(Decimal(config["statistics"]["familywise_alpha"]))
    tost_one_sided_alpha = familywise_alpha / claim_count
    method_equivalence_claim_count = sum(
        claim["kind"] == "method_equivalence" for claim in config["claims"]
    )
    # Each absolute Student-t interval consumes two tails.  The fixed count
    # covers, per method/endpoint, eight adjacent grid/root contrasts, four
    # unique disjoint-pool overlap anchors, the terminal mean, and separate
    # interaction plus L-infinity lack-of-fit bounds on both the tuning and
    # confirmation crossed surfaces; every method-equivalence claim adds four
    # high-end contrasts and four variance-envelope limits.  Bonferroni
    # allocation is intentionally conservative and is frozen by this schema.
    confidence_event_count = max(
        1,
        17 * len(config["methods"]) * len(config["endpoints"]) + 8 * method_equivalence_claim_count,
    )
    confidence_tail_alpha = familywise_alpha / (2.0 * confidence_event_count)
    critical = NormalDist().inv_cdf(1.0 - confidence_tail_alpha)
    desired_power = float(Decimal(config["statistics"]["desired_power"]))
    safety = float(Decimal(config["statistics"]["gci_safety_factor"]))
    p_min = float(Decimal(config["statistics"]["minimum_order"]))
    p_max = float(Decimal(config["statistics"]["maximum_order"]))
    order_tolerance = float(Decimal(config["statistics"]["confirmation_order_tolerance"]))
    asymptotic_min = float(Decimal(config["statistics"]["asymptotic_ratio_min"]))
    asymptotic_max = float(Decimal(config["statistics"]["asymptotic_ratio_max"]))
    grid_ratio = float(Decimal(config["grid_ratio"]))
    root_ratio = float(Decimal(config["root_ratio"]))
    reference = config["reference_method_id"]

    method_results: list[dict[str, Any]] = []
    maximum_required_outer = 2
    for method in config["methods"]:
        method_id = method["id"]
        endpoint_results: list[dict[str, Any]] = []
        method_passed = True
        for endpoint in config["endpoints"]:
            endpoint_id = endpoint["id"]
            index = _observation_index(observations, endpoint_id)
            pilot_grids = config["pilot_grids"][-3:]
            pilot_roots = config["pilot_roots"][-3:]
            confirmation_grids = config["all_grids"][-3:]
            confirmation_roots = config["all_roots"][-3:]
            high_pilot_root = pilot_roots[-1]["id"]
            high_pilot_grid = pilot_grids[-1]["id"]
            grid_tuning_cells = [
                _cell_values(
                    index,
                    "tuning_pilot",
                    method_id,
                    grid["id"],
                    high_pilot_root,
                )
                for grid in pilot_grids
            ]
            grid_confirmation_cells = [
                _cell_values(
                    index,
                    "finer_confirmation",
                    method_id,
                    grid["id"],
                    confirmation_roots[-2]["id"],
                )
                for grid in confirmation_grids
            ]
            grid_tuning = _stochastic_convergence_triplet(
                grid_tuning_cells,
                grid_ratio,
                safety,
                p_min,
                p_max,
                critical,
            )
            grid_confirmation = _stochastic_convergence_triplet(
                grid_confirmation_cells,
                grid_ratio,
                safety,
                p_min,
                p_max,
                critical,
            )
            grid_check = _confirmed_convergence(
                grid_tuning,
                grid_confirmation,
                grid_ratio,
                order_tolerance,
                asymptotic_min,
                asymptotic_max,
            )

            root_tuning_cells = [
                _cell_values(
                    index,
                    "tuning_pilot",
                    method_id,
                    high_pilot_grid,
                    root["id"],
                )
                for root in pilot_roots
            ]
            root_confirmation_cells = [
                _cell_values(
                    index,
                    "finer_confirmation",
                    method_id,
                    confirmation_grids[-2]["id"],
                    root["id"],
                )
                for root in confirmation_roots
            ]
            root_tuning = _stochastic_convergence_triplet(
                root_tuning_cells,
                root_ratio,
                safety,
                p_min,
                p_max,
                critical,
            )
            root_confirmation = _stochastic_convergence_triplet(
                root_confirmation_cells,
                root_ratio,
                safety,
                p_min,
                p_max,
                critical,
            )
            root_check = _confirmed_convergence(
                root_tuning,
                root_confirmation,
                root_ratio,
                order_tolerance,
                asymptotic_min,
                asymptotic_max,
            )

            outer_allocation = (
                float(Decimal(endpoint["uncertainty_budget"]["outer_sampling"]["allocation"]))
                if "outer_sampling" in endpoint["uncertainty_budget"]
                else 0.0
            )
            overlap_summaries: list[dict[str, Any]] = []
            for grid in pilot_grids[-2:]:
                for root in pilot_roots[-2:]:
                    overlap_summaries.append(
                        {
                            "grid_id": grid["id"],
                            "root_id": root["id"],
                            **_independent_cell_consistency_summary(
                                list(
                                    _cell_values(
                                        index,
                                        "tuning_pilot",
                                        method_id,
                                        grid["id"],
                                        root["id"],
                                    ).values()
                                ),
                                list(
                                    _cell_values(
                                        index,
                                        "finer_confirmation",
                                        method_id,
                                        grid["id"],
                                        root["id"],
                                    ).values()
                                ),
                                critical,
                            ),
                        }
                    )
            maximum_anchor = max(
                float(item["simultaneous_upper_absolute"]) for item in overlap_summaries
            )
            anchor_passed = maximum_anchor <= outer_allocation
            for label, check in (("grid", grid_check), ("root", root_check)):
                check["independent_pool_overlap_consistency"] = overlap_summaries
                check["maximum_overlap_consistency_upper_bound"] = _number_text(maximum_anchor)
                check["overlap_consistency_allocation"] = _number_text(outer_allocation)
                check["overlap_consistency_passed"] = anchor_passed
                if not anchor_passed:
                    check["passed"] = False
                    check["reasons"].append(
                        f"{label}_overlap_inconsistent_between_independent_seed_pools"
                    )

            tuning_model, tuning_interaction_bound = _crossed_stage_evidence(
                index,
                "tuning_pilot",
                method_id,
                pilot_grids,
                pilot_roots,
                grid_tuning,
                root_tuning,
                critical,
            )
            confirmation_model, confirmation_interaction_bound = _crossed_stage_evidence(
                index,
                "finer_confirmation",
                method_id,
                confirmation_grids,
                confirmation_roots,
                grid_confirmation,
                root_confirmation,
                critical,
            )
            interaction_bound = (
                None
                if tuning_interaction_bound is None or confirmation_interaction_bound is None
                else max(tuning_interaction_bound, confirmation_interaction_bound)
            )
            resolved_stage_models = [
                item
                for item in (tuning_model, confirmation_model)
                if item.get("status") == "resolved"
            ]
            model = {
                "status": ("resolved" if len(resolved_stage_models) == 2 else "unresolved"),
                "tuning_surface": tuning_model,
                "confirmation_surface": confirmation_model,
                "rms_residual": (
                    None
                    if len(resolved_stage_models) != 2
                    else _number_text(
                        max(float(item["rms_residual"]) for item in resolved_stage_models)
                    )
                ),
                "maximum_absolute_residual": (
                    None
                    if len(resolved_stage_models) != 2
                    else _number_text(
                        max(
                            float(item["maximum_absolute_residual"])
                            for item in resolved_stage_models
                        )
                    )
                ),
                "terminal_interaction_term_upper_bound": (
                    None
                    if len(resolved_stage_models) != 2
                    else _number_text(
                        max(
                            float(item["terminal_interaction_term_upper_bound"])
                            for item in resolved_stage_models
                        )
                    )
                ),
                "lack_of_fit_upper_bound": (
                    None
                    if len(resolved_stage_models) != 2
                    else _number_text(
                        max(
                            float(item["lack_of_fit_upper_bound"]) for item in resolved_stage_models
                        )
                    )
                ),
                "terminal_interaction_upper_confidence_bound": (
                    None if interaction_bound is None else _number_text(interaction_bound)
                ),
            }

            terminal_values = list(
                _cell_values(
                    index,
                    "finer_confirmation",
                    method_id,
                    config["confirmation_grid"]["id"],
                    config["confirmation_root"]["id"],
                ).values()
            )
            sampling = _paired_summary(terminal_values, critical)
            # The terminal variance is estimated, not known.  Use the
            # family-adjusted Student-t half-width; the normal critical is
            # retained in the receipt only as a diagnostic.
            sampling_bound = float(sampling["simultaneous_half_width"])
            mesh_bound = (
                None
                if not grid_check["passed"]
                else float(grid_confirmation["gci_finest_absolute"])
            )
            root_bound = (
                None
                if not root_check["passed"]
                else float(root_confirmation["gci_finest_absolute"])
            )
            ledger = _global_uncertainty_ledger(
                endpoint, mesh_bound, root_bound, interaction_bound, sampling_bound
            )
            distribution = _endpoint_distribution_status(endpoint["distribution_kind"])
            endpoint_passed = (
                grid_check["passed"]
                and root_check["passed"]
                and ledger["passed"]
                and distribution["qualification_ready"]
            )
            method_passed = method_passed and endpoint_passed
            endpoint_results.append(
                {
                    "endpoint_id": endpoint_id,
                    "distribution": distribution,
                    "grid_convergence": grid_check,
                    "root_convergence": root_check,
                    "crossed_model": model,
                    "terminal_sampling": sampling,
                    "uncertainty_ledger": ledger,
                    "diagnostic_gate_passed": endpoint_passed,
                }
            )
        timing = None
        if method_id != reference:
            timing = _method_timing_summary(observations, method_id, reference)
        method_results.append(
            {
                "method_id": method_id,
                "role": method["role"],
                "endpoint_results": endpoint_results,
                "diagnostic_physics_gate_passed": method_passed,
                "timing_vs_reference": timing,
            }
        )

    claim_results: list[dict[str, Any]] = []
    method_result_lookup = {item["method_id"]: item for item in method_results}
    for claim in config["claims"]:
        endpoint = next(item for item in config["endpoints"] if item["id"] == claim["endpoint_id"])
        distribution = _endpoint_distribution_status(endpoint["distribution_kind"])
        result: dict[str, Any] = {
            **claim,
            "multiplicity_family_claim_count": str(
                sum(1 for item in config["claims"] if item["family_id"] == claim["family_id"])
            ),
            "TOST_one_sided_alpha": _number_text(tost_one_sided_alpha),
            "distribution_engine": distribution["engine"],
        }
        if claim["kind"] == "method_equivalence" and distribution["qualification_ready"]:
            candidate = claim["candidate_method_id"]
            endpoint_index = _observation_index(observations, endpoint["id"])
            cell_results: list[dict[str, Any]] = []
            required_counts: list[int] = []
            margin = float(Decimal(endpoint["equivalence_margin_relative"])) * float(
                Decimal(endpoint["scale_contract"]["scale_basis"])
            )
            expected = float(
                Decimal(endpoint["expected_true_difference_relative"])
                * Decimal(endpoint["scale_contract"]["scale_basis"])
            )
            for grid, root in _stage_cells(config, "holdout_bracket"):
                left = _cell_values(
                    endpoint_index,
                    "finer_confirmation",
                    candidate,
                    grid["id"],
                    root["id"],
                )
                right = _cell_values(
                    endpoint_index,
                    "finer_confirmation",
                    reference,
                    grid["id"],
                    root["id"],
                )
                differences = _paired_differences(left, right)
                summary = _paired_summary(differences, critical)
                sd = _sample_sd(differences)
                sigma_upper = normal_sigma_upper_confidence_v2(
                    sd, len(differences), confidence_tail_alpha
                )
                bound = float(summary["simultaneous_upper_absolute"])
                planning_effect = max(abs(expected), bound)
                if planning_effect >= margin:
                    required = 10_000_001
                    achieved_power = 0.0
                    power_status = "unbounded_pilot_effect_reaches_equivalence_margin"
                else:
                    required = exact_normal_tost_required_n_v2(
                        sigma_upper,
                        margin,
                        planning_effect,
                        tost_one_sided_alpha,
                        desired_power,
                    )
                    achieved_power = normal_tost_power_v2(
                        required,
                        sigma_upper,
                        margin,
                        planning_effect,
                        tost_one_sided_alpha,
                    )
                    power_status = (
                        "diagnostic_pilot_effect_and_sigma_envelope_"
                        "not_candidate_blind_or_unknown_variance_exact"
                    )
                required_counts.append(required)
                cell_results.append(
                    {
                        "grid_id": grid["id"],
                        "root_id": root["id"],
                        "paired_summary": summary,
                        "equivalence_margin_absolute": _number_text(margin),
                        "equivalence_passed_diagnostic": bound <= margin,
                        "unknown_sigma_upper_confidence_bound": _number_text(sigma_upper),
                        "power_planning_absolute_effect": _number_text(planning_effect),
                        "power_planning_status": power_status,
                        "variance_envelope_one_sided_alpha": _number_text(confidence_tail_alpha),
                        "exact_normal_power_required_seed_blocks": str(required),
                        "achieved_power_at_required": _number_text(achieved_power),
                    }
                )
            required_outer = max(required_counts)
            maximum_required_outer = max(maximum_required_outer, required_outer)
            candidate_method = method_result_lookup[candidate]
            candidate_endpoint = next(
                item
                for item in candidate_method["endpoint_results"]
                if item["endpoint_id"] == endpoint["id"]
            )
            candidate_terminal_mean = float(candidate_endpoint["terminal_sampling"]["mean"])
            frozen_reference = float(Decimal(endpoint["scale_contract"]["reference_value"]))
            candidate_point_deviation = abs(candidate_terminal_mean - frozen_reference)
            candidate_ledger_upper = float(
                candidate_endpoint["uncertainty_ledger"]["conservative_upper_sum"]
            )
            candidate_absolute_upper = candidate_point_deviation + candidate_ledger_upper
            absolute_target = float(Decimal(endpoint["scale_contract"]["absolute_target"]))
            candidate_absolute_passed = (
                candidate_method["diagnostic_physics_gate_passed"]
                and candidate_absolute_upper <= absolute_target
            )
            equivalence_passed = all(item["equivalence_passed_diagnostic"] for item in cell_results)
            result.update(
                {
                    "status": (
                        "diagnostic_pass"
                        if equivalence_passed and candidate_absolute_passed
                        else "diagnostic_fail"
                    ),
                    "required_outer_seed_blocks": str(required_outer),
                    "four_cell_high_end_bracket": cell_results,
                    "all_bracket_cells_equivalent": equivalence_passed,
                    "candidate_terminal_mean": _number_text(candidate_terminal_mean),
                    "frozen_reference_value": _number_text(frozen_reference),
                    "candidate_absolute_point_deviation": _number_text(candidate_point_deviation),
                    "candidate_authenticated_uncertainty_upper_sum": _number_text(
                        candidate_ledger_upper
                    ),
                    "candidate_absolute_reference_upper_bound": _number_text(
                        candidate_absolute_upper
                    ),
                    "candidate_absolute_target": _number_text(absolute_target),
                    "candidate_absolute_gate_passed": candidate_absolute_passed,
                }
            )
        elif claim["kind"] == "exact_invariant":
            result.update(
                {
                    "status": "requires_zero_tolerance_runtime_identity_receipt",
                    "required_outer_seed_blocks": "0",
                }
            )
        elif claim["kind"] == "absolute_convergence":
            reference_method = method_result_lookup[reference]
            reference_endpoint = next(
                item
                for item in reference_method["endpoint_results"]
                if item["endpoint_id"] == endpoint["id"]
            )
            terminal_mean = float(reference_endpoint["terminal_sampling"]["mean"])
            frozen_reference = float(Decimal(endpoint["scale_contract"]["reference_value"]))
            point_deviation = abs(terminal_mean - frozen_reference)
            ledger_upper = float(reference_endpoint["uncertainty_ledger"]["conservative_upper_sum"])
            total_upper = point_deviation + ledger_upper
            target = float(Decimal(endpoint["scale_contract"]["absolute_target"]))
            method_gate = reference_method["diagnostic_physics_gate_passed"]
            passed = method_gate and total_upper <= target
            result.update(
                {
                    "status": "diagnostic_pass" if passed else "diagnostic_fail",
                    "required_outer_seed_blocks": "0",
                    "terminal_mean": _number_text(terminal_mean),
                    "frozen_reference_value": _number_text(frozen_reference),
                    "absolute_point_deviation": _number_text(point_deviation),
                    "authenticated_uncertainty_upper_sum": _number_text(ledger_upper),
                    "absolute_reference_deviation_upper_bound": _number_text(total_upper),
                    "absolute_target": _number_text(target),
                }
            )
        else:
            result.update(
                {
                    "status": "blocked_distribution_specific_power_engine_missing",
                    "required_outer_seed_blocks": None,
                }
            )
        # A declared claim is a qualification obligation, not a transient
        # loop-local diagnostic.  Retain every evaluated result so that the
        # registry-completeness and all-claims gates below cannot devolve into
        # the vacuous truth of all([]).
        claim_results.append(result)

    eligible_candidates = [
        result
        for result in method_results
        if result["role"] == "candidate"
        and result["diagnostic_physics_gate_passed"]
        and all(
            claim_result["status"] == "diagnostic_pass"
            for claim_result in claim_results
            if claim_result["kind"] == "method_equivalence"
            and claim_result["candidate_method_id"] == result["method_id"]
        )
    ]
    # Timing can only order already physics-passing candidates.  An interval
    # containing zero keeps the reference/tie; it does not manufacture speed.
    faster_candidates = [
        item
        for item in eligible_candidates
        if item["timing_vs_reference"] is not None
        and item["timing_vs_reference"]["status"] == "candidate_faster"
    ]
    selected = (
        sorted(faster_candidates, key=lambda item: item["method_id"])[0]["method_id"]
        if faster_candidates
        else reference
    )
    claim_registry_complete = not config["missing_claim_endpoints"] and bool(config["claims"])
    internal_diagnostic_gate = (
        claim_registry_complete
        and all(item["status"] == "diagnostic_pass" for item in claim_results)
        and method_result_lookup[selected]["diagnostic_physics_gate_passed"]
    )
    unsigned = {
        "schema": ANALYSIS_SCHEMA_V2,
        "schema_version": SCHEMA_VERSION_V2,
        "qualification_status": "hard_blocked_v2_foundation_only",
        "qualification_blocker": FOUNDATION_QUALIFICATION_BLOCKER,
        "physics_qualification_passed": False,
        "campaign_id": config["campaign_id"],
        "bindings": {
            "request_self_sha256": config["request_self_sha256"],
            "manifest_self_sha256": manifest[SELF_HASH_KEY],
            "state_self_sha256": state[SELF_HASH_KEY],
        },
        "design": {
            "pilot_grid_levels": [item["id"] for item in config["pilot_grids"]],
            "confirmation_grid_level": config["confirmation_grid"]["id"],
            "pilot_root_levels": [item["id"] for item in config["pilot_roots"]],
            "confirmation_root_level": config["confirmation_root"]["id"],
            "crossed_tuning_design": True,
            "separate_finer_confirmation": True,
            "fixed_physical_contract": config["fixed_physical_contract"],
        },
        "multiplicity": {
            "method": "bonferroni_v2",
            "registered_claim_count": str(len(config["claims"])),
            "familywise_alpha": config["statistics"]["familywise_alpha"],
            "TOST_one_sided_alpha_per_claim": _number_text(tost_one_sided_alpha),
            "confidence_event_count": str(confidence_event_count),
            "confidence_tail_alpha": _number_text(confidence_tail_alpha),
            "confidence_allocation": "bonferroni_two_tails_per_event_v2",
            "normal_critical": _number_text(critical),
        },
        "claim_registry": {
            "complete": claim_registry_complete,
            "missing_endpoint_claims": config["missing_claim_endpoints"],
            "claims": claim_results,
        },
        "method_results": method_results,
        "selected_method_id_diagnostic": selected,
        "internal_diagnostic_gate_passed": internal_diagnostic_gate,
        "maximum_exact_normal_power_required_outer_seed_blocks": str(maximum_required_outer),
        "independent_seed_blocks": manifest["independent_seed_blocks"],
        "materialized_feebl_processes": manifest["materialized_feebl_processes"],
        "notes": [
            "common seeds are used only for paired contrasts",
            "independent complete-run seed blocks are the uncertainty unit",
            "finer confirmation observations are disjoint from tuning observations",
            "timing never overrides a failed physics or uncertainty gate",
        ],
    }
    return seal_artifact(unsigned)


def compute_pilot_analysis_v2(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    state_value: Mapping[str, Any],
) -> dict[str, Any]:
    return _compute_pilot_analysis_v2(request_value, manifest_value, state_value)


def _blinded_floor_override(
    config: Mapping[str, Any], one_sided_alpha: float
) -> tuple[bool, int | None, dict[str, Any]]:
    blind = config["statistics"]["blinded_power_contract"]
    if blind is None:
        return False, None, {"status": "absent"}
    requested = int(blind["requested_seed_blocks"])
    sigma = float(Decimal(blind["sigma_upper_bound_relative"]))
    endpoint_powers = [
        normal_tost_power_v2(
            requested,
            sigma,
            float(Decimal(endpoint["equivalence_margin_relative"])),
            float(Decimal(endpoint["expected_true_difference_relative"])),
            one_sided_alpha,
        )
        for endpoint in config["endpoints"]
    ]
    achieved = min(endpoint_powers)
    self_declared_fields_consistent = (
        blind["status"] == "candidate_blind_verified"
        and blind["candidate_ids_blinded"] is True
        and blind["prospectively_frozen"] is True
        and achieved >= float(Decimal(config["statistics"]["desired_power"]))
    )
    # The current request contains only self-declared booleans and prose; it
    # does not carry a separately sealed pre-candidate observation set and a
    # pure recomputation certificate.  Therefore it can never lower the
    # accuracy-budget floor in v2.  Retain the arithmetic as a diagnostic,
    # but fail closed until that external certificate schema is implemented.
    valid = False
    return (
        valid,
        None,
        {
            **blind,
            "recomputed_family_adjusted_power": _number_text(achieved),
            "self_declared_fields_consistent": self_declared_fields_consistent,
            "floor_override_valid": valid,
            "blocker": (
                "separately_sealed_candidate_blind_power_certificate_"
                "and_pure_recomputation_not_implemented"
            ),
        },
    )


def _plan_realizations(
    config: Mapping[str, Any],
    methods: Sequence[str],
    seeds: Sequence[int],
    stage: str,
    cells: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if cells is None:
        cells = _stage_cells(config, "holdout_bracket")
    schedule = counterbalanced_execution_order_v2(
        methods,
        [(grid["id"], root["id"]) for grid, root in cells],
        seeds,
        stage,
    )
    by_grid = {item["id"]: item for item in config["all_grids"]}
    by_root = {item["id"]: item for item in config["all_roots"]}
    runs: list[dict[str, Any]] = []
    ordinal = 0
    for block in schedule:
        for position, method_id in enumerate(block["method_order"]):
            grid = by_grid[block["grid_id"]]
            root = by_root[block["root_id"]]
            runs.append(
                {
                    "ordinal": str(ordinal),
                    "stage": stage,
                    "timing_block_index": block["block_index"],
                    "timing_position": str(position),
                    "method_id": method_id,
                    "grid_id": grid["id"],
                    "root_id": root["id"],
                    "n_x": str(grid["n_x"]),
                    "n_y": str(grid["n_y"]),
                    "h": grid["h"],
                    "n_m": str(root["n_m"]),
                    "seed": block["seed"],
                    "seed_population": stage,
                }
            )
            ordinal += 1
    return schedule, runs


def _compute_next_stage_plan_v2(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    state_value: Mapping[str, Any],
) -> dict[str, Any]:
    config, manifest, _ = _authenticated_pilot_observations_v2(
        request_value, manifest_value, state_value
    )
    state = validate_sealed_artifact(state_value, STATE_SCHEMA_V2, "pilot state")
    analysis = _compute_pilot_analysis_v2(request_value, manifest_value, state_value)
    claim_count = max(1, len(config["claims"]))
    one_sided_alpha = float(Decimal(config["statistics"]["familywise_alpha"])) / claim_count
    blind_valid, blind_count, blind_evidence = _blinded_floor_override(config, one_sided_alpha)
    floor = int(config["statistics"]["minimum_heldout_seed_blocks"])
    power_required = int(analysis["maximum_exact_normal_power_required_outer_seed_blocks"])
    heldout_required = max(power_required, blind_count if blind_valid else floor)
    selected = analysis["selected_method_id_diagnostic"]
    reference = config["reference_method_id"]
    # The sequestered bracket must be able to re-evaluate every frozen claim,
    # including a candidate that did not win a diagnostic timing ranking.
    # Materialize all registered methods; pruning claims after seeing the
    # pilot would invalidate familywise error control.
    methods = [method["id"] for method in config["methods"]]

    actions: list[dict[str, Any]] = []
    grid_failed = any(
        not endpoint["grid_convergence"]["passed"]
        for method in analysis["method_results"]
        for endpoint in method["endpoint_results"]
    )
    root_failed = any(
        not endpoint["root_convergence"]["passed"]
        for method in analysis["method_results"]
        for endpoint in method["endpoint_results"]
    )
    if grid_failed:
        actions.append(
            {
                "action": ACTION_MESH_REFINEMENT,
                "axis": "deterministic_grid_spacing",
                "status": "diagnostic_only_requires_new_preregistered_finer_level",
                "independent_seed_blocks_added": "0",
                "may_change_n_m": False,
            }
        )
    if root_failed:
        actions.append(
            {
                "action": ACTION_WITHIN_RUN_ROOT_SUPPORT,
                "axis": "within_run_root_support",
                "status": "diagnostic_only_requires_new_nested_root_prefix_level",
                "independent_seed_blocks_added": "0",
                "outer_seeds_reused": True,
                "may_substitute_for_outer_precision": False,
            }
        )

    # Method-equivalence power is evaluated at each of the four high-end
    # bracket cells, which are populated by the disjoint confirmation pool.
    # Tuning seeds only cover one of those four cells and cannot be counted as
    # if they repeated the whole claim contrast.
    existing_claim_blocks_per_high_end_cell = len(config["seed_pools"]["confirmation"])
    additional_outer = max(0, power_required - existing_claim_blocks_per_high_end_cell)
    extension_available = len(config["seed_pools"]["extension"])
    outer_exhausted = additional_outer > extension_available
    outer_schedule: list[dict[str, Any]] = []
    outer_runs: list[dict[str, Any]] = []
    if additional_outer:
        if not outer_exhausted:
            # Precision extension is a new independent complete-run seed block
            # over the full four-by-four crossed lattice.  It may not be
            # replaced by a larger n_m job under an already counted seed.
            full_cells = [
                (grid, root) for grid in config["all_grids"] for root in config["all_roots"]
            ]
            outer_schedule, outer_runs = _plan_realizations(
                config,
                [method["id"] for method in config["methods"]],
                config["seed_pools"]["extension"][:additional_outer],
                "outer_replicate_precision",
                full_cells,
            )
        actions.append(
            {
                "action": ACTION_OUTER_REPLICATE_PRECISION,
                "axis": "independent_outer_seed_blocks",
                "status": (
                    "blocked_preregistered_extension_seed_pool_exhausted"
                    if outer_exhausted
                    else "diagnostic_plan_materialized"
                ),
                "required_additional_seed_blocks": str(additional_outer),
                "available_preregistered_extension_seed_blocks": str(extension_available),
                "independent_seed_blocks_added": (
                    "0" if outer_exhausted else str(additional_outer)
                ),
                "may_change_n_m": False,
                "larger_n_m_same_seed_fallback": "forbidden",
                "full_crossed_grid_root_method_lattice": True,
                "materialized_feebl_processes": str(len(outer_runs)),
            }
        )

    holdout_available = len(config["seed_pools"]["holdout"])
    holdout_exhausted = heldout_required > holdout_available
    holdout_schedule: list[dict[str, Any]] = []
    holdout_runs: list[dict[str, Any]] = []
    if (
        analysis["internal_diagnostic_gate_passed"]
        and additional_outer == 0
        and not holdout_exhausted
    ):
        holdout_seeds = config["seed_pools"]["holdout"][:heldout_required]
        holdout_schedule, holdout_runs = _plan_realizations(
            config, methods, holdout_seeds, "reserved_holdout"
        )
        actions.append(
            {
                "action": ACTION_HOLDOUT_BRACKET,
                "axis": "sequestered_confirmation",
                "status": "planned_but_execution_and_qualification_hard_blocked",
                "high_end_grid_levels": [item["id"] for item in config["all_grids"][-2:]],
                "high_end_root_levels": [item["id"] for item in config["all_roots"][-2:]],
                "independent_seed_blocks_added": str(heldout_required),
                "four_cell_bracket_per_method_and_seed": True,
            }
        )
    else:
        reasons: list[str] = []
        if not analysis["internal_diagnostic_gate_passed"]:
            reasons.append("pilot_internal_diagnostic_gate_failed")
        if additional_outer and not outer_exhausted:
            reasons.append("outer_precision_extension_requires_execution_and_reanalysis")
        if outer_exhausted:
            reasons.append("preregistered_extension_seed_pool_exhausted")
        if holdout_exhausted:
            reasons.append("reserved_holdout_seed_pool_exhausted")
        actions.append(
            {
                "action": ACTION_DIAGNOSTIC_FAILURE,
                "axis": "none",
                "status": "fail_closed",
                "reasons": reasons,
                "larger_n_m_same_seed_fallback": "forbidden",
            }
        )

    unsigned = {
        "schema": NEXT_STAGE_SCHEMA_V2,
        "schema_version": SCHEMA_VERSION_V2,
        "qualification_status": "hard_blocked_v2_foundation_only",
        "qualification_blocker": FOUNDATION_QUALIFICATION_BLOCKER,
        "execution_authorized": False,
        "physics_qualification_authorized": False,
        "campaign_id": config["campaign_id"],
        "bindings": {
            "request_self_sha256": config["request_self_sha256"],
            "manifest_self_sha256": manifest[SELF_HASH_KEY],
            "state_self_sha256": state[SELF_HASH_KEY],
            "analysis_self_sha256": analysis[SELF_HASH_KEY],
        },
        "selected_method_id_diagnostic": selected,
        "reference_method_id": reference,
        "heldout_seed_floor": str(floor),
        "blinded_power_floor_override": blind_evidence,
        "exact_normal_power_required_seed_blocks": str(power_required),
        "planned_heldout_independent_seed_blocks": str(heldout_required),
        "existing_claim_blocks_per_high_end_cell": str(existing_claim_blocks_per_high_end_cell),
        "available_reserved_holdout_seed_blocks": str(holdout_available),
        "holdout_seed_pool_exhausted": holdout_exhausted,
        "extension_seed_pool_exhausted": outer_exhausted,
        "actions": actions,
        "holdout_contract": {
            "cell_lattice": "{g3,g4}x{N3,N4}_v2",
            "terminal_only_confirmation_forbidden": True,
            "selection_may_not_use_holdout": True,
            "alternative_promotion_after_holdout": "forbidden",
        },
        "counterbalance_contract": "cyclic_latin_with_alternating_reversal_v2",
        "outer_replicate_schedule": outer_schedule,
        "outer_replicate_realizations": outer_runs,
        "holdout_schedule": holdout_schedule,
        "holdout_realizations": holdout_runs,
        "independent_seed_blocks": str(
            additional_outer if outer_runs else (heldout_required if holdout_runs else 0)
        ),
        "materialized_feebl_processes": str(len(outer_runs) + len(holdout_runs)),
    }
    return seal_artifact(unsigned)


def compute_next_stage_plan_v2(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    state_value: Mapping[str, Any],
) -> dict[str, Any]:
    return _compute_next_stage_plan_v2(request_value, manifest_value, state_value)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _emit_pilot_artifacts_v2(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    state_value: Mapping[str, Any],
    output_directory: Path,
) -> tuple[Path, Path]:
    analysis = _compute_pilot_analysis_v2(request_value, manifest_value, state_value)
    plan = _compute_next_stage_plan_v2(request_value, manifest_value, state_value)
    analysis_path = output_directory / "pilot_analysis_v2.json"
    plan_path = output_directory / "next_stage_plan_v2.json"
    _atomic_write_bytes(analysis_path, canonical_bytes(analysis))
    _atomic_write_bytes(plan_path, canonical_bytes(plan))
    return analysis_path, plan_path


def emit_pilot_artifacts_v2(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    state_value: Mapping[str, Any],
    output_directory: Path,
) -> tuple[Path, Path]:
    return _emit_pilot_artifacts_v2(request_value, manifest_value, state_value, output_directory)


def _validate_pilot_artifacts_v2(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    state_value: Mapping[str, Any],
    analysis_path: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_analysis = _compute_pilot_analysis_v2(request_value, manifest_value, state_value)
    expected_plan = _compute_next_stage_plan_v2(request_value, manifest_value, state_value)
    expected_analysis_bytes = canonical_bytes(expected_analysis)
    expected_plan_bytes = canonical_bytes(expected_plan)
    try:
        observed_analysis_bytes = analysis_path.read_bytes()
        observed_plan_bytes = plan_path.read_bytes()
    except OSError as exc:
        raise CampaignV2Error(f"cannot read v2 analysis artifacts: {exc}") from exc
    if observed_analysis_bytes != expected_analysis_bytes:
        raise CampaignV2Error(
            "pilot_analysis_v2.json is not canonically byte-identical to pure recomputation"
        )
    if observed_plan_bytes != expected_plan_bytes:
        raise CampaignV2Error(
            "next_stage_plan_v2.json is not canonically byte-identical to pure recomputation"
        )
    # Parse only after byte equality, so a re-signed forged JSON document has
    # no privileged path through validation.
    return (
        validate_sealed_artifact(
            json.loads(observed_analysis_bytes), ANALYSIS_SCHEMA_V2, "pilot analysis"
        ),
        validate_sealed_artifact(
            json.loads(observed_plan_bytes), NEXT_STAGE_SCHEMA_V2, "next-stage plan"
        ),
    )


def validate_pilot_artifacts_v2(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    state_value: Mapping[str, Any],
    analysis_path: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _validate_pilot_artifacts_v2(
        request_value,
        manifest_value,
        state_value,
        analysis_path,
        plan_path,
    )


def make_complete_state_v2_for_testing(
    request_value: Mapping[str, Any],
    manifest_value: Mapping[str, Any],
    value_function: Callable[[Mapping[str, Any], str], Decimal | str | int],
    *,
    elapsed_function: Callable[[Mapping[str, Any]], int] | None = None,
) -> dict[str, Any]:
    """Deterministic fixture helper; never an execution-evidence actuator."""

    config = validate_request_v2(request_value)
    manifest = validate_manifest_v2(request_value, manifest_value)
    completed: list[dict[str, Any]] = []
    for realization in manifest["realizations"]:
        values = {
            endpoint["id"]: _number_text(
                _decimal(
                    str(value_function(realization, endpoint["id"])),
                    f"fixture.{endpoint['id']}",
                )
            )
            for endpoint in config["endpoints"]
        }
        elapsed = (
            elapsed_function(realization)
            if elapsed_function is not None
            else 1_000_000 + int(realization["ordinal"])
        )
        evidence = sha256_object(
            {
                "realization_id": realization["realization_id"],
                "endpoint_values": values,
                "elapsed_ns": str(elapsed),
                "fixture_only": True,
            }
        )
        completed.append(
            {
                "realization_id": realization["realization_id"],
                "endpoint_values": values,
                "elapsed_ns": str(elapsed),
                "execution_evidence_sha256": evidence,
            }
        )
    return seal_artifact(
        {
            "schema": STATE_SCHEMA_V2,
            "schema_version": SCHEMA_VERSION_V2,
            "qualification_status": "synthetic_fixture_not_physics_qualified",
            "qualification_blocker": FOUNDATION_QUALIFICATION_BLOCKER,
            "status": "complete",
            "campaign_id": config["campaign_id"],
            "request_self_sha256": config["request_self_sha256"],
            "manifest_self_sha256": manifest[SELF_HASH_KEY],
            "completed": completed,
        }
    )


__all__ = [
    "ACTION_DIAGNOSTIC_FAILURE",
    "ACTION_HOLDOUT_BRACKET",
    "ACTION_MESH_REFINEMENT",
    "ACTION_OUTER_REPLICATE_PRECISION",
    "ACTION_WITHIN_RUN_ROOT_SUPPORT",
    "ANALYSIS_SCHEMA_V2",
    "CampaignV2Error",
    "FOUNDATION_QUALIFICATION_BLOCKER",
    "MANIFEST_SCHEMA_V2",
    "NEXT_STAGE_SCHEMA_V2",
    "REQUEST_SCHEMA_V2",
    "REQUIRED_UNCERTAINTY_COMPONENTS",
    "STATE_SCHEMA_V2",
    "SUPPORTED_ENDPOINT_KINDS",
    "authenticated_pilot_observations_v2",
    "canonical_bytes",
    "compute_next_stage_plan_v2",
    "compute_pilot_analysis_v2",
    "counterbalanced_execution_order_v2",
    "emit_pilot_artifacts_v2",
    "exact_normal_tost_required_n_v2",
    "make_complete_state_v2_for_testing",
    "materialize_pilot_manifest_v2",
    "normal_tost_power_v2",
    "seal_artifact",
    "sha256_object",
    "validate_manifest_v2",
    "validate_pilot_artifacts_v2",
    "validate_request_v2",
    "validate_sealed_artifact",
]
