#!/usr/bin/env python3
"""Pure fail-closed qualification-evidence contracts for FeeBL V3.

The functions in this module validate evidence only.  They do not launch
FeeBL, read or write files, materialize cards/seeds, authorize production, or
claim that any V3 physics capability is integrated.  G2/G4/G7 code can reuse
the same contracts when it binds real authenticated artifacts.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


OFF_CENTER_SELF_FORCE_SCHEMA_V3 = "feebl.adaptive-convergence-v3.off-center-self-force-evidence.v1"
UNCERTAINTY_LEDGER_SCHEMA_V3 = "feebl.adaptive-convergence-v3.uncertainty-ledger.v1"
RUNTIME_RECEIPT_SCHEMA_V3 = "feebl.adaptive-convergence-v3.runtime-receipt-evidence.v1"
SPECTRUM_EVIDENCE_SCHEMA_V3 = "feebl.adaptive-convergence-v3.spectrum-evidence.v1"
OFF_CENTER_FORCE_CRITERION_ID_V3 = (
    "euclidean_force_vector_defect_le_abs_plus_rel_times_" "authenticated_reference_norm.v1"
)
OFF_CENTER_FORCE_REFERENCE_SCALE_DEFINITION_V3 = (
    "euclidean_norm_of_authenticated_oracle_reference_force"
)

REQUIRED_UNCERTAINTY_COMPONENTS_V3 = (
    "source_deposition",
    "poisson_solve",
    "potential_to_field",
    "particle_gather",
    "coarse_fine_interface",
    "open_boundary_domain",
    "collision_quadrature",
    "longitudinal_time_integration",
    "independent_root_sampling",
    "split_merge_representation",
    "backend_arithmetic",
    "spectrum_binning_fitting",
    "reference_uncertainty",
)


class QualificationContractError(RuntimeError):
    """An evidence object cannot be used for V3 qualification."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationContractError(message)


def _reject_floats(value: Any, where: str) -> None:
    if isinstance(value, float):
        raise QualificationContractError(f"{where}: binary floating-point values are forbidden")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(isinstance(key, str), f"{where}: object key is not text")
            _reject_floats(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_floats(item, f"{where}[{index}]")


def canonical_bytes_v3(value: Any) -> bytes:
    """Return portable canonical JSON bytes for an exact evidence payload."""
    _reject_floats(value, "$payload")
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise QualificationContractError(
            f"$payload: value is not canonical-JSON encodable: {error}"
        ) from error


def canonical_sha256_v3(value: Any) -> str:
    return hashlib.sha256(canonical_bytes_v3(value)).hexdigest()


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{where}: expected an object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    _require(
        isinstance(value, (list, tuple)),
        f"{where}: expected an ordered sequence",
    )
    return value


def _text(value: Any, where: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{where}: expected text")
    return value


def _sha256(value: Any, where: str) -> str:
    result = _text(value, where)
    _require(
        re.fullmatch(r"[0-9a-f]{64}", result) is not None,
        f"{where}: expected canonical lowercase SHA-256",
    )
    return result


def _integer(value: Any, where: str, *, positive: bool = False) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{where}: expected an integer",
    )
    _require(value > 0 if positive else value >= 0, f"{where}: out of range")
    return value


def _decimal(value: Any, where: str, *, positive: bool = False) -> Decimal:
    _require(isinstance(value, str) and bool(value), f"{where}: expected decimal text")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise QualificationContractError(f"{where}: invalid decimal text") from error
    _require(result.is_finite(), f"{where}: non-finite decimal")
    _require(result > 0 if positive else result >= 0, f"{where}: out of range")
    return result


def _decimal_text(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    with localcontext() as context:
        context.prec = max(80, len(value.as_tuple().digits) + 20)
        return str(value.normalize(context=context))


def _force_vector(value: Any, where: str) -> tuple[Decimal, Decimal]:
    items = _sequence(value, where)
    _require(len(items) == 2, f"{where}: expected exactly two components")
    return (
        _signed_decimal(items[0], f"{where}[0]"),
        _signed_decimal(items[1], f"{where}[1]"),
    )


def _signed_decimal(value: Any, where: str) -> Decimal:
    _require(isinstance(value, str) and bool(value), f"{where}: expected decimal text")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise QualificationContractError(f"{where}: invalid decimal text") from error
    _require(result.is_finite(), f"{where}: non-finite decimal")
    return result


def validate_bound_token_v3(
    kind: str, payload: Mapping[str, Any], declared_sha256: str
) -> dict[str, str]:
    """Structurally recompute a topology/source/oracle payload binding."""
    label = _text(kind, "$token.kind")
    bound_payload = _mapping(payload, f"${label}.payload")
    declared = _sha256(declared_sha256, f"${label}.sha256")
    recomputed = canonical_sha256_v3(bound_payload)
    _require(
        recomputed == declared,
        f"stale {label} token does not bind current payload",
    )
    return {
        "kind": label,
        "sha256": recomputed,
        "status": "structural_payload_binding_pass_not_qualification",
    }


def validate_off_center_self_force_evidence_v3(
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    """Validate an independent vector-magnitude comparison and mixed bound.

    The reference scale is derived here from the authenticated oracle vector;
    no caller-supplied scale can inflate the bound.  A second immutable token
    binds the exact criterion, oracle identity, and absolute/relative tolerance
    texts that participate in the comparison.
    """
    item = _mapping(evidence, "$self_force")
    _require(
        item.get("schema") == OFF_CENTER_SELF_FORCE_SCHEMA_V3,
        "$self_force.schema: unsupported schema",
    )
    oracle_kind = item.get("oracle_kind")
    _require(
        oracle_kind in {"uniform_fine", "analytic_magnitude"},
        "off-center self-force lacks an independent magnitude oracle",
    )
    oracle_payload = _mapping(item.get("oracle_payload"), "$self_force.oracle_payload")
    oracle_sha256 = _sha256(item.get("oracle_sha256"), "$self_force.oracle_sha256")
    validate_bound_token_v3(
        "off-center self-force oracle",
        oracle_payload,
        oracle_sha256,
    )
    _require(
        oracle_payload.get("method") == oracle_kind,
        "off-center self-force oracle method is not bound",
    )
    reference_raw = _sequence(item.get("reference_force"), "$self_force.reference_force")
    _require(
        oracle_payload.get("force") == reference_raw,
        "off-center self-force reference vector is not oracle-bound",
    )
    reference = _force_vector(reference_raw, "$self_force.reference_force")
    candidate = _force_vector(item.get("candidate_force"), "$self_force.candidate_force")
    criterion_payload = _mapping(item.get("criterion_payload"), "$self_force.criterion_payload")
    _require(
        criterion_payload.get("criterion_id") == OFF_CENTER_FORCE_CRITERION_ID_V3,
        "off-center self-force criterion identity is unsupported",
    )
    _require(
        criterion_payload.get("reference_scale_definition")
        == OFF_CENTER_FORCE_REFERENCE_SCALE_DEFINITION_V3,
        "off-center self-force reference-scale definition is unsupported",
    )
    _require(
        criterion_payload.get("oracle_sha256") == oracle_sha256,
        "off-center self-force criterion is not bound to the oracle",
    )
    _require(
        criterion_payload.get("absolute_tolerance") == item.get("absolute_tolerance")
        and criterion_payload.get("relative_tolerance") == item.get("relative_tolerance"),
        "off-center self-force criterion does not bind exact tolerances",
    )
    criterion_sha256 = _sha256(item.get("criterion_sha256"), "$self_force.criterion_sha256")
    validate_bound_token_v3(
        "off-center self-force criterion",
        criterion_payload,
        criterion_sha256,
    )
    absolute_tolerance = _decimal(item.get("absolute_tolerance"), "$self_force.absolute_tolerance")
    relative_tolerance = _decimal(item.get("relative_tolerance"), "$self_force.relative_tolerance")
    with localcontext() as context:
        context.prec = 80
        reference_scale = (reference[0] ** 2 + reference[1] ** 2).sqrt()
        _require(
            reference_scale > 0,
            "off-center self-force authenticated reference norm is zero",
        )
        defect = ((candidate[0] - reference[0]) ** 2 + (candidate[1] - reference[1]) ** 2).sqrt()
        allowed = absolute_tolerance + relative_tolerance * reference_scale
    _require(
        defect <= allowed,
        "off-center self-force exceeds independent magnitude expectation",
    )
    return {
        "status": "structural_contract_pass_not_qualification",
        "defect": _decimal_text(defect),
        "allowed": _decimal_text(allowed),
        "derived_reference_scale": _decimal_text(reference_scale),
        "oracle_sha256": oracle_sha256,
        "criterion_sha256": criterion_sha256,
    }


def validate_runtime_receipt_evidence_v3(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Check the structural fields claimed by a runtime receipt.

    This function does not open any named artifact or reconstruct endpoints;
    production authentication remains a later actuator/integration obligation.
    """
    item = _mapping(evidence, "$runtime_receipt")
    _require(
        item.get("schema") == RUNTIME_RECEIPT_SCHEMA_V3,
        "$runtime_receipt.schema: unsupported schema",
    )
    hashes = {}
    for field in (
        "feebl_binary_sha256",
        "card_sha256",
        "raw_output_sha256",
        "spectrum_lattice_sha256",
        "reconstructed_endpoints_sha256",
    ):
        hashes[field] = _sha256(item.get(field), f"$runtime_receipt.{field}")
    _require(
        item.get("raw_endpoint_reconstruction_complete") is True,
        "runtime receipt did not reconstruct endpoints from raw artifacts",
    )
    count = _integer(
        item.get("authenticated_record_count"),
        "$runtime_receipt.authenticated_record_count",
        positive=True,
    )
    return {
        "status": "structural_contract_pass_not_qualification",
        "record_count": count,
        **hashes,
    }


def validate_uncertainty_ledger_evidence_v3(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Check full-ledger shape and raw-recomputed claim fields only.

    It does not read raw artifacts or repeat the declared methods at G0.
    """
    item = _mapping(evidence, "$uncertainty_ledger")
    _require(
        item.get("schema") == UNCERTAINTY_LEDGER_SCHEMA_V3,
        "$uncertainty_ledger.schema: unsupported schema",
    )
    components = _sequence(item.get("components"), "$uncertainty_ledger.components")
    observed: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(components):
        component = _mapping(raw, f"$uncertainty_ledger.components[{index}]")
        name = _text(component.get("component"), f"$component[{index}].component")
        _require(
            name in REQUIRED_UNCERTAINTY_COMPONENTS_V3,
            f"uncertainty ledger has unknown component: {name}",
        )
        _require(name not in observed, f"uncertainty ledger duplicates component: {name}")
        observed[name] = component
    missing = [name for name in REQUIRED_UNCERTAINTY_COMPONENTS_V3 if name not in observed]
    _require(not missing, "uncertainty ledger missing components: " + ",".join(missing))

    upper_sum = Decimal(0)
    for name in REQUIRED_UNCERTAINTY_COMPONENTS_V3:
        component = observed[name]
        _require(
            component.get("derivation") == "recomputed_from_authenticated_raw_artifact",
            f"uncertainty component {name} is externally asserted rather than recomputed",
        )
        _sha256(component.get("raw_artifact_sha256"), f"$component[{name}].raw_artifact_sha256")
        _text(component.get("method_id"), f"$component[{name}].method_id")
        upper_sum += _decimal(component.get("upper_bound"), f"$component[{name}].upper_bound")
    return {
        "status": "structural_contract_pass_not_qualification",
        "component_count": len(observed),
        "conservative_upper_sum": _decimal_text(upper_sum),
    }


def validate_spectrum_evidence_v3(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Check supplied count, lattice, and per-bin arithmetic structurally.

    It does not read the card or FeeBL output named by the supplied digest.
    """
    item = _mapping(evidence, "$spectrum")
    _require(
        item.get("schema") == SPECTRUM_EVIDENCE_SCHEMA_V3,
        "$spectrum.schema: unsupported schema",
    )
    _sha256(item.get("card_sha256"), "$spectrum.card_sha256")
    records = _sequence(item.get("records"), "$spectrum.records")
    declared_count = _integer(item.get("record_count"), "$spectrum.record_count")
    _require(
        declared_count == len(records),
        "spectrum record count mismatch",
    )
    _require(bool(records), "spectrum has no records")
    absolute_tolerance = _decimal(
        item.get("weight_absolute_tolerance"),
        "$spectrum.weight_absolute_tolerance",
    )
    relative_tolerance = _decimal(
        item.get("weight_relative_tolerance"),
        "$spectrum.weight_relative_tolerance",
    )
    previous_high: Decimal | None = None
    total_weight = Decimal(0)
    for index, raw in enumerate(records):
        record = _mapping(raw, f"$spectrum.records[{index}]")
        low = _decimal(record.get("low"), f"$record[{index}].low")
        high = _decimal(record.get("high"), f"$record[{index}].high")
        density = _decimal(record.get("density"), f"$record[{index}].density")
        weight = _decimal(record.get("weight"), f"$record[{index}].weight")
        _require(low < high, f"spectrum record {index} has nonpositive width")
        _require(
            previous_high is None or low == previous_high,
            "spectrum lattice is nonmonotone or gapped",
        )
        expected_weight = density * (high - low)
        allowed = absolute_tolerance + relative_tolerance * abs(expected_weight)
        _require(
            abs(weight - expected_weight) <= allowed,
            "spectrum weight closure failed",
        )
        total_weight += weight
        previous_high = high
    return {
        "status": "structural_contract_pass_not_qualification",
        "record_count": declared_count,
        "total_weight": _decimal_text(total_weight),
    }


__all__ = [
    "OFF_CENTER_FORCE_CRITERION_ID_V3",
    "OFF_CENTER_FORCE_REFERENCE_SCALE_DEFINITION_V3",
    "OFF_CENTER_SELF_FORCE_SCHEMA_V3",
    "QualificationContractError",
    "REQUIRED_UNCERTAINTY_COMPONENTS_V3",
    "RUNTIME_RECEIPT_SCHEMA_V3",
    "SPECTRUM_EVIDENCE_SCHEMA_V3",
    "UNCERTAINTY_LEDGER_SCHEMA_V3",
    "canonical_bytes_v3",
    "canonical_sha256_v3",
    "validate_bound_token_v3",
    "validate_off_center_self_force_evidence_v3",
    "validate_runtime_receipt_evidence_v3",
    "validate_spectrum_evidence_v3",
    "validate_uncertainty_ledger_evidence_v3",
]
