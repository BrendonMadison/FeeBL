#!/usr/bin/env python3
"""Fail-closed audit of the frozen V2 boundaries inherited at V3 G0.

This test deliberately proves an *unqualified* starting state.  It launches
only the standalone deterministic physics validator, reads source/build
metadata, and never creates a campaign manifest, seed, card, or run directory.
The separate strict CTest is the positive successor gate; it is intentionally
expected to fail until the numerical Stage-4 work is repaired.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping


SCHEMA = "feebl.adaptive-convergence-v3.g0-baseline-blocker-audit.v1"
RELATIVE_TOLERANCE = 5.0e-12
ABSOLUTE_TOLERANCE = 5.0e-14

# Frozen diagnostic values retain the known composite-solver limitations.
# Smooth-order and accepted/rejected interface phases each have two entries.
EXPECTED_METRICS: dict[tuple[str, str], float] = {
    ("smooth_16_to_32", "field_observed_order"): 4.380466888432065,
    ("smooth_32_to_64", "field_observed_order"): 1.2100501780976942,
    ("source_interface_sweep", "worst_potential_relative_l2"):
        0.020933700414106114,
    ("source_interface_sweep", "worst_field_relative_l2"):
        0.0085553146910598305,
    ("source_interface_sweep", "rejected_crossing_phases"): 9.0,
    ("source_interface_sweep", "accepted_complete_shape_phases"): 4.0,
    ("round_gaussian", "field_direct_relative_l2"):
        0.051418612375184948,
    ("round_gaussian", "uniform_field_direct_relative_l2"):
        0.051418910569164804,
}

EXPECTED_FAILED_GATES = {
    "smooth_order_plausible": 0,
    "interface_one_percent": 0,
    "gaussian_one_percent": 0,
    "standalone_stage4_candidate": 0,
}

EXPECTED_FINDING_IDS = {
    "P1-production-composite-not-linked-or-called",
    "P1-standalone-stage4-fails-closed",
    "P1-off-center-interface-self-force-has-no-independent-magnitude-oracle",
    "P1-analyzer-does-not-reconstruct-runtime-receipts",
    "P1-ledger-accepts-external-component-bounds",
    "P2-order-stability-compares-point-estimates",
    "P2-power-uses-known-variance-normal-model",
    "P2-timing-span-is-descriptive",
    "P2-deposit-and-gather-repeat-canonical-reconstruction",
}

METRIC_RE = re.compile(
    r"^METRIC case=(?P<case>\S+) name=(?P<name>\S+) "
    r"value=(?P<value>\S+)$"
)
GATE_RE = re.compile(r"^GATE name=(?P<name>\S+) pass=(?P<pass>[01])$")


class AuditFailure(RuntimeError):
    """The baseline evidence or a fail-closed boundary did not match."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(source_root: Path, relative: str) -> str:
    path = source_root / relative
    require(path.is_file(), f"required source evidence is missing: {relative}")
    return path.read_text(encoding="utf-8")


def require_fragments(text: str, relative: str, fragments: tuple[str, ...]) -> None:
    for fragment in fragments:
        require(
            fragment in text,
            f"{relative}: missing G0 boundary evidence {fragment!r}",
        )


def run_validator(validator: Path, strict: bool) -> subprocess.CompletedProcess[str]:
    command = [str(validator)]
    if strict:
        command.append("--require-stage4")
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired as error:
        raise AuditFailure(f"Stage-4 validator timed out: {command}") from error


def parse_metrics(output: str) -> dict[tuple[str, str], float]:
    metrics: dict[tuple[str, str], float] = {}
    for line in output.splitlines():
        match = METRIC_RE.fullmatch(line)
        if match is None:
            continue
        key = (match.group("case"), match.group("name"))
        require(key not in metrics, f"duplicate Stage-4 metric: {key}")
        try:
            value = float(match.group("value"))
        except ValueError as error:
            raise AuditFailure(f"non-numeric Stage-4 metric: {key}") from error
        require(math.isfinite(value), f"non-finite Stage-4 metric: {key}")
        metrics[key] = value
    return metrics


def parse_gates(output: str) -> dict[str, int]:
    gates: dict[str, int] = {}
    for line in output.splitlines():
        match = GATE_RE.fullmatch(line)
        if match is None:
            continue
        name = match.group("name")
        require(name not in gates, f"duplicate Stage-4 gate: {name}")
        gates[name] = int(match.group("pass"))
    return gates


def compare_stage4_output(label: str, output: str) -> dict[str, Any]:
    metrics = parse_metrics(output)
    observed: dict[str, str] = {}
    for key, expected in EXPECTED_METRICS.items():
        require(key in metrics, f"{label}: missing Stage-4 metric {key}")
        actual = metrics[key]
        if key in {
            ("source_interface_sweep", "rejected_crossing_phases"),
            ("source_interface_sweep", "accepted_complete_shape_phases"),
        }:
            require(actual == expected, f"{label}: nonintegral phase-count drift")
        else:
            require(
                math.isclose(
                    actual,
                    expected,
                    rel_tol=RELATIVE_TOLERANCE,
                    abs_tol=ABSOLUTE_TOLERANCE,
                ),
                f"{label}: metric {key} drifted: got {actual}, expected {expected}",
            )
        observed[f"{key[0]}::{key[1]}"] = format(actual, ".17g")

    phase_indices = {
        int(match.group(1))
        for case, _name in metrics
        if (match := re.fullmatch(r"source_interface_phase([0-9]+)", case))
    }
    require(
        phase_indices == set(range(13)),
        f"{label}: source/interface phase inventory is not exactly 0..12",
    )

    gates = parse_gates(output)
    for name, expected in EXPECTED_FAILED_GATES.items():
        require(gates.get(name) == expected, f"{label}: gate {name} is not {expected}")
    require(
        output.count("physics_qualification_claim=0") == 1,
        f"{label}: standalone result did not explicitly deny production qualification",
    )
    require(
        output.count("master_stage4_pass=0") == 1,
        f"{label}: standalone result did not explicitly deny master qualification",
    )
    return {"metrics": observed, "gates": gates}


def audit_production_link(feebl: Path, link_receipt: Path) -> dict[str, Any]:
    require(feebl.is_file(), f"FeeBL executable is missing: {feebl}")
    require(link_receipt.is_file(), f"FeeBL link receipt is missing: {link_receipt}")
    direct_links = [
        item.strip()
        for item in link_receipt.read_text(encoding="utf-8").splitlines()
        if item.strip()
    ]
    forbidden_targets = (
        "feebl_adaptive_composite_poisson",
        "feebl_adaptive_composite_coupling",
    )
    for target in forbidden_targets:
        require(
            all(target not in item for item in direct_links),
            f"G0 baseline unexpectedly links production FeeBL to {target}",
        )

    nm = shutil.which("nm")
    require(nm is not None, "nm is required for the G0 production-symbol audit")
    symbol_scan = subprocess.run(
        [nm, "-C", "--defined-only", str(feebl)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    require(symbol_scan.returncode == 0, f"nm failed: {symbol_scan.stderr.strip()}")
    forbidden_symbols = (
        "CompositePoissonR2",
        "build_composite_tent_stencils_r2",
        "deposit_composite_tent_source_once_r2",
        "gather_composite_tent_adjoint_r2",
    )
    present = [name for name in forbidden_symbols if name in symbol_scan.stdout]
    require(not present, f"G0 production FeeBL contains composite symbols: {present}")
    return {
        "direct_link_libraries": direct_links,
        "composite_symbols_present": present,
        "feebl_sha256": sha256_file(feebl),
    }


def load_python_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    require(
        specification is not None and specification.loader is not None,
        f"cannot load qualification module: {path}",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def observe_expected_rejection(
    identifier: str,
    expected_reason: str,
    error_type: type[Exception],
    probe: Callable[[], Any],
) -> str:
    try:
        probe()
    except error_type as error:
        reason = str(error)
        require(
            expected_reason in reason,
            f"{identifier}: unexpected rejection reason: {reason}",
        )
        return reason
    except Exception as error:
        raise AuditFailure(
            f"{identifier}: adversarial probe raised the wrong exception: "
            f"{type(error).__name__}: {error}"
        ) from error
    raise AuditFailure(f"{identifier}: adversarial evidence was accepted")


def execute_mutation(
    identifier: str,
    *,
    injection: str,
    action: str,
    expected_reason: str,
    error_type: type[Exception],
    probe: Callable[[], Any],
) -> dict[str, str]:
    return {
        "injection": injection,
        "action": action,
        "observed_rejection": observe_expected_rejection(
            identifier, expected_reason, error_type, probe
        ),
    }


def qualification_boundary_execution(
    identifier: str,
    *,
    positive_control: Mapping[str, Any],
    mutations: list[dict[str, str]],
) -> dict[str, Any]:
    require(bool(mutations), f"{identifier}: no negative mutation was executed")
    return {
        "id": identifier,
        "status": "qualification_boundary_negative_executed",
        "negative_test_executed": True,
        "positive_control": dict(positive_control),
        "injection": " | ".join(item["injection"] for item in mutations),
        "action": " | ".join(item["action"] for item in mutations),
        "observed_rejection": " | ".join(
            item["observed_rejection"] for item in mutations
        ),
        "adversarial_mutations": mutations,
        "qualification_effect": "fail_closed",
        "production_capability_status": "open",
    }


def execute_negative_tests(
    source_root: Path, stage4_metrics: Mapping[str, str]
) -> list[dict[str, Any]]:
    contracts = load_python_module(
        source_root
        / "examples" / "adaptive_grid" / "qualification_contracts.py",
        "feebl_v3_qualification_contracts",
    )
    root_pool = load_python_module(
        source_root
        / "examples" / "adaptive_particles" / "physical_root_pool_v3.py",
        "feebl_v3_physical_root_pool",
    )
    executions: list[dict[str, Any]] = []

    rejected_phases = int(
        stage4_metrics["source_interface_sweep::rejected_crossing_phases"]
    )
    accepted_phases = int(
        stage4_metrics["source_interface_sweep::accepted_complete_shape_phases"]
    )
    require(
        (rejected_phases, accepted_phases) == (9, 4),
        "cross-interface support negative did not execute",
    )
    executions.append(
        qualification_boundary_execution(
            "cross_interface_particle_support",
            positive_control={
                "source_interface_phases_exercised": 13,
                "accepted_complete_shape_phases": accepted_phases,
                "status": "retained_stage4_observation_not_qualification",
            },
            mutations=[
                {
                    "injection": (
                        "sweep all 13 retained source/interface phases, including "
                        "nine incomplete-support crossings"
                    ),
                    "action": (
                        "execute the standalone Stage-4 source-interface support "
                        "probe without dropping or renormalizing a source"
                    ),
                    "observed_rejection": (
                        f"{rejected_phases} of 13 phases rejected while "
                        f"{accepted_phases} complete-shape phases were accepted"
                    ),
                }
            ],
        )
    )

    oracle_payload = {
        "method": "uniform_fine",
        "force": ["0.3", "0.4"],
        "units": "normalized_force",
    }
    oracle_sha256 = contracts.canonical_sha256_v3(oracle_payload)
    criterion_payload = {
        "criterion_id": contracts.OFF_CENTER_FORCE_CRITERION_ID_V3,
        "reference_scale_definition": (
            contracts.OFF_CENTER_FORCE_REFERENCE_SCALE_DEFINITION_V3
        ),
        "oracle_sha256": oracle_sha256,
        "absolute_tolerance": "1e-12",
        "relative_tolerance": "0.01",
    }
    force_evidence = {
        "schema": contracts.OFF_CENTER_SELF_FORCE_SCHEMA_V3,
        "oracle_kind": "uniform_fine",
        "oracle_payload": oracle_payload,
        "oracle_sha256": oracle_sha256,
        "reference_force": ["0.3", "0.4"],
        "candidate_force": ["0.3", "0.4"],
        "criterion_payload": criterion_payload,
        "criterion_sha256": contracts.canonical_sha256_v3(criterion_payload),
        "absolute_tolerance": "1e-12",
        "relative_tolerance": "0.01",
    }
    force_positive = contracts.validate_off_center_self_force_evidence_v3(
        force_evidence
    )
    require(
        force_positive["status"] == "structural_contract_pass_not_qualification"
        and force_positive["derived_reference_scale"] == "0.5",
        "off-center self-force positive control did not derive the oracle norm",
    )
    force_mutations: list[dict[str, str]] = []

    identity_only = deepcopy(force_evidence)
    identity_only["oracle_kind"] = "energy_gradient_identity"
    force_mutations.append(
        execute_mutation(
            "off_center_self_force_independent_magnitude",
            injection=(
                "replace the independent uniform-fine oracle kind with the "
                "existing energy-gradient identity"
            ),
            action=(
                "validate pinned oracle kind before accepting the force-vector "
                "magnitude comparison"
            ),
            expected_reason="lacks an independent magnitude oracle",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_off_center_self_force_evidence_v3(
                identity_only
            ),
        )
    )

    stale_oracle = deepcopy(force_evidence)
    stale_oracle["oracle_payload"]["force"][0] = "0.31"
    force_mutations.append(
        execute_mutation(
            "off_center_self_force_independent_magnitude",
            injection="change the oracle force payload after freezing its SHA-256",
            action="recompute and compare the exact canonical oracle payload token",
            expected_reason="stale off-center self-force oracle token",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_off_center_self_force_evidence_v3(
                stale_oracle
            ),
        )
    )

    excessive_force = deepcopy(force_evidence)
    excessive_force["candidate_force"][0] = "0.306"
    force_mutations.append(
        execute_mutation(
            "off_center_self_force_independent_magnitude",
            injection=(
                "apply a 0.006 force-vector defect against a derived 0.5 "
                "reference norm and 0.005000000001 mixed bound"
            ),
            action=(
                "derive the scale from the authenticated reference vector and "
                "evaluate the Euclidean vector defect"
            ),
            expected_reason="exceeds independent magnitude expectation",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_off_center_self_force_evidence_v3(
                excessive_force
            ),
        )
    )

    stale_criterion = deepcopy(force_evidence)
    stale_criterion["relative_tolerance"] = "0.02"
    stale_criterion["criterion_payload"]["relative_tolerance"] = "0.02"
    force_mutations.append(
        execute_mutation(
            "off_center_self_force_independent_magnitude",
            injection=(
                "change the relative tolerance and criterion payload after "
                "freezing the criterion SHA-256"
            ),
            action="recompute the exact criterion/tolerance token",
            expected_reason="stale off-center self-force criterion token",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_off_center_self_force_evidence_v3(
                stale_criterion
            ),
        )
    )

    negative_tolerance = deepcopy(force_evidence)
    negative_tolerance["absolute_tolerance"] = "-1e-12"
    negative_tolerance["criterion_payload"]["absolute_tolerance"] = "-1e-12"
    negative_tolerance["criterion_sha256"] = contracts.canonical_sha256_v3(
        negative_tolerance["criterion_payload"]
    )
    force_mutations.append(
        execute_mutation(
            "off_center_self_force_independent_magnitude",
            injection=(
                "supply an authenticated criterion carrying a negative absolute "
                "tolerance"
            ),
            action="parse the exact tolerance and reject an out-of-range value",
            expected_reason="$self_force.absolute_tolerance: out of range",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_off_center_self_force_evidence_v3(
                negative_tolerance
            ),
        )
    )
    executions.append(
        qualification_boundary_execution(
            "off_center_self_force_independent_magnitude",
            positive_control=force_positive,
            mutations=force_mutations,
        )
    )

    runtime_receipt = {
        "schema": contracts.RUNTIME_RECEIPT_SCHEMA_V3,
        "feebl_binary_sha256": "1" * 64,
        "card_sha256": "2" * 64,
        "raw_output_sha256": "3" * 64,
        "spectrum_lattice_sha256": "4" * 64,
        "reconstructed_endpoints_sha256": "5" * 64,
        "raw_endpoint_reconstruction_complete": True,
        "authenticated_record_count": 10,
    }
    runtime_positive = contracts.validate_runtime_receipt_evidence_v3(
        runtime_receipt
    )
    require(
        runtime_positive["status"] == "structural_contract_pass_not_qualification",
        "runtime receipt positive control overclaimed qualification",
    )
    runtime_mutations: list[dict[str, str]] = []
    missing_raw_output = deepcopy(runtime_receipt)
    missing_raw_output.pop("raw_output_sha256")
    runtime_mutations.append(
        execute_mutation(
            "missing_runtime_identity_receipt",
            injection="remove raw_output_sha256 from an otherwise complete receipt",
            action=(
                "validate executable/card/raw-output/lattice/endpoint hashes "
                "before any evidence can be considered"
            ),
            expected_reason="raw_output_sha256",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_runtime_receipt_evidence_v3(
                missing_raw_output
            ),
        )
    )
    incomplete_reconstruction = deepcopy(runtime_receipt)
    incomplete_reconstruction["raw_endpoint_reconstruction_complete"] = False
    runtime_mutations.append(
        execute_mutation(
            "missing_runtime_identity_receipt",
            injection="mark raw endpoint reconstruction incomplete",
            action="require explicit complete raw-artifact reconstruction",
            expected_reason="did not reconstruct endpoints",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_runtime_receipt_evidence_v3(
                incomplete_reconstruction
            ),
        )
    )
    zero_records = deepcopy(runtime_receipt)
    zero_records["authenticated_record_count"] = 0
    runtime_mutations.append(
        execute_mutation(
            "missing_runtime_identity_receipt",
            injection="set the authenticated record count to zero",
            action="require a positive exact record count",
            expected_reason="authenticated_record_count: out of range",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_runtime_receipt_evidence_v3(
                zero_records
            ),
        )
    )
    executions.append(
        qualification_boundary_execution(
            "missing_runtime_identity_receipt",
            positive_control=runtime_positive,
            mutations=runtime_mutations,
        )
    )

    uncertainty_components: list[dict[str, str]] = []
    for component in contracts.REQUIRED_UNCERTAINTY_COMPONENTS_V3:
        raw_payload = {
            "component": component,
            "fixture": "g0_structural_positive_control",
        }
        uncertainty_components.append(
            {
                "component": component,
                "derivation": "recomputed_from_authenticated_raw_artifact",
                "raw_artifact_sha256": contracts.canonical_sha256_v3(raw_payload),
                "method_id": "g0_structural_probe_v1",
                "upper_bound": "0",
            }
        )
    uncertainty_ledger = {
        "schema": contracts.UNCERTAINTY_LEDGER_SCHEMA_V3,
        "components": uncertainty_components,
    }
    uncertainty_positive = contracts.validate_uncertainty_ledger_evidence_v3(
        uncertainty_ledger
    )
    require(
        uncertainty_positive["status"]
        == "structural_contract_pass_not_qualification"
        and uncertainty_positive["component_count"]
        == len(contracts.REQUIRED_UNCERTAINTY_COMPONENTS_V3),
        "uncertainty ledger positive control did not cover the full inventory",
    )
    uncertainty_mutations: list[dict[str, str]] = []
    missing_component = deepcopy(uncertainty_ledger)
    missing_component["components"] = [
        component
        for component in missing_component["components"]
        if component["component"] != "reference_uncertainty"
    ]
    uncertainty_mutations.append(
        execute_mutation(
            "externally_asserted_uncertainty_terms",
            injection="remove reference_uncertainty from the complete component set",
            action="compare the exact component inventory before summing bounds",
            expected_reason="missing components: reference_uncertainty",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_uncertainty_ledger_evidence_v3(
                missing_component
            ),
        )
    )
    asserted_component = deepcopy(uncertainty_ledger)
    asserted_component["components"][0]["derivation"] = "externally_asserted_text"
    uncertainty_mutations.append(
        execute_mutation(
            "externally_asserted_uncertainty_terms",
            injection=(
                "replace one raw-recomputed derivation with an externally "
                "asserted text bound"
            ),
            action="require the raw-recomputed-only derivation marker",
            expected_reason="externally asserted rather than recomputed",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_uncertainty_ledger_evidence_v3(
                asserted_component
            ),
        )
    )
    missing_raw_binding = deepcopy(uncertainty_ledger)
    missing_raw_binding["components"][1].pop("raw_artifact_sha256")
    uncertainty_mutations.append(
        execute_mutation(
            "externally_asserted_uncertainty_terms",
            injection="remove the bound raw-artifact SHA-256 from one component",
            action="require a canonical raw-artifact identity for every component",
            expected_reason="raw_artifact_sha256",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_uncertainty_ledger_evidence_v3(
                missing_raw_binding
            ),
        )
    )
    executions.append(
        qualification_boundary_execution(
            "externally_asserted_uncertainty_terms",
            positive_control=uncertainty_positive,
            mutations=uncertainty_mutations,
        )
    )

    topology_payload = {
        "coarse_shape": [16, 16],
        "patch": [4, 12, 4, 12],
        "ratio": 2,
    }
    topology_token = contracts.canonical_sha256_v3(topology_payload)
    topology_positive = contracts.validate_bound_token_v3(
        "topology", topology_payload, topology_token
    )
    changed_topology = deepcopy(topology_payload)
    changed_topology["patch"] = [5, 13, 4, 12]
    executions.append(
        qualification_boundary_execution(
            "stale_topology_token",
            positive_control={
                "status": "structural_contract_pass_not_qualification",
                "binding": topology_positive,
            },
            mutations=[
                execute_mutation(
                    "stale_topology_token",
                    injection="translate the patch after freezing the topology token",
                    action="recompute the canonical topology payload SHA-256",
                    expected_reason="stale topology token",
                    error_type=contracts.QualificationContractError,
                    probe=lambda: contracts.validate_bound_token_v3(
                        "topology", changed_topology, topology_token
                    ),
                )
            ],
        )
    )

    source_payload = {
        "particles": [
            {
                "stable_id": 8001,
                "charge": "1",
                "position": ["0.1", "-0.2"],
            }
        ],
        "topology_sha256": topology_token,
    }
    source_token = contracts.canonical_sha256_v3(source_payload)
    source_positive = contracts.validate_bound_token_v3(
        "source", source_payload, source_token
    )
    changed_source = deepcopy(source_payload)
    changed_source["particles"][0]["charge"] = "1.01"
    executions.append(
        qualification_boundary_execution(
            "stale_source_token",
            positive_control={
                "status": "structural_contract_pass_not_qualification",
                "binding": source_positive,
            },
            mutations=[
                execute_mutation(
                    "stale_source_token",
                    injection="change signed source charge after freezing its token",
                    action="recompute the canonical source payload SHA-256",
                    expected_reason="stale source token",
                    error_type=contracts.QualificationContractError,
                    probe=lambda: contracts.validate_bound_token_v3(
                        "source", changed_source, source_token
                    ),
                )
            ],
        )
    )

    root_positive = root_pool.extend_physical_root_prefix_v3(
        (101, 103), (107, 109), 3
    )
    require(
        root_positive["prior_prefix"] == [101, 103]
        and root_positive["added_root_ids"] == [107]
        and root_positive["resulting_prefix"] == [101, 103, 107]
        and root_positive["prefix_stable"] is True
        and root_positive["production_actuator_integrated"] is False
        and root_positive[
            "file_seed_card_or_run_materialization_performed"
        ]
        is False,
        "physical-root positive control violated stable no-materialization planning",
    )
    root_mutations: list[dict[str, str]] = []
    root_mutations.append(
        execute_mutation(
            "physical_root_pool_exhaustion",
            injection=(
                "request prefix length five from a two-root prefix with only two "
                "unused physical roots"
            ),
            action="compare exact requested extension against unused-pool capacity",
            expected_reason="physical-root extension pool exhausted",
            error_type=root_pool.PhysicalRootPoolError,
            probe=lambda: root_pool.extend_physical_root_prefix_v3(
                (101, 103), (107, 109), 5
            ),
        )
    )
    root_mutations.append(
        execute_mutation(
            "physical_root_pool_exhaustion",
            injection="place frozen root 103 at the head of the extension pool",
            action="reject an extension pool that reuses the frozen prefix",
            expected_reason="reuses or reorders the frozen prefix",
            error_type=root_pool.PhysicalRootPoolError,
            probe=lambda: root_pool.extend_physical_root_prefix_v3(
                (101, 103), (103, 107), 3
            ),
        )
    )
    root_mutations.append(
        execute_mutation(
            "physical_root_pool_exhaustion",
            injection="reverse two otherwise unused extension root IDs",
            action="require stable strictly increasing physical-root order",
            expected_reason="strictly increasing and unique",
            error_type=root_pool.PhysicalRootPoolError,
            probe=lambda: root_pool.extend_physical_root_prefix_v3(
                (101, 103), (109, 107), 3
            ),
        )
    )
    root_mutations.append(
        execute_mutation(
            "physical_root_pool_exhaustion",
            injection="set reuse_existing_roots=true on an extension request",
            action="reject explicit reuse of already counted physical roots",
            expected_reason="physical-root reuse is forbidden",
            error_type=root_pool.PhysicalRootPoolError,
            probe=lambda: root_pool.extend_physical_root_prefix_v3(
                (101, 103), (107, 109), 3, reuse_existing_roots=True
            ),
        )
    )
    root_mutations.append(
        execute_mutation(
            "physical_root_pool_exhaustion",
            injection=(
                "offer representative child count 4096 as a substitute for a "
                "new physical root"
            ),
            action="reject split/merge representation as independent-root capacity",
            expected_reason="representation-count substitution",
            error_type=root_pool.PhysicalRootPoolError,
            probe=lambda: root_pool.extend_physical_root_prefix_v3(
                (101, 103),
                (107, 109),
                3,
                representation_count_substitute=4096,
            ),
        )
    )
    executions.append(
        qualification_boundary_execution(
            "physical_root_pool_exhaustion",
            positive_control=root_positive,
            mutations=root_mutations,
        )
    )

    spectrum_evidence = {
        "schema": contracts.SPECTRUM_EVIDENCE_SCHEMA_V3,
        "card_sha256": "6" * 64,
        "record_count": 2,
        "weight_absolute_tolerance": "1e-15",
        "weight_relative_tolerance": "1e-12",
        "records": [
            {"low": "0", "high": "1", "density": "2", "weight": "2"},
            {"low": "1", "high": "2", "density": "3", "weight": "3"},
        ],
    }
    spectrum_positive = contracts.validate_spectrum_evidence_v3(
        spectrum_evidence
    )
    require(
        spectrum_positive["status"]
        == "structural_contract_pass_not_qualification"
        and spectrum_positive["record_count"] == 2,
        "spectrum positive control overclaimed qualification",
    )
    spectrum_mutations: list[dict[str, str]] = []
    bad_count = deepcopy(spectrum_evidence)
    bad_count["record_count"] = 3
    spectrum_mutations.append(
        execute_mutation(
            "spectrum_authentication_failure",
            injection="declare three records while supplying exactly two",
            action="compare the exact declared and parsed record counts",
            expected_reason="spectrum record count mismatch",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_spectrum_evidence_v3(bad_count),
        )
    )
    gapped_lattice = deepcopy(spectrum_evidence)
    gapped_lattice["records"][1]["low"] = "1.1"
    spectrum_mutations.append(
        execute_mutation(
            "spectrum_authentication_failure",
            injection="move the second bin low edge from 1 to 1.1",
            action="require exact contiguous monotone bin boundaries",
            expected_reason="spectrum lattice is nonmonotone or gapped",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_spectrum_evidence_v3(gapped_lattice),
        )
    )
    broken_closure = deepcopy(spectrum_evidence)
    broken_closure["records"][1]["weight"] = "3.1"
    spectrum_mutations.append(
        execute_mutation(
            "spectrum_authentication_failure",
            injection="change the second bin weight from 3 to 3.1",
            action="recompute density times bin width under the exact mixed tolerance",
            expected_reason="spectrum weight closure failed",
            error_type=contracts.QualificationContractError,
            probe=lambda: contracts.validate_spectrum_evidence_v3(broken_closure),
        )
    )
    executions.append(
        qualification_boundary_execution(
            "spectrum_authentication_failure",
            positive_control=spectrum_positive,
            mutations=spectrum_mutations,
        )
    )

    expected_ids = {
        "cross_interface_particle_support",
        "off_center_self_force_independent_magnitude",
        "missing_runtime_identity_receipt",
        "externally_asserted_uncertainty_terms",
        "stale_topology_token",
        "stale_source_token",
        "physical_root_pool_exhaustion",
        "spectrum_authentication_failure",
    }
    require(
        {item["id"] for item in executions} == expected_ids,
        "executable negative-test inventory is incomplete",
    )
    require(
        all(
            item["negative_test_executed"] is True
            and item["status"] == "qualification_boundary_negative_executed"
            and item["injection"]
            and item["action"]
            and item["observed_rejection"]
            for item in executions
        ),
        "an executable negative test lacks fail-closed execution evidence",
    )
    return executions


def audit_review_findings(
    source_root: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    cmake = read_source(source_root, "src/CMakeLists.txt")
    validator_source = read_source(
        source_root,
        "examples/adaptive_grid/test_adaptive_composite_physics_validation.cc",
    )
    coupling_test = read_source(
        source_root, "examples/adaptive_grid/test_adaptive_composite_coupling.cc"
    )
    coupling_source = read_source(source_root, "src/adaptiveCompositeCouplingCPP.cc")
    campaign = read_source(
        source_root,
        "examples/adaptive_particles/coupled_grid_root_campaign_v2.py",
    )
    campaign_tests = read_source(
        source_root,
        "examples/adaptive_particles/test_coupled_grid_root_campaign_v2.py",
    )
    spectrum_tests = read_source(
        source_root, "examples/adaptive_particles/test_gaussian_core_spectrum.py"
    )

    findings: list[dict[str, str]] = []

    def open_finding(identifier: str, evidence: str) -> None:
        findings.append({"id": identifier, "state": "open", "evidence": evidence})

    require_fragments(
        cmake,
        "src/CMakeLists.txt",
        (
            "add_library(feebl_adaptive_composite_poisson STATIC",
            "add_library(feebl_adaptive_composite_coupling STATIC",
            "target_link_libraries(feebl feebl_gpu_cic feebl_gpu_resident",
        ),
    )
    open_finding(
        "P1-production-composite-not-linked-or-called",
        "generated direct-link receipt plus defined-symbol scan",
    )

    require_fragments(
        validator_source,
        "examples/adaptive_grid/test_adaptive_composite_physics_validation.cc",
        (
            'std::string(argv[1]) == "--require-stage4"',
            "if (require_stage4 && !gate.pass()) return 2;",
            'std::cout << "physics_qualification_claim=0\\n";',
            'std::cout << "master_stage4_pass=0\\n";',
        ),
    )
    open_finding(
        "P1-standalone-stage4-fails-closed",
        "diagnostic exit 0, strict exit 2, fixed metrics and negative flags",
    )

    require_fragments(
        coupling_test,
        "examples/adaptive_grid/test_adaptive_composite_coupling.cc",
        (
            '"self-force equals energy-gradient diagnostic"',
            "A centered source in this exactly reflection-symmetric domain",
            '"reflection-symmetric source has non-negligible self-force"',
        ),
    )
    open_finding(
        "P1-off-center-interface-self-force-has-no-independent-magnitude-oracle",
        "off-center identity plus centered-only analytic magnitude oracle",
    )

    require_fragments(
        campaign,
        "examples/adaptive_particles/coupled_grid_root_campaign_v2.py",
        (
            'record.get("execution_evidence_sha256")',
            '"execution_evidence_sha256": evidence_hash',
        ),
    )
    require("import subprocess" not in campaign, "V2 analyzer unexpectedly gained an actuator")
    require("subprocess.run" not in campaign, "V2 analyzer unexpectedly launches FeeBL")
    open_finding(
        "P1-analyzer-does-not-reconstruct-runtime-receipts",
        "analyzer accepts supplied execution digest and has no process actuator",
    )

    require_fragments(
        campaign,
        "examples/adaptive_particles/coupled_grid_root_campaign_v2.py",
        (
            'upper_raw = entry.get("upper_bound")',
            'upper = supplied.get("upper_bound")',
            'supplied.get("method") is not None',
            'supplied.get("evidence") is not None',
        ),
    )
    open_finding(
        "P1-ledger-accepts-external-component-bounds",
        "noncomputed ledger entries accept supplied bound/method/evidence fields",
    )

    require_fragments(
        campaign,
        "examples/adaptive_particles/coupled_grid_root_campaign_v2.py",
        (
            'p_tuning = float(tuning["observed_order"])',
            'p_confirmation = float(confirmation["observed_order"])',
            "order_delta = abs(p_tuning - p_confirmation)",
        ),
    )
    open_finding(
        "P2-order-stability-compares-point-estimates",
        "point observed-order delta is compared with a fixed tolerance",
    )

    require_fragments(
        campaign,
        "examples/adaptive_particles/coupled_grid_root_campaign_v2.py",
        (
            "def normal_tost_power_v2(",
            '"engine": "exact_known_variance_normal_TOST_power_with_paired_normal_bound_v2"',
            "achieved_power = normal_tost_power_v2(",
        ),
    )
    open_finding(
        "P2-power-uses-known-variance-normal-model",
        "normal TOST power follows an upper variance envelope",
    )

    require_fragments(
        campaign,
        "examples/adaptive_particles/coupled_grid_root_campaign_v2.py",
        (
            "central 95% empirical order-statistic span",
            '"contract": "matched_counterbalanced_robust_order_statistic_interval_v2"',
        ),
    )
    open_finding(
        "P2-timing-span-is-descriptive",
        "central empirical span is timing-only and not a calibrated CI",
    )

    reconstruction_calls = coupling_source.count(
        "verify_stencil_topology(topology, stencil_set);"
    )
    require(
        reconstruction_calls == 2,
        "V2 canonical reconstruction boundary no longer has exactly deposit/gather calls",
    )
    require_fragments(
        coupling_source,
        "src/adaptiveCompositeCouplingCPP.cc",
        ("const CompositeTentStencilSetR2 canonical =", "build_composite_tent_stencils_r2("),
    )
    open_finding(
        "P2-deposit-and-gather-repeat-canonical-reconstruction",
        "deposit and gather each invoke the full canonical-set verifier",
    )

    actual_ids = {item["id"] for item in findings}
    require(actual_ids == EXPECTED_FINDING_IDS, "P1/P2 finding inventory is incomplete")

    # Preserve the real V2 evidence markers where they exist and the explicit
    # open production gap where they do not.  This is retained context only;
    # execute_negative_tests supplies one fail-closed V3 execution record for
    # every boundary before the final receipt is assembled.
    present_negative_markers = {
        "cross_interface_particle_support": "SupportCrossesRefinementBoundary",
        "missing_runtime_identity_receipt":
            "requires_zero_tolerance_runtime_identity_receipt",
        "stale_topology_token": "stale topology fingerprint did not fail closed",
        "spectrum_authentication_failure":
            "test_strict_native_contract_rejects_adversarial_mutations",
    }
    marker_sources = {
        "cross_interface_particle_support": coupling_test,
        "missing_runtime_identity_receipt": campaign_tests,
        "stale_topology_token": validator_source,
        "spectrum_authentication_failure": spectrum_tests,
    }
    for boundary, marker in present_negative_markers.items():
        require(marker in marker_sources[boundary], f"missing negative test: {boundary}")

    boundary_contracts: dict[str, dict[str, str]] = {
        boundary: {
            "retained_context": "existing_v2_negative_marker",
            "evidence": marker,
        }
        for boundary, marker in present_negative_markers.items()
    }
    boundary_contracts.update(
        {
            "off_center_self_force_independent_magnitude": {
                "retained_context": "open_production_gap",
                "evidence": "current off-center check is only a correlated energy-gradient identity",
            },
            "externally_asserted_uncertainty_terms": {
                "retained_context": "open_production_gap",
                "evidence": "current analyzer accepts supplied upper_bound/method/evidence",
            },
            "stale_source_token": {
                "retained_context": "open_production_gap",
                "evidence": "current coupling binds topology but has no independent source token",
            },
            "physical_root_pool_exhaustion": {
                "retained_context": "open_production_gap",
                "evidence": "current exhaustion test covers a campaign seed pool, not an in-process physical-root pool",
            },
        }
    )
    require(len(boundary_contracts) == 8, "required boundary inventory is incomplete")
    require(
        sum(
            item["retained_context"] == "existing_v2_negative_marker"
            for item in boundary_contracts.values()
        )
        == 4,
        "unexpected retained V2 negative-marker count",
    )
    require(
        sum(
            item["retained_context"] == "open_production_gap"
            for item in boundary_contracts.values()
        )
        == 4,
        "unexpected retained open-production-gap count",
    )
    return findings, boundary_contracts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--feebl", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--link-libraries-receipt", required=True, type=Path)
    parser.add_argument(
        "--receipt-file",
        type=Path,
        help=(
            "optionally create one immutable machine-readable G0 receipt; "
            "the parent directory must already exist"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = args.validator.resolve()
    feebl = args.feebl.resolve()
    source_root = args.source_root.resolve()
    link_receipt = args.link_libraries_receipt.resolve()
    require(validator.is_file(), f"Stage-4 validator is missing: {validator}")
    require(source_root.is_dir(), f"source root is missing: {source_root}")

    diagnostic = run_validator(validator, strict=False)
    require(diagnostic.returncode == 0, "diagnostic Stage-4 invocation did not exit 0")
    require(not diagnostic.stderr, f"diagnostic Stage-4 stderr: {diagnostic.stderr!r}")
    diagnostic_evidence = compare_stage4_output("diagnostic", diagnostic.stdout)

    strict = run_validator(validator, strict=True)
    require(strict.returncode == 2, f"frozen strict Stage-4 exit is {strict.returncode}, not 2")
    require(not strict.stderr, f"strict Stage-4 stderr: {strict.stderr!r}")
    strict_evidence = compare_stage4_output("strict", strict.stdout)
    require(
        strict_evidence["metrics"] == diagnostic_evidence["metrics"],
        "diagnostic and strict Stage-4 metrics differ",
    )

    production_link = audit_production_link(feebl, link_receipt)
    findings, boundary_contracts = audit_review_findings(source_root)
    negative_executions = execute_negative_tests(
        source_root, diagnostic_evidence["metrics"]
    )
    executed_contracts: dict[str, dict[str, Any]] = {}
    for execution in negative_executions:
        identifier = execution["id"]
        require(
            identifier in boundary_contracts,
            f"executed negative is absent from boundary inventory: {identifier}",
        )
        executed_contracts[identifier] = {
            **boundary_contracts[identifier],
            **{key: value for key, value in execution.items() if key != "id"},
            "status": "qualification_boundary_negative_executed",
        }
    require(
        set(executed_contracts) == set(boundary_contracts),
        "not every required boundary has executable negative evidence",
    )
    require(
        all(
            contract["status"]
            == "qualification_boundary_negative_executed"
            for contract in executed_contracts.values()
        ),
        "not every boundary has the executed V3 qualification status",
    )
    receipt = {
        "schema": SCHEMA,
        "status": "pass",
        "baseline_role": "frozen_v2_failing_baseline",
        "production_qualification": False,
        "strict_stage4_argument": "--require-stage4",
        "strict_stage4_exit": strict.returncode,
        "numeric_comparison": {
            "relative_tolerance": format(RELATIVE_TOLERANCE, ".17g"),
            "absolute_tolerance": format(ABSOLUTE_TOLERANCE, ".17g"),
            "phase_counts_exact": True,
        },
        "stage4": diagnostic_evidence,
        "validator_sha256": sha256_file(validator),
        "production_link": production_link,
        "open_independent_review_findings": findings,
        "required_boundary_contracts": executed_contracts,
        "execution_scope": (
            "standalone_validator_read_only_link_and_pure_in_memory_"
            "adversarial_qualification_evidence"
        ),
    }
    serialized_receipt = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    if args.receipt_file is not None:
        receipt_file = args.receipt_file.resolve(strict=False)
        require(
            receipt_file.parent.is_dir(),
            f"receipt parent directory does not exist: {receipt_file.parent}",
        )
        require(
            not receipt_file.exists(),
            f"refusing to overwrite existing receipt: {receipt_file}",
        )
        with receipt_file.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized_receipt + "\n")
    print(serialized_receipt)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditFailure, OSError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                {"schema": SCHEMA, "status": "fail", "reason": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
