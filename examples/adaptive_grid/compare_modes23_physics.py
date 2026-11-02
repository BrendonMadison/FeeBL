#!/usr/bin/env python3
"""Validate deterministic Mode-2/3 mechanisms without claiming physics convergence.

The frozen fixture exercises the adaptive controller, same-particle field-error
oracle, and Mode-3 patch/promotion state machine. Its fixed-grid ladder is a
diagnostic only: an unresolved ladder must never be presented as evidence for
the 1--3% production-physics accuracy target.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
from pathlib import Path
from typing import Any


CONFIG_PREFIX = "FEEBL_ADAPTIVE_GRID_CONFIG_V1"
ENCOUNTER_PREFIX = "FEEBL_ADAPTIVE_ENCOUNTER_V1"
SUMMARY_PREFIX = "FEEBL adaptive grid summary:"
SPACING_PREFIX = "FEEBL adaptive spacing class:"

REJECTION_GEOMETRY = 0x01
REJECTION_TAIL = 0x02
REJECTION_STRADDLER = 0x04
REJECTION_EMPTY_SUPPORT = 0x08
REJECTION_SAMPLING = 0x10
REJECTION_RESIDUAL = 0x20
REJECTION_FIELD_ORACLE = 0x40
REJECTION_LUMI_ORACLE = 0x80
KNOWN_REJECTION_MASK = 0xFF

PATTERNS = {
    "lumi_fine": re.compile(r"^lumi_fine\s*=\s*([-+0-9.eE]+)"),
    "lumi_ee": re.compile(r"^lumi_ee\s*=\s*([-+0-9.eE]+).*from beam particles collisions"),
    "upsmax": re.compile(r"^upsmax=\s*([-+0-9.eE]+)"),
    "E_cm": re.compile(r"^E_cm\s*=\s*([-+0-9.eE]+)"),
    "E_cm_var": re.compile(r"^E_cm\s*=.*E_cm_var\s*=\s*([-+0-9.eE]+)"),
}

for beam in (1, 2):
    for axis in ("x", "y"):
        PATTERNS[f"bpm_v{axis}.{beam}"] = re.compile(rf"bpm_v{axis}\.{beam}=([-+0-9.eE]+)")
        PATTERNS[f"bpm_sig_v{axis}.{beam}"] = re.compile(rf"bpm_sig_v{axis}\.{beam}=([-+0-9.eE]+)")


def parse_key_values(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise RuntimeError(f"line does not start with {prefix!r}")
    result: dict[str, str] = {}
    for token in shlex.split(line[len(prefix) :].strip()):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value
    return result


def integer(values: dict[str, str], key: str, context: str) -> int:
    try:
        return int(values[key])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"{context}: invalid or missing integer {key}") from error


def number(values: dict[str, str], key: str, context: str) -> float:
    try:
        value = float(values[key])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"{context}: invalid or missing number {key}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"{context}: non-finite {key}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close_leq(value: float, limit: float) -> bool:
    return value <= limit + 1.0e-12 * max(1.0, abs(limit))


def validate_nested_rejection_state(
    *,
    context: str,
    requested: int,
    used: int,
    promoted: int,
    rejection_mask: int,
    coarse_cells: int,
    fine_cells: int,
    iterations: int,
    tail: float,
    tail_tolerance: float,
    straddlers: int,
    nested_neff_p10: float,
    nested_neff_critical_min: float,
    minimum_neff: float,
    residual: float,
    residual_tolerance: float,
    patch_rms: float,
    patch_linf: float,
    error_target: float,
    noise_limited: int,
) -> None:
    """Validate the fail-closed, staged Mode-3 rejection receipt.

    Some comparisons are deliberately one-way because encounter values are
    human-readable, rounded telemetry. Integer stage markers and impossible
    stage combinations remain exact checks.
    """
    require(
        rejection_mask >= 0 and rejection_mask & ~KNOWN_REJECTION_MASK == 0,
        f"{context}: unknown nested-patch rejection bit",
    )
    require(noise_limited in (0, 1), f"{context}: invalid noise_limited boolean telemetry")

    if requested and tail > tail_tolerance:
        require(rejection_mask & REJECTION_TAIL, f"{context}: tail failure lacks its rejection bit")
    require(
        bool(rejection_mask & REJECTION_STRADDLER) == (straddlers > 0),
        f"{context}: straddler count and rejection bit disagree",
    )

    nested_sampling_failed = coarse_cells > 0 and (
        nested_neff_p10 < minimum_neff or nested_neff_critical_min < minimum_neff
    )
    if nested_sampling_failed:
        require(
            rejection_mask & REJECTION_SAMPLING,
            f"{context}: nested Kish-N_eff failure lacks its rejection bit",
        )
    if residual > residual_tolerance:
        require(
            rejection_mask & REJECTION_RESIDUAL,
            f"{context}: residual failure lacks its rejection bit",
        )
    if max(patch_rms, patch_linf) > error_target:
        require(
            rejection_mask & REJECTION_FIELD_ORACLE,
            f"{context}: field-oracle failure lacks its rejection bit",
        )

    pre_solve_mask = REJECTION_GEOMETRY | REJECTION_TAIL | REJECTION_STRADDLER
    if rejection_mask & pre_solve_mask:
        require(
            rejection_mask & ~pre_solve_mask == 0
            and promoted == 1
            and coarse_cells == fine_cells == iterations == 0
            and residual == patch_rms == patch_linf == 0.0,
            f"{context}: pre-solve rejection reported later-stage work",
        )
    if rejection_mask & REJECTION_EMPTY_SUPPORT:
        require(
            rejection_mask == REJECTION_EMPTY_SUPPORT
            and promoted == 1
            and coarse_cells == fine_cells == iterations == 0
            and residual == patch_rms == patch_linf == 0.0,
            f"{context}: empty-support rejection has impossible stage telemetry",
        )
    if rejection_mask & REJECTION_SAMPLING:
        require(
            rejection_mask == REJECTION_SAMPLING
            and requested == 1
            and used == 0
            and promoted == 1
            and coarse_cells > 0
            and fine_cells > 0
            and iterations == 0
            and noise_limited == 1
            and residual == patch_rms == patch_linf == 0.0,
            f"{context}: sampling rejection did not stop before harmonic/oracle work",
        )
    if rejection_mask & REJECTION_RESIDUAL:
        require(
            rejection_mask == REJECTION_RESIDUAL
            and requested == 1
            and used == 0
            and promoted == 1
            and coarse_cells > 0
            and fine_cells > 0
            and iterations > 0
            and patch_rms == patch_linf == 0.0,
            f"{context}: residual rejection did not stop before field-oracle work",
        )
    if rejection_mask & REJECTION_FIELD_ORACLE:
        require(
            rejection_mask == REJECTION_FIELD_ORACLE
            and requested == 1
            and used == 0
            and promoted == 1
            and coarse_cells > 0
            and fine_cells > 0
            and iterations > 0
            and close_leq(residual, residual_tolerance),
            f"{context}: field-oracle rejection has impossible prior-stage telemetry",
        )
    if rejection_mask & REJECTION_LUMI_ORACLE:
        require(
            rejection_mask == REJECTION_LUMI_ORACLE
            and requested == 1
            and used == 0
            and promoted == 1
            and coarse_cells > 0
            and fine_cells > 0
            and iterations > 0
            and close_leq(residual, residual_tolerance)
            and close_leq(patch_rms, error_target)
            and close_leq(patch_linf, error_target),
            f"{context}: luminosity-oracle rejection has impossible prior-stage telemetry",
        )


def require_attempted_maximum(
    summary_value: float,
    receipt_value: float,
    context: str,
    label: str,
    *,
    relative_tolerance: float,
) -> None:
    """Match an attempted-work maximum without imposing an acceptance bound."""
    require(
        math.isclose(summary_value, receipt_value, rel_tol=relative_tolerance, abs_tol=1.0e-12),
        f"{context}: maximum attempted-patch {label} is invalid",
    )


def single_record(records: list[dict[str, str]], name: str, path: Path) -> dict[str, str]:
    require(len(records) == 1, f"{path}: expected one {name}, found {len(records)}")
    return records[0]


def read_adaptive_log(path: Path) -> dict[str, Any]:
    configs: list[dict[str, str]] = []
    encounters: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    spacing: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(CONFIG_PREFIX):
            configs.append(parse_key_values(line, CONFIG_PREFIX))
        elif line.startswith(ENCOUNTER_PREFIX):
            encounters.append(parse_key_values(line, ENCOUNTER_PREFIX))
        elif line.startswith(SUMMARY_PREFIX):
            summaries.append(parse_key_values(line, SUMMARY_PREFIX))
        elif line.startswith(SPACING_PREFIX):
            spacing.append(parse_key_values(line, SPACING_PREFIX))
    return {
        "config": single_record(configs, "configuration receipt", path),
        "encounters": encounters,
        "summary": single_record(summaries, "summary receipt", path),
        "spacing": spacing,
    }


def validate_adaptive_log(
    path: Path,
    expected_mode: int,
    *,
    require_variable_spacing: bool = True,
    expect_lazy_stop: bool = False,
    require_full_mode3_state_coverage: bool = True,
    required_mode3_rejection_bits: int = 0,
) -> dict[str, Any]:
    parsed = read_adaptive_log(path)
    config = parsed["config"]
    summary = parsed["summary"]
    encounters = parsed["encounters"]
    spacing = parsed["spacing"]
    context = f"{path}: Mode {expected_mode}"

    require(
        config.get("configuration_status") == "accepted",
        f"{context}: configuration was not accepted",
    )
    require(
        integer(config, "requested_mode", context) == expected_mode,
        f"{context}: wrong requested mode",
    )
    require(
        integer(summary, "requested", context) == expected_mode
        and integer(summary, "actual", context) == expected_mode,
        f"{context}: requested/actual mode mismatch",
    )
    require(config.get("scope") == "encounter", f"{context}: validation requires encounter scope")
    require(config.get("profile") == "encounter", f"{context}: encounter telemetry is disabled")
    require(
        integer(config, "fail_on_unmet_error", context) == 1,
        f"{context}: fail-closed behavior is disabled",
    )
    require(
        integer(config, "update_interval", context) == 1
        and integer(config, "richardson_interval", context) == 1
        and integer(config, "shrink_dwell", context) == 1,
        f"{context}: fixture does not gate every encounter",
    )

    declared_factors = [int(item) for item in config["field_spacing_classes"].split(",")]
    require(len(declared_factors) >= 3, f"{context}: fewer than three spacing classes")
    require(
        all(
            declared_factors[index + 1] == 2 * declared_factors[index]
            for index in range(len(declared_factors) - 1)
        ),
        f"{context}: spacing classes are not contiguous ratio-two levels",
    )
    error_target = number(config, "error_target", context)
    require(
        math.isclose(
            number(config, "tau_richardson", context),
            error_target,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ),
        f"{context}: Richardson and field-error targets disagree",
    )
    minimum_neff = number(config, "min_neff", context)
    curvature_limit = number(config, "tau_curvature", context) * number(
        config, "refine_threshold", context
    )
    gradient_limit = number(config, "tau_field_gradient", context) * number(
        config, "refine_threshold", context
    )
    encounter_count = integer(summary, "encounters", context)
    require(encounter_count > 0, f"{context}: no adaptive encounters")
    require(
        len(encounters) == encounter_count,
        f"{context}: expected {encounter_count} encounter receipts, " f"found {len(encounters)}",
    )
    require(
        integer(summary, "mode2_unmet_error_encounters", context) == 0,
        f"{context}: fail-closed field-error gate was not met",
    )
    require(
        integer(summary, "plans", context) == encounter_count
        and integer(summary, "mode2_encounters", context) == encounter_count,
        f"{context}: plan/Mode-2 encounter summary is inconsistent",
    )
    require(
        close_leq(number(summary, "maximum_estimated_field_error", context), error_target),
        f"{context}: summary field error exceeds target",
    )
    uniform_context_misses = integer(summary, "uniform_context_misses", context)
    patch_context_misses = integer(summary, "patch_context_misses", context)
    cold_miss_count = integer(summary, "cold_miss_count", context)
    require(
        cold_miss_count == uniform_context_misses + patch_context_misses,
        f"{context}: cold misses do not equal uniform plus patch misses",
    )
    require(
        integer(summary, "context_builds", context) >= cold_miss_count,
        f"{context}: context builds are fewer than genuine cold misses",
    )

    spacing_counts: dict[int, int] = {}
    for record in spacing:
        factor = integer(record, "factor", context)
        require(
            factor not in spacing_counts,
            f"{context}: duplicate spacing counter for factor {factor}",
        )
        spacing_counts[factor] = integer(record, "selected_encounters", context)
    factors = sorted(spacing_counts)
    require(
        factors[: len(declared_factors)] == declared_factors,
        f"{context}: admitted spacing ladder does not preserve the declared " "bootstrap prefix",
    )
    require(
        all(factors[index + 1] == 2 * factors[index] for index in range(len(factors) - 1)),
        f"{context}: admitted spacing ladder is not contiguous ratio-two",
    )
    refinement_cap = integer(config, "max_refinement_factor", context)
    require(
        refinement_cap == 0 or factors[-1] <= refinement_cap,
        f"{context}: admitted spacing factor exceeds the explicit cap",
    )
    require(
        sum(spacing_counts.values()) == encounter_count,
        f"{context}: spacing counts do not sum to encounters",
    )

    selected_factors: list[int] = []
    used_count = 0
    promoted_count = 0
    coarse_cells = 0
    fine_cells = 0
    maximum_residual = 0.0
    maximum_patch_error = 0.0
    maximum_examined_patch_lumi_error = 0.0
    maximum_accepted_patch_lumi_error = 0.0
    patch_lumi_oracle_evaluations = 0
    accepted_patch_lumi_oracle_evaluations = 0
    source_beam1_in_visits = 0
    source_beam1_out_visits = 0
    source_beam2_in_visits = 0
    source_beam2_out_visits = 0
    maximum_estimated_error = 0.0
    request_count = 0
    terminal_count = 0
    no_patch_count = 0
    empty_count = 0
    nonempty_count = 0
    attempted_coarse_cells = 0
    attempted_fine_cells = 0
    harmonic_iterations = 0
    rejection_mask_or = 0
    expected_field_solves = 0
    composite_quadrature_checks: list[dict[str, float]] = []
    refinement_ratio = integer(config, "refinement_ratio", context) if expected_mode == 3 else 0
    tail_tolerance = number(config, "tau_tail", context)
    residual_tolerance = (
        number(config, "fine_residual_tolerance", context) if expected_mode == 3 else 0.0
    )
    reference_source = config.get("source_model") == "reference_cic_v1"
    lumi_error_target = number(config, "lumi_error_target", context) if reference_source else 0.0

    for index, encounter in enumerate(encounters):
        item_context = f"{context} encounter[{index}]"
        selected = integer(encounter, "selected_spacing_factor", item_context)
        selected_factors.append(selected)
        require(selected in factors, f"{item_context}: selected undeclared factor {selected}")
        selected_index = factors.index(selected)
        selected_nx = integer(encounter, "selected_grid_nx", item_context)
        selected_ny = integer(encounter, "selected_grid_ny", item_context)
        require(
            (selected_nx == 0) == (selected_ny == 0),
            f"{item_context}: only one selected grid dimension is empty",
        )
        empty = selected_nx == 0

        source_counts = {
            key: integer(encounter, key, item_context)
            for key in (
                "source_beam1_in",
                "source_beam1_out",
                "source_beam2_in",
                "source_beam2_out",
            )
        }
        require(
            all(value >= 0 for value in source_counts.values()),
            f"{item_context}: negative source-admission count",
        )
        source_beam1_in_visits += source_counts["source_beam1_in"]
        source_beam1_out_visits += source_counts["source_beam1_out"]
        source_beam2_in_visits += source_counts["source_beam2_in"]
        source_beam2_out_visits += source_counts["source_beam2_out"]

        requested = integer(encounter, "nested_patch_requested", item_context)
        used = integer(encounter, "nested_patch_used", item_context)
        promoted = integer(encounter, "nested_patch_promoted_uniform", item_context)
        rejection_mask = integer(encounter, "nested_patch_rejection_mask", item_context)
        patch_lumi_error = number(encounter, "nested_lumi_error", item_context)
        require(patch_lumi_error >= 0.0, f"{item_context}: negative patch luminosity-oracle error")
        require(
            requested in (0, 1) and used in (0, 1) and promoted in (0, 1),
            f"{item_context}: invalid Mode-3 boolean telemetry",
        )
        require(
            rejection_mask >= 0 and rejection_mask & ~KNOWN_REJECTION_MASK == 0,
            f"{item_context}: unknown nested-patch rejection bit",
        )
        request_count += requested
        rejection_mask_or |= rejection_mask

        coarse = integer(encounter, "nested_coarse_cells", item_context)
        fine = integer(encounter, "nested_fine_cells", item_context)
        iterations = integer(encounter, "nested_iterations", item_context)
        require(
            (coarse == 0) == (fine == 0), f"{item_context}: partial attempted-patch cell accounting"
        )
        harmonic_iterations += iterations
        if coarse > 0:
            require(
                expected_mode == 3 and requested == 1 and fine == coarse * refinement_ratio**2,
                f"{item_context}: attempted patch cell ratio is inconsistent",
            )
            coarse_delta_x = number(encounter, "nested_coarse_delta_x", item_context)
            coarse_delta_y = number(encounter, "nested_coarse_delta_y", item_context)
            fine_delta_x = number(encounter, "nested_fine_delta_x", item_context)
            fine_delta_y = number(encounter, "nested_fine_delta_y", item_context)
            require(
                math.isclose(
                    coarse_delta_x,
                    refinement_ratio * fine_delta_x,
                    # Text receipts use the stream's compact
                    # human-readable precision (for example,
                    # 2*1.17188 versus 2.34375).
                    rel_tol=1.0e-5,
                    abs_tol=1.0e-12,
                )
                and math.isclose(
                    coarse_delta_y, refinement_ratio * fine_delta_y, rel_tol=1.0e-5, abs_tol=1.0e-12
                ),
                f"{item_context}: nested physical spacing ratio is inconsistent",
            )
            parent_nx = selected_nx
            parent_ny = selected_ny
            if promoted:
                require(
                    parent_nx % refinement_ratio == 0 and parent_ny % refinement_ratio == 0,
                    f"{item_context}: promoted grid cannot recover its patch parent",
                )
                parent_nx //= refinement_ratio
                parent_ny //= refinement_ratio
            parent_cells = parent_nx * parent_ny
            require(coarse <= parent_cells, f"{item_context}: nested patch exceeds its parent grid")
            coarse_cell_area = coarse_delta_x * coarse_delta_y
            fine_cell_area = fine_delta_x * fine_delta_y
            parent_integral = parent_cells * coarse_cell_area
            outside_coarse_integral = (parent_cells - coarse) * coarse_cell_area
            inside_fine_integral = fine * fine_cell_area
            composite_integral = outside_coarse_integral + inside_fine_integral
            quadrature_error = symmetric_relative_change(parent_integral, composite_integral)
            receipt_quadrature_tolerance = 1.0e-5
            require(
                quadrature_error <= receipt_quadrature_tolerance,
                f"{item_context}: single-ownership composite quadrature is not conservative",
            )
            composite_quadrature_checks.append(
                {
                    "parent_constant_integral": parent_integral,
                    "outside_coarse_constant_integral": outside_coarse_integral,
                    "inside_fine_constant_integral": inside_fine_integral,
                    "composite_constant_integral": composite_integral,
                    "symmetric_relative_error": quadrature_error,
                    "tolerance": receipt_quadrature_tolerance,
                }
            )
            attempted_coarse_cells += coarse
            attempted_fine_cells += fine
        else:
            require(iterations == 0, f"{item_context}: empty patch attempt reports iterations")

        if empty:
            empty_count += 1 if expected_mode == 3 else 0
            require(
                selected == factors[0]
                and requested == used == promoted == 0
                and rejection_mask == 0,
                f"{item_context}: empty encounter has noncanonical state",
            )
            zero_numbers = (
                "selected_delta_x",
                "selected_delta_y",
                "estimated_field_error",
                "difference_rms",
                "difference_linf",
                "curvature_score",
                "field_gradient_score",
                "neff_p10",
                "neff_critical_min",
                "nested_coarse_delta_x",
                "nested_coarse_delta_y",
                "nested_fine_delta_x",
                "nested_fine_delta_y",
                "nested_neff_p10",
                "nested_neff_critical_min",
                "nested_residual",
                "nested_field_error_rms",
                "nested_field_error_linf",
                "nested_lumi_error",
                "nested_tail_fraction",
            )
            require(
                source_counts["source_beam1_in"] == 0
                and source_counts["source_beam2_in"] == 0
                and coarse == fine == 0
                and all(number(encounter, key, item_context) == 0.0 for key in zero_numbers)
                and integer(encounter, "nested_straddling_sources", item_context) == 0,
                f"{item_context}: empty encounter has nonzero field telemetry",
            )
            continue

        nonempty_count += 1
        require(
            integer(encounter, "error_target_met", item_context) == 1,
            f"{item_context}: simultaneous RMS/Linf/asymptotic gate failed",
        )
        noise_limited = integer(encounter, "noise_limited", item_context)
        require(noise_limited in (0, 1), f"{item_context}: invalid noise_limited boolean telemetry")
        estimated_error = number(encounter, "estimated_field_error", item_context)
        require(
            close_leq(estimated_error, error_target), f"{item_context}: field error exceeds target"
        )
        maximum_estimated_error = max(maximum_estimated_error, estimated_error)
        require(
            number(encounter, "difference_rms", item_context) >= 0.0
            and number(encounter, "difference_linf", item_context) >= 0.0,
            f"{item_context}: invalid RMS/Linf difference telemetry",
        )
        require(
            number(encounter, "neff_p10", item_context) >= minimum_neff
            and number(encounter, "neff_critical_min", item_context) >= minimum_neff,
            f"{item_context}: Kish-N_eff is below the configured minimum",
        )
        require(
            close_leq(number(encounter, "curvature_score", item_context), curvature_limit),
            f"{item_context}: charge-curvature indicator exceeds its gate",
        )
        require(
            close_leq(number(encounter, "field_gradient_score", item_context), gradient_limit),
            f"{item_context}: field-gradient indicator exceeds its gate",
        )
        require(
            selected_nx == integer(config, "max_nx", context) * selected
            and selected_ny == integer(config, "max_ny", context) * selected,
            f"{item_context}: selected grid dimensions do not realize spacing factor",
        )

        if expected_mode == 2:
            require(
                requested == used == promoted == 0
                and rejection_mask == 0
                and coarse == fine == 0
                and patch_lumi_error == 0.0,
                f"{item_context}: Mode 2 reported Mode-3 patch state",
            )
            expected_field_solves += min(len(factors), selected_index + 3)
            continue

        require(
            (used + promoted == 1) if requested else (used + promoted == 0),
            f"{item_context}: requested patch did not resolve to exactly one "
            "of nested use or explicit uniform promotion",
        )
        require(
            (rejection_mask != 0) == bool(promoted),
            f"{item_context}: rejection mask and promotion state disagree",
        )
        tail = number(encounter, "nested_tail_fraction", item_context)
        straddlers = integer(encounter, "nested_straddling_sources", item_context)
        nested_neff_p10 = number(encounter, "nested_neff_p10", item_context)
        nested_neff_critical_min = number(encounter, "nested_neff_critical_min", item_context)
        residual = number(encounter, "nested_residual", item_context)
        patch_rms = number(encounter, "nested_field_error_rms", item_context)
        patch_linf = number(encounter, "nested_field_error_linf", item_context)
        if iterations > 0:
            maximum_residual = max(maximum_residual, residual)
        else:
            require(residual == 0.0, f"{item_context}: residual reported without a harmonic solve")
        validate_nested_rejection_state(
            context=item_context,
            requested=requested,
            used=used,
            promoted=promoted,
            rejection_mask=rejection_mask,
            coarse_cells=coarse,
            fine_cells=fine,
            iterations=iterations,
            tail=tail,
            tail_tolerance=tail_tolerance,
            straddlers=straddlers,
            nested_neff_p10=nested_neff_p10,
            nested_neff_critical_min=nested_neff_critical_min,
            minimum_neff=minimum_neff,
            residual=residual,
            residual_tolerance=residual_tolerance,
            patch_rms=patch_rms,
            patch_linf=patch_linf,
            error_target=error_target,
            noise_limited=noise_limited,
        )
        oracle_evaluated = reference_source and (
            used == 1 or bool(rejection_mask & REJECTION_LUMI_ORACLE)
        )
        if oracle_evaluated:
            patch_lumi_oracle_evaluations += 1
            maximum_examined_patch_lumi_error = max(
                maximum_examined_patch_lumi_error, patch_lumi_error
            )
            if used:
                accepted_patch_lumi_oracle_evaluations += 1
                maximum_accepted_patch_lumi_error = max(
                    maximum_accepted_patch_lumi_error, patch_lumi_error
                )
                require(
                    close_leq(patch_lumi_error, lumi_error_target),
                    f"{item_context}: accepted patch luminosity oracle exceeds target",
                )
            # The exact comparison is made before publication.  The receipt is
            # human-readable rounded telemetry, so bit 0x80 is authoritative
            # when a value just above the threshold prints at the threshold.
        else:
            require(
                patch_lumi_error == 0.0,
                f"{item_context}: luminosity-oracle error lacks an evaluation state",
            )
        if promoted:
            promoted_count += 1
            require(
                selected_index > 0, f"{item_context}: promotion silently stayed at the coarse level"
            )
            expected_field_solves += min(len(factors), selected_index + 2)
        else:
            expected_field_solves += min(len(factors), selected_index + 3)
        if coarse > 0:
            expected_field_solves += 1

        if used:
            used_count += 1
            require(
                coarse > 0 and rejection_mask == 0,
                f"{item_context}: accepted patch lacks a clean attempted solve",
            )
            require(
                close_leq(residual, residual_tolerance),
                f"{item_context}: nested residual exceeds tolerance",
            )
            require(
                close_leq(patch_rms, error_target) and close_leq(patch_linf, error_target),
                f"{item_context}: nested RMS/Linf oracle exceeds target",
            )
            require(
                tail <= tail_tolerance and straddlers == 0,
                f"{item_context}: unsafe nested patch was accepted",
            )
            require(
                nested_neff_p10 >= minimum_neff and nested_neff_critical_min >= minimum_neff,
                f"{item_context}: nested patch Kish-N_eff gate failed",
            )
            coarse_cells += coarse
            fine_cells += fine
        elif promoted:
            require(
                rejection_mask != 0, f"{item_context}: promotion has no explicit rejection reason"
            )
        elif selected == factors[-1]:
            terminal_count += 1
        else:
            no_patch_count += 1

        if patch_rms > 0.0 or patch_linf > 0.0:
            maximum_patch_error = max(maximum_patch_error, patch_rms, patch_linf)

    nonzero_factors = sorted(factor for factor, count in spacing_counts.items() if count > 0)
    require(
        {factor: selected_factors.count(factor) for factor in factors} == spacing_counts,
        f"{context}: encounter receipts disagree with spacing summary",
    )
    require(
        integer(summary, "mode2_refined_encounters", context)
        == sum(selected > factors[0] for selected in selected_factors),
        f"{context}: refined-encounter summary disagrees with receipts",
    )
    require(
        math.isclose(
            number(summary, "maximum_estimated_field_error", context),
            maximum_estimated_error,
            rel_tol=1.0e-5,
            abs_tol=1.0e-12,
        ),
        f"{context}: maximum estimated field error disagrees with receipts",
    )
    nonempty_receipts = [
        (encounter, selected)
        for encounter, selected in zip(encounters, selected_factors)
        if integer(encounter, "selected_grid_nx", context) > 0
    ]
    require(
        len(nonempty_receipts) == nonempty_count and nonempty_count > 0,
        f"{context}: nonempty encounter accounting is inconsistent",
    )
    scaled_delta_x = [
        number(encounter, "selected_delta_x", context) * selected
        for encounter, selected in nonempty_receipts
    ]
    scaled_delta_y = [
        number(encounter, "selected_delta_y", context) * selected
        for encounter, selected in nonempty_receipts
    ]
    require(
        all(
            math.isclose(value, scaled_delta_x[0], rel_tol=1.0e-5, abs_tol=1.0e-12)
            for value in scaled_delta_x
        )
        and all(
            math.isclose(value, scaled_delta_y[0], rel_tol=1.0e-5, abs_tol=1.0e-12)
            for value in scaled_delta_y
        ),
        f"{context}: selected physical spacings do not scale with factor",
    )
    variable_spacing = (
        len(
            set(
                number(encounter, "selected_delta_x", context) for encounter, _ in nonempty_receipts
            )
        )
        >= 2
        and len(
            set(
                number(encounter, "selected_delta_y", context) for encounter, _ in nonempty_receipts
            )
        )
        >= 2
    )
    if require_variable_spacing:
        require(
            len(nonzero_factors) >= 2 and variable_spacing,
            f"{context}: physical grid spacing did not actually change",
        )
        require(
            integer(summary, "mode2_refined_encounters", context) > 0,
            f"{context}: fixture did not refine any encounter",
        )
    require(cold_miss_count > 0, f"{context}: fixture did not exercise a genuine context cold miss")
    require(
        integer(summary, "mode2_field_solves", context) == expected_field_solves,
        f"{context}: lazy uniform/patch solve accounting is inconsistent",
    )
    for key, expected in (
        ("source_beam1_in_visits", source_beam1_in_visits),
        ("source_beam1_out_visits", source_beam1_out_visits),
        ("source_beam2_in_visits", source_beam2_in_visits),
        ("source_beam2_out_visits", source_beam2_out_visits),
    ):
        require(
            integer(summary, key, context) == expected,
            f"{context}: aggregate {key} disagrees with encounter receipts",
        )

    if expected_mode == 2:
        for key in (
            "mode3_patch_requests",
            "mode3_patch_encounters",
            "mode3_uniform_promotions",
            "mode3_terminal_uniform_encounters",
            "mode3_no_patch_encounters",
            "mode3_empty_encounters",
            "mode3_patch_coarse_cells",
            "mode3_patch_fine_cells",
            "mode3_attempted_patch_coarse_cells",
            "mode3_attempted_patch_fine_cells",
            "mode3_harmonic_iterations",
            "mode3_rejection_mask_or",
        ):
            require(
                integer(summary, key, context) == 0,
                f"{context}: Mode-3 summary counter {key} is nonzero",
            )
        require(
            integer(summary, "fallback_count", context) == 0,
            f"{context}: Mode 2 reported a fallback",
        )
        require(
            integer(summary, "patch_lumi_oracle_evaluations", context) == 0
            and integer(summary, "accepted_patch_lumi_oracle_evaluations", context) == 0
            and number(summary, "maximum_examined_patch_lumi_oracle_error", context) == 0.0
            and number(summary, "maximum_accepted_patch_lumi_oracle_error", context) == 0.0,
            f"{context}: Mode 2 reported Mode-3 luminosity-oracle work",
        )
        if expect_lazy_stop:
            require(
                len(factors) >= 4
                and spacing_counts[factors[-1]] == 0
                and expected_field_solves == 3 * nonempty_count
                and expected_field_solves < len(factors) * nonempty_count,
                f"{context}: four-class fixture did not stop after its "
                "rolling three-level audit",
            )
    else:
        requests = integer(summary, "mode3_patch_requests", context)
        terminal = integer(summary, "mode3_terminal_uniform_encounters", context)
        no_patch = integer(summary, "mode3_no_patch_encounters", context)
        empty_encounters = integer(summary, "mode3_empty_encounters", context)
        if require_full_mode3_state_coverage:
            require(
                requests > 0 and used_count > 0 and promoted_count > 0,
                f"{context}: fixture did not cover patch and promotion states",
            )
        else:
            require(requests > 0, f"{context}: focused rejection fixture made no patch request")
        require(
            requests == used_count + promoted_count,
            f"{context}: patch requests do not resolve fail-closed",
        )
        require(
            requests == request_count, f"{context}: patch-request summary disagrees with receipts"
        )
        require(
            integer(summary, "mode3_patch_encounters", context) == used_count,
            f"{context}: patch-use summary disagrees with receipts",
        )
        require(
            integer(summary, "mode3_uniform_promotions", context) == promoted_count,
            f"{context}: promotion summary disagrees with receipts",
        )
        require(
            terminal == terminal_count,
            f"{context}: terminal-uniform summary disagrees with receipts",
        )
        require(no_patch == no_patch_count, f"{context}: no-patch summary disagrees with receipts")
        require(
            empty_encounters == empty_count,
            f"{context}: empty-encounter summary disagrees with receipts",
        )
        require(
            used_count + promoted_count + terminal_count + no_patch_count + empty_count
            == encounter_count,
            f"{context}: mutually exclusive Mode-3 states do not partition encounters",
        )
        require(
            integer(summary, "mode3_patch_coarse_cells", context) == coarse_cells
            and integer(summary, "mode3_patch_fine_cells", context) == fine_cells,
            f"{context}: nested cell totals disagree with receipts",
        )
        require(
            integer(summary, "mode3_attempted_patch_coarse_cells", context)
            == attempted_coarse_cells
            and integer(summary, "mode3_attempted_patch_fine_cells", context)
            == attempted_fine_cells,
            f"{context}: attempted-patch cell totals disagree with receipts",
        )
        require_attempted_maximum(
            number(summary, "maximum_patch_residual", context),
            maximum_residual,
            context,
            "residual",
            relative_tolerance=1.0e-6,
        )
        require_attempted_maximum(
            number(summary, "maximum_patch_field_error", context),
            maximum_patch_error,
            context,
            "field error",
            relative_tolerance=1.0e-5,
        )
        require(
            integer(summary, "mode3_harmonic_iterations", context) == harmonic_iterations,
            f"{context}: nested patch harmonic-iteration accounting is invalid",
        )
        require(
            integer(summary, "mode3_rejection_mask_or", context) == rejection_mask_or,
            f"{context}: rejection-mask summary disagrees with receipts",
        )
        require(
            rejection_mask_or & required_mode3_rejection_bits == required_mode3_rejection_bits,
            f"{context}: focused fixture did not exercise required rejection bits "
            f"0x{required_mode3_rejection_bits:x}",
        )
        require(
            integer(summary, "fallback_count", context) == promoted_count,
            f"{context}: explicit promotion/fallback accounting is inconsistent",
        )
        require(
            integer(summary, "patch_lumi_oracle_evaluations", context)
            == patch_lumi_oracle_evaluations
            and integer(summary, "accepted_patch_lumi_oracle_evaluations", context)
            == accepted_patch_lumi_oracle_evaluations,
            f"{context}: luminosity-oracle evaluation totals disagree with receipts",
        )
        require_attempted_maximum(
            number(summary, "maximum_examined_patch_lumi_oracle_error", context),
            maximum_examined_patch_lumi_error,
            context,
            "luminosity-oracle error",
            relative_tolerance=1.0e-5,
        )
        require_attempted_maximum(
            number(summary, "maximum_accepted_patch_lumi_oracle_error", context),
            maximum_accepted_patch_lumi_error,
            context,
            "accepted luminosity-oracle error",
            relative_tolerance=1.0e-5,
        )

    return {
        "path": str(path),
        "mode": expected_mode,
        "encounters": encounter_count,
        "declared_spacing_factors": declared_factors,
        "admitted_spacing_factors": factors,
        "selected_spacing_counts": {str(factor): spacing_counts[factor] for factor in factors},
        "variable_physical_spacing_demonstrated": variable_spacing,
        "lazy_stop_expected": expect_lazy_stop,
        "field_solves": integer(summary, "mode2_field_solves", context),
        "uniform_context_misses": uniform_context_misses,
        "patch_context_misses": patch_context_misses,
        "cold_miss_count": cold_miss_count,
        "context_evictions": integer(summary, "context_evictions", context),
        "maximum_estimated_field_error": number(summary, "maximum_estimated_field_error", context),
        "error_target": error_target,
        "minimum_kish_neff": minimum_neff,
        "curvature_limit": curvature_limit,
        "field_gradient_limit": gradient_limit,
        "mode3_patch_requests": integer(summary, "mode3_patch_requests", context),
        "mode3_patch_encounters": integer(summary, "mode3_patch_encounters", context),
        "mode3_uniform_promotions": integer(summary, "mode3_uniform_promotions", context),
        "mode3_terminal_uniform_encounters": integer(
            summary, "mode3_terminal_uniform_encounters", context
        ),
        "mode3_no_patch_encounters": integer(summary, "mode3_no_patch_encounters", context),
        "mode3_empty_encounters": integer(summary, "mode3_empty_encounters", context),
        "mode3_patch_coarse_cells": integer(summary, "mode3_patch_coarse_cells", context),
        "mode3_patch_fine_cells": integer(summary, "mode3_patch_fine_cells", context),
        "mode3_attempted_patch_coarse_cells": integer(
            summary, "mode3_attempted_patch_coarse_cells", context
        ),
        "mode3_attempted_patch_fine_cells": integer(
            summary, "mode3_attempted_patch_fine_cells", context
        ),
        "mode3_rejection_mask_or": integer(summary, "mode3_rejection_mask_or", context),
        "maximum_patch_residual": number(summary, "maximum_patch_residual", context),
        "maximum_patch_field_error": number(summary, "maximum_patch_field_error", context),
        "patch_lumi_oracle_evaluations": integer(summary, "patch_lumi_oracle_evaluations", context),
        "accepted_patch_lumi_oracle_evaluations": integer(
            summary, "accepted_patch_lumi_oracle_evaluations", context
        ),
        "maximum_examined_patch_lumi_oracle_error": number(
            summary, "maximum_examined_patch_lumi_oracle_error", context
        ),
        "maximum_accepted_patch_lumi_oracle_error": number(
            summary, "maximum_accepted_patch_lumi_oracle_error", context
        ),
        "source_admission_visits": {
            "beam1_in": source_beam1_in_visits,
            "beam1_out": source_beam1_out_visits,
            "beam2_in": source_beam2_in_visits,
            "beam2_out": source_beam2_out_visits,
        },
        "synthetic_constant_field_composite_quadrature": {
            "status": "pass" if composite_quadrature_checks else "not_applicable",
            "checks": composite_quadrature_checks,
        },
        "fail_closed_gates": [
            "same-particle RMS error",
            "same-particle Linf error",
            "three-level asymptotic contraction",
            "Kish effective-particle minimum",
            "curvature and field-gradient indicators",
        ],
    }


def read_metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for name, pattern in PATTERNS.items():
            if name in values:
                continue
            match = pattern.search(line)
            if match:
                values[name] = float(match.group(1))
    required = {
        "lumi_fine",
        "upsmax",
        "bpm_vx.1",
        "bpm_sig_vx.1",
        "bpm_vy.1",
        "bpm_sig_vy.1",
        "bpm_vx.2",
        "bpm_sig_vx.2",
        "bpm_vy.2",
        "bpm_sig_vy.2",
    }
    missing = sorted(required - set(values))
    if missing:
        raise RuntimeError(f"{path}: missing operator metrics: {', '.join(missing)}")
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"{path}: non-finite operator metric")
    require(
        values["lumi_fine"] >= 0.0 and values["upsmax"] >= 0.0,
        f"{path}: invalid negative scalar operator metric",
    )
    for name, value in values.items():
        if name.startswith("bpm_sig_"):
            require(value >= 0.0, f"{path}: negative BPM width {name}")
    return values


def symmetric_relative_change(first: float, second: float) -> float:
    return abs(second - first) / max(abs(first), abs(second), 1.0e-300)


def fixed_ladder_diagnostic(
    labels: list[str], metrics: list[dict[str, float]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    common = sorted(set.intersection(*(set(item) for item in metrics)))
    for index in range(len(metrics) - 1):
        for metric in common:
            records.append(
                {
                    "comparison": f"{labels[index]}_to_{labels[index + 1]}",
                    "metric": metric,
                    "coarser": metrics[index][metric],
                    "finer": metrics[index + 1][metric],
                    "symmetric_relative_change": symmetric_relative_change(
                        metrics[index][metric], metrics[index + 1][metric]
                    ),
                    "qualification_gate": False,
                }
            )
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_logs_command(arguments: argparse.Namespace) -> int:
    mode2 = validate_adaptive_log(arguments.mode2_log, 2)
    mode3 = validate_adaptive_log(arguments.mode3_log, 3)
    mode2_lazy = None
    mode3_residual = None
    reference_mode2 = None
    reference_mode3 = None
    reference_mode3_lumi_oracle = None
    if arguments.mode2_lazy_log is not None:
        mode2_lazy = validate_adaptive_log(
            arguments.mode2_lazy_log,
            2,
            require_variable_spacing=False,
            expect_lazy_stop=True,
        )
    if arguments.mode3_residual_log is not None:
        mode3_residual = validate_adaptive_log(
            arguments.mode3_residual_log,
            3,
            # A residual rejection publishes the registered uniform-fine
            # promotion.  It may therefore publish the same refined factor in
            # every encounter even though changed-spacing solves and patch
            # attempts occurred.  Mixed selected factors are exercised by the
            # ordinary Mode-3 fixture, not required of this focused path.
            require_variable_spacing=False,
            require_full_mode3_state_coverage=False,
            required_mode3_rejection_bits=REJECTION_RESIDUAL,
        )
        refined_receipts = sum(
            count
            for factor, count in mode3_residual["selected_spacing_counts"].items()
            if int(factor) > 1
        )
        require(
            refined_receipts > 0
            and mode3_residual["mode3_uniform_promotions"] > 0
            and mode3_residual["mode3_attempted_patch_fine_cells"] > 0,
            "Mode 3 residual fixture did not solve/refine/promote a " "changed-spacing patch",
        )
    if arguments.reference_mode2_log is not None:
        reference_mode2 = validate_adaptive_log(
            arguments.reference_mode2_log, 2, require_variable_spacing=False
        )
    if arguments.reference_mode3_log is not None:
        reference_mode3 = validate_adaptive_log(
            arguments.reference_mode3_log, 3, require_variable_spacing=False
        )
        require(
            reference_mode3["patch_lumi_oracle_evaluations"] > 0
            and reference_mode3["accepted_patch_lumi_oracle_evaluations"] > 0,
            "reference Mode 3 fixture did not exercise accepted/examined "
            "patch luminosity-oracle telemetry",
        )
    if arguments.reference_mode3_lumi_oracle_log is not None:
        reference_mode3_lumi_oracle = validate_adaptive_log(
            arguments.reference_mode3_lumi_oracle_log,
            3,
            require_variable_spacing=False,
            required_mode3_rejection_bits=REJECTION_LUMI_ORACLE,
        )
        require(
            reference_mode3_lumi_oracle["patch_lumi_oracle_evaluations"] > 0,
            "tight reference Mode 3 fixture did not examine a patch " "luminosity oracle",
        )
    adaptive_logs = {"mode2": mode2, "mode3": mode3}
    if mode2_lazy is not None:
        adaptive_logs["mode2_lazy"] = mode2_lazy
    if mode3_residual is not None:
        adaptive_logs["mode3_residual"] = mode3_residual
    if reference_mode2 is not None:
        adaptive_logs["reference_mode2"] = reference_mode2
    if reference_mode3 is not None:
        adaptive_logs["reference_mode3"] = reference_mode3
    if reference_mode3_lumi_oracle is not None:
        adaptive_logs["reference_mode3_lumi_oracle"] = reference_mode3_lumi_oracle
    receipt = {
        "schema": "FEEBL_ADAPTIVE_GRID_MODES23_MECHANISM_VALIDATION_V2",
        "status": "pass",
        "qualification_scope": "adaptive_controller_and_field_operator_mechanisms",
        "physical_convergence_claim": False,
        "production_accuracy_target_checked": False,
        "adaptive_logs": adaptive_logs,
    }
    if arguments.receipt is not None:
        write_json(arguments.receipt, receipt)
    print(
        "mode-2/3 telemetry validation PASS "
        f"mode2_encounters={mode2['encounters']} "
        f"mode3_patch_requests={mode3['mode3_patch_requests']} "
        f"mode3_uniform_promotions={mode3['mode3_uniform_promotions']} "
        f"mode2_lazy_field_solves="
        f"{mode2_lazy['field_solves'] if mode2_lazy is not None else 'not-run'} "
        f"mode3_residual_mask="
        f"{mode3_residual['mode3_rejection_mask_or'] if mode3_residual is not None else 'not-run'}"
    )
    return 0


def synthetic_rejection_receipt_command(_arguments: argparse.Namespace) -> int:
    """Exercise rejection-stage parsing without tuning a physical oracle case."""
    base = {
        "nested_patch_requested": "1",
        "nested_patch_used": "0",
        "nested_patch_promoted_uniform": "1",
        "nested_patch_rejection_mask": "0",
        "nested_coarse_cells": "64",
        "nested_fine_cells": "256",
        "nested_iterations": "0",
        "nested_tail_fraction": "0",
        "nested_straddling_sources": "0",
        "nested_neff_p10": "2",
        "nested_neff_critical_min": "2",
        "nested_residual": "0",
        "nested_field_error_rms": "0",
        "nested_field_error_linf": "0",
        "noise_limited": "0",
    }

    def validate_case(label: str, replacements: dict[str, str]) -> None:
        values = dict(base)
        values.update(replacements)
        line = ENCOUNTER_PREFIX + " " + " ".join(f"{key}={value}" for key, value in values.items())
        receipt = parse_key_values(line, ENCOUNTER_PREFIX)
        context = f"synthetic {label} receipt"
        validate_nested_rejection_state(
            context=context,
            requested=integer(receipt, "nested_patch_requested", context),
            used=integer(receipt, "nested_patch_used", context),
            promoted=integer(receipt, "nested_patch_promoted_uniform", context),
            rejection_mask=integer(receipt, "nested_patch_rejection_mask", context),
            coarse_cells=integer(receipt, "nested_coarse_cells", context),
            fine_cells=integer(receipt, "nested_fine_cells", context),
            iterations=integer(receipt, "nested_iterations", context),
            tail=number(receipt, "nested_tail_fraction", context),
            tail_tolerance=0.001,
            straddlers=integer(receipt, "nested_straddling_sources", context),
            nested_neff_p10=number(receipt, "nested_neff_p10", context),
            nested_neff_critical_min=number(receipt, "nested_neff_critical_min", context),
            minimum_neff=1.0,
            residual=number(receipt, "nested_residual", context),
            residual_tolerance=0.003,
            patch_rms=number(receipt, "nested_field_error_rms", context),
            patch_linf=number(receipt, "nested_field_error_linf", context),
            error_target=0.1,
            noise_limited=integer(receipt, "noise_limited", context),
        )

    validate_case(
        "tail/straddler",
        {
            "nested_patch_rejection_mask": str(REJECTION_TAIL | REJECTION_STRADDLER),
            "nested_coarse_cells": "0",
            "nested_fine_cells": "0",
            "nested_tail_fraction": "0.02",
            "nested_straddling_sources": "7",
        },
    )
    validate_case(
        "sampling",
        {
            "nested_patch_rejection_mask": str(REJECTION_SAMPLING),
            "nested_neff_p10": "0.5",
            "noise_limited": "1",
        },
    )
    validate_case(
        "residual",
        {
            "nested_patch_rejection_mask": str(REJECTION_RESIDUAL),
            "nested_iterations": "2",
            "nested_residual": "0.01",
        },
    )
    validate_case(
        "field-oracle",
        {
            "nested_patch_rejection_mask": str(REJECTION_FIELD_ORACLE),
            "nested_iterations": "2",
            "nested_residual": "1e-6",
            "nested_field_error_rms": "0.12",
            "nested_field_error_linf": "0.15",
        },
    )
    validate_case(
        "luminosity-oracle",
        {
            "nested_patch_rejection_mask": str(REJECTION_LUMI_ORACLE),
            "nested_iterations": "2",
            "nested_residual": "1e-6",
            "nested_field_error_rms": "0.02",
            "nested_field_error_linf": "0.03",
        },
    )

    summary = parse_key_values(
        SUMMARY_PREFIX + " maximum_patch_field_error=0.15",
        SUMMARY_PREFIX,
    )
    attempted_field_error = number(
        summary, "maximum_patch_field_error", "synthetic summary receipt"
    )
    require(
        attempted_field_error > 0.1, "synthetic field-oracle receipt does not exceed its target"
    )
    require_attempted_maximum(
        attempted_field_error,
        0.15,
        "synthetic summary receipt",
        "field error",
        relative_tolerance=1.0e-5,
    )

    try:
        validate_case(
            "missing field-oracle bit",
            {
                "nested_iterations": "2",
                "nested_residual": "1e-6",
                "nested_field_error_linf": "0.15",
            },
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("synthetic over-target field oracle passed without rejection bit")
    print("synthetic Mode-3 rejection receipt validation PASS")
    return 0


def mechanism_command(arguments: argparse.Namespace) -> int:
    fixed_paths = [arguments.fixed64, arguments.fixed128, arguments.fixed256]
    fixed_metrics = [read_metrics(path) for path in fixed_paths]
    mode2_metrics = read_metrics(arguments.mode2_out)
    mode3_metrics = read_metrics(arguments.mode3_out)
    mode2_log = validate_adaptive_log(arguments.mode2_log, 2)
    mode3_log = validate_adaptive_log(arguments.mode3_log, 3)
    ladder = fixed_ladder_diagnostic(["fixed64", "fixed128", "fixed256"], fixed_metrics)
    worst = max(
        (float(record["symmetric_relative_change"]) for record in ladder),
        default=0.0,
    )
    receipt = {
        "schema": "FEEBL_ADAPTIVE_GRID_MODES23_MECHANISM_VALIDATION_V2",
        "status": "pass",
        "qualification_scope": "adaptive_controller_and_field_operator_mechanisms",
        "physical_convergence_claim": False,
        "production_accuracy_target_checked": False,
        "accuracy_target_note": (
            "The FeeBL 1--3% production-physics target requires a separately "
            "resolved, statistically controlled convergence campaign."
        ),
        "fixed_grid_ladder": {
            "role": "diagnostic_only_not_a_pass_fail_accuracy_gate",
            "outputs": {
                label: values
                for label, values in zip(("fixed64", "fixed128", "fixed256"), fixed_metrics)
            },
            "adjacent_changes": ladder,
            "worst_symmetric_relative_change": worst,
        },
        "adaptive_operator_outputs": {
            "mode2": mode2_metrics,
            "mode3": mode3_metrics,
        },
        "adaptive_logs": {"mode2": mode2_log, "mode3": mode3_log},
    }
    write_json(arguments.receipt, receipt)
    print(
        "mode-2/3 mechanism/operator validation PASS "
        f"physical_convergence_claim=false "
        f"fixed_ladder_worst_change={worst:.8g} receipt={arguments.receipt}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    logs = subparsers.add_parser(
        "validate-logs", help="validate Mode-2/3 encounter and summary telemetry"
    )
    logs.add_argument("mode2_log", type=Path)
    logs.add_argument("mode3_log", type=Path)
    logs.add_argument("receipt", type=Path, nargs="?")
    logs.add_argument("--mode2-lazy-log", type=Path)
    logs.add_argument("--mode3-residual-log", type=Path)
    logs.add_argument("--reference-mode2-log", type=Path)
    logs.add_argument("--reference-mode3-log", type=Path)
    logs.add_argument("--reference-mode3-lumi-oracle-log", type=Path)
    logs.set_defaults(function=validate_logs_command)

    rejection_receipts = subparsers.add_parser(
        "rejection-receipt-self-test",
        help="validate synthetic staged Mode-3 rejection receipts",
    )
    rejection_receipts.set_defaults(function=synthetic_rejection_receipt_command)

    mechanism = subparsers.add_parser(
        "mechanism", help="run the non-convergence mechanism/operator comparison"
    )
    mechanism.add_argument("fixed64", type=Path)
    mechanism.add_argument("fixed128", type=Path)
    mechanism.add_argument("fixed256", type=Path)
    mechanism.add_argument("mode2_out", type=Path)
    mechanism.add_argument("mode3_out", type=Path)
    mechanism.add_argument("mode2_log", type=Path)
    mechanism.add_argument("mode3_log", type=Path)
    mechanism.add_argument("receipt", type=Path)
    mechanism.set_defaults(function=mechanism_command)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    return int(arguments.function(arguments))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # pragma: no cover - command-line diagnostic
        print(f"mode-2/3 mechanism validation FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
