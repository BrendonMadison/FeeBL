#!/usr/bin/env python3
"""Materialize the final coupled FCC-Z pilot request after the last rebuild.

This generator deliberately refuses a FeeBL binary older than any compilation
input under ``src``.  It never freezes or executes a campaign.  Run it only
after the final binary and source review are complete; then pass the emitted
request to ``coupled_grid_root_campaign.py freeze`` as a separate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


CAMPAIGN_DIRECTORY = Path(__file__).resolve().parent
SOURCE_ROOT = CAMPAIGN_DIRECTORY.parents[1]
DEFAULT_BINARY = SOURCE_ROOT / "build" / "pilot-review" / "src" / "feebl"
DEFAULT_REQUEST = CAMPAIGN_DIRECTORY / "coupled_pilot_request.json"
DEFAULT_LEGACY_REQUEST = CAMPAIGN_DIRECTORY / "fixed_legacy_diagnostic_request.json"
DEFAULT_REPLAY_SMOKE_REQUEST = CAMPAIGN_DIRECTORY / "reference_cic_replay_smoke_request.json"
DEFAULT_SOURCE_STATE = CAMPAIGN_DIRECTORY / "coupled_pilot_source_state.json"
GAUSSIAN_TRANSFORM = (
    SOURCE_ROOT / "examples" / "adaptive_particles" / "gaussian_core_spectrum.py"
)
CAMPAIGN_DRIVER = (
    SOURCE_ROOT / "examples" / "adaptive_particles" / "coupled_grid_root_campaign.py"
)
REPLAY_DIRECTORY = SOURCE_ROOT / "examples" / "adaptive_particles"
if str(REPLAY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPLAY_DIRECTORY))
import adaptive_root_replay as replay  # noqa: E402

# This block is also enforced by frozen_eight_campaign.py.  Keeping the
# preregistered unseen population in its own finite namespace prevents a seed
# that informed an exploratory/diagnostic campaign from later being relabeled
# as unseen evidence.
FROZEN_UNSEEN_SEEDS = tuple(range(5001, 5065))
SEED_POOL_KEYS = ("pilot_seeds", "extension_seeds", "holdout_seeds")
MODE_SPECIFIC_PRIMARY_ASSIGNMENTS = frozenset(
    {
        "adaptive_grid_mode",
        "adaptive_grid_terminal_patch_certification",
        "adaptive_grid_max_levels",
        "adaptive_grid_max_patches",
        "adaptive_grid_refinement_ratio",
        "adaptive_grid_blocking_factor",
        "adaptive_grid_error_buffer_cells",
        "adaptive_grid_transition_cells",
        "adaptive_grid_fine_residual_tolerance",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def checked_file(path: Path, label: str) -> Path:
    result = path.resolve()
    if not result.is_file() or result.stat().st_size == 0:
        raise RuntimeError(f"{label} is missing or empty: {result}")
    return result


def parse_card_assignments(
    path: Path,
    label: str,
    *,
    block_kind: str = "PARAMETERS",
    block_name: str = "p_fccz",
) -> dict[str, str]:
    """Parse exact FeeBL assignments; never authenticate by substring."""
    text = path.read_text(encoding="utf-8")
    try:
        block = replay.selected_block(text, block_kind, block_name)
        parsed = replay.raw_assignments(text, block)
    except replay.ReplayError as exc:
        raise RuntimeError(f"{label}: {exc}") from exc
    return {item.key: item.value for item in parsed}


def require_card_assignments(
    path: Path,
    expected: Mapping[str, str],
    label: str,
    *,
    block_kind: str = "PARAMETERS",
    block_name: str = "p_fccz",
) -> dict[str, str]:
    assignments = parse_card_assignments(path, label, block_kind=block_kind, block_name=block_name)
    for key, wanted in expected.items():
        if key not in assignments:
            raise RuntimeError(f"{label} is missing required assignment {key!r}")
        actual = assignments[key]
        if actual != wanted:
            raise RuntimeError(
                f"{label} assignment {key!r}: expected exactly {wanted!r}, found {actual!r}"
            )
    return assignments


def require_cross_template_invariants(
    reference: Mapping[str, str], candidate: Mapping[str, str], label: str
) -> list[str]:
    """Require every non-mode assignment to be present and lexically equal."""

    def is_mode_specific(key: str) -> bool:
        return key.rsplit("::", 1)[-1] in MODE_SPECIFIC_PRIMARY_ASSIGNMENTS

    reference_nonmode = {key for key in reference if not is_mode_specific(key)}
    candidate_nonmode = {key for key in candidate if not is_mode_specific(key)}
    if reference_nonmode != candidate_nonmode:
        missing = sorted(reference_nonmode - candidate_nonmode)
        extra = sorted(candidate_nonmode - reference_nonmode)
        raise RuntimeError(
            f"{label} non-mode assignment-set mismatch: missing={missing}, extra={extra}"
        )
    mismatches = sorted(key for key in reference_nonmode if reference[key] != candidate[key])
    if mismatches:
        raise RuntimeError(
            f"{label} non-mode assignments differ: "
            + ", ".join(f"{key}={reference[key]!r}/{candidate[key]!r}" for key in mismatches)
        )
    return sorted(reference_nonmode)


def request_seed_namespace(request: Mapping[str, Any], label: str) -> set[int]:
    populations: list[set[int]] = []
    for key in SEED_POOL_KEYS:
        raw = request.get(key)
        if not isinstance(raw, list) or not raw:
            raise RuntimeError(f"{label}.{key} must be a nonempty seed array")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 1 or seed > 2_147_483_647
            for seed in raw
        ):
            raise RuntimeError(f"{label}.{key} contains an invalid seed")
        population = set(raw)
        if len(population) != len(raw):
            raise RuntimeError(f"{label}.{key} contains duplicate seeds")
        populations.append(population)
    for left in range(len(populations)):
        for right in range(left + 1, len(populations)):
            overlap = populations[left] & populations[right]
            if overlap:
                raise RuntimeError(f"{label} seed populations overlap: {sorted(overlap)}")
    return set().union(*populations)


def assert_globally_disjoint_seed_namespaces(
    primary: Mapping[str, Any],
    legacy: Mapping[str, Any],
    replay_smoke: Mapping[str, Any],
) -> dict[str, list[int]]:
    namespaces = {
        "primary": request_seed_namespace(primary, "primary"),
        "legacy": request_seed_namespace(legacy, "legacy"),
        "replay_smoke": request_seed_namespace(replay_smoke, "replay_smoke"),
        "frozen_preregistered_unseen": set(FROZEN_UNSEEN_SEEDS),
    }
    names = list(namespaces)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            overlap = namespaces[names[left]] & namespaces[names[right]]
            if overlap:
                raise RuntimeError(
                    "global seed namespaces overlap between "
                    f"{names[left]} and {names[right]}: {sorted(overlap)}"
                )
    return {name: sorted(values) for name, values in namespaces.items()}


def compilation_inputs() -> list[Path]:
    result = [path.resolve() for path in (SOURCE_ROOT / "src").rglob("*") if path.is_file()]
    for relative in ("CMakeLists.txt", "src/CMakeLists.txt", "examples/CMakeLists.txt"):
        path = SOURCE_ROOT / relative
        if path.is_file():
            result.append(path.resolve())
    return sorted(set(result), key=lambda item: item.relative_to(SOURCE_ROOT).as_posix())


def source_revision() -> str:
    """Read this checkout's commit or the explicit source-export identity.

    An enclosing repository is unrelated provenance. A content digest is not a
    Git commit; compilation-input hashes separately bind the materialized files.
    """
    root = SOURCE_ROOT.resolve()

    def git_value(argument: str) -> str:
        try:
            process = subprocess.run(
                ["git", "-C", str(root), "rev-parse", argument],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                shell=False,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return process.stdout.strip() if process.returncode == 0 else ""

    git_root = git_value("--show-toplevel")
    if git_root and Path(git_root).resolve() == root:
        value = git_value("HEAD")
    else:
        revision = root / "SOURCE_REVISION"
        if revision.is_symlink() or not revision.is_file():
            raise RuntimeError("source export requires a regular SOURCE_REVISION file")
        value = revision.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"(?:[0-9a-f]{40}(?:-dirty)?|sha256:[0-9a-f]{64})", value) is None:
        raise RuntimeError(
            "source revision must be a lowercase commit id with optional -dirty "
            "suffix or a sha256 content identity"
        )
    return value


def git_head() -> str | None:
    """Return a base Git commit, or None for a content-identified source export."""
    revision = source_revision()
    return None if revision.startswith("sha256:") else revision[:40]


def write_exclusive(path: Path, value: Any) -> None:
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build_requests(
    binary: Path, python_executable: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    binary = checked_file(binary, "FeeBL binary")
    python_executable = checked_file(python_executable, "Python executable")
    gaussian_transform = checked_file(GAUSSIAN_TRANSFORM, "Gaussian-core transform")
    checked_file(CAMPAIGN_DRIVER, "coupled campaign driver")
    inputs = compilation_inputs()
    if not inputs:
        raise RuntimeError("no compilation inputs were found")
    newest_input = max(path.stat().st_mtime_ns for path in inputs)
    if binary.stat().st_mtime_ns < newest_input:
        newest_paths = [
            path.relative_to(SOURCE_ROOT).as_posix()
            for path in inputs
            if path.stat().st_mtime_ns == newest_input
        ]
        raise RuntimeError(
            "FeeBL binary predates compilation input(s) "
            + ", ".join(newest_paths)
            + "; rebuild before materializing the request"
        )

    source_files = [
        {
            "path": path.relative_to(SOURCE_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in inputs
    ]
    revision = source_revision()
    content_export = revision.startswith("sha256:")
    source_state: dict[str, Any] = {
        "schema": "feebl.coupled-pilot-source-state.v1",
        "schema_version": "1",
        "status": (
            "content_identified_source_export" if content_export else
            "exact_dirty_worktree_snapshot_not_commit_clean_claim"
        ),
        "source_revision": revision,
        "git_head": None if content_export else revision[:40],
        "source_export_sha256": revision[7:] if content_export else None,
        "source_root": str(SOURCE_ROOT),
        "compilation_input_count": str(len(source_files)),
        "compilation_inputs": source_files,
        "feebl_binary": {
            "path": str(binary),
            "sha256": sha256_file(binary),
        },
        "gaussian_core_transform": {
            "path": str(gaussian_transform),
            "sha256": sha256_file(gaussian_transform),
        },
        "python_executable": {
            "path": str(python_executable),
            "sha256": sha256_file(python_executable),
            "version": sys.version,
        },
        "request_generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "campaign_driver": {
            "path": str(CAMPAIGN_DRIVER.resolve()),
            "sha256": sha256_file(CAMPAIGN_DRIVER.resolve()),
        },
        "scientific_boundary": [
            "primary_arms_qualify_adaptive_grid_modes_2_and_3_only",
            "adaptive_macroparticle_mode_is_zero_in_all_primary_arms",
            "mode2_fixed_width_is_the_equal_discretization_reference_for_mode3_fixed_width",
            "fixed_legacy_is_a_separate_diagnostic_not_a_method_equivalence_oracle",
            "reference_cic_root_replay_requires_a_separate_sensitivity_campaign",
            "gaussian_core_is_a_hash_bound_postprocess_not_global_rms_W",
            "cpu_threads_7_selected_by_a_separate_2_vs_7_sequential_probe",
        ],
    }

    template_directory = CAMPAIGN_DIRECTORY
    template_specs = [
        (
            "mode2_fixed_width",
            template_directory / "mode2_fixed_width_cpu.dat",
            "base_grid_and_adaptive_maxima_v1",
            "fccz",
            "p_fccz",
        ),
        (
            "mode3_fixed_width",
            template_directory / "mode3_fixed_width_cpu.dat",
            "base_grid_and_adaptive_maxima_v1",
            "fccz",
            "p_fccz",
        ),
    ]
    templates = []
    primary_card_assignments: dict[str, dict[str, str]] = {}
    for template_id, path, contract, accelerator, parameters in template_specs:
        checked_file(path, f"template {template_id}")
        expected_mode = "2" if template_id.startswith("mode2") else "3"
        assignments = require_card_assignments(
            path,
            {
                "compute_backend": '"cpu"',
                "cpu_threads": "7",
                "adaptive_grid_mode": expected_mode,
                "adaptive_grid_field_spacing_classes": '"1,2,4"',
                "adaptive_grid_field_operator": '"fd4_lagrange4_v1"',
                "adaptive_grid_max_refinement_factor": "4",
                "adaptive_grid_source_model": '"fixed_width_bspline1_v1"',
                "adaptive_grid_source_width_x": "714.195",
                "adaptive_grid_source_width_y": "5.80354",
                "adaptive_grid_pilot_diagnostics": "1",
                "adaptive_grid_fail_on_unmet_error": "0",
                "adaptive_macroparticle_mode": "0",
            },
            f"template {template_id}",
            block_kind="PARAMETERS",
            block_name=parameters,
        )
        accelerator_assignments = require_card_assignments(
            path,
            {"energy": "45.6", "sigma_z": "226.562500"},
            f"template {template_id}",
            block_kind="ACCELERATOR",
            block_name=accelerator,
        )
        primary_card_assignments[template_id] = {
            **{f"ACCELERATOR::{key}": value for key, value in accelerator_assignments.items()},
            **{f"PARAMETERS::{key}": value for key, value in assignments.items()},
        }
        if template_id == "mode3_fixed_width":
            require_card_assignments(
                path,
                {"adaptive_grid_terminal_patch_certification": "1"},
                "Mode-3 terminal-patch candidate template",
                block_kind="PARAMETERS",
                block_name=parameters,
            )
        templates.append(
            {
                "id": template_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "accelerator": accelerator,
                "parameters": parameters,
                "grid_patch_contract": contract,
            }
        )
    invariant_keys = require_cross_template_invariants(
        primary_card_assignments["mode2_fixed_width"],
        primary_card_assignments["mode3_fixed_width"],
        "Mode-2/Mode-3 primary templates",
    )

    request: dict[str, Any] = {
        "schema": "feebl.coupled-grid-root-campaign-request.v1",
        "reference_template_id": "mode2_fixed_width",
        "templates": templates,
        "bindings": {
            "feebl_binary_path": str(binary),
            "feebl_binary_sha256": sha256_file(binary),
            "source_state_path": str(DEFAULT_SOURCE_STATE.resolve()),
            "source_state_sha256": "<filled-after-source-state-serialization>",
            "generator_transform": {
                "id": "feebl_legacy_mc_complete_run_v1",
                "version": "fresh_complete_process_common_seed_factorial_v1",
            },
            "physics_family": {
                "id": "fccz-short-bunch-fixed-width-mode2-vs-mode3-grid-pilot-v1",
                "enabled_channels": [
                    "beam_beam_fields",
                    "beamstrahlung_energy_loss",
                    "crab_crossing",
                    "crab_waist",
                    "ip_dispersion_monochromatization",
                    "lumi_ee",
                    "gaussian_core_postprocess",
                ],
                "adaptive_macroparticle_mode": "0",
                "qualification_scope": (
                    "mode2_vs_mode3_adaptive_grid_with_identical_fixed_width_source"
                ),
                "method_equivalence_reference_rationale": (
                    "mode2_and_mode3_share_the_same_source_kernel_spacing_classes_"
                    "refinement_cap_and_base_grid"
                ),
                "cross_template_assignment_contract": {
                    "version": "exact_non_mode_assignments_v1",
                    "allowed_mode_specific_keys": sorted(MODE_SPECIFIC_PRIMARY_ASSIGNMENTS),
                    "equal_non_mode_assignment_count": str(len(invariant_keys)),
                    "equal_non_mode_assignment_keys_sha256": hashlib.sha256(
                        canonical_bytes(invariant_keys)
                    ).hexdigest(),
                },
                "cpu_thread_selector": {
                    "selected": "7",
                    "probe_elapsed_seconds_threads_2": "25.53",
                    "probe_elapsed_seconds_threads_7": "23.13",
                    "speedup_threads_7_over_2": "1.104",
                    "peak_rss_kib_threads_2": "491472",
                    "peak_rss_kib_threads_7": "894896",
                    "spectrum_sha256_equal": "1",
                    "lumi_endpoints_equal": "1",
                    "scope": "mode2_fixed_width_base64_nz16_nm65536_sequential_probe",
                },
            },
            "gaussian_core_transform": {
                "id": "feebl_gaussian_core_spectrum_v1",
                "path": str(gaussian_transform),
                "sha256": sha256_file(gaussian_transform),
                "python_executable_path": str(python_executable),
                "python_executable_sha256": sha256_file(python_executable),
            },
        },
        "grid_levels": [
            {"id": "g64", "n_x": 64, "n_y": 64},
            {"id": "g128", "n_x": 128, "n_y": 128},
        ],
        "n_m_levels": [65536, 131072],
        "pilot_seeds": [17, 19, 23, 29],
        "extension_seeds": list(range(1001, 1033)),
        "holdout_seeds": [31, 37, 41, 43] + list(range(2001, 2029)),
        "endpoints": [
            {
                "id": "lumi_ee",
                "file": "out.dat",
                "format": "assignment",
                "key": "lumi_ee",
                "scale": "3e34",
                "target_relative_error": "0.01",
                "equivalence_margin_relative": "0.01",
                "expected_true_difference_relative": "0",
                "units": "m^-2_per_bunch_crossing",
            },
            {
                "id": "lumi_ee_high",
                "file": "out.dat",
                "format": "assignment",
                "key": "lumi_ee_high",
                "scale": "3e34",
                "target_relative_error": "0.01",
                "equivalence_margin_relative": "0.01",
                "expected_true_difference_relative": "0",
                "units": "m^-2_per_bunch_crossing",
            },
            {
                "id": "gaussian_core_sigma",
                "file": "gaussian_core.json",
                "format": "json_scalar",
                "key": "core_fit.sigma_core_GeV",
                "scale": "0.1",
                "target_relative_error": "0.01",
                "equivalence_margin_relative": "0.01",
                "expected_true_difference_relative": "0",
                "units": "GeV",
            },
            {
                "id": "gaussian_core_area",
                "file": "gaussian_core.json",
                "format": "json_scalar",
                "key": "core_fit.gaussian_core_area",
                "scale": "2e34",
                "target_relative_error": "0.01",
                "equivalence_margin_relative": "0.01",
                "expected_true_difference_relative": "0",
                "units": "m^-2_per_bunch_crossing",
            },
        ],
        "statistics": {
            "standard_error_safety_factor": "1.5",
            "confidence_level": "0.95",
            "equivalence_alpha": "0.05",
            "equivalence_power": "0.9",
            "conservative_total_budget_multiplier": "3",
        },
        "next_stage": {
            "grid": {"id": "g256", "n_x": 256, "n_y": 256},
            "n_m": 262144,
        },
        "limits": {
            "maximum_grid_cells": 65536,
            "maximum_n_m": 262144,
            "maximum_pilot_runs": 512,
            "maximum_wall_seconds_per_run": 21600,
        },
    }
    legacy_template = template_directory / "fixed_cpu.dat"
    legacy_accelerator = "fccz"
    legacy_parameters = "p_fccz"
    checked_file(legacy_template, "fixed legacy diagnostic template")
    require_card_assignments(
        legacy_template,
        {"cpu_threads": "7", "adaptive_grid_mode": "0", "n_z": "16"},
        "fixed legacy diagnostic template",
        block_kind="PARAMETERS",
        block_name=legacy_parameters,
    )
    require_card_assignments(
        legacy_template,
        {"energy": "45.6", "sigma_z": "226.562500"},
        "fixed legacy diagnostic template",
        block_kind="ACCELERATOR",
        block_name=legacy_accelerator,
    )
    legacy_request = json.loads(json.dumps(request))
    legacy_request["reference_template_id"] = "fixed_legacy_diagnostic"
    legacy_request["templates"] = [
        {
            "id": "fixed_legacy_diagnostic",
            "path": str(legacy_template.resolve()),
            "sha256": sha256_file(legacy_template),
            "accelerator": legacy_accelerator,
            "parameters": legacy_parameters,
            "grid_patch_contract": "base_grid_only_v1",
        }
    ]
    legacy_request["bindings"]["physics_family"] = {
        "id": "fccz-short-bunch-fixed-legacy-diagnostic-factorial-v1",
        "enabled_channels": request["bindings"]["physics_family"]["enabled_channels"],
        "adaptive_macroparticle_mode": "0",
        "qualification_scope": "diagnostic_baseline_only_not_adaptive_method_equivalence",
        "fixed_next_sentinel": "g256_n_m262144_requires_separate_execute_next",
        "cpu_thread_selector": request["bindings"]["physics_family"]["cpu_thread_selector"],
    }
    legacy_request["pilot_seeds"] = [3001, 3002, 3003, 3004]
    legacy_request["extension_seeds"] = list(range(3101, 3133))
    legacy_request["holdout_seeds"] = list(range(3201, 3233))

    smoke_template = template_directory / "reference_cic_replay_smoke_mode2_cpu.dat"
    smoke_accelerator = "fccz"
    smoke_parameters = "p_fccz"
    checked_file(smoke_template, "reference-CIC replay integration smoke template")
    require_card_assignments(
        smoke_template,
        {
            "compute_backend": '"cpu"',
            "cpu_threads": "7",
            "adaptive_grid_mode": "2",
            "adaptive_grid_size_classes_x": '"8,16"',
            "adaptive_grid_size_classes_y": '"8,16"',
            "adaptive_grid_field_spacing_classes": '"1,2,4"',
            "adaptive_grid_field_operator": '"fd4_lagrange4_v1"',
            "adaptive_grid_max_refinement_factor": "4",
            "adaptive_grid_source_model": '"reference_cic_v1"',
            "adaptive_grid_source_reference_nx": "64",
            "adaptive_grid_source_reference_ny": "64",
            "adaptive_grid_error_norm_policy": '"vector_rms_quantile_v1"',
            "adaptive_grid_error_quantile": "0.999",
            "adaptive_grid_error_catastrophe_limit": "0.5",
            "adaptive_macroparticle_mode": "2",
            "adaptive_macroparticle_fail_on_unmet_error": "0",
            "n_x": "16",
            "n_y": "16",
            "n_z": "2",
            "n_t": "1",
            "n_m": "512",
            "num_lumi": "50000",
        },
        "reference-CIC replay integration smoke template",
        block_kind="PARAMETERS",
        block_name=smoke_parameters,
    )
    require_card_assignments(
        smoke_template,
        {"energy": "45.6", "sigma_z": "226.562500"},
        "reference-CIC replay integration smoke template",
        block_kind="ACCELERATOR",
        block_name=smoke_accelerator,
    )
    replay_smoke_request = json.loads(json.dumps(request))
    replay_smoke_request["reference_template_id"] = "reference_cic_replay_smoke"
    replay_smoke_request["templates"] = [
        {
            "id": "reference_cic_replay_smoke",
            "path": str(smoke_template.resolve()),
            "sha256": sha256_file(smoke_template),
            "accelerator": smoke_accelerator,
            "parameters": smoke_parameters,
            "grid_patch_contract": "base_grid_and_adaptive_maxima_v1",
        }
    ]
    replay_smoke_request["bindings"].pop("gaussian_core_transform")
    replay_smoke_request["bindings"]["physics_family"] = {
        "id": "fccz-reference-cic-root-replay-integration-smoke-v1",
        "enabled_channels": ["beam_beam_fields", "lumi_ee", "runtime_replay_signal_v2"],
        "adaptive_macroparticle_mode": "2",
        "electron_lumi_sampling": "3",
        "num_lumi_capacity": "50000",
        "qualification_scope": "software_integration_only_not_physics_qualification",
        "cpu_threads": "7",
    }
    replay_smoke_request["grid_levels"] = [
        {"id": "g16", "n_x": 16, "n_y": 16},
        {"id": "g32", "n_x": 32, "n_y": 32},
    ]
    replay_smoke_request["n_m_levels"] = [512, 1024]
    replay_smoke_request["pilot_seeds"] = [4001, 4002, 4003, 4004]
    replay_smoke_request["extension_seeds"] = [4101, 4102, 4103, 4104]
    replay_smoke_request["holdout_seeds"] = [4201, 4202, 4203, 4204]
    replay_smoke_request["endpoints"] = [request["endpoints"][0]]
    replay_smoke_request["next_stage"] = {
        "grid": {"id": "g64", "n_x": 64, "n_y": 64},
        "n_m": 2048,
    }
    replay_smoke_request["limits"] = {
        "maximum_grid_cells": 4096,
        "maximum_n_m": 2048,
        "maximum_pilot_runs": 128,
        "maximum_wall_seconds_per_run": 600,
    }
    seed_namespaces = assert_globally_disjoint_seed_namespaces(
        request, legacy_request, replay_smoke_request
    )
    source_state["seed_namespace_registry"] = {
        "schema": "feebl.campaign-seed-namespaces.v1",
        "globally_pairwise_disjoint": True,
        "namespaces": seed_namespaces,
        "frozen_preregistered_unseen_policy": ("only_seed_ids_5001_through_5064_are_eligible"),
    }
    return source_state, request, legacy_request, replay_smoke_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize, but do not freeze or execute, the final FCC-Z coupled pilot request."
    )
    parser.add_argument("--feebl-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--request-output", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--legacy-request-output", type=Path, default=DEFAULT_LEGACY_REQUEST)
    parser.add_argument(
        "--replay-smoke-request-output",
        type=Path,
        default=DEFAULT_REPLAY_SMOKE_REQUEST,
    )
    parser.add_argument("--source-state-output", type=Path, default=DEFAULT_SOURCE_STATE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    request_output = args.request_output.resolve()
    legacy_output = args.legacy_request_output.resolve()
    replay_smoke_output = args.replay_smoke_request_output.resolve()
    source_output = args.source_state_output.resolve()
    output_paths = [
        request_output,
        legacy_output,
        replay_smoke_output,
        source_output,
    ]
    if len(set(output_paths)) != len(output_paths):
        raise RuntimeError("all request and source-state outputs must differ")
    if any(path.exists() for path in output_paths):
        raise RuntimeError("refusing to overwrite an existing request or source-state artifact")
    if source_output != DEFAULT_SOURCE_STATE.resolve():
        raise RuntimeError(
            "v1 request uses the canonical campaign-local source-state path; " "do not relocate it"
        )
    source_state, request, legacy_request, replay_smoke_request = build_requests(
        args.feebl_binary, args.python_executable
    )
    source_payload = canonical_bytes(source_state) + b"\n"
    request["bindings"]["source_state_sha256"] = hashlib.sha256(source_payload).hexdigest()
    legacy_request["bindings"]["source_state_sha256"] = request["bindings"]["source_state_sha256"]
    replay_smoke_request["bindings"]["source_state_sha256"] = request["bindings"][
        "source_state_sha256"
    ]
    write_exclusive(source_output, source_state)
    try:
        write_exclusive(request_output, request)
        write_exclusive(legacy_output, legacy_request)
        write_exclusive(replay_smoke_output, replay_smoke_request)
    except Exception:
        for path in output_paths:
            path.unlink(missing_ok=True)
        raise
    print(request_output)
    print(legacy_output)
    print(replay_smoke_output)
    print(source_output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"request generation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
