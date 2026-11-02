#!/usr/bin/env python3
"""Exact standard-library tests for the v2 campaign/statistics foundation."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRIVER = (
    ROOT
    / "examples" / "adaptive_particles" / "coupled_grid_root_campaign_v2.py"
)
SPEC = importlib.util.spec_from_file_location("coupled_grid_root_campaign_v2", DRIVER)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def sealed_request(
    *,
    tuning_count: int = 4,
    confirmation_count: int = 4,
    holdout_count: int = 120,
    extension_count: int = 120,
    distribution_kind: str = "continuous",
    include_complete_ledger: bool = True,
    blinded_power_contract: dict | None = None,
) -> dict:
    target = Decimal("1")
    computed = {
        "mesh_gci",
        "source_root",
        "grid_root_interaction",
        "outer_sampling",
    }
    budget = []
    for component in campaign.REQUIRED_UNCERTAINTY_COMPONENTS:
        if not include_complete_ledger and component == "external_reference":
            continue
        entry = {
            "component": component,
            "allocation": "0.1" if component in computed else "0.05",
        }
        if component not in computed:
            entry.update(
                {
                    "status": "resolved",
                    "upper_bound": "0.001",
                    "method": f"prospective_{component}_bound_v1",
                    "evidence": f"sha256-bound fixture evidence for {component}",
                }
            )
        budget.append(entry)
    statistics = {
        "familywise_alpha": "0.05",
        "desired_power": "0.9",
        "multiplicity_method": "bonferroni_v2",
        "minimum_heldout_seed_blocks": "100",
        "gci_safety_factor": "1.25",
        "minimum_order": "0.25",
        "maximum_order": "8",
        "confirmation_order_tolerance": "0.6",
        "asymptotic_ratio_min": "0.5",
        "asymptotic_ratio_max": "2",
    }
    if blinded_power_contract is not None:
        statistics["blinded_power_contract"] = blinded_power_contract
    unsigned = {
        "schema": campaign.REQUEST_SCHEMA_V2,
        "schema_version": "2",
        "qualification_status": "preregistered_v2_foundation_only",
        "campaign_id": "fccz_short_sign_plus",
        "collider_family": "fccz",
        "methods": [
            {"id": "uniform_reference", "role": "reference"},
            {"id": "adaptive_candidate", "role": "candidate"},
        ],
        "design": {
            "pilot_grid_levels": [
                {"id": "g1", "n_x": "64", "n_y": "64", "h": "4"},
                {"id": "g2", "n_x": "128", "n_y": "128", "h": "2"},
                {"id": "g3", "n_x": "256", "n_y": "256", "h": "1"},
            ],
            "confirmation_grid_level": {
                "id": "g4",
                "n_x": "512",
                "n_y": "512",
                "h": "0.5",
            },
            "pilot_root_levels": [
                {"id": "N1", "n_m": "100"},
                {"id": "N2", "n_m": "400"},
                {"id": "N3", "n_m": "1600"},
            ],
            "confirmation_root_level": {"id": "N4", "n_m": "6400"},
            "root_prefix_contract": "nested_initial_root_prefix_v2",
            "fixed_physical_contract": {
                "domain_sha256": "a" * 64,
                "source_shape_sha256": "b" * 64,
                "collision_lattice_sha256": "c" * 64,
                "longitudinal_time_sha256": "d" * 64,
                "fit_contract_sha256": "e" * 64,
            },
        },
        "seed_pools": {
            "tuning": [str(1000 + index) for index in range(tuning_count)],
            "confirmation": [
                str(2000 + index) for index in range(confirmation_count)
            ],
            "extension": [str(3000 + index) for index in range(extension_count)],
            "holdout": [str(10000 + index) for index in range(holdout_count)],
        },
        "endpoints": [
            {
                "id": "lumi_ee",
                "units": "arb",
                "distribution_kind": distribution_kind,
                "scale_contract": {
                    "reference_value": "100",
                    "near_zero_floor": "1",
                    "scale_basis": "100",
                    "target_relative": "0.01",
                    "absolute_target": str(target),
                    "reference_value_provenance": "frozen uniform-fine oracle fixture",
                    "near_zero_floor_provenance": "prospective physical sensitivity fixture",
                },
                "equivalence_margin_relative": "0.002",
                "expected_true_difference_relative": "0",
                "uncertainty_budget": budget,
            }
        ],
        "claims": [
            {
                "id": "absolute_lumi_convergence",
                "endpoint_id": "lumi_ee",
                "family_id": "fccz_primary",
                "kind": "absolute_convergence",
                "reference_method_id": "uniform_reference",
            },
            {
                "id": "adaptive_lumi_equivalence",
                "endpoint_id": "lumi_ee",
                "family_id": "fccz_primary",
                "kind": "method_equivalence",
                "candidate_method_id": "adaptive_candidate",
                "reference_method_id": "uniform_reference",
            }
        ],
        "statistics": statistics,
    }
    return campaign.seal_artifact(unsigned)


def convergent_value(realization: dict, endpoint_id: str) -> Decimal:
    del endpoint_id
    h = Decimal(realization["h"])
    n_m = Decimal(realization["n_m"])
    root_term = Decimal("4") / n_m.sqrt()
    interaction = Decimal("0.05") * h * h / n_m.sqrt()
    method_bias = (
        Decimal("-0.01")
        if realization["method_id"] == "adaptive_candidate"
        else Decimal("0")
    )
    seed_noise = Decimal(int(realization["seed"]) % 4 - 1) * Decimal("0.0002")
    return (
        Decimal("100")
        + method_bias
        + Decimal("0.1") * h * h
        + root_term
        + interaction
        + seed_noise
    )


def elapsed(realization: dict) -> int:
    base = 2_000_000 if realization["method_id"] == "uniform_reference" else 1_000_000
    return base + int(realization["seed"]) % 17


def manifest_and_state(request: dict, value_function=convergent_value):
    manifest = campaign.materialize_pilot_manifest_v2(request)
    state = campaign.make_complete_state_v2_for_testing(
        request, manifest, value_function, elapsed_function=elapsed
    )
    return manifest, state


class SchemaAndAuthenticationTests(unittest.TestCase):
    def test_v2_schemas_are_parallel_and_foundation_is_hard_blocked(self):
        request = sealed_request()
        manifest, state = manifest_and_state(request)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        plan = campaign.compute_next_stage_plan_v2(request, manifest, state)
        self.assertTrue(request["schema"].endswith(".v2"))
        self.assertTrue(manifest["schema"].endswith(".v2"))
        self.assertTrue(analysis["schema"].endswith(".v2"))
        self.assertFalse(analysis["physics_qualification_passed"])
        self.assertFalse(plan["execution_authorized"])
        self.assertEqual(
            analysis["qualification_blocker"], campaign.FOUNDATION_QUALIFICATION_BLOCKER
        )

    def test_three_pilot_levels_and_separate_finer_confirmation_are_mandatory(self):
        request = sealed_request()
        unsigned = dict(request)
        unsigned.pop("self_sha256")
        unsigned["design"] = copy.deepcopy(unsigned["design"])
        unsigned["design"]["pilot_grid_levels"] = unsigned["design"][
            "pilot_grid_levels"
        ][:2]
        with self.assertRaisesRegex(campaign.CampaignV2Error, "three pilot grid"):
            campaign.validate_request_v2(campaign.seal_artifact(unsigned))

    def test_grid_ladder_must_refine_both_axes_on_one_frozen_extent(self):
        request = sealed_request()
        unsigned = copy.deepcopy(request)
        unsigned.pop("self_sha256")
        unsigned["design"]["pilot_grid_levels"][1]["n_y"] = "64"
        with self.assertRaisesRegex(campaign.CampaignV2Error, "preserve both"):
            campaign.validate_request_v2(campaign.seal_artifact(unsigned))

        unsigned = copy.deepcopy(request)
        unsigned.pop("self_sha256")
        for level in unsigned["design"]["pilot_grid_levels"]:
            level["n_x"] = "64"
            level["n_y"] = "64"
        unsigned["design"]["confirmation_grid_level"]["n_x"] = "64"
        unsigned["design"]["confirmation_grid_level"]["n_y"] = "64"
        with self.assertRaisesRegex(campaign.CampaignV2Error, "preserve both"):
            campaign.validate_request_v2(campaign.seal_artifact(unsigned))

    def test_manifest_is_full_crossed_and_counterbalanced(self):
        request = sealed_request()
        manifest = campaign.materialize_pilot_manifest_v2(request)
        # 2 methods * (3x3 tuning * 4 seeds + 3x3 confirmation * 4 seeds)
        self.assertEqual(manifest["materialized_feebl_processes"], "144")
        schedules = manifest["schedules"]
        first_orders = [tuple(item["method_order"]) for item in schedules[:4]]
        self.assertEqual(first_orders[0], tuple(reversed(first_orders[1])))
        counts = {method: 0 for method in first_orders[0]}
        for block in schedules:
            counts[block["method_order"][0]] += 1
        self.assertEqual(len(set(counts.values())), 1)
        self.assertEqual(
            manifest["independent_seed_blocks"], "8"
        )

    def test_authenticated_observations_reject_partial_or_reordered_state(self):
        request = sealed_request()
        manifest, state = manifest_and_state(request)
        forged = copy.deepcopy(state)
        forged.pop("self_sha256")
        forged["completed"] = forged["completed"][:-1]
        forged = campaign.seal_artifact(forged)
        with self.assertRaisesRegex(campaign.CampaignV2Error, "cover every"):
            campaign.authenticated_pilot_observations_v2(request, manifest, forged)

    def test_manifest_forgery_fails_even_when_resigned(self):
        request = sealed_request()
        manifest = campaign.materialize_pilot_manifest_v2(request)
        forged = copy.deepcopy(manifest)
        forged.pop("self_sha256")
        forged["realizations"][0]["n_m"] = "999"
        forged = campaign.seal_artifact(forged)
        with self.assertRaisesRegex(campaign.CampaignV2Error, "differs from deterministic"):
            campaign.validate_manifest_v2(request, forged)

    def test_sha256_bindings_require_canonical_lowercase_hex(self):
        request = sealed_request()
        unsigned = copy.deepcopy(request)
        unsigned.pop("self_sha256")
        unsigned["design"]["fixed_physical_contract"]["domain_sha256"] = "z" * 64
        with self.assertRaisesRegex(campaign.CampaignV2Error, "canonical lowercase"):
            campaign.validate_request_v2(campaign.seal_artifact(unsigned))

    def test_exact_invariant_requires_exact_claim_and_stays_receipt_blocked(self):
        invalid = sealed_request(distribution_kind="exact_invariant")
        with self.assertRaisesRegex(
            campaign.CampaignV2Error, "requires an exact_invariant claim"
        ):
            campaign.validate_request_v2(invalid)

        unsigned = copy.deepcopy(invalid)
        unsigned.pop("self_sha256")
        unsigned["claims"] = [
            {
                "id": "lumi_identity",
                "endpoint_id": "lumi_ee",
                "family_id": "fccz_primary",
                "kind": "exact_invariant",
                "reference_method_id": "uniform_reference",
            }
        ]
        request = campaign.seal_artifact(unsigned)
        manifest, state = manifest_and_state(request)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        self.assertEqual(
            analysis["claim_registry"]["claims"][0]["status"],
            "requires_zero_tolerance_runtime_identity_receipt",
        )
        self.assertFalse(analysis["internal_diagnostic_gate_passed"])


class ConvergenceAndLedgerTests(unittest.TestCase):
    def test_synthetic_order_gci_and_crossed_model_recover(self):
        request = sealed_request()
        manifest, state = manifest_and_state(request)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        self.assertEqual(len(analysis["claim_registry"]["claims"]), 2)
        self.assertEqual(
            {item["id"] for item in analysis["claim_registry"]["claims"]},
            {"absolute_lumi_convergence", "adaptive_lumi_equivalence"},
        )
        candidate = next(
            item
            for item in analysis["method_results"]
            if item["method_id"] == "adaptive_candidate"
        )
        endpoint = candidate["endpoint_results"][0]
        self.assertTrue(endpoint["grid_convergence"]["passed"])
        self.assertTrue(endpoint["root_convergence"]["passed"])
        grid_order = float(
            endpoint["grid_convergence"]["tuning_triplet"]["observed_order"]
        )
        root_order = float(
            endpoint["root_convergence"]["tuning_triplet"]["observed_order"]
        )
        self.assertAlmostEqual(grid_order, 2.0, places=10)
        self.assertAlmostEqual(root_order, 0.5, places=10)
        self.assertLess(float(endpoint["crossed_model"]["rms_residual"]), 1e-10)
        self.assertTrue(endpoint["uncertainty_ledger"]["passed"])

    def test_failed_registered_claim_is_emitted_and_blocks_internal_gate(self):
        def nonequivalent_candidate(realization: dict, endpoint_id: str) -> Decimal:
            value = convergent_value(realization, endpoint_id)
            if realization["method_id"] == "adaptive_candidate":
                value += Decimal("1")
            return value

        request = sealed_request()
        manifest, state = manifest_and_state(request, nonequivalent_candidate)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        claims = analysis["claim_registry"]["claims"]
        equivalence = next(
            item for item in claims if item["id"] == "adaptive_lumi_equivalence"
        )
        self.assertEqual(equivalence["status"], "diagnostic_fail")
        self.assertFalse(analysis["internal_diagnostic_gate_passed"])

    def test_large_common_bias_fails_absolute_reference_claim(self):
        def common_bias(realization: dict, endpoint_id: str) -> Decimal:
            return convergent_value(realization, endpoint_id) + Decimal("900")

        request = sealed_request()
        manifest, state = manifest_and_state(request, common_bias)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        absolute = next(
            item
            for item in analysis["claim_registry"]["claims"]
            if item["kind"] == "absolute_convergence"
        )
        self.assertEqual(absolute["status"], "diagnostic_fail")
        self.assertGreater(
            float(absolute["absolute_reference_deviation_upper_bound"]),
            float(absolute["absolute_target"]),
        )
        self.assertFalse(analysis["internal_diagnostic_gate_passed"])

    def test_candidate_must_also_close_global_absolute_reference_bound(self):
        def near_edge_common_bias(realization: dict, endpoint_id: str) -> Decimal:
            value = convergent_value(realization, endpoint_id) + Decimal("0.75")
            if realization["method_id"] == "adaptive_candidate":
                value += Decimal("0.15")
            return value

        request = sealed_request()
        manifest, state = manifest_and_state(request, near_edge_common_bias)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        equivalence = next(
            item
            for item in analysis["claim_registry"]["claims"]
            if item["kind"] == "method_equivalence"
        )
        self.assertTrue(equivalence["all_bracket_cells_equivalent"])
        self.assertFalse(equivalence["candidate_absolute_gate_passed"])
        self.assertEqual(equivalence["status"], "diagnostic_fail")
        self.assertEqual(
            analysis["selected_method_id_diagnostic"], "uniform_reference"
        )
        self.assertFalse(analysis["internal_diagnostic_gate_passed"])

    def test_noncontracting_sequence_is_rejected(self):
        offsets = {"g1": "1", "g2": "0.2", "g3": "0.4", "g4": "0.1"}

        def noncontracting(realization: dict, endpoint_id: str) -> Decimal:
            del endpoint_id
            return Decimal("100") + Decimal(offsets[realization["grid_id"]])

        request = sealed_request()
        manifest, state = manifest_and_state(request, noncontracting)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        endpoint = analysis["method_results"][0]["endpoint_results"][0]
        self.assertFalse(endpoint["grid_convergence"]["passed"])
        reasons = endpoint["grid_convergence"]["tuning_triplet"]["reasons"]
        self.assertIn("point_differences_not_monotone_sign_contracting", reasons)
        plan = campaign.compute_next_stage_plan_v2(request, manifest, state)
        self.assertIn(
            campaign.ACTION_MESH_REFINEMENT,
            [item["action"] for item in plan["actions"]],
        )

    def test_grid_dependent_seed_noise_cannot_hide_behind_mean_gci(self):
        coefficients = {"g1": Decimal("10"), "g2": Decimal("-10"), "g3": Decimal("10"), "g4": Decimal("0")}

        def noisy_grid(realization: dict, endpoint_id: str) -> Decimal:
            phase = Decimal(2 * (int(realization["seed"]) % 4) - 3) / Decimal("2")
            return convergent_value(realization, endpoint_id) + coefficients[
                realization["grid_id"]
            ] * phase

        request = sealed_request()
        manifest, state = manifest_and_state(request, noisy_grid)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        endpoint = analysis["method_results"][0]["endpoint_results"][0]
        self.assertFalse(endpoint["grid_convergence"]["passed"])
        self.assertTrue(
            any(
                "confidence_intervals" in reason
                for reason in endpoint["grid_convergence"]["tuning_triplet"]["reasons"]
            )
        )
        self.assertFalse(endpoint["uncertainty_ledger"]["passed"])
        self.assertFalse(analysis["internal_diagnostic_gate_passed"])

    def test_disjoint_confirmation_pool_is_anchored_at_overlap_cells(self):
        def shifted_confirmation(realization: dict, endpoint_id: str) -> Decimal:
            value = convergent_value(realization, endpoint_id)
            if realization["stage"] == "finer_confirmation":
                value += Decimal("50")
            return value

        request = sealed_request()
        manifest, state = manifest_and_state(request, shifted_confirmation)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        endpoint = analysis["method_results"][0]["endpoint_results"][0]
        self.assertFalse(endpoint["grid_convergence"]["overlap_consistency_passed"])
        self.assertFalse(endpoint["root_convergence"]["overlap_consistency_passed"])
        self.assertFalse(analysis["internal_diagnostic_gate_passed"])

    def test_off_axis_crossed_cell_noise_enters_interaction_upper_bound(self):
        def noisy_crossed_corner(realization: dict, endpoint_id: str) -> Decimal:
            value = convergent_value(realization, endpoint_id)
            if (
                realization["stage"] == "tuning_pilot"
                and realization["grid_id"] == "g1"
                and realization["root_id"] == "N1"
            ):
                phase = Decimal(2 * (int(realization["seed"]) % 4) - 3) / Decimal("2")
                value += Decimal("100") * phase
            return value

        request = sealed_request()
        manifest, state = manifest_and_state(request, noisy_crossed_corner)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        endpoint = analysis["method_results"][0]["endpoint_results"][0]
        model = endpoint["crossed_model"]
        self.assertGreater(
            float(model["terminal_interaction_upper_confidence_bound"]), 1.0
        )
        interaction = next(
            item
            for item in endpoint["uncertainty_ledger"]["components"]
            if item["component"] == "grid_root_interaction"
        )
        self.assertGreater(float(interaction["upper_bound"]), float(interaction["allocation"]))
        self.assertFalse(endpoint["uncertainty_ledger"]["passed"])
        self.assertFalse(analysis["internal_diagnostic_gate_passed"])

    def test_crossed_model_lack_of_fit_is_an_uncertainty_not_a_hidden_residual(self):
        def bad_crossed_shape(realization: dict, endpoint_id: str) -> Decimal:
            value = convergent_value(realization, endpoint_id)
            if realization["stage"] == "tuning_pilot" and realization["grid_id"] == "g1":
                if realization["root_id"] == "N1":
                    value += Decimal("100")
                elif realization["root_id"] == "N2":
                    value += Decimal("500")
            return value

        request = sealed_request()
        manifest, state = manifest_and_state(request, bad_crossed_shape)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        endpoint = analysis["method_results"][0]["endpoint_results"][0]
        model = endpoint["crossed_model"]
        self.assertGreater(float(model["lack_of_fit_upper_bound"]), 1.0)
        interaction = next(
            item
            for item in endpoint["uncertainty_ledger"]["components"]
            if item["component"] == "grid_root_interaction"
        )
        self.assertGreater(float(interaction["upper_bound"]), float(interaction["allocation"]))
        self.assertFalse(analysis["internal_diagnostic_gate_passed"])

    def test_confirmation_crossed_surface_cannot_hide_an_off_axis_failure(self):
        def bad_confirmation_cell(realization: dict, endpoint_id: str) -> Decimal:
            value = convergent_value(realization, endpoint_id)
            if (
                realization["stage"] == "finer_confirmation"
                and realization["grid_id"] == "g2"
                and realization["root_id"] == "N2"
            ):
                value += Decimal("1000")
            return value

        request = sealed_request()
        manifest, state = manifest_and_state(request, bad_confirmation_cell)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        endpoint = analysis["method_results"][0]["endpoint_results"][0]
        model = endpoint["crossed_model"]
        self.assertGreater(
            float(model["confirmation_surface"]["lack_of_fit_upper_bound"]),
            1.0,
        )
        self.assertFalse(endpoint["uncertainty_ledger"]["passed"])
        self.assertFalse(analysis["internal_diagnostic_gate_passed"])

    def test_all_four_unique_crossed_overlap_cells_are_anchored(self):
        request = sealed_request()
        manifest, state = manifest_and_state(request)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        endpoint = analysis["method_results"][0]["endpoint_results"][0]
        summaries = endpoint["grid_convergence"][
            "independent_pool_overlap_consistency"
        ]
        self.assertEqual(
            {(item["grid_id"], item["root_id"]) for item in summaries},
            {("g2", "N2"), ("g2", "N3"), ("g3", "N2"), ("g3", "N3")},
        )
        self.assertEqual(
            summaries,
            endpoint["root_convergence"][
                "independent_pool_overlap_consistency"
            ],
        )

    def test_missing_uncertainty_component_fails_closed(self):
        request = sealed_request(include_complete_ledger=False)
        manifest, state = manifest_and_state(request)
        analysis = campaign.compute_pilot_analysis_v2(request, manifest, state)
        ledger = analysis["method_results"][0]["endpoint_results"][0][
            "uncertainty_ledger"
        ]
        missing = [item for item in ledger["components"] if item["status"] == "missing"]
        self.assertEqual([item["component"] for item in missing], ["external_reference"])
        self.assertFalse(ledger["passed"])
        self.assertFalse(analysis["internal_diagnostic_gate_passed"])

    def test_all_declared_distribution_kinds_are_versioned_and_non_normal_kinds_fail_closed(self):
        self.assertEqual(
            campaign.SUPPORTED_ENDPOINT_KINDS,
            {
                "continuous",
                "poisson_count",
                "poisson_rate",
                "binomial_proportion",
                "paired_bernoulli",
                "weighted_compound_poisson",
                "exact_invariant",
            },
        )
        for kind in campaign.SUPPORTED_ENDPOINT_KINDS - {"exact_invariant"}:
            config = campaign.validate_request_v2(sealed_request(distribution_kind=kind))
            self.assertEqual(config["endpoints"][0]["distribution_kind"], kind)


class PowerAndPlanningTests(unittest.TestCase):
    def test_student_t_and_variance_envelope_reference_values(self):
        self.assertAlmostEqual(campaign.student_t_quantile_v2(0.975, 1), 12.7062047364, places=8)
        self.assertAlmostEqual(campaign.student_t_quantile_v2(0.975, 10), 2.2281388520, places=8)
        self.assertAlmostEqual(campaign.chi_square_quantile_v2(0.025, 1), 0.0009820691, places=9)
        upper = campaign.normal_sigma_upper_confidence_v2(1.0, 2, 0.025)
        self.assertGreater(upper, 30.0)

    def test_exact_normal_tost_solver_attains_power_at_zero_and_near_margin(self):
        for true_difference in (0.0, 0.8):
            required = campaign.exact_normal_tost_required_n_v2(
                sigma=1.0,
                margin=1.0,
                true_difference=true_difference,
                one_sided_alpha=0.025,
                desired_power=0.9,
            )
            achieved = campaign.normal_tost_power_v2(
                required, 1.0, 1.0, true_difference, 0.025
            )
            self.assertGreaterEqual(achieved, 0.9)
            if required > 2:
                previous = campaign.normal_tost_power_v2(
                    required - 1, 1.0, 1.0, true_difference, 0.025
                )
                self.assertLess(previous, 0.9)
        zero_n = campaign.exact_normal_tost_required_n_v2(
            1.0, 1.0, 0.0, 0.025, 0.9
        )
        near_n = campaign.exact_normal_tost_required_n_v2(
            1.0, 1.0, 0.8, 0.025, 0.9
        )
        self.assertGreater(near_n, zero_n)

    def test_holdout_repeats_four_high_end_cells_and_reports_processes(self):
        request = sealed_request()
        manifest, state = manifest_and_state(request)
        plan = campaign.compute_next_stage_plan_v2(request, manifest, state)
        self.assertEqual(plan["planned_heldout_independent_seed_blocks"], "100")
        self.assertEqual(plan["materialized_feebl_processes"], "800")
        cells = {
            (item["grid_id"], item["root_id"])
            for item in plan["holdout_realizations"]
        }
        self.assertEqual(cells, {("g3", "N3"), ("g3", "N4"), ("g4", "N3"), ("g4", "N4")})
        self.assertTrue(plan["holdout_contract"]["terminal_only_confirmation_forbidden"])
        self.assertFalse(plan["execution_authorized"])

    def test_root_support_and_outer_precision_are_orthogonal_actions(self):
        def noisy_candidate(realization: dict, endpoint_id: str) -> Decimal:
            value = convergent_value(realization, endpoint_id)
            if realization["method_id"] == "adaptive_candidate":
                value += Decimal(int(realization["seed"]) % 4 - 1) * Decimal("0.2")
            # Destroy root contraction without changing seed semantics.
            if realization["root_id"] == "N2":
                value += Decimal("0.3")
            return value

        request = sealed_request(extension_count=0)
        manifest, state = manifest_and_state(request, noisy_candidate)
        plan = campaign.compute_next_stage_plan_v2(request, manifest, state)
        by_action = {item["action"]: item for item in plan["actions"]}
        self.assertIn(campaign.ACTION_WITHIN_RUN_ROOT_SUPPORT, by_action)
        self.assertIn(campaign.ACTION_OUTER_REPLICATE_PRECISION, by_action)
        support = by_action[campaign.ACTION_WITHIN_RUN_ROOT_SUPPORT]
        outer = by_action[campaign.ACTION_OUTER_REPLICATE_PRECISION]
        self.assertEqual(support["independent_seed_blocks_added"], "0")
        self.assertTrue(support["outer_seeds_reused"])
        self.assertFalse(outer["may_change_n_m"])
        self.assertEqual(outer["larger_n_m_same_seed_fallback"], "forbidden")
        self.assertEqual(
            outer["status"], "blocked_preregistered_extension_seed_pool_exhausted"
        )
        self.assertTrue(plan["extension_seed_pool_exhausted"])

    def test_outer_precision_materializes_fresh_seeds_on_full_crossed_lattice_before_holdout(self):
        def noisy_method_difference(realization: dict, endpoint_id: str) -> Decimal:
            value = convergent_value(realization, endpoint_id)
            if realization["method_id"] == "adaptive_candidate":
                value += Decimal(int(realization["seed"]) % 4 - 1) * Decimal("0.02")
            return value

        request = sealed_request(extension_count=1000)
        manifest, state = manifest_and_state(request, noisy_method_difference)
        plan = campaign.compute_next_stage_plan_v2(request, manifest, state)
        actions = {item["action"]: item for item in plan["actions"]}
        outer = actions[campaign.ACTION_OUTER_REPLICATE_PRECISION]
        added = int(outer["independent_seed_blocks_added"])
        self.assertGreater(added, 0)
        self.assertFalse(plan["extension_seed_pool_exhausted"])
        self.assertEqual(len(plan["outer_replicate_realizations"]), added * 4 * 4 * 2)
        self.assertFalse(plan["holdout_realizations"])
        self.assertEqual(
            {
                (item["grid_id"], item["root_id"])
                for item in plan["outer_replicate_realizations"]
            },
            {
                (f"g{grid}", f"N{root}")
                for grid in range(1, 5)
                for root in range(1, 5)
            },
        )
        self.assertTrue(
            all(
                item["seed_population"] == "outer_replicate_precision"
                for item in plan["outer_replicate_realizations"]
            )
        )

    def test_outer_power_uses_confirmation_count_and_exhaustion_fails_closed(self):
        def variable_difference(realization: dict, endpoint_id: str) -> Decimal:
            value = convergent_value(realization, endpoint_id)
            if realization["method_id"] == "adaptive_candidate":
                value += Decimal(int(realization["seed"]) % 2) * Decimal("0.002")
            return value

        request = sealed_request(
            tuning_count=100,
            confirmation_count=2,
            extension_count=1000,
        )
        manifest, state = manifest_and_state(request, variable_difference)
        plan = campaign.compute_next_stage_plan_v2(request, manifest, state)
        required = int(
            campaign.compute_pilot_analysis_v2(request, manifest, state)[
                "maximum_exact_normal_power_required_outer_seed_blocks"
            ]
        )
        self.assertEqual(plan["existing_claim_blocks_per_high_end_cell"], "2")
        outer = next(
            item
            for item in plan["actions"]
            if item["action"] == campaign.ACTION_OUTER_REPLICATE_PRECISION
        )
        self.assertEqual(
            int(outer["required_additional_seed_blocks"]), required - 2
        )
        self.assertGreater(required - 2, 1000)
        self.assertTrue(plan["extension_seed_pool_exhausted"])
        self.assertEqual(
            outer["status"],
            "blocked_preregistered_extension_seed_pool_exhausted",
        )
        self.assertEqual(outer["independent_seed_blocks_added"], "0")
        self.assertEqual(outer["larger_n_m_same_seed_fallback"], "forbidden")
        self.assertEqual(plan["outer_replicate_realizations"], [])
        failure = next(
            item
            for item in plan["actions"]
            if item["action"] == campaign.ACTION_DIAGNOSTIC_FAILURE
        )
        self.assertEqual(failure["status"], "fail_closed")
        self.assertIn(
            "preregistered_extension_seed_pool_exhausted", failure["reasons"]
        )
        self.assertEqual(failure["larger_n_m_same_seed_fallback"], "forbidden")

    def test_self_declared_blinded_power_cannot_lower_heldout_floor(self):
        blind = {
            "status": "candidate_blind_verified",
            "candidate_ids_blinded": True,
            "prospectively_frozen": True,
            "sigma_upper_bound_relative": "0.0001",
            "requested_seed_blocks": "20",
            "method": "external_family_blind_sigma_envelope_v1",
            "provenance": "sealed pre-candidate mock-study fixture",
        }
        request = sealed_request(
            holdout_count=30, extension_count=30, blinded_power_contract=blind
        )
        manifest, state = manifest_and_state(request)
        plan = campaign.compute_next_stage_plan_v2(request, manifest, state)
        self.assertTrue(
            plan["blinded_power_floor_override"]["self_declared_fields_consistent"]
        )
        self.assertFalse(plan["blinded_power_floor_override"]["floor_override_valid"])
        self.assertIn(
            "separately_sealed_candidate_blind_power_certificate",
            plan["blinded_power_floor_override"]["blocker"],
        )
        self.assertEqual(plan["planned_heldout_independent_seed_blocks"], "100")
        self.assertEqual(plan["materialized_feebl_processes"], "0")
        self.assertTrue(plan["holdout_seed_pool_exhausted"])


class PureArtifactValidationTests(unittest.TestCase):
    def test_emission_and_validation_use_exact_same_pure_recomputation(self):
        request = sealed_request()
        manifest, state = manifest_and_state(request)
        with tempfile.TemporaryDirectory() as temporary:
            analysis_path, plan_path = campaign.emit_pilot_artifacts_v2(
                request, manifest, state, Path(temporary)
            )
            analysis, plan = campaign.validate_pilot_artifacts_v2(
                request, manifest, state, analysis_path, plan_path
            )
            self.assertEqual(
                analysis_path.read_bytes(),
                campaign.canonical_bytes(
                    campaign.compute_pilot_analysis_v2(request, manifest, state)
                ),
            )
            self.assertEqual(
                plan_path.read_bytes(),
                campaign.canonical_bytes(
                    campaign.compute_next_stage_plan_v2(request, manifest, state)
                ),
            )
            self.assertFalse(analysis["physics_qualification_passed"])
            self.assertFalse(plan["execution_authorized"])

    def test_resigned_forged_analysis_and_plan_fail_canonical_byte_validation(self):
        request = sealed_request()
        manifest, state = manifest_and_state(request)
        with tempfile.TemporaryDirectory() as temporary:
            analysis_path, plan_path = campaign.emit_pilot_artifacts_v2(
                request, manifest, state, Path(temporary)
            )
            forged = json.loads(analysis_path.read_text(encoding="utf-8"))
            forged.pop("self_sha256")
            forged["selected_method_id_diagnostic"] = "uniform_reference"
            forged = campaign.seal_artifact(forged)
            analysis_path.write_bytes(campaign.canonical_bytes(forged))
            with self.assertRaisesRegex(campaign.CampaignV2Error, "byte-identical"):
                campaign.validate_pilot_artifacts_v2(
                    request, manifest, state, analysis_path, plan_path
                )


if __name__ == "__main__":
    unittest.main()
