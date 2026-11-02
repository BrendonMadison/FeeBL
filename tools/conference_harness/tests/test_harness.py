"""All data here are SYNTHETIC fixtures, never external physics validation."""

import copy
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

from harness.common import Invalid, digest, read_json, sha256, write_json
from harness.compare import compare, histogram
from harness.lhe import parse, roundtrip, validate_sidecar
from harness.runner import (
    _stop_group,
    make_plan,
    run_plan,
    selected_seed,
    validate_plan,
    verify_seed_receipt,
)
from harness.toy import conditional_covariance, invariant_mass, simulate

TEST_ROOT = Path(
    os.environ.get(
        "FEEBL_HARNESS_TEST_ROOT", str(Path(__file__).resolve().parent.parent / "test-work")
    )
).resolve()
TEST_ROOT.mkdir(parents=True, exist_ok=True)


class IsolatedTemp:
    """Default-inheritance ACLs avoid Windows sandbox/mkdtemp(0700) conflicts."""

    def __init__(self):
        self.path = TEST_ROOT / ("harness-fixture-" + uuid.uuid4().hex)
        self.path.mkdir()
        self.name = str(self.path)

    def cleanup(self):
        resolved = self.path.resolve()
        assert resolved.parent == TEST_ROOT.resolve() and resolved.name.startswith(
            "harness-fixture-"
        )
        shutil.rmtree(resolved)

    def __enter__(self):
        return self.name

    def __exit__(self, *args):
        self.cleanup()


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = IsolatedTemp()
        self.addCleanup(self.temp.cleanup)
        raw = Path(self.temp.name) / "synthetic_raw.txt"
        raw.write_text("Synthetic unittest fixture, not a collider reference.\n", encoding="utf-8")
        self.obs = {
            "name": "ee_luminosity_spectrum",
            "definition": "Synthetic dL/dW integral bin weights",
            "axes": [
                {
                    "name": "W",
                    "unit": "GeV",
                    "edges": [0, 200, 245, 250, 255],
                    "transform": "linear",
                    "log_base": None,
                    "reference_unit": None,
                }
            ],
            "value_unit": "cm^-2 s^-1",
            "normalization": "absolute_luminosity",
            "frame": "lab_invariant_mass",
            "beam_convention": "electron beam1 along +z; positron beam2 along -z",
            "selection": "ee only",
            "nominal_ecm_GeV": 250,
            "sample_kind": "ee_luminosity",
            "hemisphere_definition": "not_applicable",
            "angle_convention": "not_applicable",
            "observation": {
                "plane": "interaction_point",
                "time": "collision_integrated",
                "solenoid": "none",
                "beam_pipe": "none",
            },
            "cuts": {"selection": "none"},
            "peak_window_GeV": {"lower": 245.0, "upper": None},
        }
        prov = {
            "source_reference": "SYNTHETIC unittest fixture",
            "extraction_status": "VERIFIED",
            "auditor": "test fixture generator",
            "raw_artifacts": [{"path": str(raw), "sha256": sha256(raw)}],
        }

        def data(start, classification):
            return {
                "schema": "feebl.histogram_ensemble.v1",
                "classification": classification,
                "observable": copy.deepcopy(self.obs),
                "provenance": copy.deepcopy(prov),
                "seeds": [
                    {
                        "seed": seed,
                        "independent": True,
                        "n_records": 400,
                        "sumw": [100.0, 100.0, 100.0, 100.0],
                        "sumw2": [100.0, 100.0, 100.0, 100.0],
                        "sum_abs_w": [100.0, 100.0, 100.0, 100.0],
                    }
                    for seed in range(start, start + 8)
                ],
            }

        self.c, self.r = data(100, "SIMULATION"), data(200, "REFERENCE")
        self.contract = {
            "schema": "feebl.comparison_contract.v1",
            "registration": {
                "campaign_id": "synthetic-only",
                "registered_utc": "2026-10-26T00:00:00Z",
                "locked_before_results": True,
                "reviewer": "synthetic unittest",
            },
            "observable_name": self.obs["name"],
            "relative_tolerance": 0.01,
            "distribution_tolerance": 0.01,
            "confidence_level": 0.95,
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 7,
            "min_independent_seeds": 8,
            "ensemble_design": "independent",
            "uncertainty_bounds": {
                "candidate_numerical": 0.001,
                "reference_numerical": 0.001,
                "reference_model": 0.001,
                "reference_extraction": 0.001,
            },
            "metrics": [
                {"name": "total_luminosity", "kind": "integral", "bins": [0, 1, 2, 3]},
                {"name": "peak_top_2_percent", "kind": "integral", "bins": [2, 3]},
            ],
        }
        self.bound_evidence()

    def bound_evidence(self, zero_justification=None):
        self.contract["uncertainty_evidence"] = {}
        for component, bound in self.contract["uncertainty_bounds"].items():
            path = Path(self.temp.name) / (component + ".json")
            report = {
                "schema": "feebl.bound_evidence.v1",
                "component": component,
                "bound": bound,
                "unit": "dimensionless",
                "metric_names": [x["name"] for x in self.contract["metrics"]]
                + ["total_variation", "max_joint_cdf_distance"],
                "method": "Synthetic unittest coverage fixture, not physical evidence",
                "reviewer": "synthetic unittest",
                "status": "QUALIFIED",
                "zero_justification": zero_justification if bound == 0 else None,
            }
            path.write_text(json.dumps(report), encoding="utf-8")
            self.contract["uncertainty_evidence"][component] = {
                "path": str(path),
                "sha256": sha256(path),
            }

    def status(self):
        return compare(self.c, self.r, self.contract)["status"]

    def test_exact_fixed_ensemble_equivalence_is_scoped(self):
        self.assertEqual(self.status(), "NUMERICAL_EQUIVALENCE_PASS")

    def test_units_edges_normalization_and_angle_must_match(self):
        for field, value in (
            ("value_unit", "pb"),
            ("normalization", "probability"),
            ("angle_convention", "folded"),
        ):
            changed = copy.deepcopy(self.r)
            changed["observable"][field] = value
            self.assertEqual(compare(self.c, changed, self.contract)["status"], "BLOCKED")
        self.r["observable"]["axes"][0]["edges"][1] = 199
        self.assertEqual(self.status(), "BLOCKED")

    def test_nan_and_inconsistent_second_moments_block(self):
        self.c["seeds"][0]["sumw"][0] = float("nan")
        with self.assertRaises((Invalid, ValueError)):
            histogram(self.c)
        self.c["seeds"][0]["sumw"][0] = 100
        self.c["seeds"][0]["sumw2"][0] = 0
        self.assertEqual(self.status(), "BLOCKED")

    def test_missing_raw_and_hash_tampering_block(self):
        self.r["provenance"]["raw_artifacts"][0]["sha256"] = "0" * 64
        self.assertEqual(self.status(), "BLOCKED")

    def test_too_few_seed_replicates_block(self):
        self.c["seeds"] = self.c["seeds"][:2]
        self.assertEqual(self.status(), "BLOCKED")

    def test_numeric_floor_cannot_be_claimed_away(self):
        self.contract["uncertainty_bounds"]["reference_model"] = 0.02
        self.assertEqual(self.status(), "BLOCKED")

    def test_each_bound_requires_authentication_scope_and_value(self):
        original = copy.deepcopy(self.contract)
        for defect in ("missing", "hash", "scope", "value"):
            self.contract = copy.deepcopy(original)
            self.bound_evidence()
            pointer = self.contract["uncertainty_evidence"]["candidate_numerical"]
            if defect == "missing":
                self.contract["uncertainty_evidence"].pop("candidate_numerical")
            elif defect == "hash":
                pointer["sha256"] = "0" * 64
            else:
                path = Path(pointer["path"])
                report = read_json(path)
                if defect == "scope":
                    report["metric_names"].pop()
                else:
                    report["bound"] = 0.002
                path.write_text(json.dumps(report), encoding="utf-8")
                pointer["sha256"] = sha256(path)
            self.assertEqual(self.status(), "BLOCKED", defect)

    def test_zero_bound_is_unknown_without_explicit_justification(self):
        self.contract["uncertainty_bounds"] = {x: 0 for x in self.contract["uncertainty_bounds"]}
        self.bound_evidence()
        self.assertEqual(self.status(), "BLOCKED")
        self.bound_evidence(
            "Synthetic closed-form fixture has exactly equal registered arrays; not a machine uncertainty claim."
        )
        self.assertEqual(self.status(), "NUMERICAL_EQUIVALENCE_PASS")

    def test_not_significant_is_not_equivalent(self):
        for seed, scale in zip(self.c["seeds"], [0.4, 1.7, 0.3, 1.6, 1.9, 0.2, 0.7, 1.3]):
            seed["sumw"] = [100 * scale] * 4
            seed["sum_abs_w"] = seed["sumw"][:]
            seed["n_records"] = 10000
        self.assertEqual(self.status(), "FAIL_OR_INCONCLUSIVE")

    def test_distribution_shape_failure_even_equal_total(self):
        for seed in self.c["seeds"]:
            seed["sumw"] = seed["sum_abs_w"] = [110, 90, 100, 100]
        result = compare(self.c, self.r, self.contract)
        self.assertEqual(result["status"], "FAIL_OR_INCONCLUSIVE")
        self.assertFalse(
            next(m for m in result["metrics"] if m["name"] == "total_variation")["pass"]
        )

    def test_signed_weights_explicitly_block_positive_cdf(self):
        self.c["seeds"][0]["sum_abs_w"][0] = 110
        self.c["seeds"][0]["n_records"] = 1000
        self.assertTrue(histogram(self.c)["signed"])
        self.assertEqual(self.status(), "BLOCKED")

    def test_zero_reference_band_is_not_relative_error(self):
        for data in (self.c, self.r):
            for seed in data["seeds"]:
                seed["sumw"][2:] = [0, 0]
                seed["sumw2"][2:] = [0, 0]
                seed["sum_abs_w"][2:] = [0, 0]
        self.assertEqual(self.status(), "BLOCKED")

    def test_peak_window_cannot_slide_or_drop_high_energy_bins(self):
        self.c["observable"]["peak_window_GeV"]["lower"] = 244
        self.assertEqual(self.status(), "BLOCKED")
        self.c["observable"] = copy.deepcopy(self.r["observable"])
        self.contract["metrics"][1]["bins"] = [2]
        self.assertEqual(self.status(), "BLOCKED")

    def test_synthetic_cannot_pass_physics_gate(self):
        self.c["classification"] = "SYNTHETIC_ONLY"
        self.assertEqual(self.status(), "BLOCKED")

    def test_joint_two_dimensional_fixed_lattice(self):
        axis = {
            "name": "log10_pt_over_GeV",
            "unit": "dimensionless",
            "edges": [-4, -2, 0],
            "transform": "log",
            "log_base": 10,
            "reference_unit": "GeV",
        }
        for data in (self.c, self.r):
            data["observable"]["axes"] = [
                copy.deepcopy(axis),
                {**axis, "name": "log10_theta_out_over_rad", "reference_unit": "rad"},
            ]
            data["observable"].pop("peak_window_GeV")
            data["observable"]["sample_kind"] = "produced_background_pairs"
            data["observable"]["hemisphere_definition"] = "signed observation-plane z; backward arm"
            data["observable"][
                "angle_convention"
            ] = "theta_out=atan2(pT,abs(pz)); theta_lab retained separately"
        self.contract["metrics"] = [self.contract["metrics"][0]]
        self.bound_evidence()
        self.assertEqual(self.status(), "NUMERICAL_EQUIVALENCE_PASS")
        self.r["seeds"][0]["sumw"].pop()
        self.assertEqual(self.status(), "BLOCKED")


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = IsolatedTemp()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        binary = self.root / "feebl"
        binary.write_bytes(b"not-an-executable-test-fixture")
        card = self.root / "card.dat"
        card.write_text(
            "$ACCELERATOR:: ilc\n{energy=125;}\n$PARAMETERS:: test\n{rndm_seed=2;rndm_load=0;}\n",
            encoding="utf-8",
        )
        self.revision = "a" * 40
        self.config = {
            "schema": "feebl.run_config.v1",
            "binary": str(binary),
            "card": str(card),
            "binary_sha256": sha256(binary),
            "card_sha256": sha256(card),
            "source_revision": self.revision,
            "output_root": str(self.root),
            "run_name": "newrun",
            "accelerator": "ilc",
            "parameters": "test",
            "seed": 2,
            "timeout_seconds": 1,
            "environment": {"OMP_NUM_THREADS": "1"},
            "receipts": {
                "source": {"path": "out.dat", "contains": [self.revision]},
                "backend": {"path": "log.txt", "contains": ["actual_backend=serial"]},
            },
            "expected_outputs": [{"path": "out.dat", "min_bytes": 1}],
        }
        self.config["receipts"]["seed"] = {
            "path": "log.txt",
            "field": "actual_rng_seed",
            "evidence_kind": "runtime_rng_initialization",
        }

    def plan(self):
        file = self.root / "config.json"
        file.write_text(json.dumps(self.config), encoding="utf-8")
        return make_plan(file)

    def test_reviewed_gui_flag_is_bound_to_plan(self):
        self.config.update(schema="feebl.run_config.v2", options=["--make_gui_files"])
        plan = self.plan()
        self.assertEqual(plan["payload"]["command"][1], "--make_gui_files")
        validate_plan(plan, plan["review_token"])
        plan["payload"]["config"]["options"] = []
        with self.assertRaises(Invalid):
            validate_plan(plan, plan["review_token"])

    def test_only_bounded_gui_option_is_allowed(self):
        for options in (
            ["--acc_file", "elsewhere"],
            ["--make_gui_files"] * 2,
            "--make_gui_files",
            ["--help"],
        ):
            self.config.update(schema="feebl.run_config.v2", options=options)
            with self.assertRaises(Invalid):
                self.plan()
        self.config.update(schema="feebl.run_config.v1", options=["--make_gui_files"])
        with self.assertRaises(Invalid):
            self.plan()

    def test_plan_dry_run_does_not_create_run_directory(self):
        plan = self.plan()
        path = self.root / "plan.json"
        write_json(path, plan)
        self.assertEqual(run_plan(path, plan["review_token"])["status"], "REVIEWED_NOT_EXECUTED")
        self.assertFalse((self.root / "newrun").exists())

    def test_plan_tamper_and_card_tamper_fail_closed(self):
        plan = self.plan()
        tampered = copy.deepcopy(plan)
        tampered["payload"]["config"]["seed"] = 99
        with self.assertRaises(Invalid):
            validate_plan(tampered, plan["review_token"])
        Path(self.config["card"]).write_text("changed", encoding="utf-8")
        with self.assertRaises(Invalid):
            validate_plan(plan, plan["review_token"])

    def test_binary_hash_and_seed_mismatch_fail(self):
        self.config["binary_sha256"] = "0" * 64
        with self.assertRaises(Invalid):
            self.plan()

    def test_selected_active_seed_ignores_comments_and_other_blocks(self):
        text = "# rndm_seed=2;\n$ACCELERATOR:: ilc {energy=125;}\n$PARAMETERS:: unused {rndm_seed=99;rndm_load=0;}\n$PARAMETERS:: test { /* rndm_seed=5; */ rndm_seed=7; // rndm_seed=8;\nrndm_load=0;}"
        self.assertEqual(selected_seed(text, "ilc", "test"), 7)
        bad = text.replace("rndm_seed=7;", "rndm_seed=3+4;")
        with self.assertRaises(Invalid):
            selected_seed(bad, "ilc", "test")

    def test_existing_quoted_fast_backend_selectors_can_plan(self):
        for backend in ("cpu", "gpu"):
            text = (
                '$ACCELERATOR:: ilc {energy=125;} $PARAMETERS:: test {rndm_seed=2;rndm_load=0;compute_backend="'
                + backend
                + '";collision_schedule="wavefront_parallel";wavefront_field_solver="auto";cut_z=3*sigma_z.1;}'
            )
            Path(self.config["card"]).write_text(text, encoding="utf-8")
            self.config["card_sha256"] = sha256(self.config["card"])
            plan = self.plan()
            self.assertEqual(plan["payload"]["config"]["seed"], 2)

    def test_quoted_seed_lookalikes_delimiters_and_headers_are_refused(self):
        base = (
            "$ACCELERATOR:: ilc {energy=125;} $PARAMETERS:: test {rndm_seed=2;rndm_load=0;LABEL;}"
        )
        for field in (
            'label="# rndm_seed=99;"',
            'label="rndm_seed=99"',
            'label="cpu//comment"',
            'label="cpu/*comment*/"',
            'label="{$PARAMETERS:: fake}"',
            'label="cpu"+"gpu"',
            'label="cpu\\gpu"',
        ):
            with self.assertRaises(Invalid, msg=field):
                selected_seed(base.replace("LABEL", field), "ilc", "test")

    def test_simple_card_unsupported_syntax_and_rng_load_fail_closed(self):
        base = "$ACCELERATOR:: ilc {energy=125;} $PARAMETERS:: test {rndm_seed=2;rndm_load=0;}"
        for text in (
            base.replace("rndm_seed=2;", "rndm_seed=2;rndm_seed=3;"),
            base.replace("rndm_load=0;", "rndm_load=1;"),
            base.replace("rndm_load=0;", ""),
            base.replace("test {", "test inherits other {"),
            base.replace("rndm_seed=2;", 'rndm_seed="2";'),
            base.replace("rndm_seed=2;", "rndm_seed={2};"),
            "#include other.dat\n" + base.replace("$PARAMETERS", "include other.dat\n$PARAMETERS"),
            base + "/* unterminated",
        ):
            with self.assertRaises(Invalid, msg=text):
                selected_seed(text, "ilc", "test")

    def test_executed_seed_is_not_a_requested_echo_or_duplicate(self):
        receipt = self.config["receipts"]["seed"]
        self.assertEqual(verify_seed_receipt("actual_rng_seed = 2\n", receipt, 2), 2)
        for text in (
            "# actual_rng_seed = 2\n",
            "requested_rng_seed = 2\n",
            "actual_rng_seed = 7\n",
            "actual_rng_seed = 2\nactual_rng_seed = 2\n",
        ):
            with self.assertRaises(Invalid):
                verify_seed_receipt(text, receipt, 2)
        bad = {**receipt, "evidence_kind": "requested_input_echo"}
        with self.assertRaises(Invalid):
            verify_seed_receipt("actual_rng_seed = 2\n", bad, 2)
        current = {**receipt, "field": "the random number seed is"}
        self.assertEqual(verify_seed_receipt("  the random number seed is : 2\n", current, 2), 2)
        with self.assertRaises(Invalid):
            verify_seed_receipt("  requested the random number seed is : 2\n", current, 2)
        self.config["binary_sha256"] = sha256(self.config["binary"])
        self.config["seed"] = 3
        with self.assertRaises(Invalid):
            self.plan()

    def test_path_traversal_existing_root_and_receipts_required(self):
        self.config["expected_outputs"][0]["path"] = "../out.dat"
        with self.assertRaises(Invalid):
            self.plan()
        self.config["expected_outputs"][0]["path"] = "out.dat"
        (self.root / "newrun").mkdir()
        with self.assertRaises(Invalid):
            self.plan()
        (self.root / "newrun").rmdir()
        self.config["receipts"].pop("source")
        with self.assertRaises(Invalid):
            self.plan()

    def test_never_overwrite_reviewed_plan(self):
        file = self.root / "plan.json"
        write_json(file, self.plan())
        with self.assertRaises(FileExistsError):
            write_json(file, {})

    @unittest.skipUnless(
        platform.system() == "Linux",
        "Linux process-group cleanup is explicitly not claimed on Windows",
    )
    def test_linux_timeout_group_cleanup(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
        )
        self.assertTrue(_stop_group(process))
        self.assertIsNotNone(process.poll())


def lhe_fixture(final_count):
    lines = [
        '<LesHouchesEvents version="3.0">',
        '<header><initrwgt><weight id="scale">synthetic</weight></initrwgt></header>',
        "<init>",
        "11 -11 125 125 0 0 0 0 3 1",
        "1.0 0.0 1.0 1",
        "</init>",
        "<event>",
        f"{2 + final_count} 1 1.0 250.0 0.007297 0.118",
        "11 -1 0 0 0 0 0 0 125 125 0.000511 0 9",
        "-11 -1 0 0 0 0 0 0 -125 125 0.000511 0 9",
    ]
    for index in range(final_count):
        lines.append(f"{11 if index % 2 else -11} 1 1 2 0 0 0.1 0 -10 10.0005 0.000511 0 9")
    lines += [
        '<rwgt><wgt id="scale">1.1</wgt></rwgt>',
        '<scales muf="250" mur="250"/>',
        "</event>",
        "</LesHouchesEvents>",
    ]
    return "\n".join(lines)


class LHETests(unittest.TestCase):
    def setUp(self):
        self.temp = IsolatedTemp()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def file(self, text):
        file = self.root / "events.lhe"
        file.write_text(text, encoding="utf-8")
        return file

    def test_one_two_three_final_state_particles_and_reweights(self):
        for count in (1, 2, 3):
            source = self.file(lhe_fixture(count))
            _, result = parse(source)
            self.assertEqual(result["events"][0]["n_outgoing"], count)
            self.assertEqual(result["events"][0]["reweight_ids"], ["scale"])
            result = roundtrip(source, self.root / f"roundtrip{count}.lhe")
            self.assertEqual(result["status"], "SEMANTIC_ROUNDTRIP_PASS")

    def test_nup_wrong_record_count_and_nonfinite_rejected(self):
        for source in (
            lhe_fixture(1).replace("3 1 1.0 250.0", "4 1 1.0 250.0"),
            lhe_fixture(1).replace("0.1 0 -10", "NaN 0 -10"),
        ):
            with self.assertRaises((Invalid, ValueError)):
                parse(self.file(source))

    def test_no_fabricated_incoming_or_entity_expansion(self):
        for source in (
            lhe_fixture(1).replace("11 -1 0 0", "11 1 0 0"),
            '<!DOCTYPE x [<!ENTITY p "x">]>' + lhe_fixture(1),
        ):
            with self.assertRaises(Invalid):
                parse(self.file(source))

    def test_sidecar_keeps_lab_and_folded_angles_separate(self):
        source = self.file(lhe_fixture(1))
        _, parsed = parse(source)
        sidecar = {
            "schema": "feebl.transport_sidecar.v1",
            "lhe_sha256": sha256(source),
            "frame": "lab",
            "units": {"momentum": "GeV", "position": "mm", "time": "ns"},
            "transport": "not_applied",
            "particles": [
                {
                    "event_id": 1,
                    "particle_id": 3,
                    "x_mm": 0.0,
                    "y_mm": 0.0,
                    "z_mm": -100.0,
                    "time_ns": 1.0,
                    "weight": 1.0,
                    "observation_plane": "z=-100 mm",
                    "theta_lab_rad": math.atan2(0.1, -10),
                    "theta_out_rad": math.atan2(0.1, 10),
                    "phi_rad": 0.0,
                    "hemisphere_definition": "signed observation-plane z",
                }
            ],
        }
        self.assertEqual(validate_sidecar(sidecar, parsed)["particles"], 1)
        sidecar["particles"][0]["theta_lab_rad"] = sidecar["particles"][0]["theta_out_rad"]
        with self.assertRaises(Invalid):
            validate_sidecar(sidecar, parsed)


class ToyTests(unittest.TestCase):
    def test_covariance_opposite_sign_narrows_sum_same_sign_does_not(self):
        opposite, vop, overlap = conditional_covariance(0.001, 0.001, 0.001, -0.001, 5e-7, 5e-7)
        same, vsame, _ = conditional_covariance(0.001, 0.001, 0.001, 0.001, 5e-7, 5e-7)
        self.assertLess(vop, vsame)
        self.assertAlmostEqual(vsame, 2e-6)
        self.assertLess(overlap, 1)
        self.assertLess(opposite[0][1], 0)
        self.assertGreater(same[0][1], 0)
        self.assertGreater(opposite[0][0] * opposite[1][1] - opposite[0][1] ** 2, 0)

    def test_exact_asymmetric_invariant_not_sum(self):
        self.assertAlmostEqual(invariant_mass(500, 31.25), 250, places=7)
        self.assertNotAlmostEqual(invariant_mass(500, 31.25), 531.25)

    def test_toy_conditional_moments_with_statistical_bound(self):
        config = {
            "schema": "feebl.gaussian_dispersion_toy.v1",
            "seed": 17,
            "samples": 30000,
            "energy1_GeV": 125,
            "energy2_GeV": 125,
            "sigma_delta1": 0.001,
            "sigma_delta2": 0.001,
            "dispersion1_m": 0.001,
            "dispersion2_m": -0.001,
            "sigma_xbeta1_m": 5e-7,
            "sigma_xbeta2_m": 5e-7,
        }
        result = simulate(config)
        relative = abs(
            result["measured_variance_delta_sum"] / result["analytic_variance_delta_sum"] - 1
        )
        self.assertLess(relative, 6 * math.sqrt(2 / (config["samples"] - 1)))
        self.assertEqual(result["classification"], "SYNTHETIC_ONLY")
        with self.assertRaises(Invalid):
            conditional_covariance(0, 0.001, 0, 0, 1e-6, 1e-6)


class JSONTests(unittest.TestCase):
    def test_duplicate_json_keys_and_overwrites_rejected(self):
        with IsolatedTemp() as directory:
            path = Path(directory) / "data.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaises(Invalid):
                read_json(path)


if __name__ == "__main__":
    unittest.main()
