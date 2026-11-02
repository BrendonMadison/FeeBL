#!/usr/bin/env python3
"""Standard-library contract tests for the coupled grid-by-root campaign."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = (
    ROOT
    / "examples" / "adaptive_particles" / "coupled_grid_root_campaign.py"
)
SPEC = importlib.util.spec_from_file_location("coupled_grid_root_campaign", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


BASE_CARD = """$ACCELERATOR:: fccz
{energy=45.6;dist_z=0;fake_method_bias=BIAS;}

$PARAMETERS:: p_fccz
{n_x=16;n_y=16;n_z=2;n_m=1000;
charge_symmetric=0;force_symmetric=0;
load_beam=0;rndm_load=0;rndm_save=0;rndm_seed=7;
do_photons=0;do_pairs=0;do_lumi=1;num_lumi=1000;
spectrum_w_min=84;spectrum_w_max=92;spectrum_w_bins=4000;}
"""


FAKE_FEEBL = r'''#!/usr/bin/env python3
from decimal import Decimal
from pathlib import Path
import re
import sys

card = Path("acc.dat").read_text(encoding="utf-8")
def integer(name):
    return int(re.search(r"(?:^|[;{])\s*" + re.escape(name) + r"\s*=\s*([0-9]+)", card, re.M).group(1))
def number(name):
    return Decimal(re.search(r"(?:^|[;{])\s*" + re.escape(name) + r"\s*=\s*([0-9.]+)", card, re.M).group(1))
nx = integer("n_x")
ny = integer("n_y")
nm = integer("n_m")
seed = integer("rndm_seed")
bias = number("fake_method_bias")
mode_match = re.search(r"(?:^|[;{])\s*adaptive_macroparticle_mode\s*=\s*([0-9]+)", card, re.M)
adaptive_mode = int(mode_match.group(1)) if mode_match else 0
value = Decimal(100) + Decimal(1) / Decimal(nx) + Decimal(1) / Decimal(ny)
value += Decimal(2) / Decimal(nm) + bias + Decimal(seed % 7 - 3) / Decimal(10000)
Path("out.dat").write_text("lumi_ee=" + str(value) + ";\n", encoding="utf-8")
Path("lumi.ee.out.dLdW.fne").write_text(
    "#rms_W " + str(value / Decimal(100)) + " +- 0.00001\n", encoding="utf-8"
)
if adaptive_mode:
    print(
        "FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2 "
        "schema=feebl.adaptive-root-replay-signal.v2 "
        "request_schema=feebl.adaptive-root-replay-request.v1 "
        "exit_code=0 publication_blocked=0 exploratory_outputs_authorized=1 "
        "final_physics_outputs_published=1 "
        "publication_scope=final_summary_outputs_only_v1 "
        "pre_gate_diagnostic_artifacts_may_exist=1 "
        "pre_gate_artifacts_physics_qualification_claim=0 "
        "preexisting_artifacts_not_removed=1 fresh_run_directory_required=1 "
        "final_polarization_sample_complete=1 bpm_finalization_complete=1 "
        "global_luminosity_publication_complete=1 "
        "complete_collision=1 physics_qualification_claim=0 "
        "recommended_action=request_initial_state_replay reason=fixture "
        "generator_id=feebl_legacy_mc_complete_run_v1 "
        "normalization_contract=feebl_n_m_attempted_legacy_markers_v1 "
        f"rndm_seed={seed} n_m_beam1={nm} n_m_beam2={nm} common_n_m=1 "
        "independent_complete_runs=1 minimum_complete_runs=4 "
        "controller_observations=1 controller_trace_hash=1 "
        "roots_added_online=0 fresh_complete_process_required=1"
    )
trace = Path(__file__).with_name("trace.txt")
with trace.open("a", encoding="utf-8") as handle:
    handle.write(f"{nx},{ny},{nm},{seed}\n")
sys.exit(0)
'''


STRICT_FEEBL = r'''#!/usr/bin/env python3
from pathlib import Path
import re
import sys
card = Path("acc.dat").read_text(encoding="utf-8")
def integer(name):
    return int(re.search(r"(?:^|[;{])\s*" + re.escape(name) + r"\s*=\s*([0-9]+)", card, re.M).group(1))
nm = integer("n_m")
seed = integer("rndm_seed")
print(
    "FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2 "
    "schema=feebl.adaptive-root-replay-signal.v2 "
    "request_schema=feebl.adaptive-root-replay-request.v1 "
    "exit_code=75 publication_blocked=1 exploratory_outputs_authorized=0 "
    "final_physics_outputs_published=0 "
    "publication_scope=final_summary_outputs_only_v1 "
    "pre_gate_diagnostic_artifacts_may_exist=1 "
    "pre_gate_artifacts_physics_qualification_claim=0 "
    "preexisting_artifacts_not_removed=1 fresh_run_directory_required=1 "
    "final_polarization_sample_complete=1 bpm_finalization_complete=1 "
    "global_luminosity_publication_complete=1 "
    "complete_collision=1 physics_qualification_claim=0 "
    "recommended_action=request_initial_state_replay reason=fixture "
    "generator_id=feebl_legacy_mc_complete_run_v1 "
    "normalization_contract=feebl_n_m_attempted_legacy_markers_v1 "
    f"rndm_seed={seed} n_m_beam1={nm} n_m_beam2={nm} common_n_m=1 "
    "independent_complete_runs=1 minimum_complete_runs=4 "
    "controller_observations=1 controller_trace_hash=1 "
    "roots_added_online=0 fresh_complete_process_required=1"
)
sys.exit(75)
'''


FAKE_CORE_TRANSFORM = r'''#!/usr/bin/env python3
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("spectrum", type=Path)
parser.add_argument("--json", required=True, type=Path, dest="json_path")
parser.add_argument("--csv", required=True, type=Path, dest="csv_path")
parser.add_argument("--expected-seed", required=True)
parser.add_argument("--expected-w-min", required=True)
parser.add_argument("--expected-w-max", required=True)
parser.add_argument("--expected-bins", required=True)
parser.add_argument("--expected-total-luminosity", required=True)
parser.add_argument("--expected-run-id", required=True)
args = parser.parse_args()
raw = args.spectrum.read_bytes()
text = raw.decode("utf-8")
value = Decimal(re.search(r"(?m)^#rms_W\s+([0-9.eE+-]+)", text).group(1))
numeric = str(value)
result = {
    "schema": "feebl.gaussian-core-spectrum.v1",
    "schema_version": "1",
    "status": "accepted_core_fit_not_physics_qualification",
    "input": {
        "file_name": args.spectrum.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
    },
    "qualification_binding": {
        "status": "qualification_input_bound",
        "native_format": "FEEBL-NATIVE 1.0",
        "feebl_version": "fixture",
        "rndm_seed": args.expected_seed,
        "run_id": args.expected_run_id,
        "spectrum_w_min_GeV": args.expected_w_min,
        "spectrum_w_max_GeV": args.expected_w_max,
        "spectrum_w_bins": args.expected_bins,
        "expected_total_luminosity": args.expected_total_luminosity,
        "lattice_phase_proven": True,
    },
    "core_fit": {
        "mu_core_GeV": "91.2",
        "sigma_core_GeV": numeric,
        "gaussian_core_area": numeric,
        "peak_dL_dW": numeric,
        "mu_fit_uncertainty_GeV_diagnostic": "0.0001",
        "sigma_fit_uncertainty_GeV_diagnostic": "0.0001",
    },
    "stability": {
        "maximum_window_mu_shift_GeV": "0.0001",
        "maximum_window_sigma_relative_shift": "0.001",
        "pair_rebin_mu_shift_GeV": "0.0001",
        "pair_rebin_sigma_relative_shift": "0.001",
    },
}
args.json_path.write_text(
    json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
args.csv_path.write_text(
    "schema,input_file_name,input_sha256,binding_status,run_id,rndm_seed,"
    "spectrum_w_min_GeV,spectrum_w_max_GeV,spectrum_w_bins,lattice_phase_proven,"
    "expected_total_luminosity,"
    "status,mu_core_GeV,sigma_core_GeV,"
    "gaussian_core_area,peak_dL_dW,mu_fit_uncertainty_GeV_diagnostic,"
    "sigma_fit_uncertainty_GeV_diagnostic,maximum_window_mu_shift_GeV,"
    "maximum_window_sigma_relative_shift,pair_rebin_mu_shift_GeV,"
    "pair_rebin_sigma_relative_shift\n"
    + ",".join(
        [
            result["schema"],
            result["input"]["file_name"],
            result["input"]["sha256"],
            result["qualification_binding"]["status"],
            result["qualification_binding"]["run_id"],
            result["qualification_binding"]["rndm_seed"],
            result["qualification_binding"]["spectrum_w_min_GeV"],
            result["qualification_binding"]["spectrum_w_max_GeV"],
            result["qualification_binding"]["spectrum_w_bins"],
            "true",
            result["qualification_binding"]["expected_total_luminosity"],
            result["status"],
            result["core_fit"]["mu_core_GeV"],
            result["core_fit"]["sigma_core_GeV"],
            result["core_fit"]["gaussian_core_area"],
            result["core_fit"]["peak_dL_dW"],
            result["core_fit"]["mu_fit_uncertainty_GeV_diagnostic"],
            result["core_fit"]["sigma_fit_uncertainty_GeV_diagnostic"],
            result["stability"]["maximum_window_mu_shift_GeV"],
            result["stability"]["maximum_window_sigma_relative_shift"],
            result["stability"]["pair_rebin_mu_shift_GeV"],
            result["stability"]["pair_rebin_sigma_relative_shift"],
        ]
    )
    + "\n",
    encoding="utf-8",
)
'''


def write_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return campaign.sha256_file(path)


class CampaignFixture:
    def __init__(self, root: Path, *, adaptive_bias: str = "0.1") -> None:
        self.root = root
        self.reference = root / "reference.dat"
        self.adaptive = root / "adaptive.dat"
        self.binary = root / "fake-feebl"
        self.core_transform = root / "fake-gaussian-core.py"
        self.source = root / "source-state.json"
        self.reference_hash = write_bytes(
            self.reference, BASE_CARD.replace("BIAS", "0").encode("utf-8")
        )
        self.adaptive_hash = write_bytes(
            self.adaptive,
            BASE_CARD.replace("BIAS", adaptive_bias)
            .replace(
                "{n_x=16;n_y=16;",
                "{n_x=16;n_y=16;adaptive_grid_max_nx=16;adaptive_grid_max_ny=16;"
                "adaptive_macroparticle_mode=2;",
            )
            .encode("utf-8"),
        )
        self.binary_hash = write_bytes(self.binary, FAKE_FEEBL.encode("utf-8"))
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR)
        self.core_transform_hash = write_bytes(
            self.core_transform, FAKE_CORE_TRANSFORM.encode("utf-8")
        )
        self.python_path = Path(sys.executable).resolve()
        self.python_hash = campaign.sha256_file(self.python_path)
        self.source_hash = write_bytes(self.source, b'{"source":"fixture-v1"}\n')
        self.request: dict[str, object] = {
            "schema": campaign.REQUEST_SCHEMA,
            "reference_template_id": "reference",
            "templates": [
                {
                    "id": "reference",
                    "path": str(self.reference),
                    "sha256": self.reference_hash,
                    "accelerator": "fccz",
                    "parameters": "p_fccz",
                    "grid_patch_contract": campaign.BASE_GRID_ONLY_CONTRACT,
                },
                {
                    "id": "adaptive",
                    "path": str(self.adaptive),
                    "sha256": self.adaptive_hash,
                    "accelerator": "fccz",
                    "parameters": "p_fccz",
                    "grid_patch_contract": campaign.COUPLED_ADAPTIVE_MAXIMA_CONTRACT,
                },
            ],
            "bindings": {
                "feebl_binary_path": str(self.binary),
                "feebl_binary_sha256": self.binary_hash,
                "source_state_path": str(self.source),
                "source_state_sha256": self.source_hash,
                "generator_transform": {
                    "id": campaign.GENERATOR_ID,
                    "version": "legacy-complete-run-fixture-v1",
                },
                "physics_family": {
                    "id": "fccz-adaptive-grid-root-fixture-v1",
                    "enabled_channels": ["beam_beam_fields", "lumi_ee"],
                },
                "gaussian_core_transform": {
                    "id": campaign.GAUSSIAN_CORE_TRANSFORM_ID,
                    "path": str(self.core_transform),
                    "sha256": self.core_transform_hash,
                    "python_executable_path": str(self.python_path),
                    "python_executable_sha256": self.python_hash,
                },
            },
            "grid_levels": [
                {"id": "g16", "n_x": 16, "n_y": 16},
                {"id": "g32", "n_x": 32, "n_y": 32},
            ],
            "n_m_levels": [1000, 2000],
            "pilot_seeds": [11, 13, 17, 19],
            "extension_seeds": [23, 29, 31, 37],
            "holdout_seeds": [101, 103, 107, 109],
            "endpoints": [
                {
                    "id": "lumi_ee",
                    "file": "out.dat",
                    "format": "assignment",
                    "key": "lumi_ee",
                    "scale": "100",
                    "target_relative_error": "0.01",
                    "equivalence_margin_relative": "0.01",
                    "expected_true_difference_relative": "0",
                    "units": "arb",
                },
                {
                    "id": "rms_w",
                    "file": "lumi.ee.out.dLdW.fne",
                    "format": "hash_comment",
                    "key": "rms_W",
                    "scale": "1",
                    "target_relative_error": "0.01",
                    "equivalence_margin_relative": "0.01",
                    "expected_true_difference_relative": "0",
                    "units": "arb",
                },
                {
                    "id": "gaussian_core_sigma",
                    "file": "gaussian_core.json",
                    "format": "json_scalar",
                    "key": "core_fit.sigma_core_GeV",
                    "scale": "1",
                    "target_relative_error": "0.01",
                    "equivalence_margin_relative": "0.01",
                    "expected_true_difference_relative": "0",
                    "units": "GeV",
                },
            ],
            "statistics": {
                "standard_error_safety_factor": "1",
                "confidence_level": "0.95",
                "equivalence_alpha": "0.05",
                "equivalence_power": "0.9",
                "conservative_total_budget_multiplier": "3",
            },
            "next_stage": {
                "grid": {"id": "g64", "n_x": 64, "n_y": 64},
                "n_m": 4000,
            },
            "limits": {
                "maximum_grid_cells": 4096,
                "maximum_n_m": 4000,
                "maximum_pilot_runs": 128,
                "maximum_wall_seconds_per_run": 30,
            },
        }

    def write_request(self, name: str = "request.json") -> Path:
        path = self.root / name
        path.write_text(
            json.dumps(self.request, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return path


class CoupledCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="feebl-coupled-campaign-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_reference_cic_smoke_grid_contract_covers_pilot_and_next_stage(self) -> None:
        template_path = (
            ROOT
            / "examples" / "adaptive_particles" / "reference_cic_replay_smoke_mode2_cpu.dat"
        )
        raw = template_path.read_bytes()
        for n_xy in (16, 32, 64):
            card = campaign.patch_card(
                raw,
                "fccz",
                "p_fccz",
                n_xy,
                n_xy,
                512,
                4001,
                campaign.COUPLED_ADAPTIVE_MAXIMA_CONTRACT,
            )
            info = campaign.inspect_campaign_card(
                card,
                "fccz",
                "p_fccz",
                campaign.COUPLED_ADAPTIVE_MAXIMA_CONTRACT,
            )
            self.assertEqual(info["adaptive_grid_max_nx"], n_xy)
            self.assertEqual(info["adaptive_grid_max_ny"], n_xy)
            self.assertEqual(info["adaptive_grid_size_classes_x"], [8, 16])
            self.assertEqual(info["adaptive_grid_size_classes_y"], [8, 16])
            self.assertEqual(info["adaptive_grid_source_reference_nx"], 64)
            self.assertEqual(info["adaptive_grid_source_reference_ny"], 64)
            self.assertEqual(
                info["adaptive_grid_field_spacing_classes"], [1, 2, 4]
            )
            text = card.decode("utf-8")
            block = campaign.replay.selected_block(text, "PARAMETERS", "p_fccz")
            self.assertEqual(
                campaign.replay.one_assignment(text, block, "num_lumi").value,
                50000,
            )

        class_bank_too_large = raw.replace(b'"8,16"', b'"8,16,32"')
        with self.assertRaisesRegex(campaign.CampaignError, "lie inside"):
            campaign.inspect_campaign_card(
                class_bank_too_large,
                "fccz",
                "p_fccz",
                campaign.COUPLED_ADAPTIVE_MAXIMA_CONTRACT,
            )

        source_too_small = raw.replace(
            b"adaptive_grid_source_reference_nx=64;adaptive_grid_source_reference_ny=64;",
            b"adaptive_grid_source_reference_nx=32;adaptive_grid_source_reference_ny=32;",
        )
        source_too_small_at_g64 = campaign.patch_card(
            source_too_small,
            "fccz",
            "p_fccz",
            64,
            64,
            2048,
            4001,
            campaign.COUPLED_ADAPTIVE_MAXIMA_CONTRACT,
        )
        with self.assertRaisesRegex(campaign.CampaignError, "integer multiples"):
            campaign.inspect_campaign_card(
                source_too_small_at_g64,
                "fccz",
                "p_fccz",
                campaign.COUPLED_ADAPTIVE_MAXIMA_CONTRACT,
            )

        # The source-32 card is valid at the source template's 16 grid and at
        # the pilot 32 grid.  Freeze must nevertheless reject it because the
        # preregistered 64 next-stage grid is not compatible.
        fixture = CampaignFixture(self.root)
        fixture.adaptive.write_bytes(source_too_small)
        fixture.adaptive_hash = campaign.sha256_file(fixture.adaptive)
        adaptive_request = fixture.request["templates"][1]
        assert isinstance(adaptive_request, dict)
        adaptive_request["sha256"] = fixture.adaptive_hash
        with self.assertRaisesRegex(campaign.CampaignError, "integer multiples"):
            campaign.freeze_campaign(
                fixture.write_request(), self.root / "invalid-next-grid-campaign"
            )
        self.assertFalse((self.root / "invalid-next-grid-campaign").exists())

    def test_student_t_quantiles_and_familywise_policy_are_small_sample_safe(self) -> None:
        self.assertAlmostEqual(
            float(campaign._student_t_quantile(campaign.Decimal("0.975"), 3)),
            3.182446305,
            places=8,
        )
        self.assertAlmostEqual(
            float(campaign._student_t_quantile(campaign.Decimal("0.95"), 9)),
            1.833112933,
            places=8,
        )
        fixture = CampaignFixture(self.root)
        policy, _ = campaign.freeze_campaign(
            fixture.write_request(), self.root / "campaign"
        )
        statistics = policy["statistics"]
        self.assertEqual(statistics["familywise_error_control"], "bonferroni_v1")
        self.assertEqual(statistics["pilot_degrees_of_freedom"], "3")
        self.assertGreater(
            float(
                statistics[
                    "two_sided_confidence_familywise_student_t_quantile"
                ]
            ),
            3.182446305,
        )

    def test_endpoint_scale_contract_authenticates_the_declared_one_percent(self) -> None:
        raw = {
            "id": "lumi_ee",
            "file": "out.dat",
            "format": "assignment",
            "key": "lumi_ee",
            "scale": "3e34",
            "target_relative_error": "0.01",
            "equivalence_margin_relative": "0.01",
            "expected_true_difference_relative": "0",
            "units": "m^-2",
            "scale_contract": {
                "reference_value": "3e34",
                "near_zero_floor": "1e20",
                "scale_basis": "3e34",
                "absolute_target": "3e32",
                "reference_value_provenance": "preregistered external design reference v1",
                "near_zero_floor_provenance": "preregistered physical resolution floor v1",
            },
        }
        endpoint = campaign._endpoint_specs([raw])[0]
        contract = endpoint["scale_contract"]
        self.assertEqual(
            campaign.Decimal(contract["scale_basis"]),
            max(
                abs(campaign.Decimal(contract["reference_value"])),
                campaign.Decimal(contract["near_zero_floor"]),
            ),
        )
        self.assertEqual(
            campaign.Decimal(contract["absolute_target"]),
            campaign.Decimal(contract["scale_basis"])
            * campaign.Decimal(endpoint["target_relative_error"]),
        )
        broken = json.loads(json.dumps(raw))
        broken["scale_contract"]["absolute_target"] = "3e31"
        with self.assertRaisesRegex(campaign.CampaignError, "scale_basis times"):
            campaign._endpoint_specs([broken])

        signed_near_zero = json.loads(json.dumps(raw))
        signed_near_zero["scale"] = "5"
        signed_near_zero["scale_contract"].update(
            {
                "reference_value": "-2",
                "near_zero_floor": "5",
                "scale_basis": "5",
                "absolute_target": "0.05",
            }
        )
        signed_contract = campaign._endpoint_specs([signed_near_zero])[0][
            "scale_contract"
        ]
        self.assertEqual(signed_contract["reference_value"], "-2")
        self.assertEqual(signed_contract["scale_basis"], "5")

        missing_floor = json.loads(json.dumps(raw))
        del missing_floor["scale_contract"]["near_zero_floor"]
        with self.assertRaisesRegex(campaign.CampaignError, "near_zero_floor"):
            campaign._endpoint_specs([missing_floor])

        # Legacy requests without provenance remain reproducible inputs but
        # cannot support a new absolute 1% qualification claim.
        del raw["scale_contract"]
        config = {"grids": [{}, {}], "roots": [1, 2], "endpoints": campaign._endpoint_specs([raw])}
        status = campaign._solution_verification_design_status(config)
        self.assertIn(
            "endpoint_scale_reference_provenance_missing:lumi_ee",
            status["blockers"],
        )

    def test_adaptive_grid_certificate_is_strict_and_hash_bound(self) -> None:
        log_path = self.root / "grid-certificate.log"
        fields = {
            "schema": campaign.GRID_CERTIFICATE_SCHEMA,
            "requested_mode": "3",
            "actual_mode": "3",
            "error_norm_policy": "vector_rms_quantile_v1",
            "source_model": "fixed_width_bspline1_v1",
            "error_target": "0.01",
            "error_quantile": "0.999",
            "pilot_diagnostics": "1",
            "fail_on_unmet_error": "0",
            "encounters": "8",
            "certified_encounters": "8",
            "unmet_encounters": "0",
            "noise_stop_encounters": "0",
            "resource_unmet_encounters": "0",
            "vector_catastrophe_failure_encounters": "0",
            "maximum_vector_difference": "0.2",
            "catastrophe_limit": "0.5",
            "terminal_patch_enabled": "1",
            "terminal_patch_attempts": "3",
            "terminal_patch_accepts": "3",
            "terminal_patch_rejects": "0",
            "terminal_patch_rejection_mask_or": "0",
            "external_replay_recommended": "0",
            "all_encounters_certified": "1",
            "disposition": "accepted_all_encounters",
        }
        line = campaign.GRID_CERTIFICATE_PREFIX + " ".join(
            f"{key}={fields[key]}" for key in campaign.GRID_CERTIFICATE_FIELDS
        )
        log_path.write_text(line + "\n", encoding="utf-8")
        config = {
            "templates": [
                {
                    "id": "mode3",
                    "card_info": {
                        "adaptive_grid_mode": 3,
                        "adaptive_grid_pilot_diagnostics": 1,
                        "adaptive_grid_fail_on_unmet_error": 0,
                        "adaptive_grid_terminal_patch_certification": 1,
                        "adaptive_grid_error_norm_policy": "vector_rms_quantile_v1",
                        "adaptive_grid_source_model": "fixed_width_bspline1_v1",
                        "adaptive_grid_error_target": "0.01",
                        "adaptive_grid_error_quantile": "0.999",
                        "adaptive_grid_error_catastrophe_limit": "0.5",
                    },
                }
            ]
        }
        evidence = campaign._adaptive_grid_certificate_evidence(
            log_path, {"template_id": "mode3"}, config
        )
        self.assertTrue(evidence["qualification_passed"])
        self.assertEqual(
            evidence["combined_log_sha256"], campaign.sha256_file(log_path)
        )
        fields["terminal_patch_attempts"] = "0"
        fields["terminal_patch_accepts"] = "0"
        line = campaign.GRID_CERTIFICATE_PREFIX + " ".join(
            f"{key}={fields[key]}" for key in campaign.GRID_CERTIFICATE_FIELDS
        )
        log_path.write_text(line + "\n", encoding="utf-8")
        unexercised = campaign._adaptive_grid_certificate_evidence(
            log_path, {"template_id": "mode3"}, config
        )
        self.assertTrue(unexercised["certificate_internal_passed"])
        self.assertFalse(unexercised["terminal_patch_qualification_exercised"])
        self.assertFalse(unexercised["qualification_passed"])

        fields["maximum_vector_difference"] = "1"
        log_path.write_text(
            campaign.GRID_CERTIFICATE_PREFIX
            + " ".join(
                f"{key}={fields[key]}" for key in campaign.GRID_CERTIFICATE_FIELDS
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(campaign.CampaignError, "internally inconsistent"):
            campaign._adaptive_grid_certificate_evidence(
                log_path, {"template_id": "mode3"}, config
            )

        fields["maximum_vector_difference"] = "0.2"
        fields["all_encounters_certified"] = "1"
        fields["disposition"] = "accepted_all_encounters"
        fields["error_target"] = "0.02"
        log_path.write_text(
            campaign.GRID_CERTIFICATE_PREFIX
            + " ".join(
                f"{key}={fields[key]}" for key in campaign.GRID_CERTIFICATE_FIELDS
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(campaign.CampaignError, "target differs"):
            campaign._adaptive_grid_certificate_evidence(
                log_path, {"template_id": "mode3"}, config
            )

    def test_freeze_materializes_factorial_and_declared_coupled_grid_changes(self) -> None:
        fixture = CampaignFixture(self.root)
        policy, manifest = campaign.freeze_campaign(
            fixture.write_request(), self.root / "campaign"
        )
        self.assertEqual(len(manifest["realizations"]), 32)
        self.assertEqual(policy["design"]["pilot_run_count"], "32")
        self.assertEqual(policy["design"]["reference_template_id"], "reference")
        self.assertEqual(
            policy["statistics"]["conservative_total_budget_multiplier"], "3"
        )
        for realization in manifest["realizations"]:
            card = (self.root / "campaign" / realization["card"]).read_bytes()
            template = next(
                item
                for item in policy["templates"]
                if item["id"] == realization["template_id"]
            )
            info = campaign.inspect_campaign_card(
                card,
                template["accelerator"],
                template["parameters"],
                template["grid_patch_contract"],
            )
            self.assertEqual(
                info["campaign_family_sha256"], realization["card_family_sha256"]
            )
            if template["grid_patch_contract"] == campaign.COUPLED_ADAPTIVE_MAXIMA_CONTRACT:
                self.assertEqual(info["adaptive_grid_max_nx"], int(realization["n_x"]))
                self.assertEqual(info["adaptive_grid_max_ny"], int(realization["n_y"]))
        campaign.validate_campaign(
            self.root / "campaign" / "campaign_policy.json",
            self.root / "campaign" / "pilot_manifest.json",
            self.root / "campaign",
        )

    @unittest.skipIf(os.name == "nt", "POSIX executable and relocation fixture")
    def test_frozen_closure_survives_original_mutation_deletion_and_relocation(self) -> None:
        fixture = CampaignFixture(self.root)
        request = fixture.write_request()
        output = self.root / "campaign"
        campaign.freeze_campaign(request, output)

        # Every request-addressed input may disappear or change after freeze.
        # The campaign must neither read nor launch any of these original paths.
        request.write_text('{"superseded":true}\n', encoding="utf-8")
        fixture.source.unlink()
        fixture.reference.unlink()
        fixture.adaptive.unlink()
        fixture.binary.unlink()
        fixture.core_transform.unlink()

        relocated = self.root / "relocated-campaign"
        shutil.move(str(output), str(relocated))
        frozen_python = relocated / campaign.FROZEN_PYTHON_PATH
        frozen_driver = relocated / campaign.FROZEN_DRIVER_PATH
        for command in ("validate", "execute", "analyze"):
            process = subprocess.run(
                [
                    str(frozen_python),
                    "-B",
                    str(frozen_driver),
                    command,
                    "--output-dir",
                    str(relocated),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(
                process.returncode,
                0,
                msg=f"{command} failed after relocation: {process.stderr}",
            )
        self.assertTrue((relocated / "pilot_analysis.json").is_file())

    def test_every_frozen_input_is_hash_authenticated_and_symlinks_fail_closed(self) -> None:
        fixture = CampaignFixture(self.root)
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        frozen = policy["frozen_inputs"]
        bindings = [
            frozen["request"],
            frozen["source_state"],
            frozen["campaign_driver"],
            frozen["replay_helper"],
            frozen["feebl_binary"],
            frozen["python_executable"],
            frozen["gaussian_core_transform"],
            *frozen["templates"],
        ]
        for binding in bindings:
            path = output / binding["path"]
            original = path.read_bytes()
            with self.subTest(path=binding["path"]):
                path.write_bytes(original + b"tamper")
                with self.assertRaises(campaign.CampaignError):
                    campaign.validate_campaign(policy_path, manifest_path, output)
                path.write_bytes(original)

        if os.name != "nt":
            source_path = output / frozen["source_state"]["path"]
            payload = source_path.read_bytes()
            source_path.unlink()
            outside = self.root / "outside-source-state.json"
            outside.write_bytes(payload)
            source_path.symlink_to(outside)
            with self.assertRaisesRegex(campaign.CampaignError, "symlink"):
                campaign.validate_campaign(policy_path, manifest_path, output)

    def test_running_driver_helper_and_python_must_match_frozen_bytes(self) -> None:
        fixture = CampaignFixture(self.root)
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        changed = self.root / "changed-runtime-identity"
        changed.write_bytes(b"not-the-frozen-runtime-identity\n")
        identities = (
            (campaign, "__file__", str(changed), "driver"),
            (campaign.replay, "__file__", str(changed), "helper"),
            (campaign.sys, "executable", str(changed), "interpreter"),
        )
        for owner, attribute, replacement, label in identities:
            original = getattr(owner, attribute)
            try:
                setattr(owner, attribute, replacement)
                with self.subTest(identity=label):
                    with self.assertRaisesRegex(
                        campaign.CampaignError, "SHA-256 mismatch"
                    ):
                        campaign.validate_campaign(
                            policy_path, manifest_path, output
                        )
            finally:
                setattr(owner, attribute, original)

        campaign.validate_campaign(policy_path, manifest_path, output)

    def test_literal_qualification_values_ignore_quoted_and_reject_comment_decoys(self) -> None:
        cases = [
            (
                'note="spectrum_w_min=84.0;";spectrum_w_max=92.0;',
                campaign._one_literal_decimal_card_value,
                "spectrum_w_min",
            ),
            (
                'spectrum_w_min=84.0;# spectrum_w_max=92.0;\nother=1;',
                campaign._one_literal_decimal_card_value,
                "spectrum_w_max",
            ),
            (
                "note='adaptive_grid_source_model=\"reference_cic_v1\";';other=1;",
                campaign._one_literal_string_card_value,
                "adaptive_grid_source_model",
            ),
            (
                'other=1;# adaptive_grid_error_norm_policy="vector_rms_quantile_v1";\n',
                campaign._one_literal_string_card_value,
                "adaptive_grid_error_norm_policy",
            ),
        ]
        for body, reader, key in cases:
            text = f"$PARAMETERS:: p\n{{{body}}}\n"
            block = campaign.replay.selected_block(text, "PARAMETERS", "p")
            with self.subTest(key=key, body=body):
                with self.assertRaises(campaign.CampaignError):
                    reader(text, block, key)

    @unittest.skipIf(os.name == "nt", "WSL worktree Git metadata fixture")
    def test_real_request_generator_separates_physics_and_diagnostics(self) -> None:
        generator_path = (
            ROOT
            / "examples" / "adaptive_particles" / "generate_coupled_pilot_request.py"
        )
        generator_spec = importlib.util.spec_from_file_location(
            "coupled_pilot_request_generator", generator_path
        )
        assert generator_spec is not None and generator_spec.loader is not None
        generator = importlib.util.module_from_spec(generator_spec)
        generator_spec.loader.exec_module(generator)
        binary = self.root / "post-build-feebl-placeholder"
        binary.write_bytes(b"test-only-binary-evidence\n")
        newest_input = max(
            path.stat().st_mtime_ns for path in generator.compilation_inputs()
        )
        os.utime(binary, ns=(newest_input + 1, newest_input + 1))
        source, primary, legacy, smoke = generator.build_requests(
            binary, Path(sys.executable).resolve()
        )
        self.assertEqual(source["schema"], "feebl.coupled-pilot-source-state.v1")
        self.assertEqual(primary["reference_template_id"], "mode2_fixed_width")
        self.assertEqual(
            [item["id"] for item in primary["templates"]],
            ["mode2_fixed_width", "mode3_fixed_width"],
        )
        self.assertEqual(
            len(primary["templates"])
            * len(primary["grid_levels"])
            * len(primary["n_m_levels"])
            * len(primary["pilot_seeds"]),
            32,
        )
        self.assertEqual(primary["bindings"]["physics_family"]["cpu_thread_selector"]["selected"], "7")
        self.assertEqual(legacy["reference_template_id"], "fixed_legacy_diagnostic")
        self.assertEqual(len(legacy["templates"]), 1)
        self.assertIn(
            "diagnostic_baseline_only",
            legacy["bindings"]["physics_family"]["qualification_scope"],
        )
        self.assertEqual(smoke["reference_template_id"], "reference_cic_replay_smoke")
        self.assertEqual(smoke["n_m_levels"], [512, 1024])
        self.assertNotIn("gaussian_core_transform", smoke["bindings"])
        smoke_assignments = generator.parse_card_assignments(
            Path(smoke["templates"][0]["path"]), "smoke test card"
        )
        self.assertEqual(smoke_assignments["n_x"], "16")
        self.assertEqual(smoke_assignments["n_y"], "16")
        self.assertEqual(smoke_assignments["n_z"], "2")
        self.assertEqual(smoke_assignments["num_lumi"], "50000")
        self.assertEqual(smoke_assignments["adaptive_grid_size_classes_x"], '"8,16"')
        self.assertEqual(smoke_assignments["adaptive_grid_size_classes_y"], '"8,16"')
        self.assertEqual(smoke_assignments["adaptive_grid_source_reference_nx"], "64")
        self.assertEqual(smoke_assignments["adaptive_grid_source_reference_ny"], "64")
        self.assertEqual(
            smoke_assignments["adaptive_grid_field_spacing_classes"], '"1,2,4"'
        )
        self.assertEqual(
            smoke["bindings"]["physics_family"]["num_lumi_capacity"], "50000"
        )
        namespaces = source["seed_namespace_registry"]["namespaces"]
        self.assertEqual(
            namespaces["frozen_preregistered_unseen"], list(range(5001, 5065))
        )
        namespace_sets = {name: set(values) for name, values in namespaces.items()}
        names = list(namespace_sets)
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                self.assertTrue(
                    namespace_sets[names[left]].isdisjoint(namespace_sets[names[right]])
                )
        broken_smoke = json.loads(json.dumps(smoke))
        broken_smoke["pilot_seeds"][0] = primary["pilot_seeds"][0]
        with self.assertRaisesRegex(RuntimeError, "global seed namespaces overlap"):
            generator.assert_globally_disjoint_seed_namespaces(
                primary, legacy, broken_smoke
            )
        misleading = self.root / "misleading-card.dat"
        misleading.write_text(
            "$PARAMETERS:: decoy\n{cpu_threads=7;}\n"
            "$PARAMETERS:: p_fccz\n{cpu_threads=70;}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "expected exactly '7'"):
            generator.require_card_assignments(
                misleading, {"cpu_threads": "7"}, "misleading card"
            )
        duplicate = self.root / "duplicate-card.dat"
        duplicate.write_text(
            "$PARAMETERS:: p_fccz\n{cpu_threads=7;cpu_threads=7;}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate assignment 'cpu_threads'"):
            generator.parse_card_assignments(duplicate, "duplicate card")
        duplicate_block = self.root / "duplicate-block-card.dat"
        duplicate_block.write_text(
            "$PARAMETERS:: p_fccz\n{cpu_threads=7;}\n"
            "$PARAMETERS:: p_fccz\n{cpu_threads=7;}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one PARAMETERS::p_fccz"):
            generator.parse_card_assignments(duplicate_block, "duplicate block card")
        accelerator_decoy = self.root / "accelerator-decoy-card.dat"
        accelerator_decoy.write_text(
            "$ACCELERATOR:: decoy\n{energy=45.6;sigma_z=226.562500;}\n"
            "$ACCELERATOR:: fccz\n{energy=45.6;sigma_z=999;}\n"
            "$PARAMETERS:: p_fccz\n{cpu_threads=7;}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "found '999'"):
            generator.require_card_assignments(
                accelerator_decoy,
                {"sigma_z": "226.562500"},
                "accelerator decoy card",
                block_kind="ACCELERATOR",
                block_name="fccz",
            )
        quoted_spoof = self.root / "quoted-spoof-card.dat"
        quoted_spoof.write_text(
            "$PARAMETERS:: p_fccz\n"
            '{adaptive_grid_schedule_save="probe; '
            'adaptive_grid_source_width_x=714.195;";cpu_threads=7;}\n',
            encoding="utf-8",
        )
        quoted_assignments = generator.parse_card_assignments(
            quoted_spoof, "quoted spoof card"
        )
        self.assertEqual(
            quoted_assignments,
            {
                "adaptive_grid_schedule_save": (
                    '"probe; adaptive_grid_source_width_x=714.195;"'
                ),
                "cpu_threads": "7",
            },
        )
        with self.assertRaisesRegex(
            RuntimeError, "missing required assignment 'adaptive_grid_source_width_x'"
        ):
            generator.require_card_assignments(
                quoted_spoof,
                {"adaptive_grid_source_width_x": "714.195"},
                "quoted spoof card",
            )
        quoted_normal = self.root / "quoted-normal-card.dat"
        quoted_normal.write_text(
            '$PARAMETERS:: p_fccz\n{probe="alpha; beta=gamma;";cpu_threads=7;}\n',
            encoding="utf-8",
        )
        self.assertEqual(
            generator.parse_card_assignments(quoted_normal, "quoted normal card"),
            {"probe": '"alpha; beta=gamma;"', "cpu_threads": "7"},
        )
        mode2_assignments = generator.parse_card_assignments(
            Path(primary["templates"][0]["path"]), "Mode-2 card"
        )
        forged_mode3_assignments = dict(mode2_assignments)
        forged_mode3_assignments["adaptive_grid_mode"] = "3"
        forged_mode3_assignments["do_photons"] = "1"
        with self.assertRaisesRegex(RuntimeError, "non-mode assignments differ"):
            generator.require_cross_template_invariants(
                mode2_assignments,
                forged_mode3_assignments,
                "forged Mode-3 card",
            )
        source_path = self.root / "generated-source-state.json"
        source_path.write_bytes(campaign.replay.canonical_bytes(source))
        source_hash = campaign.sha256_file(source_path)
        for request, expected_runs in (
            (primary, 32),
            (legacy, 16),
            (smoke, 16),
        ):
            request["bindings"]["source_state_path"] = str(source_path)
            request["bindings"]["source_state_sha256"] = source_hash
            config = campaign._configuration(request, self.root)
            self.assertEqual(
                len(config["templates"])
                * len(config["grids"])
                * len(config["roots"])
                * len(config["pilot_seeds"]),
                expected_runs,
            )

    def test_design_and_limit_violations_fail_before_output_creation(self) -> None:
        cases = {
            "three_seeds": lambda request: request.__setitem__("pilot_seeds", [1, 2, 3]),
            "overlap": lambda request: request.__setitem__("holdout_seeds", [11, 31, 37, 41]),
            "run_limit": lambda request: request["limits"].__setitem__("maximum_pilot_runs", 31),
            "margin": lambda request: request["endpoints"][0].__setitem__(
                "equivalence_margin_relative", "0.011"
            ),
            "wrong_grid_contract": lambda request: request["templates"][1].__setitem__(
                "grid_patch_contract", campaign.BASE_GRID_ONLY_CONTRACT
            ),
            "missing_total_multiplier": lambda request: request["statistics"].pop(
                "conservative_total_budget_multiplier"
            ),
            "small_total_multiplier": lambda request: request["statistics"].__setitem__(
                "conservative_total_budget_multiplier", "0.9"
            ),
            "large_total_multiplier": lambda request: request["statistics"].__setitem__(
                "conservative_total_budget_multiplier", "4.1"
            ),
            "missing_core_transform": lambda request: request["bindings"].pop(
                "gaussian_core_transform"
            ),
            "unsupported_core_key": lambda request: request["endpoints"][2].__setitem__(
                "key", "spectrum.global_rms_W_GeV_diagnostic_only"
            ),
        }
        for name, mutation in cases.items():
            with self.subTest(name=name):
                case_root = self.root / name
                case_root.mkdir()
                fixture = CampaignFixture(case_root)
                mutation(fixture.request)
                output = case_root / "campaign"
                with self.assertRaises(campaign.CampaignError):
                    campaign.freeze_campaign(fixture.write_request(), output)
                self.assertFalse(output.exists())

    def test_diagnostic_mode1_is_refused_during_freeze(self) -> None:
        fixture = CampaignFixture(self.root)
        mode1 = fixture.adaptive.read_text(encoding="utf-8").replace(
            "adaptive_macroparticle_mode=2", "adaptive_macroparticle_mode=1"
        )
        fixture.adaptive_hash = write_bytes(fixture.adaptive, mode1.encode("utf-8"))
        fixture.request["templates"][1]["sha256"] = fixture.adaptive_hash
        output = self.root / "campaign"
        with self.assertRaisesRegex(
            campaign.CampaignError, "diagnostic Mode 1 has no runtime replay handoff"
        ):
            campaign.freeze_campaign(fixture.write_request(), output)
        self.assertFalse(output.exists())

    def test_resigned_manifest_path_escape_and_card_tamper_fail_closed(self) -> None:
        fixture = CampaignFixture(self.root)
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        manifest_path = output / "pilot_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("self_sha256")
        manifest["realizations"][0]["run_directory"] = "../escape"
        campaign.write_canonical(manifest_path, campaign.with_self_hash(manifest))
        with self.assertRaises(campaign.CampaignError):
            campaign.validate_campaign(
                output / "campaign_policy.json", manifest_path, output
            )

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_preexisting_gaussian_output_blocks_execution_before_launch(self) -> None:
        fixture = CampaignFixture(self.root)
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        manifest = json.loads((output / "pilot_manifest.json").read_text(encoding="utf-8"))
        first_directory = output / manifest["realizations"][0]["run_directory"]
        (first_directory / "gaussian_core.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(campaign.CampaignError, "unrecorded artifacts"):
            campaign.execute_pilot(
                output / "campaign_policy.json",
                output / "pilot_manifest.json",
                output,
            )
        self.assertFalse((self.root / "trace.txt").exists())

    def test_recommendation_logic_covers_all_bounded_actions(self) -> None:
        common = {
            "method_pass": True,
            "method_point_pass": True,
            "required_power_runs": 4,
            "pilot_count": 4,
            "extension_count": 4,
            "holdout_count": 4,
        }
        self.assertEqual(
            campaign._choose_action(
                campaign.Decimal("0.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.4"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.3"),
                campaign.Decimal("3"),
                **common,
            )[0],
            "hold",
        )
        self.assertEqual(
            campaign._choose_action(
                campaign.Decimal("1.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("1.4"),
                campaign.Decimal("1.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("1.3"),
                campaign.Decimal("3"),
                **common,
            )[0],
            "grid",
        )
        self.assertEqual(
            campaign._choose_action(
                campaign.Decimal("0.1"),
                campaign.Decimal("1.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("1.4"),
                campaign.Decimal("0.1"),
                campaign.Decimal("1.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("1.3"),
                campaign.Decimal("3"),
                **common,
            )[0],
            "root",
        )
        self.assertEqual(
            campaign._choose_action(
                campaign.Decimal("1.1"),
                campaign.Decimal("1.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("2.4"),
                campaign.Decimal("1.1"),
                campaign.Decimal("1.1"),
                campaign.Decimal("0.1"),
                campaign.Decimal("2.3"),
                campaign.Decimal("3"),
                **common,
            )[0],
            "joint",
        )
        self.assertEqual(
            campaign._choose_action(
                campaign.Decimal("0.8"),
                campaign.Decimal("0.8"),
                campaign.Decimal("0.8"),
                campaign.Decimal("0.8"),
                campaign.Decimal("3.1"),
                campaign.Decimal("0.8"),
                campaign.Decimal("0.8"),
                campaign.Decimal("0.8"),
                campaign.Decimal("3.1"),
                campaign.Decimal("3"),
                **common,
            )[0],
            "joint",
        )
        replicate_common = dict(common)
        replicate_common["required_power_runs"] = 6
        replicate_common["holdout_count"] = 8
        self.assertEqual(
            campaign._choose_action(
                campaign.Decimal("1.2"),
                campaign.Decimal("0.2"),
                campaign.Decimal("0.2"),
                campaign.Decimal("1.2"),
                campaign.Decimal("2.8"),
                campaign.Decimal("0.2"),
                campaign.Decimal("0.2"),
                campaign.Decimal("0.2"),
                campaign.Decimal("0.6"),
                campaign.Decimal("3"),
                **replicate_common,
            )[0],
            "replicate",
        )
        replicate_common["required_power_runs"] = 9
        self.assertEqual(
            campaign._choose_action(
                campaign.Decimal("1.2"),
                campaign.Decimal("0.2"),
                campaign.Decimal("0.2"),
                campaign.Decimal("1.2"),
                campaign.Decimal("2.8"),
                campaign.Decimal("0.2"),
                campaign.Decimal("0.2"),
                campaign.Decimal("0.2"),
                campaign.Decimal("0.6"),
                campaign.Decimal("3"),
                **replicate_common,
            )[0],
            "insufficient_preregistered_seeds",
        )

    def test_replication_design_uses_new_complete_process_seeds_not_higher_n_m(self) -> None:
        templates = [{"id": "reference"}, {"id": "candidate"}]
        config = {
            "templates": templates,
            "grids": [
                {"id": "g16", "n_x": 16, "n_y": 16},
                {"id": "g32", "n_x": 32, "n_y": 32},
            ],
            "roots": [1000, 2000],
            "pilot_seeds": [11, 13, 17, 19],
            "extension_seeds": [23, 29, 31, 37],
            "holdout_seeds": [101, 103, 107, 109, 113, 127],
            "next_grid": {"id": "g64", "n_x": 64, "n_y": 64},
            "next_n_m": 4000,
        }
        specs, design = campaign._next_stage_run_specs(config, "replicate", 6)
        self.assertEqual(design["materialized_extension_seed_count"], "2")
        self.assertEqual({item["seed"] for item in specs}, {23, 29})
        self.assertTrue({item["seed"] for item in specs}.isdisjoint({11, 13, 17, 19}))
        self.assertEqual({item["n_m"] for item in specs}, {1000, 2000})
        self.assertEqual(len(specs), 2 * 2 * 2 * 2)
        self.assertEqual(
            {item["design_role"] for item in specs},
            {"complete_paired_factorial_replication"},
        )
        holdout_specs, holdout_design = campaign._next_stage_run_specs(
            config, "hold", 5
        )
        self.assertEqual(holdout_design["materialized_holdout_seed_count"], "5")
        self.assertEqual(
            {item["seed"] for item in holdout_specs},
            {101, 103, 107, 109, 113},
        )
        self.assertEqual(len(holdout_specs), 2 * 5)

    def test_runtime_replay_forces_fresh_seed_replication_and_capacity_gate(self) -> None:
        root_action = campaign._apply_runtime_action_constraint(
            "hold", True, [], {"request_initial_state_replay": 1}
        )
        self.assertEqual(root_action[0], "root_replicate")
        self.assertFalse(root_action[1])
        joint_action = campaign._apply_runtime_action_constraint(
            "hold", True, [], {"joint_replay_and_mesh_refinement": 1}
        )
        self.assertEqual(joint_action[0], "joint_replicate")
        self.assertFalse(joint_action[1])
        self.assertEqual(
            campaign._apply_replication_seed_capacity(
                root_action[0], root_action[1], root_action[2], 6, 4, 2
            )[0],
            "root_replicate",
        )
        for available in (0, 1):
            constrained = campaign._apply_replication_seed_capacity(
                joint_action[0], joint_action[1], joint_action[2], 6, 4, available
            )
            self.assertEqual(constrained[0], "diagnostic_joint")
            self.assertFalse(constrained[1])
            self.assertIn(
                "insufficient_preregistered_extension_seeds_after_runtime_constraint",
                constrained[2],
            )
            self.assertIn(
                "extension_seed_pool_exhausted_diagnostic_axis_only_not_qualification",
                constrained[2],
            )
        pure_replication = campaign._apply_replication_seed_capacity(
            "replicate", False, [], 6, 4, 1
        )
        self.assertEqual(pure_replication[0], "insufficient_preregistered_seeds")

    def test_tost_power_planning_and_timing_ranking_are_conservative(self) -> None:
        requested_power = campaign.Decimal("0.9")
        z_alpha = campaign.Decimal(
            str(campaign.NormalDist().inv_cdf(0.95))
        )
        z_power = campaign.Decimal(
            str(campaign.NormalDist().inv_cdf(float((1 + requested_power) / 2)))
        )
        summary = campaign._equivalence_summary(
            [campaign.Decimal("-1"), campaign.Decimal("1"),
             campaign.Decimal("-1"), campaign.Decimal("1")],
            campaign.Decimal("10"),
            campaign.Decimal("0.1"),
            campaign.Decimal("0"),
            campaign.Decimal("1"),
            z_alpha,
            z_power,
        )
        n = int(summary["approximate_power_required_complete_runs"])
        sd = float(summary["paired_sample_standard_deviation"])
        standardized = 1.0 * math.sqrt(n) / sd - float(z_alpha)
        achieved = 2.0 * campaign.NormalDist().cdf(standardized) - 1.0
        self.assertGreaterEqual(achieved, float(requested_power))
        self.assertIn("conservative_normal_TOST", summary["power_formula"])

        timing = campaign._paired_timing_summaries(
            {
                "reference": {(index,): 10_000_000_000 for index in range(4)},
                "fast": {(index,): 8_000_000_000 for index in range(4)},
                "unresolved": {
                    (index,): value
                    for index, value in enumerate(
                        (8_000_000_000, 12_000_000_000,
                         8_000_000_000, 12_000_000_000)
                    )
                },
            },
            "reference",
            campaign.Decimal("0.95"),
        )
        self.assertEqual(timing["fast"]["comparison_to_reference"], "significantly_faster")
        self.assertEqual(
            timing["unresolved"]["comparison_to_reference"], "unresolved_timing_tie"
        )

    def test_seed_exhaustion_emits_nonqualifying_diagnostic_axis_plan(self) -> None:
        config = {
            "templates": [{"id": "reference"}, {"id": "candidate"}],
            "grids": [
                {"id": "g16", "n_x": 16, "n_y": 16},
                {"id": "g32", "n_x": 32, "n_y": 32},
            ],
            "roots": [1000, 2000],
            "pilot_seeds": [11, 13, 17, 19],
            "extension_seeds": [],
            "holdout_seeds": [101, 103, 107, 109],
            "next_grid": {"id": "g64", "n_x": 64, "n_y": 64},
            "next_n_m": 4000,
            "endpoints": [],
        }
        action = campaign._choose_action(
            campaign.Decimal("2"), campaign.Decimal("0.2"),
            campaign.Decimal("0.2"), campaign.Decimal("2"),
            campaign.Decimal("2.5"), campaign.Decimal("2"),
            campaign.Decimal("0.2"), campaign.Decimal("0.2"),
            campaign.Decimal("2.4"), campaign.Decimal("3"),
            True, True, 8, 4, 0, 4,
        )
        self.assertEqual(action[0], "diagnostic_grid")
        self.assertFalse(action[1])
        specs, design = campaign._next_stage_run_specs(config, action[0], 8)
        self.assertTrue(specs)
        self.assertEqual({item["seed"] for item in specs}, {11, 13, 17, 19})
        self.assertTrue(
            all(item["design_role"].startswith("nonqualifying_diagnostic_") for item in specs)
        )
        self.assertFalse(design["qualification_eligible"])
        self.assertEqual(design["materialized_feebl_process_count"], str(len(specs)))
        self.assertEqual(design["materialized_distinct_seed_block_count"], "4")
        status = campaign._solution_verification_design_status(config)
        self.assertFalse(status["absolute_solution_convergence_qualification_ready"])
        self.assertIn("fewer_than_three_preregistered_grid_levels", status["blockers"])
    def test_hash_comment_accepts_only_numeric_plus_minus_uncertainty(self) -> None:
        run = self.root / "run"
        run.mkdir()
        path = run / "spectrum.out"
        spec = {
            "id": "rms_w",
            "file": "spectrum.out",
            "format": "hash_comment",
            "key": "rms_W",
        }
        path.write_text("#rms_W 0.123 +- 0.004\n", encoding="utf-8")
        value, _ = campaign.parse_endpoint(run, spec)
        self.assertEqual(value, "0.123")
        path.write_text("#rms_W 0.123 +- not-a-number\n", encoding="utf-8")
        with self.assertRaisesRegex(campaign.CampaignError, "was not found"):
            campaign.parse_endpoint(run, spec)

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_execute_resume_analyze_and_holdout_gate(self) -> None:
        fixture = CampaignFixture(self.root)
        request = fixture.write_request()
        output = self.root / "campaign"
        campaign.freeze_campaign(request, output)
        self.assertFalse((output / "next_stage").exists())
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        self.assertEqual(campaign.execute_pilot(policy_path, manifest_path, output), 0)
        trace_path = output / campaign.FROZEN_BINARY_PATH.parent / "trace.txt"
        trace_before = trace_path.read_bytes()
        self.assertEqual(len(trace_before.splitlines()), 32)
        frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_trace = "".join(
            f"{item['n_x']},{item['n_y']},{item['n_m']},{item['seed']}\n"
            for item in frozen_manifest["realizations"]
        ).encode("utf-8")
        self.assertEqual(trace_before, expected_trace)
        state = json.loads((output / "pilot_state.json").read_text(encoding="utf-8"))
        adaptive_records = [
            record
            for realization, record in zip(
                frozen_manifest["realizations"], state["completed"]
            )
            if realization["template_id"] == "adaptive"
        ]
        self.assertTrue(adaptive_records)
        self.assertTrue(
            all(record["runtime_replay_signal"]["present"] for record in adaptive_records)
        )
        self.assertTrue(
            all(
                record["runtime_replay_signal"]["schema"]
                == "feebl.adaptive-root-replay-signal.v2"
                for record in adaptive_records
            )
        )
        self.assertEqual(campaign.execute_pilot(policy_path, manifest_path, output), 0)
        self.assertEqual(trace_path.read_bytes(), trace_before)
        analysis, next_manifest = campaign.analyze_pilot(
            policy_path, manifest_path, output
        )
        self.assertEqual(trace_path.read_bytes(), trace_before)
        self.assertTrue(analysis["pilot_gate_passed"])
        self.assertEqual(analysis["recommendation"], "hold")
        self.assertTrue(next_manifest["holdout_execution_authorized"])
        self.assertEqual(
            next_manifest["stage_design"]["seed_populations"],
            ["reserved_holdout"],
        )
        self.assertEqual(
            {int(item["seed"]) for item in next_manifest["runs"]},
            {101, 103, 107, 109},
        )
        adaptive = next(
            item for item in analysis["template_results"] if item["template_id"] == "adaptive"
        )
        self.assertTrue(
            adaptive["method_equivalence"]["passes_all_frozen_equivalence_margins"]
        )
        self.assertEqual(
            len(adaptive["method_equivalence"]["equal_cell_paired_shifts"]), 12
        )
        self.assertEqual(adaptive["recommendation"], "root_replicate")
        self.assertFalse(adaptive["pilot_gate_passed"])
        self.assertEqual(
            adaptive["runtime_controller_signals"][
                "terminal_recommended_action_counts"
            ],
            {"request_initial_state_replay": "4"},
        )
        analysis_sha = campaign.sha256_file(output / "pilot_analysis.json")
        with self.assertRaisesRegex(campaign.CampaignError, "state is missing"):
            campaign.analyze_next_stage(policy_path, manifest_path, output)
        with self.assertRaisesRegex(campaign.CampaignError, "explicit --allow-holdout"):
            campaign.execute_next_stage(
                policy_path,
                manifest_path,
                output,
                analysis_sha,
                allow_holdout=False,
            )
        self.assertEqual(
            campaign.execute_next_stage(
                policy_path,
                manifest_path,
                output,
                analysis_sha,
                allow_holdout=True,
            ),
            0,
        )
        trace_after_holdout = trace_path.read_bytes()
        self.assertEqual(len(trace_after_holdout.splitlines()), 40)
        self.assertEqual(
            campaign.execute_next_stage(
                policy_path,
                manifest_path,
                output,
                analysis_sha,
                allow_holdout=True,
            ),
            0,
        )
        self.assertEqual(trace_path.read_bytes(), trace_after_holdout)
        final = campaign.analyze_next_stage(policy_path, manifest_path, output)
        self.assertEqual(final["analysis_kind"], "sequestered_holdout_confirmation")
        self.assertEqual(
            final["selected_template_id"], analysis["selected_template_id"]
        )
        self.assertTrue(final["holdout_declared_estimator_gate_passed"])
        self.assertFalse(final["holdout_gate_passed"])
        self.assertIsNone(final["qualified_template_id"])
        self.assertIn(
            "absolute_solution_verification_design_is_hard_blocked",
            final["decision_reasons"],
        )
        self.assertFalse(
            final["solution_verification_design"][
                "absolute_solution_convergence_qualification_ready"
            ]
        )
        self.assertFalse(
            final["selection_firewall"][
                "holdout_observations_used_to_rerank_or_replace_selection"
            ]
        )
        self.assertFalse(final["absolute_gaussian_fit_model_validity_established"])
        self.assertTrue(
            final["gaussian_fit_diagnostics"]["direct_diagnostic_guard_passed"]
        )
        policy, manifest, config = campaign.validate_campaign(
            policy_path, manifest_path, output
        )
        campaign.validate_final_analysis(
            output / "final_analysis.json",
            policy_path,
            manifest_path,
            output,
            policy,
            manifest,
            config,
        )
        self.assertEqual(campaign.main(["validate", "--output-dir", str(output)]), 0)
        with self.assertRaisesRegex(campaign.CampaignError, "refusing to overwrite"):
            campaign.analyze_next_stage(policy_path, manifest_path, output)
        final_path = output / "final_analysis.json"
        original_final = final_path.read_bytes()
        forged = json.loads(original_final.decode("utf-8"))
        forged["qualified_template_id"] = analysis["selected_template_id"]
        forged = campaign.with_self_hash(
            {key: value for key, value in forged.items() if key != "self_sha256"}
        )
        final_path.write_bytes(campaign.replay.canonical_bytes(forged))
        with self.assertRaisesRegex(campaign.CampaignError, "recomputation"):
            campaign.validate_final_analysis(
                final_path,
                policy_path,
                manifest_path,
                output,
                policy,
                manifest,
                config,
            )
        final_path.write_bytes(original_final)

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_extension_final_analysis_uses_fresh_roots_and_cannot_promote(self) -> None:
        fixture = CampaignFixture(self.root)
        fixture.request["templates"] = [fixture.request["templates"][1]]
        fixture.request["reference_template_id"] = "adaptive"
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        self.assertEqual(campaign.execute_pilot(policy_path, manifest_path, output), 0)
        analysis, next_manifest = campaign.analyze_pilot(
            policy_path, manifest_path, output
        )
        self.assertEqual(analysis["recommendation"], "root_replicate")
        analysis_sha = campaign.sha256_file(output / "pilot_analysis.json")
        self.assertEqual(
            campaign.execute_next_stage(
                policy_path,
                manifest_path,
                output,
                analysis_sha,
                allow_holdout=False,
            ),
            0,
        )
        final = campaign.analyze_next_stage(policy_path, manifest_path, output)
        self.assertEqual(final["analysis_kind"], "extension_or_axis_followup")
        self.assertEqual(final["completed_next_stage_action"], "root_replicate")
        used = final["seed_accounting"]["used_extension_seeds"]
        self.assertEqual(used, ["23"])
        self.assertIsNone(final["qualified_template_id"])
        self.assertFalse(final["expanded_gate_passed"])
        self.assertEqual(
            final["recommended_disposition"],
            "diagnostic_failure_no_manual_promotion",
        )
        self.assertTrue(
            all(
                run["seed"] == "23"
                for run in next_manifest["runs"]
                if run["seed_population"] == "preregistered_extension"
            )
        )

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_gaussian_direct_fit_diagnostic_blocks_holdout_confirmation(self) -> None:
        fixture = CampaignFixture(self.root)
        changed = FAKE_CORE_TRANSFORM.replace(
            '"sigma_fit_uncertainty_GeV_diagnostic": "0.0001"',
            '"sigma_fit_uncertainty_GeV_diagnostic": "1"',
        )
        fixture.core_transform_hash = write_bytes(
            fixture.core_transform, changed.encode("utf-8")
        )
        fixture.request["bindings"]["gaussian_core_transform"][
            "sha256"
        ] = fixture.core_transform_hash
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        self.assertEqual(campaign.execute_pilot(policy_path, manifest_path, output), 0)
        analysis, _ = campaign.analyze_pilot(policy_path, manifest_path, output)
        self.assertTrue(analysis["pilot_gate_passed"])
        self.assertEqual(
            campaign.execute_next_stage(
                policy_path,
                manifest_path,
                output,
                campaign.sha256_file(output / "pilot_analysis.json"),
                allow_holdout=True,
            ),
            0,
        )
        final = campaign.analyze_next_stage(policy_path, manifest_path, output)
        self.assertFalse(final["holdout_gate_passed"])
        self.assertIsNone(final["qualified_template_id"])
        self.assertFalse(
            final["gaussian_fit_diagnostics"]["direct_diagnostic_guard_passed"]
        )
        self.assertIn(
            "holdout_gaussian_direct_fit_diagnostic_failed",
            final["decision_reasons"],
        )

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_output_tampering_is_detected_and_biased_method_is_not_selected(self) -> None:
        fixture = CampaignFixture(self.root, adaptive_bias="2")
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        self.assertEqual(campaign.execute_pilot(policy_path, manifest_path, output), 0)
        analysis, _ = campaign.analyze_pilot(policy_path, manifest_path, output)
        adaptive = next(
            item for item in analysis["template_results"] if item["template_id"] == "adaptive"
        )
        self.assertFalse(
            adaptive["method_equivalence"]["passes_all_frozen_equivalence_margins"]
        )
        self.assertEqual(adaptive["recommendation"], "rejected_method_bias")
        self.assertNotEqual(analysis["selected_template_id"], "adaptive")
        policy, manifest, config = campaign.validate_campaign(
            policy_path, manifest_path, output
        )
        first_output = output / manifest["realizations"][0]["run_directory"] / "out.dat"
        original_output = first_output.read_bytes()
        first_output.write_text("lumi=999;\n", encoding="utf-8")
        with self.assertRaises(campaign.CampaignError):
            campaign.validate_state(
                output / "pilot_state.json",
                policy_path,
                manifest_path,
                output,
                policy,
                manifest,
                config,
            )
        first_output.write_bytes(original_output)
        campaign.validate_state(
            output / "pilot_state.json",
            policy_path,
            manifest_path,
            output,
            policy,
            manifest,
            config,
        )
        core_json = first_output.with_name(campaign.GAUSSIAN_CORE_JSON)
        original_core_json = core_json.read_bytes()
        forged_core = json.loads(original_core_json.decode("utf-8"))
        forged_core["qualification_binding"]["run_id"] = "forged-run"
        core_json.write_bytes(campaign.replay.canonical_bytes(forged_core) + b"\n")
        with self.assertRaisesRegex(
            campaign.CampaignError, "qualification binding run_id"
        ):
            campaign.validate_state(
                output / "pilot_state.json",
                policy_path,
                manifest_path,
                output,
                policy,
                manifest,
                config,
            )
        core_json.write_bytes(original_core_json)
        core_csv = first_output.with_name(campaign.GAUSSIAN_CORE_CSV)
        core_csv.write_bytes(core_csv.read_bytes() + b"tampered\n")
        with self.assertRaisesRegex(campaign.CampaignError, "Gaussian-core CSV"):
            campaign.validate_state(
                output / "pilot_state.json",
                policy_path,
                manifest_path,
                output,
                policy,
                manifest,
                config,
            )

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_gaussian_core_transform_failure_and_preexisting_output_fail_closed(self) -> None:
        overwrite_root = self.root / "overwrite"
        overwrite_root.mkdir()
        overwrite_fixture = CampaignFixture(overwrite_root)
        overwrite_output = overwrite_root / "campaign"
        campaign.freeze_campaign(
            overwrite_fixture.write_request(), overwrite_output
        )
        manifest = json.loads(
            (overwrite_output / "pilot_manifest.json").read_text(encoding="utf-8")
        )
        first_run = overwrite_output / manifest["realizations"][0]["run_directory"]
        (first_run / campaign.GAUSSIAN_CORE_JSON).write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(campaign.CampaignError, "unrecorded artifacts"):
            campaign.execute_pilot(
                overwrite_output / "campaign_policy.json",
                overwrite_output / "pilot_manifest.json",
                overwrite_output,
            )

        failure_root = self.root / "failure"
        failure_root.mkdir()
        failure_fixture = CampaignFixture(failure_root)
        failed_transform = FAKE_CORE_TRANSFORM.replace(
            "parser = argparse.ArgumentParser()",
            "raise SystemExit(2)\nparser = argparse.ArgumentParser()",
        )
        failure_fixture.core_transform_hash = write_bytes(
            failure_fixture.core_transform, failed_transform.encode("utf-8")
        )
        failure_fixture.request["bindings"]["gaussian_core_transform"][
            "sha256"
        ] = failure_fixture.core_transform_hash
        failure_output = failure_root / "campaign"
        campaign.freeze_campaign(
            failure_fixture.write_request(), failure_output
        )
        with self.assertRaisesRegex(campaign.CampaignError, "transform returned 2"):
            campaign.execute_pilot(
                failure_output / "campaign_policy.json",
                failure_output / "pilot_manifest.json",
                failure_output,
            )
        failed_state = json.loads(
            (failure_output / "pilot_state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(failed_state["status"], "failed")
        self.assertEqual(failed_state["failure"]["kind"], "invalid_output")

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_failed_gate_materializes_only_pilot_followup_seeds(self) -> None:
        fixture = CampaignFixture(self.root)
        for endpoint in fixture.request["endpoints"]:
            endpoint["target_relative_error"] = "0.00001"
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        self.assertEqual(campaign.execute_pilot(policy_path, manifest_path, output), 0)
        analysis, next_manifest = campaign.analyze_pilot(
            policy_path, manifest_path, output
        )
        self.assertFalse(analysis["pilot_gate_passed"])
        self.assertIn(analysis["recommendation"], campaign.NEXT_STAGE_ACTIONS - {"hold"})
        self.assertFalse(next_manifest["holdout_execution_authorized"])
        self.assertNotIn("reserved_holdout", next_manifest["stage_design"]["seed_populations"])
        if analysis["recommendation"] == "insufficient_preregistered_seeds":
            self.assertFalse(next_manifest["execution_authorized"])
            self.assertEqual(next_manifest["runs"], [])
        with self.assertRaisesRegex(
            campaign.CampaignError,
            "nonholdout follow-up|insufficient preregistered seeds",
        ):
            campaign.execute_next_stage(
                policy_path,
                manifest_path,
                output,
                campaign.sha256_file(output / "pilot_analysis.json"),
                allow_holdout=True,
            )

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_strict_exit_75_is_bound_request_only_failure(self) -> None:
        fixture = CampaignFixture(self.root)
        fixture.request["templates"] = [fixture.request["templates"][1]]
        fixture.request["reference_template_id"] = "adaptive"
        fixture.binary_hash = write_bytes(
            fixture.binary, STRICT_FEEBL.encode("utf-8")
        )
        fixture.binary.chmod(fixture.binary.stat().st_mode | stat.S_IXUSR)
        fixture.request["bindings"]["feebl_binary_sha256"] = fixture.binary_hash
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        self.assertEqual(
            campaign.execute_pilot(policy_path, manifest_path, output),
            campaign.replay.REPLAY_REQUIRED_EXIT_CODE,
        )
        state = json.loads((output / "pilot_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["completed"], [])
        self.assertEqual(state["failure"]["kind"], "runtime_replay_required")
        self.assertEqual(
            state["failure"]["runtime_replay_signal"]["parsed_fields"]["exit_code"],
            "75",
        )
        first = json.loads(manifest_path.read_text(encoding="utf-8"))["realizations"][0]
        self.assertFalse((output / first["run_directory"] / "out.dat").exists())
        with self.assertRaisesRegex(campaign.CampaignError, "failed"):
            campaign.execute_pilot(policy_path, manifest_path, output)
        policy, manifest, config = campaign.validate_campaign(
            policy_path, manifest_path, output
        )
        campaign.validate_state(
            output / "pilot_state.json",
            policy_path,
            manifest_path,
            output,
            policy,
            manifest,
            config,
        )
        failed_log = output / first["run_directory"] / "log.txt"
        failed_log.write_text(
            failed_log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(campaign.CampaignError, "SHA-256 mismatch"):
            campaign.validate_state(
                output / "pilot_state.json",
                policy_path,
                manifest_path,
                output,
                policy,
                manifest,
                config,
            )

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_malformed_exit_75_is_campaign_integrity_code_4(self) -> None:
        fixture = CampaignFixture(self.root)
        fixture.request["templates"] = [fixture.request["templates"][1]]
        fixture.request["reference_template_id"] = "adaptive"
        malformed = STRICT_FEEBL.replace(
            "FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V2",
            "FEEBL_ADAPTIVE_ROOT_REPLAY_REQUIRED_V1",
        )
        fixture.binary_hash = write_bytes(fixture.binary, malformed.encode("utf-8"))
        fixture.binary.chmod(fixture.binary.stat().st_mode | stat.S_IXUSR)
        fixture.request["bindings"]["feebl_binary_sha256"] = fixture.binary_hash
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        self.assertEqual(campaign.execute_pilot(policy_path, manifest_path, output), 4)
        state = json.loads((output / "pilot_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["failure"]["exit_code"], "75")
        self.assertEqual(state["failure"]["kind"], "invalid_runtime_signal")
        self.assertNotIn("runtime_replay_signal", state["failure"])

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_adaptive_success_without_runtime_signal_fails_closed(self) -> None:
        fixture = CampaignFixture(self.root)
        fixture.request["templates"] = [fixture.request["templates"][1]]
        fixture.request["reference_template_id"] = "adaptive"
        no_signal = FAKE_FEEBL.replace("if adaptive_mode:\n", "if adaptive_mode and False:\n")
        fixture.binary_hash = write_bytes(fixture.binary, no_signal.encode("utf-8"))
        fixture.binary.chmod(fixture.binary.stat().st_mode | stat.S_IXUSR)
        fixture.request["bindings"]["feebl_binary_sha256"] = fixture.binary_hash
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        self.assertEqual(
            campaign.execute_pilot(policy_path, manifest_path, output), 4
        )
        state = json.loads((output / "pilot_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["failure"]["kind"], "invalid_runtime_signal")

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_terminal_joint_signal_constrains_action_and_next_signal_failure(self) -> None:
        fixture = CampaignFixture(self.root, adaptive_bias="0")
        fixture.request["templates"] = [fixture.request["templates"][1]]
        fixture.request["reference_template_id"] = "adaptive"
        joint_then_missing = FAKE_FEEBL.replace(
            "recommended_action=request_initial_state_replay",
            "recommended_action=joint_replay_and_mesh_refinement",
        ).replace(
            "if adaptive_mode:\n",
            "if adaptive_mode and nm < 4000:\n",
        )
        fixture.binary_hash = write_bytes(
            fixture.binary, joint_then_missing.encode("utf-8")
        )
        fixture.binary.chmod(fixture.binary.stat().st_mode | stat.S_IXUSR)
        fixture.request["bindings"]["feebl_binary_sha256"] = fixture.binary_hash
        output = self.root / "campaign"
        campaign.freeze_campaign(fixture.write_request(), output)
        policy_path = output / "campaign_policy.json"
        manifest_path = output / "pilot_manifest.json"
        self.assertEqual(campaign.execute_pilot(policy_path, manifest_path, output), 0)
        analysis, next_manifest = campaign.analyze_pilot(
            policy_path, manifest_path, output
        )
        self.assertEqual(analysis["recommendation"], "joint_replicate")
        self.assertFalse(analysis["pilot_gate_passed"])
        result = analysis["template_results"][0]
        self.assertEqual(
            result["runtime_controller_signals"][
                "terminal_recommended_action_counts"
            ],
            {"joint_replay_and_mesh_refinement": "4"},
        )
        self.assertIn(
            "runtime_controller_requested_joint_replay_and_mesh_refinement",
            result["recommendation_reasons"],
        )
        self.assertEqual(
            next_manifest["stage_design"]["seed_populations"],
            ["pilot_followup", "preregistered_extension"],
        )
        extension_count = int(
            next_manifest["stage_design"]["materialized_extension_seed_count"]
        )
        self.assertGreaterEqual(extension_count, 1)
        extension_ids = {23, 29, 31, 37}
        used_extension_ids = {
            int(run["seed"])
            for run in next_manifest["runs"]
            if run["seed_population"] == "preregistered_extension"
        }
        self.assertEqual(
            used_extension_ids,
            set([23, 29, 31, 37][:extension_count]),
        )
        analysis_sha = campaign.sha256_file(output / "pilot_analysis.json")
        self.assertEqual(
            campaign.execute_next_stage(
                policy_path,
                manifest_path,
                output,
                analysis_sha,
                allow_holdout=False,
            ),
            4,
        )
        next_state_path = output / "next_stage" / "next_stage_state.json"
        next_state = json.loads(next_state_path.read_text(encoding="utf-8"))
        self.assertEqual(next_state["status"], "failed")
        self.assertEqual(next_state["failure"]["kind"], "invalid_runtime_signal")
        policy, manifest, config = campaign.validate_campaign(
            policy_path, manifest_path, output
        )
        _, validated_next = campaign.validate_analysis_and_next(
            policy_path, manifest_path, output, policy, manifest, config
        )
        campaign.validate_next_state(
            next_state_path,
            output / "pilot_analysis.json",
            output / "next_stage" / "next_stage_manifest.json",
            output,
            validated_next,
            config,
        )
        failed_run = next_manifest["runs"][0]
        failed_log = output / failed_run["run_directory"] / "log.txt"
        failed_log.write_text(
            failed_log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(campaign.CampaignError, "SHA-256 mismatch"):
            campaign.validate_next_state(
                next_state_path,
                output / "pilot_analysis.json",
                output / "next_stage" / "next_stage_manifest.json",
                output,
                validated_next,
                config,
            )


if __name__ == "__main__":
    unittest.main()
