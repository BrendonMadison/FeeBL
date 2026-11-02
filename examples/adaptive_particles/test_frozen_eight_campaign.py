#!/usr/bin/env python3
"""Standard-library tests for the frozen FCC-Z serial campaign harness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "adaptive_particles" / "frozen_eight_campaign.py"
SPEC = importlib.util.spec_from_file_location("frozen_eight_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def rewrite_frozen_manifest(fixture: "Fixture", manifest: dict[str, object]) -> None:
    write_json(fixture.manifest, manifest)
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["input_sha256"]["frozen_manifest"] = sha256(fixture.manifest)
    write_json(fixture.config, config)


def card_text(sigma_um: str, seed: int, variant: str) -> str:
    sign = "-0.002797928" if variant == "opp" else "0.002797928"
    return f"""$ACCELERATOR:: fccz
{{energy=45.6;particles=24.3000;
beta_x=25;beta_y=0.8;
emitt_x=63.0;emitt_y=0.13;
sigma_z={sigma_um};espread.1=0.0026;espread.2=0.0026;
which_espread.1=3;which_espread.2=3;dist_z=0;
f_rep=3288.131;n_b=10000;waist_y=0;
angle_x.1=0.01500000;angle_x.2=0.01500000;
angle_phi.1=0.0;angle_phi.2=0.0;}}

$PARAMETERS:: p_fccz
{{n_x=512;n_y=512;n_z=160;
n_t=1;n_m=250000;
cut_x=1389005.7;cut_y=4096.62;cut_z=3.0*sigma_z.1;
force_symmetric=0;electron_ratio=1.0;
do_photons=1;store_photons=0;photon_ratio=1.0;
ecm_min=1.98*energy.1;
do_hadrons=0;store_hadrons=0;do_coherent=0;do_trident=0;
do_pairs=0;track_pairs=0;store_pairs=0;do_compt=0;do_muons=0;
grids=1;integration_method=2;electron_distribution_rho=2;
rndm_load=0;rndm_save=0;rndm_seed={seed};
load_beam=0;charge_sign=-1.0;
do_espread=1;do_eloss=1;
waist_mode=1;waist_fraction=1.0;
crab_full_crossing_angle=0.03;
crab_longitudinal_sign.1=1;crab_longitudinal_sign.2=-1;
waist_symplecticity_tolerance=1e-9;
store_native=1;do_lumi=1;num_lumi=500000;
ip_dispersion.1=0.002797928;ip_dispersion.2={sign};
spectrum_w_min=84.0;spectrum_w_max=92.0;spectrum_w_bins=4000;
hist_ee_bins=1010;hist_ee_max=2.02*energy.1;}}
"""


FAKE_BINARY = r'''#!/usr/bin/env python3
import os
import re
from pathlib import Path

card = Path("acc.dat").read_text(encoding="utf-8")
mode = int(re.search(r"adaptive_macroparticle_mode\s*=\s*([0-9]+)", card).group(1))
adaptive = mode > 0
base = 101.0 if adaptive else 100.0
Path("out.dat").write_text(
    f"lumi_ee = {base}\nlumi_ee_high = {base * 0.9}\n", encoding="utf-8"
)
Path("lumi.ee.out.dLdW.fne").write_text("# fake spectrum\n90 1\n91 2\n", encoding="utf-8")
print("FAKE_FEEBL_COMPLETE arm=" + ("adaptive" if adaptive else "control"))
if adaptive:
    Path("adaptive_policy.json").write_text('{"policy":"fixed"}\n', encoding="utf-8")
    Path("adaptive_trace.json").write_text('{"trace":"realization"}\n', encoding="utf-8")
    Path("adaptive_schedule.json").write_text('{"schedule":[1]}\n', encoding="utf-8")
    unmet = 1 if os.environ.get("FAKE_UNMET") == "1" else 0
    certificate_target = os.environ.get("FAKE_CERTIFICATE_TARGET", "0.01")
    terminal_patch_enabled = os.environ.get("FAKE_TERMINAL_PATCH_ENABLED", "0")
    certified = 3 if unmet else 4
    disposition = "unqualified_diagnostics" if unmet else "accepted_all_encounters"
    print(
        "FEEBL_ADAPTIVE_MACROPARTICLE_SUMMARY_V1 "
        "requested_mode=2 actual_mode=2 fail_on_unmet_error=1 "
        "live_distinct_roots1=10 representative_visits1=20 "
        "many_root_representative_visits1=2 maximum_weight_ess1=9 "
        "controller=coupled_error_budget_v1 controller_replay_required=0 "
        "controller_action_unmet=0 physical_primary_population_mutations=0 "
        "source_enters_lumi_ee=0 source_enters_stochastic_physics=0 max_bytes=4096"
    )
    print("FEEBL_ADAPTIVE_ENCOUNTER_V1 error_target_met=" + ("0" if unmet else "1") +
          " vector_catastrophe_guard_passed=1")
    print(
        "FEEBL_ADAPTIVE_GRID_CERTIFICATE_V1 "
        "schema=feebl.adaptive-grid-certificate.v1 requested_mode=2 actual_mode=2 "
        "error_norm_policy=vector_rms_quantile_v1 source_model=fixed_width_bspline1_v1 "
        "error_budget_sha256=not_applicable "
        "card_hash=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc "
        f"error_target={certificate_target} error_quantile=0.999 "
        "error_reference_field=0 error_absolute_floor_fraction=0 "
        "error_catastrophe_absolute_fraction=0 error_exposure_limit=0 "
        "maximum_beam_above_target_exposure=0 "
        "pilot_diagnostics=0 fail_on_unmet_error=1 encounters=4 "
        f"certified_encounters={certified} unmet_encounters={unmet} noise_stop_encounters=0 "
        "resource_unmet_encounters=0 vector_catastrophe_failure_encounters=0 "
        "maximum_vector_difference=0.005 catastrophe_limit=0.5 "
        f"terminal_patch_enabled={terminal_patch_enabled} terminal_patch_attempts=0 "
        "terminal_patch_accepts=0 terminal_patch_rejects=0 terminal_patch_rejection_mask_or=0 "
        f"external_replay_recommended=0 all_encounters_certified={0 if unmet else 1} "
        f"disposition={disposition}"
    )
'''


FAKE_FITTER = r'''#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json

parser = argparse.ArgumentParser()
parser.add_argument("spectrum")
parser.add_argument("--json", required=True)
parser.add_argument("--csv", required=True)
parser.add_argument("--expected-seed", required=True)
parser.add_argument("--expected-w-min", required=True)
parser.add_argument("--expected-w-max", required=True)
parser.add_argument("--expected-bins", required=True)
parser.add_argument("--expected-run-id", required=True)
args = parser.parse_args()
data = open(args.spectrum, "rb").read()
result = {
    "schema": "feebl.gaussian-core-spectrum.v1",
    "status": "accepted_core_fit_not_physics_qualification",
    "input": {"file_name": "lumi.ee.out.dLdW.fne", "sha256": hashlib.sha256(data).hexdigest()},
    "qualification_binding": {
        "status": "qualification_input_bound",
        "native_format": "FEEBL-NATIVE 1.0",
        "feebl_version": "fixture",
        "rndm_seed": args.expected_seed,
        "run_id": args.expected_run_id,
        "spectrum_w_min_GeV": args.expected_w_min,
        "spectrum_w_max_GeV": args.expected_w_max,
        "spectrum_w_bins": args.expected_bins,
        "lattice_phase_proven": True,
    },
    "core_fit": {
        "mu_core_GeV": "91.0",
        "sigma_core_GeV": "0.2",
        "gaussian_core_area": "5.0",
        "peak_dL_dW": "10.0"
    }
}
with open(args.json, "x", encoding="utf-8", newline="") as stream:
    json.dump(result, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
with open(args.csv, "x", encoding="utf-8", newline="") as stream:
    fields = [
        "schema", "input_file_name", "input_sha256", "binding_status", "run_id",
        "rndm_seed", "spectrum_w_min_GeV", "spectrum_w_max_GeV", "spectrum_w_bins",
        "lattice_phase_proven", "status", "mu_core_GeV", "sigma_core_GeV",
        "gaussian_core_area", "peak_dL_dW",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerow({
        "schema": result["schema"],
        "input_file_name": result["input"]["file_name"],
        "input_sha256": result["input"]["sha256"],
        "binding_status": result["qualification_binding"]["status"],
        "run_id": result["qualification_binding"]["run_id"],
        "rndm_seed": result["qualification_binding"]["rndm_seed"],
        "spectrum_w_min_GeV": result["qualification_binding"]["spectrum_w_min_GeV"],
        "spectrum_w_max_GeV": result["qualification_binding"]["spectrum_w_max_GeV"],
        "spectrum_w_bins": result["qualification_binding"]["spectrum_w_bins"],
        "lattice_phase_proven": "true",
        "status": result["status"],
        "mu_core_GeV": result["core_fit"]["mu_core_GeV"],
        "sigma_core_GeV": result["core_fit"]["sigma_core_GeV"],
        "gaussian_core_area": result["core_fit"]["gaussian_core_area"],
        "peak_dL_dW": result["core_fit"]["peak_dL_dW"],
    })
print("fake Gaussian-core fit accepted")
'''


class Fixture:
    def __init__(
        self,
        root: Path,
        *,
        unmet: bool = False,
        with_history: bool = False,
        certificate_target: str = "0.01",
        terminal_patch_enabled: str = "0",
        terminal_patch_card: str = "0",
    ):
        self.root = root
        self.source_root = root / "source"
        self.output = root / "campaign"
        self.template = root / "adaptive-template.dat"
        self.binary = root / "fake-feebl.py"
        self.fitter = root / "fake-fitter.py"
        self.config = root / "config.json"
        self.manifest = root / "frozen.json"
        self.additional: Path | None = None
        self.binary.write_text(FAKE_BINARY, encoding="utf-8")
        self.fitter.write_text(FAKE_FITTER, encoding="utf-8")
        self.template.write_text(
            """$PARAMETERS:: p_fccz
{compute_backend="cpu";
n_x=64;
adaptive_grid_mode=2;
adaptive_grid_error_norm_policy="vector_rms_quantile_v1";
adaptive_grid_error_budget_sha256="";
adaptive_grid_source_model="fixed_width_bspline1_v1";
adaptive_grid_error_target=0.01;
adaptive_grid_error_quantile=0.999;
adaptive_grid_error_catastrophe_limit=0.5;
adaptive_grid_error_reference_field=0;
adaptive_grid_error_absolute_floor_fraction=0;
adaptive_grid_error_catastrophe_absolute_fraction=0;
adaptive_grid_error_exposure_limit=0;
adaptive_grid_fail_on_unmet_error=1;
adaptive_grid_pilot_diagnostics=0;
adaptive_grid_terminal_patch_certification=0;
adaptive_grid_schedule_save="adaptive_schedule.json";
adaptive_macroparticle_mode=2;
adaptive_macroparticle_policy_save="adaptive_policy.json";
adaptive_macroparticle_trace_save="adaptive_trace.json";}
""".replace(
                "adaptive_grid_terminal_patch_certification=0;",
                "adaptive_grid_terminal_patch_certification="
                + terminal_patch_card
                + ";",
            ),
            encoding="utf-8",
        )
        common = {
            "beam_energy_gev": 45.6,
            "n_x": 512,
            "n_y": 512,
            "n_z": 160,
            "n_m": 250000,
            "num_lumi": 500000,
            "crab_full_crossing_angle_rad": 0.03,
            "crab_longitudinal_sign_1": 1,
            "crab_longitudinal_sign_2": -1,
            "ip_dispersion_1_m": 0.002797928,
            "spectrum_w_min_gev": 84.0,
            "spectrum_w_max_gev": 92.0,
            "spectrum_w_bins": 4000,
        }
        runs = []
        for sigma_tag, sigma_mm, sigma_um in (
            ("sz0002266", 0.2265625, "226.562500"),
            ("sz0145000", 14.5, "14500.0"),
        ):
            for seed in (1, 2):
                for variant in ("opp", "same"):
                    run_id = f"{sigma_tag}_s{seed}_{variant}"
                    rel = Path("benchmark_results") / "FeeBL_FCCZ_sz_ladder_CPU_GPU_20260830" / "hi" / run_id / "original_acc.dat"
                    path = self.source_root / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(card_text(sigma_um, seed, variant), encoding="utf-8")
                    runs.append(
                        {
                            "run_id": run_id,
                            "sigma_z_mm": sigma_mm,
                            "sigma_z_card_micrometres": float(sigma_um),
                            "rndm_seed": seed,
                            "variant": variant,
                            "ip_dispersion_2_m": -0.002797928 if variant == "opp" else 0.002797928,
                            "card_relative_path": rel.as_posix(),
                            "card_sha256": sha256(path),
                        }
                    )
        write_json(
            self.manifest,
            {
                "schema_version": 1,
                "campaign_id": "test-holdout",
                "pristine_unseen_qualification_holdout": False,
                "common_card_contract": common,
                "statistical_contract": {
                    "independent_unit": "complete run",
                    "seeds_per_sigma_z_and_sign_configuration": 2,
                    "two_seed_status": "diagnostic_only",
                },
                "runs": runs,
            },
        )
        if with_history:
            first = runs[0]
            source_card = self.source_root / first["card_relative_path"]
            for backend in ("cpu", "gpu"):
                self._write_history(source_card.parent / backend, source_card, 99.0)
            convergence = self.source_root / "benchmark_results" / "FeeBL_FCCZ_convergence_8_20260830"
            self._write_history(convergence / f"xgrid_{first['run_id']}", source_card, 102.0)
            self._write_history(convergence / f"xstat_{first['run_id']}", source_card, 103.0)
        config = {
            "schema": campaign.CONFIG_SCHEMA,
            "schema_version": 1,
            "campaign_id": "fake-frozen-eight",
            "finalized": True,
            "adaptive_arm_id": "selected_adaptive",
            "input_sha256": {
                "frozen_manifest": sha256(self.manifest),
                "adaptive_template": sha256(self.template),
                "binary": sha256(self.binary),
                "gaussian_core_fitter": sha256(self.fitter),
            },
            "card_overlay": {
                "backend_keys": ["compute_backend"],
                "shared_numerical_keys": ["n_x"],
                "adaptive_keys": [
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
                    "adaptive_grid_fail_on_unmet_error",
                    "adaptive_grid_pilot_diagnostics",
                    "adaptive_grid_terminal_patch_certification",
                    "adaptive_grid_schedule_save",
                    "adaptive_macroparticle_mode",
                    "adaptive_macroparticle_policy_save",
                    "adaptive_macroparticle_trace_save",
                ],
                "control_values": {
                    "adaptive_grid_mode": "0",
                    "adaptive_macroparticle_mode": "0",
                },
            },
            "execution": {
                "argv": ["{python}", "{binary}", "fccz", "p_fccz", "out.dat"],
                "timeout_seconds": 20,
                "inherit_environment": ["PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"],
                "environment": {
                    "FAKE_UNMET": "1" if unmet else "0",
                    "FAKE_CERTIFICATE_TARGET": certificate_target,
                    "FAKE_TERMINAL_PATCH_ENABLED": terminal_patch_enabled,
                },
            },
            "telemetry": {
                "artifacts": [
                    {"role": "policy", "path": "adaptive_policy.json"},
                    {"role": "execution_trace", "path": "adaptive_trace.json"},
                    {"role": "decision_schedule", "path": "adaptive_schedule.json"},
                ],
                "required_record_prefixes": [
                    "FEEBL_ADAPTIVE_MACROPARTICLE_SUMMARY_V1",
                    "FEEBL_ADAPTIVE_GRID_CERTIFICATE_V1",
                ],
                "required_fields": ["live_distinct_roots1", "representative_visits1"],
            },
            "statistics": {
                "minimum_unseen_seeds_per_configuration": 4,
                "relative_equivalence_margin": "0.01",
                "expected_relative_difference": "0",
                "equivalence_alpha": "0.05",
                "equivalence_power": "0.90",
                "standard_error_safety_factor": "1.5",
                "maximum_supported_unseen_seeds_per_configuration": 64,
            },
            "additional_manifest_sha256": None,
            "notes": "fake standard-library test fixture",
        }
        write_json(self.config, config)

    def _write_history(self, directory: Path, source_card: Path, lumi: float) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "acc.dat").write_bytes(source_card.read_bytes())
        (directory / "out.dat").write_text(
            f"lumi_ee = {lumi}\nlumi_ee_high = {lumi * 0.9}\n", encoding="utf-8"
        )
        (directory / "lumi.ee.out.dLdW.fne").write_text(
            "# historical fake\n90 1\n91 2\n", encoding="utf-8"
        )
        (directory / "log.txt").write_text("historical\n", encoding="utf-8")

    def freeze(self, *, additional: Path | None = None) -> str:
        return campaign.freeze_campaign(
            argparse.Namespace(
                manifest=self.manifest,
                source_root=self.source_root,
                adaptive_template=self.template,
                config=self.config,
                binary=self.binary,
                fitter=self.fitter,
                additional_manifest=additional,
                output=self.output,
            )
        )

    def run_args(self, digest: str) -> argparse.Namespace:
        return argparse.Namespace(campaign=self.output, campaign_sha256=digest)


class FrozenEightCampaignTest(unittest.TestCase):
    @staticmethod
    def _grid_certificate_case(
        *, mixed: bool = False, field_overrides: dict[str, str] | None = None,
        card_overrides: dict[str, str] | None = None,
        field_order: list[str] | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, campaign.Assignment]]:
        budget = "a" * 64 if mixed else "not_applicable"
        values = {
            "schema": campaign.GRID_CERTIFICATE_SCHEMA,
            "requested_mode": "2",
            "actual_mode": "2",
            "error_norm_policy": (
                "vector_mixed_rms_quantile_v2"
                if mixed else "vector_rms_quantile_v1"
            ),
            "source_model": "fixed_width_bspline1_v1",
            "error_budget_sha256": budget,
            "card_hash": "c" * 64,
            "error_target": "0.01",
            "error_quantile": "0.999",
            "error_reference_field": "12.5" if mixed else "0",
            "error_absolute_floor_fraction": "0.001" if mixed else "0",
            "error_catastrophe_absolute_fraction": "0.01" if mixed else "0",
            "error_exposure_limit": "0.02" if mixed else "0",
            "maximum_beam_above_target_exposure": "0.01" if mixed else "0",
            "pilot_diagnostics": "0",
            "fail_on_unmet_error": "1",
            "encounters": "4",
            "certified_encounters": "4",
            "unmet_encounters": "0",
            "noise_stop_encounters": "0",
            "resource_unmet_encounters": "0",
            "vector_catastrophe_failure_encounters": "0",
            "maximum_vector_difference": "0.005",
            "catastrophe_limit": "0.5",
            "terminal_patch_enabled": "0",
            "terminal_patch_attempts": "0",
            "terminal_patch_accepts": "0",
            "terminal_patch_rejects": "0",
            "terminal_patch_rejection_mask_or": "0",
            "external_replay_recommended": "0",
            "all_encounters_certified": "1",
            "disposition": "accepted_all_encounters",
        }
        if field_overrides:
            values.update(field_overrides)
        order = field_order or list(campaign.GRID_CERTIFICATE_FIELD_ORDER)
        ordered = {key: values[key] for key in order}
        raw = campaign.GRID_CERTIFICATE_RECORD + " " + " ".join(
            f"{key}={value}" for key, value in ordered.items()
        )
        records: list[dict[str, object]] = [
            {
                "line_number": 1,
                "record": campaign.GRID_CERTIFICATE_RECORD,
                "fields": ordered,
                "raw": raw,
            }
        ]
        card_values = {
            "adaptive_grid_mode": values["requested_mode"],
            "adaptive_grid_error_norm_policy": values["error_norm_policy"],
            "adaptive_grid_error_budget_sha256": (
                "" if values["error_budget_sha256"] == "not_applicable"
                else values["error_budget_sha256"]
            ),
            "adaptive_grid_source_model": values["source_model"],
            "adaptive_grid_error_target": values["error_target"],
            "adaptive_grid_error_quantile": values["error_quantile"],
            "adaptive_grid_error_catastrophe_limit": values["catastrophe_limit"],
            "adaptive_grid_error_reference_field": values["error_reference_field"],
            "adaptive_grid_error_absolute_floor_fraction": values[
                "error_absolute_floor_fraction"
            ],
            "adaptive_grid_error_catastrophe_absolute_fraction": values[
                "error_catastrophe_absolute_fraction"
            ],
            "adaptive_grid_error_exposure_limit": values["error_exposure_limit"],
            "adaptive_grid_pilot_diagnostics": values["pilot_diagnostics"],
            "adaptive_grid_fail_on_unmet_error": values["fail_on_unmet_error"],
            "adaptive_grid_terminal_patch_certification": values[
                "terminal_patch_enabled"
            ],
        }
        if card_overrides:
            card_values.update(card_overrides)
        quoted = {
            "adaptive_grid_error_norm_policy",
            "adaptive_grid_error_budget_sha256",
            "adaptive_grid_source_model",
        }
        card = "$PARAMETERS:: p_fccz\n{" + ";".join(
            f'{key}="{value}"' if key in quoted else f"{key}={value}"
            for key, value in card_values.items()
        ) + ";}\n"
        assignments = campaign.parse_assignments(card, "certificate test card")
        return records, assignments

    def test_grid_certificate_current_schema_is_strict_and_bound(self) -> None:
        records, assignments = self._grid_certificate_case()
        accepted = campaign.validate_grid_certificate(records, assignments)
        self.assertEqual(
            accepted["canonical_fields"]["error_budget_sha256"],
            "not_applicable",
        )

        mixed_records, mixed_assignments = self._grid_certificate_case(mixed=True)
        mixed_accepted = campaign.validate_grid_certificate(
            mixed_records, mixed_assignments
        )
        self.assertEqual(
            mixed_accepted["canonical_fields"][
                "maximum_beam_above_target_exposure"
            ],
            "0.01",
        )

        order = list(campaign.GRID_CERTIFICATE_FIELD_ORDER)
        order[5], order[6] = order[6], order[5]
        reordered, reordered_assignments = self._grid_certificate_case(
            field_order=order
        )
        with self.assertRaisesRegex(campaign.CampaignError, "order is noncanonical"):
            campaign.validate_grid_certificate(reordered, reordered_assignments)

        missing_order = [
            key for key in campaign.GRID_CERTIFICATE_FIELD_ORDER
            if key != "maximum_beam_above_target_exposure"
        ]
        missing, missing_assignments = self._grid_certificate_case(
            field_order=missing_order
        )
        with self.assertRaisesRegex(campaign.CampaignError, "noncanonical tokens"):
            campaign.validate_grid_certificate(missing, missing_assignments)

        unknown_order = list(campaign.GRID_CERTIFICATE_FIELD_ORDER) + ["unknown"]
        unknown, unknown_assignments = self._grid_certificate_case(
            field_overrides={"unknown": "0"}, field_order=unknown_order
        )
        with self.assertRaisesRegex(campaign.CampaignError, "noncanonical tokens"):
            campaign.validate_grid_certificate(unknown, unknown_assignments)

        adversarial = (
            (
                "uppercase_card_hash",
                {"card_hash": "C" * 64},
                None,
                "card_hash is not a canonical lowercase SHA-256",
            ),
            (
                "short_card_hash",
                {"card_hash": "c" * 63},
                None,
                "card_hash is not a canonical lowercase SHA-256",
            ),
            (
                "budget_card_mismatch",
                None,
                {"adaptive_grid_error_budget_sha256": "b" * 64},
                "error_budget_sha256 does not match acc.dat",
            ),
            (
                "per_beam_exposure_above_limit",
                {"maximum_beam_above_target_exposure": "0.021"},
                None,
                "invalid scale or per-beam exposure evidence",
            ),
            (
                "nonfinite_reference",
                {"error_reference_field": "Infinity"},
                None,
                "nonfinite decimal error_reference_field",
            ),
        )
        for name, field_overrides, card_overrides, error in adversarial:
            with self.subTest(name=name):
                bad_records, bad_assignments = self._grid_certificate_case(
                    mixed=True,
                    field_overrides=field_overrides,
                    card_overrides=card_overrides,
                )
                with self.assertRaisesRegex(campaign.CampaignError, error):
                    campaign.validate_grid_certificate(
                        bad_records, bad_assignments
                    )

        nonmixed_live, nonmixed_live_assignments = self._grid_certificate_case(
            field_overrides={"error_reference_field": "1"},
            card_overrides={"adaptive_grid_error_reference_field": "1"},
        )
        with self.assertRaisesRegex(
            campaign.CampaignError, "live mixed-V2 fields"
        ):
            campaign.validate_grid_certificate(
                nonmixed_live, nonmixed_live_assignments
            )

    def test_conflicting_duplicate_output_metric_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "out.dat"
            path.write_text(
                "lumi_ee = 1\nlumi_ee = 2\nlumi_ee_high = 0.9\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                campaign.CampaignError, "conflicting duplicate lumi_ee"
            ):
                campaign.parse_out_metrics(path)

    def test_manifest_physics_identity_is_bound_exactly(self) -> None:
        mutations = {
            "unsupported_sigma": (
                lambda manifest: manifest["runs"][0].update(
                    sigma_z_mm=0.3, sigma_z_card_micrometres=300.0
                ),
                "must be exactly 0.2265625 or 14.5",
            ),
            "inconsistent_sigma_units": (
                lambda manifest: manifest["runs"][0].update(
                    sigma_z_card_micrometres=226.5
                ),
                "must equal 1000\\*sigma_z_mm",
            ),
            "wrong_dispersion_magnitude": (
                lambda manifest: manifest["runs"][0].update(
                    ip_dispersion_2_m=-0.001
                ),
                "must equal -common_card_contract.ip_dispersion_1_m",
            ),
        }
        for name, (mutate, error) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
                mutate(manifest)
                rewrite_frozen_manifest(fixture, manifest)
                with self.assertRaisesRegex(campaign.CampaignError, error):
                    fixture.freeze()

    def test_manifest_rejects_duplicate_physical_card_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
            manifest["runs"][1]["card_relative_path"] = manifest["runs"][0][
                "card_relative_path"
            ]
            manifest["runs"][1]["card_sha256"] = manifest["runs"][0]["card_sha256"]
            rewrite_frozen_manifest(fixture, manifest)
            with self.assertRaisesRegex(campaign.CampaignError, "duplicate physical card path"):
                fixture.freeze()

    def test_selected_blocks_cannot_be_satisfied_by_decoys(self) -> None:
        cases = {
            "accelerator_decoy": (
                lambda text: (
                    "$ACCELERATOR:: decoy\n{sigma_z=226.562500;}\n"
                    + text.replace("sigma_z=226.562500;", "", 1)
                ),
                "missing required assignment 'sigma_z'",
            ),
            "parameters_decoy": (
                lambda text: (
                    "$PARAMETERS:: decoy\n{rndm_seed=1;}\n"
                    + text.replace("rndm_seed=1;", "", 1)
                ),
                "missing required assignment 'rndm_seed'",
            ),
            "duplicate_selected_parameters_block": (
                lambda text: text
                + "\n$PARAMETERS:: p_fccz\n{rndm_seed=1;}\n",
                "exactly one PARAMETERS::p_fccz block; found 2",
            ),
            "quoted_rndm_seed_spoof": (
                lambda text: text.replace(
                    "rndm_seed=1;", 'probe="rndm_seed=1;";', 1
                ),
                "missing required assignment 'rndm_seed'",
            ),
            "quoted_ip_dispersion_spoof": (
                lambda text: text.replace(
                    "ip_dispersion.2=-0.002797928;",
                    'probe="ip_dispersion.2=-0.002797928;";',
                    1,
                ),
                "missing required assignment 'ip_dispersion.2'",
            ),
        }
        for name, (mutate, error) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
                first = manifest["runs"][0]
                card = fixture.source_root / first["card_relative_path"]
                card.write_text(mutate(card.read_text(encoding="utf-8")), encoding="utf-8")
                first["card_sha256"] = sha256(card)
                rewrite_frozen_manifest(fixture, manifest)
                with self.assertRaisesRegex(campaign.CampaignError, error):
                    fixture.freeze()

    def test_quote_aware_assignment_parser_preserves_outer_string_rhs(self) -> None:
        text = (
            '$PARAMETERS:: p_fccz\n{probe="alpha; beta=gamma;";'
            'rndm_seed=5001;}\n'
        )
        assignments = campaign.parse_assignments(text, "quoted normal card")
        self.assertEqual(set(assignments), {"probe", "rndm_seed"})
        self.assertEqual(assignments["probe"].value, '"alpha; beta=gamma;"')
        self.assertEqual(assignments["rndm_seed"].value, "5001")
        escaped = (
            '$PARAMETERS:: p_fccz\n{probe="alpha\\\"; beta=gamma;";'
            'rndm_seed=5001;}\n'
        )
        escaped_assignments = campaign.parse_assignments(
            escaped, "escaped quoted card"
        )
        self.assertEqual(
            escaped_assignments["probe"].value, '"alpha\\\"; beta=gamma;"'
        )
        malformed_cards = (
            '$PARAMETERS:: p_fccz\n{probe="unterminated;}\n',
            '$PARAMETERS:: p_fccz\n{probe="valid";trailing}\n',
        )
        for malformed in malformed_cards:
            with self.assertRaisesRegex(campaign.CampaignError, "unterminated|malformed|no equals"):
                campaign.parse_assignments(malformed, "malformed quoted card")

    def test_unseen_seed_must_use_reserved_global_namespace(self) -> None:
        campaign.validate_preregistered_unseen_seed(5001)
        with self.assertRaisesRegex(
            campaign.CampaignError, "outside the reserved preregistered-unseen namespace"
        ):
            campaign.validate_preregistered_unseen_seed(5000)

    def test_four_unseen_complete_seeds_can_pass_statistical_gate(self) -> None:
        identities = {}
        for sigma_z in ("0.2265625", "14.5"):
            for variant in ("opp", "same"):
                for seed in (5001, 5002, 5003, 5004):
                    run_id = f"s{sigma_z}_{variant}_{seed}"
                    identities[run_id] = {
                        "origin": "preregistered_unseen",
                        "sigma_z_mm": sigma_z,
                        "rndm_seed": seed,
                        "variant": variant,
                        "arms": {
                            "control": {"lumi_ee": 100.0},
                            "selected_adaptive": {"lumi_ee": 100.1},
                        },
                    }
        plan = campaign._independent_seed_power_plan(
            identities,
            "selected_adaptive",
            {
                "minimum_unseen_seeds_per_configuration": 4,
                "relative_equivalence_margin": "0.01",
                "expected_relative_difference": "0",
                "equivalence_alpha": "0.05",
                "equivalence_power": "0.90",
                "standard_error_safety_factor": "1.5",
                "maximum_supported_unseen_seeds_per_configuration": 64,
            },
        )
        self.assertEqual(
            plan["current_preregistered_unseen_seed_count_per_configuration"], 4
        )
        self.assertEqual(
            plan["recommended_total_preregistered_unseen_seed_count_per_configuration"],
            4,
        )
        self.assertEqual(plan["additional_unseen_seeds_required"], 0)
        self.assertEqual(
            plan["disposition"],
            "statistical_equivalence_gate_passed_not_overall_qualification",
        )

    def test_student_t_precision_count_extends_four_seed_pilot(self) -> None:
        identities = {}
        deltas = (-0.003, -0.001, 0.001, 0.003)
        metrics = tuple(f"metric_{index}" for index in range(6))
        for sigma_z in ("0.2265625", "14.5"):
            for variant in ("opp", "same"):
                for seed, delta in zip((5001, 5002, 5003, 5004), deltas):
                    run_id = f"s{sigma_z}_{variant}_{seed}"
                    identities[run_id] = {
                        "origin": "preregistered_unseen",
                        "sigma_z_mm": sigma_z,
                        "rndm_seed": seed,
                        "variant": variant,
                        "arms": {
                            "control": {metric: 100.0 for metric in metrics},
                            "selected_adaptive": {
                                metric: 100.0 * (1.0 + delta) for metric in metrics
                            },
                        },
                    }
        plan = campaign._independent_seed_power_plan(
            identities,
            "selected_adaptive",
            {
                "minimum_unseen_seeds_per_configuration": 4,
                "relative_equivalence_margin": "0.01",
                "expected_relative_difference": "0",
                "equivalence_alpha": "0.05",
                "equivalence_power": "0.90",
                "standard_error_safety_factor": "1.5",
                "maximum_supported_unseen_seeds_per_configuration": 64,
            },
        )
        self.assertGreater(
            plan["recommended_total_preregistered_unseen_seed_count_per_configuration"],
            4,
        )
        self.assertGreater(plan["additional_unseen_seeds_required"], 0)
        comparison = plan["populations"]["preregistered_unseen_only"][
            "comparisons"
        ][0]
        self.assertGreater(
            comparison["fixed_critical_precision_required_complete_processes"],
            comparison["normal_power_required_complete_processes"],
        )
        self.assertEqual(
            plan["disposition"], "add_preregistered_unseen_complete_process_seeds"
        )

    def test_nonpositive_observed_margin_gap_requires_method_revision(self) -> None:
        identities = {}
        for sigma_z in ("0.2265625", "14.5"):
            for variant in ("opp", "same"):
                for seed in (5001, 5002, 5003, 5004):
                    run_id = f"s{sigma_z}_{variant}_{seed}"
                    identities[run_id] = {
                        "origin": "preregistered_unseen",
                        "sigma_z_mm": sigma_z,
                        "rndm_seed": seed,
                        "variant": variant,
                        "arms": {
                            "control": {"lumi_ee": 100.0},
                            "selected_adaptive": {"lumi_ee": 101.0},
                        },
                    }
        plan = campaign._independent_seed_power_plan(
            identities,
            "selected_adaptive",
            {
                "minimum_unseen_seeds_per_configuration": 4,
                "relative_equivalence_margin": "0.01",
                "expected_relative_difference": "0",
                "equivalence_alpha": "0.05",
                "equivalence_power": "0.90",
                "standard_error_safety_factor": "1.5",
                "maximum_supported_unseen_seeds_per_configuration": 64,
            },
        )
        self.assertEqual(
            plan["disposition"],
            "point_estimate_outside_margin_method_revision_required",
        )
        self.assertEqual(plan["additional_unseen_seeds_required"], 0)
        comparison = plan["populations"]["preregistered_unseen_only"][
            "comparisons"
        ][0]
        self.assertEqual(
            comparison["fixed_critical_precision_status"],
            "nonpositive_observed_margin_gap_method_revision_required",
        )
        self.assertIsNone(
            comparison["fixed_critical_precision_required_complete_processes"]
        )

    def test_freeze_preserves_source_physics_and_never_runs_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            digest = fixture.freeze()
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(list((fixture.output / "runs").iterdir()), [])
            source = (fixture.output / "cards" / "sz0002266_s1_opp" / "original_acc.dat").read_text(encoding="utf-8")
            control = (fixture.output / "cards" / "sz0002266_s1_opp" / "control_acc.dat").read_text(encoding="utf-8")
            adaptive = (fixture.output / "cards" / "sz0002266_s1_opp" / "selected_adaptive_acc.dat").read_text(encoding="utf-8")
            for exact in (
                "sigma_z=226.562500",
                "rndm_seed=1",
                "ip_dispersion.2=-0.002797928",
                "cut_x=1389005.7",
            ):
                self.assertIn(exact, source)
                self.assertIn(exact, control)
                self.assertIn(exact, adaptive)
            self.assertIn("n_x=64", control)
            self.assertIn("adaptive_grid_mode=0", control)
            self.assertIn("adaptive_macroparticle_mode=0", control)
            self.assertIn("adaptive_grid_mode=2", adaptive)
            self.assertEqual(campaign.status(fixture.run_args(digest)), "0/16 verified; next=0001_sz0002266_s1_opp_control")
            frozen_policy = fixture.output / "inputs" / "frozen_adaptive_policy.json"
            self.assertEqual(
                campaign.load_json(frozen_policy)["realization_scope"],
                "independent_of_run_identity_seed_and_outputs",
            )
            frozen_policy.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(campaign.CampaignError, "static file|tampered"):
                campaign.status(fixture.run_args(digest))

    def test_run_next_is_single_step_resumable_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            digest = fixture.freeze()
            first = campaign.run_next(fixture.run_args(digest))
            self.assertIn("(1/16)", first)
            self.assertEqual(len(list((fixture.output / "runs").iterdir())), 1)
            second = campaign.run_next(fixture.run_args(digest))
            self.assertIn("selected_adaptive", second)
            adaptive_dir = fixture.output / "runs" / "0002_sz0002266_s1_opp_selected_adaptive"
            telemetry = campaign.load_json(adaptive_dir / "adaptive_telemetry.json")
            self.assertEqual(telemetry["grid_certificate"]["qualification_gate"], "accepted_all_encounters")
            self.assertEqual(
                telemetry["realization_independent_policy"]["sha256"],
                sha256(adaptive_dir / "frozen_adaptive_policy.json"),
            )
            self.assertTrue((adaptive_dir / "gaussian_core.csv").is_file())
            (adaptive_dir / "out.dat").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(campaign.CampaignError, "tampered|mismatch"):
                campaign.run_next(fixture.run_args(digest))

    def test_gaussian_binding_must_match_executed_card_and_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            digest = fixture.freeze()
            campaign.run_next(fixture.run_args(digest))
            run_key = "0001_sz0002266_s1_opp_control"
            run_dir = fixture.output / "runs" / run_key
            output_path = run_dir / "gaussian_core.json"
            output = campaign.load_json(output_path)
            output["qualification_binding"]["rndm_seed"] = "999"
            output_path.write_bytes(campaign.canonical_json_bytes(output))
            with self.assertRaisesRegex(
                campaign.CampaignError,
                "qualification binding rndm_seed disagrees with executed acc.dat",
            ):
                campaign.validate_gaussian_output(run_dir, run_key)

    def test_partial_or_locked_output_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            digest = fixture.freeze()
            partial = fixture.output / "runs" / "0001_sz0002266_s1_opp_control"
            partial.mkdir()
            (partial / "out.dat").write_text("preexisting\n", encoding="utf-8")
            with self.assertRaises(campaign.CampaignError):
                campaign.run_next(fixture.run_args(digest))
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            digest = fixture.freeze()
            (fixture.output / ".campaign.lock").write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(campaign.CampaignError, "lock already exists"):
                campaign.run_next(fixture.run_args(digest))

    def test_internal_unmet_grid_error_fails_despite_valid_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), unmet=True)
            digest = fixture.freeze()
            campaign.run_next(fixture.run_args(digest))  # nonadaptive control
            with self.assertRaisesRegex(
                campaign.CampaignError, "does not cover every encounter|unmet|error_target_met"
            ):
                campaign.run_next(fixture.run_args(digest))
            failed = fixture.output / "runs" / "0002_sz0002266_s1_opp_selected_adaptive"
            self.assertTrue((failed / "out.dat").is_file(), "fake endpoints were produced")
            receipt = campaign.load_json(failed / "run_receipt.json")
            self.assertEqual(receipt["status"], "failed")
            with self.assertRaisesRegex(campaign.CampaignError, "sealed failed run"):
                campaign.status(fixture.run_args(digest))

    def test_certificate_is_bound_to_the_executed_card(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), certificate_target="0.02")
            digest = fixture.freeze()
            campaign.run_next(fixture.run_args(digest))  # nonadaptive control
            with self.assertRaisesRegex(campaign.CampaignError, "does not match acc.dat"):
                campaign.run_next(fixture.run_args(digest))
            failed = (
                fixture.output
                / "runs"
                / "0002_sz0002266_s1_opp_selected_adaptive"
            )
            self.assertEqual(
                campaign.load_json(failed / "run_receipt.json")["status"], "failed"
            )

    def test_terminal_patch_enable_is_bound_to_the_executed_card(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), terminal_patch_enabled="1")
            digest = fixture.freeze()
            campaign.run_next(fixture.run_args(digest))  # nonadaptive control
            with self.assertRaisesRegex(
                campaign.CampaignError, "terminal_patch_enabled does not match acc.dat"
            ):
                campaign.run_next(fixture.run_args(digest))
            failed = (
                fixture.output
                / "runs"
                / "0002_sz0002266_s1_opp_selected_adaptive"
            )
            self.assertEqual(
                campaign.load_json(failed / "run_receipt.json")["status"], "failed"
            )

    def test_enabled_terminal_patch_must_be_exercised_for_frozen_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(
                Path(temporary),
                terminal_patch_enabled="1",
                terminal_patch_card="1",
            )
            digest = fixture.freeze()
            campaign.run_next(fixture.run_args(digest))  # nonadaptive control
            with self.assertRaisesRegex(
                campaign.CampaignError, "enabled but no terminal patch was exercised"
            ):
                campaign.run_next(fixture.run_args(digest))
            failed = (
                fixture.output
                / "runs"
                / "0002_sz0002266_s1_opp_selected_adaptive"
            )
            self.assertEqual(
                campaign.load_json(failed / "run_receipt.json")["status"], "failed"
            )

    def test_complete_analysis_pairs_arms_and_uses_available_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), with_history=True)
            digest = fixture.freeze()
            for _ in range(16):
                campaign.run_next(fixture.run_args(digest))
            analysis_hash = campaign.analyze(fixture.run_args(digest))
            self.assertRegex(analysis_hash, r"^[0-9a-f]{64}$")
            paired = campaign.load_json(fixture.output / "analysis" / "paired_adaptive_vs_control.json")
            self.assertEqual(len(paired["pairs"]), 8)
            self.assertEqual(paired["energy_spread_definition"], "gaussian_core_sigma_GeV")
            self.assertFalse(paired["global_spectrum_rms_used"])
            limits = campaign.load_json(fixture.output / "analysis" / "two_seed_limits.json")
            self.assertFalse(limits["qualification_allowed"])
            self.assertTrue(
                all(item["degrees_of_freedom"] == 1 for item in limits["paired_limits"])
            )
            power_plan = campaign.load_json(
                fixture.output / "analysis" / "independent_seed_power_plan.json"
            )
            self.assertTrue(power_plan["larger_n_m_is_not_independent_replication"])
            self.assertEqual(
                power_plan["current_preregistered_unseen_seed_count_per_configuration"],
                0,
            )
            self.assertGreaterEqual(
                power_plan[
                    "recommended_total_preregistered_unseen_seed_count_per_configuration"
                ],
                4,
            )
            self.assertEqual(
                power_plan["disposition"],
                "add_preregistered_unseen_complete_process_seeds",
            )
            historical = campaign.load_json(
                fixture.output / "analysis" / "historical_comparisons.json"
            )
            categories = {row["historical_category"] for row in historical["comparisons"]}
            self.assertEqual(categories, {"fixed512", "xgrid1024", "xstat1m"})
            with self.assertRaisesRegex(campaign.CampaignError, "preexisting analysis"):
                campaign.analyze(fixture.run_args(digest))

    def test_additional_seed_requires_exact_preregistered_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            extra_runs = []
            for sigma_tag, sigma_mm, sigma_um in (
                ("sz0002266", 0.2265625, "226.562500"),
                ("sz0145000", 14.5, "14500.0"),
            ):
                for variant in ("opp", "same"):
                    run_id = f"{sigma_tag}_s5001_{variant}"
                    rel = Path("unseen") / run_id / "original_acc.dat"
                    path = fixture.source_root / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(card_text(sigma_um, 5001, variant), encoding="utf-8")
                    extra_runs.append(
                        {
                            "run_id": run_id,
                            "sigma_z_mm": sigma_mm,
                            "sigma_z_card_micrometres": float(sigma_um),
                            "rndm_seed": 5001,
                            "variant": variant,
                            "ip_dispersion_2_m": -0.002797928 if variant == "opp" else 0.002797928,
                            "card_relative_path": rel.as_posix(),
                            "card_sha256": sha256(path),
                        }
                    )
            additional = fixture.root / "additional.json"
            write_json(
                additional,
                {
                    "schema": campaign.UNSEEN_SCHEMA,
                    "schema_version": 1,
                    "preregistered": True,
                    "preregistered_at_utc": "2026-09-08T00:00:00Z",
                    "runs": extra_runs,
                },
            )
            with self.assertRaisesRegex(campaign.CampaignError, "forbidden"):
                fixture.freeze(additional=additional)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            extra_runs = []
            for sigma_tag, sigma_mm, sigma_um in (
                ("sz0002266", 0.2265625, "226.562500"),
                ("sz0145000", 14.5, "14500.0"),
            ):
                for variant in ("opp", "same"):
                    run_id = f"{sigma_tag}_s5001_{variant}"
                    rel = Path("unseen") / run_id / "original_acc.dat"
                    path = fixture.source_root / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(card_text(sigma_um, 5001, variant), encoding="utf-8")
                    extra_runs.append(
                        {
                            "run_id": run_id,
                            "sigma_z_mm": sigma_mm,
                            "sigma_z_card_micrometres": float(sigma_um),
                            "rndm_seed": 5001,
                            "variant": variant,
                            "ip_dispersion_2_m": -0.002797928 if variant == "opp" else 0.002797928,
                            "card_relative_path": rel.as_posix(),
                            "card_sha256": sha256(path),
                        }
                    )
            additional = fixture.root / "additional.json"
            write_json(
                additional,
                {
                    "schema": campaign.UNSEEN_SCHEMA,
                    "schema_version": 1,
                    "preregistered": True,
                    "preregistered_at_utc": "2026-09-08T00:00:00Z",
                    "runs": extra_runs,
                },
            )
            config = json.loads(fixture.config.read_text(encoding="utf-8"))
            config["additional_manifest_sha256"] = sha256(additional)
            write_json(fixture.config, config)
            digest = fixture.freeze(additional=additional)
            plan = campaign.load_json(fixture.output / "campaign.json")
            self.assertEqual(plan["additional_run_count"], 4)
            self.assertEqual(len(plan["runs"]), 24)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
