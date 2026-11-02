#!/usr/bin/env python3
"""Fixed contract tests for the external adaptive-root replay driver."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = (
    ROOT
    / "examples" / "adaptive_particles" / "adaptive_root_replay.py"
)
SPEC = importlib.util.spec_from_file_location("adaptive_root_replay", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


BASE_CARD = """$ACCELERATOR:: fccz
{energy=45.6;dist_z=0;}

$PARAMETERS:: p_fccz
{n_x=16;n_y=16;n_z=2;n_m=1000;
charge_symmetric=0;force_symmetric=0;
load_beam=0;rndm_load=0;rndm_save=0;rndm_seed=7;
do_photons=0;do_pairs=0;do_lumi=1;num_lumi=1000;}
"""


def write_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return replay.sha256_file(path)


def contains_forbidden_policy_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            folded = key.casefold()
            if any(part in folded for part in replay.FORBIDDEN_POLICY_KEY_PARTS):
                return True
            if contains_forbidden_policy_key(item):
                return True
    elif isinstance(value, list):
        return any(contains_forbidden_policy_key(item) for item in value)
    return False


class ReplayFixture:
    def __init__(self, root: Path, card_text: str = BASE_CARD, executable: bytes = b"fake-feebl-v1\n"):
        self.root = root
        self.template = root / "template.dat"
        self.binary = root / "feebl"
        self.source = root / "source-state.sha256"
        self.template_hash = write_bytes(self.template, card_text.encode("utf-8"))
        self.binary_hash = write_bytes(self.binary, executable)
        self.source_hash = write_bytes(self.source, b"source-state-e687d0aa\n")
        self.generator = {
            "id": replay.GENERATOR_ID,
            "version": "legacy-sequential-transform-audited-v1",
        }
        self.physics = {
            "id": "fccz-fields-and-lumi-family-v1",
            "beam_species": "electron_positron",
            "enabled_channels": ["beam_beam_fields", "lumi_ee"],
        }
        self.generator_hash = replay.sha256_bytes(replay.canonical_bytes(self.generator))
        self.physics_hash = replay.sha256_bytes(replay.canonical_bytes(self.physics))
        template_raw = self.template.read_bytes()
        self.family_hash = replay.sha256_bytes(
            replay.masked_card_bytes(template_raw, "fccz", "p_fccz")
        )
        self.expected_bindings = {
            "template_card_family_sha256": self.family_hash,
            "feebl_binary_sha256": self.binary_hash,
            "source_state_sha256": self.source_hash,
            "generator_transform_sha256": self.generator_hash,
            "physics_family_sha256": self.physics_hash,
        }
        values = ["100", "104", "96", "108"]
        self.pilots: list[dict[str, object]] = []
        for index, (seed, value) in enumerate(zip((11, 13, 17, 19), values), start=1):
            card_path = root / f"pilot-{index}.dat"
            card = replay.patch_card(template_raw, "fccz", "p_fccz", 1000, seed)
            card_hash = write_bytes(card_path, card)
            evidence = root / f"pilot-{index}.complete"
            evidence_hash = write_bytes(evidence, f"exit_code=0\nseed={seed}\n".encode())
            self.pilots.append(
                {
                    "seed": seed,
                    "complete": True,
                    "exit_code": 0,
                    "n_m": 1000,
                    "card_path": str(card_path),
                    "card_sha256": card_hash,
                    "evidence_path": str(evidence),
                    "evidence_sha256": evidence_hash,
                    "bindings": dict(self.expected_bindings),
                    "values": {"lumi.total": value, "spectrum.core": str(int(value) / 100)},
                }
            )
        self.request: dict[str, object] = {
            "schema": replay.REQUEST_SCHEMA,
            "template_card": {
                "path": str(self.template),
                "sha256": self.template_hash,
                "accelerator": "fccz",
                "parameters": "p_fccz",
            },
            "bindings": {
                "feebl_binary_path": str(self.binary),
                "feebl_binary_sha256": self.binary_hash,
                "source_state_path": str(self.source),
                "source_state_sha256": self.source_hash,
                "generator_transform": self.generator,
                "physics_family": self.physics,
            },
            "sampling": {
                "n_now": 1000,
                "normalization_contract": replay.NORMALIZATION_ID,
                "round_up_to_multiple": 64,
                "minimum_complete_runs": 4,
                "standard_error_safety_factor": "1.5",
            },
            "endpoints": [
                {
                    "id": "lumi.total",
                    "scale": "100",
                    "target_relative_standard_error": "0.005",
                    "units": "arb",
                },
                {
                    "id": "spectrum.core",
                    "scale": "1",
                    "target_relative_standard_error": "0.005",
                    "units": "arb",
                },
            ],
            "pilot_runs": self.pilots,
            "production_seeds": [101, 103, 107, 109],
            "replay_limits": {
                "maximum_attempts": 3,
                "attempt_index": 1,
                "maximum_n_m": 1_000_000,
                "minimum_progress_fraction": "0.05",
            },
            "replay_history": [],
        }

    def write_request(self, name: str = "request.json") -> Path:
        path = self.root / name
        path.write_text(
            json.dumps(self.request, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return path


class AdaptiveRootReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="feebl-root-replay-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_runtime_replay_signal_contract(self) -> None:
        blocked = (
            "ordinary log line\n"
            f"{replay.RUNTIME_SIGNAL_PREFIX} "
            f"schema={replay.RUNTIME_SIGNAL_SCHEMA} "
            f"request_schema={replay.REQUEST_SCHEMA} "
            f"exit_code={replay.REPLAY_REQUIRED_EXIT_CODE} "
            "publication_blocked=1 exploratory_outputs_authorized=0 "
            "final_physics_outputs_published=0 "
            "publication_scope=final_summary_outputs_only_v1 "
            "pre_gate_diagnostic_artifacts_may_exist=1 "
            "pre_gate_artifacts_physics_qualification_claim=0 "
            "preexisting_artifacts_not_removed=1 fresh_run_directory_required=1 "
            "final_polarization_sample_complete=1 bpm_finalization_complete=1 "
            "global_luminosity_publication_complete=1 "
            "complete_collision=1 "
            "physics_qualification_claim=0 "
            "recommended_action=request_initial_state_replay "
            "reason=source_sampling_uncertainty_unknown_requires_independent_runs_or_replay "
            f"generator_id={replay.GENERATOR_ID} "
            f"normalization_contract={replay.NORMALIZATION_ID} "
            "rndm_seed=17 n_m_beam1=250000 n_m_beam2=250000 common_n_m=1 "
            f"independent_complete_runs=1 minimum_complete_runs={replay.MINIMUM_INDEPENDENT_RUNS} "
            "controller_observations=12 controller_trace_hash=987654321 "
            "roots_added_online=0 fresh_complete_process_required=1\n"
        )
        fields = replay.parse_runtime_replay_signal(blocked)
        self.assertEqual(fields["publication_blocked"], "1")
        self.assertEqual(fields["exit_code"], "75")
        self.assertEqual(fields["roots_added_online"], "0")

        exploratory = blocked.replace(
            "exit_code=75 publication_blocked=1 exploratory_outputs_authorized=0",
            "exit_code=0 publication_blocked=0 exploratory_outputs_authorized=1",
        ).replace(
            "final_physics_outputs_published=0",
            "final_physics_outputs_published=1",
        )
        fields = replay.parse_runtime_replay_signal(exploratory)
        self.assertEqual(fields["publication_blocked"], "0")
        self.assertEqual(fields["exit_code"], "0")

        malformed = {
            "online insertion": blocked.replace(
                "roots_added_online=0", "roots_added_online=1"
            ),
            "blocked publication contradiction": blocked.replace(
                "final_physics_outputs_published=0",
                "final_physics_outputs_published=1",
            ),
            "exploratory publication contradiction": exploratory.replace(
                "final_physics_outputs_published=1",
                "final_physics_outputs_published=0",
            ),
            "wrong status": blocked.replace("exit_code=75", "exit_code=2"),
            "duplicate signal": blocked + blocked,
        }
        for name, record in malformed.items():
            with self.subTest(name=name):
                with self.assertRaises(replay.ReplayError):
                    replay.parse_runtime_replay_signal(record)

    def test_valid_plan_preserves_card_and_validates(self) -> None:
        fixture = ReplayFixture(self.root)
        request = fixture.write_request()
        output = self.root / "plan"
        policy, manifest = replay.plan_request(request, output)
        self.assertGreater(int(policy["allocation"]["target_n_m"]), 1000)
        self.assertEqual(policy["allocation"]["pilot_complete_run_count"], "4")
        self.assertEqual(policy["allocation"]["production_complete_run_count"], "4")
        self.assertFalse(contains_forbidden_policy_key(policy))
        self.assertNotIn("production_seeds", policy)
        self.assertEqual(len(manifest["realizations"]), 4)
        target = int(policy["allocation"]["target_n_m"])
        for realization, expected_seed in zip(manifest["realizations"], (101, 103, 107, 109)):
            card = (output / realization["card"]).read_bytes()
            info = replay.inspect_card(card, "fccz", "p_fccz")
            self.assertEqual(info["n_m"], target)
            self.assertEqual(info["rndm_seed"], expected_seed)
            self.assertEqual(info["card_family_sha256"], fixture.family_hash)
        replay.validate_artifacts(
            output / "adaptive_root_policy.json",
            output / "execution_manifest.json",
            output,
        )

    def test_output_is_byte_deterministic(self) -> None:
        fixture = ReplayFixture(self.root)
        request = fixture.write_request()
        first = self.root / "first"
        second = self.root / "second"
        replay.plan_request(request, first)
        replay.plan_request(request, second)
        for relative in ("adaptive_root_policy.json", "execution_manifest.json"):
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
        first_cards = sorted(path.relative_to(first) for path in (first / "runs").glob("*/acc.dat"))
        second_cards = sorted(path.relative_to(second) for path in (second / "runs").glob("*/acc.dat"))
        self.assertEqual(first_cards, second_cards)
        for relative in first_cards:
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())

    def test_tampering_and_binding_mismatch_are_refused(self) -> None:
        fixture = ReplayFixture(self.root)
        request = fixture.write_request()
        output = self.root / "plan"
        _, manifest = replay.plan_request(request, output)
        card = output / manifest["realizations"][0]["card"]
        card.write_bytes(card.read_bytes() + b"\n# tampered\n")
        with self.assertRaisesRegex(replay.ReplayError, "SHA-256 mismatch"):
            replay.validate_artifacts(
                output / "adaptive_root_policy.json",
                output / "execution_manifest.json",
                output,
            )

        mismatch_root = self.root / "mismatch"
        mismatch_root.mkdir()
        mismatched = ReplayFixture(mismatch_root)
        mismatched.request["template_card"]["sha256"] = "0" * 64  # type: ignore[index]
        mismatch_request = mismatched.write_request()
        with self.assertRaisesRegex(replay.ReplayError, "SHA-256 mismatch"):
            replay.plan_request(mismatch_request, mismatch_root / "out")

    def test_duplicate_pilot_seeds_are_refused(self) -> None:
        fixture = ReplayFixture(self.root)
        fixture.pilots[1]["seed"] = fixture.pilots[0]["seed"]
        request = fixture.write_request()
        with self.assertRaisesRegex(replay.ReplayError, "duplicates another pilot"):
            replay.plan_request(request, self.root / "out")

    def test_inadequate_independent_repetitions_are_refused(self) -> None:
        fixture = ReplayFixture(self.root)
        fixture.request["pilot_runs"] = fixture.pilots[:3]
        request = fixture.write_request()
        with self.assertRaisesRegex(replay.ReplayError, "inadequate independent repetitions"):
            replay.plan_request(request, self.root / "out")

    def test_inadequate_production_repetitions_are_refused(self) -> None:
        fixture = ReplayFixture(self.root)
        fixture.request["production_seeds"] = [101, 103, 107]
        request = fixture.write_request()
        with self.assertRaisesRegex(
            replay.ReplayError, "inadequate independent production repetitions"
        ):
            replay.plan_request(request, self.root / "out")

    def test_v1_pilot_and_production_replication_counts_must_match(self) -> None:
        fixture = ReplayFixture(self.root)
        fixture.request["production_seeds"] = [101, 103, 107, 109, 113]
        request = fixture.write_request()
        with self.assertRaisesRegex(
            replay.ReplayError,
            "production complete-run count must equal pilot complete-run count",
        ):
            replay.plan_request(request, self.root / "out")

    def test_resigned_manifest_path_escape_is_refused(self) -> None:
        cases = {
            "run_directory": ("run_directory", "../escape"),
            "expected_output": ("expected_output", "../../escape.out"),
            "combined_log": ("combined_log", "/tmp/escape.log"),
            "windows_drive": ("run_directory", "C:/escape"),
        }
        for name, (field, value) in cases.items():
            with self.subTest(name=name):
                case_root = self.root / name
                case_root.mkdir()
                fixture = ReplayFixture(case_root)
                request = fixture.write_request()
                output = case_root / "plan"
                replay.plan_request(request, output)
                manifest_path = output / "execution_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                del manifest["self_sha256"]
                manifest["realizations"][0][field] = value
                replay.write_canonical(manifest_path, replay.with_self_hash(manifest))
                with self.assertRaisesRegex(replay.ReplayError, "safe relative|safe filename"):
                    replay.validate_artifacts(
                        output / "adaptive_root_policy.json",
                        manifest_path,
                        output,
                    )

    def test_missing_explicit_scalar_scale_is_refused(self) -> None:
        fixture = ReplayFixture(self.root)
        del fixture.request["endpoints"][0]["scale"]  # type: ignore[index]
        request = fixture.write_request()
        with self.assertRaisesRegex(replay.ReplayError, "scale"):
            replay.plan_request(request, self.root / "out")

    def test_symmetry_and_external_load_are_refused(self) -> None:
        cases = {
            "charge_symmetric": BASE_CARD.replace("charge_symmetric=0", "charge_symmetric=1"),
            "symmetric_constant_z": BASE_CARD.replace("dist_z=0", "dist_z=1").replace(
                "charge_symmetric=0", "charge_symmetric=1"
            ),
            "external_load": BASE_CARD.replace("load_beam=0", "load_beam=3"),
        }
        patterns = {
            "charge_symmetric": "correlated family",
            "symmetric_constant_z": "factor-four",
            "external_load": "cannot invent independent roots",
        }
        for name, card in cases.items():
            with self.subTest(name=name):
                case_root = self.root / name
                case_root.mkdir()
                fixture = ReplayFixture(case_root, card)
                request = fixture.write_request()
                with self.assertRaisesRegex(replay.ReplayError, patterns[name]):
                    replay.plan_request(request, case_root / "out")

    def test_unsymmetric_constant_z_is_explicitly_warned(self) -> None:
        fixture = ReplayFixture(self.root, BASE_CARD.replace("dist_z=0", "dist_z=1"))
        request = fixture.write_request()
        policy, _ = replay.plan_request(request, self.root / "out")
        warnings = policy["source_distribution_contract"]["warnings"]
        self.assertTrue(any("constant_z_unsymmetric" in item for item in warnings))

    def test_generator_mismatch_is_refused(self) -> None:
        fixture = ReplayFixture(self.root)
        fixture.pilots[2]["bindings"]["generator_transform_sha256"] = "f" * 64  # type: ignore[index]
        request = fixture.write_request()
        with self.assertRaisesRegex(replay.ReplayError, "generator_transform_sha256"):
            replay.plan_request(request, self.root / "out")

    def test_attempt_limit_and_nonprogress_are_fail_closed(self) -> None:
        fixture = ReplayFixture(self.root)
        fixture.request["replay_limits"]["attempt_index"] = 2  # type: ignore[index]
        fixture.request["replay_history"] = [
            {
                "attempt_index": 1,
                "n_m": 999,
                "endpoint_conservative_standard_errors": {
                    "lumi.total": "3.8",
                    "spectrum.core": "0.038",
                },
            }
        ]
        request = fixture.write_request()
        with self.assertRaisesRegex(replay.ReplayError, "non-progress"):
            replay.plan_request(request, self.root / "out")

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_requested_execution_is_strictly_sequential(self) -> None:
        trace = self.root / "sequence.txt"
        program = f"""#!/usr/bin/env python3
import pathlib, sys
trace = pathlib.Path({str(trace)!r})
previous = trace.read_text().splitlines() if trace.exists() else []
ordinal = int(pathlib.Path.cwd().name.split('_', 1)[0])
if len(previous) + 1 != ordinal:
    raise SystemExit(9)
with trace.open('a') as handle:
    handle.write(str(ordinal) + '\\n')
pathlib.Path('out.dat').write_text('complete\\n')
""".encode()
        fixture = ReplayFixture(self.root, executable=program)
        fixture.binary.chmod(fixture.binary.stat().st_mode | stat.S_IXUSR)
        # chmod does not alter bytes, so its SHA binding remains valid.
        request = fixture.write_request()
        output = self.root / "out"
        replay.plan_request(request, output)
        code = replay.execute_artifacts(
            output / "adaptive_root_policy.json",
            output / "execution_manifest.json",
            output,
        )
        self.assertEqual(code, 0)
        self.assertEqual(trace.read_text().splitlines(), ["1", "2", "3", "4"])
        results = json.loads((output / "execution_results.json").read_text())
        self.assertTrue(results["completed_all_planned_runs"])
        self.assertFalse(results["physics_qualification_claim"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
