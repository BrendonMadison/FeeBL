#!/usr/bin/env python3
"""Mutation tests for the retained photon-luminosity benchmark evidence.

The fixture is deliberately synthetic and tiny: these tests audit the evidence
validator, not FeeBL physics or performance.  Every mutation starts from a
staging document that the validator first accepts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import statistics
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any

import photon_lumi_benchmark_support as support


def artifact(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": support.sha256_file(path),
    }


def write_result(path: Path, aggregates: dict[str, float]) -> None:
    path.write_text(
        "".join(f"{name} = {aggregates[name]:.17g}\n"
                for name in support.EXPECTED_AGGREGATES),
        encoding="utf-8",
    )


class EvidenceFixture:
    """Materialize one complete, valid, five-pair staging evidence tree."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.json_path = root / "photon_lumi_benchmark.staging.json"
        self.csv_path = root / "photon_lumi_benchmark.staging.csv"
        self.final_json = root / "photon_lumi_benchmark.json"
        self.final_csv = root / "photon_lumi_benchmark.csv"
        self.document = self._make_document()
        self.write_document()
        support.write_csv(self.csv_path, support.computed_rows(self.document))

    def _write_artifact(self, relative: str, content: str) -> dict[str, str]:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return artifact(path, self.root)

    def refresh_build_manifest(self, provenance: dict[str, Any] | None = None
                               ) -> None:
        provenance = provenance or self.document["provenance"]
        artifacts = provenance["artifacts"]
        manifest_path = self.root / "provenance/build_manifest.json"
        manifest = {
            "schema": "feebl.legacy_photon_lumi_fast.build.v1",
            "compiler": provenance["compiler"],
            "compile_flags": provenance["compile_flags"],
            "binary_sha256": artifacts["benchmark_binary"]["sha256"],
            "git_revision": provenance["git_revision"],
            "git_dirty": provenance["git_dirty"],
            "git_status_sha256": provenance["git_status_sha256"],
            "working_tree_diff_sha256": (
                artifacts["working_tree_diff"]["sha256"]
                if provenance["git_dirty"] else None
            ),
            "retained_artifact_sha256": {
                name: record["sha256"]
                for name, record in sorted(artifacts.items())
                if name != "build_manifest"
            },
            "retained_library_sha256": [
                row["retained"]["sha256"]
                for row in provenance["linked_libraries"]
            ],
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts["build_manifest"] = artifact(manifest_path, self.root)

    def _make_sample(self, relative: str, case: str, pair: int,
                     order: int, variant: str, seconds: float,
                     warmup: bool) -> dict[str, Any]:
        run = self.root / relative
        run.mkdir(parents=True)
        aggregates = {
            name: 1000.0 + float(index)
            for index, name in enumerate(support.EXPECTED_AGGREGATES)
        }
        result = run / "result.out"
        stdout = run / "stdout.txt"
        stderr = run / "stderr.txt"
        rng = run / "rndm.save"
        timing = run / "timing.json"
        elapsed_ns = int(round(seconds * 1.0e9))
        seconds = elapsed_ns * 1.0e-9
        start_ns = 10_000_000_000 + pair * 1_000_000_000 + order * 100_000
        write_result(result, aggregates)
        stdout.write_text(f"{case} {variant}\n", encoding="utf-8")
        if variant == "factorized":
            stderr.write_text(
                "FEEBL legacy photon-lumi summary: "
                "requested=factorized-exact-v2 factorized_calls=1 "
                "cartesian_calls=0 cartesian_pairs_visited=0 "
                "cartesian_pairs_avoided=16000000\n",
                encoding="utf-8",
            )
        else:
            stderr.write_text("Cartesian reference\n", encoding="utf-8")
        rng.write_text(f"rng-state-{pair}\n", encoding="utf-8")
        timing.write_text(json.dumps({
            "clock": "time.perf_counter_ns",
            "start_ns": start_ns,
            "end_ns": start_ns + elapsed_ns,
            "elapsed_ns": elapsed_ns,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sample: dict[str, Any] = {
            "case": case,
            "pair": pair,
            "repetition": pair,
            "order_index": order,
            "variant": variant,
            "warmup": warmup,
            "seconds": seconds,
            "exit_code": 0,
            "command": [
                "taskset", "--cpu-list", "0", "feebl", variant, "result.out"
            ],
            "aggregates": aggregates,
            "artifacts": {
                "stdout": artifact(stdout, self.root),
                "stderr": artifact(stderr, self.root),
                "result": artifact(result, self.root),
                "rng": artifact(rng, self.root),
                "timing": artifact(timing, self.root),
            },
        }
        if variant == "factorized":
            sample["provenance"] = {
                "factorized_calls": 1,
                "cartesian_calls": 0,
                "cartesian_pairs_visited": 0,
                "cartesian_pairs_avoided": 16000000,
            }
        if not warmup:
            sample["physics_equivalent"] = True
            sample["rng_identical"] = True
        return sample

    def _make_document(self) -> dict[str, Any]:
        deck = self._write_artifact(
            "raw/n4000/benchmark.dat", "n_m=4000\nn_m=4000\n"
        )
        required = {
            "benchmark_binary", "deck_template", "runner", "validator",
            "validator_mutation_tests", "fast_source", "fast_header",
            "grid_source", "switch_source", "ldd_output", "lscpu_output",
        }
        provenance_artifacts = {
            name: self._write_artifact(
                f"provenance/inputs/{name}.txt", f"fixture {name}\n"
            )
            for name in sorted(required)
        }
        ldd_output = self.root / provenance_artifacts["ldd_output"]["path"]
        ldd_output.write_text(
            "libfixture.so => /original/host/libfixture.so (0x00000000)\n",
            encoding="utf-8",
        )
        provenance_artifacts["ldd_output"]["sha256"] = (
            support.sha256_file(ldd_output)
        )
        retained_library_path = self.root / "provenance/libraries/000-libfixture.so"
        retained_library_path.parent.mkdir(parents=True, exist_ok=True)
        retained_library_path.write_bytes(b"synthetic retained shared library\n")
        retained_library = {
            "original_host_path": "/original/host/libfixture.so",
            "resolved_host_path": "/resolved/host/libfixture.so",
            "retained": artifact(retained_library_path, self.root),
        }
        provenance = {
            "git_revision": "0" * 40,
            "git_dirty": False,
            "git_status_porcelain": [],
            "git_status_sha256": hashlib.sha256(b"").hexdigest(),
            "compiler": "fixture-c++",
            "compile_flags": "-O2 -fno-fast-math",
            "command_template": ["taskset", "feebl", "<variant>"],
            "performance_environment": {
                "OMP_NUM_THREADS": "1",
                "OMP_DYNAMIC": "FALSE",
                "OMP_PROC_BIND": "close",
                "OMP_PLACES": "{0}",
            },
            "host": {
                "pinned_cpu": 0,
                "allowed_cpu_list": [0],
            },
            "artifacts": provenance_artifacts,
            "linked_libraries": [retained_library],
        }
        self.refresh_build_manifest(provenance)
        case: dict[str, Any] = {
            "name": "n4000",
            "macroparticles_per_beam": 4000,
            "deck": deck,
            "warmups": [],
            "samples": [],
        }
        for order, variant in enumerate(("cartesian", "factorized")):
            case["warmups"].append(self._make_sample(
                f"raw/n4000/warmup-00-{order}-{variant}", "n4000", 0,
                order, variant, 0.5 + 0.1 * order, True,
            ))
        for pair in range(5):
            variants = (("cartesian", "factorized") if pair % 2 == 0
                        else ("factorized", "cartesian"))
            for order, variant in enumerate(variants):
                seconds = ((2.0 + 0.1 * pair) if variant == "cartesian"
                           else (1.0 + 0.05 * pair))
                case["samples"].append(self._make_sample(
                    f"raw/n4000/pair-{pair:02d}-{order}-{variant}",
                    "n4000", pair, order, variant, seconds, False,
                ))
        cartesian = [sample["seconds"] for sample in case["samples"]
                     if sample["variant"] == "cartesian"]
        factorized = [sample["seconds"] for sample in case["samples"]
                      if sample["variant"] == "factorized"]
        paired = []
        for pair in range(5):
            group = {sample["variant"]: sample for sample in case["samples"]
                     if sample["pair"] == pair}
            paired.append(group["cartesian"]["seconds"] /
                          group["factorized"]["seconds"])
        case["summary"] = {
            "cartesian_median_seconds": statistics.median(cartesian),
            "factorized_median_seconds": statistics.median(factorized),
            "ratio_of_medians_speedup": (
                statistics.median(cartesian) / statistics.median(factorized)
            ),
            "median_paired_speedup": statistics.median(paired),
            "cartesian_min_seconds": min(cartesian),
            "cartesian_max_seconds": max(cartesian),
            "factorized_min_seconds": min(factorized),
            "factorized_max_seconds": max(factorized),
        }
        return {
            "schema": support.SCHEMA,
            "valid": False,
            "validation_status": "unvalidated_staging",
            "classification": support.CLASSIFICATION,
            "eligible_for_project_level_2x_claim": False,
            "disclaimer": (
                "Synthetic validator fixture; not a whole-program 2x claim."
            ),
            "evidence_authenticity": {
                "external_cryptographic_witness": False,
                "statement": (
                    "Unkeyed local SHA-256 integrity checks do not provide "
                    "independent cryptographic witness authenticity."
                ),
            },
            "evidence_root": ".",
            "accuracy_gate_relative": 2.0e-6,
            "accuracy_gate_absolute": 2.0e-6,
            "warmups_per_variant_case": 1,
            "repetitions": 5,
            "provenance": provenance,
            "cases": [case],
        }

    def write_document(self) -> None:
        self.json_path.write_text(
            json.dumps(self.document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def sample(self, variant: str, pair: int = 0) -> dict[str, Any]:
        return next(
            sample for sample in self.document["cases"][0]["samples"]
            if sample["variant"] == variant and sample["pair"] == pair
        )

    def artifact_path(self, sample: dict[str, Any], name: str) -> Path:
        return self.root / sample["artifacts"][name]["path"]


class ValidatorMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="feebl-photon-validator-"
        )
        self.fixture = EvidenceFixture(Path(self.temporary.name))
        # Every mutation is meaningful only if its unmodified parent is valid.
        support.validate_pair(
            self.fixture.json_path, self.fixture.csv_path, False, True
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_staging_rejected(self) -> None:
        with self.assertRaises(support.ValidationError):
            support.validate_pair(
                self.fixture.json_path, self.fixture.csv_path, False, True
            )

    def test_clean_staging_is_accepted(self) -> None:
        support.validate_pair(
            self.fixture.json_path, self.fixture.csv_path, False, True
        )

    def test_valid_dirty_tree_retains_patch_and_build_manifest(self) -> None:
        provenance = self.fixture.document["provenance"]
        status = [" M src/gridCPP.cc", "?? src/legacyPhotonLumiFastCPP.cc"]
        provenance["git_dirty"] = True
        provenance["git_status_porcelain"] = status
        provenance["git_status_sha256"] = support.sha256_text("\n".join(status))
        patch_path = self.fixture.root / "provenance/working_tree.patch"
        patch_path.write_text(
            "diff --git a/src/gridCPP.cc b/src/gridCPP.cc\n"
            "--- a/src/gridCPP.cc\n+++ b/src/gridCPP.cc\n",
            encoding="utf-8",
        )
        provenance["artifacts"]["working_tree_diff"] = artifact(
            patch_path, self.fixture.root
        )
        self.fixture.refresh_build_manifest()
        self.fixture.write_document()
        support.write_csv(
            self.fixture.csv_path, support.computed_rows(self.fixture.document)
        )
        support.validate_pair(
            self.fixture.json_path, self.fixture.csv_path, False, True
        )

    def test_dirty_tree_without_retained_patch_is_rejected(self) -> None:
        provenance = self.fixture.document["provenance"]
        status = [" M src/gridCPP.cc"]
        provenance["git_dirty"] = True
        provenance["git_status_porcelain"] = status
        provenance["git_status_sha256"] = support.sha256_text("\n".join(status))
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_unknown_compiler_is_rejected(self) -> None:
        self.fixture.document["provenance"]["compiler"] = "unknown"
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_unknown_compile_flags_are_rejected(self) -> None:
        self.fixture.document["provenance"]["compile_flags"] = (
            "unknown (binary supplied)"
        )
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_tampered_build_manifest_is_rejected(self) -> None:
        provenance = self.fixture.document["provenance"]
        record = provenance["artifacts"]["build_manifest"]
        path = self.fixture.root / record["path"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["compiler"] = "different-c++"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record["sha256"] = support.sha256_file(path)
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_widened_absolute_gate_is_rejected(self) -> None:
        self.fixture.document["accuracy_gate_absolute"] = 3.0e-6
        self.fixture.write_document()
        support.write_csv(
            self.fixture.csv_path, support.computed_rows(self.fixture.document)
        )
        self.assert_staging_rejected()

    def test_hybrid_gate_separates_absolute_and_relative_terms(self) -> None:
        left = {name: 0.0 for name in support.EXPECTED_AGGREGATES}
        right = copy.deepcopy(left)
        right["lumi_gg"] = 1.5e-6
        self.assertTrue(support.physics_close(left, right, 2.0e-6, 2.0e-6))
        right["lumi_gg"] = 2.5e-6
        self.assertFalse(support.physics_close(left, right, 2.0e-6, 2.0e-6))

        left["lumi_gg"] = 1000.0
        right["lumi_gg"] = 1000.0019
        self.assertTrue(support.physics_close(left, right, 2.0e-6, 2.0e-6))
        right["lumi_gg"] = 1000.0021
        self.assertFalse(support.physics_close(left, right, 2.0e-6, 2.0e-6))

    def test_tampered_timing_is_rejected(self) -> None:
        self.fixture.sample("cartesian")["seconds"] *= 7.0
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_internally_inconsistent_raw_timing_is_rejected(self) -> None:
        sample = self.fixture.sample("factorized")
        timing_path = self.fixture.artifact_path(sample, "timing")
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        timing["elapsed_ns"] += 1
        timing_path.write_text(
            json.dumps(timing, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sample["artifacts"]["timing"]["sha256"] = (
            support.sha256_file(timing_path)
        )
        self.fixture.write_document()
        support.write_csv(
            self.fixture.csv_path, support.computed_rows(self.fixture.document)
        )
        self.assert_staging_rejected()

    def test_tampered_physics_aggregate_is_rejected(self) -> None:
        sample = self.fixture.sample("factorized")
        changed = copy.deepcopy(sample["aggregates"])
        changed["lumi_gg"] += 100.0
        result = self.fixture.artifact_path(sample, "result")
        write_result(result, changed)
        sample["aggregates"] = changed
        sample["artifacts"]["result"]["sha256"] = support.sha256_file(result)
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_tampered_rng_hash_is_rejected(self) -> None:
        sample = self.fixture.sample("factorized")
        rng = self.fixture.artifact_path(sample, "rng")
        rng.write_text("different-final-rng-state\n", encoding="utf-8")
        sample["artifacts"]["rng"]["sha256"] = support.sha256_file(rng)
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_tampered_factorized_provenance_is_rejected(self) -> None:
        sample = self.fixture.sample("factorized")
        stderr = self.fixture.artifact_path(sample, "stderr")
        stderr.write_text(
            "FEEBL legacy photon-lumi summary: "
            "requested=factorized-exact-v2 factorized_calls=1 "
            "cartesian_calls=1 cartesian_pairs_visited=1 "
            "cartesian_pairs_avoided=15999999\n",
            encoding="utf-8",
        )
        sample["provenance"] = {
            "factorized_calls": 1,
            "cartesian_calls": 1,
            "cartesian_pairs_visited": 1,
            "cartesian_pairs_avoided": 15999999,
        }
        sample["artifacts"]["stderr"]["sha256"] = support.sha256_file(stderr)
        self.fixture.write_document()
        support.write_csv(
            self.fixture.csv_path, support.computed_rows(self.fixture.document)
        )
        self.assert_staging_rejected()

    def test_tampered_stored_speedup_is_rejected(self) -> None:
        summary = self.fixture.document["cases"][0]["summary"]
        summary["ratio_of_medians_speedup"] *= 1.25
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_duplicate_raw_aggregate_row_is_rejected(self) -> None:
        sample = self.fixture.sample("cartesian")
        result = self.fixture.artifact_path(sample, "result")
        with result.open("a", encoding="utf-8") as stream:
            stream.write(f"lumi_gg = {sample['aggregates']['lumi_gg']:.17g}\n")
        sample["artifacts"]["result"]["sha256"] = support.sha256_file(result)
        self.fixture.write_document()
        support.write_csv(
            self.fixture.csv_path, support.computed_rows(self.fixture.document)
        )
        self.assert_staging_rejected()

    def test_bogus_warmup_variant_is_rejected(self) -> None:
        warmup = self.fixture.document["cases"][0]["warmups"][1]
        warmup["variant"] = "cartesian"
        warmup["command"][-2] = "cartesian"
        warmup.pop("provenance")
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_reused_raw_run_artifacts_are_rejected(self) -> None:
        first = self.fixture.sample("cartesian", 0)
        reused = self.fixture.sample("cartesian", 2)
        reused["artifacts"] = copy.deepcopy(first["artifacts"])
        reused["aggregates"] = copy.deepcopy(first["aggregates"])
        self.fixture.write_document()
        support.write_csv(
            self.fixture.csv_path, support.computed_rows(self.fixture.document)
        )
        self.assert_staging_rejected()

    def test_inconsistent_openmp_affinity_is_rejected(self) -> None:
        environment = self.fixture.document["provenance"][
            "performance_environment"
        ]
        environment["OMP_PLACES"] = "{7}"
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_missing_retained_linked_library_is_rejected(self) -> None:
        library = self.fixture.document["provenance"]["linked_libraries"][0]
        (self.fixture.root / library["retained"]["path"]).unlink()
        self.assert_staging_rejected()

    def test_ldd_library_omission_is_rejected(self) -> None:
        provenance = self.fixture.document["provenance"]
        ldd_record = provenance["artifacts"]["ldd_output"]
        ldd_path = self.fixture.root / ldd_record["path"]
        with ldd_path.open("a", encoding="utf-8") as stream:
            stream.write(
                "libmissing.so => /original/host/libmissing.so (0x00000001)\n"
            )
        ldd_record["sha256"] = support.sha256_file(ldd_path)
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_authenticity_overclaim_is_rejected(self) -> None:
        authenticity = self.fixture.document["evidence_authenticity"]
        authenticity["external_cryptographic_witness"] = True
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_csv_divergence_is_rejected(self) -> None:
        rows = support.read_csv(self.fixture.csv_path)
        rows[0]["seconds"] = "999.0"
        support.write_csv(self.fixture.csv_path, rows)
        self.assert_staging_rejected()

    def test_missing_raw_file_is_rejected(self) -> None:
        sample = self.fixture.sample("cartesian")
        self.fixture.artifact_path(sample, "result").unlink()
        self.assert_staging_rejected()

    def test_nonzero_exit_is_rejected(self) -> None:
        self.fixture.sample("factorized")["exit_code"] = 17
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_path_traversal_is_rejected(self) -> None:
        self.fixture.sample("cartesian")["artifacts"]["result"][
            "path"
        ] = "../outside-result.out"
        self.fixture.write_document()
        self.assert_staging_rejected()

    def test_checksum_manifest_tampering_is_rejected(self) -> None:
        support.promote(
            self.fixture.json_path,
            self.fixture.csv_path,
            self.fixture.final_json,
            self.fixture.final_csv,
        )
        manifest = self.fixture.root / "SHA256SUMS"
        lines = manifest.read_text(encoding="utf-8").splitlines()
        digest, name = lines[0].split("  ", 1)
        replacement = ("1" if digest[0] != "1" else "2") + digest[1:]
        lines[0] = f"{replacement}  {name}"
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(support.ValidationError):
            support.validate_pair(
                self.fixture.final_json, self.fixture.final_csv, True, True
            )

    def test_promotion_attests_retained_validator(self) -> None:
        support.promote(
            self.fixture.json_path,
            self.fixture.csv_path,
            self.fixture.final_json,
            self.fixture.final_csv,
        )
        document = json.loads(
            self.fixture.final_json.read_text(encoding="utf-8")
        )
        retained = document["provenance"]["artifacts"]["validator"]
        self.assertEqual(document["validation"]["validator"], retained["path"])
        self.assertEqual(
            document["validation"]["validator_sha256"], retained["sha256"]
        )
        self.assertFalse(self.fixture.json_path.exists())
        support.validate_pair(
            self.fixture.final_json, self.fixture.final_csv, True, True
        )

    def test_tampered_retained_validator_attestation_is_rejected(self) -> None:
        support.promote(
            self.fixture.json_path,
            self.fixture.csv_path,
            self.fixture.final_json,
            self.fixture.final_csv,
        )
        document = json.loads(
            self.fixture.final_json.read_text(encoding="utf-8")
        )
        document["validation"]["validator_sha256"] = "f" * 64
        self.fixture.final_json.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        support.write_manifest(self.fixture.root)
        with self.assertRaises(support.ValidationError):
            support.validate_pair(
                self.fixture.final_json, self.fixture.final_csv, True, True
            )

    def test_manifest_failure_leaves_no_valid_json(self) -> None:
        with mock.patch.object(
            support, "write_manifest", side_effect=OSError("manifest failure")
        ):
            with self.assertRaises(OSError):
                support.promote(
                    self.fixture.json_path,
                    self.fixture.csv_path,
                    self.fixture.final_json,
                    self.fixture.final_csv,
                )
        self.assertFalse(self.fixture.final_json.exists())

    def test_publication_io_failure_leaves_no_valid_json(self) -> None:
        original = support.atomic_write_bytes

        def fail_manifest(destination: Path, content: bytes) -> None:
            if destination.name == "SHA256SUMS":
                raise OSError("injected publication failure")
            original(destination, content)

        with mock.patch.object(
            support, "atomic_write_bytes", side_effect=fail_manifest
        ):
            with self.assertRaises(OSError):
                support.promote(
                    self.fixture.json_path,
                    self.fixture.csv_path,
                    self.fixture.final_json,
                    self.fixture.final_csv,
                )
        self.assertFalse(self.fixture.final_json.exists())

    def test_prepublication_validation_failure_leaves_no_valid_json(self) -> None:
        original = support.validate_manifest

        def fail_prospective(root: Path,
                             prospective_files: dict[str, str] | None = None
                             ) -> None:
            if prospective_files:
                raise support.ValidationError(
                    "injected final prepublication validation failure"
                )
            original(root, prospective_files)

        with mock.patch.object(
            support, "validate_manifest", side_effect=fail_prospective
        ):
            with self.assertRaises(support.ValidationError):
                support.promote(
                    self.fixture.json_path,
                    self.fixture.csv_path,
                    self.fixture.final_json,
                    self.fixture.final_csv,
                )
        self.assertFalse(self.fixture.final_json.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
