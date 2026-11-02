#!/usr/bin/env python3
"""Mutation tests for the executor microbenchmark validator."""

from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modern_cpu_executor_benchmark_support as support  # noqa: E402


class ValidatorMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="feebl-executor-validator-test.")
        root = Path(self.temporary.name)
        artifact_names = {
            "source_fixture": "source.cc",
            "harness_fixture": "harness.py",
            "benchmark_binary": "benchmark.bin",
            "openmp_runtime": "libgomp.so.1",
            "libgomp": "libgomp.so.1",
        }
        artifacts: dict[str, dict[str, str]] = {}
        for label, filename in artifact_names.items():
            path = root / filename
            if not path.exists():
                path.write_bytes((filename + "\n").encode("ascii"))
            artifacts[label] = {
                "path": str(path.resolve()),
                "sha256": support.sha256_file(path),
            }
        validator_path = Path(support.__file__).resolve()
        artifacts["harness_validator"] = {
            "path": str(validator_path),
            "sha256": support.sha256_file(validator_path),
        }
        checksum = "0x0123456789abcdef"
        repetitions = 4  # Deliberately even: standard median is exercised.
        document = {
            "schema": support.SCHEMA,
            "valid": False,
            "validation_status": "unvalidated_staging",
            "classification": support.CLASSIFICATION,
            "eligible_for_project_level_2x_claim": False,
            "disclaimer": "unit fixture; not claim eligible",
            "oracle_scope": (
                "SerialReference is a same-implementation scheduling oracle; "
                "this is deterministic self-consistency only."
            ),
            "timing_scope": "execute only",
            "timing_scope_id": "executor_execute_only",
            "preset": "unit",
            "warmups_per_variant_case": 1,
            "repetitions_per_variant_case": repetitions,
            "git_revision": "0" * 40,
            "compiler_command": "unit-c++",
            "compile_flags": "-O3 -fopenmp",
            "openmp_macro": 201511,
            "host_metadata": {
                "schema": support.SCHEMA,
                "selected_worker_capacity": 1,
                "benchmark_worker_counts": [1],
                "selected_physical_cpus": [{"cpu": 0, "core": 0, "socket": 0}],
                "openmp_environment": {
                    "OMP_DYNAMIC": "FALSE",
                    "OMP_THREAD_LIMIT": "1",
                    "OMP_NUM_THREADS": "1",
                    "OMP_MAX_ACTIVE_LEVELS": "1",
                    "OMP_PROC_BIND": "close",
                    "OMP_PLACES": "{0}",
                    "OMP_WAIT_POLICY": "ACTIVE",
                    "OMP_SCHEDULE": "static",
                },
                "provenance": {
                    "git_revision": "0" * 40,
                    "git_dirty": False,
                    "git_status_porcelain": [],
                    "git_status_sha256": hashlib.sha256(b"").hexdigest(),
                    "compiler_invocation": "unit-c++",
                    "artifacts": artifacts,
                },
            },
            "runtime_binding": {
                "initial_thread_affinity_at_probe": [0],
                "worker_cpus_at_probe": [0],
                "worker_places_at_probe": [0],
                "omp_proc_bind": "close",
                "omp_num_places": 1,
            },
            "cases": [],
        }
        case = {
            "shape": "unit_2x2_t1",
            "beam1_slices": 2,
            "beam2_slices": 2,
            "substeps": 1,
            "encounters": 4,
            "waves": 3,
            "intensity": "unit",
            "integer_mix_iterations": 1,
            "oracle_checksum": checksum,
            "aggregates": [],
        }
        for variant, execution, workers, times, order_index in (
            ("serial_ref", "SerialReference", 1, [1.0, 2.0, 4.0, 8.0], 0),
            ("openmp_1", "OpenMP", 1, [0.5, 1.0, 2.0, 4.0], 1),
        ):
            aggregate = {
                "variant": variant,
                "execution": execution,
                "workers": workers,
                "schedule": "Static",
                "construction_seconds": 0.01,
                "preflight_bytes": 100,
                "checksum": checksum,
                "deterministic_checksum_match_same_implementation_oracle": True,
                "median_seconds": (times[1] + times[2]) / 2.0,
                "p95_seconds": times[-1],
                "min_seconds": times[0],
                "max_seconds": times[-1],
                "exploratory_speedup_vs_serial_median": 1.0,
                "samples": [],
            }
            for repetition, seconds in enumerate(times):
                aggregate["samples"].append(
                    {
                        "repetition": repetition,
                        "order_index": order_index,
                        "seconds": seconds,
                        "checksum": checksum,
                        "report_success": True,
                        "used_serial_fallback": False,
                        "attempted_encounters": 4,
                        "completed_encounters": 4,
                        "published_encounters": 0,
                        "requested_workers": workers,
                        "actual_workers": workers,
                        "parallel_regions": 1 if execution == "OpenMP" else 0,
                    }
                )
            case["aggregates"].append(aggregate)
        serial_median = case["aggregates"][0]["median_seconds"]
        for aggregate in case["aggregates"]:
            aggregate["exploratory_speedup_vs_serial_median"] = (
                serial_median / aggregate["median_seconds"]
            )
        document["cases"].append(case)
        rows = [
            support.as_csv_row(document, case, aggregate, sample)
            for aggregate in case["aggregates"]
            for sample in aggregate["samples"]
        ]
        self.document = document
        self.rows = rows
        self.repetitions = repetitions

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def validate(self, document=None, rows=None) -> None:
        support.validate_pair(
            self.document if document is None else document,
            self.rows if rows is None else rows,
            self.repetitions,
            expected_valid=False,
            verify_provenance_files=True,
        )

    def assert_rejected(self, document=None, rows=None) -> None:
        with self.assertRaises(support.ValidationError):
            self.validate(document, rows)

    def test_valid_even_sample_fixture_uses_standard_median(self) -> None:
        self.validate()
        self.assertEqual(self.document["cases"][0]["aggregates"][0]["median_seconds"], 3.0)

    def test_rejects_tampered_nonpositive_time(self) -> None:
        document = copy.deepcopy(self.document)
        document["cases"][0]["aggregates"][0]["samples"][0]["seconds"] = -1.0
        self.assert_rejected(document=document)

    def test_rejects_tampered_speedup(self) -> None:
        document = copy.deepcopy(self.document)
        document["cases"][0]["aggregates"][1]["exploratory_speedup_vs_serial_median"] = 99.0
        self.assert_rejected(document=document)

    def test_rejects_nonfinite_time(self) -> None:
        document = copy.deepcopy(self.document)
        document["cases"][0]["aggregates"][0]["samples"][1]["seconds"] = float("nan")
        self.assert_rejected(document=document)

    def test_rejects_tampered_checksum(self) -> None:
        document = copy.deepcopy(self.document)
        document["cases"][0]["aggregates"][1]["samples"][2]["checksum"] = "0xffffffffffffffff"
        self.assert_rejected(document=document)

    def test_rejects_csv_divergence(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["seconds"] = "999.0"
        self.assert_rejected(rows=rows)

    def test_promotion_occurs_only_after_validation(self) -> None:
        root = Path(self.temporary.name)
        json_path, csv_path = root / "final.json", root / "final.csv"
        support.promote_pair(self.document, self.rows, self.repetitions, json_path, csv_path)
        document, rows = support.validate_paths(
            json_path,
            csv_path,
            self.repetitions,
            expected_valid=True,
            verify_provenance_files=True,
        )
        self.assertTrue(document["valid"])
        self.assertTrue(all(row["valid"] == "true" for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
