#!/usr/bin/env python3
"""Non-vacuous regression tests for tools/modern_benchmark.py."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "tools" / "modern_benchmark.py"


FAKE_FEEBL = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import sys
import time
from pathlib import Path

card = Path("acc.dat")
candidate = 'engine="modern_cpu"' in card.read_text()
scenario = os.environ.get("FAKE_SCENARIO", "good")
if candidate and scenario == "dead":
    print("controlled candidate death", file=sys.stderr)
    raise SystemExit(7)

time.sleep(0.002 if candidate else 0.004)
output = Path(sys.argv[3])
output.parent.mkdir(parents=True, exist_ok=True)
text = "wrong-physics-result\n" if candidate and scenario == "wrong_output" else "physics-result\n"
output.write_text(text)
if candidate and scenario == "extra_output":
    Path("unexpected.dat").write_text("unexpected\n")
print("fake FeeBL completed")

if candidate:
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    engine = "wrong_engine" if scenario == "wrong_engine" else "modern_cpu"
    fallback = scenario == "fallback"
    manifest = {
        "schema": "feebl-modern-run-manifest-v1",
        "requested_engine": "modern_cpu",
        "actual_engine": engine,
        "executable_sha256": sha(Path(sys.argv[0]).resolve()),
        "input_card_sha256": sha(card),
        "input_inventory_sha256": os.environ["FEEBL_INPUT_INVENTORY_SHA256"],
        "output_sha256": sha(output),
        "capabilities": {
            "beam_soa": True,
            "cell_csr": True,
            "stable_ids": True,
            "counter_rng": True,
            "encounter_dag": True,
        },
        "work_counters": {
            "encounters": 16,
            "primary_particles": 1024,
            "physics_events": 32,
        },
        "fallback": fallback,
        "fallback_reason": "controlled fallback" if fallback else "",
        "host_stage_count": 0,
        "transfer_count": 0,
        "transfer_bytes": 0,
        "allocation_count": 0,
        "synchronization_count": 0,
        "completion_marker": "complete",
        "nan_detected": False,
    }
    Path(os.environ["FEEBL_RUN_MANIFEST"]).write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
'''


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prepare(directory: Path) -> tuple[Path, Path, Path]:
    binary = directory / "fake_feebl.py"
    binary.write_text(FAKE_FEEBL)
    binary.chmod(0o755)
    card = directory / "source.dat"
    card.write_text(
        "$ACCELERATOR:: c_fake\n{energy=1;}\n"
        "$PARAMETERS:: p_fake\n{rndm_load=0;rndm_save=0;}\n"
    )
    dependency = directory / "physics-plan.json"
    dependency.write_text('{"plan":"controlled"}\n')
    return binary, card, dependency


def run_harness(directory: Path, scenario: str, name: str,
                *, repetitions: int = 1, dry_run: bool = False,
                smoke: bool = False) -> subprocess.CompletedProcess[str]:
    binary, card, dependency = prepare(directory)
    output = directory / name
    command = [
        sys.executable, str(HARNESS),
        "--binary", str(binary),
        "--card", str(card),
        "--accelerator", "c_fake",
        "--parameters", "p_fake",
        "--candidate-fragment", 'engine="modern_cpu";rng_backend="counter";',
        "--expected-candidate-engine", "modern_cpu",
        "--candidate-label", "modern",
        "--output", str(output),
        "--repetitions", str(repetitions),
        "--warmups", "0",
        "--timeout", "5",
        "--bootstrap-draws", "200",
        "--dependency", str(dependency),
        "--env", f"FAKE_SCENARIO={scenario}",
    ]
    if dry_run:
        command.append("--dry-run")
    if smoke:
        command.append("--smoke")
    return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)


def test_dry_run_and_source_immutability() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        binary, card, dependency = prepare(directory)
        before = hashlib.sha256(card.read_bytes()).hexdigest()
        output = directory / "dry"
        completed = subprocess.run([
            sys.executable, str(HARNESS),
            "--binary", str(binary), "--card", str(card),
            "--accelerator", "c_fake", "--parameters", "p_fake",
            "--candidate-fragment", 'engine="modern_cpu";',
            "--expected-candidate-engine", "modern_cpu",
            "--output", str(output), "--dependency", str(dependency),
            "--dry-run",
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        assert completed.returncode == 0, completed.stderr
        assert hashlib.sha256(card.read_bytes()).hexdigest() == before
        campaign = json.loads((output / "campaign_manifest.json").read_text())
        assert campaign["status"] == "dry_run"
        assert campaign["source_card_sha256"] == before
        assert campaign["materialized_card_sha256"]["legacy"] == before
        explicit = [item for item in campaign["hash_inventory"]
                    if "explicit_dependency" in item["roles"]]
        assert len(explicit) == 1 and explicit[0]["sha256"] == hashlib.sha256(
            dependency.read_bytes()).hexdigest()
        assert not (output / "runs").exists()


def test_good_fake_is_balanced_and_exploratory() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        completed = run_harness(directory, "good", "good", repetitions=2)
        assert completed.returncode == 0, completed.stderr
        output = directory / "good"
        samples = load_jsonl(output / "samples.jsonl")
        assert len(samples) == 4 and all(sample["valid"] for sample in samples)
        by_block = {}
        for sample in samples:
            by_block.setdefault(sample["block"], []).append(sample)
        assert [item["variant"] for item in sorted(
            by_block[0], key=lambda item: item["order_in_block"]
        )] == ["legacy", "modern"]
        assert [item["variant"] for item in sorted(
            by_block[1], key=lambda item: item["order_in_block"]
        )] == ["modern", "legacy"]
        candidate = next(item for item in samples if item["candidate"])
        assert candidate["candidate_manifest_present"]
        assert candidate["stdout_sha256"] and candidate["stderr_sha256"]
        assert "result.dat" in candidate["output_sha256"]
        analysis = json.loads((output / "analysis.json").read_text())
        assert analysis["label"] == "EXPLORATORY_TIMING_ONLY"
        assert analysis["qualification_pass"] is None
        assert analysis["simultaneous_physics_and_performance_pass"] is None
        assert analysis["paired_samples"] == 2
        assert analysis["paired_log_median_speedup"] is not None


def test_smoke_mode_is_one_paired_block() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        completed = run_harness(
            directory, "good", "smoke", repetitions=9, smoke=True
        )
        assert completed.returncode == 0, completed.stderr
        samples = load_jsonl(directory / "smoke" / "samples.jsonl")
        assert len(samples) == 2
        campaign = json.loads(
            (directory / "smoke" / "campaign_manifest.json").read_text()
        )
        assert campaign["smoke"] is True and campaign["valid_paired_blocks"] == 1


def assert_controlled_failure(scenario: str, expected_error: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        completed = run_harness(directory, scenario, "failed")
        assert completed.returncode != 0
        samples = load_jsonl(directory / "failed" / "samples.jsonl")
        assert len(samples) == 2
        candidate = next(item for item in samples if item["candidate"])
        assert not candidate["valid"]
        assert any(expected_error in error for error in candidate["errors"]), candidate
        campaign = json.loads(
            (directory / "failed" / "campaign_manifest.json").read_text()
        )
        assert campaign["status"] == "failed"
        assert campaign["invalid_sample_count"] >= 1
        analysis = json.loads((directory / "failed" / "analysis.json").read_text())
        assert analysis["qualification_pass"] is None


def test_dead_binary_is_retained_and_rejected() -> None:
    assert_controlled_failure("dead", "process return code is 7")


def test_wrong_output_is_retained_and_rejected() -> None:
    assert_controlled_failure("wrong_output", "paired output hash mismatch")


def test_fallback_manifest_is_retained_and_rejected() -> None:
    assert_controlled_failure("fallback", "candidate reported fallback")


def test_backend_mismatch_is_retained_and_rejected() -> None:
    assert_controlled_failure("wrong_engine", "actual engine/backend mismatch")


def test_extra_output_is_retained_and_rejected() -> None:
    assert_controlled_failure("extra_output", "extra output files: unexpected.dat")


def main() -> int:
    test_dry_run_and_source_immutability()
    test_good_fake_is_balanced_and_exploratory()
    test_smoke_mode_is_one_paired_block()
    test_dead_binary_is_retained_and_rejected()
    test_wrong_output_is_retained_and_rejected()
    test_fallback_manifest_is_retained_and_rejected()
    test_backend_mismatch_is_retained_and_rejected()
    test_extra_output_is_retained_and_rejected()
    print("modern benchmark harness checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
