#!/usr/bin/env python3
"""Independent verifier for the sealed six-cell luminosity benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
from typing import Any


EXPECTED_BINARY = "fde8656602fd35baa28604bc67ada21d769835eec5bde406269f1a5a5bc9d734"
EXPECTED_DECK = "9a1de68dae835271efcb7b4d1f45099f726a976d3b2295c3274c58107d0ec77e"
EXPECTED_RNG = "f766bd9a36452c94208a47c60637aee992931e6adabd75838b0d1528cc932892"
EXPECTED_PREREGISTRATION = (
    "e028fea9b142592fb6fb02321abecf414c94442041f0e8ad5fbbdb7b41ed2dea"
)
EXPECTED_RUN_MATRIX = (
    "2a5ac42804d91264af85ad5975a7d908b5a12284401d2c8a3897d4a66ed713b9"
)
EXPECTED_RUNNER = (
    "92cc2d333652a9e64a05425f42c16081f9ac6e3d04e5e0c434a864b5557c2b2d"
)
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20260826
RELATIVE_GATE = 2.0e-6
ABSOLUTE_GATE = 2.0e-6
VARIANTS = {
    "e0p0": {"electron_lumi_precompute": 0, "photon_lumi_factorized": 0},
    "e1p0": {"electron_lumi_precompute": 1, "photon_lumi_factorized": 0},
    "e2p0": {"electron_lumi_precompute": 2, "photon_lumi_factorized": 0},
    "e0p1": {"electron_lumi_precompute": 0, "photon_lumi_factorized": 1},
    "e1p1": {"electron_lumi_precompute": 1, "photon_lumi_factorized": 1},
    "e2p1": {"electron_lumi_precompute": 2, "photon_lumi_factorized": 1},
}
EXPECTED_COMPARISONS = {
    "e0p1_vs_e0p0", "e1p0_vs_e0p0", "e1p1_vs_e0p1",
    "e1p1_vs_e1p0", "e2p0_vs_e0p0", "e2p0_vs_e1p0",
    "e2p1_vs_e0p0", "e2p1_vs_e0p1", "e2p1_vs_e1p1",
    "e2p1_vs_e2p0",
}
PHOTON_NAMES = tuple(
    ["lumi_eg", "lumi_ge", "lumi_gg", "lumi_gg_high"] +
    [f"lumi[{row}][{column}]" for row in range(3) for column in range(3)]
)
PHOTON_RE = re.compile(
    r"^(lumi_(?:eg|ge|gg|gg_high)|lumi\[[0-2]\]\[[0-2]\])\s*=\s*"
    r"([-+0-9.eE]+)", re.MULTILINE,
)
ELECTRON_MARKER = re.compile(
    br"^ electron_lumi_precompute = [12] .*\n", re.MULTILINE
)
PHOTON_MARKER = re.compile(
    br"^ photon_lumi_factorized = 1 .*\n", re.MULTILINE
)
FAST_SUMMARY = re.compile(
    r"FEEBL legacy photon-lumi summary:\s*"
    r"requested=factorized-exact-v2.*?factorized_calls=(\d+).*?"
    r"cartesian_calls=(\d+).*?cartesian_pairs_visited=(\d+).*?"
    r"cartesian_pairs_avoided=(\d+)", re.DOTALL,
)


class Invalid(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_artifact(root: pathlib.Path, record: Any, description: str) -> pathlib.Path:
    require(isinstance(record, dict), f"{description}: record missing")
    relative = pathlib.Path(record.get("path", ""))
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"{description}: unsafe path")
    resolved = (root / relative).resolve()
    require(root.resolve() in resolved.parents, f"{description}: path escapes root")
    require(resolved.is_file(), f"{description}: file missing")
    require(resolved.stat().st_size == record.get("bytes"),
            f"{description}: byte count mismatch")
    require(sha256(resolved) == record.get("sha256"),
            f"{description}: SHA256 mismatch")
    return resolved


def verify_aborted_attempt(prereg_root: pathlib.Path) -> None:
    """Verify that the superseded card interpretation remains fully retained."""
    outer = json.loads(
        (prereg_root / "ABORTED-ATTEMPT-01.json").read_text(encoding="utf-8")
    )
    inner = json.loads(
        (prereg_root / "run" / "ABORTED.json").read_text(encoding="utf-8")
    )
    for record, description in ((outer, "outer"), (inner, "inner")):
        require(record.get("status") == "aborted-before-first-timed-sample",
                f"{description} abort status mismatch")
        require(record.get("timed_samples") == 0,
                f"{description} abort unexpectedly contains timed samples")
    require(outer.get("classification") ==
            "operator-card-identity-abort-not-performance-evidence",
            "outer abort classification mismatch")
    require(outer.get("interpret_elapsed_as_performance") is False,
            "outer abort permits a performance interpretation")
    require(inner.get("classification") ==
            "operator-card-identity-abort-not-performance-evidence",
            "inner abort classification mismatch")
    require(inner.get("interpret_any_partial_elapsed_state_as_performance") is False,
            "inner abort permits a performance interpretation")

    manifest = prereg_root / "ABORTED-ATTEMPT-01.SHA256SUMS"
    require(manifest.is_file(), "aborted-attempt checksum manifest missing")
    recorded: set[pathlib.Path] = set()
    for line_number, line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), start=1):
        pieces = line.split("  ", 1)
        require(len(pieces) == 2 and re.fullmatch(r"[0-9a-f]{64}", pieces[0]) is not None,
                f"aborted manifest line {line_number}: malformed")
        relative = pathlib.Path(pieces[1])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"aborted manifest line {line_number}: unsafe path")
        require(relative not in recorded,
                f"aborted manifest line {line_number}: duplicate path")
        recorded.add(relative)
        path = prereg_root / relative
        require(path.is_file() and sha256(path) == pieces[0],
                f"aborted manifest line {line_number}: hash mismatch")
    retained = {
        path.relative_to(prereg_root) for path in (prereg_root / "run").rglob("*")
        if path.is_file()
    }
    retained.add(pathlib.Path("runner-console.log"))
    require(recorded == retained,
            "aborted manifest does not exactly cover the retained partial attempt")


def electron_payload(payload: bytes) -> bytes:
    selected: list[bytes] = []
    in_steps = False
    steps = 0
    for line in payload.splitlines(keepends=True):
        if (line.startswith(b"lumi_ee = ") or
                line.startswith(b"lumi_ee_high = ") or
                line.startswith(b"E_cm = ") or
                line.startswith(b"lumi_ee=") or
                line.startswith(b"lumi_ee_high=")):
            selected.append(line)
        if line.startswith(b"step ") and b"lumi total" in line:
            selected.append(line)
            in_steps = True
            continue
        if in_steps:
            if re.fullmatch(br"[0-9]+[ \t]+[-+0-9.eE]+[ \t]+[-+0-9.eE]+\r?\n?", line):
                selected.append(line)
                steps += 1
            else:
                in_steps = False
    require(steps > 0 and len(selected) >= 6, "electron payload parse failed")
    return b"".join(selected)


def photon_aggregates(payload: bytes) -> dict[str, float]:
    matches = PHOTON_RE.findall(payload.decode("utf-8", errors="strict"))
    found = {name: float(value) for name, value in matches}
    require(len(matches) == 13 and set(found) == set(PHOTON_NAMES),
            "photon aggregate parse failed")
    require(all(math.isfinite(value) for value in found.values()),
            "nonfinite photon aggregate")
    return found


def close_photon(first: dict[str, float], second: dict[str, float]) -> bool:
    return all(
        abs(first[name] - second[name]) <= max(
            ABSOLUTE_GATE,
            RELATIVE_GATE * max(abs(first[name]), abs(second[name])),
        ) for name in PHOTON_NAMES
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def recompute_comparison(seconds: dict[str, list[float]], baseline: str,
                         candidate: str) -> dict[str, Any]:
    first, second = seconds[baseline], seconds[candidate]
    paired = [left / right for left, right in zip(first, second)]
    material = f"{BOOTSTRAP_SEED}:{baseline}:{candidate}".encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = random.Random(seed)
    bootstrap = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [generator.randrange(9) for _ in range(9)]
        bootstrap.append(
            statistics.median(first[index] for index in indices) /
            statistics.median(second[index] for index in indices)
        )
    return {
        "ratio": statistics.median(first) / statistics.median(second),
        "paired": statistics.median(paired),
        "paired_values": paired,
        "lower": percentile(bootstrap, 0.025),
        "upper": percentile(bootstrap, 0.975),
        "seed": seed,
    }


def same_float(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=2.0e-15)


def validate_csv_summaries(document: dict[str, Any], prereg_root: pathlib.Path) -> None:
    with (prereg_root / "VARIANT_SUMMARY.csv").open(
            newline="", encoding="utf-8") as stream:
        variant_rows = {row["variant"]: row for row in csv.DictReader(stream)}
    require(set(variant_rows) == set(VARIANTS), "variant CSV row set mismatch")
    for variant, settings in VARIANTS.items():
        row = variant_rows[variant]
        summary = document["summary_by_variant"][variant]
        require(int(row["electron_lumi_precompute"]) ==
                settings["electron_lumi_precompute"] and
                int(row["photon_lumi_factorized"]) ==
                settings["photon_lumi_factorized"] and
                int(row["timed_blocks"]) == 9,
                f"{variant}: CSV selector/sample mismatch")
        for column, result_key in (
                ("median_seconds", "median_seconds"),
                ("mean_seconds", "mean_seconds"),
                ("minimum_seconds", "minimum_seconds"),
                ("maximum_seconds", "maximum_seconds")):
            require(same_float(float(row[column]), summary[result_key]),
                    f"{variant}: CSV {column} mismatch")

    with (prereg_root / "COMPARISON_SUMMARY.csv").open(
            newline="", encoding="utf-8") as stream:
        comparison_rows = {
            row["comparison"]: row for row in csv.DictReader(stream)
        }
    require(set(comparison_rows) == EXPECTED_COMPARISONS,
            "comparison CSV row set mismatch")
    for key, stored in document["comparisons"].items():
        row = comparison_rows[key]
        interval = stored["paired_bootstrap_ratio_of_medians_95pct"]
        candidate, baseline = key.split("_vs_")
        require(row["baseline"] == baseline and row["candidate"] == candidate and
                row["broad_claim_eligible"] == "false" and
                int(row["bootstrap_replicates"]) == BOOTSTRAP_REPLICATES,
                f"{key}: CSV identity/classification mismatch")
        for column, value in (
                ("ratio_of_medians_speedup", stored["ratio_of_medians_speedup"]),
                ("median_paired_speedup", stored["median_paired_speedup"]),
                ("bootstrap_95pct_lower", interval["lower"]),
                ("bootstrap_95pct_upper", interval["upper"])):
            require(same_float(float(row[column]), value),
                    f"{key}: CSV {column} mismatch")


def validate(evidence: pathlib.Path, prereg_root: pathlib.Path) -> dict[str, Any]:
    result_path = evidence / "results.json"
    require(result_path.is_file(), "results.json missing")
    document = json.loads(result_path.read_text(encoding="utf-8"))
    prereg_path = prereg_root / "PREREGISTRATION.json"
    amendment_path = prereg_root / "AMENDMENT-01.json"
    run_matrix_path = prereg_root / "RUN_MATRIX.csv"
    prereg = json.loads(prereg_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    require(sha256(prereg_path) == EXPECTED_PREREGISTRATION,
            "preregistration identity mismatch")
    require(sha256(run_matrix_path) == EXPECTED_RUN_MATRIX,
            "run matrix identity mismatch")
    require(amendment.get("prior_preregistration_sha256") ==
            EXPECTED_PREREGISTRATION, "amendment preregistration hash mismatch")
    require(amendment.get("run_matrix_sha256") == EXPECTED_RUN_MATRIX,
            "amendment run-matrix hash mismatch")
    require(amendment.get("amended_runner_sha256") == EXPECTED_RUNNER,
            "amendment runner hash mismatch")
    require(prereg.get("recorded_before_first_timing") is True and
            amendment.get("recorded_before_first_timed_sample") is True,
            "matrix/amendment timing freeze mismatch")
    verify_aborted_attempt(prereg_root)
    require(document.get("schema") ==
            "feebl-electron-photon-lumi-fixed-matrix-v1", "schema mismatch")
    require(document.get("valid") is True, "runner did not mark evidence valid")
    require(document.get("eligible_for_broad_or_project_level_speedup_claim") is False,
            "evidence incorrectly claims broad eligibility")
    require(document.get("matrix_frozen_before_first_timing") is True,
            "matrix freeze claim missing")
    require(document.get("cpu") == 0 and document.get("timed_blocks") == 9 and
            document.get("warmup_cycles") == 1, "run-count/CPU contract mismatch")
    require(document.get("variants") == VARIANTS, "variant matrix mismatch")
    require(document.get("schedule") == prereg.get("schedule") ==
            amendment.get("schedule"), "schedule mismatch")
    require(amendment.get("exact_reduced_deck_sha256") == EXPECTED_DECK,
            "amendment deck hash mismatch")

    provenance = document.get("provenance", {})
    binary = safe_artifact(evidence, provenance.get("binary"), "binary")
    source_deck = safe_artifact(evidence, provenance.get("source_deck"), "source deck")
    runner = safe_artifact(evidence, provenance.get("runner"), "runner")
    safe_artifact(evidence, provenance.get("preregistration"), "preregistration")
    safe_artifact(evidence, provenance.get("amendment"), "amendment")
    safe_artifact(evidence, provenance.get("toolchain"), "toolchain")
    safe_artifact(evidence, provenance.get("system_preflight"), "preflight")
    safe_artifact(evidence, provenance.get("system_postflight"), "postflight")
    require(sha256(binary) == EXPECTED_BINARY, "binary identity mismatch")
    require(sha256(source_deck) == EXPECTED_DECK, "source deck identity mismatch")
    require(sha256(runner) == EXPECTED_RUNNER, "runner identity mismatch")
    require(provenance.get("git", {}).get("commit") ==
            "bd5271c2c3549155c8ed06eed6096e1d7e3b3a20", "git commit mismatch")
    require(provenance.get("environment") == {
        "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close", "OMP_PLACES": "{0}",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }, "environment mismatch")
    libraries = provenance.get("linked_libraries")
    require(isinstance(libraries, list) and libraries, "linked library hashes absent")
    for library in libraries:
        path = pathlib.Path(library["path"])
        require(path.is_file() and sha256(path) == library["sha256"] and
                path.stat().st_size == library["bytes"], "linked library changed")

    base = source_deck.read_bytes()
    selector = b"electron_lumi_precompute=0;photon_lumi_factorized=0;"
    require(base.count(selector) == 1, "frozen selector prefix absent")
    work_signatures = set()
    derived_paths: set[pathlib.Path] = set()
    for name, settings in VARIANTS.items():
        card_record = document["derived_cards"][name]
        card = safe_artifact(evidence, card_record, f"card {name}")
        require(card not in derived_paths, f"card {name}: duplicate artifact path")
        derived_paths.add(card)
        replacement = (
            f"electron_lumi_precompute={settings['electron_lumi_precompute']};"
            f"photon_lumi_factorized={settings['photon_lumi_factorized']};"
        ).encode()
        require(card.read_bytes() == base.replace(selector, replacement, 1),
                f"card {name} differs beyond selectors")
        work_signatures.add(card_record["work_signature_sha256"])
    require(len(work_signatures) == 1 and next(iter(work_signatures)) ==
            document.get("input_work_signature_sha256"), "work signature mismatch")

    samples = document.get("samples")
    require(isinstance(samples, list) and len(samples) == 60, "sample count mismatch")
    expected_timed = {
        (block, slot): variant for block, order in enumerate(document["schedule"])
        for slot, variant in enumerate(order)
    }
    expected_warmup = {
        slot: variant for slot, variant in enumerate(prereg["warmup_order"])
    }
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    parsed: dict[int, dict[str, Any]] = {}
    loads: list[float] = []
    temperature_readings = 0
    seen_sample_paths: set[pathlib.Path] = set()
    for index, sample in enumerate(samples):
        variant = sample.get("variant")
        require(variant in VARIANTS and sample.get("selectors") == VARIANTS[variant],
                f"sample {index}: selector mismatch")
        if sample.get("warmup"):
            require(sample.get("block") == 0 and
                    expected_warmup.get(sample.get("order_index")) == variant,
                    f"sample {index}: warmup order mismatch")
        else:
            require(expected_timed.get((sample.get("block"), sample.get("order_index"))) ==
                    variant, f"sample {index}: timed order mismatch")
        command = sample.get("command")
        require(isinstance(command, list) and command[:3] ==
                ["taskset", "--cpu-list", "0"] and command[-3:] ==
                ["s9", "p_s9", "result.out"], f"sample {index}: command mismatch")
        artifacts = sample.get("artifacts", {})
        paths = {name: safe_artifact(evidence, artifacts.get(name),
                                     f"sample {index} {name}")
                 for name in ("stdout", "stderr", "result", "rng", "timing",
                              "system_before", "system_after")}
        for name, path in paths.items():
            require(path not in seen_sample_paths,
                    f"sample {index} {name}: artifact path reused")
            seen_sample_paths.add(path)
        timing = json.loads(paths["timing"].read_text())
        require(timing["clock"] == "time.perf_counter_ns" and
                timing["end_ns"] > timing["start_ns"] and
                timing["elapsed_ns"] == timing["end_ns"] - timing["start_ns"] and
                timing["elapsed_ns"] == sample["elapsed_ns"] and
                same_float(timing["seconds"], sample["seconds"]),
                f"sample {index}: timing mismatch")
        for snapshot_name in ("system_before", "system_after"):
            snapshot = json.loads(paths[snapshot_name].read_text())
            require(all(key in snapshot for key in
                        ("load1", "load5", "load15", "temperatures",
                         "temperature_status", "cpu_frequencies",
                         "frequency_status")), f"sample {index}: snapshot incomplete")
            loads.append(float(snapshot["load1"]))
            temperature_readings += len(snapshot["temperatures"])
        payload = paths["result"].read_bytes()
        rng_hash = sha256(paths["rng"])
        normalized = PHOTON_MARKER.sub(b"", ELECTRON_MARKER.sub(b"", payload))
        electron = electron_payload(payload)
        aggregates = photon_aggregates(payload)
        require(hashlib.sha256(normalized).hexdigest() ==
                sample["normalized_result_sha256"],
                f"sample {index}: normalized output digest mismatch")
        require(hashlib.sha256(electron).hexdigest() ==
                sample["electron_payload_sha256"],
                f"sample {index}: electron digest mismatch")
        require(aggregates == sample["photon_aggregates"],
                f"sample {index}: photon aggregates mismatch")
        stderr = paths["stderr"].read_text(errors="replace")
        fast = FAST_SUMMARY.search(stderr)
        if VARIANTS[variant]["photon_lumi_factorized"]:
            require(fast is not None, f"sample {index}: fast provenance absent")
            counters = tuple(map(int, fast.groups()))
            require(counters == (64, 0, 0, 151024448),
                    f"sample {index}: fast work counters mismatch")
        else:
            require(fast is None, f"sample {index}: unexpected fast provenance")
        require(sample.get("exit_code") == 0, f"sample {index}: nonzero exit")
        grouped[variant].append({
            "result_sha": sha256(paths["result"]), "rng_sha": rng_hash,
            "normalized": hashlib.sha256(normalized).hexdigest(),
            "electron": hashlib.sha256(electron).hexdigest(),
            "photon": aggregates, "seconds": sample["seconds"],
            "warmup": sample["warmup"], "block": sample["block"],
        })
        parsed[index] = grouped[variant][-1]

    require(all(len(rows) == 10 for rows in grouped.values()),
            "per-variant repetition count mismatch")
    rng_hashes = {row["rng_sha"] for rows in grouped.values() for row in rows}
    require(rng_hashes == {EXPECTED_RNG}, "RNG bytes differ")
    for variant, rows in grouped.items():
        require(len({row["result_sha"] for row in rows}) == 1,
                f"{variant}: repeated output differs")
    electron_hashes = {row["electron"] for rows in grouped.values() for row in rows}
    require(len(electron_hashes) == 1, "electron payload differs across matrix")
    for photon in (0, 1):
        hashes = {grouped[f"e{electron}p{photon}"][0]["normalized"]
                  for electron in (0, 1, 2)}
        require(len(hashes) == 1, f"photon {photon}: electron modes not exact")
    for electron in (0, 1, 2):
        require(close_photon(grouped[f"e{electron}p0"][0]["photon"],
                             grouped[f"e{electron}p1"][0]["photon"]),
                f"electron {electron}: photon physics gate failed")

    seconds: dict[str, list[float]] = {name: [0.0] * 9 for name in VARIANTS}
    for rows in grouped.values():
        for row in rows:
            if not row["warmup"]:
                # Variant recovered by identity below.
                pass
    for variant, rows in grouped.items():
        for row in rows:
            if not row["warmup"]:
                seconds[variant][row["block"]] = row["seconds"]
    require(seconds == document["seconds_by_variant_in_block_order"],
            "seconds-by-block table mismatch")
    for variant, values in seconds.items():
        summary = document["summary_by_variant"][variant]
        require(same_float(summary["median_seconds"], statistics.median(values)) and
                same_float(summary["minimum_seconds"], min(values)) and
                same_float(summary["maximum_seconds"], max(values)) and
                same_float(summary["mean_seconds"], statistics.mean(values)),
                f"{variant}: summary mismatch")
    require(set(document.get("comparisons", {})) == EXPECTED_COMPARISONS,
            "comparison matrix incomplete or unexpected")
    require(set(prereg.get("required_comparisons", ())) <= EXPECTED_COMPARISONS,
            "preregistered comparison absent")
    for key, stored in document["comparisons"].items():
        candidate, baseline = key.split("_vs_")
        computed = recompute_comparison(seconds, baseline, candidate)
        interval = stored["paired_bootstrap_ratio_of_medians_95pct"]
        require(same_float(stored["ratio_of_medians_speedup"], computed["ratio"]) and
                same_float(stored["median_paired_speedup"], computed["paired"]) and
                all(same_float(a, b) for a, b in
                    zip(stored["paired_speedups"], computed["paired_values"])) and
                same_float(interval["lower"], computed["lower"]) and
                same_float(interval["upper"], computed["upper"]) and
                interval["seed"] == computed["seed"] and
                interval["replicates"] == BOOTSTRAP_REPLICATES,
                f"{key}: statistic/bootstrap mismatch")
    validate_csv_summaries(document, prereg_root)

    artifact_hash_records = len(samples) * 7 + 8 + len(VARIANTS)
    return {
        "schema": "feebl-electron-photon-lumi-fixed-matrix-validation-v1",
        "valid": True,
        # The runner emitted one results document plus the artifacts addressed
        # below.  Derive this count from the sealed schema so later validator
        # reports stored beside the evidence cannot change it.
        "verified_files": artifact_hash_records + 1,
        "samples": len(samples),
        "timed_samples": sum(not sample["warmup"] for sample in samples),
        "fresh_process_cpu0_commands": len(samples),
        "artifact_hash_records_verified": artifact_hash_records,
        "exact_rng_sha256": next(iter(rng_hashes)),
        "exact_electron_payload_sha256": next(iter(electron_hashes)),
        "factorized_calls_per_run": 64,
        "factorized_cartesian_pairs_avoided_per_run": 151024448,
        "load1_min": min(loads),
        "load1_max": max(loads),
        "temperature_readings": temperature_readings,
        "temperature_status": (
            "available" if temperature_readings else "unavailable-in-WSL"
        ),
        "statistics_recomputed": sorted(document["comparisons"]),
        "broad_claim_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument("--prereg-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    evidence = args.evidence.resolve(strict=True)
    prereg_root = args.prereg_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    report = validate(evidence, prereg_root)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
