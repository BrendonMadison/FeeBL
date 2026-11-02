#!/usr/bin/env python3
"""Combine disjoint complete collision families after explicit version controls.

Different native versions may define different families, but are never pooled
inside one sampling ensemble. Reusing an older family requires an exact-spectrum
and scientific-summary control using its first declared seed under both builds.
These controls test the exercised configurations, not arbitrary equivalence of
the two programs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from comparison_statistics import PANELS, panel_case, read_rows
from seed_campaign import DEFAULT_CASES, seed_for_case


def digest(path):
    """Identify one retained compact input without modifying it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_hash(value):
    """Require an explicit SHA-256 provenance value."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check_controls(document, historical, new_binary):
    """Require four honest per-case controls, preserving any allowed metadata."""
    if (
        document.get("schema") != "feebl.cross-version-controls/1"
        or document.get("status") != "PASS"
    ):
        raise ValueError("Cross-version controls have not passed")
    controls = document.get("controls", [])
    if len({row["case"] for row in controls}) != len(controls):
        raise ValueError("Repeated case in version-control receipt")
    by_case = {row["case"]: row for row in controls}
    for case, old_binary in historical.items():
        if case not in by_case:
            raise ValueError(f"Missing version control for historical family {case}")
        row = by_case[case]
        if row.get("seed") != seed_for_case(case, 0):
            raise ValueError(f"Control did not use the first declared seed for {case}")
        if (
            row.get("old_binary_sha256") != old_binary
            or row.get("new_binary_sha256") != new_binary
        ):
            raise ValueError(f"Control binary mismatch for {case}")
        old_hash, new_hash = (
            row.get("old_spectrum_sha256"),
            row.get("new_spectrum_sha256"),
        )
        if not valid_hash(old_hash) or not valid_hash(new_hash):
            raise ValueError(f"Raw spectrum identities are missing for {case}")
        if old_hash != new_hash:
            payload = row.get("old_spectrum_payload_sha256")
            if not valid_hash(payload) or payload != row.get(
                "new_spectrum_payload_sha256"
            ):
                raise ValueError(f"Spectrum changed under the control for {case}")
            metadata = row.get("ignored_spectrum_metadata", [])
            if (
                len(metadata) != 1
                or set(metadata[0]) != {"name", "old", "new"}
                or metadata[0]["name"] != "source_revision"
            ):
                raise ValueError(f"Unapproved spectrum-header omission for {case}")
            if any(
                not isinstance(metadata[0][key], str)
                or not metadata[0][key]
                or "\n" in metadata[0][key]
                for key in ("old", "new")
            ):
                raise ValueError(
                    f"Source-revision metadata must be explicit for {case}"
                )
        if row.get("scientific_summary_equal") is not True:
            raise ValueError(f"Scientific summary changed under the control for {case}")
        ignored = row.get("ignored_summary_metadata", {})
        if ignored not in (
            {},
            [],
            {"imported_spin_mode": 0},
            {"imported_spin_mode": "0"},
            [{"name": "imported_spin_mode", "value": 0}],
        ):
            raise ValueError(f"Unapproved scientific-summary omission for {case}")


def merge(datasets, source_hashes, controls_path, output):
    """Write a new complete eight-family ensemble with per-family provenance."""
    datasets = [Path(path) for path in datasets]
    output = Path(output)
    if output.exists():
        raise FileExistsError("Merged output must be new")
    if len(datasets) != 2 or len(source_hashes) != 2:
        raise ValueError(
            "Supply historical and repaired complete-family reductions, in that order"
        )
    if not all(valid_hash(value) for value in source_hashes):
        raise ValueError("Each native source archive needs its SHA-256")
    controls = json.loads(Path(controls_path).read_text())
    for key, expected in zip(
        ("old_source_archive_sha256", "new_source_archive_sha256"), source_hashes
    ):
        if key in controls and controls[key] != expected:
            raise ValueError(
                "Control source-archive identity disagrees with the reduced campaign"
            )
    all_cases, all_panels, matrices, receipts = set(), [], {}, []
    family_binaries = []
    for index, (dataset, source_hash) in enumerate(zip(datasets, source_hashes)):
        metadata_path = dataset.with_suffix(".json")
        inventory_path = dataset.parent / "campaign_inventory.json"
        metadata = json.loads(metadata_path.read_text())
        inventory = json.loads(inventory_path.read_text())
        inventory_by_case = {row["case"]: row for row in inventory["cases"]}
        panel_records = metadata["panels"]
        cases = {row["case"] for row in panel_records}
        if cases & all_cases:
            raise ValueError("Do not pool, repeat or replace a case across reductions")
        if cases - set(DEFAULT_CASES):
            raise ValueError("Reduction includes an unexpected family")
        if any(
            not inventory_by_case[case]["complete_requested_ensemble"] for case in cases
        ):
            raise ValueError("A reduced family is incomplete in its original inventory")
        with np.load(dataset, allow_pickle=False) as values:
            expected_keys = {
                row["panel"] + suffix
                for row in panel_records
                for suffix in ("", "__seeds", "__encounter_neff")
            }
            if set(values.files) != expected_keys:
                raise ValueError("NPZ arrays and panel provenance disagree")
            binaries = {}
            for panel in panel_records:
                name, case = panel["panel"], panel["case"]
                if panel_case(name) != case or name in matrices:
                    raise ValueError("Repeated or misidentified panel")
                expected_seeds = [seed_for_case(case, replica) for replica in range(12)]
                if (
                    panel["seeds"] != expected_seeds
                    or values[name + "__seeds"].tolist() != expected_seeds
                ):
                    raise ValueError(
                        "The original twelve canonical seeds must be retained"
                    )
                if len(panel["binary_sha256"]) != 1 or not valid_hash(
                    panel["binary_sha256"][0]
                ):
                    raise ValueError(
                        "One complete family requires exactly one native binary"
                    )
                binary = panel["binary_sha256"][0]
                original_records = inventory_by_case[case]["records"]
                if {row["seed"] for row in original_records} != set(
                    expected_seeds
                ) or any(
                    row["status"] != "EXECUTED" or row["binary_sha256"] != binary
                    for row in original_records
                ):
                    raise ValueError("Original inventory and reduced family disagree")
                if case in binaries and binaries[case] != binary:
                    raise ValueError(
                        "Panels from the same case have different native versions"
                    )
                binaries[case] = binary
                expected_shape = (12, len(read_rows(PANELS / (name + ".csv"))))
                if (
                    values[name].shape != expected_shape
                    or not np.isfinite(values[name]).all()
                ):
                    raise ValueError("Malformed complete-family panel matrix")
                native_runs = panel["native_runs"]
                if [row["seed"] for row in native_runs] != expected_seeds:
                    raise ValueError("Native-run receipts do not cover all seeds")
                for row in native_runs:
                    if not valid_hash(row["native_sha256"]) or not valid_hash(
                        row["execution_receipt_sha256"]
                    ):
                        raise ValueError(
                            "Native output or execution-receipt identity is missing"
                        )
                all_panels.append(
                    {
                        **panel,
                        "native_source_archive_sha256": source_hash,
                        "reduction_input_index": index,
                    }
                )
            for key in values.files:
                matrices[key] = values[key].copy()
        family_binaries.append(binaries)
        all_cases.update(cases)
        receipts.append(
            {
                "input_index": index,
                "native_source_archive_sha256": source_hash,
                "seed_vectors_sha256": digest(dataset),
                "metadata_sha256": digest(metadata_path),
                "campaign_inventory_sha256": digest(inventory_path),
                "original_metadata": metadata,
                "original_campaign_inventory": inventory,
            }
        )
    if all_cases != set(DEFAULT_CASES):
        raise ValueError(
            "The complete comparison requires all eight requested families"
        )
    new_binaries = set(family_binaries[1].values())
    if len(new_binaries) != 1:
        raise ValueError("The repaired campaign must identify one new native binary")
    check_controls(controls, family_binaries[0], next(iter(new_binaries)))
    expected_panels = {
        path.stem
        for path in PANELS.glob("*_pull.csv")
        if not path.stem.startswith("lep_")
    }
    if {row["panel"] for row in all_panels} != expected_panels:
        raise ValueError("At least one required comparison panel is missing")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "seed_vectors.npz", **matrices)
    result = {
        "schema": "feebl.merged-seed-ensemble/1",
        "panels": all_panels,
        "sampling_unit": "independent full native collision",
        "all_requested_cases_complete": True,
        "purpose": "Disjoint complete families with tested cross-version controls; no within-family pooling",
        "bin_support": "packaged comparison bins, preserved by the recorded reduction sources",
        "producer_sha256": digest(__file__),
        "seed_vectors_sha256": digest(output / "seed_vectors.npz"),
        "reduction_inputs": receipts,
        "cross_version_controls": controls,
        "cross_version_controls_sha256": digest(controls_path),
        "qualification": "Exact controls establish unchanged exercised cases only. Seed covariance does not include model or discretization uncertainty.",
    }
    (output / "seed_vectors.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output / "campaign_inventory.json").write_text(
        json.dumps(
            {
                "schema": "feebl.merged-campaign-inventory/1",
                "all_requested_cases_complete": True,
                "original_campaigns": [
                    row["original_campaign_inventory"] for row in receipts
                ],
                "qualification": "Original failed and unexecuted attempts remain visible; the new complete ensemble is separately identified.",
            },
            indent=2,
        )
        + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs=2,
        required=True,
        type=Path,
        metavar=("HISTORICAL_NPZ", "REPAIRED_NPZ"),
    )
    parser.add_argument(
        "--source-archives",
        nargs=2,
        required=True,
        metavar=("HISTORICAL_SHA256", "REPAIRED_SHA256"),
    )
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = merge(args.inputs, args.source_archives, args.controls, args.output)
    print(
        json.dumps(
            {
                "families": len(DEFAULT_CASES),
                "panels": len(result["panels"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )
