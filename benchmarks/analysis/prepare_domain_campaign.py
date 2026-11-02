#!/usr/bin/env python3
"""Freeze HALHF transverse-domain tests at fixed cell spacing.

Execute each generated job with seed_campaign.py execute. This program does
not submit a scheduler job or execute the simulation. Baseline seeds match the
first three replicas of the eight-family independent-seed campaign.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import shutil

from seed_campaign import BENCHMARKS, digest, seed_for_case


def assignment(card, key):
    """Read exactly one scalar card assignment without evaluating expressions."""
    values = re.findall(rf"(?<!\w){re.escape(key)}\s*=\s*([^;]+);", card)
    if len(values) != 1:
        raise ValueError(f"Expected one {key} assignment")
    return float(values[0])


def replace(card, key, value):
    """Replace one scalar setting and reject ambiguous input cards."""
    card, count = re.subn(
        rf"(?<!\w){re.escape(key)}\s*=\s*[^;]+;", f"{key}={value};", card
    )
    if count != 1:
        raise ValueError(f"Expected one {key} assignment")
    return card


def prepare(output):
    """Expand both transverse half-widths and cell counts by two and four."""
    cases = {
        case["id"]: case
        for case in json.loads((BENCHMARKS / "manifest.json").read_text())["cases"]
    }
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for case_id in ("halhf380", "halhf550"):
        case = cases[case_id]
        for factor in (2, 4):
            for replicate in range(3):
                seed = seed_for_case(case_id, replicate)
                study_id = f"{case_id}_domain_x{factor}"
                relative = Path(study_id) / f"seed_{seed}"
                directory = output / relative
                directory.mkdir(parents=True)
                for name, source in case["inputs"].items():
                    shutil.copyfile(BENCHMARKS / source, directory / name)
                card = (directory / "acc.dat").read_text(encoding="utf-8-sig")
                baseline = {
                    key: assignment(card, key)
                    for key in ("n_x", "n_y", "cut_x", "cut_y")
                }
                for key, value in baseline.items():
                    changed = (
                        int(value * factor)
                        if key.startswith("n_")
                        else format(value * factor, ".14g")
                    )
                    card = replace(card, key, changed)
                for key, value in [
                    ("rndm_seed", seed),
                    ("rndm_load", 0),
                    ("rndm_save", 0),
                ]:
                    card = replace(card, key, value)
                (directory / "acc.dat").write_text(card, encoding="utf-8", newline="\n")
                jobs.append(
                    {
                        "index": len(jobs),
                        "case": study_id,
                        "baseline_case": case_id,
                        "domain_factor": factor,
                        "seed": seed,
                        "directory": relative.as_posix(),
                        "argv": [
                            *case["options"],
                            case["accelerator"],
                            case["parameters"],
                            "out.dat",
                        ],
                        "timeout_seconds": 1800,
                        "baseline_transverse_grid": baseline,
                        "input_sha256": {
                            p.name: digest(p) for p in directory.iterdir()
                        },
                    }
                )
    document = {
        "schema": "feebl.seed-campaign/1",
        "seed_count": 3,
        "purpose": "Transverse-domain refinement with fixed transverse cell spacing",
        "qualification": "Matched RNG seeds; branching histories may decorrelate. Not a grid-spacing test.",
        "jobs": jobs,
    }
    (output / "campaign.json").write_text(json.dumps(document, indent=2) + "\n")
    return document


def reduce(campaign, output):
    """Export native luminosity, photon/loss and outside-grid diagnostics."""
    document = json.loads((campaign / "campaign.json").read_text())
    records = []
    keys = (
        "lumi_ee",
        "lumi_ee_high",
        "lumi_gg",
        "n_phot1",
        "n_phot2",
        "de1",
        "de2",
        "out.1",
        "out.2",
    )
    number = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?"
    for job in document["jobs"]:
        directory = campaign / job["directory"]
        receipt = json.loads((directory / "execution.json").read_text())
        if receipt["status"] != "EXECUTED":
            raise ValueError(f"Unsuccessful native run: {job['directory']}")
        text = (directory / "out.dat").read_text()
        row = {
            key: job.get(key, "")
            for key in ("case", "baseline_case", "domain_factor", "seed")
        }
        for key in keys:
            values = re.findall(rf"(?<![\w.]){re.escape(key)}\s*=\s*({number})", text)
            if not values:
                raise ValueError(f"Missing {key} in {job['directory']}")
            row[key] = float(values[-1])
        misses = re.findall(rf"beam\s+([12])\s*:\s*miss\s*=\s*({number})", text)
        for beam, value in misses:
            row[f"outside_fraction_beam{beam}"] = float(value)
        if len(misses) != 2:
            raise ValueError("Native outside-grid fractions missing")
        row.update(
            elapsed_seconds=receipt["elapsed_seconds"],
            binary_sha256=receipt["binary_sha256"],
            output_sha256=digest(directory / "out.dat"),
        )
        records.append(row)
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "reduce"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        print(
            f"Prepared {len(prepare(args.output)['jobs'])} domain replicas; no simulation executed."
        )
    elif args.campaign is None:
        parser.error("reduce requires --campaign")
    else:
        reduce(args.campaign, args.output)
