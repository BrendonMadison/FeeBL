#!/usr/bin/env python3
"""Prepare matched field-on, zero-field and time-refined polarized probes.

Preparation does not run FeeBL or submit jobs. The generated campaign is accepted
by benchmarks/analysis/seed_campaign.py execute, which preserves input hashes,
native outputs, process limits and failure receipts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from generate_polarized_bhabha_events import generate


def replace_assignment(card, name, value):
    """Change one existing assignment, rejecting absent or ambiguous settings."""
    pattern = rf"(?<![\w.]){re.escape(name)}\s*=\s*[^;]+;"
    result, count = re.subn(pattern, f"{name}={value};", card)
    if count != 1:
        raise ValueError(f"Expected one assignment for {name}; found {count}.")
    return result


def prepare(destination, events=8192):
    """Freeze three matched inputs with unchanged event momenta and spin labels."""
    root = Path(__file__).resolve().parents[1]
    source_card = root / "benchmarks/cases/viewer_bhabha_field_on/acc.dat"
    card = source_card.read_text()
    if not re.search(r"\bn_z\s*=\s*16\s*;", card):
        raise ValueError("The manufactured admission counter requires n_z=16.")
    if "imported_spin_mode" in card:
        raise ValueError("The source card already supplies an imported-spin mode.")
    card = card.replace("load_events=1;", "load_events=1;imported_spin_mode=1;")
    destination.mkdir(parents=True, exist_ok=False)
    nominal = destination / "field_on"
    generate(nominal, events, 125, 1, (0.8, 0.3), 7292917, 240)
    zero = destination / "field_off"
    zero.mkdir()
    for name in ("event.ini", "polarized_event_identity.json"):
        shutil.copyfile(nominal / name, zero / name)
    refined = destination / "time_refined"
    generate(refined, events, 125, 1, (0.8, 0.3), 7292917, 480)
    nominal_rows = (nominal / "event.ini").read_text().splitlines()
    refined_rows = (refined / "event.ini").read_text().splitlines()
    if any(a.split()[1:] != b.split()[1:] for a, b in zip(nominal_rows, refined_rows)):
        raise ValueError("Time refinement changed a physical event input.")
    jobs = []
    for index, name in enumerate(("field_on", "field_off", "time_refined")):
        directory = destination / name
        selected = card
        if name == "field_off":
            selected = replace_assignment(selected, "particles.1", 0)
            selected = replace_assignment(selected, "particles.2", 0)
        if name == "time_refined":
            selected = replace_assignment(selected, "n_t", 4)
        (directory / "acc.dat").write_text(selected, newline="\n")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(directory.iterdir()) if p.is_file()}
        jobs.append(dict(index=index, case=name, seed=6102711, directory=name,
                         argv=["ilc250", "diagnostic", "out.dat"],
                         timeout_seconds=600, input_sha256=hashes))
    receipt = dict(
        schema="feebl.seed-campaign/1",
        purpose="Paired BMT transport of polarized imported Bhabha-like probes",
        events_per_run=events, tracks_per_run=2 * events,
        source_card_sha256=hashlib.sha256(source_card.read_bytes()).hexdigest(),
        qualification="Fixed central vertex and one sampled beam field; not a Bhabha spin-density-matrix prediction",
        processes="Imported finite-mass BMT and Lorentz deflection; imported radiation and spin flips disabled",
        field_model="Three transverse field grids, 5000 primary macroparticles per beam",
        refinement="n_t 2 to 4; physical event inputs fixed, admission counter 240 to 480",
        jobs=jobs)
    (destination / "campaign.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=8192)
    options = parser.parse_args()
    print(json.dumps(prepare(options.output, options.events), indent=2))
