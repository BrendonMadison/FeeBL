#!/usr/bin/env python3
"""Reproduce the manufactured back-to-back inputs used by the viewer examples.

Writes event.ini and event_identity.json into a new directory. The 128 events
have 125 GeV electron and positron energies, 1/179 degree polar angles, exact
opposite three-velocities and seeded uniform azimuth. No cross section is implied.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def generate(destination):
    """Write native event rows and their explicit identity at n_z=16, n_t=2."""
    destination.mkdir(parents=True, exist_ok=False)
    rng = random.Random(6102701)
    speed = math.sqrt(1 - (.00051099895 / 125) ** 2)
    lines, events = [], []
    for event_id in range(128):
        phi = rng.uniform(0, 2 * math.pi)
        velocity = [speed * math.sin(math.pi / 180) * math.cos(phi),
                    speed * math.sin(math.pi / 180) * math.sin(phi),
                    speed * math.cos(math.pi / 180)]
        particles = []
        for sign, role, pdg in [(1, "electron", 11), (-1, "positron", -11)]:
            lines.append(f"240 {sign * 125} 0 0 0 " + " ".join(f"{sign*x:.17g}" for x in velocity) + "\n")
            particles.append(dict(source_record=len(lines), role=role, event_slot=role, pdg_id=pdg))
        events.append(dict(event_id=event_id, particles=particles))
    event_path = destination / "event.ini"
    event_path.write_text("".join(lines), encoding="ascii", newline="\r\n")
    identity = dict(
        schema_version="feebl.collider.manufactured-events/1", coordinate_frame="native_pair_tracking",
        source_artifact=dict(path="event.ini", bytes=event_path.stat().st_size,
                             sha256=hashlib.sha256(event_path.read_bytes()).hexdigest()),
        events=events,
        generation=dict(kind="manufactured back-to-back transport fixture; no cross-section claim",
                        events=128, tracks=256, energy_GeV_per_particle=125, electron_mass_GeV=.00051099895,
                        electron_theta_from_positive_z_deg=1, positron_theta_from_positive_z_deg=179,
                        azimuth_seed=6102701, azimuth_rng="Python random.Random, MT19937",
                        azimuth="uniform [0,2pi), opposite vectors", vertex_nm=[0,0,0],
                        injection_diagonal=15, injection_time_counter=240, n_z=16, n_t=2,
                        native_seed=6102711, selection="all records",
                        signed_energy_convention="native +125 electron; -125 positron"))
    (destination / "event_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8", newline="\r\n")
    return identity["source_artifact"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(generate(parser.parse_args().output), indent=2))

