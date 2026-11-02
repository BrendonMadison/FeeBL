#!/usr/bin/env python3
"""Generate polarized, back-to-back Bhabha-like transport probes.

Each species samples a longitudinal pure-state helicity of +1 or -1 with mean
P supplied by the corresponding beam. Spin is aligned with the actual outgoing
momentum. This is a controlled transport fixture, not a polarized hard-scattering
generator: cross sections and the Bhabha spin density matrix are not modeled.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def generate(output, events, energy_gev, theta_deg, polarizations, seed, counter):
    """Write eleven-column native records and retain every initial spin label.

    Coordinates are nm, signed energy is GeV (+ electron, - positron), velocity
    is beta, and spin is a unit expectation vector in the tracking frame.
    `counter` is the native admission counter, not a physical time in seconds.
    The same generated file must be used in field-on and field-off controls.
    """
    electron_mass_gev = 0.00051099906
    if (events < 1 or not math.isfinite(energy_gev) or
            energy_gev <= electron_mass_gev or not math.isfinite(theta_deg) or
            not 0 < theta_deg < 90 or counter < 0):
        raise ValueError('Invalid event count, energy or forward polar angle.')
    if any(not math.isfinite(p) or abs(p) > 1 for p in polarizations):
        raise ValueError('Nominal polarizations must lie in [-1,1].')
    output.mkdir(parents=True, exist_ok=False)
    azimuth_rng = random.Random(seed)
    spin_rng = random.Random(seed + 1)
    beta = math.sqrt(1 - (electron_mass_gev / energy_gev) ** 2)
    theta = math.radians(theta_deg)
    records, identities = [], []
    sum_helicity = [0, 0]
    for event_id in range(events):
        phi = azimuth_rng.uniform(0, 2 * math.pi)
        direction = (math.sin(theta) * math.cos(phi),
                     math.sin(theta) * math.sin(phi), math.cos(theta))
        tracks = []
        for species, (sign, pdg, name) in enumerate(((1, 11, 'electron'), (-1, -11, 'positron'))):
            helicity = 1 if spin_rng.random() < (1 + polarizations[species]) / 2 else -1
            velocity = [sign * beta * component for component in direction]
            spin = [sign * helicity * component for component in direction]
            row = [counter, sign * energy_gev, 0, 0, 0, *velocity, *spin]
            records.append(' '.join(format(value, '.17g') for value in row) + '\n')
            tracks.append(dict(source_record=len(records), pdg_id=pdg, role=name,
                               initial_helicity=helicity, spin=spin, beta=velocity))
            sum_helicity[species] += helicity
        identities.append(dict(event_id=event_id, particles=tracks))
    source = output / 'event.ini'
    source.write_text(''.join(records), encoding='ascii', newline='\n')
    receipt = dict(
        schema='feebl.polarized-bhabha-input/1', events=events, tracks=2 * events,
        energy_GeV=energy_gev, electron_mass_GeV=electron_mass_gev, electron_theta_deg=theta_deg,
        nominal_helicity=dict(electron=polarizations[0], positron=polarizations[1]),
        realized_helicity=dict(electron=sum_helicity[0] / events, positron=sum_helicity[1] / events),
        sampling='independent Bernoulli helicities; p(+)=(1+P)/2',
        azimuth_seed=seed, helicity_seed=seed + 1, rng='Python random.Random MT19937',
        vertex_nm=[0, 0, 0], admission_counter=counter,
        spin_definition='unit vector parallel/antiparallel to each outgoing momentum',
        physics_scope='central-vertex transport fixture; no hard-scattering spin density matrix or cross section',
        source_artifact=dict(path='event.ini', sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        event_records=identities)
    (output / 'polarized_event_identity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return {key: value for key, value in receipt.items() if key != 'event_records'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--events', type=int, default=8192)
    parser.add_argument('--energy-gev', type=float, default=125)
    parser.add_argument('--theta-deg', type=float, default=1)
    parser.add_argument('--electron-polarization', type=float, default=0.8)
    parser.add_argument('--positron-polarization', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=7292917)
    parser.add_argument('--admission-counter', type=int, default=240)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.events, args.energy_gev, args.theta_deg,
                              (args.electron_polarization, args.positron_polarization),
                              args.seed, args.admission_counter), indent=2))


if __name__ == '__main__':
    main()
