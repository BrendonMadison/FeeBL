"""Strict, declared LHEF 3 subset and an explicit vertex/transport sidecar.

This is an interoperability scaffold, NOT electromagnetic transport.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .common import finite, integer, keys, require, sha256


def _number(token):
    return finite(float(token.replace("D", "E").replace("d", "e")), "LHE real")


def _integer(token):
    value = int(token)
    require(
        str(value) == token.lstrip("+") or token.lstrip("+-").isdigit(), "Invalid integer token"
    )
    return value


def _lines(text):
    return [
        line.split("#", 1)[0].strip()
        for line in (text or "").splitlines()
        if line.split("#", 1)[0].strip()
    ]


def parse(path):
    require(
        Path(path).stat().st_size <= 256 * 1024 * 1024,
        "LHE XML exceeds the scaffold's 256 MiB limit",
    )
    source = Path(path).read_bytes()
    require(
        b"<!DOCTYPE" not in source.upper() and b"<!ENTITY" not in source.upper(),
        "DTD/entity declarations are not accepted",
    )
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(source, parser=parser)
    require(
        root.tag == "LesHouchesEvents" and root.attrib.get("version") in {"3.0", "3.1"},
        "Supported strict subset requires unnamespaced LHEF 3.0 or 3.1",
    )
    inits = root.findall("init")
    require(len(inits) == 1, "Exactly one init block is required")
    lines = _lines(inits[0].text)
    require(lines and len(lines[0].split()) == 10, "Malformed init beam header")
    header = lines[0].split()
    for index in (0, 1, 4, 5, 6, 7, 8, 9):
        _integer(header[index])
    require(
        _number(header[2]) > 0 and _number(header[3]) > 0, "Two positive beam energies are required"
    )
    nprocess = _integer(header[9])
    require(nprocess >= 1 and len(lines) == nprocess + 1, "NPRUP/process record mismatch")
    process_ids = set()
    for line in lines[1:]:
        fields = line.split()
        require(len(fields) == 4, "Malformed process record")
        for field in fields[:3]:
            _number(field)
        process_id = _integer(fields[3])
        require(process_id not in process_ids, "Duplicate process ID")
        process_ids.add(process_id)
    # eventgroup/compact alternative encodings are deliberately not silently skipped.
    require(
        not root.findall("eventgroup"), "eventgroup format is not supported by this strict subset"
    )
    events = []
    for event_id, element in enumerate(root.findall("event"), 1):
        lines = _lines(element.text)
        require(lines and len(lines[0].split()) == 6, "Malformed event header")
        header = lines[0].split()
        nup = _integer(header[0])
        process_id = _integer(header[1])
        require(process_id in process_ids, "Event refers to an undefined process ID")
        require(
            nup >= 3 and len(lines) == nup + 1,
            "NUP does not match particle records (this collider subset requires two incoming plus one or more outgoing)",
        )
        event_weight = _number(header[2])
        for value in header[3:]:
            _number(value)
        particles = []
        for particle_id, line in enumerate(lines[1:], 1):
            fields = line.split()
            require(len(fields) == 13, f"Particle {particle_id}: expected all 13 HEPEUP fields")
            ints = [_integer(x) for x in fields[:6]]
            reals = [_number(x) for x in fields[6:]]
            require(ints[0] != 0 and ints[1] != 0, "Zero PDG or status is not accepted")
            require(
                all(0 <= x <= nup and x != particle_id for x in ints[2:4]),
                "Mother reference outside event or self-parent",
            )
            require(ints[2] <= ints[3], "Mother range is reversed")
            require((ints[2] == 0) == (ints[3] == 0), "Incomplete zero mother range")
            require(all(x >= 0 for x in ints[4:6]), "Negative color index")
            require(reals[3] >= 0, "Negative particle energy")
            particles.append(
                {
                    "particle_id": particle_id,
                    "pdg": ints[0],
                    "status": ints[1],
                    "mothers": ints[2:4],
                    "colors": ints[4:6],
                    "px": reals[0],
                    "py": reals[1],
                    "pz": reals[2],
                    "energy": reals[3],
                    "mass": reals[4],
                    "lifetime": reals[5],
                    "spin": reals[6],
                }
            )
        incoming = [p for p in particles if p["status"] == -1]
        outgoing = [p for p in particles if p["status"] == 1]

        def visit(particle_id, stack, visited):
            require(particle_id not in stack, "Cyclic mother references")
            if particle_id in visited:
                return
            particle = particles[particle_id - 1]
            first, last = particle["mothers"]
            if first:
                for mother in range(first, last + 1):
                    visit(mother, stack | {particle_id}, visited)
            visited.add(particle_id)

        visited = set()
        for particle_id in range(1, nup + 1):
            visit(particle_id, set(), visited)
        require(
            len(incoming) == 2 and len(outgoing) >= 1,
            "Collider LHE subset requires exactly two incoming and at least one final-state particle; no incoming particles will be fabricated",
        )
        require(
            sorted(p["pdg"] for p in incoming) == [-11, 11],
            "This FeeBL scaffold expects an electron/positron initial state",
        )
        # Record, but do not require, conservation: beam-field transport exchanges
        # four-momentum with an external field and must document that in a sidecar.
        residual = [
            sum(p[key] for p in incoming) - sum(p[key] for p in outgoing)
            for key in ("px", "py", "pz", "energy")
        ]
        rwgt_ids = []
        for weight in element.findall("rwgt/wgt"):
            require(weight.attrib.get("id"), "Reweight ID is missing")
            require(weight.attrib["id"] not in rwgt_ids, "Duplicate event reweight ID")
            rwgt_ids.append(weight.attrib["id"])
            _number((weight.text or "").strip())
        events.append(
            {
                "event_id": event_id,
                "weight": event_weight,
                "particles": particles,
                "n_incoming": len(incoming),
                "n_outgoing": len(outgoing),
                "four_momentum_residual_GeV": residual,
                "reweight_ids": rwgt_ids,
            }
        )
    require(events, "No directly encoded event elements")
    return root, {
        "schema": "feebl.lhe_validation.v1",
        "status": "STRUCTURAL_SUBSET_PASS_NOT_TRANSPORT_VALIDATION",
        "lhe_version": root.attrib["version"],
        "sha256": sha256(path),
        "events": events,
        "limits": "Direct text HEPEUP/init records only, two electron/positron incoming particles, variable final-state multiplicity, semantic XML extension preservation; no vertex or beam-field propagation is inferred.",
    }


def validate_sidecar(sidecar, lhe_result):
    keys(sidecar, ("schema", "lhe_sha256", "frame", "units", "transport", "particles"))
    require(sidecar["schema"] == "feebl.transport_sidecar.v1", "Unknown transport sidecar")
    require(sidecar["lhe_sha256"] == lhe_result["sha256"], "Sidecar belongs to different LHE bytes")
    require(
        sidecar["frame"] == "lab"
        and sidecar["units"] == {"momentum": "GeV", "position": "mm", "time": "ns"},
        "Explicit lab GeV/mm/ns conventions are required",
    )
    require(
        sidecar["transport"] in {"not_applied", "external_beam_fields_applied"},
        "Transport application state is missing",
    )
    expected = {
        (e["event_id"], p["particle_id"])
        for e in lhe_result["events"]
        for p in e["particles"]
        if p["status"] == 1
    }
    seen = set()
    for particle in sidecar["particles"]:
        keys(
            particle,
            (
                "event_id",
                "particle_id",
                "x_mm",
                "y_mm",
                "z_mm",
                "time_ns",
                "weight",
                "observation_plane",
                "theta_lab_rad",
                "theta_out_rad",
                "phi_rad",
                "hemisphere_definition",
            ),
        )
        key = (particle["event_id"], particle["particle_id"])
        require(
            key in expected and key not in seen, "Duplicate or nonexistent final-state sidecar ID"
        )
        seen.add(key)
        for name in (
            "x_mm",
            "y_mm",
            "z_mm",
            "time_ns",
            "weight",
            "theta_lab_rad",
            "theta_out_rad",
            "phi_rad",
        ):
            finite(particle[name], name)
        require(
            0 <= particle["theta_lab_rad"] <= math.pi
            and 0 <= particle["theta_out_rad"] <= math.pi / 2,
            "Angle convention outside range",
        )
        require(
            particle["observation_plane"] and particle["hemisphere_definition"],
            "Observation and hemisphere definitions required",
        )
        event = lhe_result["events"][particle["event_id"] - 1]
        require(particle["weight"] == event["weight"], "Sidecar must preserve the event weight")
        momentum = event["particles"][particle["particle_id"] - 1]
        pt = math.hypot(momentum["px"], momentum["py"])
        require(
            abs(particle["theta_lab_rad"] - math.atan2(pt, momentum["pz"])) <= 1e-10,
            "theta_lab disagrees with LHE momentum",
        )
        require(
            abs(particle["theta_out_rad"] - math.atan2(pt, abs(momentum["pz"]))) <= 1e-10,
            "theta_out disagrees with folded outgoing-arm convention",
        )
        if pt > 0:
            require(
                abs(
                    math.remainder(
                        particle["phi_rad"] - math.atan2(momentum["py"], momentum["px"]),
                        2 * math.pi,
                    )
                )
                <= 1e-10,
                "phi disagrees with LHE momentum",
            )
    require(seen == expected, "Sidecar must cover every transported final-state particle")
    return {"status": "SIDECAR_STRUCTURAL_PASS_NOT_TRANSPORT_VALIDATION", "particles": len(seen)}


def roundtrip(source, destination):
    tree, before = parse(source)
    require(not Path(destination).exists(), "Refusing to replace an existing LHE file")
    with open(destination, "xb") as stream:
        ET.ElementTree(tree).write(stream, encoding="utf-8", xml_declaration=True)
    _, after = parse(destination)
    require(before["events"] == after["events"], "Roundtrip changed HEPEUP records or weights")
    require(
        ET.tostring(tree)
        == ET.tostring(
            ET.parse(
                destination, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
            ).getroot()
        ),
        "Roundtrip changed XML extensions",
    )
    return {
        "status": "SEMANTIC_ROUNDTRIP_PASS",
        "input_sha256": before["sha256"],
        "output_sha256": after["sha256"],
        "events": len(after["events"]),
        "note": "Not byte-identical; formatting may change. No physics or standards-complete interoperability claim.",
    }
