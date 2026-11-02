"""Generate the native card keyword reference from its C++ registry.

Types and defaults come from src/readData.cc. Units and short interpretation
notes are maintained here at the input boundary. --check detects stale output.
The unsupported TWOBEAM compile-time branch is excluded from this reference.
"""

import argparse
import ast
import json
from pathlib import Path
import re

from input_reference_metadata import metadata

ROOT = Path(__file__).resolve().parents[1]
UNITS = {
    "particles": "10^10 particles per bunch", "energy": "GeV",
    "f_rep": "Hz", "n_b": "bunch crossings per train",
    "sigma_x": "nm", "sigma_y": "nm", "sigma_z": "um",
    "emitt_x": "mm mrad (normalized)", "emitt_y": "mm mrad (normalized)",
    "beta_x": "mm", "beta_y": "mm", "offset_x": "nm", "offset_y": "nm",
    "offset_z": "um", "waist_x": "um", "waist_y": "um",
    "angle_phi": "rad", "angle_x": "rad", "angle_y": "rad",
    "cut_x": "nm", "cut_y": "nm", "cut_z": "um",
    "crab_frequency": "Hz", "crab_phase": "rad", "crab_target_tilt": "rad",
    "ip_dispersion": "m", "crab_full_crossing_angle": "rad",
    "adaptive_grid_anchor_x": "nm", "adaptive_grid_anchor_y": "nm",
    "adaptive_grid_source_width_x": "nm", "adaptive_grid_source_width_y": "nm",
    "adaptive_grid_error_reference_field": "native electric-equivalent field units (GV/nm)",
    "adaptive_grid_field_floor": "native electric-equivalent field units (GV/nm)",
    "adaptive_grid_rho_floor": "native deposited charge weight per grid cell",
    "adaptive_grid_lumi_floor": "native overlap normalization; see adaptive-grid contract",
    "electron_lumi_spatial_x_min": "nm", "electron_lumi_spatial_x_max": "nm",
    "electron_lumi_spatial_y_min": "nm", "electron_lumi_spatial_y_max": "nm",
    "lumi_p": "m^2 (luminosity-event sampling scale)",
    "lumi_p_eg": "m^2 (luminosity-event sampling scale)",
    "lumi_p_gg": "m^2 (luminosity-event sampling scale)",
    "bhabha_scal": "m^2 (event sampling scale)", "prod_scal": "m^2 (event sampling scale)",
}
for _key in ("ecm_min", "ecm_min_gg", "jet_ptmin", "emin", "bhabha_ecmload",
             "pair_ecut", "muon_ecut", "prod_e", "hist_ee_min", "hist_ee_max",
             "hist_espec_min", "hist_espec_max", "compt_emax", "spectrum_w_min", "spectrum_w_max"):
    UNITS[_key] = "GeV"

def strip_comments(source):
    """Remove C++ comments while retaining strings and source line numbers."""
    pattern = r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*'
    return re.sub(pattern, lambda match: match[0] if match[0].startswith('"')
                  else "\n" * match[0].count("\n"), source, flags=re.S)


def registry():
    """Read active declarations and literal defaults; fail on unknown grammar."""
    source = strip_comments((ROOT / "src/readData.cc").read_text(encoding="utf-8"))
    source = re.sub(r"#ifdef TWOBEAM\b.*?#endif",
                    lambda match: "\n" * match[0].count("\n"), source, flags=re.S)
    entries = {}
    for match in re.finditer(r'def_named_variable\("([^"]+)",\s*(T_\w+)\)', source):
        name, native_type = match.groups()
        entries[name] = {
            "name": name, "native_type": native_type,
            "section": "ACCELERATOR" if match.start() < source.index("void def_param(") else "PARAMETERS",
            "default": "0 (zero-initialized)" if native_type != "T_STRING" else "unset",
            "registry_source": "src/readData.cc",
            "registry_line": source.count("\n", 0, match.start()) + 1,
        }
    for match in re.finditer(r'define_values\(("(?:\\.|[^"\\])*")\s*,\s*m_account\)', source):
        assignments = ast.literal_eval(match[1]).strip("{}")
        for assignment in assignments.split(";"):
            if not assignment.strip():
                continue
            name, value = [part.strip() for part in assignment.split("=", 1)]
            if name not in entries:
                raise ValueError(f"Default for undeclared key {name}")
            entries[name]["default"] = value
    descriptions = metadata(ROOT)
    if set(entries) != set(descriptions):
        raise ValueError("Input metadata mismatch: " + repr({
            "missing": sorted(set(entries) - set(descriptions)),
            "unregistered": sorted(set(descriptions) - set(entries))}))
    for name, entry in entries.items():
        native_type = entry["native_type"]
        entry["type"] = "string" if native_type == "T_STRING" else (
            "integer" if "INT" in native_type else "real")
        entry["suffix"] = "mirrored .1/.2" if "MIRROR" in native_type else (
            ".1/.2" if native_type.endswith("_2") else "none")
        entry["unit"] = UNITS.get(name, "dimensionless")
        if native_type == "T_STRING":
            entry["unit"] = "text (selector, list or path)"
        elif "bytes" in name:
            entry["unit"] = "bytes"
        description = descriptions[name]
        if description["unit"]:
            entry["unit"] = description["unit"]
        entry["meaning"] = description["meaning"]
        entry["conditions"] = description["conditions"]
        entry["notes"] = description["meaning"] + " " + description["conditions"]
        entry["semantic_sources"] = description["sources"]
        if not entry["meaning"].strip() or not entry["conditions"].strip():
            raise ValueError(f"Empty input description: {name}")
        for relative in entry["semantic_sources"]:
            if not (ROOT / relative).is_file():
                raise ValueError(f"Missing input source for {name}: {relative}")
    return sorted(entries.values(), key=lambda item: (item["section"], item["name"]))


def products():
    rows = registry()
    header = """# Native input reference

Generated by `python tools/generate_input_reference.py` from the active native
registry. `python tools/generate_input_reference.py --check` detects drift.
Units in this table are card units, not native storage or viewer display units.
Defaults do not certify a physically valid collider; start from a packaged card.
Every registered key has an explicit description and implementation/contract
source. The companion `input_reference.json` retains all source paths and the
registry declaration line. Missing metadata makes generation fail.

The value column states implemented selectors and conditions, not an assurance
that every combination is supported. Many legacy flags test nonzero values;
use the stated 0/1 convention rather than assuming every other integer is
rejected. Experimental adaptive and GPU options have stricter coupled admission
rules. Numerical tolerances are declared targets, not achieved accuracy.

Cards contain `$ACCELERATOR:: name { ... }` followed by
`$PARAMETERS:: name { ... }`. Assignments end in semicolons; expressions support
`+`, `-`, `*`, `/`, parentheses and previously declared keys. Integer division
truncates toward zero. Overflow, nonfinite expressions, fractional integer
assignments and unsupported keywords are errors. LF and CRLF are accepted.
Do not concatenate tokens across physical lines. Blocks are bounded by the
native parser's 10,000-byte buffer; physical lines must be shorter than 511 bytes.

For `.1/.2` keys, an unsuffixed assignment sets both beams; later suffixed
assignments override one beam. A mirrored unsuffixed value sets beam 1 to `v`
and beam 2 to `-v`; explicit suffixes use the supplied sign. Scalar and string
keys have no beam suffix. The unsupported TWOBEAM build branch is excluded.

`rndm_load=0;rndm_save=0` is now the default. Explicit checkpoint switches control saving and restoring the random stream.
Coordinate and execution conventions are described in the
[manual](FeeBL_Manual_Paper.pdf) and the linked implementation sources.

"""
    sections = []
    for section in ("ACCELERATOR", "PARAMETERS"):
        sections.append(f"## {section}\n\n| Keyword | Type | Beam suffix | Card unit | Default | Meaning and accepted values | Sources |\n|---|---|---|---|---|---|---|\n")
        for row in rows:
            if row["section"] == section:
                links = [f'[declaration](../{row["registry_source"]}#L{row["registry_line"]})']
                links.extend(f'[{Path(path).stem}](../{path})' for path in row["semantic_sources"])
                values = (f'`{row["name"]}`', row["type"], row["suffix"], row["unit"],
                          f'`{row["default"]}`', row["notes"], "; ".join(links))
                sections.append("| " + " | ".join(v.replace("|", "\\|") for v in values) + " |\n")
        sections.append("\n")
    return {ROOT / "docs/INPUT_REFERENCE.md": header + "".join(sections),
            ROOT / "docs/input_reference.json": json.dumps(rows, indent=2) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, content in products().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Stale generated reference: {path.relative_to(ROOT)}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    print(f"{'Checked' if args.check else 'Generated'} {len(registry())} input keys")


if __name__ == "__main__":
    main()
