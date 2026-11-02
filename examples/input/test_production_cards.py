"""Exercise input admission using the actual production executable.

Only a 32 x 32 mesh and 1,000 macroparticles per beam are used. Outputs are
retained under --output, including failed cards; no fixture is silently removed.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def execute(binary, root, label, card):
    """Run one card with a 30 s bound and retain its process evidence."""
    directory = root / label
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "acc.dat").write_bytes(card)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(
        [str(binary), "ilc500", "diagnostic", "out.dat"],
        cwd=directory, env=env, capture_output=True, timeout=30,
    )
    (directory / "stdout.txt").write_bytes(result.stdout)
    (directory / "stderr.txt").write_bytes(result.stderr)
    summary = directory / "out.dat"
    output = summary.read_text(errors="replace") if summary.exists() else ""
    return result.returncode, output, (result.stdout + result.stderr).decode(errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feebl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    card = (source / "benchmarks/cases/smoke/acc.dat").read_text()
    # The default must use the seed even when a stale checkpoint is present.
    card = card.replace("rndm_load=0;rndm_save=0;", "")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("seed_first", "seed_repeat"):
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        (directory / "rndm.save").write_bytes(b"deliberately stale checkpoint")
    checks = []
    outputs = []
    for name, line_endings in (("seed_first", "lf"), ("seed_repeat", "lf"),
                               ("crlf", "crlf"), ("no_final_newline", "none")):
        variant = card.replace("\n", "\r\n") if line_endings == "crlf" else card
        if line_endings == "none":
            variant = variant.rstrip("\n")
        code, output, diagnostics = execute(args.feebl.resolve(), args.output, name, variant.encode())
        if code != 0:
            raise AssertionError(f"{name}: exit {code}: {diagnostics[-1000:]}")
        luminosity = re.search(r"lumi_ee\s*=\s*(\S+)", output)
        if not luminosity:
            raise AssertionError(f"{name}: no luminosity in successful output")
        outputs.append(luminosity.group(1))
        checks.append({"case": name, "returncode": code, "luminosity": outputs[-1]})
    if len(set(outputs)) != 1:
        raise AssertionError(f"seed or line-ending replay differs: {outputs}")
    invalid = {
        "overflow_literal": "ip_dispersion.1=1e999;",
        "overflow_expression": "ip_dispersion.1=1e300*1e300;",
        "zero_divisor": "ip_dispersion.1=1.0/0.0;",
        "integer_overflow": "n_x=999999999999999999999999;",
        "integer_narrowing": "n_x=1e50;",
        "integer_fraction": "n_x=3.5;",
        "missing_checkpoint": "rndm_load=1;",
        "wrong_dispersion_key": "ip_dispersion_m.1=0.001;",
        "unsupported_imported_spin_flip": "imported_spin_mode=2;",
        "imported_spin_without_events": "imported_spin_mode=1;",
    }
    for name, assignment in invalid.items():
        closing = card.rfind("}")
        variant = card[:closing] + assignment + card[closing:]
        code, output, diagnostics = execute(args.feebl.resolve(), args.output, name, variant.encode())
        if code == 0 or re.search(r"lumi_ee\s*=", output):
            raise AssertionError(f"{name}: invalid input published success")
        if not diagnostics.strip():
            raise AssertionError(f"{name}: rejected without a diagnostic")
        checks.append({"case": name, "returncode": code, "rejected": True})
    (args.output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"PASS: {len(checks)} production input, checkpoint-default and line-ending checks")


if __name__ == "__main__":
    main()
