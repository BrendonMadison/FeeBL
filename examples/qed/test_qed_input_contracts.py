"""Check QED mode admission using the actual native executable.

Invalid cards must fail with their expected diagnostic before simulation.
A small legacy-compatible control verifies that the test fixture itself is
accepted. Artifacts are retained in a distinct directory for every invocation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid


def set_parameter(card, name, value):
    """Change exactly one scoped assignment, or add a newly introduced option."""
    opening = card.index("{", card.index("$PARAMETERS::"))
    closing = card.index("}", opening)
    body = card[opening + 1:closing]
    pattern = re.compile(r"(?<![\w.])" + re.escape(name) + r"\s*=\s*[^;]+;")
    if len(pattern.findall(body)) > 1:
        raise ValueError("Ambiguous card assignment: " + name)
    replacement = f"{name}={value};"
    body = pattern.sub(replacement, body) if pattern.search(body) else body + "\n" + replacement + "\n"
    return card[:opening + 1] + body + card[closing:]


def checks():
    """Return finite negative tests and the exact error text they must expose."""
    result = []
    for mode in (1, 3):
        result.append((f"timing{mode}_legacy_coherent",
                       {"photon_timing": mode, "do_coherent": 1, "coherent_model": 0},
                       "requires coherent_model = 1"))
    result.append(("polarisation_legacy_pair",
                   {"photon_polarisation": 1, "pair_model": 0},
                   "requires pair_model = 1 or 2"))
    result.append(("polarisation_legacy_coherent",
                   {"photon_polarisation": 1, "pair_model": 1,
                    "do_coherent": 1, "coherent_model": 0},
                   "requires coherent_model = 1"))
    for name, maximum in (("pair_model", 2), ("photon_timing", 3),
                          ("photon_angle", 2), ("photon_polarisation", 1),
                          ("coherent_model", 1)):
        for value in (-1, maximum + 1):
            result.append((f"invalid_{name}_{value}", {name: value}, name + " must be"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feebl", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.feebl.resolve()
    base = args.card.read_text(encoding="utf-8")
    # Keep the native fixture tiny even if its packaged display settings grow.
    common = {"do_photons": 1, "do_eloss": 1, "do_coherent": 0,
              "pair_model": 0, "photon_timing": 0, "photon_angle": 0,
              "photon_polarisation": 0, "coherent_model": 0,
              "store_native": 0, "store_photons": 0}
    for name, value in common.items():
        base = set_parameter(base, name, value)
    output = args.output.resolve() / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    results = []
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for name, changes, expected in [("valid_zero_modes", {}, None), *checks()]:
        directory = output / name
        directory.mkdir()
        card = base
        for key, value in changes.items():
            card = set_parameter(card, key, value)
        (directory / "acc.dat").write_text(card, encoding="utf-8", newline="\n")
        started = time.monotonic()
        completed = subprocess.run([str(binary), "ilc500", "diagnostic", "out.dat"],
                                   cwd=directory, env=environment, capture_output=True,
                                   text=True, timeout=10)
        (directory / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (directory / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        diagnostic = completed.stdout + completed.stderr
        passed = completed.returncode == 0 if expected is None else (
            completed.returncode != 0 and expected in diagnostic)
        row = {"name": name, "returncode": completed.returncode,
               "expected_diagnostic": expected, "passed": passed,
               "elapsed_seconds": time.monotonic() - started,
               "card_sha256": hashlib.sha256(card.encode()).hexdigest()}
        results.append(row)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        if not passed:
            raise RuntimeError(f"Native QED admission failed: {name}; retained outputs in {directory}")
    print(f"PASS: {len(results)} native QED input contracts; artifacts: {output}")


if __name__ == "__main__":
    main()
