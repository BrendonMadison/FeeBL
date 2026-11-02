"""Check the workspace cap against the actual executable using tiny cards.

Each process is bounded to 120 seconds. The output directory must be new, so
all accepted and rejected cards retain independent execution evidence.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def scientific_files(directory):
    """Return artifacts after removing only declared execution provenance."""
    excluded = {"acc.dat", "stdout.txt", "stderr.txt", "receipt.json"}
    result = {}
    for path in directory.iterdir():
        if not path.is_file() or path.name in excluded:
            continue
        content = path.read_bytes()
        if path.name == "result.out":
            content = b"\n".join(
                line for line in content.splitlines()
                if not line.startswith(b"wavefront_workspace_max_bytes = ")
            )
        result[path.name] = hashlib.sha256(content).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feebl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    binary = args.feebl.resolve(strict=True)
    template = (args.source_root / "examples/legacy_wavefront_scheduler/stateful_tiny.dat").read_text()
    backend = ('compute_backend="cpu";cpu_threads=2;cpu_variant="pool_dual";'
               'reduction_mode="strict";allow_backend_fallback=0;sparse_cells=1;'
               'collision_schedule="wavefront_parallel";')
    template = template.replace('collision_schedule="legacy";', backend)
    assert template != (args.source_root / "examples/legacy_wavefront_scheduler/stateful_tiny.dat").read_text()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("automatic", "0.0", True, None),
        ("explicit_64gib", "68719476736", True, None),
        ("largest_exact_integer", "9007199254740991", True, None),
        ("negative", "-1.0", False, "wavefront_workspace_max_bytes"),
        ("fractional", "4096.5", False, "wavefront_workspace_max_bytes"),
        ("beyond_exact_integer", "9007199254740992", False, "wavefront_workspace_max_bytes"),
        ("nonfinite_literal", "1e999", False, None),
        ("insufficient", "1", False, "workspace"),
        ("unused_legacy", "68719476736", False, "wavefront_workspace_max_bytes"),
    ]
    records = []
    accepted_artifacts = []
    for name, value, accept, diagnostic in cases:
        directory = args.output / name
        directory.mkdir()
        card = template
        if name == "unused_legacy":
            card = card.replace('collision_schedule="wavefront_parallel";',
                                'collision_schedule="legacy";')
        position = card.rfind("}")
        card = (card[:position] + f"wavefront_workspace_max_bytes={value};\n" + card[position:])
        (directory / "acc.dat").write_text(card, encoding="utf-8")
        started = time.perf_counter()
        result = subprocess.run(
            [str(binary), "tiny", "exact", "result.out"], cwd=directory,
            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
            capture_output=True, timeout=120,
        )
        (directory / "stdout.txt").write_bytes(result.stdout)
        (directory / "stderr.txt").write_bytes(result.stderr)
        receipt = {"case": name, "value": value, "returncode": result.returncode,
                   "wall_seconds": time.perf_counter() - started,
                   "expected_accept": accept,
                   "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        logs = (result.stdout + result.stderr).decode(errors="replace")
        output = directory / "result.out"
        text = output.read_text(errors="replace") if output.exists() else ""
        if accept:
            assert result.returncode == 0, (name, logs[-2000:])
            assert re.search(r"lumi_ee\s*=", text), (name, "no completed luminosity")
            if value != "0.0":
                assert f"wavefront_workspace_max_bytes = {value}" in text
            accepted_artifacts.append(scientific_files(directory))
        else:
            assert result.returncode != 0, (name, "accepted invalid budget")
            assert not re.search(r"lumi_ee\s*=", text), (name, "published physics")
            if diagnostic:
                assert diagnostic in logs, (name, logs[-2000:])
        records.append(receipt)
    assert all(result == accepted_artifacts[0] for result in accepted_artifacts[1:]), \
        "Changing a nonbinding budget changed a scientific artifact"
    (args.output / "checks.json").write_text(json.dumps(records, indent=2) + "\n")
    print("PASS: workspace budget admission, bounded allocation and nonbinding-cap equivalence")


if __name__ == "__main__":
    main()
