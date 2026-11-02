#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_backends", ROOT / "tools" / "benchmark_backends.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def main() -> int:
    variant = MODULE.parse_variant("cpu4:FEEBL_BACKEND=cpu,FEEBL_THREADS=4")
    assert variant.name == "cpu4"
    assert variant.environment == {"FEEBL_BACKEND": "cpu", "FEEBL_THREADS": "4"}
    assert variant.deck_fragment == ""

    name, fragment = MODULE.parse_deck_fragment(
        'cpu4=compute_backend="cpu";cpu_threads=4'
    )
    assert name == "cpu4"
    assert fragment == 'compute_backend="cpu";cpu_threads=4;'

    source = (
        b'$ACCELERATOR:: c1\r\n{x=1;}\r\n'
        b'$PARAMETERS:: p_c1\r\n{n_m=10;}\r\n'
        b'$PARAMETERS:: p_c2\r\n{n_m=20;}\r\n'
    )
    rendered = MODULE.materialize_deck(source, "p_c1", fragment)
    assert source == (
        b'$ACCELERATOR:: c1\r\n{x=1;}\r\n'
        b'$PARAMETERS:: p_c1\r\n{n_m=10;}\r\n'
        b'$PARAMETERS:: p_c2\r\n{n_m=20;}\r\n'
    )
    assert b'{\r\ncompute_backend="cpu";cpu_threads=4;n_m=10;}' in rendered
    assert b'$PARAMETERS:: p_c2\r\n{n_m=20;}' in rendered
    assert MODULE.materialize_deck(source, "p_c1", "") is source
    assert hashlib.sha256(source).digest() != hashlib.sha256(rendered).digest()

    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "deck.dat"
        source_path.write_bytes(source)
        before = hashlib.sha256(source_path.read_bytes()).hexdigest()
        _ = MODULE.materialize_deck(source_path.read_bytes(), "p_c1", fragment)
        after = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert before == after

        binary = Path(directory) / "fake-feebl"
        binary.write_text(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            "grep -q 'compute_backend=\"cpu\"' acc.dat\n"
            "printf 'lumi[0][1] = 42\\nlumi_fine = 7\\nupsmax = 0.5\\n' > \"$3\"\n"
        )
        binary.chmod(0o755)
        output = Path(directory) / "benchmark"
        assert MODULE.main([
            "--binary", str(binary),
            "--source-root", str(ROOT),
            "--deck", str(source_path),
            "--case", "c1",
            "--variant", "cpu4",
            "--deck-fragment", 'cpu4=compute_backend="cpu";cpu_threads=4;',
            "--baseline", "cpu4",
            "--warmups", "0",
            "--repetitions", "1",
            "--output", str(output),
        ]) == 0
        manifest = json.loads((output / "manifest.json").read_text())
        sample = json.loads((output / "samples.jsonl").read_text())
        run_deck = Path(sample["run_directory"]) / "acc.dat"
        assert hashlib.sha256(run_deck.read_bytes()).hexdigest() == sample[
            "input_deck_sha256"
        ]
        assert manifest["materialized_deck_sha256"]["c1"]["cpu4"] == sample[
            "input_deck_sha256"
        ]
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before

    normalized = MODULE.normalize_output(
        "* version 1\nphysics=42\nelapsed time 3\n",
        [__import__("re").compile(pattern) for pattern in MODULE.DEFAULT_NORMALIZE],
    )
    assert normalized == "physics=42\n"

    values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98]
    low, high = MODULE.bootstrap_interval(values, statistics.median, 7, draws=1000)
    assert low <= statistics.median(values) <= high
    assert math.isclose(MODULE.median_absolute_deviation(values), 0.05,
                        rel_tol=0.0, abs_tol=1e-12)
    assert MODULE.count_processes_named("a-process-name-that-cannot-exist") == 0
    print("benchmark harness unit checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
