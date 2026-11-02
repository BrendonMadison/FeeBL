#!/usr/bin/env python3

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "render_parallel_results", ROOT / "tools" / "render_parallel_results.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cpu = root / "cpu.csv"
        gpu = root / "gpu.csv"
        output = root / "rendered"
        write_csv(cpu, [
            "case", "variant", "samples", "median_seconds", "mad_seconds",
            "median_ci95_low", "median_ci95_high", "speedup_vs_baseline",
            "speedup_ci95_low", "speedup_ci95_high",
        ], [{
            "case": "c1", "variant": "cpu4", "samples": 7,
            "median_seconds": 1.0, "mad_seconds": 0.1,
            "median_ci95_low": 0.9, "median_ci95_high": 1.1,
            "speedup_vs_baseline": 2.0, "speedup_ci95_low": 1.8,
            "speedup_ci95_high": 2.2,
        }])
        write_csv(gpu, ["backend", "phase", "particles", "total_ms"], [
            {"backend": "cpu_serial", "phase": "steady", "particles": 1024,
             "total_ms": 2.0},
            {"backend": "atomic", "phase": "steady", "particles": 1024,
             "total_ms": 1.0},
            {"backend": "atomic", "phase": "cold", "particles": 1024,
             "total_ms": 50.0},
        ])
        MODULE.render(MODULE.read_rows(cpu), MODULE.read_rows(gpu), output)
        report = (output / "BENCHMARK_RESULTS.generated.md").read_text()
        assert "c1" in report and "cpu4" in report
        assert "1024" in report and "atomic" in report
        assert (output / "cpu_speedup.svg").read_text().startswith("<svg")
        assert (output / "gpu_cic_speedup.svg").read_text().startswith("<svg")
        summary = list(csv.DictReader((output / "gpu_summary.csv").open()))
        atomic = next(row for row in summary if row["backend"] == "atomic")
        assert float(atomic["speedup_vs_cpu_serial"]) == 2.0
    print("parallel result renderer checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
