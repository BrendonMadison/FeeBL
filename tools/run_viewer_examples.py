#!/usr/bin/env python3
"""Run and authenticate the compact native viewer examples on a compute node.

The output directory must be new. Runs are sequential and retain execution
receipts, native files and failed attempts. Finalization requires NumPy/SciPy.
These examples exercise interfaces; their low resolution is not a physics test.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

CASES = ("viewer_radiation_spin_pairs", "viewer_bhabha_field_on", "viewer_bhabha_zero_field")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feebl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--finalize-timeout", type=int, default=900)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in args.cases:
        if case not in (*CASES, "lep1994_focusing", "smoke"):
            parser.error("unknown viewer example")
        run = args.output / case
        commands = [
            [sys.executable, "-B", str(source / "benchmarks/run.py"), "run", case,
             "--feebl", str(args.feebl.resolve()), "--output", str(run.resolve())],
            [sys.executable, "-B", str(source / "tools/finalize_gui_dataset.py"), str(run.resolve()),
             "--binary", str(args.feebl.resolve()), "--card", str(source / "benchmarks/cases" / case / "acc.dat"),
             "--source", str(source)],
        ]
        record = dict(case=case, status="FAILED", stages=[])
        results.append(record)
        try:
            for stage, command in zip(("native", "finalize"), commands):
                started = time.monotonic()
                with (args.output / (case + "_" + stage + ".log")).open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=900 if stage == "native" else args.finalize_timeout)
                record["stages"].append(dict(stage=stage, returncode=result.returncode,
                                              elapsed_s=time.monotonic() - started))
                if result.returncode:
                    raise RuntimeError(stage + " failed; inspect retained log")
            record["status"] = "PASS_INTERFACE_CONTRACT"
            record["bytes"] = sum(p.stat().st_size for p in run.iterdir() if p.is_file())
            record["physics_qualification"] = "NOT_ESTABLISHED"
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            record["error"] = str(error)
        (args.output / "viewer_example_execution.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return int(any(r["status"] != "PASS_INTERFACE_CONTRACT" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
