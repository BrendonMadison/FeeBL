"""Offline command line. No simulation is executed without an exact review token."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .common import Invalid, read_json, write_json
from .compare import compare
from .lhe import parse, roundtrip, validate_sidecar
from .runner import doctor, make_plan, run_plan
from .toy import simulate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "doctor", help="Read-only path/tool inventory; no installs or implicit WSL launch"
    )
    p.add_argument("--inventory")
    p.add_argument("--wsl", action="store_true")
    p.add_argument("--output")
    p = sub.add_parser(
        "plan", help="Authenticate a single run and write an immutable reviewed plan"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("run", help="Dry-run by default; actual execution only in native Linux")
    p.add_argument("--plan", required=True)
    p.add_argument("--review-token", required=True)
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser(
        "compare", help="Preregistered seed-bootstrap equivalence, not a noisy-bin plot test"
    )
    p.add_argument("--candidate", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser(
        "lhe-validate", help="Strict supported LHEF 3 structural subset, no transport claim"
    )
    p.add_argument("file")
    p.add_argument("--sidecar")
    p = sub.add_parser("lhe-roundtrip", help="Preserve HEPEUP data and semantic XML extensions")
    p.add_argument("source")
    p.add_argument("destination")
    p = sub.add_parser(
        "toy-monochromatization", help="Synthetic Gaussian head-on analytic workshop"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser(
        "selftest", help="Run isolated synthetic unittest fixtures, not FeeBL physics"
    )
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(args.inventory, args.wsl)
        elif args.command == "plan":
            result = make_plan(args.config)
        elif args.command == "run":
            result = run_plan(args.plan, args.review_token, args.execute)
        elif args.command == "compare":
            result = compare(
                read_json(args.candidate), read_json(args.reference), read_json(args.contract)
            )
        elif args.command == "lhe-validate":
            _, result = parse(args.file)
            if args.sidecar:
                result["sidecar"] = validate_sidecar(read_json(args.sidecar), result)
        elif args.command == "lhe-roundtrip":
            result = roundtrip(args.source, args.destination)
        elif args.command == "toy-monochromatization":
            result = simulate(read_json(args.config))
        elif args.command == "selftest":
            root = Path(__file__).resolve().parent.parent
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-v"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )
            result = {
                "schema": "feebl.harness_selftest.v1",
                "classification": "SYNTHETIC_ONLY",
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "scope": "Harness negative/roundtrip/analytic fixtures only. No actual FeeBL simulation, external reference or machine feasibility has been validated.",
            }
        if getattr(args, "output", None):
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return (
            2
            if result.get("status") == "BLOCKED"
            else (
                1
                if result.get("status")
                in {"FAIL", "FAIL_OR_INCONCLUSIVE", "FAILED", "TIMEOUT", "TIMEOUT_CLEANUP_FAILED"}
                else 0
            )
        )
    except (Invalid, ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
