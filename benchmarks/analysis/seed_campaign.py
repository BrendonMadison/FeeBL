#!/usr/bin/env python3
"""Prepare or execute independent collision replicas for sampling covariance.

Preparation is light: it copies native inputs into a new campaign directory,
changes only the RNG seed, and records every input hash. Execution is explicit,
bounded and uses one solver process per invocation. Submit these invocations
through the site's scheduler; this program never submits a cluster job.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


BENCHMARKS = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    "geometry_d0",
    "geometry_opposite",
    "geometry_same",
    "ilc1000_tdr_b1b_gaussian_proxy",
    "halhf250_wide_domain",
    "halhf380",
    "halhf550",
    "ilc500_ussc_pairs_2006",
)


def seed_for_case(case_id, replicate):
    """Preserve the case's canonical RNG block when selecting or reordering cases."""
    if case_id not in DEFAULT_CASES:
        raise ValueError(f"No canonical seed block is assigned to {case_id}.")
    if not 0 <= replicate < 100:
        raise ValueError("A canonical seed block supports replicates 0 through 99.")
    return 729000000 + DEFAULT_CASES.index(case_id) * 10000 + replicate * 101


def digest(path):
    """Return the SHA-256 identity of one input or output file."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def prepare(output, case_ids, seeds):
    """Freeze fresh seeds and native inputs; reject an existing campaign."""
    if not 4 <= seeds <= 100:
        raise ValueError("Request between four and 100 independent seeds.")
    manifest = json.loads((BENCHMARKS / "manifest.json").read_text())
    cases = {case["id"]: case for case in manifest["cases"]}
    if set(case_ids) - cases.keys():
        raise ValueError("Unknown case in requested campaign.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("A campaign cannot repeat a case.")
    for case_id in case_ids:
        seed_for_case(case_id, seeds - 1)
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for case_id in case_ids:
        case = cases[case_id]
        for replicate in range(seeds):
            # Disjoint seed blocks identify cases and cannot reuse discovery seeds.
            seed = seed_for_case(case_id, replicate)
            relative = Path(case_id) / f"seed_{seed}"
            directory = output / relative
            directory.mkdir(parents=True)
            for name, source in case["inputs"].items():
                shutil.copyfile(BENCHMARKS / source, directory / name)
            card = (directory / "acc.dat").read_text(encoding="utf-8-sig")
            card, count = re.subn(
                r"(?<!\w)rndm_seed\s*=\s*[^;]+;", f"rndm_seed={seed};", card
            )
            if count != 1:
                raise ValueError(f"Expected one seed assignment in {case_id}.")
            for key in ("rndm_load", "rndm_save"):
                card, count = re.subn(rf"(?<!\w){key}\s*=\s*[^;]+;", f"{key}=0;", card)
                if count != 1:
                    raise ValueError(f"Expected one {key} assignment in {case_id}.")
            (directory / "acc.dat").write_text(card, encoding="utf-8", newline="\n")
            jobs.append(
                {
                    "index": len(jobs),
                    "case": case_id,
                    "seed": seed,
                    "directory": relative.as_posix(),
                    "argv": [
                        *case["options"],
                        case["accelerator"],
                        case["parameters"],
                        "out.dat",
                    ],
                    "timeout_seconds": case["timeout_seconds"],
                    "input_sha256": {p.name: digest(p) for p in directory.iterdir()},
                }
            )
    document = {
        "schema": "feebl.seed-campaign/1",
        "seed_count": seeds,
        "purpose": "Independent sampling covariance at fixed numerical settings",
        "qualification": "Not a spatial, time-step or physical-input convergence study",
        "jobs": jobs,
    }
    (output / "campaign.json").write_text(json.dumps(document, indent=2) + "\n")
    return document


def execute(campaign, index, executable):
    """Execute one frozen replica, retaining timeout/failure receipts and logs."""
    campaign, executable = campaign.resolve(), executable.resolve()
    document = json.loads((campaign / "campaign.json").read_text())
    job = document["jobs"][index]
    directory = campaign / job["directory"]
    if (directory / "execution.json").exists() or (directory / "stdout.txt").exists():
        raise FileExistsError(
            "This replica has already been attempted; create a new campaign."
        )
    for name, expected in job["input_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Changed campaign input: {name}")
    receipt = {**job, "binary_sha256": digest(executable), "status": "FAILED"}
    started = time.monotonic()
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    try:
        with (directory / "stdout.txt").open("wb") as stdout, (
            directory / "stderr.txt"
        ).open("wb") as stderr:
            process = subprocess.Popen(
                [str(executable), *job["argv"]],
                cwd=directory,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name == "posix"),
            )
            try:
                receipt["returncode"] = process.wait(timeout=job["timeout_seconds"])
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                receipt["status"] = "TIMEOUT"
                raise
        if receipt["returncode"] != 0:
            raise RuntimeError(f"Native exit code {receipt['returncode']}")
        if not (directory / "out.dat").is_file():
            raise RuntimeError("Missing native scientific summary")
        receipt["status"] = "EXECUTED"
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["outputs"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in directory.iterdir()
            if p.is_file()
        }
        (directory / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    plan = actions.add_parser("prepare")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--seeds", type=int, default=12)
    plan.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    run = actions.add_parser("execute")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--feebl", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.output, args.cases, args.seeds)
        print(
            f"Prepared {len(result['jobs'])} independent replicas; no simulation executed."
        )
    else:
        execute(args.campaign, args.index, args.feebl)


if __name__ == "__main__":
    main()
