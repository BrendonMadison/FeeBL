#!/usr/bin/env python3
"""Run a packaged collision example or verify its preserved input data.

Run from any directory. Paths in manifest.json are relative to this directory;
--feebl and --output are relative to the caller. Each execution uses a new output
directory and records input/binary hashes, seed, return code and elapsed time.
Successful process completion establishes execution, not physics convergence.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    """Return a file digest without retaining the complete file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def packaged_path(relative: str) -> Path:
    """Resolve an existing packaged file; reject paths outside the bundle."""
    path = (HERE / relative).resolve()
    if not path.is_relative_to(HERE) or not path.is_file():
        raise ValueError(f"Invalid packaged input: {relative}")
    return path


def verify(manifest: dict) -> int:
    """Check preserved data against the published SHA-256 inventory."""
    for item in manifest["files"]:
        path = packaged_path(item["path"])
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Packaged file differs from manifest: {item['path']}")
    for case in manifest["cases"]:
        for target, source in case["inputs"].items():
            if Path(target).name != target or target in (".", ".."):
                raise ValueError(f"Invalid native input name: {target}")
            packaged_path(source)
    print(f"Verified {len(manifest['files'])} files and {len(manifest['cases'])} cases.")
    return 0


def run_case(case: dict, executable: Path, output: Path, timeout: float,
             memory_gib: float) -> int:
    """Run one native case under a wall-time limit; never overwrite a result.

    The optional address-space limit is applied on POSIX; zero leaves memory
    limits to the scheduler or operating system. Native execution uses
    one OpenMP/BLAS thread. A timeout terminates the process group on POSIX and
    the native process on Windows, leaving a failed receipt and its logs.
    """
    executable, output = executable.resolve(), output.resolve()
    if not executable.is_file():
        raise ValueError(f"FeeBL executable does not exist: {executable}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--timeout must be positive and finite.")
    if not math.isfinite(memory_gib) or memory_gib < 0:
        raise ValueError("--memory-gib must be nonnegative and finite.")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True)
    inputs = {}
    for target, relative in case["inputs"].items():
        source = packaged_path(relative)
        shutil.copyfile(source, output / target)
        inputs[target] = sha256(source)

    command = [str(executable), *case["options"], case["accelerator"],
               case["parameters"], "out.dat"]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
    receipt = {
        "schema": "feebl.example-execution/1", "case": case["id"],
        "binary_sha256": sha256(executable), "input_sha256": inputs,
        "origin_source_revision": case["origin_source_revision"],
        "seed": case["settings"].get("rndm_seed"),
        "argv_after_executable": command[1:], "timeout_seconds": timeout,
        "memory_gib": memory_gib if os.name == "posix" and memory_gib > 0 else None,
        "status": "FAILED", "physics_qualification": "not assessed by runner",
    }

    def limit_memory() -> None:
        import resource
        byte_limit = int(memory_gib * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (byte_limit, byte_limit))

    process = None
    started = time.monotonic()
    try:
        with (output / "stdout.txt").open("wb") as stdout, \
                (output / "stderr.txt").open("wb") as stderr:
            options = {"start_new_session": True} if os.name == "posix" else {}
            if os.name == "posix" and memory_gib > 0:
                options["preexec_fn"] = limit_memory
            process = subprocess.Popen(command, cwd=output, env=environment,
                                       stdout=stdout, stderr=stderr, **options)
            try:
                return_code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                raise
        receipt["returncode"] = return_code
        if return_code != 0:
            raise RuntimeError(f"FeeBL exited with code {return_code}; inspect stderr.txt.")
        if not (output / "out.dat").is_file() or (output / "out.dat").stat().st_size < 100:
            raise RuntimeError("FeeBL did not produce a nonempty out.dat.")
        receipt["status"] = "EXECUTED"
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        receipt["error"] = str(error)
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        # The dataset finalizer reads the runtime backend receipt from log.txt.
        # Keep the streams separately, and combine them without altering bytes;
        # concatenation preserves each stream's order, not inter-stream timing.
        with (output / "log.txt").open("wb") as combined:
            for name in ("stdout.txt", "stderr.txt"):
                path = output / name
                if path.exists():
                    with path.open("rb") as stream:
                        shutil.copyfileobj(stream, combined)
        receipt["outputs"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(output.iterdir()) if p.is_file()
        }
        (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                encoding="utf-8")
        print(f"{case['id']}: {receipt['status']} ({receipt['elapsed_seconds']:.2f} s)")


def main() -> int:
    """Parse the public command line and report explicit failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="List native examples and numerical settings.")
    actions.add_parser("verify", help="Verify preserved data and input hashes.")
    run = actions.add_parser("run", help="Execute one case in a new directory.")
    run.add_argument("case")
    run.add_argument("--feebl", type=Path, required=True, help="Native FeeBL executable.")
    run.add_argument("--output", type=Path, required=True, help="New result directory.")
    run.add_argument("--timeout", type=float, help="Wall-time limit in seconds.")
    run.add_argument("--memory-gib", type=float, default=8.0,
                     help="POSIX address-space limit in GiB (default: 8; 0 uses only OS/scheduler limits).")
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    try:
        if args.action == "verify":
            return verify(manifest)
        if args.action == "list":
            for case in manifest["cases"]:
                settings = case["settings"]
                grid = "x".join(settings.get(k, "?") for k in ("n_x", "n_y", "n_z"))
                print(f"{case['id']:<44} N={settings.get('n_m', '?'):<7} grid={grid}")
            return 0
        case = next((row for row in manifest["cases"] if row["id"] == args.case), None)
        if case is None:
            raise ValueError(f"Unknown case {args.case!r}; use the list command.")
        verify(manifest)
        return run_case(case, args.feebl, args.output,
                        args.timeout if args.timeout is not None else case["timeout_seconds"],
                        args.memory_gib)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
