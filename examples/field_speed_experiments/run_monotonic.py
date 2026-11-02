#!/usr/bin/env python3
"""Run one command with a monotonic wall clock and auditable resource data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


def elapsed_text(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600.0)
    minutes, remainder = divmod(remainder, 60.0)
    if hours >= 1.0:
        return f"{int(hours)}:{int(minutes):02d}:{remainder:05.2f}"
    return f"{int(minutes)}:{remainder:05.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start_ns = time.perf_counter_ns()
    with args.stdout.open("wb") as stdout, args.stderr.open("wb") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    elapsed_ns = time.perf_counter_ns() - start_ns
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_seconds = elapsed_ns / 1.0e9
    payload = {
        "schema": "feebl.monotonic-command-timing.v1",
        "clock": "python.time.perf_counter_ns",
        "clock_monotonic": True,
        "command": command,
        "exit_code": completed.returncode,
        "wall_nanoseconds": elapsed_ns,
        "wall_seconds": wall_seconds,
        "child_user_seconds": after.ru_utime - before.ru_utime,
        "child_system_seconds": after.ru_stime - before.ru_stime,
        "maximum_resident_set_kbytes": after.ru_maxrss,
        "voluntary_context_switches": after.ru_nvcsw - before.ru_nvcsw,
        "involuntary_context_switches": after.ru_nivcsw - before.ru_nivcsw,
    }
    args.json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.resource.write_text(
        "\n".join(
            (
                f"Command being timed: {' '.join(command)}",
                "Clock source: python.time.perf_counter_ns (monotonic)",
                "Elapsed (wall clock) time (h:mm:ss or m:ss): "
                f"{elapsed_text(wall_seconds)}",
                f"Elapsed monotonic seconds: {wall_seconds:.9f}",
                f"User time (seconds): {payload['child_user_seconds']:.9f}",
                f"System time (seconds): {payload['child_system_seconds']:.9f}",
                "Maximum resident set size (kbytes): "
                f"{payload['maximum_resident_set_kbytes']}",
                f"Voluntary context switches: {payload['voluntary_context_switches']}",
                "Involuntary context switches: "
                f"{payload['involuntary_context_switches']}",
                f"Exit status: {completed.returncode}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return completed.returncode if completed.returncode >= 0 else 128 - completed.returncode


if __name__ == "__main__":
    sys.exit(main())
