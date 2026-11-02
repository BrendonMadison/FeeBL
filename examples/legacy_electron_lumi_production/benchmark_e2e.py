#!/usr/bin/env python3
"""Paired, exactness-gated timing harness for electron_lumi_precompute.

Results are diagnostic until an independent benchmark owner controls machine
state, freezes the dirty tree into an identified binary, and runs the required
broad production set. This script never promotes its own result to a claim.
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import statistics
import subprocess
import tempfile
import time


MARKER = re.compile(br"^ electron_lumi_precompute = 1 .*\n", re.MULTILINE)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_model():
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def candidate_deck(source, profile):
    raw = pathlib.Path(source).read_bytes()
    header = re.compile(
        br"(^\$PARAMETERS::[ \t]+" + re.escape(profile.encode()) +
        br"[ \t]*\r?\n[ \t]*\{)", re.MULTILINE)
    matches = list(header.finditer(raw))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one $PARAMETERS:: {profile} block, found {len(matches)}")
    start = matches[0].end()
    next_block = raw.find(b"\n$", start)
    block_end = len(raw) if next_block < 0 else next_block
    if b"electron_lumi_precompute" in raw[start:block_end]:
        raise SystemExit(f"{profile} already sets electron_lumi_precompute")
    return raw[:start] + b"electron_lumi_precompute=1;" + raw[start:]


def run_once(binary, deck, accelerator, profile, directory):
    directory.mkdir()
    stdout = directory / "stdout.txt"
    stderr = directory / "stderr.txt"
    command = [str(binary), f"--acc_file={deck}", accelerator, profile,
               "result.out"]
    before = time.perf_counter_ns()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        completed = subprocess.run(command, cwd=directory, stdout=out, stderr=err,
                                   check=False)
    elapsed = (time.perf_counter_ns() - before) / 1.0e9
    if completed.returncode != 0:
        tail = stderr.read_text(errors="replace")[-2000:]
        raise SystemExit(
            f"run failed with exit {completed.returncode}: {' '.join(command)}\n{tail}")
    return elapsed, directory / "result.out", directory / "rndm.save"


def compare_exact(cartesian, precomputed):
    legacy = pathlib.Path(cartesian).read_bytes()
    fast = pathlib.Path(precomputed).read_bytes()
    markers = MARKER.findall(fast)
    if len(markers) != 1 or MARKER.search(legacy):
        raise SystemExit("missing, duplicate, or misplaced fast-path provenance")
    if legacy != MARKER.sub(b"", fast):
        raise SystemExit("candidate/Cartesian output mismatch; timing is invalid")


def git_state(root):
    def git(*arguments):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *arguments], stderr=subprocess.DEVNULL,
                text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {
        "commit": git("rev-parse", "HEAD"),
        "status_porcelain": git("status", "--porcelain=v1"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--deck", required=True, type=pathlib.Path)
    parser.add_argument("--accelerator", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path,
                        help="write new JSON file (refuses to overwrite)")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be >=1 and warmups >=0")

    binary = args.binary.resolve(strict=True)
    deck = args.deck.resolve(strict=True)
    root = pathlib.Path(__file__).resolve().parents[2]
    fast_bytes = candidate_deck(deck, args.profile)
    timings = {"cartesian": [], "precomputed": []}

    with tempfile.TemporaryDirectory(prefix="feebl-ee-precompute-benchmark-") as tmp:
        work = pathlib.Path(tmp)
        cart_deck = work / "cartesian.dat"
        fast_deck = work / "precomputed.dat"
        cart_deck.write_bytes(deck.read_bytes())
        fast_deck.write_bytes(fast_bytes)

        sequence = []
        for warmup in range(args.warmups):
            sequence.extend((f"warmup-{warmup}-cartesian",
                             f"warmup-{warmup}-precomputed"))
        for repetition in range(args.repetitions):
            modes = ("cartesian", "precomputed") if repetition % 2 == 0 \
                    else ("precomputed", "cartesian")
            sequence.extend(f"sample-{repetition}-{mode}" for mode in modes)

        paired_outputs = {}
        for tag in sequence:
            mode = tag.rsplit("-", 1)[1]
            selected_deck = cart_deck if mode == "cartesian" else fast_deck
            elapsed, result, rng = run_once(
                binary, selected_deck, args.accelerator, args.profile, work / tag)
            if tag.startswith("sample-"):
                repetition = int(tag.split("-", 2)[1])
                timings[mode].append(elapsed)
                paired_outputs.setdefault(repetition, {})[mode] = (result, rng)

        rng_checked = False
        for repetition, outputs in sorted(paired_outputs.items()):
            compare_exact(outputs["cartesian"][0], outputs["precomputed"][0])
            cart_rng, fast_rng = outputs["cartesian"][1], outputs["precomputed"][1]
            if cart_rng.exists() or fast_rng.exists():
                if not cart_rng.exists() or not fast_rng.exists() or \
                        cart_rng.read_bytes() != fast_rng.read_bytes():
                    raise SystemExit(f"sample {repetition}: RNG states differ")
                rng_checked = True

    cart_median = statistics.median(timings["cartesian"])
    fast_median = statistics.median(timings["precomputed"])
    report = {
        "schema": "feebl-electron-lumi-precompute-benchmark-v1",
        "claim_status": "diagnostic-only; not a broad-set performance claim",
        "exact_output_for_every_timed_pair": True,
        "rng_state_checked": rng_checked,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "source_deck": str(deck),
        "source_deck_sha256": sha256(deck),
        "candidate_deck_sha256": hashlib.sha256(fast_bytes).hexdigest(),
        "accelerator": args.accelerator,
        "profile": args.profile,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "order": "paired alternating after fixed-order warmups",
        "seconds": timings,
        "median_seconds": {
            "cartesian": cart_median,
            "precomputed": fast_median,
        },
        "median_speedup": cart_median / fast_median,
        "host": {
            "platform": platform.platform(),
            "cpu_model": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity") else "unavailable",
        },
        "selected_environment": {
            name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "CUDA_VISIBLE_DEVICES")
        },
        "git": git_state(root),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
