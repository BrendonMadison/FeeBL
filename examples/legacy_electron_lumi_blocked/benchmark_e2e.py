#!/usr/bin/env python3
"""Paired diagnostic screen: ordered precompute mode 1 versus blocked mode 2.

This harness deliberately cannot promote a performance claim.  Run it only in
an explicitly granted idle window on the reduced p_s9 fixture, and retain the
JSON together with the exact binary/build/vectorization provenance.
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


MARKERS = {
    "mode1": re.compile(
        br"^ electron_lumi_precompute = 1 .*\n", re.MULTILINE),
    "mode2": re.compile(
        br"^ electron_lumi_precompute = 2 .*\n", re.MULTILINE),
}


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


def candidate_deck(source, profile, mode):
    raw = pathlib.Path(source).read_bytes()
    header = re.compile(
        br"(^\$PARAMETERS::[ \t]+" + re.escape(profile.encode()) +
        br"[ \t]*\r?\n[ \t]*\{)", re.MULTILINE)
    matches = list(header.finditer(raw))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one $PARAMETERS:: {profile} block, "
            f"found {len(matches)}")
    start = matches[0].end()
    next_block = raw.find(b"\n$", start)
    block_end = len(raw) if next_block < 0 else next_block
    if b"electron_lumi_precompute" in raw[start:block_end]:
        raise SystemExit(f"{profile} already sets electron_lumi_precompute")
    return (raw[:start] +
            f"electron_lumi_precompute={mode};".encode() + raw[start:])


def run_once(binary, deck, accelerator, profile, directory):
    directory.mkdir()
    stdout = directory / "stdout.txt"
    stderr = directory / "stderr.txt"
    command = [str(binary), f"--acc_file={deck}", accelerator, profile,
               "result.out"]
    before = time.perf_counter_ns()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        completed = subprocess.run(command, cwd=directory, stdout=out,
                                   stderr=err, check=False)
    elapsed = (time.perf_counter_ns() - before) / 1.0e9
    if completed.returncode != 0:
        tail = stderr.read_text(errors="replace")[-2000:]
        raise SystemExit(
            f"run failed with exit {completed.returncode}: "
            f"{' '.join(command)}\n{tail}")
    return elapsed, directory / "result.out", directory / "rndm.save"


def stripped_output(path, mode):
    payload = pathlib.Path(path).read_bytes()
    own = MARKERS[mode]
    other = MARKERS["mode2" if mode == "mode1" else "mode1"]
    matches = own.findall(payload)
    if len(matches) != 1 or other.search(payload):
        raise SystemExit(
            f"{mode}: missing, duplicate, or incorrect provenance marker")
    return own.sub(b"", payload)


def compare_exact(first, second, repetition):
    mode1 = stripped_output(first, "mode1")
    mode2 = stripped_output(second, "mode2")
    if mode1 != mode2:
        for line, (left, right) in enumerate(
                zip(mode1.splitlines(), mode2.splitlines()), start=1):
            if left != right:
                raise SystemExit(
                    f"sample {repetition}: output mismatch at line {line}:\n"
                    f"mode1={left!r}\nmode2={right!r}")
        raise SystemExit(
            f"sample {repetition}: output lengths differ: "
            f"{len(mode1)} != {len(mode2)}")


def git_state(root):
    def git(*arguments):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *arguments],
                stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {
        "commit": git("rev-parse", "HEAD"),
        "status_porcelain": git("status", "--porcelain=v1"),
        "diff_sha256": hashlib.sha256(
            git("diff", "--binary", "HEAD").encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--deck", required=True, type=pathlib.Path)
    parser.add_argument("--accelerator", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cpu", type=int,
                        help="pin the harness and child runs to one logical CPU")
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="write a new JSON file (refuses to overwrite)")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be >=1 and warmups >=0")
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu requires sched_setaffinity support")
        try:
            os.sched_setaffinity(0, {args.cpu})
        except OSError as error:
            parser.error(f"cannot pin to CPU {args.cpu}: {error}")

    binary = args.binary.resolve(strict=True)
    deck = args.deck.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    root = pathlib.Path(__file__).resolve().parents[2]
    decks = {
        "mode1": candidate_deck(deck, args.profile, 1),
        "mode2": candidate_deck(deck, args.profile, 2),
    }
    timings = {"mode1": [], "mode2": []}

    with tempfile.TemporaryDirectory(
            prefix="feebl-ee-blocked-benchmark-") as temporary:
        work = pathlib.Path(temporary)
        deck_paths = {}
        for mode, content in decks.items():
            deck_paths[mode] = work / f"{mode}.dat"
            deck_paths[mode].write_bytes(content)

        sequence = []
        for warmup in range(args.warmups):
            sequence.extend((f"warmup-{warmup}-mode1",
                             f"warmup-{warmup}-mode2"))
        for repetition in range(args.repetitions):
            modes = ("mode1", "mode2") if repetition % 2 == 0 \
                    else ("mode2", "mode1")
            sequence.extend(f"sample-{repetition}-{mode}" for mode in modes)

        paired_outputs = {}
        for tag in sequence:
            mode = tag.rsplit("-", 1)[1]
            elapsed, result, rng = run_once(
                binary, deck_paths[mode], args.accelerator, args.profile,
                work / tag)
            if tag.startswith("sample-"):
                repetition = int(tag.split("-", 2)[1])
                timings[mode].append(elapsed)
                paired_outputs.setdefault(repetition, {})[mode] = (result, rng)

        rng_checked = False
        for repetition, outputs in sorted(paired_outputs.items()):
            compare_exact(outputs["mode1"][0], outputs["mode2"][0],
                          repetition)
            first_rng = outputs["mode1"][1]
            second_rng = outputs["mode2"][1]
            if first_rng.exists() or second_rng.exists():
                if (not first_rng.exists() or not second_rng.exists() or
                        first_rng.read_bytes() != second_rng.read_bytes()):
                    raise SystemExit(f"sample {repetition}: RNG states differ")
                rng_checked = True

    first_median = statistics.median(timings["mode1"])
    second_median = statistics.median(timings["mode2"])
    report = {
        "schema": "feebl-electron-lumi-blocked-screen-v1",
        "claim_status": (
            "diagnostic-only reduced-fixture screen; not a broad-set claim"),
        "comparison": "electron_lumi_precompute=1 / =2",
        "exact_output_for_every_timed_pair": True,
        "rng_state_checked": rng_checked,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "source_deck": str(deck),
        "source_deck_sha256": sha256(deck),
        "derived_deck_sha256": {
            mode: hashlib.sha256(content).hexdigest()
            for mode, content in decks.items()
        },
        "accelerator": args.accelerator,
        "profile": args.profile,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "order": "paired AB/BA alternating after fixed-order warmup",
        "seconds": timings,
        "median_seconds": {
            "mode1": first_median,
            "mode2": second_median,
        },
        "median_speedup_mode1_over_mode2": first_median / second_median,
        "host": {
            "platform": platform.platform(),
            "cpu_model": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "affinity": (sorted(os.sched_getaffinity(0))
                         if hasattr(os, "sched_getaffinity")
                         else "unavailable"),
        },
        "selected_environment": {
            name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
             "CUDA_VISIBLE_DEVICES")
        },
        "git": git_state(root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
