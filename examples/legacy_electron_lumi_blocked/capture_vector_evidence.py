#!/usr/bin/env python3
"""Seal compiler/disassembly evidence from the linked production executable."""

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symbols(binary):
    raw = subprocess.check_output(
        ["nm", "--defined-only", str(binary)], text=True)
    records = []
    names = []
    for line in raw.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3:
            records.append((fields[0], fields[1], fields[2]))
            names.append(fields[2])
    demangled = subprocess.check_output(
        ["c++filt"], input="\n".join(names) + "\n", text=True).splitlines()
    return [(address, kind, raw_name, pretty)
            for (address, kind, raw_name), pretty in zip(records, demangled)]


def unique_symbol(records, needle):
    matches = [record for record in records
               if needle in record[3] and "[clone " not in record[3]]
    if len(matches) != 1:
        rendered = "\n".join(record[3] for record in matches[:20])
        raise SystemExit(
            f"expected one defined symbol containing {needle!r}, "
            f"found {len(matches)}\n{rendered}")
    return matches[0]


def disassemble(binary, raw_name):
    # GNU objdump applies --demangle while resolving --disassemble=SYMBOL;
    # passing a raw ELF name together with -C therefore produces an empty
    # section listing. Keep lookup/output raw here; symbols.txt records both
    # spellings and the instruction checks do not depend on presentation.
    return subprocess.check_output(
        ["objdump", "-d", "--no-show-raw-insn",
         f"--disassemble={raw_name}", str(binary)], text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--vector-report", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    binary = args.binary.resolve(strict=True)
    vector_report = args.vector_report.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing {output}")

    records = symbols(binary)
    evaluate = unique_symbol(
        records,
        "feebl::legacy_electron_lumi_blocked::evaluate_block(")
    dispatcher = unique_symbol(records, "GRID::step_lumi_blocked(")
    evaluate_disassembly = disassemble(binary, evaluate[2])
    dispatcher_disassembly = disassemble(binary, dispatcher[2])
    report_text = vector_report.read_text(errors="replace")

    vectorized_lines = [
        line for line in report_text.splitlines()
        if "legacyElectronLumiBlockedCPP.cc" in line and
        "loop vectorized" in line
    ]
    if len(vectorized_lines) < 2:
        raise SystemExit(
            "compiler report does not prove at least two blocked-kernel loops "
            "were vectorized")
    packed_sqrt = bool(re.search(r"\bv?sqrtp[sd]\b", evaluate_disassembly))
    packed_arithmetic = bool(re.search(
        r"\bv?(?:add|sub|mul)p[sd]\b", evaluate_disassembly))
    dispatcher_calls_kernel = "evaluate_block" in dispatcher_disassembly
    if not packed_sqrt or not packed_arithmetic or not dispatcher_calls_kernel:
        raise SystemExit(
            "linked-binary disassembly lacks packed sqrt/arithmetic or the "
            "production dispatcher call to evaluate_block")

    output.mkdir(parents=True)
    shutil.copy2(vector_report, output / "compiler-vectorization.txt")
    (output / "evaluate_block.disassembly.txt").write_text(
        evaluate_disassembly)
    (output / "step_lumi_blocked.disassembly.txt").write_text(
        dispatcher_disassembly)
    (output / "symbols.txt").write_text(
        "\n".join(
            f"{address} {kind} {raw_name} {pretty}"
            for address, kind, raw_name, pretty in (evaluate, dispatcher)) +
        "\n")
    manifest = {
        "schema": "feebl-electron-lumi-vector-evidence-v1",
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "input_vector_report": str(vector_report),
        "input_vector_report_sha256": sha256(vector_report),
        "evaluate_symbol": evaluate[3],
        "dispatcher_symbol": dispatcher[3],
        "source_vectorized_lines": vectorized_lines,
        "packed_sqrt_in_linked_symbol": packed_sqrt,
        "packed_arithmetic_in_linked_symbol": packed_arithmetic,
        "production_dispatcher_calls_kernel": dispatcher_calls_kernel,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hashes = []
    for path in sorted(output.iterdir()):
        if path.name != "SHA256SUMS":
            hashes.append(f"{sha256(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
