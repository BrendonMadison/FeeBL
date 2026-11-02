#!/usr/bin/env python3
"""Render machine-readable FeeBL backend results as Markdown and SVG.

Only Python's standard library is used so the reproducibility driver does not
need a plotting environment.  The renderer never supplies missing values: a
section is labelled "not generated" unless its input CSV was provided.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


def read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def format_number(value: str | float, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "not available"
    return f"{number:.{digits}g}"


def aggregate_gpu(rows: Iterable[dict[str, str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("phase") != "steady":
            continue
        try:
            key = (int(row["particles"]), row["backend"])
            grouped[key].append(float(row["total_ms"]))
        except (KeyError, TypeError, ValueError):
            continue
    particles = sorted({key[0] for key in grouped})
    output: list[dict[str, float | int | str]] = []
    for count in particles:
        serial_values = grouped.get((count, "cpu_serial"), [])
        serial = statistics.median(serial_values) if serial_values else math.nan
        for backend in ("cpu_serial", "atomic", "shared", "sorted"):
            values = grouped.get((count, backend), [])
            if not values:
                continue
            median = statistics.median(values)
            output.append(
                {
                    "particles": count,
                    "backend": backend,
                    "samples": len(values),
                    "median_ms": median,
                    "speedup_vs_cpu_serial": serial / median if serial_values else math.nan,
                }
            )
    return output


def write_gpu_summary(rows: Sequence[dict[str, float | int | str]], path: Path) -> None:
    fields = ["particles", "backend", "samples", "median_ms", "speedup_vs_cpu_serial"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def svg_bar_chart(entries: Sequence[tuple[str, float]], title: str, destination: Path) -> None:
    width = 920
    row_height = 38
    margin_left = 250
    margin_right = 80
    margin_top = 70
    height = max(180, margin_top + row_height * len(entries) + 55)
    finite = [value for _, value in entries if math.isfinite(value) and value >= 0]
    maximum = max(finite, default=1.0)
    plot_width = width - margin_left - margin_right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="36" font-family="sans-serif" font-size="22" '
        f'font-weight="bold">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(entries):
        y = margin_top + index * row_height
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 20}" text-anchor="end" '
            f'font-family="monospace" font-size="14">{html.escape(label)}</text>'
        )
        if math.isfinite(value) and value >= 0:
            bar_width = 0 if maximum == 0 else value / maximum * plot_width
            parts.append(
                f'<rect x="{margin_left}" y="{y + 5}" width="{bar_width:.2f}" '
                'height="21" fill="#276fbf"/>'
            )
            parts.append(
                f'<text x="{margin_left + bar_width + 8:.2f}" y="{y + 21}" '
                f'font-family="monospace" font-size="13">{value:.4g}</text>'
            )
        else:
            parts.append(
                f'<text x="{margin_left}" y="{y + 20}" font-family="sans-serif" '
                'font-size="13" fill="#777">not available</text>'
            )
    parts.append("</svg>")
    destination.write_text("\n".join(parts) + "\n")


def render(cpu_rows: list[dict[str, str]], gpu_rows: list[dict[str, str]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    gpu_summary = aggregate_gpu(gpu_rows)
    if gpu_summary:
        write_gpu_summary(gpu_summary, output / "gpu_summary.csv")

    document = [
        "# Generated FeeBL parallel-backend results",
        "",
        "This file is generated only from the CSV inputs named below. Missing",
        "sections are left explicit; the renderer does not synthesize timings.",
        "",
        "## CPU end-to-end summary",
        "",
    ]
    if cpu_rows:
        document.append(
            markdown_table(
                ["case", "variant", "n", "median s", "MAD s", "speedup", "speedup 95% CI"],
                (
                    [
                        row.get("case", ""),
                        row.get("variant", ""),
                        row.get("samples", ""),
                        format_number(row.get("median_seconds", "")),
                        format_number(row.get("mad_seconds", "")),
                        format_number(row.get("speedup_vs_baseline", "")),
                        (
                            format_number(row.get("speedup_ci95_low", ""))
                            + " to "
                            + format_number(row.get("speedup_ci95_high", ""))
                        ),
                    ]
                    for row in cpu_rows
                ),
            )
        )
        cpu_entries = [
            (
                f"{row.get('case', '')} / {row.get('variant', '')}",
                float(row.get("speedup_vs_baseline", "nan")),
            )
            for row in cpu_rows
        ]
        svg_bar_chart(cpu_entries, "CPU end-to-end speedup vs serial", output / "cpu_speedup.svg")
    else:
        document.append("Not generated: no CPU summary CSV was supplied.")

    document.extend(["", "## GPU CIC steady-call summary", ""])
    if gpu_summary:
        document.append(
            markdown_table(
                ["particles/slice", "backend", "n", "median ms", "speedup"],
                (
                    [
                        str(row["particles"]),
                        str(row["backend"]),
                        str(row["samples"]),
                        format_number(row["median_ms"]),
                        format_number(row["speedup_vs_cpu_serial"]),
                    ]
                    for row in gpu_summary
                ),
            )
        )
        gpu_entries = [
            (f"{row['particles']} / {row['backend']}", float(row["speedup_vs_cpu_serial"]))
            for row in gpu_summary
            if row["backend"] != "cpu_serial"
        ]
        svg_bar_chart(
            gpu_entries, "GPU CIC speedup vs serial kernel", output / "gpu_cic_speedup.svg"
        )
    else:
        document.append("Not generated: no GPU raw benchmark CSV was supplied.")

    document.extend(
        [
            "",
            "## Interpretation guard",
            "",
            "A kernel speedup is not an end-to-end FeeBL speedup. Candidate selection",
            "also requires the registered accuracy gates and controlled whole-program",
            "measurements. A speedup whose 95% interval includes 1 is inconclusive.",
            "",
        ]
    )
    (output / "BENCHMARK_RESULTS.generated.md").write_text("\n".join(document))


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-summary", type=Path)
    parser.add_argument("--gpu-raw", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    if args.cpu_summary is not None and not args.cpu_summary.is_file():
        raise SystemExit(f"CPU summary is missing: {args.cpu_summary}")
    if args.gpu_raw is not None and not args.gpu_raw.is_file():
        raise SystemExit(f"GPU raw CSV is missing: {args.gpu_raw}")
    render(read_rows(args.cpu_summary), read_rows(args.gpu_raw), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
