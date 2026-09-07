#!/usr/bin/env python3
"""Summarise a portable-vs-native encode A/B (P4-133, #464).

Reads the text printed by `examples/bench_encode_matrix.rs` (one file per
build variant) and by `examples/bench_c_encode_linux.c`, and prints one
markdown table: a row per benchmark case, and for every variant its time plus
its ratio to a named baseline build and, optionally, to the C reference.

    perf_ab_summary.py --baseline stock --reference C \\
        stock=out/stock.txt fma=out/fma.txt native=out/native.txt C=out/c.txt

Labels are free text; `--baseline` and `--reference` must name one of them.
A run file with no benchmark lines is an error rather than an empty column,
because the point of the table is a measured delta — a variant whose build
failed must not read as "no difference".
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

# Both benches print `encode_<w>x<h>_<sub>` with the geometry padded to a
# fixed width, so the width and height carry stray spaces; the C bench adds a
# `C_` prefix. The time is the first `<float> us` after the case name.
_LINE = re.compile(
    r"^(?:C_)?encode_\s*(\d+)x(\d+)\s*_(\d{3})\b.*?(\d+(?:\.\d+)?)\s*us\b"
)


def parse_bench_output(text: str) -> Dict[str, float]:
    """Map `"<w>x<h>_<sub>"` to microseconds for every benchmark line."""
    cases: Dict[str, float] = {}
    for line in text.splitlines():
        match = _LINE.match(line.strip())
        if match is None:
            continue
        width, height, subsampling, time_us = match.groups()
        cases[f"{width}x{height}_{subsampling}"] = float(time_us)
    return cases


@dataclass(frozen=True)
class Cell:
    time_us: float
    vs_baseline: float
    vs_reference: Optional[float]


@dataclass(frozen=True)
class Row:
    case: str
    columns: Dict[str, Optional[Cell]]


def _sort_key(case: str) -> tuple:
    geometry, subsampling = case.rsplit("_", 1)
    width, height = (int(part) for part in geometry.split("x"))
    return (width * height, subsampling)


def summarize(
    runs: Dict[str, Dict[str, float]],
    baseline: str,
    reference: Optional[str],
) -> List[Row]:
    """One `Row` per case seen in any run, in pixel-count order.

    Raises `KeyError` if `baseline` (or a given `reference`) is not a run
    label, so a typo cannot produce a table of ratios against nothing.
    """
    baseline_times: Dict[str, float] = runs[baseline]
    reference_times: Optional[Dict[str, float]] = runs[reference] if reference else None
    cases: List[str] = sorted({case for times in runs.values() for case in times}, key=_sort_key)
    rows: List[Row] = []
    for case in cases:
        columns: Dict[str, Optional[Cell]] = {}
        for label, times in runs.items():
            time_us: Optional[float] = times.get(case)
            base: Optional[float] = baseline_times.get(case)
            if time_us is None or base is None:
                columns[label] = None
                continue
            ref: Optional[float] = reference_times.get(case) if reference_times else None
            columns[label] = Cell(
                time_us=time_us,
                vs_baseline=time_us / base,
                vs_reference=(time_us / ref) if ref else None,
            )
        rows.append(Row(case=case, columns=columns))
    return rows


def render_table(rows: Iterable[Row]) -> str:
    rows = list(rows)
    if not rows:
        return ""
    labels: List[str] = list(rows[0].columns.keys())
    has_reference: bool = any(
        cell is not None and cell.vs_reference is not None
        for row in rows
        for cell in row.columns.values()
    )
    header: List[str] = ["Case"]
    for label in labels:
        header.append(f"{label} (µs)")
        header.append(f"{label} / baseline")
        if has_reference:
            header.append(f"{label} / reference")
    lines: List[str] = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in rows:
        cells: List[str] = [row.case]
        for label in labels:
            cell: Optional[Cell] = row.columns.get(label)
            if cell is None:
                cells.append("—")
                cells.append("—")
                if has_reference:
                    cells.append("—")
                continue
            cells.append(f"{cell.time_us:.1f}")
            cells.append(f"{cell.vs_baseline:.3f}")
            if has_reference:
                cells.append("—" if cell.vs_reference is None else f"{cell.vs_reference:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: List[str], write: Callable[[str], None] = sys.stdout.write) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, help="label whose times are the 1.000 column")
    parser.add_argument("--reference", default=None, help="label of the C reference run, if any")
    parser.add_argument("runs", nargs="+", metavar="LABEL=PATH")
    args = parser.parse_args(argv)

    runs: Dict[str, Dict[str, float]] = {}
    for spec in args.runs:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            write(f"error: expected LABEL=PATH, got {spec!r}\n")
            return 2
        with open(path, encoding="utf-8") as handle:
            cases: Dict[str, float] = parse_bench_output(handle.read())
        if not cases:
            write(f"error: {path} ({label}) contains no benchmark lines\n")
            return 1
        runs[label] = cases

    try:
        rows: List[Row] = summarize(runs, baseline=args.baseline, reference=args.reference)
    except KeyError as missing:
        write(f"error: no run labelled {missing}\n")
        return 2
    write(render_table(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
