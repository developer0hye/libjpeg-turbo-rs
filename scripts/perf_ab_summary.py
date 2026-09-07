#!/usr/bin/env python3
"""Summarise a portable-vs-native encode A/B (P4-133, #464).

Reads the text printed by `examples/bench_encode_matrix.rs` (one file per
build variant) and by `examples/bench_c_encode_linux.c`, and prints one
markdown table: a row per benchmark case, and for every variant its time plus
its ratio to a named baseline build and, optionally, to the C reference.

    perf_ab_summary.py --baseline stock --reference C \\
        --noise-pair stock,stock-again \\
        stock=out/stock.txt fma=out/fma.txt native=out/native.txt C=out/c.txt

Labels are free text; `--baseline`, `--reference` and both halves of
`--noise-pair` must name one of them. The noise pair is the same binary timed
twice, first and last: its per-case spread is the run's own drift, and a
variant whose ratio to the baseline sits inside that spread is marked
`(noise)` rather than left for a reader to mistake for a win.

A run file with no benchmark lines is an error rather than an empty column,
because the point of the table is a measured delta — a variant whose build
failed must not read as "no difference". A case a run lacks keeps whatever
the other runs measured; only the ratios that need the missing value become
unknown.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Both benches print `encode_<w>x<h>_<sub>` with the geometry padded to a
# fixed width, so the width and height carry stray spaces; the C bench adds a
# `C_` prefix. The time is the first `<float> us` after the case name.
_LINE = re.compile(
    r"^(?:C_)?encode_\s*(\d+)x(\d+)\s*_(\d{3})\b.*?(\d+(?:\.\d+)?)\s*us\b"
)


def parse_bench_output(text: str) -> Dict[str, float]:
    """Map `"<w>x<h>_<sub>"` to microseconds for every benchmark line.

    Raises `ValueError` if a case appears twice: the key is geometry plus
    subsampling, so a second fixture of the same shape would otherwise
    overwrite the first without a trace.
    """
    cases: Dict[str, float] = {}
    for line in text.splitlines():
        match = _LINE.match(line.strip())
        if match is None:
            continue
        width, height, subsampling, time_us = match.groups()
        key: str = f"{width}x{height}_{subsampling}"
        if key in cases:
            raise ValueError(f"case {key} appears more than once in one run")
        cases[key] = float(time_us)
    return cases


@dataclass(frozen=True)
class Cell:
    time_us: float
    vs_baseline: Optional[float]
    vs_reference: Optional[float]
    # True when |vs_baseline - 1| is inside the run's own noise for this case;
    # None when either the ratio or the noise is unknown.
    within_noise: Optional[bool]


@dataclass(frozen=True)
class Row:
    case: str
    columns: Dict[str, Optional[Cell]]
    # |again / first - 1| for the noise pair, or None without a pair or a value.
    noise: Optional[float]


def _sort_key(case: str) -> tuple:
    geometry, subsampling = case.rsplit("_", 1)
    width, height = (int(part) for part in geometry.split("x"))
    return (width * height, subsampling)


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def summarize(
    runs: Dict[str, Dict[str, float]],
    baseline: str,
    reference: Optional[str],
    noise_pair: Optional[Tuple[str, str]] = None,
) -> List[Row]:
    """One `Row` per case seen in any run, in pixel-count order.

    Raises `KeyError` if `baseline`, a given `reference`, or a noise-pair
    label is not a run label, so a typo cannot produce a table of ratios
    against nothing.
    """
    baseline_times: Dict[str, float] = runs[baseline]
    reference_times: Optional[Dict[str, float]] = runs[reference] if reference else None
    pair: Optional[Tuple[Dict[str, float], Dict[str, float]]] = None
    if noise_pair is not None:
        pair = (runs[noise_pair[0]], runs[noise_pair[1]])
    cases: List[str] = sorted({case for times in runs.values() for case in times}, key=_sort_key)
    rows: List[Row] = []
    for case in cases:
        noise: Optional[float] = None
        if pair is not None:
            spread: Optional[float] = _ratio(pair[1].get(case), pair[0].get(case))
            noise = None if spread is None else abs(spread - 1.0)
        columns: Dict[str, Optional[Cell]] = {}
        for label, times in runs.items():
            time_us: Optional[float] = times.get(case)
            if time_us is None:
                columns[label] = None
                continue
            vs_baseline: Optional[float] = _ratio(time_us, baseline_times.get(case))
            within: Optional[bool] = None
            if vs_baseline is not None and noise is not None:
                within = abs(vs_baseline - 1.0) <= noise
            columns[label] = Cell(
                time_us=time_us,
                vs_baseline=vs_baseline,
                vs_reference=_ratio(time_us, reference_times.get(case)) if reference_times else None,
                within_noise=within,
            )
        rows.append(Row(case=case, columns=columns, noise=noise))
    return rows


def _ratio_text(value: Optional[float], within_noise: Optional[bool] = None) -> str:
    if value is None:
        return "—"
    text: str = f"{value:.3f}"
    return f"{text} (noise)" if within_noise else text


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
    has_noise: bool = any(row.noise is not None for row in rows)
    header: List[str] = ["Case"]
    if has_noise:
        header.append("noise (pair spread)")
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
        if has_noise:
            cells.append("—" if row.noise is None else f"{row.noise * 100:.1f}%")
        for label in labels:
            cell: Optional[Cell] = row.columns.get(label)
            if cell is None:
                cells.extend(["—"] * (3 if has_reference else 2))
                continue
            cells.append(f"{cell.time_us:.1f}")
            cells.append(_ratio_text(cell.vs_baseline, cell.within_noise))
            if has_reference:
                cells.append(_ratio_text(cell.vs_reference))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: List[str], write: Callable[[str], None] = sys.stdout.write) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, help="label whose times are the 1.000 column")
    parser.add_argument("--reference", default=None, help="label of the C reference run, if any")
    parser.add_argument(
        "--noise-pair",
        default=None,
        metavar="FIRST,AGAIN",
        help="two labels of the same binary timed first and last; their spread marks noise",
    )
    parser.add_argument("runs", nargs="+", metavar="LABEL=PATH")
    args = parser.parse_args(argv)

    noise_pair: Optional[Tuple[str, str]] = None
    if args.noise_pair is not None:
        first, sep, again = args.noise_pair.partition(",")
        if not sep or not first or not again:
            write(f"error: --noise-pair expects FIRST,AGAIN, got {args.noise_pair!r}\n")
            return 2
        noise_pair = (first, again)

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
        rows: List[Row] = summarize(
            runs, baseline=args.baseline, reference=args.reference, noise_pair=noise_pair
        )
    except KeyError as missing:
        write(f"error: no run labelled {missing}\n")
        return 2
    write(render_table(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
