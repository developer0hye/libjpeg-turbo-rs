#!/usr/bin/env python3
"""Regression budgets for the downstream-consumer report (P4-214, #640).

Reads one `report.json` written by the consumer harness and the reference
reports the budgets were set from, and prints, per case, the same-run timing
ratios the budgets are stated in, the budget each ratio is held to, and
whether this report meets it. See BUDGETS.md for the rules and for why only
same-run ratios are budgeted.

    python3 experiments/downstream/budgets.py <report.json> \\
        [--first <reference.json>]... [--markdown]

Each `--first` names one reference report (the committed reference set);
without any, the report under test is its own single reference. Every report
must be a full run (not `--smoke`, not `--only`) with the same CPU model,
architecture, build variant, recorded runtime CPU features, consumer
sources, rustc, iterations and warmup, and concurrent thread and decode
counts. Exit status is 0 when every budgeted row is within budget, 1 when
one is over, 2 on a usage error or an unusable set. Standard library only.
"""

import json
import statistics
import sys

# README.md "Regression budget" fixed this floor before any data existed.
MIN_BAND = 0.03

# A pair whose ratio moved more than this between reference runs of one
# consumer binary on one CPU model cannot be budgeted on hosted runners: a
# limit that wide would pass real regressions. Such rows are reported, not
# scored, until a quiet machine measures them.
MAX_RESOLVABLE_RANGE = 0.10

# Same-run pairs. Every pair is held to the reference set (its median times
# 1 + band). `parity` pairs compare with the published release and also say
# whether the reference is behind it; `lead` pairs compare with another codec.
DECODE_PAIRS = [
    ("candidate-fresh", "baseline-fresh", "parity"),
    ("candidate-reuse", "baseline-reuse", "parity"),
    ("candidate-reuse", "zune-reuse", "lead"),
    ("candidate-image-adapter", "image-builtin", "lead"),
]
ENCODE_PAIRS = [
    ("candidate", "baseline", "parity"),
    ("candidate-444", "baseline-444", "parity"),
    ("candidate-444", "image-builtin", "lead"),
    ("candidate-image-adapter", "image-builtin", "lead"),
]
THUMBNAIL_PAIRS = [
    ("candidate", "baseline", "parity"),
    ("candidate-image-adapter", "image-builtin", "lead"),
    ("candidate-scaled-decode", "image-builtin", "lead"),
]
CONCURRENT_PAIRS = DECODE_PAIRS


def timing_of(row):
    # Concurrent rows time a whole batch; the other sections time one call.
    return row.get("timing") or row.get("batch_timing")


def spread(timing):
    return (timing["p90_ms"] - timing["p10_ms"]) / timing["median_ms"]


def ratio_rows(case_id, rows, pairs):
    by_backend = {row["backend"]: row for row in rows if timing_of(row)}
    out = []
    for numerator, denominator, kind in pairs:
        if numerator not in by_backend or denominator not in by_backend:
            continue
        a = timing_of(by_backend[numerator])
        b = timing_of(by_backend[denominator])
        out.append(
            {
                "key": (case_id, f"{numerator} / {denominator}"),
                "kind": kind,
                "ratio": a["median_ms"] / b["median_ms"],
                "spread": max(spread(a), spread(b)),
            }
        )
    return out


def all_ratios(report):
    ratios = []
    for case in report["decode"]:
        ratios += ratio_rows(case["id"], case["rows"], DECODE_PAIRS)
    for case in report["encode"]:
        ratios += ratio_rows(case["id"], case["rows"], ENCODE_PAIRS)
    ratios += ratio_rows("thumbnail", report["thumbnail"]["rows"], THUMBNAIL_PAIRS)
    if report.get("concurrent"):
        concurrent = report["concurrent"]
        ratios += ratio_rows(concurrent["id"], concurrent["rows"], CONCURRENT_PAIRS)
    return ratios


def runtime_features(report):
    """The architecture and the `name=yes|no` dispatch inputs a report records."""
    tokens = report["environment"]["runtime_cpu_features"].split()
    return tokens[0], dict(token.split("=", 1) for token in tokens[1:])


def comparability(report):
    """Everything that must match for two reports' ratios to be comparable.

    Ratios move with the CPU model, with the SIMD kernels runtime dispatch
    picks, with the build variant, the compiler, the sampling settings, the
    concurrent workload's shape, and with the harness binary itself (BUDGETS.md: one library, two
    consumer builds, 5 % and 30 % apart on two parity rows).
    """
    architecture, features = runtime_features(report)
    concurrent = report.get("concurrent") or {}
    return {
        # The same features on another microarchitecture still move ratios:
        # hosted x86_64 runners alternate between Zen 3 and Zen 4 parts.
        "CPU": report["environment"]["cpu"],
        "architecture": architecture,
        "runtime features": features,
        "VARIANT": report["build"]["variant"],
        "consumer sources": report["build"].get("consumer_source_sha256"),
        # The compiler is part of the binary; stable moves every six weeks,
        # and the reference has to be regenerated when it does.
        "rustc": report["environment"]["rustc"].splitlines()[0],
        # A band measured over 30 samples says nothing about a 3-sample run.
        "iterations": report["iterations"],
        "warmup": report["warmup"],
        # T comes from the runner's parallelism, so two runners of one CPU
        # model can still run different concurrent workloads.
        "concurrent threads": concurrent.get("threads"),
        "concurrent decodes per thread": concurrent.get("decodes_per_thread"),
    }


def unusable(paths, reports):
    """Why the set cannot be scored, or None."""
    for path, report in zip(paths, reports):
        # A smoke run times two iterations and says it is not a measurement.
        if report.get("smoke"):
            return f"{path} is a --smoke run, not a measurement"
        # An --only run leaves cases out, so a pass would cover only them.
        if report.get("only") is not None:
            return f"{path} is filtered with --only {report['only']!r}"
        # Reports from before the hash was recorded cannot show they share a
        # consumer binary with anything, so they cannot join a set.
        if len(set(paths)) > 1 and not report["build"].get("consumer_source_sha256"):
            return f"{path} predates consumer_source_sha256, so its binary cannot be matched"
    expected = comparability(reports[0])
    for path, report in zip(paths[1:], reports[1:]):
        actual = comparability(report)
        for name, value in expected.items():
            if actual[name] != value:
                return f"{path} differs from {paths[0]} in {name}: {actual[name]} vs {value}"
    return None


def load(path):
    with open(path) as handle:
        return json.load(handle)


def parse_arguments(argv):
    """(report path, reference paths, markdown?) or None on a usage error."""
    args = argv[1:]
    as_markdown = "--markdown" in args
    args = [arg for arg in args if arg != "--markdown"]
    references = []
    while "--first" in args:
        index = args.index("--first")
        if index + 1 >= len(args):
            return None
        references.append(args[index + 1])
        del args[index : index + 2]
    if len(args) != 1:
        return None
    return args[0], references or [args[0]], as_markdown


def budgets(reference_reports):
    """Per pair: the reference median, cross-run range, band and limit."""
    by_key = {}
    for report in reference_reports:
        for entry in all_ratios(report):
            by_key.setdefault(entry["key"], []).append(entry)
    out = {}
    for key, entries in by_key.items():
        ratios = [entry["ratio"] for entry in entries]
        cross_run = max(ratios) - min(ratios)
        within_run = 2.0 * max(entry["spread"] for entry in entries)
        band = max(MIN_BAND, within_run, cross_run)
        median = statistics.median(ratios)
        kind = entries[0]["kind"]
        out[key] = {
            "median": median,
            "range": cross_run,
            "band": band,
            "resolvable": cross_run <= MAX_RESOLVABLE_RANGE,
            # The regression limit is the reference itself plus its noise:
            # a later report may not be slower than the reference set by
            # more than the reference runs disagreed. Holding parity rows to
            # 1.0 instead would flag every known gap on every run and bury a
            # new regression among them; the gaps are reported separately.
            "limit": median * (1.0 + band),
            # Parity rows only: the reference is behind the published
            # release by more than its band. These are the losing cases
            # BUDGETS.md lists; they do not fail the check.
            "behind_release": kind == "parity" and median > 1.0 + band,
        }
    return out


def main(argv):
    parsed = parse_arguments(argv)
    if parsed is None:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    report_path, reference_paths, as_markdown = parsed
    paths = [report_path] + reference_paths
    reports = [load(path) for path in paths]
    problem = unusable(paths, reports)
    if problem:
        print(f"budgets.py: {problem}", file=sys.stderr)
        return 2
    limits = budgets(reports[1:])

    over = 0
    if as_markdown:
        print(
            "| case | pair | kind | ratio | reference median | cross-run range "
            "| band | budget | verdict | reference vs 0.8.0 |"
        )
        print("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for entry in all_ratios(reports[0]):
        budget = limits.get(entry["key"])
        release_text = "-"
        if budget is None:
            verdict = "no reference row"
            median_text = range_text = band_text = limit_text = "-"
        else:
            if entry["kind"] == "parity":
                release_text = "behind" if budget["behind_release"] else "within band"
            median_text = f"{budget['median']:.3f}"
            range_text = f"{100 * budget['range']:.1f}%"
            band_text = f"{100 * budget['band']:.1f}%"
            limit_text = f"{budget['limit']:.3f}"
            if not budget["resolvable"]:
                verdict = "not resolvable on these runners"
            elif entry["ratio"] <= budget["limit"]:
                verdict = "ok"
            else:
                verdict = "OVER"
                over += 1
        case, pair = entry["key"]
        if as_markdown:
            print(
                f"| {case} | {pair} | {entry['kind']} | {entry['ratio']:.3f} | "
                f"{median_text} | {range_text} | {band_text} | {limit_text} | {verdict} | "
                f"{release_text} |"
            )
        else:
            print(
                f"{case:32} {pair:44} {entry['ratio']:6.3f} {median_text:>6} "
                f"{range_text:>6} {band_text:>6} {limit_text:>6} {verdict:8} {release_text}"
            )
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
