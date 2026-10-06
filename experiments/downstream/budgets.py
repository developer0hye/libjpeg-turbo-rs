#!/usr/bin/env python3
"""Regression budgets for the downstream-consumer report (P4-214, #640).

Reads one `report.json` written by the consumer harness and prints, per case,
the same-run timing ratios the budgets are stated in, the budget each ratio is
held to, and whether this report meets it. See BUDGETS.md for the rules and
for why only same-run ratios are budgeted.

    python3 experiments/downstream/budgets.py <report.json> [--first <first.json>] [--markdown]

`--first` names the report the bands and lead budgets were set from (the
committed first report); without it the report under test is its own first.
Both reports must be full runs (not `--smoke`, not `--only`) with the same
architecture, build variant and recorded runtime CPU features. Exit status is 0 when every budgeted row is within budget, 1
when one is over, 2 on a usage error or an unusable pair. Standard library
only.
"""

import json
import sys

# The rule README.md "Regression budget" fixed before any data existed: a
# ratio's band is twice the wider of the two rows' relative p10-p90 spreads in
# the first report, and never under 3 %.
MIN_BAND = 0.03

# Same-run pairs. `parity` pairs hold the candidate to the published release
# (ratio 1.0 plus the band); `lead` pairs hold it to the lead it had over a
# competitor in the first report (measured ratio times 1 + the band).
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


def spread(row):
    timing = row["timing"]
    return (timing["p90_ms"] - timing["p10_ms"]) / timing["median_ms"]


def ratio_rows(case_id, rows, pairs):
    by_backend = {row["backend"]: row for row in rows if row.get("timing")}
    out = []
    for numerator, denominator, kind in pairs:
        if numerator not in by_backend or denominator not in by_backend:
            continue
        a, b = by_backend[numerator], by_backend[denominator]
        ratio = a["timing"]["median_ms"] / b["timing"]["median_ms"]
        band = max(MIN_BAND, 2.0 * max(spread(a), spread(b)))
        out.append(
            {
                "case": case_id,
                "pair": f"{numerator} / {denominator}",
                "kind": kind,
                "ratio": ratio,
                "band": band,
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
    return ratios


def runtime_features(report):
    """The architecture and the `name=yes|no` dispatch inputs a report records."""
    tokens = report["environment"]["runtime_cpu_features"].split()
    return tokens[0], dict(token.split("=", 1) for token in tokens[1:])


def unusable(report, first_report):
    """Why the pair cannot be scored, or None.

    Ratios move with the SIMD kernels runtime dispatch picks and with the
    build variant, so a band set on one runner does not apply to another.
    """
    for name, candidate in (("report", report), ("first report", first_report)):
        # A smoke run times two iterations and says it is not a measurement.
        if candidate.get("smoke"):
            return f"the {name} is a --smoke run, not a measurement"
        # An --only run leaves cases out, so a pass would cover only them.
        if candidate.get("only") is not None:
            return f"the {name} is filtered with --only {candidate['only']!r}"
    if report["build"]["variant"] != first_report["build"]["variant"]:
        return (
            f"the report is VARIANT={report['build']['variant']} but the first "
            f"report is VARIANT={first_report['build']['variant']}"
        )
    architecture, features = runtime_features(report)
    first_architecture, first_features = runtime_features(first_report)
    if architecture != first_architecture:
        return f"the report is {architecture} but the first report is {first_architecture}"
    differing = sorted(
        name
        for name in features.keys() & first_features.keys()
        if features[name] != first_features[name]
    )
    if differing:
        return f"runtime features differ from the first report: {', '.join(differing)}"
    # The 2026-10-07 first report predates recording ssse3, bmi1 and lzcnt.
    # Say which dispatch inputs this comparison could not check rather than
    # refusing every later report.
    unchecked = sorted(features.keys() ^ first_features.keys())
    if unchecked:
        print(
            "budgets.py: not recorded by both reports, so not compared: "
            + ", ".join(unchecked),
            file=sys.stderr,
        )
    return None


def load(path):
    with open(path) as handle:
        return json.load(handle)


def main(argv):
    args = argv[1:]
    as_markdown = "--markdown" in args
    if as_markdown:
        args.remove("--markdown")
    first_path = None
    if "--first" in args:
        index = args.index("--first")
        if index + 1 >= len(args):
            print("--first needs a path", file=sys.stderr)
            return 2
        first_path = args[index + 1]
        del args[index : index + 2]
    if len(args) != 1:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    report = load(args[0])
    first_report = load(first_path) if first_path else report
    problem = unusable(report, first_report)
    if problem:
        print(f"budgets.py: {problem}", file=sys.stderr)
        return 2
    ratios = all_ratios(report)
    first = all_ratios(first_report)
    # README.md: the first report's spread is the noise floor for every
    # later report, so bands come from it, not from the run under test.
    first_band = {(entry["case"], entry["pair"]): entry["band"] for entry in first}
    lead_limit = {
        (entry["case"], entry["pair"]): entry["ratio"] * (1.0 + entry["band"])
        for entry in first
        if entry["kind"] == "lead"
    }

    over = 0
    if as_markdown:
        print("| case | pair | ratio | band | budget | verdict |")
        print("|---|---|---:|---:|---:|---|")
    for entry in ratios:
        # Parity: the release's speed plus the first report's band. Lead: the
        # first report's ratio times (1 + its band), so a lead can shrink only by
        # what the first run could not resolve.
        key = (entry["case"], entry["pair"])
        if entry["kind"] == "parity":
            budget = 1.0 + first_band[key] if key in first_band else None
        else:
            budget = lead_limit.get(key)
        if budget is None:
            verdict, budget_text = "no first-report row", "-"
        else:
            verdict = "ok" if entry["ratio"] <= budget else "OVER"
            over += verdict == "OVER"
            budget_text = f"{budget:.3f}"
        if as_markdown:
            print(
                f"| {entry['case']} | {entry['pair']} | {entry['ratio']:.3f} | "
                f"{100 * first_band.get(key, entry['band']):.1f}% | {budget_text} | {verdict} |"
            )
        else:
            print(
                f"{entry['case']:32} {entry['pair']:44} {entry['ratio']:6.3f} "
                f"{100 * first_band.get(key, entry['band']):5.1f}% {budget_text:>20} {verdict}"
            )
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
