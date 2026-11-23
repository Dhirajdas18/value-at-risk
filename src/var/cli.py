"""Command line: `var report <topic>` builds a topic's charts and prints its numbers."""

import argparse
from pathlib import Path

OUT = Path("out")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="var", description="Value at Risk on NSE portfolios")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="build a topic's charts and print its numbers")
    rep.add_argument("topic")
    args = parser.parse_args(argv)

    from var import plots, reports

    if args.topic not in reports.REPORTS:
        parser.error(f"unknown topic {args.topic!r}; topics: {', '.join(reports.REPORTS)}")
    plots.style()
    for key, value in reports.REPORTS[args.topic](OUT).items():
        print(f"{key}: {value}")
    print(f"charts -> {OUT.resolve()}")
    return 0
