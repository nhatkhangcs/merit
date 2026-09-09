#!/usr/bin/env python3
"""Validate completed MERIT runs and export comparison tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from merit.export_tables import export_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", help="completed run directories")
    parser.add_argument("--output-dir", required=True, help="directory for results.csv/results.md")
    parser.add_argument(
        "--allow-fewer-stream-orders",
        action="store_true",
        help="diagnostic only; reported exports require seeds 0, 1, and 2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = export_results(
        args.run_directories,
        args.output_dir,
        require_three_stream_orders=not args.allow_fewer_stream_orders,
    )
    print(f"Exported {len(rows)} validated run rows to {args.output_dir}")


if __name__ == "__main__":
    main()
