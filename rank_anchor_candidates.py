from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.qa_anchor_candidates import rank_anchor_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank anchor candidates from an existing qa_report.json.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/qa_report.json"),
        help="Path to qa_report.json",
    )
    parser.add_argument(
        "--lane",
        type=str,
        required=True,
        help="Target lane, e.g. side_90 or back_180",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of candidates to print",
    )
    parser.add_argument(
        "--prefer-shadow-router",
        action="store_true",
        help="Prefer shadow router output when report already contains it.",
    )
    parser.add_argument(
        "--recompute-shadow-router",
        action="store_true",
        help="Recompute shadow router for items missing router output in the report.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root used to resolve input/ and runtime assets.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional input directory override.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = rank_anchor_candidates(
        report_path=args.report,
        lane=args.lane,
        top_n=args.top_n,
        prefer_shadow_router=bool(args.prefer_shadow_router),
        recompute_shadow_router=bool(args.recompute_shadow_router),
        base_dir=args.base_dir,
        input_dir=args.input_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
