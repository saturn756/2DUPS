#!/usr/bin/env python3
"""评测入口（骨架）：按切片统计模块指标并产出 EvaluationRecord。

用法：python scripts/evaluate.py --list-slices
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from twodups.evaluation.slices import DEFAULT_SLICES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="2DUPS 评测入口")
    ap.add_argument("--list-slices", action="store_true", help="打印预定义数据切片")
    args = ap.parse_args()
    if args.list_slices:
        for s in DEFAULT_SLICES:
            print(s)
        return 0
    print("评测尚未接通：见 docs/architecture/05_实施规划.md（S6）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
