#!/usr/bin/env python3
"""跑完整链路（骨架）。

用法：
    python scripts/run_pipeline.py --module-config configs/m4_detection_tracking.yaml --dry-run
链路接通前只做配置解析与打印，便于先校验配置。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from twodups.utils.config import load_module  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="2DUPS 链路运行入口")
    ap.add_argument("--module-config", required=True, help="模块配置，如 configs/m5_segmentation.yaml")
    ap.add_argument("--common", default="configs/common.yaml", help="公共配置")
    ap.add_argument("--dry-run", action="store_true", help="只解析并打印配置")
    args = ap.parse_args()

    config = load_module(args.module_config, args.common)
    if args.dry_run:
        print(f"模块：{config.get('module')} {config.get('name')}")
        print(f"选定实现：{config.get('impl')}")
        print(f"日志目录：{config.get('runtime', {}).get('log_dir')}")
        return 0

    from twodups.pipeline.runner import run
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
