#!/usr/bin/env python3
"""打印接口契约与必需随行字段，用于自查与评审。

用法：python scripts/check_contracts.py
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import fields

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from twodups import contracts  # noqa: E402

CASES = (
    (contracts.DatasetManifest, "I1"), (contracts.CalibratedFrame, "I2"),
    (contracts.ImageVariantSet, "I3"), (contracts.QualityReport, "I3"),
    (contracts.MatchResult, "I4"), (contracts.DetectionSet, "I5"),
    (contracts.TrackSet, "I6"), (contracts.SegmentationMap, "I7"),
    (contracts.SceneGraph2D, "I8"), (contracts.EvaluationRecord, "I9"),
)


def main() -> int:
    print(f"接口总数：{len(contracts.INTERFACES)} -> {', '.join(contracts.INTERFACES)}\n")
    for cls, iface in CASES:
        names = ", ".join(f.name for f in fields(cls))
        print(f"[{iface}] {cls.__name__}\n    {names}")
    print("\n强制随行字段：frame_id / variant_id / producer_ref(config_ref) / calibration(geometry_status)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
