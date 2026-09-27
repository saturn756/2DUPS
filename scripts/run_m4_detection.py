#!/usr/bin/env python3
"""BDD100K DataLoader → M2 → M4 detection-only smoke test. No implicit download."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twodups.contracts import Status, to_wire  # noqa: E402
from twodups.data.bdd100k_loader import BDD100KVideoLoader  # noqa: E402
from twodups.data.manifest import load  # noqa: E402
from twodups.modules.m2_preprocess import M2Preprocess  # noqa: E402
from twodups.modules.m4_detection_tracking import M4DetectionTracking  # noqa: E402
from twodups.pipeline.runner import iter_detection_only  # noqa: E402
from twodups.utils.config import REPO_ROOT, load_module  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 YOLO11 检测子块冒烟测试（不包含跟踪）")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/m4_detection_tracking.yaml")
    parser.add_argument("--sample-id", help="BDD100K 五段样本中的一个视频 ID")
    parser.add_argument("--start-frame", type=int, default=0, help="首个推理帧索引；之前帧只解码、不做 M2/M4")
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="逐帧打印 I5 JSON，便于检查真实检测框")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置，不加载视频或权重")
    args = parser.parse_args()
    if args.max_frames <= 0 or args.start_frame < 0:
        parser.error("--max-frames must be positive and --start-frame non-negative")
    config_path = args.config.resolve()
    try:
        config_ref = config_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        parser.error("--config must be inside the repository")
    detector = M4DetectionTracking(load_module(config_path), config_ref=config_ref)
    params = detector.config["params"]["detection"]
    if args.dry_run:
        print(f"M4 detection=yolo11n; checkpoint={params['checkpoint_path'] or 'not configured'}; "
              "tracking=not implemented; no download")
        return 0
    if not args.sample_id:
        parser.error("--sample-id is required unless --dry-run is used")
    if params["checkpoint_path"] is None:
        parser.error("configure checkpoint_path and checkpoint_sha256 before inference; no download is attempted")
    manifest = load(REPO_ROOT / "configs/dataset_manifest.yaml")
    loader = BDD100KVideoLoader(manifest)
    m2 = M2Preprocess()
    seen = 0
    for step in iter_detection_only(loader, m2, detector, args.sample_id,
                                    start_frame=args.start_frame, max_frames=args.max_frames):
        detections = step.detections
        seen += 1
        if args.json:
            print(json.dumps(to_wire(detections), ensure_ascii=False))
        else:
            print(f"{detections.frame_id} variant={detections.variant_id} "
                  f"status={detections.status.value} objects={len(detections.objects)} "
                  f"labels={sum(len(a.objects) for a in step.frame.annotations)} "
                  f"reason={detections.reason or '-'}")
        if detections.status is not Status.OK:
            return 2
    if seen == 0:
        print("No frames in requested range", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
