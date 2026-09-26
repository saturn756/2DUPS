#!/usr/bin/env python3
"""M1→M2→M4 detection-only smoke test. No checkpoint download."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twodups.contracts import Status  # noqa: E402
from twodups.data.bdd100k_loader import BDD100KVideoLoader  # noqa: E402
from twodups.data.manifest import load  # noqa: E402
from twodups.modules.m2_preprocess import M2Preprocess  # noqa: E402
from twodups.modules.m4_detection_tracking import M4DetectionTracking  # noqa: E402
from twodups.utils.config import REPO_ROOT, load_module  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 YOLO11 检测子块冒烟测试（不包含跟踪）")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/m4_detection_tracking.yaml")
    parser.add_argument("--sample-id", help="BDD100K 五段样本中的一个视频 ID")
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="只检查配置，不加载视频或权重")
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
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
    for frame in loader.iter_sample(args.sample_id, max_frames=args.max_frames):
        processed = m2.process(frame)
        detections = detector.detect(processed.variants, processed.images_rgb)
        print(f"{detections.frame_id} variant={detections.variant_id} "
              f"status={detections.status.value} objects={len(detections.objects)} "
              f"reason={detections.reason or '-'}")
        if detections.status is not Status.OK:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
