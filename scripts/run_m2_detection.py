#!/usr/bin/env python3
"""M0 → M1 → M2 detection smoke test for a selected manifest. No download."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twodups.contracts import Status, to_wire  # noqa: E402
from twodups.data.manifest import load  # noqa: E402
from twodups.modules.m0_data_ingest import M0DataIngest  # noqa: E402
from twodups.modules.m1_preprocess import M1Preprocess  # noqa: E402
from twodups.modules.m2_detection import M2Detection  # noqa: E402
from twodups.pipeline.runner import iter_m0_m1_m2_chain  # noqa: E402
from twodups.utils.config import REPO_ROOT, load_module  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 YOLO11 检测冒烟测试（不包含 M4 跟踪）")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/m2_detection.yaml")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs/dataset_manifest.yaml")
    parser.add_argument("--sample-id", help="数据清单中的样本 ID")
    parser.add_argument("--start-frame", type=int, default=0, help="首个推理帧索引；之前帧只解码、不做 M1/M2")
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
    detection_config = load_module(config_path)
    params = detection_config["params"]["detection"]
    if args.dry_run:
        print(f"M2 detection=yolo11n; checkpoint={params['checkpoint_path'] or 'not configured'}; "
              "M4 tracking=separate module; no download")
        return 0
    if not args.sample_id:
        parser.error("--sample-id is required unless --dry-run is used")
    if params["checkpoint_path"] is None:
        parser.error("configure checkpoint_path and checkpoint_sha256 before inference; no download is attempted")
    manifest = load(args.manifest)
    m0 = M0DataIngest(load_module(REPO_ROOT / "configs/m0_data_ingest.yaml"))
    m1 = M1Preprocess()
    m2 = M2Detection(load_module(config_path), config_ref=config_ref)
    seen = 0
    for step in iter_m0_m1_m2_chain(m0, manifest, m1, m2, args.sample_id,
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
