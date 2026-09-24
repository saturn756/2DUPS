#!/usr/bin/env python3
"""Check BDD100K decoding and sparse-label alignment without writing frames."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twodups.data.bdd100k_loader import BDD100KVideoLoader  # noqa: E402
from twodups.data.manifest import load  # noqa: E402
from twodups.utils.config import REPO_ROOT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="流式检查 BDD100K 视频和单帧标注")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs/dataset_manifest.yaml")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="媒体引用的仓库根目录")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-id", help="仅检查一个样本")
    parser.add_argument("--max-frames", type=int, help="每段最多读取 N 帧；截断时不要求匹配未到达的标签")
    args = parser.parse_args()

    manifest = load(args.manifest)
    loader = BDD100KVideoLoader(manifest, root=args.root)
    sample_ids = [args.sample_id] if args.sample_id else manifest.splits[args.split]
    for sample_id in sample_ids:
        frames = 0
        corrected = 0
        labels = []
        size = None
        rotation = None
        for frame in loader.iter_sample(sample_id, max_frames=args.max_frames):
            frames += 1
            metadata = frame.metadata
            size = (metadata.image_size.width, metadata.image_size.height)
            rotation = metadata.rotation_degrees_clockwise
            if metadata.source_timestamp is None or abs(metadata.timestamp - metadata.source_timestamp) > 1e-6:
                corrected += 1
            for annotation in frame.annotations:
                labels.append((metadata.frame_index, len(annotation.objects),
                               abs(metadata.timestamp - annotation.timestamp)))
        label_text = ", ".join(f"frame={index}, objects={count}, error={error:.4f}s"
                                for index, count, error in labels) or "none"
        print(f"{sample_id}: frames={frames}, size={size}, rotation_cw={rotation}, "
              f"timestamps_corrected={corrected}, labels=[{label_text}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
