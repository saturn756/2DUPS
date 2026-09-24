#!/usr/bin/env python3
"""M1 DataLoader → M2 quality/variant smoke test on BDD100K samples."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twodups.data.bdd100k_loader import BDD100KVideoLoader  # noqa: E402
from twodups.data.manifest import load  # noqa: E402
from twodups.modules.m2_preprocess import M2Preprocess  # noqa: E402
from twodups.utils.config import REPO_ROOT, load_module  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 M2 简单质量评价与条件增强")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs/dataset_manifest.yaml")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/m2_preprocess.yaml")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-id", help="只运行一段视频")
    parser.add_argument("--max-frames", type=int, default=20, help="每段最多解码 N 帧，默认 20")
    parser.add_argument("--stride", type=int, default=1, help="每 N 帧处理一帧，默认 1")
    args = parser.parse_args()
    if args.max_frames <= 0 or args.stride <= 0:
        parser.error("--max-frames and --stride must be positive")

    manifest = load(args.manifest)
    loader = BDD100KVideoLoader(manifest, root=args.root)
    m2 = M2Preprocess(load_module(args.config))
    sample_ids = [args.sample_id] if args.sample_id else manifest.splits[args.split]
    for sample_id in sample_ids:
        processed_count = 0
        selected: dict[str, int] = {}
        for frame in loader.iter_sample(sample_id, max_frames=args.max_frames):
            if frame.metadata.frame_index % args.stride:
                continue
            result = m2.process(frame)
            processed_count += 1
            choice = result.variants.default_variant
            selected[choice] = selected.get(choice, 0) + 1
            metrics = result.quality.metrics
            print(f"{sample_id} frame={frame.metadata.frame_index}: "
                  f"brightness={metrics['brightness_mean']:.1f}, "
                  f"under={metrics['underexposed_fraction']:.3f}, "
                  f"blur={metrics['blur_laplacian_variance']:.1f}, "
                  f"noise={metrics['noise_sigma']:.1f}, "
                  f"flags={result.quality.quality_flags}, default={choice}")
        print(f"{sample_id}: processed={processed_count}, selected={selected}")
    print(f"Reports and generated PNGs: {m2.output_dir / m2.config_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
