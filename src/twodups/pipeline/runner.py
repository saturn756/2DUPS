"""离线链路编排。

顺序：输入 → M1 → M2 →｛M3 ∥ M4 ∥ M5｝→ M6 → 输出。
对齐规则（见 docs/architecture/02_模块规范.md）：
  · 三条支路必须使用同一 variant_id；
  · 缺失输入只能等待或降级，不得用其它帧或其它版本顶替；
  · M3 消费 pair(t-1, t)，其结果只能用于第 t 帧及之后。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from ..contracts.i5_detection_set import DetectionSet
from ..data.bdd100k_loader import BDD100KVideoLoader, LoadedFrame
from ..modules.m2_preprocess.module import M2Preprocess, ProcessedFrame
from ..modules.m4_detection_tracking.module import M4DetectionTracking


@dataclass
class RunResult:
    frame_id: str
    scene_graph: Any | None = None          # SceneGraph2D（I8）
    degraded: list[str] = field(default_factory=list)


@dataclass
class DetectionOnlyResult:
    """One verified I2 → I3 → I5 handoff; pixels stay in ProcessedFrame."""

    frame: LoadedFrame
    processed: ProcessedFrame
    detections: DetectionSet


def iter_detection_only(
    loader: BDD100KVideoLoader,
    preprocess: M2Preprocess,
    detector: M4DetectionTracking,
    sample_id: str,
    *,
    start_frame: int = 0,
    max_frames: int = 1,
) -> Iterator[DetectionOnlyResult]:
    """Run the current small chain, without implying tracking or full M1–M6 support."""
    if start_frame < 0 or max_frames <= 0:
        raise ValueError("start_frame must be non-negative and max_frames positive")
    for frame in loader.iter_sample(sample_id, max_frames=start_frame + max_frames):
        if frame.metadata.frame_index < start_frame:
            continue
        processed = preprocess.process(frame)
        variants = processed.variants
        if ((variants.sequence_id, variants.frame_id) !=
                (frame.metadata.sequence_id, frame.metadata.frame_id)):
            raise ValueError("M2 I3 frame identity does not match I2")
        if processed.quality.selected_variant_id != variants.default_variant:
            raise ValueError("M2 quality decision does not match I3 default_variant")
        detections = detector.detect(variants, processed.images_rgb)
        if ((detections.sequence_id, detections.frame_id, detections.variant_id,
             detections.coordinate_space) !=
                (variants.sequence_id, variants.frame_id, variants.default_variant,
                 variants.coordinate_space)):
            raise ValueError("M4 I5 is not aligned with I3")
        yield DetectionOnlyResult(frame, processed, detections)


def run(config: dict[str, Any]) -> list[RunResult]:
    """完整链路尚未实现；只允许各模块已有的独立入口运行。"""
    raise NotImplementedError(
        "完整链路未接通；当前可运行 scripts/run_dataloader.py 和 scripts/run_m2.py"
    )
