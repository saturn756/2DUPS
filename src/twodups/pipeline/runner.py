"""离线链路编排。

主链路：M0 数据接入 → M1 预处理 → M2 检测 → M3 分割 → M4 跟踪
→ M5 集成；E 是贯穿评测层。
对齐规则（见 docs/architecture/02_模块规范.md）：
  · 三条支路必须使用同一 variant_id；
  · 缺失输入只能等待或降级，不得用其它帧或其它版本顶替；
  · M3 消费 pair(t-1, t)，其结果只能用于第 t 帧及之后。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from ..contracts.i5_detection_set import DetectionSet
from ..contracts.i1_dataset_manifest import DatasetManifest
from ..data.bdd100k_loader import BDD100KVideoLoader, LoadedFrame
from ..modules.m0_data_ingest.module import M0DataIngest
from ..modules.m1_preprocess.module import M1Preprocess, ProcessedFrame
from ..modules.m2_detection.module import M2Detection


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


def _iter_detection_frames(
    frames: Iterator[LoadedFrame],
    preprocess: M1Preprocess,
    detector: M2Detection,
    *,
    start_frame: int,
) -> Iterator[DetectionOnlyResult]:
    for frame in frames:
        if frame.metadata.frame_index < start_frame:
            continue
        processed = preprocess.process(frame)
        variants = processed.variants
        if ((variants.sequence_id, variants.frame_id) !=
                (frame.metadata.sequence_id, frame.metadata.frame_id)):
            raise ValueError("M1 I3 frame identity does not match M0 I2")
        if processed.quality.selected_variant_id != variants.default_variant:
            raise ValueError("M1 quality decision does not match I3 default_variant")
        detections = detector.detect(variants, processed.images_rgb)
        if ((detections.sequence_id, detections.frame_id, detections.variant_id,
             detections.coordinate_space) !=
                (variants.sequence_id, variants.frame_id, variants.default_variant,
                 variants.coordinate_space)):
            raise ValueError("M2 I5 is not aligned with I3")
        yield DetectionOnlyResult(frame, processed, detections)


def iter_detection_chain(
    loader: BDD100KVideoLoader,
    preprocess: M1Preprocess,
    detector: M2Detection,
    sample_id: str,
    *,
    start_frame: int = 0,
    max_frames: int = 1,
) -> Iterator[DetectionOnlyResult]:
    """Run the current verified M0 → M1 → M2 chain."""
    if start_frame < 0 or max_frames <= 0:
        raise ValueError("start_frame must be non-negative and max_frames positive")
    yield from _iter_detection_frames(
        loader.iter_sample(sample_id, max_frames=start_frame + max_frames),
        preprocess, detector, start_frame=start_frame,
    )


def iter_m0_m1_m2_chain(
    ingest: M0DataIngest,
    manifest: DatasetManifest,
    preprocess: M1Preprocess,
    detector: M2Detection,
    sample_id: str,
    *,
    start_frame: int = 0,
    max_frames: int = 1,
) -> Iterator[DetectionOnlyResult]:
    """Run the same verified chain through the M0 adapter boundary."""
    if start_frame < 0 or max_frames <= 0:
        raise ValueError("start_frame must be non-negative and max_frames positive")
    yield from _iter_detection_frames(
        ingest.iter_sample(manifest, sample_id, max_frames=start_frame + max_frames),
        preprocess, detector, start_frame=start_frame,
    )


# Compatibility name for callers that used the pre-refactor smoke-test helper.
iter_detection_only = iter_detection_chain


def run(config: dict[str, Any]) -> list[RunResult]:
    """完整链路尚未实现；只允许各模块已有的独立入口运行。"""
    raise NotImplementedError(
        "完整 M0–M5 链路未接通；当前可运行 M0→M1→M2 detection chain"
    )
