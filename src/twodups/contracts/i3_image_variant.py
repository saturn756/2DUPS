"""I3 ImageVariantSet / QualityReport（边 E04–E06、E16）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import CalibrationStatus, Ref, Status
from .i2_calibrated_frame import ImageSize


@dataclass
class ImageVariant:
    variant_id: str
    kind: str                                  # raw / enhanced / ...
    media_ref: Ref
    params_ref: Ref | None = None
    parent_variant_id: str | None = None
    coordinate_space: str = "pixel"
    transform_ref: Ref | None = None
    operations: list[dict] = field(default_factory=list)


@dataclass
class ImageVariantSet:
    """raw 版本必须始终存在；三条支路必须使用同一 variant_id。"""

    INTERFACE = "I3"

    frame_id: str
    variants: list[ImageVariant]
    default_variant: str
    sequence_id: str = ""
    image_size: ImageSize | None = None
    coordinate_space: str = "pixel"
    calibration_status: CalibrationStatus = CalibrationStatus.MISSING
    manifest_ref: str | None = None
    schema_version: str = "1.0"
    producer_ref: str = "m2.clahe_lab"
    config_ref: str = "configs/m2_preprocess.yaml"

    def get(self, variant_id: str) -> ImageVariant:
        for v in self.variants:
            if v.variant_id == variant_id:
                return v
        raise KeyError(f"未找到图像版本：{variant_id}")


@dataclass
class QualityReport:
    """门控决策必须带理由；处理失败也要给出 use_raw 而不是阻塞下游。"""

    INTERFACE = "I3"

    frame_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    gate_decision: str = "use_raw"             # use_raw / use_processed
    reason: list[str] = field(default_factory=list)
    sequence_id: str = ""
    quality_flags: list[str] = field(default_factory=list)
    operations: list[dict] = field(default_factory=list)
    selected_variant_id: str = "raw"
    manifest_ref: str | None = None
    processing_ms: float = 0.0
    status: Status = Status.OK
    schema_version: str = "1.0"
    producer_ref: str = "m2.quality_gate"
    config_ref: str = "configs/m2_preprocess.yaml"
