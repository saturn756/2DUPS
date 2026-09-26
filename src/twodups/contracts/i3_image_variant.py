"""I3 ImageVariantSet and QualityReport."""
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

    sequence_id: str
    frame_id: str
    variants: list[ImageVariant]
    default_variant: str
    image_size: ImageSize
    coordinate_space: str
    calibration_status: CalibrationStatus
    manifest_ref: str | None = None
    schema_version: str = "1.0"
    producer_ref: str = "m2.clahe_lab"
    config_ref: str = "configs/m2_preprocess.yaml"

    def __post_init__(self) -> None:
        ids = [variant.variant_id for variant in self.variants]
        if ids.count("raw") != 1 or len(ids) != len(set(ids)):
            raise ValueError("I3 requires exactly one raw and unique variant IDs")
        if self.default_variant not in ids:
            raise ValueError("I3 default_variant must name an existing version")
        if self.get("raw").parent_variant_id is not None:
            raise ValueError("I3 raw variant cannot have a parent")
        if self.get("raw").kind != "raw":
            raise ValueError("I3 raw variant must have raw kind")
        seen: set[str] = set()
        for variant in self.variants:
            if variant.variant_id != "raw" and variant.parent_variant_id is None:
                raise ValueError("I3 derived variant requires a parent")
            if variant.parent_variant_id is not None and variant.parent_variant_id not in seen:
                raise ValueError(f"I3 parent must precede child: {variant.parent_variant_id}")
            if variant.kind == "rectified" and self.calibration_status is not CalibrationStatus.VALID:
                raise ValueError("I3 rectified variant requires valid calibration")
            seen.add(variant.variant_id)
        if self.get(self.default_variant).coordinate_space != self.coordinate_space:
            raise ValueError("I3 default variant coordinate space mismatch")

    def get(self, variant_id: str) -> ImageVariant:
        for v in self.variants:
            if v.variant_id == variant_id:
                return v
        raise KeyError(f"未找到图像版本：{variant_id}")


@dataclass
class QualityReport:
    """门控决策必须带理由；处理失败也要给出 use_raw 而不是阻塞下游。"""

    INTERFACE = "I3"

    sequence_id: str
    frame_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    gate_decision: str = "use_raw"             # use_raw / use_processed
    reason: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    operations: list[dict] = field(default_factory=list)
    selected_variant_id: str = "raw"
    manifest_ref: str | None = None
    processing_ms: float = 0.0
    status: Status = Status.OK
    schema_version: str = "1.0"
    producer_ref: str = "m2.quality_gate"
    config_ref: str = "configs/m2_preprocess.yaml"

    def __post_init__(self) -> None:
        if self.gate_decision not in {"use_raw", "use_processed"}:
            raise ValueError("Unknown I3 gate decision")
        if (self.gate_decision == "use_raw") != (self.selected_variant_id == "raw"):
            raise ValueError("I3 gate decision and selected variant disagree")
