"""I3 ImageVariantSet / QualityReport（边 E04–E06、E16）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import Ref


@dataclass
class ImageVariant:
    variant_id: str
    kind: str                                  # raw / enhanced / ...
    media_ref: Ref
    params_ref: Ref | None = None


@dataclass
class ImageVariantSet:
    """raw 版本必须始终存在；三条支路必须使用同一 variant_id。"""

    INTERFACE = "I3"

    frame_id: str
    variants: list[ImageVariant]
    default_variant: str

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
