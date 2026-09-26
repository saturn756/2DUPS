"""Shared wire-level states and references. See docs/interface/04_接口规范.md."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class Status(str, Enum):
    """通用可用性状态。失败必须显式，不允许用缺省值掩盖。"""

    OK = "ok"
    UNRELIABLE = "unreliable"
    FAILED = "failed"


class CalibrationStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    STALE = "stale"


class GeometryStatus(str, Enum):
    DEGRADED = "degraded"
    IMAGE_PLANE = "image_plane"


class TrackState(str, Enum):
    TRACKED = "tracked"
    LOST = "lost"
    NEW = "new"
    UNRELIABLE = "unreliable"


@dataclass(frozen=True)
class Ref:
    """大对象引用。图像、掩码、点集一律用引用传递，不内嵌像素或高维特征张量。"""

    uri: str
    kind: str = "file"
    checksum: str | None = None


@dataclass(frozen=True)
class Producer:
    """实现与配置标识，评测可复现的前提。"""

    producer_ref: str
    config_ref: str


@dataclass
class CalibrationState:
    """标定状态三态；非 valid 时下游禁止输出带物理尺度的结论。"""

    status: CalibrationStatus
    params_ref: Ref | None = None
    reason: str | None = None


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if (not all(math.isfinite(value) for value in (self.x1, self.y1, self.x2, self.y2))
                or self.x2 <= self.x1 or self.y2 <= self.y1):
            raise ValueError("Invalid bbox coordinates")
