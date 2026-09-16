"""接口通用约定：强制随行字段、状态枚举与大对象引用。

每个接口都必须能回答四件事：哪一帧（frame_id）、哪个图像版本（variant_id）、
哪个实现与配置（producer_ref / config_ref）、哪个标定状态（calibration / geometry_status）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    FULL = "full"
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
