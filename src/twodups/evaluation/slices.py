"""数据切片定义。切片口径必须与 SceneGraph2D.window 一致。"""
from __future__ import annotations

DEFAULT_SLICES = (
    "normal",
    "low_light",
    "blur",
    "occlusion",
    "far",
    "camera_motion",
    "calibration_missing",
)


def slice_of(sample) -> str:
    """由样本的 slice_tags 归一到一个评测切片。TODO：阶段六实现。"""
    raise NotImplementedError
