"""M4 跟踪子块主选：ByteTrack（BYTE 关联）。

机制：按阈值把检测分成高分框与低分框；第一次用高分框与全部轨迹（含丢失轨迹）关联；
第二次用剩余轨迹与低分框关联，且只用 IoU（低分框多来自遮挡或模糊，外观不可靠）。
IoU 低于阈值拒绝匹配，丢失超过 N 帧才删除以支持重关联。
"""
from __future__ import annotations


def associate(detections, tracks, **kwargs):
    """返回更新后的 TrackSet。TODO(阶段四)。"""
    raise NotImplementedError
