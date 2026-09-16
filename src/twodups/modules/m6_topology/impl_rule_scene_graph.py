"""M6 选定实现：规则化 2D 场景图。

机制：节点为对象 / 区域 / 轨迹；边为隶属、前后左右、相邻、冲突；
每条边必须携带证据类型与置信度，证据不足时标记为不确定，不用缺省关系填充。
标定失效时 geometry_status 降级为 image_plane；输入缺失时列出 missing_inputs 并收窄关系类型。
"""
from __future__ import annotations


def build(track_set, segmentation, match_result=None, calibration=None, **kwargs):
    """返回 SceneGraph2D。TODO(阶段五)。"""
    raise NotImplementedError
