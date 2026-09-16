"""M4 跟踪子块升级：UCMCTrack 相机运动补偿。

机制：状态建在地平面 [x, ẋ, y, ẏ] 并按匀速模型滤波；观测取检测框底边中点的单应投影；
投影引入的相关噪声经映射矩阵逆变换为地平面非对角协方差，关联用带 ln|S| 项的映射马氏距离；
相机运动被当作加速度噪声写入过程噪声，全序列共用一组补偿参数。
前提：需要相机内外参与地平面单应（依赖 M1）。
"""
from __future__ import annotations


def compensate(detections, tracks, calibration, **kwargs):
    """提供相机运动补偿后的关联代价。TODO(阶段四)。"""
    raise NotImplementedError
