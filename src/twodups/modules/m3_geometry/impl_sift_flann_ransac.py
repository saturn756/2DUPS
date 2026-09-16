"""M3 选定实现：SIFT + FLANN(BBF) + RANSAC。

机制：DoG 尺度空间极值定位关键点 → 三维二次 Taylor 拟合求亚像素位置 → Hessian 去边缘响应
→ 36 bin 方向直方图定方向 → 4×4×8 共 128 维描述子；匹配用最近邻/次近邻距离比 0.8，
FLANN 加速，RANSAC 估计单应并剔除外点。
对照实现：XFeat（64 维稠密描述子 + 可靠性图，CPU 实时）。
"""
from __future__ import annotations


def estimate(image_a, image_b, **kwargs):
    """返回 MatchResult 所需的对应关系、内点与几何变换。TODO(阶段四)。"""
    raise NotImplementedError
