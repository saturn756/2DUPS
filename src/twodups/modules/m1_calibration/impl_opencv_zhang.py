"""M1 选定实现：OpenCV 张氏平面标定 + 畸变校正。

机制：平面靶标求单应 H = A[r1 r2 t]；r1⊥r2 给出绝对二次曲线约束，
每个单应只提供 2 个内参约束，故必须多视图；闭式解后以极大似然精化重投影误差，
径向畸变只建模 k1、k2 并与其它参数整体优化。
对照实现：自动标定（GNN 单应估计），见 docs/methods/。
"""
from __future__ import annotations


def calibrate(images, pattern_size, square_size, **kwargs):
    """返回 (内参矩阵, 畸变系数, 重投影误差)。TODO(阶段二)。"""
    raise NotImplementedError
