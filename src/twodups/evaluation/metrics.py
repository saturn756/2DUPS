"""模块级指标。

按模块的验收维度独立统计；检测与跟踪子块必须分开：
  M1 重投影误差、标定可用率；M2 质量指标与下游指标差；M3 内点率、残差、耗时；
  M4 检测 mAP/AP-S 与跟踪 MOTA/IDF1/ID Switch（分开统计）；M5 mIoU、边界与细长结构；
  M6 关系准确率、图合法率、降级判定准确性。
"""
from __future__ import annotations


def compute(module_ref: str, predictions, targets, **kwargs) -> dict[str, float]:
    """返回 {指标名: 数值}。TODO：随各模块实现补充。"""
    raise NotImplementedError
