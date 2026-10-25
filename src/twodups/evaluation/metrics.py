"""模块级指标。

按模块的验收维度独立统计；检测与跟踪子块必须分开：
  M0 接入完整率；M1 质量指标与下游指标差；M2 检测 mAP/AP-S；
  M3 mIoU、边界与细长结构；M4 跟踪 MOTA/IDF1/ID Switch；
  M5 关系准确率、图合法率与降级判定准确性。
"""
from __future__ import annotations


def compute(module_ref: str, predictions, targets, **kwargs) -> dict[str, float]:
    """返回 {指标名: 数值}。TODO：随各模块实现补充。"""
    raise NotImplementedError
