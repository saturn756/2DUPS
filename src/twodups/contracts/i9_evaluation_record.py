"""I9 EvaluationRecord（边 E17，运行级产物）。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Failure:
    sample_ref: str
    failure_tag: str
    cause_module: str


@dataclass
class EvaluationRecord:
    """任一总体指标可追溯到模块、切片与失败样本；检测与跟踪子块指标必须分开统计。"""

    INTERFACE = "I9"

    run_id: str
    module_ref: str
    sub_block: str | None = None               # 检测 / 跟踪 必须可分
    slice: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    implementation_ref: str | None = None
    config_ref: str | None = None
