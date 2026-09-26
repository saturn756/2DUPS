"""I9 EvaluationRecord: run-level metrics and failures."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Failure:
    sample_ref: str
    failure_tag: str
    cause_module: str


@dataclass
class EvaluationSlice:
    name: str
    tags: list[str] = field(default_factory=list)


@dataclass
class EvaluationRecord:
    """任一总体指标可追溯到模块、切片与失败样本；检测与跟踪子块指标必须分开统计。"""

    INTERFACE = "I9"

    run_id: str
    module_ref: str
    implementation_ref: str
    manifest_ref: str
    evaluation_config_ref: str
    producer_ref: str
    config_ref: str
    sub_block: str | None = None               # 检测 / 跟踪 必须可分
    sequence_id: str | None = None
    slice: EvaluationSlice | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    schema_version: str = "1.0"
