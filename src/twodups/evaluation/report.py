"""汇总评测结果，产出 EvaluationRecord（I9）。"""
from __future__ import annotations

from typing import Any

from ..contracts.i9_evaluation_record import EvaluationRecord


def build_record(run_id: str, module_ref: str, metrics: dict[str, float], **kwargs) -> EvaluationRecord:
    """构造一条评测记录。TODO：接入失败标签与样本引用。"""
    return EvaluationRecord(run_id=run_id, module_ref=module_ref, metrics=metrics, **kwargs)


def dump(records: list[EvaluationRecord], path: str) -> None:
    """把评测记录写到 outputs/ 下。TODO：选定序列化格式（jsonl）。"""
    raise NotImplementedError
