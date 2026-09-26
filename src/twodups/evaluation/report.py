"""汇总评测结果，产出 EvaluationRecord（I9）。"""
from __future__ import annotations

from typing import Any

from ..contracts.i9_evaluation_record import EvaluationRecord


def build_record(run_id: str, module_ref: str, metrics: dict[str, float], *,
                 implementation_ref: str, manifest_ref: str, evaluation_config_ref: str,
                 producer_ref: str, config_ref: str, **kwargs) -> EvaluationRecord:
    """Construct an I9 record only when its provenance is supplied."""
    return EvaluationRecord(run_id=run_id, module_ref=module_ref, metrics=metrics,
                            implementation_ref=implementation_ref, manifest_ref=manifest_ref,
                            evaluation_config_ref=evaluation_config_ref, producer_ref=producer_ref,
                            config_ref=config_ref, **kwargs)


def dump(records: list[EvaluationRecord], path: str) -> None:
    """把评测记录写到 outputs/ 下。TODO：选定序列化格式（jsonl）。"""
    raise NotImplementedError
