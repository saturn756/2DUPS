"""离线链路编排。

顺序：输入 → M1 → M2 →｛M3 ∥ M4 ∥ M5｝→ M6 → 输出。
对齐规则（见 docs/dataflow/03_数据流与连接逻辑.md）：
  · 三条支路必须使用同一 variant_id；
  · 缺失输入只能等待或降级，不得用其它帧或其它版本顶替；
  · M3 消费 pair(t-1, t)，其结果只能用于第 t 帧及之后。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    frame_id: str
    scene_graph: Any | None = None          # SceneGraph2D（I8）
    degraded: list[str] = field(default_factory=list)


def run(config: dict[str, Any]) -> list[RunResult]:
    """按配置跑完整链路。TODO：按实施规划 S1–S6 逐模块接通。"""
    raise NotImplementedError(
        "链路尚未接通：先完成 M1、M2（docs/architecture/05_实施规划.md 的 S1–S2）"
    )
