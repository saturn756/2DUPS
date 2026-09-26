"""模块 M4 多目标检测与跟踪 的入口。

职责与边界：docs/architecture/02_模块规范.md
输入输出：I3, I4(可选) → I5, I6（接口见 docs/interface/04_接口规范.md）
配置：configs/m4_detection_tracking.yaml
"""
from __future__ import annotations

from typing import Any

from ..base import ModuleContext, ModuleRun


class M4DetectionTracking:
    NAME = "M4"
    INTERFACE_IN = ('I3', 'I4(可选)')
    INTERFACE_OUT = ('I5', 'I6')

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        raise NotImplementedError(
            "M4 尚未实现；见 docs/architecture/02_模块规范.md"
        )
