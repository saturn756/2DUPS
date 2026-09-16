"""模块 M5 语义分割 的入口。

职责与边界：docs/architecture/02_模块规范.md
输入输出：I3 → I7（接口见 docs/interface/04_接口规范.md）
配置：configs/m5_segmentation.yaml
"""
from __future__ import annotations

from typing import Any

from ..base import ModuleContext, ModuleRun


class M5Segmentation:
    NAME = "M5"
    INTERFACE_IN = ('I3',)
    INTERFACE_OUT = ('I7',)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        raise NotImplementedError(
            "M5 待阶段五实现：见 docs/architecture/05_实施规划.md（S5）"
        )
