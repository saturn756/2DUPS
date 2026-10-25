"""M5 系统集成与场景理解的入口。

职责与边界：docs/architecture/02_模块规范.md
输入输出：I5, I6, I7, I4(可选) → I8（接口见 docs/interface/04_接口规范.md）
配置：configs/m5_integration.yaml
"""
from __future__ import annotations

from typing import Any

from ..base import ModuleContext, ModuleRun


class M5Integration:
    NAME = "M5"
    INTERFACE_IN = ('I5', 'I6', 'I7', 'I4(可选)')
    INTERFACE_OUT = ('I8',)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        raise NotImplementedError(
            "M5 尚未实现；见 docs/architecture/02_模块规范.md"
        )
