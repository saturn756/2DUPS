"""模块 M6 场景理解与拓扑推理 的入口。

职责与边界：docs/architecture/02_模块规范.md
输入输出：I6, I7, I4(可选) → I8（接口见 docs/interface/04_接口规范.md）
配置：configs/m6_topology.yaml
"""
from __future__ import annotations

from typing import Any

from ..base import ModuleContext, ModuleRun


class M6Topology:
    NAME = "M6"
    INTERFACE_IN = ('I6', 'I7', 'I4(可选)')
    INTERFACE_OUT = ('I8',)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        raise NotImplementedError(
            "M6 待阶段五实现：见 docs/architecture/05_实施规划.md（S5）"
        )
