"""模块 M2 图像预处理与质量评价 的入口。

职责与边界：docs/architecture/02_模块规范.md
输入输出：I2 → I3（接口见 docs/interface/04_接口规范.md）
配置：configs/m2_preprocess.yaml
"""
from __future__ import annotations

from typing import Any

from ..base import ModuleContext, ModuleRun


class M2Preprocess:
    NAME = "M2"
    INTERFACE_IN = ('I2',)
    INTERFACE_OUT = ('I3',)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        raise NotImplementedError(
            "M2 待阶段三实现：见 docs/architecture/05_实施规划.md（S2）"
        )
