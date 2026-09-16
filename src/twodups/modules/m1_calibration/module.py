"""模块 M1 采集与相机标定 的入口。

职责与边界：docs/architecture/02_模块规范.md
输入输出：外部输入 → I2, I1（接口见 docs/interface/04_接口规范.md）
配置：configs/m1_calibration.yaml
"""
from __future__ import annotations

from typing import Any

from ..base import ModuleContext, ModuleRun


class M1Calibration:
    NAME = "M1"
    INTERFACE_IN = ('外部输入',)
    INTERFACE_OUT = ('I2', 'I1')

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        raise NotImplementedError(
            "M1 待阶段二实现：见 docs/architecture/05_实施规划.md（S1）"
        )
