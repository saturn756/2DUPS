"""可选帧间几何扩展的入口，不属于课程主模块编号。

职责与边界：docs/architecture/02_模块规范.md
输入输出：I3 → I4（接口见 docs/interface/04_接口规范.md）
配置：configs/optional_geometry.yaml
"""
from __future__ import annotations

from typing import Any

from ..base import ModuleContext, ModuleRun


class OptionalGeometry:
    NAME = "OPTIONAL_GEOMETRY"
    INTERFACE_IN = ('I3',)
    INTERFACE_OUT = ('I4',)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        raise NotImplementedError(
            "可选帧间几何尚未实现；见 docs/architecture/02_模块规范.md"
        )
