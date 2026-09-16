"""模块基类与运行上下文。模块之间只通过 contracts 中的接口通信。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..contracts.base import Producer


@dataclass
class ModuleContext:
    """模块运行上下文：配置、实现与配置标识、数据清单。"""

    config: dict[str, Any]
    producer: Producer
    manifest: Any | None = None


@dataclass
class ModuleRun:
    """一次模块调用的输出：接口名 → 对象（I1–I9 中的类型）。"""

    outputs: dict[str, Any] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


@runtime_checkable
class Module(Protocol):
    NAME: str
    INTERFACE_IN: tuple[str, ...]
    INTERFACE_OUT: tuple[str, ...]

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun: ...
