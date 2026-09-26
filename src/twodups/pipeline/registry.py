"""实现注册表：把配置里的 impl 名映射到具体实现（模块内部的可替换点）。

换实现不改接口：注册表让“选定实现 / 对照实现”可以只靠配置切换。
"""
from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """把实现函数登记到某个名字下（name 与配置里的 impl 字段对应）。"""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> Callable[..., Any]:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的实现：{name}（检查模块实现与 configs/ 配置）")
    return _REGISTRY[name]


def available() -> list[str]:
    """已注册的实现名，供配置校验与评测记录使用。"""
    return sorted(_REGISTRY)
