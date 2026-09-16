"""配置加载：模块配置与 common.yaml 组合。"""
from __future__ import annotations

import copy
import pathlib
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_COMMON = REPO_ROOT / "configs" / "common.yaml"


def load(path: str | pathlib.Path) -> dict[str, Any]:
    """读取一个 YAML 配置。"""
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}


def load_module(module_config: str | pathlib.Path,
                common: str | pathlib.Path = DEFAULT_COMMON) -> dict[str, Any]:
    """公共项 + 模块项合并；模块项优先。"""
    merged = copy.deepcopy(load(common))
    merged.update(load(module_config))
    return merged
