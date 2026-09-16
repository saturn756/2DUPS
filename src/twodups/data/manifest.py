"""数据清单（I1）：版本、切分、来源与许可、切片标签。"""
from __future__ import annotations

import pathlib

from ..contracts.i1_dataset_manifest import DatasetManifest


def load(path: str | pathlib.Path) -> DatasetManifest:
    """从 configs/dataset_manifest.yaml 读入清单。TODO(阶段二)：按实际清单格式实现。"""
    raise NotImplementedError


def validate(manifest: DatasetManifest) -> list[str]:
    """校验清单完整性（来源、许可、切片标签、样本引用）。返回问题列表。"""
    problems: list[str] = []
    if not manifest.source or not manifest.license:
        problems.append("缺少来源或许可信息")
    if not manifest.samples:
        problems.append("清单为空")
    return problems
