"""M0 数据适配器的最小协议。"""
from __future__ import annotations

from typing import Iterator, Protocol

from ..bdd100k_loader import LoadedFrame
from ...contracts.i1_dataset_manifest import DatasetManifest


class DatasetAdapter(Protocol):
    name: str

    def iter_sample(self, sample_id: str, *, max_frames: int | None = None) -> Iterator[LoadedFrame]:
        """把一个原始序列转换为统一的运行时 I2 + RGB 载体。"""

    def validate(self, manifest: DatasetManifest) -> list[str]:
        """检查数据集清单，不应在这里执行模型推理。"""
