"""BDD100K 适配器。

BDD100KVideoLoader 是具体的解码实现；该包装器让 M0 可以通过统一适配器名
发现它，未来新增 KITTI/Cityscapes/CULane/MOT 时不修改下游模块。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ...contracts.i1_dataset_manifest import DatasetManifest
from ..bdd100k_loader import BDD100KVideoLoader, LoadedFrame


class BDD100KAdapter:
    name = "bdd100k"

    def __init__(self, manifest: DatasetManifest, root: str | Path) -> None:
        self.loader = BDD100KVideoLoader(manifest, root=root)

    def iter_sample(self, sample_id: str, *, max_frames: int | None = None) -> Iterator[LoadedFrame]:
        yield from self.loader.iter_sample(sample_id, max_frames=max_frames)

    def validate(self, manifest: DatasetManifest) -> list[str]:
        return [] if manifest.source == "BDD100K" else ["manifest source is not BDD100K"]
