"""M0 dataset adapter registry.

Adapters are runtime infrastructure, so they use a dedicated explicit registry
instead of ``pipeline.registry`` (which intentionally starts empty).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import DatasetAdapter  # noqa: F401
from .bdd100k import BDD100KAdapter, build_bdd100k_adapter
from .bdd100k_image import BddImageAdapter, build_bdd100k_image_adapter
from .image_sequence import ImageSequenceAdapter, build_image_sequence_adapter
from .kitti_road import KittiRoadAdapter, SingleImageAdapter, build_kitti_road_adapter
from ...contracts.i1_dataset_manifest import DatasetManifest


class AdapterNotRegistered(ValueError):
    """The manifest source has no M0 adapter."""


SOURCE_TO_ADAPTER = {
    "BDD100K": "bdd100k",
    "BDD100K_IMAGE": "bdd100k_image",
    "image_sequence": "image_sequence",
    "KITTI_ROAD": "kitti_road",
}

ADAPTERS: dict[str, Callable[..., DatasetAdapter]] = {
    "bdd100k": build_bdd100k_adapter,
    "bdd100k_image": build_bdd100k_image_adapter,
    "image_sequence": build_image_sequence_adapter,
    "kitti_road": build_kitti_road_adapter,
}


def registered() -> list[str]:
    return sorted(ADAPTERS)


def adapter_name(source: str) -> str:
    try:
        return SOURCE_TO_ADAPTER[source]
    except KeyError as exc:
        available = ", ".join(registered()) or "none"
        raise AdapterNotRegistered(
            f"No M0 adapter is registered for {source!r}; available adapters: {available}") from exc


def build(manifest: DatasetManifest, *, root: str | Path = ".") -> DatasetAdapter:
    name = adapter_name(manifest.source)
    return ADAPTERS[name](manifest, root=root)


__all__ = [
    "ADAPTERS", "SOURCE_TO_ADAPTER", "AdapterNotRegistered", "DatasetAdapter",
    "BDD100KAdapter", "BddImageAdapter", "ImageSequenceAdapter", "KittiRoadAdapter",
    "SingleImageAdapter", "adapter_name", "build", "registered",
]
