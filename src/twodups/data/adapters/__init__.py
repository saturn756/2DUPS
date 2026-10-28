"""Select one of the implemented M0 readers explicitly."""
from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter
from .bdd100k import BDD100KAdapter
from .kitti_road import KittiRoadAdapter
from ...contracts.i1_dataset_manifest import DatasetManifest
from ...utils.config import REPO_ROOT


def build(manifest: DatasetManifest, *, root: str | Path = REPO_ROOT) -> DatasetAdapter:
    if manifest.source == "BDD100K":
        return BDD100KAdapter(manifest, root=root)
    if manifest.source == "KITTI_ROAD":
        return KittiRoadAdapter(manifest, root=root)
    raise ValueError(f"No M0 adapter is available for {manifest.source!r}")
