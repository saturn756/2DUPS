"""M0 数据接入与标准化的入口。

职责与边界：docs/architecture/02_模块规范.md
输入输出：数据源 → I2, I1（接口见 docs/interface/04_接口规范.md）
配置：configs/m0_data_ingest.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from ...contracts.i1_dataset_manifest import DatasetManifest
from ...data.adapters.bdd100k import BDD100KAdapter
from ...data.bdd100k_loader import LoadedFrame
from ...data.manifest import validate
from ...utils.config import REPO_ROOT
from ..base import ModuleContext, ModuleRun


class M0DataIngest:
    NAME = "M0"
    INTERFACE_IN = ('数据源',)
    INTERFACE_OUT = ('I2', 'I1')

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def adapter(self, manifest: DatasetManifest, *, root: str | Path = REPO_ROOT) -> BDD100KAdapter:
        """Create the configured dataset adapter without coupling later modules to BDD100K."""
        if manifest.source != "BDD100K":
            raise ValueError(f"No M0 adapter is registered for {manifest.source!r}")
        adapter = BDD100KAdapter(manifest, root=root)
        problems = validate(manifest) + adapter.validate(manifest)
        if problems:
            raise ValueError("Invalid M0 manifest: " + "; ".join(problems))
        return adapter

    def iter_sample(self, manifest: DatasetManifest, sample_id: str, *,
                    root: str | Path = REPO_ROOT,
                    max_frames: int | None = None) -> Iterator[LoadedFrame]:
        """Yield the normalized I2 plus RGB runtime carrier for one sequence."""
        yield from self.adapter(manifest, root=root).iter_sample(sample_id, max_frames=max_frames)

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        manifest = inputs[0] if inputs else ctx.manifest
        if not isinstance(manifest, DatasetManifest):
            raise TypeError("M0.run expects DatasetManifest or ctx.manifest")
        self.adapter(manifest, root=self.config.get("root", REPO_ROOT))
        return ModuleRun(outputs={"DatasetManifest": manifest})
