"""I1 DatasetManifest（边 E03）：运行级前提，先于任何样本处理。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import Ref


@dataclass
class Sample:
    sample_id: str
    sequence_id: str
    media_ref: Ref
    timestamp: float | None = None
    annotation_ref: Ref | None = None
    slice_tags: list[str] = field(default_factory=list)


@dataclass
class DatasetManifest:
    """样本必须可追溯到来源与许可；切片标签必须显式，禁止按文件名推断。"""

    INTERFACE = "I1"

    manifest_id: str
    version: str
    source: str
    license: str
    samples: list[Sample] = field(default_factory=list)
    splits: dict[str, list[str]] = field(default_factory=dict)
    checksum: str | None = None
    schema_version: str = "1.0"
    producer_ref: str = "m1.dataset_manifest"
    config_ref: str = "configs/dataset_manifest.yaml"
