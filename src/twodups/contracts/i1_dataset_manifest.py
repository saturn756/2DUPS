"""I1 DatasetManifest: required run-level dataset identity and split."""
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
    # Image sequences have no container PTS, so their frame rate must be
    # declared in I1. Video sources may leave it unset and use the container
    # metadata instead.
    fps: float | None = None
    calibration_ref: Ref | None = None


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
    producer_ref: str = "m0.dataset_manifest"
    config_ref: str = "configs/dataset_manifest.yaml"
