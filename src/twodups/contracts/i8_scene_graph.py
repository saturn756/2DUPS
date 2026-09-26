"""I8 SceneGraph2D: auditable 2D relations, never metric depth or risk."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .base import GeometryStatus


RELATIONS = frozenset({"belongs_to", "inside_region", "left_of", "right_of", "overlaps", "appears_in"})


@dataclass
class Node:
    node_id: str
    type: str                                  # track / region
    ref: str
    class_ref: str

    def __post_init__(self) -> None:
        if self.type not in {"track", "region"}:
            raise ValueError("I8 supports track and region nodes only")


@dataclass
class Evidence:
    type: str
    source: str
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.type or not self.source:
            raise ValueError("I8 evidence needs type and source")


@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    evidence: list[Evidence]
    confidence: float

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"Unsupported 2D relation: {self.relation}")
        if not self.evidence:
            raise ValueError("I8 edge requires evidence")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("I8 edge confidence must be within [0, 1]")


@dataclass
class FrameWindow:
    frame_ids: list[str]
    uses_future_frames: bool = False

    def __post_init__(self) -> None:
        if not self.frame_ids:
            raise ValueError("I8 window must contain frames")
        if self.uses_future_frames:
            raise ValueError("I8 cannot use future frames")


@dataclass
class SceneGraph2D:
    """只输出证据充分的关系；geometry_status 必须真实反映几何可用性。"""

    INTERFACE = "I8"

    sequence_id: str
    frame_id: str
    variant_id: str
    coordinate_space: str
    window: FrameWindow
    producer_ref: str
    config_ref: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    geometry_status: GeometryStatus = GeometryStatus.IMAGE_PLANE
    missing_inputs: list[str] = field(default_factory=list)
    provenance: dict[str, str | None] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.frame_id not in self.window.frame_ids:
            raise ValueError("Current frame must occur in I8 window")
        if self.missing_inputs and self.geometry_status is not GeometryStatus.DEGRADED:
            raise ValueError("I8 missing inputs require degraded geometry status")
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate I8 node IDs")
        if any(edge.src not in ids or edge.dst not in ids for edge in self.edges):
            raise ValueError("I8 edge references an unknown node")
