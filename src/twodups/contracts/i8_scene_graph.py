"""I8 SceneGraph2D（边 E10 必需，唯一对外契约）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import GeometryStatus


@dataclass
class Node:
    node_id: str
    type: str                                  # object / region / track
    ref: str


@dataclass
class Edge:
    src: str
    dst: str
    relation: str                              # membership / front_behind / left_right / adjacent / conflict
    evidence: list[str] = field(default_factory=list)
    confidence: float | None = None


@dataclass
class SceneGraph2D:
    """只输出证据充分的关系；geometry_status 必须真实反映几何可用性。"""

    INTERFACE = "I8"

    frame_id: str
    window: str | None = None                  # 帧窗口定义，须与评测切片口径一致
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    geometry_status: GeometryStatus = GeometryStatus.FULL
    missing_inputs: list[str] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)
