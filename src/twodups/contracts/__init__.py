"""接口契约（I1–I9）：字段、状态与不变量。

唯一来源：docs/interface/04_接口规范.md。任何模块之间传递的数据都必须能用这里的类型表达。
"""
from .base import (  # noqa: F401
    CalibrationState, CalibrationStatus, GeometryStatus, Producer, Ref, Status, TrackState,
)
from .i1_dataset_manifest import DatasetManifest, Sample  # noqa: F401
from .i2_calibrated_frame import CalibratedFrame  # noqa: F401
from .i3_image_variant import ImageVariant, ImageVariantSet, QualityReport  # noqa: F401
from .i4_match_result import FramePair, MatchResult, Transform  # noqa: F401
from .i5_detection_set import Detection, DetectionSet  # noqa: F401
from .i6_track_set import Track, TrackSet  # noqa: F401
from .i7_segmentation_map import SegmentationMap  # noqa: F401
from .i8_scene_graph import Edge, Node, SceneGraph2D  # noqa: F401
from .i9_evaluation_record import EvaluationRecord, Failure  # noqa: F401

INTERFACES = ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")
