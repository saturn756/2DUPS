"""M4 tracking-by-detection 的最小可运行实现。

第一版使用同类别 bbox 的贪心 IoU 关联，不依赖标定或 M3 几何。
它的目的是真正建立 I5 → I6 的模块边界；后续可替换为 ByteTrack。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...contracts.base import BBox, Status, TrackState
from ...contracts.i4_match_result import MatchResult
from ...contracts.i5_detection_set import DetectionSet
from ...contracts.i6_track_set import Association, Track, TrackSet
from ...utils.config import REPO_ROOT, load_module
from ..base import ModuleContext, ModuleRun


def _iou(left: BBox, right: BBox) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    return intersection / (left_area + right_area - intersection)


@dataclass
class _TrackState:
    track_id: str
    class_ref: str
    bbox: BBox
    confidence: float
    age: int = 0
    missed_frames: int = 0
    last_detection_id: str | None = None


class M4Tracking:
    NAME = "M4"
    INTERFACE_IN = ("I5", "I4(可选)")
    INTERFACE_OUT = ("I6",)

    def __init__(self, config: dict[str, Any] | None = None,
                 *, config_ref: str = "configs/m4_tracking.yaml") -> None:
        self.config = config if config is not None else load_module(REPO_ROOT / config_ref)
        self.config_ref = config_ref
        params = self.config.get("params", {})
        self.match_iou_threshold = float(params.get("match_iou_threshold", 0.2))
        self.max_lost_frames = int(params.get("max_lost_frames", 30))
        if not 0 <= self.match_iou_threshold <= 1 or self.max_lost_frames < 0:
            raise ValueError("Invalid M4 tracking parameters")
        self._sequence_id: str | None = None
        self._variant_id: str | None = None
        self._coordinate_space: str | None = None
        self._tracks: dict[str, _TrackState] = {}
        self._next_id = 0

    def reset(self) -> None:
        self._sequence_id = None
        self._variant_id = None
        self._coordinate_space = None
        self._tracks.clear()
        self._next_id = 0

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        if not inputs or not isinstance(inputs[0], DetectionSet):
            raise TypeError("M4.run expects DetectionSet and optional MatchResult")
        match = inputs[1] if len(inputs) > 1 else None
        if match is not None and not isinstance(match, MatchResult):
            raise TypeError("M4 optional input must be MatchResult")
        result = self.update(inputs[0], match_result=match)
        return ModuleRun(outputs={"TrackSet": result},
                         degraded=[] if result.status is Status.OK else [result.reason or "tracking_failed"])

    def update(self, detections: DetectionSet,
               *, match_result: MatchResult | None = None) -> TrackSet:
        if self._sequence_id != detections.sequence_id:
            self.reset()
            self._sequence_id = detections.sequence_id

        variant_switched = (
            self._variant_id is not None and
            (self._variant_id != detections.variant_id or
             self._coordinate_space != detections.coordinate_space)
        )
        if variant_switched:
            # A bbox in another image version/coordinate space cannot inherit an
            # identity from the previous space. Start a new association window.
            self._tracks.clear()
        self._variant_id = detections.variant_id
        self._coordinate_space = detections.coordinate_space

        if detections.status is not Status.OK:
            return TrackSet(
                sequence_id=detections.sequence_id, frame_id=detections.frame_id,
                variant_id=detections.variant_id, coordinate_space=detections.coordinate_space,
                status=Status.FAILED,
                association=Association("iou_greedy", False, reason="detection_failed"),
                producer_ref="m4.iou_tracker", config_ref=self.config_ref,
                detection_missing=True, reason="detection_failed",
            )

        unmatched = set(self._tracks)
        assigned: dict[str, str] = {}
        candidates = sorted(
            ((-_iou(track.bbox, detection.bbox), track_id, index)
             for track_id, track in self._tracks.items()
             for index, detection in enumerate(detections.objects)
             if track.class_ref == detection.class_ref),
            key=lambda item: item[0],
        )
        used_detections: set[int] = set()
        for negative_iou, track_id, index in candidates:
            if track_id not in unmatched or index in used_detections:
                continue
            if -negative_iou < self.match_iou_threshold:
                continue
            assigned[track_id] = detections.objects[index].detection_id
            unmatched.remove(track_id)
            used_detections.add(index)

        output: list[Track] = []
        for track_id, state in list(self._tracks.items()):
            if track_id in assigned:
                detection = next(item for item in detections.objects
                                 if item.detection_id == assigned[track_id])
                state.bbox = detection.bbox
                state.confidence = detection.score
                state.age += 1
                state.missed_frames = 0
                state.last_detection_id = detection.detection_id
                output.append(Track(track_id, state.class_ref, TrackState.TRACKED,
                                    state.bbox, state.confidence, state.age, 0,
                                    state.last_detection_id))
            else:
                state.age += 1
                state.missed_frames += 1
                if state.missed_frames <= self.max_lost_frames:
                    output.append(Track(track_id, state.class_ref, TrackState.LOST,
                                        None, None, state.age, state.missed_frames,
                                        state.last_detection_id))
                else:
                    del self._tracks[track_id]

        for index, detection in enumerate(detections.objects):
            if index in used_detections:
                continue
            track_id = f"{detections.sequence_id}:track:{self._next_id:06d}"
            self._next_id += 1
            self._tracks[track_id] = _TrackState(
                track_id, detection.class_ref, detection.bbox, detection.score,
                age=1, last_detection_id=detection.detection_id,
            )
            output.append(Track(track_id, detection.class_ref, TrackState.NEW,
                                detection.bbox, detection.score, 1, 0,
                                detection.detection_id))

        used_geometry = bool(match_result and match_result.status is Status.OK and
                             self.config.get("params", {}).get("use_geometry", False))
        if variant_switched:
            reason = "variant_switch"
        elif not (match_result and match_result.status is not Status.OK):
            reason = None
        else:
            reason = "match_unreliable"
        return TrackSet(
            sequence_id=detections.sequence_id, frame_id=detections.frame_id,
            variant_id=detections.variant_id, coordinate_space=detections.coordinate_space,
            status=Status.OK,
            association=Association("iou_greedy", used_geometry, reason=reason),
            producer_ref="m4.iou_tracker", config_ref=self.config_ref,
            tracks=output, detection_missing=False,
        )
