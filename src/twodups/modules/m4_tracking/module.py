"""I5 to I6 using pinned Ultralytics association and optional pixel GMC."""
from __future__ import annotations

from types import SimpleNamespace
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT

from ...contracts.base import BBox, Status, TrackState
from ...contracts.i4_match_result import MatchResult
from ...contracts.i5_detection_set import DetectionSet
from ...contracts.i6_track_set import Association, Track, TrackSet
from ...utils.config import REPO_ROOT, load_module
from ..base import ModuleContext, ModuleRun
from .motion import MotionCompensator


class _ClassTracker(BYTETracker):
    @staticmethod
    def reset_id():
        # Ultralytics' counter is process-global; another class/instance must not rewind it.
        pass


class _GroupBoTSORT(BOTSORT):
    @staticmethod
    def reset_id():
        pass


@dataclass
class _Observation:
    label: str
    last_id: str | None = None
    labels: deque = field(default_factory=lambda: deque(maxlen=5))
    hits: deque = field(default_factory=lambda: deque(maxlen=5))
    confirmed: bool = False


class M4Tracking:
    NAME = "M4"
    INTERFACE_IN = ("I5", "I4(可选)")
    INTERFACE_OUT = ("I6",)

    def __init__(self, config: dict[str, Any] | None = None, *,
                 config_ref: str = "configs/m4_tracking.yaml") -> None:
        self.config = config if config is not None else load_module(REPO_ROOT / config_ref)
        self.config_ref = config_ref
        params = self.config["params"]
        self.method = self.config["impl"]["association"]
        self.use_geometry = params.get("use_geometry", False)
        self.merge_vehicles = params.get("merge_vehicle_classes", False)
        self.min_confirmations = params.get("min_confirmations", 1)
        if self.method not in {"bytetrack", "botsort"} or (self.use_geometry and self.method != "botsort"):
            raise ValueError("ByteTrack runs without geometry; use botsort for GMC")
        if (not isinstance(self.use_geometry, bool) or not isinstance(self.merge_vehicles, bool) or
                isinstance(self.min_confirmations, bool) or not isinstance(self.min_confirmations, int) or
                not 1 <= self.min_confirmations <= 5):
            raise ValueError("Invalid stability parameters")
        self.args = SimpleNamespace(**{key: params[key] for key in (
            "track_high_thresh", "track_low_thresh", "new_track_thresh",
            "track_buffer", "match_thresh", "fuse_score")})
        a = self.args
        if (not 0 <= a.track_low_thresh < a.track_high_thresh <= a.new_track_thresh <= 1 or
                not 0 < a.match_thresh <= 1 or isinstance(a.track_buffer, bool) or
                not isinstance(a.track_buffer, int) or a.track_buffer < 0 or
                not isinstance(a.fuse_score, bool)):
            raise ValueError("Invalid ByteTrack parameters")
        self.args.gmc_method = "none"
        self.args.with_reid = False
        self.args.proximity_thresh = .5
        self.args.appearance_thresh = .8
        self._motion = MotionCompensator()
        self._shared_warp = SimpleNamespace(apply=lambda *args: self._warp)
        self._generation = -1
        self.reset()

    def reset(self) -> None:
        self._generation += 1
        self._sequence_id = None
        self._alignment = None
        self._last_index = None
        self._trackers = {}
        self._observations = {}
        self._motion.reset()
        self._motion_frame_id = None
        self._warp = np.eye(2, 3)
        self.last_motion = {"used": False, "reason": "gmc_disabled"}

    def _group(self, class_ref):
        namespace, _, name = class_ref.rpartition(".")
        if self.merge_vehicles and name in {"car", "truck", "bus"}:
            return f"{namespace}.vehicle" if namespace else "vehicle"
        return class_ref

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        if len(inputs) not in (2, 3, 4):
            raise TypeError("M4.run expects I5, runtime frame_index, optional I4 and aligned RGB")
        result = self.update(inputs[0], frame_index=inputs[1],
                             match_result=inputs[2] if len(inputs) >= 3 else None,
                             image_rgb=inputs[3] if len(inputs) == 4 else None)
        return ModuleRun(outputs={"TrackSet": result},
                         degraded=[] if result.status is Status.OK else [result.reason])

    def update(self, detections: DetectionSet, *, frame_index: int,
               match_result: MatchResult | None = None, image_rgb: np.ndarray | None = None) -> TrackSet:
        if not isinstance(detections, DetectionSet):
            raise TypeError("M4 requires DetectionSet")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("M4 requires a non-negative frame_index")
        if match_result is not None and not isinstance(match_result, MatchResult):
            raise TypeError("Optional geometry input must be MatchResult")
        if image_rgb is not None and (not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8 or
                image_rgb.shape != (detections.image_size.height, detections.image_size.width, 3)):
            raise ValueError("GMC requires aligned uint8 RGB pixels")
        objects = detections.objects
        ids = [obj.detection_id for obj in objects]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate detection IDs")
        if any(not obj.class_ref or not obj.detection_id for obj in objects):
            raise ValueError("Detection class and ID must be present")
        if any(not (0 <= obj.bbox.x1 < obj.bbox.x2 <= detections.image_size.width and
                    0 <= obj.bbox.y1 < obj.bbox.y2 <= detections.image_size.height)
               for obj in objects):
            raise ValueError("Detection bbox is outside the image")

        if self._sequence_id != detections.sequence_id:
            self.reset()
            self._sequence_id = detections.sequence_id
        if self._last_index is not None and frame_index != self._last_index + 1:
            raise ValueError("M4 requires consecutive frames; reset before seeking")
        alignment = (detections.variant_id, detections.coordinate_space,
                     detections.image_size, detections.class_schema_ref)
        reason = "match_missing" if match_result is None else "geometry_disabled"
        if match_result is not None and match_result.status is not Status.OK:
            reason = "match_unreliable"
        if self._alignment is not None and self._alignment != alignment:
            self._generation += 1
            self._trackers.clear()
            self._observations.clear()
            self._motion.reset()
            self._motion_frame_id = None
            reason = "variant_switch" if self._alignment[:2] != alignment[:2] else "input_schema_or_size_switch"
        reset_reason = reason if reason in {"variant_switch", "input_schema_or_size_switch"} else None
        self.last_motion = {"used": False, "reason": "gmc_disabled"}
        if self.use_geometry:
            previous_motion_id = self._motion_frame_id
            if image_rgb is None:
                self._motion.reset()
                self._warp = np.eye(2, 3)
                self.last_motion = {"used": False, "reason": "gmc_missing_image"}
            else:
                moving = [o.bbox for o in objects if o.class_ref.split(".")[-1] not in {"traffic_light", "traffic_sign", "stop_sign"}]
                self._warp, self.last_motion = self._motion.update(image_rgb, moving)
            self.last_motion.update(frame_a=previous_motion_id, frame_b=detections.frame_id,
                                    variant_id=detections.variant_id)
            self._motion_frame_id = detections.frame_id if image_rgb is not None else None
            reason = reset_reason or self.last_motion["reason"]
        self._alignment = alignment
        self._last_index = frame_index

        failed = detections.status is not Status.OK
        if failed:
            reason = detections.reason or "detection_failed"
        grouped = {class_ref: [] for class_ref in self._trackers}
        if not failed:
            for obj in objects:
                grouped.setdefault(self._group(obj.class_ref), []).append(obj)
        output = []
        retained = {}
        for class_ref, current in grouped.items():
            if class_ref not in self._trackers:
                # frame_rate=30 makes track_buffer an exact count of input frames.
                self._trackers[class_ref] = (_ClassTracker if self.method == "bytetrack" else _GroupBoTSORT)(self.args, frame_rate=30)
                if self.use_geometry:
                    # Estimate once per frame, then reuse the same warp for every association group.
                    self._trackers[class_ref].gmc = self._shared_warp
            backend = self._trackers[class_ref]
            data = np.array([[o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2, o.score, 0]
                             for o in current], dtype=np.float32).reshape(-1, 6)
            backend.update(Boxes(data, (detections.image_size.height, detections.image_size.width)),
                           img=image_rgb if self.use_geometry else None)
            backend.lost_stracks = [t for t in backend.lost_stracks
                                    if backend.frame_id - t.end_frame <= self.args.track_buffer]
            high = [o for o, row in zip(current, data) if row[4] >= self.args.track_high_thresh]
            low = [o for o, row in zip(current, data)
                   if self.args.track_low_thresh < row[4] < self.args.track_high_thresh]
            for state in backend.tracked_stracks + backend.lost_stracks:
                missed = backend.frame_id - state.end_frame
                if not state.is_activated or missed > self.args.track_buffer:
                    continue
                track_id = f"{detections.sequence_id}:track:{self._generation}:{class_ref}:{state.track_id}"
                observation = self._observations.get(track_id)
                if missed:
                    if observation is None:
                        continue
                    observation.hits.append(False)
                    track = Track(track_id, observation.label, TrackState.LOST, None, None,
                                  backend.frame_id - state.start_frame + 1, missed, observation.last_id)
                else:
                    pool = high if state.score >= self.args.track_high_thresh else low
                    detection = pool[int(state.idx)]
                    first = observation is None
                    if first:
                        observation = _Observation(detection.class_ref)
                    observation.labels.append((detection.class_ref, detection.score))
                    scores = Counter()
                    votes = Counter()
                    for label, score in observation.labels:
                        scores[label] += score
                        votes[label] += 1
                    winner = scores.most_common(1)[0][0]
                    if votes[winner] >= 3:
                        observation.label = winner
                    observation.last_id = detection.detection_id
                    observation.hits.append(detection.score >= self.args.track_high_thresh)
                    observation.confirmed |= sum(observation.hits) >= self.min_confirmations
                    track = Track(track_id, observation.label,
                                  TrackState.NEW if first or not observation.confirmed else TrackState.TRACKED,
                                  BBox(*(float(x) for x in state.xyxy)), float(state.score),
                                  backend.frame_id - state.start_frame + 1, 0, observation.last_id)
                retained[track_id] = observation
                output.append(track)
        self._observations = retained
        return TrackSet(sequence_id=detections.sequence_id, frame_id=detections.frame_id,
                        variant_id=detections.variant_id, coordinate_space=detections.coordinate_space,
                        status=Status.FAILED if failed else Status.OK,
                        association=Association(self.method, self.last_motion["used"], reason=reason),
                        producer_ref=f"m4.ultralytics_{self.method}", config_ref=self.config_ref,
                        tracks=output, detection_missing=failed, reason=reason if failed else None)
