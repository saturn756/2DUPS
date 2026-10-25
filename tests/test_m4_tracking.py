"""M4 I5→I6 tracking-by-detection tests."""
from __future__ import annotations

from twodups.contracts import BBox, Detection, DetectionSet, ImageSize, Status, TrackState
from twodups.modules.m4_tracking import M4Tracking


def _detections(frame: int, boxes: list[tuple[float, float, float, float]],
                variant_id: str = "raw") -> DetectionSet:
    return DetectionSet(
        sequence_id="seq", frame_id=f"seq:{frame:06d}", variant_id=variant_id,
        coordinate_space="pixel", image_size=ImageSize(100, 100), status=Status.OK,
        class_schema_ref="test", producer_ref="m2.test", config_ref="configs/m2_detection.yaml",
        objects=[Detection(f"d{frame}-{index}", "coco.car", BBox(*box), 0.9)
                 for index, box in enumerate(boxes)],
    )


def test_iou_tracker_keeps_id_for_overlapping_detection() -> None:
    tracker = M4Tracking({"params": {"match_iou_threshold": 0.2, "max_lost_frames": 2}})
    first = tracker.update(_detections(0, [(10, 10, 30, 30)]))
    second = tracker.update(_detections(1, [(11, 11, 31, 31)]))
    assert first.tracks[0].state is TrackState.NEW
    assert second.tracks[0].state is TrackState.TRACKED
    assert second.tracks[0].track_id == first.tracks[0].track_id


def test_iou_tracker_marks_missing_object_lost_without_stale_bbox() -> None:
    tracker = M4Tracking({"params": {"match_iou_threshold": 0.2, "max_lost_frames": 2}})
    first = tracker.update(_detections(0, [(10, 10, 30, 30)]))
    second = tracker.update(_detections(1, []))
    assert second.tracks[0].track_id == first.tracks[0].track_id
    assert second.tracks[0].state is TrackState.LOST
    assert second.tracks[0].current_bbox is None


def test_tracker_resets_ids_on_sequence_change() -> None:
    tracker = M4Tracking({"params": {"match_iou_threshold": 0.2}})
    first = tracker.update(_detections(0, [(10, 10, 30, 30)]))
    next_sequence = _detections(0, [(10, 10, 30, 30)])
    next_sequence.sequence_id = "other"
    second = tracker.update(next_sequence)
    assert first.tracks[0].track_id != second.tracks[0].track_id


def test_variant_switch_does_not_reuse_previous_identity() -> None:
    tracker = M4Tracking({"params": {"match_iou_threshold": 0.2}})
    first = tracker.update(_detections(0, [(10, 10, 30, 30)], "raw"))
    second = tracker.update(_detections(1, [(11, 11, 31, 31)], "enhanced"))
    assert second.association.reason == "variant_switch"
    assert first.tracks[0].track_id != second.tracks[0].track_id
