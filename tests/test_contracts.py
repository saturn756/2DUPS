"""接口契约自检：字段齐全、状态枚举完整、默认值符合规范。"""
from __future__ import annotations

from dataclasses import fields

from twodups import contracts
from twodups.contracts.base import CalibrationStatus, GeometryStatus, Status, TrackState
import pytest


def test_interfaces_registered() -> None:
    assert contracts.INTERFACES == ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")


def _graph() -> contracts.SceneGraph2D:
    return contracts.SceneGraph2D(
        sequence_id="s", frame_id="s:000000", variant_id="raw", coordinate_space="pixel",
        window=contracts.FrameWindow(frame_ids=["s:000000"]),
        producer_ref="m6.rule_based_scene_graph", config_ref="configs/m6_topology.yaml",
    )


def test_scene_graph_defaults_to_image_plane() -> None:
    graph = _graph()
    assert graph.geometry_status is GeometryStatus.IMAGE_PLANE
    assert graph.missing_inputs == []
    assert graph.edges == []
    with pytest.raises(ValueError, match="degraded"):
        contracts.SceneGraph2D(
            sequence_id="s", frame_id="s:000000", variant_id="raw", coordinate_space="pixel",
            window=contracts.FrameWindow(["s:000000"]), producer_ref="m6.rules",
            config_ref="configs/m6_topology.yaml", missing_inputs=["I7"],
        )


def test_scene_graph_edge_carries_evidence() -> None:
    edge = contracts.Edge(src="t1", dst="r1", relation="belongs_to",
                          evidence=[contracts.Evidence(type="mask_overlap", source="I6+I7")],
                          confidence=0.8)
    assert edge.evidence and edge.relation == "belongs_to"
    with pytest.raises(ValueError):
        contracts.Edge(src="t1", dst="r1", relation="front_behind",
                       evidence=edge.evidence, confidence=0.8)


def test_track_set_states_are_explicit() -> None:
    track = contracts.Track(track_id="t1", class_ref="car", state=TrackState.UNRELIABLE)
    assert {s.value for s in TrackState} == {"tracked", "lost", "new", "unreliable"}
    assert track.state is TrackState.UNRELIABLE
    with pytest.raises(ValueError):
        contracts.Track(track_id="lost", class_ref="car", state=TrackState.LOST,
                        current_bbox=contracts.BBox(0, 0, 1, 1))


def test_match_result_has_status_and_producer() -> None:
    names = {f.name for f in fields(contracts.MatchResult)}
    assert {"sequence_id", "pair", "status", "producer_ref", "config_ref"} <= names
    assert {s.value for s in Status} == {"ok", "unreliable", "failed"}
    with pytest.raises(ValueError):
        contracts.Transform(type="homography", semantics="metric_motion")
    with pytest.raises(ValueError, match="requires a transform"):
        contracts.MatchResult(
            sequence_id="s", pair=contracts.FramePair("s:0", "s:1", "raw", "pixel"),
            status=Status.OK, producer_ref="m3.geometry", config_ref="configs/m3_geometry.yaml",
        )


def test_image_variant_set_requires_raw_variant() -> None:
    variants = contracts.ImageVariantSet(
        sequence_id="s", frame_id="f0",
        variants=[contracts.ImageVariant(variant_id="raw", kind="raw",
                                         media_ref=contracts.Ref(uri="data/raw/f0.png"))],
        default_variant="raw",
        image_size=contracts.ImageSize(2, 2), coordinate_space="pixel",
        calibration_status=CalibrationStatus.MISSING,
    )
    assert variants.get("raw").kind == "raw"


def test_calibration_status_three_states() -> None:
    assert {s.value for s in CalibrationStatus} == {"valid", "missing", "stale"}


def test_detection_failure_cannot_claim_objects() -> None:
    with pytest.raises(ValueError):
        contracts.DetectionSet(
            sequence_id="s", frame_id="s:0", variant_id="raw", coordinate_space="pixel",
            image_size=contracts.ImageSize(10, 10), status=Status.FAILED,
            class_schema_ref="schema.json", producer_ref="m4.detector", config_ref="configs/m4.yaml",
            objects=[contracts.Detection("d1", "car", contracts.BBox(1, 1, 2, 2), 0.9)],
        )


def test_resized_segmentation_requires_alignment() -> None:
    with pytest.raises(ValueError):
        contracts.SegmentationMap(
            sequence_id="s", frame_id="s:0", variant_id="raw", coordinate_space="pixel",
            source_size=contracts.ImageSize(1280, 720), map_size=contracts.ImageSize(640, 360),
            class_map_ref=contracts.Ref("mask.png"), class_schema_ref="schema.json",
            alignment_ref=None, status=Status.OK, producer_ref="m5.segmenter",
            config_ref="configs/m5_segmentation.yaml",
        )


def test_i3_rejects_missing_raw_or_unknown_default() -> None:
    with pytest.raises(ValueError):
        contracts.ImageVariantSet(sequence_id="s", frame_id="f0", variants=[], default_variant="raw",
                                  image_size=contracts.ImageSize(2, 2), coordinate_space="pixel",
                                  calibration_status=CalibrationStatus.MISSING)
    with pytest.raises(ValueError, match="valid calibration"):
        contracts.ImageVariantSet(
            sequence_id="s", frame_id="s:0", image_size=contracts.ImageSize(2, 2),
            coordinate_space="pixel", calibration_status=CalibrationStatus.MISSING,
            default_variant="rectified", variants=[
                contracts.ImageVariant("raw", "raw", contracts.Ref("raw.png")),
                contracts.ImageVariant("rectified", "rectified", contracts.Ref("rectified.png"),
                                       parent_variant_id="raw"),
            ],
        )
