"""接口契约自检：字段齐全、状态枚举完整、默认值符合规范。"""
from __future__ import annotations

from dataclasses import fields

from twodups import contracts
from twodups.contracts.base import CalibrationStatus, GeometryStatus, Status, TrackState


def test_interfaces_registered() -> None:
    assert contracts.INTERFACES == ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9")


def test_scene_graph_defaults_to_full_geometry() -> None:
    graph = contracts.SceneGraph2D(frame_id="f0")
    assert graph.geometry_status is GeometryStatus.FULL
    assert graph.missing_inputs == []
    assert graph.edges == []


def test_scene_graph_edge_carries_evidence() -> None:
    edge = contracts.Edge(src="t1", dst="r1", relation="membership", evidence=["bbox_overlap"])
    assert edge.evidence and edge.relation == "membership"


def test_track_set_states_are_explicit() -> None:
    track = contracts.Track(track_id="t1", class_ref="car", state=TrackState.UNRELIABLE)
    assert {s.value for s in TrackState} == {"tracked", "lost", "new", "unreliable"}
    assert track.state is TrackState.UNRELIABLE


def test_match_result_has_status_and_producer() -> None:
    names = {f.name for f in fields(contracts.MatchResult)}
    assert {"pair", "status", "producer"} <= names
    assert {s.value for s in Status} == {"ok", "unreliable", "failed"}


def test_image_variant_set_requires_raw_variant() -> None:
    variants = contracts.ImageVariantSet(
        frame_id="f0",
        variants=[contracts.ImageVariant(variant_id="raw", kind="raw",
                                         media_ref=contracts.Ref(uri="data/raw/f0.png"))],
        default_variant="raw",
    )
    assert variants.get("raw").kind == "raw"


def test_calibration_status_three_states() -> None:
    assert {s.value for s in CalibrationStatus} == {"valid", "missing", "stale"}
