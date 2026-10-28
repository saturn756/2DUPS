"""Exercise the real Ultralytics ByteTrack backend with synthetic I5 records."""
from dataclasses import replace
import pytest
import cv2
import numpy as np
from twodups.modules.m4_tracking.motion import MotionCompensator
from twodups.modules.m4_tracking.render import TrackRenderer
from twodups.contracts import BBox, Detection, DetectionSet, ImageSize, Status, TrackState
from twodups.modules.m4_tracking import M4Tracking
from twodups.utils.config import REPO_ROOT, load_module

def detections(frame, rows=(), variant="raw", sequence="seq", failed=False):
    return DetectionSet(sequence_id=sequence, frame_id=f"{sequence}:{frame:06d}",
                        variant_id=variant, coordinate_space="pixel", image_size=ImageSize(100,100),
                        status=Status.FAILED if failed else Status.OK,
                        class_schema_ref="test", producer_ref="test", config_ref="test",
                        objects=[Detection(f"d{frame}-{i}",cls,BBox(*box),score)
                                 for i,(cls,box,score) in enumerate(rows)],
                        detection_missing=failed)

CAR = ("car",(10,10,30,30),.9)

def tracker(buffer=2):
    config=load_module(REPO_ROOT/"configs/m4_tracking.yaml")
    config["params"]["track_buffer"]=buffer
    return M4Tracking(config)

def step(t, frame, rows=(), **kwargs):
    return t.update(detections(frame,rows,**kwargs),frame_index=frame)

def test_id_survives_movement_and_a_missed_frame():
    t=tracker()
    first=step(t,0,[CAR]).tracks[0]
    second=step(t,1,[("car",(11,11,31,31),.9)]).tracks[0]
    assert first.state is TrackState.NEW and second.state is TrackState.TRACKED
    assert first.track_id==second.track_id
    lost=step(t,2).tracks[0]
    assert lost.state is TrackState.LOST and lost.current_bbox is None
    assert lost.last_detection_id=="d1-0"
    restored=step(t,3,[("car",(12,12,32,32),.9)]).tracks[0]
    assert restored.track_id==first.track_id

def test_low_confidence_recovers_existing_track_but_does_not_start_one():
    t=tracker()
    first=step(t,0,[CAR]).tracks[0]
    low=step(t,1,[("car",(11,11,31,31),.15)]).tracks[0]
    assert low.track_id==first.track_id and low.last_detection_id=="d1-0"
    assert not step(tracker(),0,[("car",(10,10,30,30),.15)]).tracks

def test_detection_ids_survive_confidence_filtering():
    t=tracker()
    first=step(t,0,[("car",(60,60,70,70),.05),CAR]).tracks[0]
    assert first.last_detection_id=="d0-1"
    second=step(t,1,[("car",(60,60,70,70),.05),("car",(11,11,31,31),.15)]).tracks[0]
    assert second.last_detection_id=="d1-1"

def test_classes_never_share_an_identity():
    t=tracker()
    car=step(t,0,[CAR]).tracks[0]
    switched=step(t,1,[("person",CAR[1],.9)])
    assert any(x.track_id==car.track_id and x.state is TrackState.LOST for x in switched.tracks)
    assert all(x.track_id!=car.track_id for x in switched.tracks if x.class_ref=="person")

def test_failed_frames_age_and_expire_tracks():
    t=tracker(buffer=1)
    original=step(t,0,[CAR]).tracks[0]
    lost=step(t,1,failed=True)
    assert lost.status is Status.FAILED and lost.detection_missing
    assert lost.tracks[0].state is TrackState.LOST and lost.tracks[0].current_bbox is None
    assert not step(t,2,failed=True).tracks
    step(t,3,[CAR])
    recovered=step(t,4,[CAR])
    assert all(x.track_id!=original.track_id for x in recovered.tracks)

def test_sequence_and_variant_switch_reset_association():
    t=tracker()
    first=step(t,0,[CAR]).tracks[0]
    switched=step(t,1,[CAR],variant="enhanced")
    assert switched.association.reason=="variant_switch"
    assert switched.tracks[0].track_id!=first.track_id
    other=step(t,0,[CAR],sequence="other")
    assert other.tracks[0].track_id!=switched.tracks[0].track_id

@pytest.mark.parametrize("frame",[0,2])
def test_duplicate_or_skipped_frame_is_rejected(frame):
    t=tracker()
    step(t,0,[CAR])
    with pytest.raises(ValueError,match="consecutive"):
        step(t,frame,[CAR])

def test_new_tracker_or_class_does_not_rewind_shared_ids():
    t=tracker()
    initial=step(t,0,[CAR]).tracks[0]
    step(tracker(),0,[CAR])
    step(t,1,[CAR,("person",(50,50,60,60),.9)])
    result=step(t,2,[CAR,("car",(70,70,90,90),.9)])
    result=step(t,3,[CAR,("car",(70,70,90,90),.9)])
    assert len({x.track_id for x in result.tracks})==len(result.tracks)
    assert any(x.track_id==initial.track_id for x in result.tracks)

def test_geometry_cannot_be_falsely_enabled():
    config=load_module(REPO_ROOT/"configs/m4_tracking.yaml")
    config["params"]["use_geometry"]=True
    with pytest.raises(ValueError,match="without geometry"):
        M4Tracking(config)

def test_invalid_bbox_is_rejected_before_updating_state():
    t=tracker()
    bad=detections(0,[("car",(-1,1,20,20),.9)])
    with pytest.raises(ValueError,match="outside"):
        t.update(bad,frame_index=0)
    assert step(t,0,[CAR]).tracks


def stable():
    return M4Tracking(load_module(REPO_ROOT/"configs/m4_tracking_stable.yaml"))


def test_vehicle_class_flip_keeps_id_but_person_stays_separate():
    tracker = stable()
    original = tracker.update(detections(0, [CAR]), frame_index=0).tracks[0]
    for frame in range(1, 5):
        result = tracker.update(detections(frame, [("truck", CAR[1], .9)]), frame_index=frame)
        track = next(t for t in result.tracks if t.current_bbox is not None)
        assert track.track_id == original.track_id
        assert track.class_ref == ("car" if frame < 3 else "truck")
        assert track.state is (TrackState.NEW if frame < 2 else TrackState.TRACKED)
    result = tracker.update(detections(5, [("person", CAR[1], .9)]), frame_index=5)
    assert next(t for t in result.tracks if t.track_id == original.track_id).state is TrackState.LOST
    assert all(t.track_id != original.track_id for t in result.tracks if t.class_ref == "person")


def test_low_score_does_not_confirm_candidate():
    tracker = stable()
    tracker.update(detections(0, [CAR]), frame_index=0)
    for frame in range(1, 4):
        result = tracker.update(detections(frame, [("car", CAR[1], .15)]), frame_index=frame)
        assert result.tracks[0].state is TrackState.NEW
    tracker.update(detections(4, [CAR]), frame_index=4)
    assert tracker.update(detections(5, [CAR]), frame_index=5).tracks[0].state is TrackState.NEW
    result = tracker.update(detections(6, [CAR]), frame_index=6)
    assert result.tracks[0].state is TrackState.TRACKED


def texture(width=640, height=480):
    rng = np.random.default_rng(42)
    gray = rng.integers(0, 256, (height, width), dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(gray, (5, 5), 0), cv2.COLOR_GRAY2RGB)


def test_motion_estimates_known_translation_and_resets():
    motion = MotionCompensator()
    image = texture()
    assert not motion.update(image, [])[1]["used"]
    moved = cv2.warpAffine(image, np.array([[1.,0.,12.],[0.,1.,6.]]), (640,480))
    matrix, report = motion.update(moved, [])
    assert report["used"]
    np.testing.assert_allclose(matrix, [[1,0,12],[0,1,6]], atol=.3)
    motion.reset()
    assert motion.update(moved, [])[1]["reason"] == "gmc_first_frame"


def test_textureless_and_masked_frames_fall_back_to_identity():
    for image, boxes in [(np.zeros((480,640,3),np.uint8), []),
                         (texture(), [BBox(0,0,640,480)])]:
        motion = MotionCompensator()
        motion.update(image, boxes)
        matrix, report = motion.update(image, boxes)
        assert not report["used"]
        np.testing.assert_array_equal(matrix, np.eye(2,3))


def test_gmc_hook_moves_prediction_before_matching_and_reports_only_actual_use():
    config = load_module(REPO_ROOT/"configs/m4_tracking_stable.yaml")
    config["params"]["min_confirmations"] = 1
    tracker = M4Tracking(config)
    # A 12px shift would break the 20px-wide box's IoU without compensation.
    image = texture(100,100)
    # Small images cannot satisfy the conservative feature spread gate; inject a known accepted warp.
    tracker._motion.update = lambda rgb, boxes: (np.array([[1.,0.,12.],[0.,1.,0.]]),
                                                {"used":True,"reason":"gmc_ok"})
    first = tracker.update(detections(0, [CAR]), frame_index=0, image_rgb=image).tracks[0]
    second = tracker.update(detections(1, [("car",(22,10,42,30),.9)]), frame_index=1, image_rgb=image)
    assert second.tracks[0].track_id == first.track_id
    assert second.association.motion_compensation_used
    assert second.tracks[0].current_bbox.x1 == pytest.approx(22, abs=1)
    missing = tracker.update(detections(2), frame_index=2)
    assert not missing.association.motion_compensation_used
    assert missing.association.reason == "gmc_missing_image"


def test_motion_variant_switch_and_invalid_rgb_reset_safely():
    tracker = stable()
    image = texture(100,100)
    with pytest.raises(ValueError, match="aligned"):
        tracker.update(detections(0, [CAR]), frame_index=0, image_rgb=image[:50])
    first = tracker.update(detections(0, [CAR]), frame_index=0, image_rgb=image)
    assert not first.association.motion_compensation_used
    switched = tracker.update(detections(1, [CAR], variant="enhanced"), frame_index=1, image_rgb=image)
    assert not switched.association.motion_compensation_used
    assert switched.association.reason == "variant_switch"
    assert tracker.last_motion["reason"] == "gmc_first_frame"
    assert tracker.last_motion["frame_a"] is None


def test_display_smoothing_never_changes_i6_and_clears_after_loss():
    config = load_module(REPO_ROOT/"configs/m4_tracking_stable.yaml")
    tracker = M4Tracking(config)
    image = np.zeros((100,100,3), np.uint8)
    renderer = TrackRenderer(.5)
    first = tracker.update(detections(0, [CAR]), frame_index=0)
    original = first.tracks[0]
    renderer.draw(image, first, 0)
    moved = replace(original, current_bbox=BBox(14,10,34,30))
    second = replace(first, tracks=[moved])
    renderer.draw(image, second, 1)
    assert renderer.previous[original.track_id][0] == 22
    assert moved.current_bbox.x1 == 14
    renderer.draw(image, replace(first, tracks=[]), 2)
    assert not renderer.previous
    renderer.draw(image, second, 3)
    assert renderer.previous[original.track_id][0] == 24


def test_motion_quality_gate_rejects_implausible_transform(monkeypatch):
    motion = MotionCompensator()
    image = texture()
    motion.update(image, [])
    def excessive_translation(previous, current, **kwargs):
        return np.array([[1.,0.,100.],[0.,1.,0.]]), np.ones((len(previous),1),np.uint8)
    monkeypatch.setattr(cv2, "estimateAffinePartial2D", excessive_translation)
    matrix, report = motion.update(image, [])
    assert not report["used"] and report["reason"] == "gmc_quality_rejected"
    np.testing.assert_array_equal(matrix, np.eye(2,3))


def test_tiny_image_does_not_crash_gmc():
    matrix, report = MotionCompensator().update(np.zeros((1,1,3),np.uint8), [])
    assert not report["used"] and report["reason"] == "gmc_small_image"
    np.testing.assert_array_equal(matrix, np.eye(2,3))


def test_stable_profile_keeps_moderate_score_target_until_confirmation():
    t=stable()
    rows=[("car",CAR[1],.30)]
    first=t.update(detections(0,rows),frame_index=0).tracks[0]
    assert first.state is TrackState.NEW
    assert t.update(detections(1,rows),frame_index=1).tracks[0].track_id==first.track_id
    third=t.update(detections(2,rows),frame_index=2).tracks[0]
    assert third.track_id==first.track_id and third.state is TrackState.TRACKED
