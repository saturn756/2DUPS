"""Exercise the real Ultralytics ByteTrack backend with synthetic I5 records."""
from dataclasses import replace
import pytest
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
