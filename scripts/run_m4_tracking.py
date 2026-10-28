#!/usr/bin/env python3
"""M0 -> M1 -> M2 -> ByteTrack; export I5/I6 JSONL and an H.264 preview."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import av
import cv2
import yaml

from twodups.contracts import Status, to_wire
from twodups.data.manifest import load
from twodups.modules.m0_data_ingest import M0DataIngest
from twodups.modules.m1_preprocess import M1Preprocess
from twodups.modules.m2_detection import M2Detection
from twodups.modules.m4_tracking import M4Tracking
from twodups.pipeline.runner import iter_m0_m1_m2_chain
from twodups.utils.config import REPO_ROOT, load_module


def annotate(rgb, tracks, frame_index):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    visible = 0
    for track in tracks.tracks:
        if track.current_bbox is None:
            continue
        visible += 1
        number = int(track.track_id.rsplit(":", 1)[1])
        color = ((number * 67) % 180 + 60, (number * 97) % 180 + 60, (number * 37) % 180 + 60)
        box = track.current_bbox
        x1,y1,x2,y2 = (int(round(v)) for v in (box.x1,box.y1,box.x2,box.y2))
        cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
        name = track.class_ref.split(".",1)[-1].replace("_"," ")
        cv2.putText(image,f"ID {number} {name} {track.confidence:.2f}",
                    (max(0,x1),max(62,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
    cv2.rectangle(image,(0,0),(image.shape[1],45),(20,20,20),-1)
    cv2.putText(image,f"YOLO11n + ByteTrack | frame {frame_index} | visible {visible} | {tracks.status.value}",
                (10,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT/"configs/dataset_manifest.yaml")
    parser.add_argument("--detection-config", type=Path, default=REPO_ROOT/"configs/m2_detection.yaml")
    parser.add_argument("--tracking-config", type=Path, default=REPO_ROOT/"configs/m4_tracking.yaml")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True, help="New output directory inside repository")
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    output = args.output.resolve()
    tracker_path = args.tracking_config.resolve()
    if not output.is_relative_to(REPO_ROOT) or not tracker_path.is_relative_to(REPO_ROOT):
        parser.error("Output and tracker config must be inside the repository")
    manifest = load(args.manifest)
    samples = [s for s in manifest.samples if s.sample_id == args.sample_id]
    if not samples or samples[0].media_ref.kind != "video":
        parser.error("Tracking requires a video sample in the manifest")
    source = (REPO_ROOT/samples[0].media_ref.uri).resolve()
    if not source.is_relative_to(REPO_ROOT):
        parser.error("Video reference escapes repository")
    with av.open(str(source)) as container:
        rate = container.streams.video[0].average_rate
    if rate is None or rate <= 0:
        parser.error("Video has no usable frame rate")
    tracker = M4Tracking(load_module(tracker_path),config_ref=tracker_path.relative_to(REPO_ROOT).as_posix())
    detector_config = load_module(args.detection_config)
    # ByteTrack's second association needs the low-score detections M2 normally filters out.
    detector_config["params"]["detection"]["score_threshold"] = tracker.args.track_low_thresh
    output.mkdir(parents=True,exist_ok=False)
    config_path = output/"detection.config.yaml"
    config_path.write_text(yaml.safe_dump(detector_config,allow_unicode=True,sort_keys=False))
    (output/"tracking.config.yaml").write_text(yaml.safe_dump(tracker.config,allow_unicode=True,sort_keys=False))
    detector = M2Detection(detector_config,config_ref=config_path.relative_to(REPO_ROOT).as_posix())
    frames = failures = 0
    identities = set()
    with ExitStack() as stack:
        video = stack.enter_context(av.open(str(output/"tracking.mp4"),"w",options={"movflags":"+faststart"}))
        detection_log = stack.enter_context((output/"detections.jsonl").open("w"))
        track_log = stack.enter_context((output/"tracks.jsonl").open("w"))
        stream = None
        for step in iter_m0_m1_m2_chain(M0DataIngest(),manifest,M1Preprocess(),detector,
                                        args.sample_id,max_frames=args.max_frames):
            index = step.frame.metadata.frame_index
            tracks = tracker.update(step.detections,frame_index=index)
            detection_log.write(json.dumps(to_wire(step.detections),ensure_ascii=False)+"\n")
            track_log.write(json.dumps(to_wire(tracks),ensure_ascii=False)+"\n")
            failures += tracks.status is not Status.OK
            identities.update(t.track_id for t in tracks.tracks)
            image = annotate(step.processed.default_image_rgb,tracks,index)
            if stream is None:
                stream = video.add_stream("libx264",rate=rate)
                stream.height,stream.width = image.shape[:2]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf":"22","preset":"veryfast"}
            for packet in stream.encode(av.VideoFrame.from_ndarray(image,format="bgr24")):
                video.mux(packet)
            frames += 1
            if frames % 100 == 0:
                print(f"processed={frames}",flush=True)
        if stream is None:
            raise ValueError("Video yielded no frames")
        for packet in stream.encode():
            video.mux(packet)
    with av.open(str(output/"tracking.mp4")) as saved:
        assert sum(1 for _ in saved.decode(video=0)) == frames
    summary = {"frames":frames,"fps":float(rate),"unique_tracks":len(identities),"failed_frames":failures,
               "detector_config":config_path.relative_to(REPO_ROOT).as_posix(),
               "video":(output/"tracking.mp4").relative_to(REPO_ROOT).as_posix()}
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
