"""Stream BDD100K MOV frames into M1's I2 metadata and RGB pixels.

The decoded image is an in-process value.  I2 itself remains serializable and
points back to the source video and frame index; no frame cache is written.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import struct
from typing import Iterator

import av
import numpy as np

from ..contracts.base import CalibrationState, CalibrationStatus, Ref
from ..contracts.i1_dataset_manifest import DatasetManifest, Sample
from ..contracts.i2_calibrated_frame import CalibratedFrame, ImageSize
from ..utils.config import REPO_ROOT


@dataclass(frozen=True)
class ObjectAnnotation:
    """A one-frame object label; ``local_id`` is not a tracking identity."""

    local_id: str
    category: str
    bbox_xyxy: tuple[float, float, float, float]
    attributes: dict


@dataclass(frozen=True)
class FrameAnnotation:
    timestamp: float
    objects: tuple[ObjectAnnotation, ...]


@dataclass
class LoadedFrame:
    """Runtime wrapper: serializable I2 metadata plus the actual RGB pixels."""

    metadata: CalibratedFrame
    image_rgb: np.ndarray
    annotations: list[FrameAnnotation] = field(default_factory=list)


def rotation_from_display_matrix(frame: av.VideoFrame) -> int:
    """Return clockwise degrees from FFmpeg's MOV display matrix."""
    for item in frame.side_data:
        if item.type.name != "DISPLAYMATRIX":
            continue
        values = struct.unpack("<9i", bytes(item))
        angle = math.degrees(math.atan2(values[1], values[0]))
        quarter_turns = round(angle / 90)
        if abs(angle - 90 * quarter_turns) > 1:
            raise ValueError(f"Unsupported video rotation: {angle:.2f} degrees")
        return (90 * quarter_turns) % 360
    return 0


def _resolve_ref(root: Path, ref: Ref) -> Path:
    candidate = (root / ref.uri).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Media reference escapes repository: {ref.uri}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing dataset file: {candidate}")
    return candidate


def _load_annotations(path: Path | None, sequence_id: str) -> list[FrameAnnotation]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("name") != sequence_id:
        raise ValueError(f"Annotation/video ID mismatch: {path}")
    result = []
    for row in data.get("frames", []):
        objects = []
        for obj in row.get("objects", []):
            box = obj.get("box2d")
            if box is None:
                continue
            xyxy = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
            if xyxy[2] < xyxy[0] or xyxy[3] < xyxy[1]:
                raise ValueError(f"Invalid bbox in {path}")
            objects.append(ObjectAnnotation(
                local_id=str(obj["id"]),
                category=obj["category"],
                bbox_xyxy=xyxy,
                attributes=obj.get("attributes") or {},
            ))
        result.append(FrameAnnotation(timestamp=float(row["timestamp"]) / 1000.0,
                                      objects=tuple(objects)))
    return sorted(result, key=lambda annotation: annotation.timestamp)


class BDD100KVideoLoader:
    def __init__(self, manifest: DatasetManifest, root: str | Path = REPO_ROOT) -> None:
        if manifest.source != "BDD100K":
            raise ValueError(f"Expected BDD100K manifest, got {manifest.source}")
        self.manifest = manifest
        self.root = Path(root).resolve()
        self.samples = {sample.sample_id: sample for sample in manifest.samples}

    def iter_split(self, split: str, max_frames_per_sequence: int | None = None) -> Iterator[LoadedFrame]:
        """Yield all frames in split order, closing each video before the next."""
        if split not in self.manifest.splits:
            raise KeyError(f"Unknown split: {split}")
        for sample_id in self.manifest.splits[split]:
            yield from self.iter_sample(sample_id, max_frames=max_frames_per_sequence)

    def iter_sample(self, sample_id: str, max_frames: int | None = None) -> Iterator[LoadedFrame]:
        """Decode one sequence and attach each sparse label to its nearest frame."""
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        sample = self.samples[sample_id]
        video_path = _resolve_ref(self.root, sample.media_ref)
        annotation_path = (_resolve_ref(self.root, sample.annotation_ref)
                           if sample.annotation_ref is not None else None)
        annotations = _load_annotations(annotation_path, sample.sequence_id)
        with av.open(str(video_path)) as container:
            if len(container.streams.video) != 1:
                raise ValueError(f"Expected one video stream: {video_path}")
            stream = container.streams.video[0]
            if stream.average_rate is None or float(stream.average_rate) <= 0:
                raise ValueError(f"Missing frame rate: {video_path}")
            fps = float(stream.average_rate)
            frames = self._decode(container, sample, fps, max_frames)
            yield from self._attach_annotations(frames, annotations, fps, complete=max_frames is None)

    def _decode(self, container: av.container.InputContainer, sample: Sample,
                fps: float, max_frames: int | None) -> Iterator[LoadedFrame]:
        rotation: int | None = None
        last_timestamp: float | None = None
        for index, decoded in enumerate(container.decode(video=0)):
            if max_frames is not None and index >= max_frames:
                break
            if rotation is None:
                rotation = rotation_from_display_matrix(decoded)
            # Some official MOVs repeat or regress in PTS at the start. Keep the
            # unmodified PTS for audit, but expose a strictly increasing clock
            # to downstream modules. Use only a 1 ms minimum step so later good
            # PTS can catch up instead of shifting the whole sequence.
            source_timestamp = float(decoded.time) if decoded.time is not None else None
            candidate = source_timestamp if source_timestamp is not None else index / fps
            if not math.isfinite(candidate):
                raise ValueError(f"Non-finite PTS in {sample.sequence_id} frame {index}")
            timestamp = max(0.0, candidate)
            if last_timestamp is not None:
                timestamp = max(timestamp, last_timestamp + 0.001)
            image = decoded.to_ndarray(format="rgb24")
            if rotation:
                image = np.rot90(image, -(rotation // 90))
            image = np.ascontiguousarray(image)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Unexpected decoded frame shape: {image.shape}")
            height, width = image.shape[:2]
            metadata = CalibratedFrame(
                sequence_id=sample.sequence_id,
                frame_id=f"{sample.sequence_id}:{index:06d}",
                frame_index=index,
                image_size=ImageSize(width=width, height=height),
                media_ref=Ref(uri=f"{sample.media_ref.uri}#frame={index}", kind="video_frame"),
                coordinate_space="pixel",
                calibration=CalibrationState(
                    status=CalibrationStatus.MISSING,
                    reason="dataset_does_not_provide_verified_calibration",
                ),
                camera_id="camera_front",
                timestamp=timestamp,
                source_timestamp=source_timestamp,
                rotation_degrees_clockwise=rotation or 0,
                manifest_ref=self.manifest.manifest_id,
            )
            yield LoadedFrame(metadata=metadata, image_rgb=image)
            last_timestamp = timestamp

    @staticmethod
    def _attach_annotations(frames: Iterator[LoadedFrame], annotations: list[FrameAnnotation],
                            fps: float, complete: bool) -> Iterator[LoadedFrame]:
        pending: LoadedFrame | None = None
        next_index = 0
        tolerance = max(0.05, 1.5 / fps)
        for current in frames:
            if pending is None:
                pending = current
                continue
            while next_index < len(annotations) and annotations[next_index].timestamp <= current.metadata.timestamp:
                annotation = annotations[next_index]
                target = min((pending, current),
                             key=lambda item: abs(item.metadata.timestamp - annotation.timestamp))
                error = abs(target.metadata.timestamp - annotation.timestamp)
                if error > tolerance:
                    raise ValueError(f"No video frame near annotation at {annotation.timestamp:.3f}s")
                target.annotations.append(annotation)
                next_index += 1
            yield pending
            pending = current
        if pending is not None:
            while next_index < len(annotations):
                annotation = annotations[next_index]
                error = abs(pending.metadata.timestamp - annotation.timestamp)
                if error <= tolerance:
                    pending.annotations.append(annotation)
                    next_index += 1
                elif complete:
                    raise ValueError(f"Unmatched annotation at {annotation.timestamp:.3f}s")
                else:
                    break
            yield pending
        elif complete and annotations:
            raise ValueError("Video contains no frames for its annotations")
