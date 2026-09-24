"""Unit tests for the BDD100K manifest and streaming alignment."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from twodups.contracts.base import CalibrationStatus, Ref
from twodups.contracts.i1_dataset_manifest import DatasetManifest, Sample
from twodups.data.bdd100k_loader import (
    BDD100KVideoLoader, FrameAnnotation, LoadedFrame, _load_annotations,
    _resolve_ref, rotation_from_display_matrix,
)
from twodups.data.manifest import load


class TestBDD100KLoader(unittest.TestCase):
    def test_manifest_contains_five_distinct_bdd_sequences(self) -> None:
        manifest = load(Path(__file__).resolve().parents[1] / "configs/dataset_manifest.yaml")
        self.assertEqual(manifest.source, "BDD100K")
        self.assertTrue(manifest.checksum.startswith("sha256:"))
        self.assertEqual(len(manifest.samples), len(manifest.splits["train"]))
        self.assertEqual(len(manifest.samples), 5)
        self.assertEqual(len({sample.sequence_id for sample in manifest.samples}), 5)

    def test_rotation_from_display_matrix(self) -> None:
        import struct

        class FakeSideData:
            type = SimpleNamespace(name="DISPLAYMATRIX")

            def __init__(self, matrix: tuple[int, ...]) -> None:
                self.matrix = matrix

            def __bytes__(self) -> bytes:
                return struct.pack("<9i", *self.matrix)

        for matrix, expected in [
            ((0, 65536, 0, -65536, 0, 0, 0, 0, 1073741824), 90),
            ((0, -65536, 0, 65536, 0, 0, 0, 0, 1073741824), 270),
        ]:
            frame = SimpleNamespace(side_data=[FakeSideData(matrix)])
            self.assertEqual(rotation_from_display_matrix(frame), expected)

    def test_decode_corrects_regressing_pts_but_keeps_original(self) -> None:
        class FakeFrame:
            def __init__(self, time: float) -> None:
                self.time = time

            def to_ndarray(self, format: str) -> np.ndarray:
                assert format == "rgb24"
                return np.zeros((2, 3, 3), dtype=np.uint8)

        manifest = DatasetManifest("test", "1", "BDD100K", "research")
        loader = BDD100KVideoLoader(manifest)
        sample = Sample("seq", "seq", Ref("video.mov"))
        container = SimpleNamespace(decode=lambda **kwargs: iter(map(FakeFrame, [0, 1/30, 1/30, 0.008, 0.2])))
        with patch("twodups.data.bdd100k_loader.rotation_from_display_matrix", return_value=0):
            frames = list(loader._decode(container, sample, 30.0, None))
        self.assertEqual([frame.metadata.source_timestamp for frame in frames], [0, 1/30, 1/30, 0.008, 0.2])
        self.assertTrue(all(b.metadata.timestamp > a.metadata.timestamp for a, b in zip(frames, frames[1:])))
        self.assertGreater(frames[2].metadata.timestamp, frames[2].metadata.source_timestamp)
        self.assertIs(frames[0].metadata.calibration.status, CalibrationStatus.MISSING)
        self.assertEqual(frames[0].metadata.image_size.width, 3)

    def test_sparse_annotation_is_attached_once_to_nearest_frame(self) -> None:
        frames = [LoadedFrame(SimpleNamespace(timestamp=t, frame_index=i), np.empty((0,)))
                  for i, t in enumerate([9.96, 9.99, 10.023, 10.056])]
        annotation = FrameAnnotation(timestamp=10.0, objects=())
        result = list(BDD100KVideoLoader._attach_annotations(iter(frames), [annotation], 30, True))
        self.assertEqual([[item.timestamp for item in frame.annotations] for frame in result],
                         [[], [10.0], [], []])

    def test_sparse_annotation_outside_tolerance_fails(self) -> None:
        frames = [LoadedFrame(SimpleNamespace(timestamp=t), np.empty((0,))) for t in [9.0, 11.0]]
        with self.assertRaisesRegex(ValueError, "No video frame near annotation"):
            list(BDD100KVideoLoader._attach_annotations(iter(frames), [FrameAnnotation(10.0, ())], 30, True))

    def test_truncated_video_does_not_require_future_annotation(self) -> None:
        frames = [LoadedFrame(SimpleNamespace(timestamp=t), np.empty((0,))) for t in [0.0, 0.033]]
        result = list(BDD100KVideoLoader._attach_annotations(iter(frames),
                      [FrameAnnotation(10.0, ())], 30, False))
        self.assertEqual(len(result), 2)
        self.assertFalse(any(frame.annotations for frame in result))

    def test_media_ref_cannot_escape_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            _resolve_ref(Path(__file__).resolve().parents[1], Ref("../outside.mov"))

    def test_old_bdd_label_id_is_local_not_track_id(self) -> None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.json"
            path.write_text(json.dumps({"name": "seq", "frames": [{"timestamp": 10000, "objects": [
                {"id": 7, "category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}
            ]}]}), encoding="utf-8")
            annotations = _load_annotations(path, "seq")
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].objects[0].local_id, "7")
        self.assertEqual(annotations[0].objects[0].bbox_xyxy, (1, 2, 3, 4))
