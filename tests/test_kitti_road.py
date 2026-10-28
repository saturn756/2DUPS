"""KITTI Road：适配器、文件名配对与单类二值指标（不依赖 data/raw/）。"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest

import cv2
import numpy as np

from twodups.contracts.base import CalibrationStatus, Ref
from twodups.contracts.i1_dataset_manifest import DatasetManifest, Sample
from twodups.data.adapters import build
from twodups.data.adapters.kitti_road import KittiRoadAdapter
from twodups.evaluation.kitti_road import (
    evaluate_binary, gt_key, pair, pred_key,
)


def _write_png(path: Path, array: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", array)
    assert success
    path.write_bytes(encoded.tobytes())


def _manifest(uri: str = "img/um_000000.png", *, kind: str = "image_file",
              fps: float | None = None, annotation: Ref | None = None,
              calibration: Ref | None = None) -> DatasetManifest:
    return DatasetManifest(
        manifest_id="kitti-road-test", version="1", source="KITTI_ROAD", license="test",
        samples=[Sample("um_000000", "um_000000", Ref(uri, kind),
                        annotation_ref=annotation, slice_tags=["um"], fps=fps,
                        calibration_ref=calibration)],
        splits={"eval": ["um_000000"]},
    )


class TestKittiRoadAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "img").mkdir()
        _write_png(self.root / "img" / "um_000000.png",
                   np.full((8, 12, 3), 90, dtype=np.uint8))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_yields_one_frame_per_image(self) -> None:
        adapter = KittiRoadAdapter(_manifest(), root=self.root)
        frames = list(adapter.iter_sample("um_000000"))
        self.assertEqual(len(frames), 1)
        metadata = frames[0].metadata
        self.assertEqual(metadata.frame_id, "um_000000")
        self.assertEqual(metadata.frame_index, 0)
        self.assertEqual((metadata.image_size.width, metadata.image_size.height), (12, 8))
        # 单帧没有时间轴，不伪造 PTS 也不伪造时间戳
        self.assertIsNone(metadata.timestamp)
        self.assertIsNone(metadata.source_timestamp)
        self.assertIs(metadata.calibration.status, CalibrationStatus.MISSING)
        self.assertEqual(metadata.calibration.reason, "kitti_road_calib_file_not_parsed")
        self.assertEqual(frames[0].image_rgb.dtype, np.uint8)
        self.assertEqual(frames[0].image_rgb.shape, (8, 12, 3))

    def test_manifest_declaring_fps_is_rejected(self) -> None:
        adapter = KittiRoadAdapter(_manifest(fps=10.0), root=self.root)
        problems = adapter.validate(_manifest(fps=10.0))
        self.assertTrue(any("must not declare fps" in problem for problem in problems))

    def test_wrong_media_kind_is_rejected(self) -> None:
        manifest = _manifest(kind="image_sequence")
        problems = KittiRoadAdapter(manifest, root=self.root).validate(manifest)
        self.assertTrue(any("media_ref.kind" in problem for problem in problems))

    def test_annotation_kind_is_checked(self) -> None:
        manifest = _manifest(annotation=Ref("gt.png", "file"))
        problems = KittiRoadAdapter(manifest, root=self.root).validate(manifest)
        self.assertTrue(any("annotation_ref.kind" in problem for problem in problems))

    def test_missing_file_is_reported(self) -> None:
        adapter = KittiRoadAdapter(_manifest("img/nope.png"), root=self.root)
        with self.assertRaises(FileNotFoundError):
            list(adapter.iter_sample("um_000000"))

    def test_reference_escaping_repository_is_rejected(self) -> None:
        adapter = KittiRoadAdapter(_manifest("../outside.png"), root=self.root)
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            list(adapter.iter_sample("um_000000"))

    def test_registry_builds_kitti_road_adapter(self) -> None:
        self.assertIsInstance(build(_manifest(), root=self.root), KittiRoadAdapter)

    def _write_calibration(self, *, width: int = 12, height: int = 8) -> Ref:
        payload = {
            "schema": "twodups.camera_params/1",
            "camera_id": "camera_2",
            "image_size": {"width": width, "height": height},
            "model": "pinhole",
            "intrinsics": {"fx": 7.0, "fy": 7.0, "cx": 6.0, "cy": 4.0},
            "distortion": {"model": "none", "coefficients": []},
            "provenance": {"method": "KITTI_P_rect_02"},
        }
        path = self.root / "camera_params.json"
        content = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        path.write_bytes(content)
        checksum = "sha256:" + hashlib.sha256(content).hexdigest()
        return Ref(path.name, "camera_params", checksum)

    def test_calibration_reference_is_valid_when_size_matches(self) -> None:
        ref = self._write_calibration()
        frame = next(KittiRoadAdapter(_manifest(calibration=ref), root=self.root)
                     .iter_sample("um_000000"))
        self.assertIs(frame.metadata.calibration.status, CalibrationStatus.VALID)
        self.assertEqual(frame.metadata.calibration.params_ref, ref)
        self.assertEqual(frame.metadata.camera_id, "camera_2")

    def test_calibration_reference_is_stale_when_size_differs(self) -> None:
        ref = self._write_calibration(width=24, height=16)
        frame = next(KittiRoadAdapter(_manifest(calibration=ref), root=self.root)
                     .iter_sample("um_000000"))
        self.assertIs(frame.metadata.calibration.status, CalibrationStatus.STALE)
        self.assertIn("calibration_image_size_mismatch", frame.metadata.calibration.reason)


class TestKittiRoadKeys(unittest.TestCase):
    def test_ground_truth_key_strips_the_tag(self) -> None:
        self.assertEqual(gt_key("um_road_000000"), "um_000000")
        self.assertEqual(gt_key("umm_lane_000094"), "umm_000094")
        self.assertEqual(gt_key("uu_000012"), "uu_000012")

    def test_prediction_key_strips_the_artifact_suffix(self) -> None:
        self.assertEqual(pred_key("um_000000_class_ab12cd34ef56"), "um_000000")
        self.assertEqual(pred_key("um_000000"), "um_000000")


class TestKittiRoadPairingAndMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pred = self.root / "pred"
        self.gt = self.root / "gt"
        (self.pred / "um").mkdir(parents=True)
        self.gt.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _gt_mask(self, name: str, positive_rows: slice, *, ignore_patch: bool = False) -> None:
        mask = np.zeros((10, 10, 3), np.uint8)
        mask[:, :] = (255, 0, 0)                      # 红=背景（BGR 存同值不影响对称色）
        mask[positive_rows, :] = (255, 0, 255)        # 洋红=正类
        if ignore_patch:
            mask[0:2, 0:2] = 0                        # 黑=ignore
        _write_png(self.gt / name, mask)

    def test_pairs_prediction_artifacts_with_tagged_masks(self) -> None:
        _write_png(self.pred / "um" / "um_000000_class_aaaa.png", np.zeros((10, 10), np.uint8))
        self._gt_mask("um_road_000000.png", slice(5, 10))
        pairs = pair(self.pred, self.gt, tag="road")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1].name, "um_road_000000.png")

    def test_only_masks_of_the_requested_tag_are_used(self) -> None:
        _write_png(self.pred / "um" / "um_000000_class_aaaa.png", np.zeros((10, 10), np.uint8))
        self._gt_mask("um_road_000000.png", slice(5, 10))
        self._gt_mask("um_lane_000000.png", slice(8, 10))
        road = pair(self.pred, self.gt, tag="road")
        lane = pair(self.pred, self.gt, tag="lane")
        self.assertEqual(road[0][1].name, "um_road_000000.png")
        self.assertEqual(lane[0][1].name, "um_lane_000000.png")

    def test_perfect_prediction_scores_one(self) -> None:
        prediction = np.zeros((10, 10), np.uint8)
        prediction[5:, :] = 1
        _write_png(self.pred / "um" / "um_000000_class_aaaa.png", prediction)
        self._gt_mask("um_road_000000.png", slice(5, 10))
        score = evaluate_binary(pair(self.pred, self.gt, tag="road"), target_id=1)
        self.assertAlmostEqual(score.iou, 1.0)
        self.assertAlmostEqual(score.f1, 1.0)
        self.assertAlmostEqual(score.pixel_accuracy, 1.0)

    def test_missed_positive_region_lowers_iou(self) -> None:
        prediction = np.zeros((10, 10), np.uint8)     # 全部预测成背景
        _write_png(self.pred / "um" / "um_000000_class_aaaa.png", prediction)
        self._gt_mask("um_road_000000.png", slice(5, 10))
        score = evaluate_binary(pair(self.pred, self.gt, tag="road"), target_id=1)
        self.assertAlmostEqual(score.iou, 0.0)
        self.assertAlmostEqual(score.recall, 0.0)
        self.assertAlmostEqual(score.pixel_accuracy, 0.5)
        self.assertEqual(score.false_negative, 50)

    def test_wrong_target_class_counts_as_false_positives(self) -> None:
        prediction = np.zeros((10, 10), np.uint8)
        prediction[:, :] = 2                          # 全predict成车道线，目标是 road
        _write_png(self.pred / "um" / "um_000000_class_aaaa.png", prediction)
        self._gt_mask("um_road_000000.png", slice(5, 10))
        score = evaluate_binary(pair(self.pred, self.gt, tag="road"), target_id=1)
        self.assertEqual(score.true_positive, 0)
        self.assertEqual(score.false_negative, 50)

    def test_ignore_pixels_do_not_enter_metrics(self) -> None:
        prediction = np.zeros((10, 10), np.uint8)
        prediction[5:, :] = 1
        _write_png(self.pred / "um" / "um_000000_class_aaaa.png", prediction)
        self._gt_mask("um_road_000000.png", slice(5, 10), ignore_patch=True)
        score = evaluate_binary(pair(self.pred, self.gt, tag="road"), target_id=1)
        self.assertEqual(score.evaluated_pixels, 96)  # 100 减去 2x2 ignore
        self.assertAlmostEqual(score.iou, 1.0)

    def test_missing_mask_for_tag_is_an_error(self) -> None:
        _write_png(self.pred / "um" / "um_000000_class_aaaa.png", np.zeros((10, 10), np.uint8))
        self._gt_mask("um_road_000000.png", slice(5, 10))
        with self.assertRaises(FileNotFoundError):
            pair(self.pred, self.gt, tag="lane")

    def test_size_mismatch_is_an_error(self) -> None:
        _write_png(self.pred / "um" / "um_000000_class_aaaa.png", np.zeros((9, 10), np.uint8))
        self._gt_mask("um_road_000000.png", slice(5, 10))
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            evaluate_binary(pair(self.pred, self.gt, tag="road"), target_id=1)


if __name__ == "__main__":
    unittest.main()
