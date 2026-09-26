"""M2 quality gate and variant invariants without external dataset files."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np

from twodups.contracts.base import CalibrationState, CalibrationStatus, Producer, Ref, Status
from twodups.contracts.i2_calibrated_frame import CalibratedFrame, ImageSize
from twodups.data.bdd100k_loader import LoadedFrame
from twodups.modules.base import ModuleContext
from twodups.modules.m2_preprocess import M2Preprocess
from twodups.modules.m2_preprocess.quality_metrics import measure
from twodups.utils.config import REPO_ROOT, load_module


def _frame(image: np.ndarray, calibration: CalibrationStatus = CalibrationStatus.MISSING) -> LoadedFrame:
    height, width = image.shape[:2]
    metadata = CalibratedFrame(
        sequence_id="test_seq", frame_id="test_seq:000000", frame_index=0,
        image_size=ImageSize(width, height), media_ref=Ref("data/raw/test.mov#frame=0", "video_frame"),
        coordinate_space="pixel", calibration=CalibrationState(calibration),
    )
    return LoadedFrame(metadata=metadata, image_rgb=image)


class TestM2(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        config = load_module(REPO_ROOT / "configs/m2_preprocess.yaml")
        config["params"]["output_dir"] = self.tmp.name
        self.m2 = M2Preprocess(config)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dark_frame_generates_traceable_enhanced_version(self) -> None:
        image = np.full((64, 96, 3), 10, dtype=np.uint8)
        original = image.copy()
        result = self.m2.process(_frame(image))
        self.assertEqual(result.variants.default_variant, "enhanced")
        self.assertEqual([v.variant_id for v in result.variants.variants], ["raw", "enhanced"])
        self.assertEqual(result.variants.get("enhanced").parent_variant_id, "raw")
        self.assertEqual(result.variants.get("enhanced").coordinate_space, "pixel")
        self.assertTrue(result.variants.get("enhanced").media_ref.checksum.startswith("sha256:"))
        self.assertTrue(Path(result.variants.get("enhanced").media_ref.uri).is_file())
        self.assertTrue(result.report_path.is_file())
        self.assertEqual(result.default_image_rgb.shape, image.shape)
        np.testing.assert_array_equal(image, original)
        self.assertIn("low_light", result.quality.quality_flags)
        self.assertEqual(result.quality.selected_variant_id, "enhanced")
        self.assertEqual(result.quality.status, Status.OK)
        self.assertNotIn("rectified", [v.variant_id for v in result.variants.variants])

    def test_normal_frame_keeps_raw(self) -> None:
        image = np.full((64, 96, 3), 150, dtype=np.uint8)
        result = self.m2.process(_frame(image))
        self.assertEqual(result.variants.default_variant, "raw")
        self.assertEqual(result.quality.gate_decision, "use_raw")
        self.assertEqual(len(result.variants.variants), 1)
        self.assertEqual(result.quality.metrics["brightness_mean"], 150.0)

    def test_noise_gate_generates_denoised(self) -> None:
        image = np.full((64, 96, 3), 120, dtype=np.uint8)
        metrics = {"brightness_mean": 120.0, "underexposed_fraction": 0.0,
                   "overexposed_fraction": 0.0, "blur_laplacian_variance": 100.0,
                   "noise_sigma": 10.0}
        with patch("twodups.modules.m2_preprocess.module.measure", return_value=metrics):
            result = self.m2.process(_frame(image))
        self.assertEqual(result.variants.default_variant, "denoised")
        self.assertEqual(result.variants.get("denoised").parent_variant_id, "raw")

    def test_noise_metric_responds_to_strong_synthetic_noise(self) -> None:
        rng = np.random.default_rng(1)
        clean = np.full((240, 320, 3), 100, dtype=np.uint8)
        noisy = np.clip(clean.astype(np.float32) + rng.normal(0, 20, clean.shape), 0, 255).astype(np.uint8)
        self.assertLess(measure(clean)["noise_sigma"], 1)
        self.assertGreater(measure(noisy)["noise_sigma"], 8)

    def test_processing_failure_falls_back_to_raw(self) -> None:
        image = np.full((64, 96, 3), 10, dtype=np.uint8)
        with patch("twodups.modules.m2_preprocess.module.enhance", side_effect=ValueError("bad transform")):
            result = self.m2.process(_frame(image))
        self.assertEqual(result.variants.default_variant, "raw")
        self.assertEqual(result.quality.status, Status.UNRELIABLE)
        self.assertTrue(any(op["status"] == "failed" for op in result.quality.operations))

    def test_invalid_input_is_rejected(self) -> None:
        bad = _frame(np.zeros((64, 96, 3), dtype=np.uint8))
        bad.image_rgb = bad.image_rgb[:, :-1]
        with self.assertRaisesRegex(ValueError, "matching I2 image_size"):
            self.m2.process(bad)

    def test_valid_calibration_does_not_create_fake_rectification(self) -> None:
        image = np.full((64, 96, 3), 150, dtype=np.uint8)
        result = self.m2.process(_frame(image, CalibrationStatus.VALID))
        self.assertNotIn("rectified", [v.variant_id for v in result.variants.variants])
        self.assertIn("rectification_deferred_until_calibration_schema", result.quality.reason)

    def test_module_run_emits_both_i3_records(self) -> None:
        image = np.full((64, 96, 3), 150, dtype=np.uint8)
        ctx = ModuleContext(config=self.m2.config,
                            producer=Producer("m2.clahe_lab", "configs/m2_preprocess.yaml"))
        result = self.m2.run(_frame(image), ctx=ctx)
        self.assertEqual(set(result.outputs), {"ImageVariantSet", "QualityReport"})
        self.assertEqual(result.outputs["ImageVariantSet"].producer_ref, "m2.clahe_lab")

    def test_experiment_config_ref_is_preserved(self) -> None:
        config = load_module(REPO_ROOT / "configs/m2_preprocess.yaml")
        config["params"]["output_dir"] = self.tmp.name
        m2 = M2Preprocess(config, config_ref="configs/experiments/m2/lowlight-a.yaml")
        result = m2.process(_frame(np.full((64, 96, 3), 150, dtype=np.uint8)))
        self.assertEqual(result.variants.config_ref, "configs/experiments/m2/lowlight-a.yaml")
        self.assertEqual(result.quality.config_ref, result.variants.config_ref)
        stored = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["interface"], "I3")
        self.assertEqual(stored["kind"], "QualityReport")
        self.assertEqual(stored["config_ref"], result.quality.config_ref)

    def test_unsupported_rectification_config_is_rejected(self) -> None:
        config = load_module(REPO_ROOT / "configs/m2_preprocess.yaml")
        config["impl"]["rectification"] = "opencv_undistort"
        with self.assertRaisesRegex(ValueError, "cannot enable rectification"):
            M2Preprocess(config)
