"""M4 I3→I5 contract tests without Ultralytics or real weights."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from twodups.contracts import (CalibrationStatus, ImageSize, ImageVariant,
                               ImageVariantSet, Ref, Status)
from twodups.modules.m4_detection_tracking import M4DetectionTracking
from twodups.modules.m4_detection_tracking.impl_yolo11 import YOLO11Detector
from twodups.utils.config import REPO_ROOT, load_module


def _input() -> tuple[ImageVariantSet, dict[str, np.ndarray]]:
    variants = ImageVariantSet(
        sequence_id="seq", frame_id="seq:000001", image_size=ImageSize(32, 24),
        coordinate_space="pixel", calibration_status=CalibrationStatus.MISSING,
        default_variant="raw", variants=[ImageVariant("raw", "raw", Ref("raw.mov#frame=1"))],
    )
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]  # RGB carrier; Ultralytics receives BGR.
    return variants, {"raw": image}


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)


class FakeResult:
    def __init__(self, boxes):
        self.orig_shape = (24, 32)
        self.boxes = boxes


class FakeModel:
    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
             5: "bus", 7: "truck", 9: "traffic light", 11: "stop sign"}

    def __init__(self, boxes):
        self.boxes = boxes
        self.source = None
        self.options = None

    def predict(self, *, source, **options):
        self.source = source
        self.options = options
        return [FakeResult(self.boxes)]


def _config():
    return load_module(REPO_ROOT / "configs/m4_detection_tracking.yaml")


def test_one_yolo11_detection_becomes_aligned_i5() -> None:
    model = FakeModel(FakeBoxes([[2, 3, 20, 21]], [0.91], [2]))
    module = M4DetectionTracking(_config(), model=model)
    variants, images = _input()
    result = module.detect(variants, images)
    assert result.status is Status.OK
    assert result.detection_missing is False
    assert result.frame_id == variants.frame_id
    assert result.variant_id == variants.default_variant
    assert result.coordinate_space == variants.coordinate_space
    assert result.objects[0].class_ref == "coco.car"
    assert result.class_schema_ref == "configs/m4_detection_tracking.yaml#params.detection.class_map"
    assert result.objects[0].bbox.x2 == 20
    assert model.source[0, 0].tolist() == [30, 20, 10]
    assert model.options["classes"] == sorted(_config()["params"]["detection"]["class_map"])
    assert model.options["device"] == 0


def test_empty_detection_is_not_failure() -> None:
    module = M4DetectionTracking(_config(), model=FakeModel(FakeBoxes([], [], [])))
    result = module.detect(*_input())
    assert result.status is Status.OK and result.objects == []
    assert result.detection_missing is False


def test_no_checkpoint_never_pretends_success() -> None:
    module = M4DetectionTracking(_config())
    result = module.detect(*_input())
    assert result.status is Status.FAILED
    assert result.detection_missing is True and result.objects == []
    assert "checkpoint_not_configured" in result.reason


def test_wrong_variant_pixels_are_rejected() -> None:
    module = M4DetectionTracking(_config(), model=FakeModel(FakeBoxes([], [], [])))
    variants, _ = _input()
    with pytest.raises(ValueError, match="default_variant"):
        module.detect(variants, {"enhanced": np.zeros((24, 32, 3), dtype=np.uint8)})


def test_checkpoint_class_mismatch_fails_explicitly() -> None:
    model = FakeModel(FakeBoxes([[2, 3, 20, 21]], [0.91], [2]))
    model.names = {**FakeModel.names, 2: "truck"}
    result = M4DetectionTracking(_config(), model=model).detect(*_input())
    assert result.status is Status.FAILED and result.objects == []
    assert "checkpoint_class_schema_mismatch" in result.reason


def test_invalid_bbox_fails_without_partial_objects() -> None:
    boxes = FakeBoxes([[2, 3, 20, 21], [-20, 2, 5, 8]], [0.91, 0.8], [2, 2])
    result = M4DetectionTracking(_config(), model=FakeModel(boxes)).detect(*_input())
    assert result.status is Status.FAILED and result.objects == []


def test_checkpoint_path_and_hash_must_be_paired() -> None:
    config = deepcopy(_config())
    config["params"]["detection"]["checkpoint_path"] = "checkpoints/m4/yolo11n.pt"
    with pytest.raises(ValueError, match="configured together"):
        M4DetectionTracking(config)


def test_missing_or_wrong_checkpoint_fails_without_model_loading(tmp_path) -> None:
    config = deepcopy(_config())
    config["params"]["detection"]["checkpoint_path"] = "checkpoints/m4/yolo11n.pt"
    config["params"]["detection"]["checkpoint_sha256"] = "0" * 64
    detector = YOLO11Detector(config, root=tmp_path)
    missing = detector.detect(*_input())
    assert missing.status is Status.FAILED and "checkpoint_missing" in missing.reason
    weight = tmp_path / "checkpoints/m4/yolo11n.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"not a model")
    mismatch = detector.detect(*_input())
    assert mismatch.status is Status.FAILED and "checkpoint_sha256_mismatch" in mismatch.reason
