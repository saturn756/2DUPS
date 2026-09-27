"""编排与配置骨架的冒烟测试。"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from twodups.contracts import (CalibratedFrame, CalibrationState, CalibrationStatus,
                               DetectionSet, ImageSize, Ref, Status)
from twodups.data.bdd100k_loader import LoadedFrame
from twodups.modules import MODULES
from twodups.modules.m1_calibration import M1Calibration
from twodups.modules.m2_preprocess import M2Preprocess
from twodups.pipeline import registry
from twodups.pipeline.runner import iter_detection_only
from twodups.utils.config import REPO_ROOT, load_module

CONFIGS = ["m1_calibration", "m2_preprocess", "m3_geometry",
           "m4_detection_tracking", "m5_segmentation", "m6_topology"]


def test_six_modules_declared() -> None:
    assert MODULES == ("M1", "M2", "M3", "M4", "M5", "M6")


def test_module_declares_interfaces() -> None:
    mod = M1Calibration(config={})
    assert mod.NAME == "M1"
    assert "I2" in mod.INTERFACE_OUT


def test_registry_starts_empty() -> None:
    assert registry.available() == []


@pytest.mark.parametrize("name", CONFIGS)
def test_module_config_loads(name: str) -> None:
    cfg = load_module(REPO_ROOT / "configs" / f"{name}.yaml")
    assert cfg["module"] == f"M{CONFIGS.index(name) + 1}"
    assert "interface_version" in cfg


def test_common_config_has_alignment_policy() -> None:
    cfg = load_module(REPO_ROOT / "configs" / "m1_calibration.yaml")
    assert cfg["alignment"]["variant_policy"] == "same_for_all_branches"
    assert cfg["alignment"]["temporal_variant_policy"] == "skip_i4_on_change"
    assert cfg["alignment"]["allow_future_frames"] is False


def test_detection_only_chain_keeps_i2_i3_i5_aligned(tmp_path: pathlib.Path) -> None:
    class Loader:
        def iter_sample(self, sample_id: str, max_frames: int):
            assert sample_id == "seq"
            assert max_frames == 2
            for index in range(max_frames):
                metadata = CalibratedFrame(
                    sequence_id="seq", frame_id=f"seq:{index:06d}", frame_index=index,
                    image_size=ImageSize(8, 8), media_ref=Ref(f"video.mov#frame={index}"),
                    coordinate_space="pixel",
                    calibration=CalibrationState(CalibrationStatus.MISSING),
                )
                yield LoadedFrame(metadata, np.zeros((8, 8, 3), dtype=np.uint8))

    class Detector:
        def detect(self, variants, images_rgb):
            assert variants.default_variant in images_rgb
            assert images_rgb[variants.default_variant].shape == (8, 8, 3)
            return DetectionSet(
                sequence_id=variants.sequence_id, frame_id=variants.frame_id,
                variant_id=variants.default_variant,
                coordinate_space=variants.coordinate_space, image_size=variants.image_size,
                status=Status.OK, class_schema_ref="test", producer_ref="test.detector",
                config_ref="configs/m4_detection_tracking.yaml",
            )

    config = load_module(REPO_ROOT / "configs/m2_preprocess.yaml")
    config["params"]["output_dir"] = str(tmp_path)
    result = list(iter_detection_only(Loader(), M2Preprocess(config), Detector(),
                                      "seq", start_frame=1, max_frames=1))
    assert len(result) == 1
    assert result[0].frame.metadata.frame_index == 1
    assert result[0].detections.frame_id == result[0].processed.variants.frame_id
    assert result[0].detections.variant_id == result[0].processed.variants.default_variant


def test_config_status_does_not_claim_unimplemented_modules_work() -> None:
    expected = {"m1_calibration": "partial", "m2_preprocess": "initial_implementation",
                "m3_geometry": "planned", "m4_detection_tracking": "partial_detection",
                "m5_segmentation": "planned", "m6_topology": "planned"}
    for name, status in expected.items():
        config = load_module(REPO_ROOT / "configs" / f"{name}.yaml")
        assert config["status"] == status
        if status == "planned":
            assert config["producer_ref"] is None
