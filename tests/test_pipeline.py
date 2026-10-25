"""编排与配置骨架的冒烟测试。"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from twodups.contracts import (CalibratedFrame, CalibrationState, CalibrationStatus,
                               DetectionSet, ImageSize, Ref, Status)
from twodups.data.bdd100k_loader import LoadedFrame
from twodups.modules import MODULES
from twodups.modules.m0_data_ingest import M0DataIngest
from twodups.modules.m1_preprocess import M1Preprocess
from twodups.pipeline import registry
from twodups.pipeline.runner import iter_detection_chain
from twodups.utils.config import REPO_ROOT, load_module

CONFIGS = ["m0_data_ingest", "m1_preprocess", "m2_detection",
           "m3_segmentation", "m4_tracking", "m5_integration"]


def test_six_modules_declared() -> None:
    assert MODULES == ("M0", "M1", "M2", "M3", "M4", "M5")


def test_module_declares_interfaces() -> None:
    mod = M0DataIngest(config={})
    assert mod.NAME == "M0"
    assert "I2" in mod.INTERFACE_OUT


def test_registry_starts_empty() -> None:
    assert registry.available() == []


@pytest.mark.parametrize("name", CONFIGS)
def test_module_config_loads(name: str) -> None:
    cfg = load_module(REPO_ROOT / "configs" / f"{name}.yaml")
    assert cfg["module"] == f"M{CONFIGS.index(name)}"
    assert "interface_version" in cfg


def test_common_config_has_alignment_policy() -> None:
    cfg = load_module(REPO_ROOT / "configs" / "m0_data_ingest.yaml")
    assert cfg["alignment"]["variant_policy"] == "same_for_all_branches"
    assert cfg["alignment"]["temporal_variant_policy"] == "fixed_per_run"
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
                config_ref="configs/m2_detection.yaml",
            )

    config = load_module(REPO_ROOT / "configs/m1_preprocess.yaml")
    config["params"]["output_dir"] = str(tmp_path)
    result = list(iter_detection_chain(Loader(), M1Preprocess(config), Detector(),
                                      "seq", start_frame=1, max_frames=1))
    assert len(result) == 1
    assert result[0].frame.metadata.frame_index == 1
    assert result[0].detections.frame_id == result[0].processed.variants.frame_id
    assert result[0].detections.variant_id == result[0].processed.variants.default_variant


def test_config_status_does_not_claim_unimplemented_modules_work() -> None:
    expected = {"m0_data_ingest": "partial", "m1_preprocess": "initial_implementation",
                "m2_detection": "partial_detection", "m3_segmentation": "planned",
                "m4_tracking": "initial_implementation", "m5_integration": "planned"}
    for name, status in expected.items():
        config = load_module(REPO_ROOT / "configs" / f"{name}.yaml")
        assert config["status"] == status
        if status == "planned":
            assert config["producer_ref"] is None
