"""编排与配置骨架的冒烟测试。"""
from __future__ import annotations

import pathlib

import pytest

from twodups.modules import MODULES
from twodups.modules.m1_calibration import M1Calibration
from twodups.pipeline import registry
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
    assert cfg["alignment"]["allow_future_frames"] is False


def test_config_status_does_not_claim_unimplemented_modules_work() -> None:
    expected = {"m1_calibration": "partial", "m2_preprocess": "initial_implementation",
                "m3_geometry": "planned", "m4_detection_tracking": "partial_detection",
                "m5_segmentation": "planned", "m6_topology": "planned"}
    for name, status in expected.items():
        config = load_module(REPO_ROOT / "configs" / f"{name}.yaml")
        assert config["status"] == status
        if status == "planned":
            assert config["producer_ref"] is None
