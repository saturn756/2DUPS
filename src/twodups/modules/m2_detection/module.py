"""M2 目标检测入口；跟踪属于独立的 M4。

职责与边界：docs/architecture/02_模块规范.md
输入输出：I3 → I5（接口见 docs/interface/04_接口规范.md）
配置：configs/m2_detection.yaml
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts.base import Status
from ...contracts.i3_image_variant import ImageVariantSet
from ...contracts.i5_detection_set import DetectionSet
from ...utils.config import REPO_ROOT, load_module

from ..base import ModuleContext, ModuleRun
from .impl_yolo11 import YOLO11Detector


class M2Detection:
    NAME = "M2"
    INTERFACE_IN = ("I3",)
    INTERFACE_OUT = ("I5",)

    def __init__(self, config: dict[str, Any] | None = None, *, model: Any | None = None,
                 config_ref: str = "configs/m2_detection.yaml") -> None:
        self.config = config if config is not None else load_module(REPO_ROOT / config_ref)
        self.detector = YOLO11Detector(self.config, config_ref=config_ref, model=model)

    def detect(self, variants: ImageVariantSet, images_rgb: dict[str, np.ndarray]) -> DetectionSet:
        return self.detector.detect(variants, images_rgb)

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        if len(inputs) != 2 or not isinstance(inputs[0], ImageVariantSet):
            raise TypeError("M2.run expects ImageVariantSet and images_rgb")
        if not isinstance(inputs[1], dict):
            raise TypeError("M2.run expects the in-process images_rgb mapping")
        detections = self.detect(inputs[0], inputs[1])
        degraded = [] if detections.status is Status.OK else [detections.reason or "detection_failed"]
        return ModuleRun(outputs={"DetectionSet": detections}, degraded=degraded)
