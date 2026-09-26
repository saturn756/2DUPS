"""模块 M4 入口：检测子块可用，跟踪及完整 run 尚未实现。

职责与边界：docs/architecture/02_模块规范.md
输入输出：I3, I4(可选) → I5, I6（接口见 docs/interface/04_接口规范.md）
配置：configs/m4_detection_tracking.yaml
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts.i3_image_variant import ImageVariantSet
from ...contracts.i5_detection_set import DetectionSet
from ...utils.config import REPO_ROOT, load_module

from ..base import ModuleContext, ModuleRun
from .impl_yolo11 import YOLO11Detector


class M4DetectionTracking:
    NAME = "M4"
    INTERFACE_IN = ("I3",)
    INTERFACE_OUT = ("I5",)

    def __init__(self, config: dict[str, Any] | None = None, *, model: Any | None = None,
                 config_ref: str = "configs/m4_detection_tracking.yaml") -> None:
        self.config = config if config is not None else load_module(REPO_ROOT / config_ref)
        self.detector = YOLO11Detector(self.config, config_ref=config_ref, model=model)

    def detect(self, variants: ImageVariantSet, images_rgb: dict[str, np.ndarray]) -> DetectionSet:
        return self.detector.detect(variants, images_rgb)

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        raise NotImplementedError(
            "M4 跟踪与完整 run 尚未实现；当前仅可调用 detect() 生成 I5"
        )
