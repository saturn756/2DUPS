"""YOLO11 detection adapter: I3 pixels in, current-frame I5 detections out.

No model or checkpoint is downloaded here. Real inference requires an existing
local checkpoint and an optional Ultralytics installation. Tests inject a fake
model so the contract can be checked without either dependency.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...contracts.base import BBox, Status
from ...contracts.i3_image_variant import ImageVariantSet
from ...contracts.i5_detection_set import Detection, DetectionSet
from ...utils.config import REPO_ROOT


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YOLO11Detector:
    """Detection-only subset of M4; tracking remains unimplemented."""

    def __init__(self, config: dict[str, Any], *, config_ref: str = "configs/m4_detection_tracking.yaml",
                 model: Any | None = None, root: Path = REPO_ROOT) -> None:
        if Path(config_ref).is_absolute():
            raise ValueError("config_ref must be repository-relative")
        self.config_ref = config_ref
        self.root = root.resolve()
        detection_impl = config.get("impl", {}).get("detection", {})
        if detection_impl.get("selected") != "yolo11n":
            raise ValueError("M4 detection adapter supports only yolo11n")
        self.params = config["params"]["detection"]
        if self.params.get("scope") != "full_frame" or self.params.get("frequency") != "every_frame":
            raise ValueError("M4 first detection pass requires full-frame, every-frame inference")
        self.class_map = {int(key): value for key, value in self.params["class_map"].items()}
        if not self.class_map or any(
            key < 0 or not isinstance(value, dict) or
            not isinstance(value.get("name"), str) or not value["name"] or
            not isinstance(value.get("ref"), str) or not value["ref"]
            for key, value in self.class_map.items()
        ):
            raise ValueError("M4 class_map requires COCO IDs, names and class refs")
        self.score_threshold = float(self.params["score_threshold"])
        self.nms_iou = float(self.params["nms_iou"])
        self.input_size = int(self.params["input_size"])
        self.max_detections = int(self.params["max_detections"])
        if (not 0 <= self.score_threshold <= 1 or not 0 < self.nms_iou < 1 or
                self.input_size <= 0 or self.max_detections <= 0):
            raise ValueError("Invalid YOLO11 detection thresholds or dimensions")
        self.checkpoint_path = self.params.get("checkpoint_path")
        self.checkpoint_sha256 = self.params.get("checkpoint_sha256")
        if bool(self.checkpoint_path) != bool(self.checkpoint_sha256):
            raise ValueError("checkpoint_path and checkpoint_sha256 must be configured together")
        if self.checkpoint_path is not None and not isinstance(self.checkpoint_path, str):
            raise ValueError("checkpoint_path must be a repository-relative string")
        if self.checkpoint_sha256 is not None and not isinstance(self.checkpoint_sha256, str):
            raise ValueError("checkpoint_sha256 must be a string")
        if self.checkpoint_sha256 is not None and (
            len(self.checkpoint_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in self.checkpoint_sha256)
        ):
            raise ValueError("checkpoint_sha256 must be 64 lowercase hex characters")
        self.model = model
        self.class_schema_ref = f"{config_ref}#params.detection.class_map"
        self.producer_ref = "m4.yolo11_detector"

    def _local_checkpoint(self) -> Path:
        if self.checkpoint_path is None:
            raise FileNotFoundError("checkpoint_not_configured")
        relative = Path(self.checkpoint_path)
        if relative.is_absolute() or relative.parts[:1] != ("checkpoints",):
            raise ValueError("checkpoint_path must be relative to repository checkpoints/")
        checkpoint = (self.root / relative).resolve()
        if not checkpoint.is_relative_to(self.root / "checkpoints"):
            raise ValueError("checkpoint_path escapes checkpoints/")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint_missing: {self.checkpoint_path}")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self.checkpoint_sha256:
            raise ValueError("checkpoint_sha256_mismatch")
        return checkpoint

    def _get_model(self) -> Any:
        if self.model is not None:
            return self.model
        checkpoint = self._local_checkpoint()  # Check before import: no implicit download.
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics_not_installed") from exc
        self.model = YOLO(str(checkpoint))
        return self.model

    def _result(self, variants: ImageVariantSet, *, status: Status,
                objects: list[Detection] | None = None, reason: str | None = None) -> DetectionSet:
        return DetectionSet(
            sequence_id=variants.sequence_id, frame_id=variants.frame_id,
            variant_id=variants.default_variant, coordinate_space=variants.coordinate_space,
            image_size=variants.image_size, status=status,
            class_schema_ref=self.class_schema_ref, producer_ref=self.producer_ref,
            config_ref=self.config_ref, objects=objects or [],
            detection_missing=status is not Status.OK, reason=reason,
        )

    def detect(self, variants: ImageVariantSet, images_rgb: Mapping[str, np.ndarray]) -> DetectionSet:
        """Use exactly I3.default_variant; reject invalid carrier pixels."""
        if not isinstance(variants, ImageVariantSet):
            raise TypeError("M4 detection requires ImageVariantSet")
        if variants.default_variant not in images_rgb:
            raise ValueError("Missing pixels for I3.default_variant")
        image = images_rgb[variants.default_variant]
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or
                image.shape != (variants.image_size.height, variants.image_size.width, 3)):
            raise ValueError("M4 requires uint8 RGB pixels matching I3 image_size")
        try:
            model = self._get_model()
            names = model.names
            for class_id, item in self.class_map.items():
                actual = names[class_id]
                if str(actual).strip().lower() != item["name"].strip().lower():
                    raise ValueError(f"checkpoint_class_schema_mismatch: {class_id}")
            # Ultralytics interprets NumPy arrays as BGR. M2's runtime carrier is RGB.
            image_bgr = np.ascontiguousarray(image[:, :, ::-1])
            results = model.predict(
                source=image_bgr, conf=self.score_threshold, iou=self.nms_iou,
                imgsz=self.input_size, max_det=self.max_detections,
                classes=sorted(self.class_map), device=self.params["device"],
                verbose=False, stream=False,
            )
            if len(results) != 1 or tuple(results[0].orig_shape) != image.shape[:2]:
                raise ValueError("YOLO11 result is not aligned with input image")
            boxes = results[0].boxes
            xyxy = _array(boxes.xyxy)
            scores = _array(boxes.conf)
            class_ids = _array(boxes.cls)
            if xyxy.ndim != 2 or xyxy.shape[1] != 4 or scores.shape != (len(xyxy),) or class_ids.shape != (len(xyxy),):
                raise ValueError("Invalid YOLO11 boxes shape")
            objects: list[Detection] = []
            width, height = variants.image_size.width, variants.image_size.height
            for index, (coords, score, raw_class_id) in enumerate(zip(xyxy, scores, class_ids)):
                if not math.isfinite(float(raw_class_id)) or int(raw_class_id) != raw_class_id:
                    raise ValueError("Invalid YOLO11 class ID")
                class_id = int(raw_class_id)
                if class_id not in self.class_map or float(score) < self.score_threshold:
                    continue
                if not np.isfinite(coords).all() or not np.isfinite(score).all():
                    raise ValueError("Non-finite YOLO11 output")
                x1, y1, x2, y2 = (float(value) for value in coords)
                if x1 < -1 or y1 < -1 or x2 > width + 1 or y2 > height + 1:
                    raise ValueError("YOLO11 bbox is outside the source image")
                bbox = BBox(max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
                objects.append(Detection(
                    detection_id=f"{variants.frame_id}:det:{index:04d}",
                    class_ref=self.class_map[class_id]["ref"], bbox=bbox, score=float(score),
                ))
            return self._result(variants, status=Status.OK, objects=objects)
        except (AttributeError, FileNotFoundError, RuntimeError, ValueError,
                KeyError, OSError, TypeError, IndexError) as exc:
            return self._result(variants, status=Status.FAILED,
                                reason=f"{type(exc).__name__}: {exc}")
