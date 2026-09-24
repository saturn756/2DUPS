"""M2 first pass: RGB validation, quality metrics, conditional non-geometric variants.

The public I3 records contain file references for generated PNGs. Pixels are
also returned in ProcessedFrame for immediate in-process consumption.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from ...contracts.base import CalibrationStatus, Ref, Status
from ...contracts.i3_image_variant import ImageVariant, ImageVariantSet, QualityReport
from ...data.bdd100k_loader import LoadedFrame
from ...utils.config import REPO_ROOT, load_module
from ..base import ModuleContext, ModuleRun
from .impl_clahe import enhance
from .impl_denoise import denoise
from .quality_metrics import measure


@dataclass
class ProcessedFrame:
    variants: ImageVariantSet
    quality: QualityReport
    images_rgb: dict[str, np.ndarray]
    report_path: Path

    @property
    def default_image_rgb(self) -> np.ndarray:
        return self.images_rgb[self.variants.default_variant]


class M2Preprocess:
    NAME = "M2"
    INTERFACE_IN = ("I2",)
    INTERFACE_OUT = ("I3",)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_module(REPO_ROOT / "configs/m2_preprocess.yaml")
        selected = self.config.get("impl", {}).get("enhancement", {}).get("selected")
        if selected != "clahe_lab":
            raise ValueError(f"M2 first pass supports only clahe_lab, got {selected!r}")
        if self.config.get("impl", {}).get("rectification") != "deferred":
            raise ValueError("M2 first pass cannot enable rectification without a calibration schema")
        self.params = self.config["params"]
        normalize = self.params.get("normalize", {})
        if (not self.params.get("preserve_raw") or normalize.get("color_space") != "RGB" or
                normalize.get("dtype") != "uint8" or normalize.get("resize") is not None):
            raise ValueError("M2 first pass requires preserved uint8 RGB at source resolution")
        config_bytes = json.dumps(self.config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.config_hash = hashlib.sha256(config_bytes).hexdigest()[:12]
        output_dir = Path(self.params.get("output_dir", "outputs/m2"))
        self.output_dir = (output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir).resolve()

    def run(self, *inputs: Any, ctx: ModuleContext) -> ModuleRun:
        if len(inputs) != 1 or not isinstance(inputs[0], LoadedFrame):
            raise TypeError("M2.run expects one LoadedFrame (I2 metadata plus RGB pixels)")
        processed = self.process(inputs[0], producer_ref=ctx.producer.producer_ref,
                                 config_ref=ctx.producer.config_ref)
        failures = [op["operation"] for op in processed.quality.operations if op["status"] == "failed"]
        return ModuleRun(outputs={"ImageVariantSet": processed.variants,
                                  "QualityReport": processed.quality}, degraded=failures)

    def process(self, frame: LoadedFrame, producer_ref: str | None = None,
                config_ref: str = "configs/m2_preprocess.yaml") -> ProcessedFrame:
        """Process one frame; fail fast on invalid input, fall back on transform errors."""
        self._validate_input(frame)
        started = perf_counter()
        metadata = frame.metadata
        sequence_id = metadata.sequence_id
        producer = producer_ref or self.config.get("producer_ref", "m2.clahe_lab")
        raw = ImageVariant(variant_id="raw", kind="raw", media_ref=metadata.media_ref,
                           coordinate_space=metadata.coordinate_space)
        variants = [raw]
        images = {"raw": frame.image_rgb}
        operations: list[dict] = []
        flags: list[str] = []
        reasons: list[str] = []
        failed = False

        if metadata.calibration.status is CalibrationStatus.MISSING:
            rectification_reason = "calibration_missing"
        elif metadata.calibration.status is CalibrationStatus.STALE:
            rectification_reason = "calibration_stale"
        else:
            rectification_reason = "rectification_deferred_until_calibration_schema"
        operations.append({"operation": "rectification", "status": "skipped",
                           "reason": rectification_reason})
        if metadata.calibration.status is CalibrationStatus.VALID:
            reasons.append(rectification_reason)

        exposure = self.params["exposure_pixels"]
        metrics = measure(frame.image_rgb, int(exposure["under_lt"]), int(exposure["over_gt"]))
        gate = self.params["quality_gate"]
        low = (metrics["brightness_mean"] < gate["low_light"]["brightness_mean_lt"]
               and metrics["underexposed_fraction"] > gate["low_light"]["underexposed_fraction_gt"])
        severe = (metrics["brightness_mean"] < gate["severe_low_light"]["brightness_mean_lt"]
                  and metrics["underexposed_fraction"] > gate["severe_low_light"]["underexposed_fraction_gt"])
        blurry = metrics["blur_laplacian_variance"] < gate["blur"]["laplacian_variance_lt"]
        noisy = metrics["noise_sigma"] > gate["noisy"]["noise_sigma_gt"]
        overexposed = metrics["overexposed_fraction"] > gate["overexposed"]["overexposed_fraction_gt"]
        for condition, name in [(low, "low_light"), (severe, "severe_low_light"),
                                (blurry, "blur"), (noisy, "noisy"), (overexposed, "overexposed")]:
            if condition:
                flags.append(name)

        current_id = "raw"
        if noisy and not blurry:
            denoise_params = self.params["denoise"]
            try:
                denoised = denoise(images[current_id], **denoise_params)
                self._validate_transform(denoised, frame.image_rgb)
                ref = self._save_png(denoised, sequence_id, metadata.frame_index, "denoised")
                variants.append(ImageVariant("denoised", "denoised", ref,
                                             parent_variant_id=current_id,
                                             coordinate_space=metadata.coordinate_space,
                                             operations=[{"name": "bilateral_filter", **denoise_params}]))
                images["denoised"] = denoised
                current_id = "denoised"
                operations.append({"operation": "denoising", "status": "generated",
                                   "input_variant_id": "raw", "output_variant_id": "denoised",
                                   "params": denoise_params})
                reasons.append("noise_sigma_above_threshold")
            except (cv2.error, OSError, ValueError) as exc:
                failed = True
                operations.append({"operation": "denoising", "status": "failed",
                                   "reason": f"{type(exc).__name__}: {exc}"})
                reasons.append("denoising_failed_fallback")
        else:
            operations.append({"operation": "denoising", "status": "skipped",
                               "reason": "blur_gate" if noisy else "noise_below_threshold"})

        if low:
            lowlight = self.params["lowlight"]
            clip_limit = float(lowlight["clahe"]["clip_limit"])
            if blurry:
                clip_limit = min(clip_limit, 1.5)
            gamma = (float(lowlight["gamma"]["severe"] if severe else lowlight["gamma"]["normal"])
                     if lowlight["gamma"]["enabled"] else 1.0)
            if blurry:
                gamma = max(gamma, float(lowlight["gamma"]["normal"]))
            if overexposed:
                gamma = 1.0  # avoid lifting already clipped highlights
            tile_grid = tuple(lowlight["clahe"]["tile_grid_size"])
            params = {"clip_limit": clip_limit, "tile_grid": tile_grid, "gamma": gamma}
            try:
                enhanced = enhance(images[current_id], **params)
                self._validate_transform(enhanced, frame.image_rgb)
                ref = self._save_png(enhanced, sequence_id, metadata.frame_index, "enhanced")
                variants.append(ImageVariant("enhanced", "enhanced", ref,
                                             parent_variant_id=current_id,
                                             coordinate_space=metadata.coordinate_space,
                                             operations=[{"name": "clahe_lab_gamma", **params}]))
                images["enhanced"] = enhanced
                operations.append({"operation": "lowlight_enhancement", "status": "generated",
                                   "input_variant_id": current_id, "output_variant_id": "enhanced",
                                   "params": params})
                current_id = "enhanced"
                reasons.append("low_light_gate")
                if blurry:
                    reasons.append("blur_limited_enhancement")
                if overexposed:
                    reasons.append("overexposure_disabled_gamma")
            except (cv2.error, OSError, ValueError) as exc:
                failed = True
                operations.append({"operation": "lowlight_enhancement", "status": "failed",
                                   "reason": f"{type(exc).__name__}: {exc}"})
                reasons.append("enhancement_failed_fallback")
        else:
            operations.append({"operation": "lowlight_enhancement", "status": "skipped",
                               "reason": "low_light_gate_not_met"})

        if current_id == "raw":
            reasons.append("use_raw_no_accepted_processing")
        variant_set = ImageVariantSet(
            frame_id=metadata.frame_id, sequence_id=sequence_id, variants=variants,
            default_variant=current_id, image_size=metadata.image_size,
            coordinate_space=metadata.coordinate_space,
            calibration_status=metadata.calibration.status,
            manifest_ref=metadata.manifest_ref,
            producer_ref=producer, config_ref=config_ref,
        )
        report = QualityReport(
            frame_id=metadata.frame_id, sequence_id=sequence_id, metrics=metrics,
            quality_flags=flags, operations=operations,
            gate_decision="use_raw" if current_id == "raw" else "use_processed",
            selected_variant_id=current_id, manifest_ref=metadata.manifest_ref, reason=reasons,
            processing_ms=(perf_counter() - started) * 1000,
            status=Status.UNRELIABLE if failed else Status.OK,
            producer_ref=producer, config_ref=config_ref,
        )
        report_path = self._save_report(report, sequence_id, metadata.frame_index)
        return ProcessedFrame(variant_set, report, images, report_path)

    @staticmethod
    def _validate_input(frame: LoadedFrame) -> None:
        if not isinstance(frame, LoadedFrame):
            raise TypeError("M2 requires LoadedFrame with I2 metadata and RGB pixels")
        image, metadata = frame.image_rgb, frame.metadata
        if not re.fullmatch(r"[A-Za-z0-9_-]+", metadata.sequence_id):
            raise ValueError("Unsafe sequence_id for artifact path")
        if metadata.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                image.ndim != 3 or image.shape[2] != 3 or
                image.shape[:2] != (metadata.image_size.height, metadata.image_size.width)):
            raise ValueError("M2 expects uint8 RGB HxWx3 matching I2 image_size")
        if metadata.coordinate_space != "pixel" or metadata.variant_id != "raw":
            raise ValueError("M2 first pass accepts only raw pixel-coordinate frames")

    @staticmethod
    def _validate_transform(image: np.ndarray, raw: np.ndarray) -> None:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != raw.shape:
            raise ValueError("M2 transform changed dtype or image geometry")

    def _artifact_dir(self, sequence_id: str) -> Path:
        path = self.output_dir / self.config_hash / sequence_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _ref(path: Path, kind: str, checksum: str) -> Ref:
        try:
            uri = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            uri = path.as_posix()
        return Ref(uri=uri, kind=kind, checksum=checksum)

    def _save_png(self, image: np.ndarray, sequence_id: str,
                  frame_index: int, variant_id: str) -> Ref:
        success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("OpenCV PNG encoding failed")
        content = encoded.tobytes()
        digest = hashlib.sha256(content).hexdigest()
        path = self._artifact_dir(sequence_id) / f"{frame_index:06d}_{variant_id}_{digest[:12]}.png"
        if not path.exists():
            path.write_bytes(content)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Artifact checksum mismatch: {path}")
        return self._ref(path, "image", f"sha256:{digest}")

    def _save_report(self, report: QualityReport, sequence_id: str, frame_index: int) -> Path:
        content = json.dumps(asdict(report), ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        path = self._artifact_dir(sequence_id) / f"{frame_index:06d}_quality_{digest[:12]}.json"
        if not path.exists():
            path.write_bytes(content)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Quality report checksum mismatch: {path}")
        return path
