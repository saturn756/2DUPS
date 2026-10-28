"""KITTI Road 适配器：一张图一个样本（该数据集是独立单帧，没有序列与 PTS）。

为什么不用 image_sequence：KITTI Road 的 289 张训练图彼此独立，既没有容器 PTS 也没有
时间顺序；伪造 fps 或帧序会违反 I2 的语义。这里让 ``sample`` 直接对应一个图像文件，
于是 I1 的 ``annotation_ref``（每样本一条）正好可以指向该帧的 GT 掩码。

GT 掩码不进入 I2：它只被评测层按文件名配对读取（见 ``evaluation/kitti_road.py``）。
标定：生成 KITTI Road 清单时把 `calib/` 里的 P2 转成标准 camera_params 文件，
本适配器再按样本解析并输出 ``valid/stale``。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ...contracts.base import CalibrationState, CalibrationStatus, Ref
from ...contracts.i1_dataset_manifest import DatasetManifest, Sample
from ...contracts.i2_calibrated_frame import CalibratedFrame, ImageSize
from ..bdd100k_loader import LoadedFrame
from ..calibration import CalibrationResolver

SOURCE = "KITTI_ROAD"
IMAGE_KIND = "image_file"
MASK_KIND = "segmentation_mask"


class SingleImageAdapter:
    """一张图一个样本的通用实现；子类只改 ``name``/``source``/标定原因。"""

    name = "kitti_road"
    source = SOURCE
    calibration_reason = "kitti_road_calib_file_not_parsed"

    def __init__(self, manifest: DatasetManifest, root: str | Path) -> None:
        if manifest.source != self.source:
            raise ValueError(f"Expected {self.source} manifest, got {manifest.source}")
        self.manifest = manifest
        self.root = Path(root).resolve()
        self.samples = {sample.sample_id: sample for sample in manifest.samples}
        self.calibration = CalibrationResolver(self.root)

    def validate(self, manifest: DatasetManifest) -> list[str]:
        problems: list[str] = []
        if manifest.source != self.source:
            return [f"manifest source is not {self.source}"]
        for sample in manifest.samples:
            if sample.media_ref.kind != IMAGE_KIND:
                problems.append(f"sample {sample.sample_id} media_ref.kind must be {IMAGE_KIND!r}")
            if sample.fps is not None:
                problems.append(f"sample {sample.sample_id} is a single image and must not declare fps")
            if sample.annotation_ref is not None and sample.annotation_ref.kind != MASK_KIND:
                problems.append(f"sample {sample.sample_id} annotation_ref.kind must be {MASK_KIND!r}")
        return problems

    def iter_sample(self, sample_id: str, *, max_frames: int | None = None) -> Iterator[LoadedFrame]:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        sample = self.samples[sample_id]
        path = self._resolve(sample.media_ref.uri)
        image = _decode(path)
        height, width = image.shape[:2]
        image_size = ImageSize(width=width, height=height)
        if sample.calibration_ref is None:
            calibration = CalibrationState(
                status=CalibrationStatus.MISSING,
                reason=self.calibration_reason,
            )
        else:
            calibration = self.calibration.state(sample, image_size)
        yield LoadedFrame(metadata=CalibratedFrame(
            sequence_id=sample.sequence_id,
            frame_id=path.stem,
            frame_index=0,
            image_size=image_size,
            media_ref=Ref(uri=sample.media_ref.uri, kind=IMAGE_KIND),
            coordinate_space="pixel",
            calibration=calibration,
            camera_id="camera_2",
            timestamp=None,
            source_timestamp=None,
            rotation_degrees_clockwise=0,
            manifest_ref=manifest_ref(self.manifest),
        ), image_rgb=image)

    def _resolve(self, uri: str) -> Path:
        path = (self.root / uri).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Media reference escapes repository: {uri}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {uri}")
        return path


class KittiRoadAdapter(SingleImageAdapter):
    """KITTI Road：``training/image_2`` 与其他任意目录下的单帧图像。"""


def manifest_ref(manifest: DatasetManifest) -> str:
    return manifest.manifest_id


def _decode(path: Path) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"Cannot decode image: {path.name}")
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image: {path.name}")
    return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
