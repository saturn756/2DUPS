"""M0 标定通道：相机参数文件的读取、校验与 I2 状态映射。

标定是"可以作为依据"的参数，不是随手填的矩阵。因此这里：

- 文件必须声明来源方法（``provenance.method``），``valid`` 只表示参数可用且适用，
  不表示已经获得深度或三维位置；
- 文件里的 ``image_size`` 指 **M0 输出的帧尺寸**（已应用视频旋转元数据）。与当前帧
  不一致时输出 ``stale`` 并写明两边尺寸，而不是把参数硬套上去；
- 文件缺失、字段越界、未知字段一律失败：清单声称有标定却读不到，不能降级成 missing；
- 校验和（文件字节 sha256）写进 I2 的 ``params_ref``，可追溯。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts.base import CalibrationState, CalibrationStatus, Ref
from ..contracts.i1_dataset_manifest import Sample
from ..contracts.i2_calibrated_frame import ImageSize
from ..utils.config import REPO_ROOT

SCHEMA = "twodups.camera_params/1"
CAMERA_KIND = "camera_params"
PINHOLE = "pinhole"
DISTORTION_LENGTHS = {"radial_tangential": (4, 5, 8)}
REQUIRED_KEYS = {"schema", "camera_id", "image_size", "model", "intrinsics", "distortion",
                 "provenance"}


@dataclass(frozen=True)
class CameraParams:
    """一份通过校验的相机参数；``checksum`` 是参数文件字节的 sha256。"""

    camera_id: str
    image_size: ImageSize
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    coefficients: tuple[float, ...]
    method: str
    checksum: str
    uri: str

    def ref(self) -> Ref:
        return Ref(uri=self.uri, kind=CAMERA_KIND, checksum=self.checksum)

    def reason(self) -> str:
        return f"camera_params_verified:{self.method}"


def _fail(message: str, path: Path) -> ValueError:
    return ValueError(f"Invalid camera params {path.name}: {message}")


def _require_number(value: Any, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _fail(f"{field} must be a finite number", path)
    return float(value)


def _parse_image_size(value: Any, path: Path) -> ImageSize:
    if not isinstance(value, dict) or set(value) != {"width", "height"}:
        raise _fail("image_size must be {width, height}", path)
    width, height = value["width"], value["height"]
    for name, item in (("width", width), ("height", height)):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise _fail(f"image_size.{name} must be a positive integer", path)
    return ImageSize(width=width, height=height)


def parse(payload: Any, *, uri: str, checksum: str, path: Path | None = None) -> CameraParams:
    """校验并构造 ``CameraParams``；任何越界或未知字段都会失败。"""
    path = path or Path(uri)
    if not isinstance(payload, dict):
        raise _fail("root must be an object", path)
    unknown = set(payload) - REQUIRED_KEYS
    if unknown:
        raise _fail(f"unknown fields {sorted(unknown)}", path)
    missing = REQUIRED_KEYS - set(payload)
    if missing:
        raise _fail(f"missing fields {sorted(missing)}", path)
    if payload["schema"] != SCHEMA:
        raise _fail(f"schema must be {SCHEMA!r}", path)
    camera_id = payload["camera_id"]
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise _fail("camera_id must be a non-empty string", path)
    if payload["model"] != PINHOLE:
        raise _fail(f"model must be {PINHOLE!r}", path)
    image_size = _parse_image_size(payload["image_size"], path)

    intrinsics = payload["intrinsics"]
    if not isinstance(intrinsics, dict) or set(intrinsics) != {"fx", "fy", "cx", "cy"}:
        raise _fail("intrinsics must be {fx, fy, cx, cy}", path)
    fx = _require_number(intrinsics["fx"], "intrinsics.fx", path)
    fy = _require_number(intrinsics["fy"], "intrinsics.fy", path)
    cx = _require_number(intrinsics["cx"], "intrinsics.cx", path)
    cy = _require_number(intrinsics["cy"], "intrinsics.cy", path)
    if fx <= 0 or fy <= 0:
        raise _fail("focal lengths must be positive", path)
    if not 0 <= cx < image_size.width or not 0 <= cy < image_size.height:
        raise _fail("principal point must lie inside the image", path)

    distortion = payload["distortion"]
    if not isinstance(distortion, dict) or set(distortion) != {"model", "coefficients"}:
        raise _fail("distortion must be {model, coefficients}", path)
    model = distortion["model"]
    if model not in {"none", *DISTORTION_LENGTHS}:
        raise _fail(f"unsupported distortion model {model!r}", path)
    coefficients = distortion["coefficients"]
    if not isinstance(coefficients, list):
        raise _fail("distortion.coefficients must be a list", path)
    if model == "none":
        if coefficients:
            raise _fail("distortion model 'none' must have empty coefficients", path)
    elif len(coefficients) not in DISTORTION_LENGTHS[model]:
        raise _fail(f"{model} needs {DISTORTION_LENGTHS[model]} coefficients, "
                    f"got {len(coefficients)}", path)
    values = tuple(_require_number(item, "distortion.coefficients[]", path) for item in coefficients)

    provenance = payload["provenance"]
    if not isinstance(provenance, dict) or not isinstance(provenance.get("method"), str) \
            or not provenance["method"].strip():
        raise _fail("provenance.method must be a non-empty string", path)

    return CameraParams(camera_id=camera_id, image_size=image_size, fx=fx, fy=fy, cx=cx, cy=cy,
                        distortion_model=model, coefficients=values,
                        method=provenance["method"], checksum=checksum, uri=uri)


def _resolve(root: Path, uri: str) -> Path:
    candidate = (root / uri).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Calibration reference escapes repository: {uri}") from exc
    return candidate


def load(ref: Ref, *, root: str | Path = REPO_ROOT) -> CameraParams:
    """读取并校验参数文件；``ref.checksum`` 存在时必须匹配文件字节。"""
    if ref.kind != CAMERA_KIND:
        raise ValueError(f"Calibration reference kind must be {CAMERA_KIND!r}, got {ref.kind!r}")
    path = _resolve(Path(root).resolve(), ref.uri)
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera params file: {ref.uri}")
    content = path.read_bytes()
    checksum = "sha256:" + hashlib.sha256(content).hexdigest()
    if ref.checksum is not None and ref.checksum != checksum:
        raise ValueError(f"Camera params checksum mismatch: {ref.uri}")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Camera params is not valid JSON: {ref.uri}: {exc}") from exc
    return parse(payload, uri=ref.uri, checksum=checksum, path=path)


def state_for(params: CameraParams, image_size: ImageSize) -> CalibrationState:
    """参数适用 → valid；分辨率不一致 → stale（并写明两边尺寸）。"""
    if params.image_size != image_size:
        return CalibrationState(
            status=CalibrationStatus.STALE,
            params_ref=params.ref(),
            reason=(f"calibration_image_size_mismatch: params "
                    f"{params.image_size.width}x{params.image_size.height} vs frame "
                    f"{image_size.width}x{image_size.height}"),
        )
    return CalibrationState(status=CalibrationStatus.VALID, params_ref=params.ref(),
                            reason=params.reason())


def parse_kitti_projection(text: str, *, key: str = "P2") -> tuple[float, ...]:
    """Read one KITTI 3x4 projection matrix from a native calib text file."""
    aliases = {key, "P2" if key == "P_rect_02" else "P_rect_02"}
    values: list[float] | None = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, raw_values = line.split(":", 1)
        if name.strip() not in aliases:
            continue
        if values is not None:
            raise ValueError(f"Duplicate KITTI calibration key: {name.strip()}")
        tokens = raw_values.split()
        if len(tokens) != 12:
            raise ValueError(f"KITTI {name.strip()} must contain 12 values")
        try:
            values = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(f"KITTI {name.strip()} contains a non-numeric value") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"KITTI {name.strip()} contains a non-finite value")
    if values is None:
        raise ValueError(f"KITTI calibration is missing {key}")
    return tuple(values)


def kitti_camera_payload(text: str, *, image_size: ImageSize,
                         source_uri: str, source_checksum: str,
                         camera_id: str = "camera_2") -> dict[str, Any]:
    """Convert KITTI's rectified ``P2`` into the project camera schema."""
    projection = parse_kitti_projection(text, key="P2")
    fx, cx, fy, cy = projection[0], projection[2], projection[5], projection[6]
    if fx <= 0 or fy <= 0:
        raise ValueError("KITTI P2 has non-positive focal length")
    if not 0 <= cx < image_size.width or not 0 <= cy < image_size.height:
        raise ValueError("KITTI P2 principal point is outside the image")
    return {
        "schema": SCHEMA,
        "camera_id": camera_id,
        "image_size": {"width": image_size.width, "height": image_size.height},
        "model": PINHOLE,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "distortion": {"model": "none", "coefficients": []},
        "provenance": {
            "method": "KITTI_P_rect_02",
            "source_uri": source_uri,
            "source_checksum": source_checksum,
            "projection_key": "P2",
        },
    }


class CalibrationResolver:
    """按样本解析标定状态；同一文件只读一次，避免逐帧 IO。"""

    def __init__(self, root: str | Path = REPO_ROOT) -> None:
        self.root = Path(root).resolve()
        self._cache: dict[tuple[str, str | None], CameraParams] = {}

    def state(self, sample: Sample, image_size: ImageSize) -> CalibrationState:
        if sample.calibration_ref is None:
            return CalibrationState(
                status=CalibrationStatus.MISSING,
                reason="dataset_does_not_provide_verified_calibration",
            )
        key = (sample.calibration_ref.uri, sample.calibration_ref.checksum)
        params = self._cache.get(key)
        if params is None:
            params = load(sample.calibration_ref, root=self.root)
            self._cache[key] = params
        return state_for(params, image_size)
