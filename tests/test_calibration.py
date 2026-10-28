"""M0 标定通道：参数文件校验、状态映射与适配器接入（不依赖 data/raw/）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from twodups.contracts.base import CalibrationStatus, Ref
from twodups.contracts.i1_dataset_manifest import DatasetManifest, Sample
from twodups.contracts.i2_calibrated_frame import ImageSize
from twodups.data.adapters.image_sequence import ImageSequenceAdapter
from twodups.data.calibration import (
    SCHEMA, CalibrationResolver, CameraParams, kitti_camera_payload, load,
    parse, parse_kitti_projection, state_for,
)
from twodups.data.manifest import dump, load as load_manifest, to_dict, validate

PARAMS = {
    "schema": SCHEMA,
    "camera_id": "front",
    "image_size": {"width": 64, "height": 48},
    "model": "pinhole",
    "intrinsics": {"fx": 60.0, "fy": 61.0, "cx": 32.0, "cy": 24.0},
    "distortion": {"model": "radial_tangential", "coefficients": [-0.2, 0.05, 0.0, 0.0]},
    "provenance": {"method": "synthetic_render", "verified": True},
}


def _write_params(root: Path, payload: dict | None = None, name: str = "camera_params.json") -> Path:
    path = root / name
    path.write_text(json.dumps(payload or PARAMS, ensure_ascii=False), encoding="utf-8")
    return path


def _checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_png(path: Path, width: int = 64, height: int = 48) -> None:
    success, encoded = cv2.imencode(".png", np.full((height, width, 3), 120, dtype=np.uint8))
    assert success
    path.write_bytes(encoded.tobytes())


class TestCameraParamsParsing(unittest.TestCase):
    def test_valid_payload_parses(self) -> None:
        params = parse(PARAMS, uri="camera_params.json", checksum="sha256:abc")
        self.assertIsInstance(params, CameraParams)
        self.assertEqual(params.camera_id, "front")
        self.assertEqual((params.fx, params.fy), (60.0, 61.0))
        self.assertEqual(params.image_size, ImageSize(64, 48))
        self.assertEqual(params.method, "synthetic_render")
        self.assertEqual(params.ref().kind, "camera_params")

    def test_unknown_field_is_rejected(self) -> None:
        payload = dict(PARAMS, extra=1)
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            parse(payload, uri="p", checksum="sha256:abc")

    def test_missing_field_is_rejected(self) -> None:
        payload = {key: value for key, value in PARAMS.items() if key != "provenance"}
        with self.assertRaisesRegex(ValueError, "missing fields"):
            parse(payload, uri="p", checksum="sha256:abc")

    def test_wrong_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema must be"):
            parse(dict(PARAMS, schema="other/1"), uri="p", checksum="sha256:abc")

    def test_non_pinhole_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "model must be"):
            parse(dict(PARAMS, model="fisheye"), uri="p", checksum="sha256:abc")

    def test_non_positive_focal_length_is_rejected(self) -> None:
        payload = dict(PARAMS, intrinsics=dict(PARAMS["intrinsics"], fx=0.0))
        with self.assertRaisesRegex(ValueError, "focal lengths must be positive"):
            parse(payload, uri="p", checksum="sha256:abc")

    def test_principal_point_outside_image_is_rejected(self) -> None:
        payload = dict(PARAMS, intrinsics=dict(PARAMS["intrinsics"], cx=-1.0))
        with self.assertRaisesRegex(ValueError, "principal point"):
            parse(payload, uri="p", checksum="sha256:abc")

    def test_image_size_must_be_positive_ints(self) -> None:
        payload = dict(PARAMS, image_size={"width": 0, "height": 48})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            parse(payload, uri="p", checksum="sha256:abc")

    def test_distortion_model_none_requires_empty_coefficients(self) -> None:
        payload = dict(PARAMS, distortion={"model": "none", "coefficients": [0.1]})
        with self.assertRaisesRegex(ValueError, "must have empty coefficients"):
            parse(payload, uri="p", checksum="sha256:abc")
        ok = dict(PARAMS, distortion={"model": "none", "coefficients": []})
        self.assertEqual(parse(ok, uri="p", checksum="sha256:abc").coefficients, ())

    def test_distortion_coefficient_count_is_enforced(self) -> None:
        payload = dict(PARAMS, distortion={"model": "radial_tangential",
                                          "coefficients": [-0.2, 0.05, 0.0]})
        with self.assertRaisesRegex(ValueError, "coefficients"):
            parse(payload, uri="p", checksum="sha256:abc")

    def test_provenance_method_is_required(self) -> None:
        payload = dict(PARAMS, provenance={"verified": True})
        with self.assertRaisesRegex(ValueError, "provenance.method"):
            parse(payload, uri="p", checksum="sha256:abc")


class TestKittiCalibration(unittest.TestCase):
    P2 = (
        "P0: 1 0 0 0\n"
        "P2: 7.215377000000e+02 0 6.095593000000e+02 4.485728000000e+01 "
        "0 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 "
        "0 0 1 2.745884000000e-03\n"
    )

    def test_p2_projection_is_parsed(self) -> None:
        projection = parse_kitti_projection(self.P2)
        self.assertEqual(len(projection), 12)
        self.assertAlmostEqual(projection[0], 721.5377)
        self.assertAlmostEqual(projection[5], 721.5377)
        self.assertAlmostEqual(projection[2], 609.5593)
        self.assertAlmostEqual(projection[6], 172.854)

    def test_missing_p2_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing P2"):
            parse_kitti_projection("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")

    def test_p2_is_converted_to_camera_params_payload(self) -> None:
        payload = kitti_camera_payload(
            self.P2,
            image_size=ImageSize(1242, 375),
            source_uri="data/raw/kitti_road/training/calib/um_000000.txt",
            source_checksum="sha256:" + "a" * 64,
        )
        params = parse(payload, uri="calibration/um_000000.json", checksum="sha256:abc")
        self.assertEqual(params.camera_id, "camera_2")
        self.assertEqual(params.image_size, ImageSize(1242, 375))
        self.assertAlmostEqual(params.fx, 721.5377)
        self.assertAlmostEqual(params.fy, 721.5377)
        self.assertAlmostEqual(params.cx, 609.5593)
        self.assertAlmostEqual(params.cy, 172.854)
        self.assertEqual(params.distortion_model, "none")
        self.assertEqual(params.coefficients, ())
        self.assertEqual(params.method, "KITTI_P_rect_02")


class TestCalibrationLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = _write_params(self.root)
        self.uri = self.path.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_load_records_checksum_and_reference(self) -> None:
        params = load(Ref(self.uri, "camera_params"), root=self.root)
        self.assertEqual(params.checksum, _checksum(self.path))
        self.assertEqual(params.ref().checksum, params.checksum)

    def test_checksum_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load(Ref(self.uri, "camera_params", "sha256:" + "0" * 64), root=self.root)

    def test_wrong_reference_kind_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "kind must be"):
            load(Ref(self.uri, "file"), root=self.root)

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load(Ref("nope.json", "camera_params"), root=self.root)

    def test_escaping_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            load(Ref("../outside.json", "camera_params"), root=self.root)

    def test_invalid_json_is_rejected(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            load(Ref(self.uri, "camera_params"), root=self.root)

    def test_state_is_valid_when_size_matches(self) -> None:
        params = load(Ref(self.uri, "camera_params"), root=self.root)
        state = state_for(params, ImageSize(64, 48))
        self.assertIs(state.status, CalibrationStatus.VALID)
        self.assertIn("camera_params_verified:synthetic_render", state.reason)
        self.assertEqual(state.params_ref.checksum, params.checksum)

    def test_state_is_stale_when_frame_size_differs(self) -> None:
        params = load(Ref(self.uri, "camera_params"), root=self.root)
        state = state_for(params, ImageSize(128, 96))
        self.assertIs(state.status, CalibrationStatus.STALE)
        self.assertIn("64x48", state.reason)
        self.assertIn("128x96", state.reason)


class TestCalibrationResolverAndAdapters(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "seq").mkdir()
        _write_png(self.root / "seq" / "frame_000000.png")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _manifest(self, *, calibration_ref: Ref | None, width: int = 64,
                  height: int = 48) -> DatasetManifest:
        return DatasetManifest(
            manifest_id="calib", version="1", source="image_sequence", license="test",
            samples=[Sample("s1", "s1", Ref("seq", "image_sequence"),
                            slice_tags=["synthetic"], fps=10.0,
                            calibration_ref=calibration_ref)],
            splits={"train": ["s1"]},
        )

    def test_resolver_reports_missing_without_reference(self) -> None:
        resolver = CalibrationResolver(self.root)
        state = resolver.state(self._manifest(calibration_ref=None).samples[0],
                               ImageSize(64, 48))
        self.assertIs(state.status, CalibrationStatus.MISSING)
        self.assertEqual(state.reason, "dataset_does_not_provide_verified_calibration")
        self.assertIsNone(state.params_ref)

    def test_adapter_uses_calibration_reference(self) -> None:
        path = _write_params(self.root)
        ref = Ref(path.name, "camera_params", _checksum(path))
        manifest = self._manifest(calibration_ref=ref)
        frame = next(ImageSequenceAdapter(manifest, root=self.root).iter_sample("s1"))
        self.assertIs(frame.metadata.calibration.status, CalibrationStatus.VALID)
        self.assertEqual(frame.metadata.calibration.params_ref.checksum, ref.checksum)

    def test_adapter_marks_stale_on_size_mismatch(self) -> None:
        payload = dict(PARAMS, image_size={"width": 128, "height": 96})
        path = _write_params(self.root, payload)
        ref = Ref(path.name, "camera_params", _checksum(path))
        frame = next(ImageSequenceAdapter(self._manifest(calibration_ref=ref),
                                         root=self.root).iter_sample("s1"))
        self.assertIs(frame.metadata.calibration.status, CalibrationStatus.STALE)
        self.assertIn("calibration_image_size_mismatch", frame.metadata.calibration.reason)

    def test_manifest_round_trip_keeps_calibration_reference(self) -> None:
        path = _write_params(self.root)
        ref = Ref(path.name, "camera_params", _checksum(path))
        manifest = self._manifest(calibration_ref=ref)
        self.assertEqual(validate(manifest), [])
        out = self.root / "manifest.yaml"
        dump(manifest, out)
        reloaded = load_manifest(out)
        self.assertEqual(reloaded.samples[0].calibration_ref, ref)
        self.assertEqual(to_dict(reloaded)["samples"][0]["calibration_ref"]["uri"], path.name)

    def test_manifest_rejects_wrong_calibration_kind(self) -> None:
        manifest = self._manifest(calibration_ref=Ref("camera_params.json", "file"))
        problems = validate(manifest)
        self.assertTrue(any("calibration_ref.kind" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
