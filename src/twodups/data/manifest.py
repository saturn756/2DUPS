"""Read and validate the run-level I1 dataset manifest."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..contracts.base import Ref
from ..contracts.i1_dataset_manifest import DatasetManifest
from ..contracts.i1_dataset_manifest import Sample


def _ref(value: dict | None, field: str) -> Ref | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("uri"), str) or not value["uri"]:
        raise ValueError(f"{field} must contain a non-empty uri")
    return Ref(uri=value["uri"], kind=value.get("kind", "file"), checksum=value.get("checksum"))


def canonical_checksum(content: Mapping[str, Any]) -> str:
    """Return the checksum of manifest content, excluding the checksum field."""
    body = {key: value for key, value in content.items() if key != "checksum"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _ref_to_dict(ref: Ref | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    result: dict[str, Any] = {"uri": ref.uri, "kind": ref.kind}
    if ref.checksum is not None:
        result["checksum"] = ref.checksum
    return result


def to_dict(manifest: DatasetManifest, *, include_checksum: bool = True) -> dict[str, Any]:
    """Convert I1 to YAML/JSON data while omitting unset optional fields."""
    payload: dict[str, Any] = {
        "manifest_id": manifest.manifest_id,
        "version": manifest.version,
        "schema_version": manifest.schema_version,
        "source": manifest.source,
        "license": manifest.license,
        "producer_ref": manifest.producer_ref,
        "config_ref": manifest.config_ref,
        "splits": manifest.splits,
        "samples": [],
    }
    for sample in manifest.samples:
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "sequence_id": sample.sequence_id,
            "media_ref": _ref_to_dict(sample.media_ref),
        }
        for key, value in (
            ("annotation_ref", _ref_to_dict(sample.annotation_ref)),
            ("timestamp", sample.timestamp),
            ("slice_tags", sample.slice_tags),
            ("fps", sample.fps),
            ("calibration_ref", _ref_to_dict(sample.calibration_ref)),
        ):
            if value is not None and value != []:
                row[key] = value
        payload["samples"].append(row)
    if include_checksum and manifest.checksum is not None:
        payload["checksum"] = manifest.checksum
    return payload


def load(path: str | Path, *, verify_checksum: bool = True) -> DatasetManifest:
    """Load a YAML manifest and optionally verify its canonical checksum."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest must be a YAML mapping: {path}")
    checksum = canonical_checksum(raw)
    if verify_checksum and raw.get("checksum") not in (None, checksum):
        raise ValueError(f"Manifest checksum mismatch: {path}")

    sample_rows = raw.get("samples")
    if not isinstance(sample_rows, list):
        raise ValueError("samples must be a list")
    samples = []
    for row in sample_rows:
        if not isinstance(row, dict):
            raise ValueError("Each sample must be a mapping")
        media_ref = _ref(row.get("media_ref"), "media_ref")
        if media_ref is None:
            raise ValueError(f"sample {row.get('sample_id')} missing media_ref")
        samples.append(Sample(
            sample_id=row["sample_id"],
            sequence_id=row["sequence_id"],
            media_ref=media_ref,
            annotation_ref=_ref(row.get("annotation_ref"), "annotation_ref"),
            timestamp=row.get("timestamp"),
            slice_tags=list(row.get("slice_tags", [])),
            fps=row.get("fps"),
            calibration_ref=_ref(row.get("calibration_ref"), "calibration_ref"),
        ))
    manifest = DatasetManifest(
        manifest_id=raw["manifest_id"],
        version=str(raw["version"]),
        source=raw["source"],
        license=raw["license"],
        samples=samples,
        splits=raw.get("splits", {}),
        checksum=checksum,
        schema_version=str(raw.get("schema_version", "1.0")),
        producer_ref=raw.get("producer_ref", "m0.dataset_manifest"),
        config_ref=raw.get("config_ref", str(path)),
    )
    problems = validate(manifest)
    if problems:
        raise ValueError("Invalid manifest: " + "; ".join(problems))
    return manifest


def dump(manifest: DatasetManifest, path: str | Path,
         *, header: list[str] | None = None) -> str:
    """Write a canonical YAML manifest and return its content checksum."""
    problems = validate(manifest)
    if problems:
        raise ValueError("Invalid manifest: " + "; ".join(problems))
    path = Path(path)
    content = to_dict(manifest, include_checksum=False)
    checksum = canonical_checksum(content)
    payload = dict(content)
    payload["checksum"] = checksum
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False,
                          default_flow_style=False)
    prefix = ""
    if header:
        prefix = "\n".join(header).rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prefix + text, encoding="utf-8", newline="\n")
    manifest.checksum = checksum
    return checksum


def rewrite(path: str | Path) -> str:
    """Recompute a manifest checksum while preserving its leading comments."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header: list[str] = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            header.append(line)
        else:
            break
    manifest = load(path, verify_checksum=False)
    return dump(manifest, path, header=header or None)


def validate(manifest: DatasetManifest) -> list[str]:
    """Return semantic problems without reading media files."""
    problems: list[str] = []
    if not manifest.manifest_id or not manifest.version:
        problems.append("missing manifest identity or version")
    if not manifest.source or not manifest.license:
        problems.append("missing source or license")
    if not manifest.samples:
        problems.append("empty samples")
    sample_ids = [sample.sample_id for sample in manifest.samples]
    sequence_ids = [sample.sequence_id for sample in manifest.samples]
    if len(sample_ids) != len(set(sample_ids)):
        problems.append("duplicate sample_id")
    if len(sequence_ids) != len(set(sequence_ids)):
        problems.append("duplicate sequence_id")
    for sample in manifest.samples:
        if not sample.sample_id or not sample.sequence_id:
            problems.append("sample missing sample_id or sequence_id")
        if not sample.media_ref.uri:
            problems.append(f"sample {sample.sample_id} missing media_ref")
        if not all(isinstance(tag, str) and tag for tag in sample.slice_tags):
            problems.append(f"sample {sample.sample_id} has invalid slice_tags")
        if len(sample.slice_tags) != len(set(sample.slice_tags)):
            problems.append(f"sample {sample.sample_id} repeats a slice_tag")
        if sample.fps is not None:
            if isinstance(sample.fps, bool) or not isinstance(sample.fps, (int, float)) \
                    or not math.isfinite(float(sample.fps)) or not 0 < float(sample.fps) < 1000:
                problems.append(f"sample {sample.sample_id} has invalid fps")
        if manifest.source == "image_sequence" or sample.media_ref.kind == "image_sequence":
            if sample.fps is None:
                problems.append(f"sample {sample.sample_id} has no fps")
        if sample.calibration_ref is not None and sample.calibration_ref.kind != "camera_params":
            problems.append(f"sample {sample.sample_id} calibration_ref.kind must be 'camera_params'")
    if not isinstance(manifest.splits, dict) or not manifest.splits:
        problems.append("missing splits")
    else:
        assigned: dict[str, str] = {}
        for split, members in manifest.splits.items():
            if not isinstance(members, list) or any(member not in sample_ids for member in members):
                problems.append(f"split {split} references unknown samples")
                continue
            if len(members) != len(set(members)):
                problems.append(f"split {split} contains duplicate samples")
            for member in members:
                if member in assigned and assigned[member] != split:
                    problems.append(f"sample {member} occurs in both {assigned[member]} and {split}")
                assigned[member] = split
        for sample_id in sample_ids:
            if sample_id not in assigned:
                problems.append(f"sample {sample_id} is not assigned to any split")
    return problems


def _resolve_media(root: Path, uri: str) -> Path:
    candidate = (root / uri).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"reference escapes the repository: {uri}")
    return candidate


def _check_ref(ref: Ref, *, root: Path, label: str) -> list[str]:
    try:
        path = _resolve_media(root, ref.uri)
    except ValueError as exc:
        return [f"{label} {exc}"]
    if ref.kind == "image_sequence":
        if not path.is_dir():
            return [f"{label} is not a directory: {ref.uri}"]
    else:
        if not path.is_file():
            return [f"{label} is missing: {ref.uri}"]
        if path.stat().st_size == 0:
            return [f"{label} is empty: {ref.uri}"]
    if ref.checksum is not None and path.is_file():
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != ref.checksum:
            return [f"{label} checksum mismatch: {ref.uri}"]
    return []


def validate_media(manifest: DatasetManifest, *, root: str | Path) -> list[str]:
    """Check media references without decoding images or videos."""
    root_path = Path(root).resolve()
    problems: list[str] = []
    for sample in manifest.samples:
        problems.extend(_check_ref(sample.media_ref, root=root_path,
                                   label=f"sample {sample.sample_id} media"))
        if sample.annotation_ref is not None:
            problems.extend(_check_ref(sample.annotation_ref, root=root_path,
                                       label=f"sample {sample.sample_id} annotation"))
        if sample.calibration_ref is not None:
            problems.extend(_check_ref(sample.calibration_ref, root=root_path,
                                       label=f"sample {sample.sample_id} calibration"))
    return problems
