"""Read and validate the run-level I1 dataset manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def load(path: str | Path) -> DatasetManifest:
    """Load a YAML manifest; fill and verify its canonical content checksum."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest must be a YAML mapping: {path}")
    content = {key: value for key, value in raw.items() if key != "checksum"}
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    checksum = "sha256:" + hashlib.sha256(encoded).hexdigest()
    if raw.get("checksum") not in (None, checksum):
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
        producer_ref=raw.get("producer_ref", "m1.dataset_manifest"),
        config_ref=raw.get("config_ref", str(path)),
    )
    problems = validate(manifest)
    if problems:
        raise ValueError("Invalid manifest: " + "; ".join(problems))
    return manifest


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
    if not isinstance(manifest.splits, dict) or not manifest.splits:
        problems.append("missing splits")
    else:
        for split, members in manifest.splits.items():
            if not isinstance(members, list) or any(member not in sample_ids for member in members):
                problems.append(f"split {split} references unknown samples")
    return problems
