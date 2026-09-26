"""Executable guard against wire examples drifting from required contract fields."""
from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
import re
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

from twodups import contracts


EXAMPLES = {
    "I1": contracts.DatasetManifest,
    "I2": contracts.CalibratedFrame,
    "I4": contracts.MatchResult,
    "I5": contracts.DetectionSet,
    "I6": contracts.TrackSet,
    "I7": contracts.SegmentationMap,
    "I8": contracts.SceneGraph2D,
    "I9": contracts.EvaluationRecord,
}


def _hydrate(annotation, value):
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        for choice in get_args(annotation):
            if choice is type(None):
                continue
            try:
                return _hydrate(choice, value)
            except (TypeError, ValueError, KeyError):
                pass
        raise ValueError(f"Cannot hydrate {annotation}: {value!r}")
    if origin is list:
        return [_hydrate(get_args(annotation)[0], item) for item in value]
    if origin is dict:
        return value
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _construct(annotation, value)
    return value


def _construct(cls, payload):
    annotations = get_type_hints(cls)
    allowed = {field.name for field in fields(cls)}
    assert isinstance(payload, dict)
    assert payload.keys() <= allowed, f"{cls.__name__}: unknown fields {payload.keys() - allowed}"
    return cls(**{key: _hydrate(annotations[key], value) for key, value in payload.items()})


def _assert_subset(expected, actual):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            assert key in actual
            _assert_subset(value, actual[key])
    elif isinstance(expected, list):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            _assert_subset(left, right)
    else:
        assert expected == actual


def test_all_json_examples_parse_and_cover_required_fields() -> None:
    path = Path(__file__).resolve().parents[1] / "docs/interface/04_接口规范.md"
    blocks = re.findall(r"```json\n(.*?)\n```", path.read_text(encoding="utf-8"), re.DOTALL)
    payloads = [json.loads(block) for block in blocks]
    examples = [value for value in payloads if value.get("interface") in contracts.INTERFACES]
    assert {item["interface"] for item in examples} == set(contracts.INTERFACES)
    for payload in examples:
        cls = EXAMPLES.get(payload["interface"])
        if cls is None:
            cls = {"ImageVariantSet": contracts.ImageVariantSet,
                   "QualityReport": contracts.QualityReport}[payload["kind"]]
        required = {field.name for field in fields(cls)
                    if field.default is MISSING and field.default_factory is MISSING}
        assert required <= payload.keys(), f"{cls.__name__}: missing {required - payload.keys()}"
        assert payload["schema_version"] == "1.0"
        body = {key: value for key, value in payload.items() if key not in {"interface", "kind"}}
        _assert_subset(payload, contracts.to_wire(_construct(cls, body)))


def test_to_wire_emits_interface_discriminator() -> None:
    graph = contracts.SceneGraph2D(
        sequence_id="s", frame_id="s:0", variant_id="raw", coordinate_space="pixel",
        window=contracts.FrameWindow(["s:0"]), producer_ref="m6.rules",
        config_ref="configs/m6_topology.yaml",
    )
    wire = contracts.to_wire(graph)
    assert wire["interface"] == "I8"
    assert wire["geometry_status"] == "image_plane"
    assert json.loads(json.dumps(wire))["window"]["frame_ids"] == ["s:0"]
