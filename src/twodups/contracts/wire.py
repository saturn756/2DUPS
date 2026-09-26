"""Serialize a contract dataclass with its declared I1–I9 wire discriminator."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_wire(value: Any) -> dict[str, Any]:
    """Create a JSON-compatible payload; payload references, not pixels, cross boundaries."""
    if not is_dataclass(value) or not hasattr(value, "INTERFACE"):
        raise TypeError("Expected an I1–I9 contract dataclass instance")
    payload = asdict(value)
    payload["interface"] = value.INTERFACE
    if value.INTERFACE == "I3":
        payload["kind"] = type(value).__name__
    return payload
