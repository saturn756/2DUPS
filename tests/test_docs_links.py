"""The small authoritative documentation set must not point to removed files."""
from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    "README.md",
    "AGENTS.md",
    "docs/architecture/02_模块规范.md",
    "docs/interface/04_接口规范.md",
    "docs/team/01_协作开发与复现规范.md",
    "docs/datasets/08_BDD100K_五段小样本.md",
)


def test_canonical_relative_links_resolve() -> None:
    for name in CANONICAL:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "mailto:")):
                continue
            target_path = target.split("#", 1)[0]
            if target_path:
                assert (path.parent / target_path).exists(), f"{name}: broken link {target}"
