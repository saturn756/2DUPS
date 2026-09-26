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


def test_repository_guidance_has_no_machine_absolute_paths() -> None:
    """Shared instructions must work from each collaborator's own checkout."""
    paths = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "environment.yml"]
    for dirname in ("docs", "configs", "scripts", "src", "tests"):
        paths.extend((ROOT / dirname).rglob("*"))
    machine_path = re.compile(r"/(?:home|Users|path/to|opt|mnt|tmp|var)/|~/|\b10\.10\.\d+\.\d+\b")
    for path in paths:
        if path == Path(__file__).resolve() or path.suffix not in {".md", ".py", ".yaml", ".yml"}:
            continue
        assert machine_path.search(path.read_text(encoding="utf-8")) is None, path
