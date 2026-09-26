"""Keep the committed five-video asset checksum list aligned with I1."""
from __future__ import annotations

from pathlib import Path
import re
import unittest

from twodups.data.manifest import load, validate
from twodups.utils.config import REPO_ROOT


class TestBDD100KAssetList(unittest.TestCase):
    def test_sha256_list_covers_exactly_the_manifest_assets(self) -> None:
        manifest = load(REPO_ROOT / "configs/dataset_manifest.yaml")
        expected = {
            ref.uri
            for sample in manifest.samples
            for ref in (sample.media_ref, sample.annotation_ref)
            if ref is not None
        }
        lines = (REPO_ROOT / "configs/datasets/bdd100k-five.sha256").read_text(encoding="utf-8").splitlines()
        entries = [re.fullmatch(r"([0-9a-f]{64})  (data/raw/bdd100k/[^\s]+)", line) for line in lines]
        self.assertTrue(all(entries), "Each checksum line must be SHA-256 plus a relative asset path")
        paths = [entry.group(2) for entry in entries if entry is not None]
        self.assertEqual(len(paths), len(set(paths)), "Duplicate checksum paths")
        self.assertEqual(set(paths), expected)

    def test_split_leakage_is_rejected(self) -> None:
        manifest = load(REPO_ROOT / "configs/dataset_manifest.yaml")
        sample_id = manifest.splits["train"][0]
        manifest.splits["test"] = [sample_id]
        self.assertTrue(any("both train and test" in issue for issue in validate(manifest)))


if __name__ == "__main__":
    unittest.main()
