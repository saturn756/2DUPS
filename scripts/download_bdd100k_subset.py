"""Read selected BDD100K files from the official ZIP with HTTP Range requests.

The official video archive is about 2 TB.  This script reads its ZIP directory
and only the bytes needed for selected members.  It never downloads the archive.
"""

from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.error
import urllib.request
import zipfile


VIDEO_URL = "http://128.32.162.150/bdd100k/bdd100k_videos.zip"
LABEL_URL = "http://128.32.162.150/bdd100k/bdd100k_labels.zip"
BLOCK_SIZE = 1024 * 1024
MAX_BLOCKS = 16


class HTTPRangeFile:
    """Seekable, cached view of a remote HTTP object."""

    def __init__(self, url: str) -> None:
        self.url = url
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            if "bytes" not in response.headers.get("Accept-Ranges", ""):
                raise RuntimeError("Server does not advertise byte ranges")
        self.position = 0
        self.cache: collections.OrderedDict[int, bytes] = collections.OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Invalid seek mode: {whence}")
        if position < 0:
            raise ValueError("Seek before start of archive")
        self.position = position
        return position

    def _get_block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, self.size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )
        expected_range = f"bytes {start}-{end}/{self.size}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(f"Server did not return requested range: {expected_range}")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise IOError(f"Short HTTP range at byte {start}")
                break
            except (OSError, urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))
        self.cache[index] = data
        if len(self.cache) > MAX_BLOCKS:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        pieces = []
        while size > 0:
            block_index, offset = divmod(self.position, BLOCK_SIZE)
            block = self._get_block(block_index)
            piece = block[offset : offset + size]
            if not piece:
                raise IOError("Empty HTTP range read")
            pieces.append(piece)
            self.position += len(piece)
            size -= len(piece)
        return b"".join(pieces)


def safe_target(root: Path, member_name: str) -> Path:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_name}")
    return root.joinpath(*path.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=VIDEO_URL)
    parser.add_argument("--list", action="store_true", help="List a sample of ZIP entries")
    parser.add_argument("--match", help="Only list entries containing this text")
    parser.add_argument("--name", action="append", default=[], help="Exact ZIP member to extract")
    parser.add_argument("--dest", type=Path, default=Path("data/raw/bdd100k"))
    parser.add_argument("--max-bytes", type=int, default=500 * 1024 * 1024)
    args = parser.parse_args()

    if not args.list and not args.name:
        parser.error("Choose --list or at least one --name")
    if len(args.name) > 5:
        parser.error("At most five files can be extracted per run")

    remote = HTTPRangeFile(args.url)
    with zipfile.ZipFile(remote) as archive:
        if args.list:
            entries = [
                item for item in archive.infolist()
                if not item.is_dir() and (not args.match or args.match in item.filename)
            ]
            print(f"archive_bytes={remote.size} matched_entries={len(entries)}", flush=True)
            for item in entries[:30]:
                print(f"{item.filename}\t{item.file_size}\t{item.compress_size}")

        if not args.name:
            return

        entries_by_name = {item.filename: item for item in archive.infolist()}
        missing = [name for name in args.name if name not in entries_by_name]
        if missing:
            raise SystemExit(f"Missing archive members: {missing}")
        selected = [entries_by_name[name] for name in args.name]
        total = sum(item.file_size for item in selected)
        if total > args.max_bytes:
            raise SystemExit(f"Selected files total {total} bytes; limit {args.max_bytes}")

        root = args.dest.resolve()
        for item in selected:
            target = safe_target(root, item.filename)
            if target.exists():
                if target.stat().st_size != item.file_size:
                    raise SystemExit(f"Existing file has unexpected size: {target}")
                print(f"exists {target} ({item.file_size} bytes)", flush=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            print(f"extract {item.filename} ({item.file_size} bytes)", flush=True)
            try:
                with archive.open(item) as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if partial.stat().st_size != item.file_size:
                    raise IOError(f"Unexpected extracted size for {item.filename}")
                partial.replace(target)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
            print(f"saved {target}", flush=True)


if __name__ == "__main__":
    main()
