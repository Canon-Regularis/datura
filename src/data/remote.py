"""Reading remote files without downloading them whole.

The Watkins metadata lives inside parquet shards that also carry the audio, about
4.5 GB in total, and only a few text columns are wanted. Parquet stores columns
separately and records where each one sits, so a reader that can seek fetches those
few columns and nothing else. In practice that is a fifth of a megabyte out of a
587 MB shard.

Requests go through curl for the same reason the archive download does: Python's
stdlib SSL rejects this machine's certificate chain, and curl carries its own CA
bundle.
"""

from __future__ import annotations

import io
import subprocess

from src.errors import DaturaError

CHUNK_TIMEOUT_SECONDS = 180
HEAD_TIMEOUT_SECONDS = 60


class RemoteReadError(DaturaError):
    """Raised when a remote file cannot be sized or read."""


def content_length(url: str) -> int:
    """Size of a remote file, following redirects."""
    result = subprocess.run(
        ["curl", "-sIL", "--max-time", str(HEAD_TIMEOUT_SECONDS), url],
        capture_output=True,
        text=True,
        check=False,
    )
    lengths = [
        line.split(":", 1)[1].strip()
        for line in result.stdout.splitlines()
        if line.lower().startswith("content-length")
    ]
    if not lengths:
        raise RemoteReadError(f"no content length reported for {url}")
    return int(lengths[-1])


class RangeReader(io.RawIOBase):
    """A seekable, read only file over HTTP.

    Enough of the file interface for pyarrow to open a parquet footer, decide
    which byte ranges it needs, and fetch only those.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._position = 0
        self._size = content_length(url)
        self.request_count = 0
        self.bytes_read = 0

    def __repr__(self) -> str:
        return f"RangeReader({self.url!r}, size={self._size})"

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._position = offset
        elif whence == io.SEEK_CUR:
            self._position += offset
        elif whence == io.SEEK_END:
            self._position = self._size + offset
        else:
            raise RemoteReadError(f"unsupported seek origin {whence}")
        return self._position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._position
        if size <= 0 or self._position >= self._size:
            return b""

        last = min(self._position + size, self._size) - 1
        result = subprocess.run(
            [
                "curl",
                "-sL",
                "--max-time",
                str(CHUNK_TIMEOUT_SECONDS),
                "-r",
                f"{self._position}-{last}",
                self.url,
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RemoteReadError(
                f"curl failed reading bytes {self._position}-{last} of {self.url}"
            )

        block = result.stdout
        self.request_count += 1
        self.bytes_read += len(block)
        self._position += len(block)
        return block

    def readinto(self, buffer) -> int:
        block = self.read(len(buffer))
        buffer[: len(block)] = block
        return len(block)
