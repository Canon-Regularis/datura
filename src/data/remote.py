"""Everything this project fetches over HTTP.

One module builds curl commands, so retries, timeouts and redirect handling are
decided once. Nothing else in the codebase shells out to a network tool.

Requests go through curl rather than urllib because Python's stdlib SSL rejects
this machine's certificate chain, and curl carries its own CA bundle.

The reader here matters as much as the fetchers. The Watkins metadata sits inside
parquet shards that also carry the audio, about 4.5 GB in total, and only a few
text columns are wanted. Parquet stores columns separately and records where each
one sits, so a reader that can seek fetches those columns and nothing else: in
practice a fifth of a megabyte out of a 587 MB shard.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

from src.errors import DaturaError

CHUNK_TIMEOUT_SECONDS = 180
HEAD_TIMEOUT_SECONDS = 60
TEXT_TIMEOUT_SECONDS = 60
DOWNLOAD_RETRIES = 10
DOWNLOAD_RETRY_DELAY_SECONDS = 5


class RemoteReadError(DaturaError):
    """Raised when a remote resource cannot be sized, read or fetched."""


def _curl(*arguments: str, timeout: int, text: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["curl", *arguments],
        capture_output=True,
        text=text,
        check=False,
        timeout=timeout + 30,
    )


def _parse_content_length(headers: str) -> int | None:
    """The last content length in a redirect chain, which is the real one."""
    lengths = [
        line.split(":", 1)[1].strip()
        for line in headers.splitlines()
        if line.lower().startswith("content-length")
    ]
    return int(lengths[-1]) if lengths else None


def optional_content_length(url: str) -> int | None:
    """Size of a remote file, or None when the server does not report one.

    Some mirrors answer a HEAD without a length. That is worth tolerating for a
    resumable download, and not worth tolerating for a seeking reader, which is why
    there are two of these.
    """
    result = _curl(
        "-sIL",
        "--max-time",
        str(HEAD_TIMEOUT_SECONDS),
        url,
        timeout=HEAD_TIMEOUT_SECONDS,
        text=True,
    )
    return _parse_content_length(result.stdout) if result.returncode == 0 else None


def content_length(url: str) -> int:
    """Size of a remote file, following redirects."""
    size = optional_content_length(url)
    if size is None:
        raise RemoteReadError(f"no content length reported for {url}")
    return size


def fetch_text(url: str, timeout: int = TEXT_TIMEOUT_SECONDS) -> str:
    """Fetch a small document, such as an index of files."""
    result = _curl("-sL", "--max-time", str(timeout), url, timeout=timeout, text=True)
    if result.returncode != 0:
        raise RemoteReadError(f"curl failed fetching {url}")
    return result.stdout


def download(url: str, destination: Path, *, timeout: int = 24 * 3600) -> Path:
    """Fetch a large file to disk, resuming a partial one rather than restarting it.

    Output goes straight to the file so a multi gigabyte archive never passes
    through this process.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "curl",
            "-L",
            "-C",
            "-",
            "--retry",
            str(DOWNLOAD_RETRIES),
            "--retry-delay",
            str(DOWNLOAD_RETRY_DELAY_SECONDS),
            "--retry-all-errors",
            "-o",
            str(destination),
            url,
        ],
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RemoteReadError(f"curl failed to fetch {url}")
    return destination


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
        result = _curl(
            "-sL",
            "--max-time",
            str(CHUNK_TIMEOUT_SECONDS),
            "-r",
            f"{self._position}-{last}",
            self.url,
            timeout=CHUNK_TIMEOUT_SECONDS,
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
