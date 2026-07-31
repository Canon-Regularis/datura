"""File access for audio. Returns arrays and headers, nothing more.

Feature code never opens a file. Keeping decode here means the rest of the
pipeline can be tested against synthetic arrays without touching the disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioHeader:
    """Everything readable without decoding the samples."""

    sample_rate: int
    channels: int
    frames: int
    subtype: str
    bytes_on_disk: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def probe(path: str | Path) -> AudioHeader:
    """Read the header only. Used to build the manifest without decoding 4,600 files."""
    path = Path(path)
    info = sf.info(str(path))
    return AudioHeader(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        subtype=str(info.subtype),
        bytes_on_disk=path.stat().st_size,
    )


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode to mono float32 in [-1, 1] and return the signal with its native rate."""
    signal, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = signal.mean(axis=1) if signal.shape[1] > 1 else signal[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sample_rate)
