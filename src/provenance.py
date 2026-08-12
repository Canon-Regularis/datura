"""Record what produced a result.

A number in a report is only reproducible if you can tell which code, which
package versions and which configuration made it. Every run writes one of these
next to its metrics, so a result that looks wrong six months from now can be
traced back to the commit and the environment that produced it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, Config
from src.results import PROVENANCE_FILE

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "torch",
    "librosa",
    "soundfile",
    "soxr",
)


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_state() -> dict[str, Any]:
    """Commit, branch, and whether the tree had uncommitted changes at run time.

    A dirty tree is recorded rather than refused. The point is to know afterwards,
    since most runs during development happen on a dirty tree.
    """
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def accelerator() -> dict[str, Any]:
    """What torch was actually running on, which changes CNN results slightly."""
    try:
        import torch
    except ImportError:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False, "torch_build": torch.__version__}
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "torch_build": torch.__version__,
        "device": properties.name,
        "total_memory_gb": round(properties.total_memory / 1e9, 2),
        "cuda": torch.version.cuda,
    }


def record(cfg: Config, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything needed to place a result in time, code and environment."""
    return {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "name": cfg.name,
            "source": cfg.source.name,
            "audio_digest": cfg.audio_digest,
            "spectrogram_digest": cfg.spectrogram_digest,
            "species": list(cfg.dataset.species),
            "target_sample_rate": cfg.audio.target_sample_rate,
            "archive_sha256": cfg.dataset.archive_sha256,
        },
        "git": git_state(),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "packages": package_versions(),
        "accelerator": accelerator(),
        **(extra or {}),
    }


def write(cfg: Config, directory: Path, extra: dict[str, Any] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROVENANCE_FILE
    path.write_text(json.dumps(record(cfg, extra), indent=2) + "\n", encoding="utf-8")
    return path
