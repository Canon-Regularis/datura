"""Acquire the Watkins full cuts archive and unpack the species under study.

The WHOI original at cis.whoi.edu currently redirects to a maintenance page, so
the Internet Archive mirror is the acquisition path.

Downloads shell out to curl. Python's stdlib SSL rejects this machine's
certificate chain, and curl carries its own CA bundle.

Usage:
    python -m src.data.download [--config configs/base.yaml] [--force]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from src.config import Config, load_config


class DownloadError(RuntimeError):
    pass


def remote_size(url: str) -> int | None:
    """Content length of the archive, or None if the server does not report one."""
    result = subprocess.run(
        ["curl", "-sIL", "--max-time", "60", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    sizes = re.findall(r"(?im)^content-length:\s*(\d+)", result.stdout)
    return int(sizes[-1]) if sizes else None


def download_archive(cfg: Config, *, force: bool = False) -> Path:
    """Fetch the zip, resuming a partial file rather than restarting it."""
    cfg.paths.ensure()
    destination = cfg.paths.raw / cfg.dataset.zip_name
    expected = remote_size(cfg.dataset.archive_url)

    if destination.exists() and not force:
        actual = destination.stat().st_size
        if expected is None:
            print(f"archive present at {destination} ({actual / 1e9:.2f} GB), size unverified")
            return destination
        if actual == expected:
            print(f"archive already complete at {destination} ({actual / 1e9:.2f} GB)")
            return destination
        print(f"resuming download at {actual / 1e9:.2f} of {expected / 1e9:.2f} GB")

    command = [
        "curl",
        "-L",
        "-C",
        "-",
        "--retry",
        "10",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "-o",
        str(destination),
        cfg.dataset.archive_url,
    ]
    if subprocess.run(command, check=False).returncode != 0:
        raise DownloadError(f"curl failed to fetch {cfg.dataset.archive_url}")

    if expected is not None and destination.stat().st_size != expected:
        raise DownloadError(
            f"downloaded {destination.stat().st_size} bytes, expected {expected}; rerun to resume"
        )
    return destination


def extract_species(cfg: Config, archive: Path, *, force: bool = False) -> Path:
    """Unpack only the configured species directories.

    The full archive holds 54 species. Unpacking three keeps the working set small
    while the zip itself stays available for the global tape audit.
    """
    root = cfg.paths.raw / cfg.dataset.archive_root
    prefixes = tuple(f"{cfg.dataset.archive_root}/{name}/" for name in cfg.dataset.species)

    with zipfile.ZipFile(archive) as bundle:
        wanted = [
            member
            for member in bundle.namelist()
            if member.startswith(prefixes) and member.lower().endswith(".wav")
        ]
        if not wanted:
            raise DownloadError(
                f"no members matched {prefixes}; check dataset.archive_root and dataset.species"
            )

        pending = [m for m in wanted if force or not (cfg.paths.raw / m).exists()]
        print(f"{len(wanted)} clips requested, {len(pending)} to extract")
        for index, member in enumerate(pending, start=1):
            bundle.extract(member, cfg.paths.raw)
            if index % 500 == 0 or index == len(pending):
                print(f"  extracted {index}/{len(pending)}")

    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--force", action="store_true", help="redownload and re-extract")
    parser.add_argument("--skip-download", action="store_true", help="extract an existing archive")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    archive = cfg.paths.raw / cfg.dataset.zip_name
    if not args.skip_download:
        archive = download_archive(cfg, force=args.force)
    elif not archive.exists():
        raise DownloadError(f"--skip-download given but {archive} does not exist")

    root = extract_species(cfg, archive, force=args.force)
    print(f"dataset root: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
