"""Acquire the Watkins full cuts archive and unpack the species under study.

The WHOI original at cis.whoi.edu currently redirects to a maintenance page, so
the Internet Archive mirror is the acquisition path.

Every network call goes through ``src.data.remote``, which is the one place in the
project that builds a curl command.

Usage:
    python -m src.data.download [--config configs/base.yaml] [--force]
"""

from __future__ import annotations

import hashlib
import logging
import sys
import zipfile
from pathlib import Path

from src import cli
from src.config import Config
from src.data import remote
from src.errors import DaturaError

logger = logging.getLogger(__name__)


class DownloadError(DaturaError):
    """Raised when the archive cannot be fetched or does not match its digest."""


def file_digest(path: Path, chunk_bytes: int = 8 << 20) -> str:
    """SHA256 of a file, read in chunks so a 6.7 GB archive never lands in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def verify_digest(path: Path, expected: str, what: str = "file", setting: str = "") -> str:
    """Check one fetched file against the digest pinned beside its url.

    Size alone is a weak check. A truncated resume or a mirror serving different
    content can match on length, and everything downstream would then be measured
    against data nobody can reproduce. Both the corpus and the encoder weights arrive
    over the network, so both come through here.
    """
    logger.info("verifying %s", path.name)
    actual = file_digest(path)
    if actual != expected:
        named = setting or f"the digest for this {what}"
        raise DownloadError(
            f"{what} digest mismatch for {path}\n"
            f"  expected {expected}\n"
            f"  actual   {actual}\n"
            f"delete the file and fetch it again, or update {named} if the upstream "
            "release genuinely changed"
        )
    logger.info("digest matches %s...", actual[:16])
    return actual


def verify_archive(cfg: Config, path: Path) -> str:
    """Check the archive against the digest pinned in the config."""
    logger.info("this takes about half a minute for the archive")
    return verify_digest(
        path, cfg.dataset.archive_sha256, what="archive", setting="dataset.archive_sha256"
    )


def download_archive(cfg: Config, *, force: bool = False, verify: bool = True) -> Path:
    """Fetch the zip, resuming a partial file rather than restarting it."""
    cfg.paths.ensure()
    destination = cfg.paths.raw / cfg.dataset.zip_name
    expected = remote.optional_content_length(cfg.dataset.archive_url)

    if destination.exists() and not force:
        actual = destination.stat().st_size
        if expected is None or actual == expected:
            logger.info("archive already present at %s (%.2f GB)", destination, actual / 1e9)
            if verify:
                verify_archive(cfg, destination)
            return destination
        logger.info("resuming download at %.2f of %.2f GB", actual / 1e9, expected / 1e9)

    remote.download(cfg.dataset.archive_url, destination)

    if expected is not None and destination.stat().st_size != expected:
        raise DownloadError(
            f"downloaded {destination.stat().st_size} bytes, expected {expected}; rerun to resume"
        )
    if verify:
        verify_archive(cfg, destination)
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
        logger.info("%d clips requested, %d to extract", len(wanted), len(pending))
        for index, member in enumerate(pending, start=1):
            bundle.extract(member, cfg.paths.raw)
            if index % 500 == 0 or index == len(pending):
                logger.info("  extracted %d/%d", index, len(pending))

    return root


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    parser.add_argument("--force", action="store_true", help="redownload and extract again")
    parser.add_argument("--skip-download", action="store_true", help="extract an existing archive")
    parser.add_argument(
        "--skip-verify", action="store_true", help="skip the SHA256 check on an existing archive"
    )
    args = parser.parse_args(argv)

    cfg = cli.prepare(args)
    archive = cfg.paths.raw / cfg.dataset.zip_name
    if not args.skip_download:
        archive = download_archive(cfg, force=args.force, verify=not args.skip_verify)
    elif not archive.exists():
        raise DownloadError(f"--skip-download given but {archive} does not exist")
    elif not args.skip_verify:
        verify_archive(cfg, archive)

    root = extract_species(cfg, archive, force=args.force)
    logger.info("dataset root: %s", root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
