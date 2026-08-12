"""Call types, sites and recording conditions for every clip in the collection.

Nothing here downloads audio. The metadata sits in parquet shards alongside the
audio, and ``src.data.remote`` fetches only the few text columns, about a fifth of
a megabyte out of each 587 MB shard. The parsed result is written once and read
from disk afterwards.

How a note is read lives in ``src.data.notes``; this module is about where the
notes come from and where the result goes.

Usage:
    python -m src.data.annotations [--config configs/base.yaml] [--refresh]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src import cli
from src.config import Config
from src.data import remote
from src.data.notes import (
    CALL_PREFIX,
    CONDITION_PREFIX,
    Vocabulary,
    collection_code,
    coordinates_of,
    load_vocabulary,
    site_of,
    tag_note,
)
from src.errors import DaturaError

logger = logging.getLogger(__name__)

MIRROR = "ivangtorre/watkins-marine-mammal-full-cuts"
SHARD_INDEX = f"https://huggingface.co/api/datasets/{MIRROR}/parquet/default/train"
# Only what the parse reads. The shards carry an observation_date too, and it was
# fetched for months without anything looking at it: the recording year already
# comes off the clip path, so the column paid for bandwidth and reached no artifact.
WANTED_COLUMNS = ("record_number", "display_name", "note", "location")


class AnnotationError(DaturaError):
    """Raised when the metadata mirror cannot be read or does not match the audio."""


def shard_urls() -> list[str]:
    try:
        urls = json.loads(remote.fetch_text(SHARD_INDEX))
    except json.JSONDecodeError as error:
        raise AnnotationError(f"could not list parquet shards for {MIRROR}") from error
    if not urls:
        raise AnnotationError(f"{MIRROR} reported no parquet shards")
    return urls


def fetch_metadata() -> pd.DataFrame:
    """Pull the text columns out of every shard, leaving the audio behind."""
    frames = []
    fetched = 0
    for number, url in enumerate(shard_urls()):
        reader = remote.RangeReader(url)
        table = pq.ParquetFile(reader).read(columns=list(WANTED_COLUMNS))
        frames.append(table.to_pandas())
        fetched += reader.bytes_read
        logger.info(
            "  shard %d: %d rows, %.1f MB of %.0f MB",
            number,
            table.num_rows,
            reader.bytes_read / 1e6,
            reader.size / 1e6,
        )
    frame = pd.concat(frames, ignore_index=True)
    logger.info("fetched %d rows using %.1f MB in total", len(frame), fetched / 1e6)
    return frame


def annotate(metadata: pd.DataFrame, vocabulary: Vocabulary) -> pd.DataFrame:
    """Turn the raw metadata into one row per clip with flag columns."""
    rows = []
    for record in metadata.itertuples(index=False):
        calls, conditions = tag_note(record.note, vocabulary)
        latitude, longitude = coordinates_of(record.location)
        row: dict[str, object] = {
            "clip_id": str(record.record_number),
            "species": record.display_name,
            "site": site_of(record.location),
            "latitude": latitude,
            "longitude": longitude,
            "note": record.note or "",
            "collection_code": collection_code(record.note),
            "n_call_types": len(calls),
        }
        row.update({f"{CALL_PREFIX}{label}": label in calls for label in vocabulary.call_labels})
        row.update(
            {
                f"{CONDITION_PREFIX}{label}": label in conditions
                for label in vocabulary.condition_labels
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def annotations_path(cfg: Config) -> Path:
    return cfg.paths.metadata / "watkins_annotations.parquet"


def note_columns(vocabulary: Vocabulary) -> list[str]:
    """Every column derived from the note text, in the order ``annotate`` writes them."""
    return [
        "collection_code",
        "n_call_types",
        *[f"{CALL_PREFIX}{label}" for label in vocabulary.call_labels],
        *[f"{CONDITION_PREFIX}{label}" for label in vocabulary.condition_labels],
    ]


def reparse(frame: pd.DataFrame, vocabulary: Vocabulary) -> pd.DataFrame:
    """Recompute every note derived column from the notes already on disk.

    The parquet stores the raw note, so a vocabulary change costs a reparse rather
    than a refetch. Nothing here touches the network.
    """
    parsed = []
    for note in frame["note"].fillna(""):
        calls, conditions = tag_note(note, vocabulary)
        row: dict[str, object] = {
            "collection_code": collection_code(note),
            "n_call_types": len(calls),
        }
        row.update({f"{CALL_PREFIX}{label}": label in calls for label in vocabulary.call_labels})
        row.update(
            {
                f"{CONDITION_PREFIX}{label}": label in conditions
                for label in vocabulary.condition_labels
            }
        )
        parsed.append(row)

    columns = note_columns(vocabulary)
    kept = frame.drop(columns=[c for c in frame.columns if c in columns])
    return pd.concat([kept.reset_index(drop=True), pd.DataFrame(parsed, columns=columns)], axis=1)


def build(cfg: Config, *, refresh: bool = False) -> pd.DataFrame:
    """Read the annotations from disk, fetching and parsing them the first time."""
    path = annotations_path(cfg)
    if path.exists() and not refresh:
        logger.info("annotations already built at %s", path)
        return load(cfg)

    vocabulary = load_vocabulary()
    frame = annotate(fetch_metadata(), vocabulary)
    frame.to_parquet(path, index=False)
    logger.info("wrote %d annotated clips to %s", len(frame), path)
    return frame


def load(cfg: Config) -> pd.DataFrame:
    """The annotations, reparsed in place if the vocabulary has moved since.

    One file serves every configuration, because it covers all 15,248 clips of all
    54 species and nothing in it depends on which species are under study. What it
    does depend on is the vocabulary, and a stale parse feeding a control is the
    failure worth guarding. The note is stored, so healing costs a reparse and no
    network at all.
    """
    path = annotations_path(cfg)
    if not path.exists():
        raise AnnotationError(f"{path} not found; run python -m src.data.annotations first")

    frame = pd.read_parquet(path)
    vocabulary = load_vocabulary()
    missing = [column for column in note_columns(vocabulary) if column not in frame.columns]
    if not missing:
        return frame

    logger.info("vocabulary has moved since %s was written; reparsing %s", path.name, missing)
    frame = reparse(frame, vocabulary)
    frame.to_parquet(path, index=False)
    return frame


def call_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(CALL_PREFIX)]


def condition_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(CONDITION_PREFIX)]


# What a note says about the circumstances of a recording, as opposed to what it
# says about the call. Named here because this module is what parses them, and
# because the controls that consume them were each carrying their own copy of this
# list: the collection code went unmeasured for months partly because adding a
# field meant remembering three separate merges.
CONTEXT_COLUMNS = ("site", "latitude", "longitude", "collection_code")


def context_columns(frame: pd.DataFrame) -> list[str]:
    """Every note derived column that describes the recording rather than the call."""
    return [*CONTEXT_COLUMNS, *condition_columns(frame)]


def attach_context(index: pd.DataFrame, parsed: pd.DataFrame) -> pd.DataFrame:
    """Join the circumstances of each recording onto a frame of windows or clips.

    The cached window index carries what the extractor wrote and nothing else, and
    adding a column to it would invalidate every cached feature array. Joining here
    costs one merge and leaves the cache alone, which is why every control that sees
    the notes is built this way.

    Only what is missing is joined, and that is not a tidiness measure. The manifest
    gained a ``site`` and a ``collection_code`` when the place held out experiment was
    built, so this merge started producing ``site_x`` and ``site_y`` and the plain name
    stopped existing. The call type stage passes its own output back through here, so
    it broke, and nothing caught it because no test runs that stage. Joining only the
    absent columns makes the call idempotent, which is what its callers already assume.
    """
    columns = context_columns(parsed)
    missing = set(columns) - set(parsed.columns)
    if missing:
        raise AnnotationError(f"parsed notes are missing context columns: {sorted(missing)}")

    wanted = [column for column in columns if column not in index.columns]
    if not wanted:
        return index
    return index.merge(parsed[["clip_id", *wanted]], on="clip_id", how="left")


def _log_summary(frame: pd.DataFrame, species: tuple[str, ...]) -> None:
    subset = frame[frame["species"].isin(species)]
    logger.info(
        "\n%d clips annotated, %d of them in the species under study", len(frame), len(subset)
    )

    calls = call_columns(frame)
    coverage = subset.groupby("species")[calls].sum().T.rename_axis("call type").astype(int)
    coverage = coverage[coverage.sum(axis=1) > 0].sort_values(
        by=list(coverage.columns), ascending=False
    )
    logger.info("\nCall type counts per species\n%s", coverage.to_string())

    logger.info(
        "\nClips with no call type recognised: %d of %d",
        int((subset["n_call_types"] == 0).sum()),
        len(subset),
    )
    sites = subset.groupby("species")["site"].nunique()
    logger.info("\nDistinct sites per species\n%s", sites.to_string())


def main(argv: list[str] | None = None) -> int:
    parser = cli.parser_for(__doc__)
    parser.add_argument("--refresh", action="store_true", help="refetch and reparse")
    args = parser.parse_args(argv)

    cfg = cli.prepare(args)
    frame = build(cfg, refresh=args.refresh)
    _log_summary(frame, cfg.dataset.species)
    return 0


if __name__ == "__main__":
    sys.exit(main())
