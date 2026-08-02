"""Call types, sites and recording conditions, read from the Watkins field notes.

The collection carries a written note against every cut, for example "Squeal;
chirp. Reverberation present. Good cut." and "Clicks; ship noise." Those notes were
never a schema, so this module reads them as prose: it looks for known terms and
records every one it finds. A clip can be several call types at once, and usually
is.

Nothing here downloads audio. The metadata sits in parquet shards alongside the
audio, and ``src.data.remote`` fetches only the few text columns, about a fifth of
a megabyte out of each 587 MB shard. The parsed result is written once and read
from disk afterwards.

Usage:
    python -m src.data.annotations [--config configs/base.yaml] [--refresh]
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src import cli
from src.config import Config, load_yaml
from src.data.remote import RangeReader
from src.errors import DaturaError

logger = logging.getLogger(__name__)

MIRROR = "ivangtorre/watkins-marine-mammal-full-cuts"
SHARD_INDEX = f"https://huggingface.co/api/datasets/{MIRROR}/parquet/default/train"
WANTED_COLUMNS = ("record_number", "display_name", "note", "location", "observation_date")
VOCABULARY_FILE = "configs/call_types.yaml"

CALL_PREFIX = "call_"
CONDITION_PREFIX = "cond_"


class AnnotationError(DaturaError):
    """Raised when the metadata mirror cannot be read or does not match the audio."""


@dataclass(frozen=True)
class Vocabulary:
    """Terms to look for in a note, and what to record when they are found.

    Terms are held longest first so that a longer phrase wins: a note saying
    "pulsed call" records a pulsed call, and is not also counted as a call.
    """

    call_types: dict[str, list[str]]
    conditions: dict[str, list[str]]

    @property
    def call_labels(self) -> list[str]:
        return list(self.call_types)

    @property
    def condition_labels(self) -> list[str]:
        return list(self.conditions)

    def ordered_terms(self, groups: dict[str, list[str]]) -> list[tuple[str, str]]:
        pairs = [(term, label) for label, terms in groups.items() for term in terms]
        return sorted(pairs, key=lambda pair: -len(pair[0]))


def load_vocabulary(path: str = VOCABULARY_FILE) -> Vocabulary:
    raw = load_yaml(path)
    missing = {"call_types", "conditions"} - set(raw)
    if missing:
        raise AnnotationError(f"{path} is missing sections: {sorted(missing)}")
    return Vocabulary(call_types=raw["call_types"], conditions=raw["conditions"])


def tag_note(note: str | None, vocabulary: Vocabulary) -> tuple[set[str], set[str]]:
    """Every call type and condition mentioned in one note.

    Matched spans are blanked as they are consumed, which is what stops a longer
    phrase from being counted twice under a shorter one.
    """
    if not note:
        return set(), set()

    remaining = note.lower()
    found: dict[str, set[str]] = {"call": set(), "condition": set()}
    for kind, groups in (("call", vocabulary.call_types), ("condition", vocabulary.conditions)):
        for term, label in vocabulary.ordered_terms(groups):
            pattern = re.compile(re.escape(term.lower()))
            if pattern.search(remaining):
                found[kind].add(label)
                remaining = pattern.sub(" " * len(term), remaining)
    return found["call"], found["condition"]


def _first(value: object) -> object:
    """The first element of a sequence, whatever kind of sequence it is.

    Arrow nests these fields as lists, and ``to_pandas`` hands them back as numpy
    arrays rather than lists. Testing for ``list`` alone therefore matched nothing
    and silently emptied every site in the collection.
    """
    if value is None or isinstance(value, str | bytes | dict):
        return value
    try:
        return next(iter(value), None)
    except TypeError:
        return value


def _site_of(location: object) -> str:
    """The named place a recording was made, or an empty string."""
    if not isinstance(location, dict):
        return ""
    name = _first(location.get("name"))
    return str(name) if name else ""


def _coordinates_of(location: object) -> tuple[float | None, float | None]:
    if not isinstance(location, dict):
        return None, None
    point = _first(location.get("coordinates"))
    if isinstance(point, dict):
        return point.get("lat"), point.get("lon")
    return None, None


def shard_urls() -> list[str]:
    result = subprocess.run(
        ["curl", "-sL", "--max-time", "60", SHARD_INDEX],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        urls = json.loads(result.stdout)
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
        reader = RangeReader(url)
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
        latitude, longitude = _coordinates_of(record.location)
        row: dict[str, object] = {
            "clip_id": str(record.record_number),
            "species": record.display_name,
            "site": _site_of(record.location),
            "latitude": latitude,
            "longitude": longitude,
            "note": record.note or "",
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


def build(cfg: Config, *, refresh: bool = False) -> pd.DataFrame:
    """Read the annotations from disk, fetching and parsing them the first time."""
    path = annotations_path(cfg)
    if path.exists() and not refresh:
        logger.info("annotations already built at %s", path)
        return pd.read_parquet(path)

    vocabulary = load_vocabulary()
    frame = annotate(fetch_metadata(), vocabulary)
    frame.to_parquet(path, index=False)
    logger.info("wrote %d annotated clips to %s", len(frame), path)
    return frame


def load(cfg: Config) -> pd.DataFrame:
    path = annotations_path(cfg)
    if not path.exists():
        raise AnnotationError(f"{path} not found; run python -m src.data.annotations first")
    return pd.read_parquet(path)


def call_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(CALL_PREFIX)]


def condition_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(CONDITION_PREFIX)]


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
