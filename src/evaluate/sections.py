"""Composing the report document, section by section.

Everything here returns lines of markdown and writes nothing. The tables come from
``src.evaluate.tables``, the figures are drawn elsewhere, and this module decides
what a reader is shown and in what order.

The order is deliberate. A margin comes before the score it was computed from, and
the columns saying what the design can resolve come before the first table, because
a reader who stops after the first number should still have stopped on the right
one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.evaluate import families, tables

MARGIN_COLUMNS = (
    "Columns beside the margin say what the design resolves. `folds` counts every fold of "
    "every repeat, so a run of ten repeats over five folds shows 50. `low` and `high` bound "
    "the paired difference at 95%, and `p_value` is the corrected resampled test, which "
    "accounts for the training data those folds share. `agreeing` counts the folds that "
    "pointed the same way as the mean, and is worth reading where the p value settles "
    "nothing."
)


def overview(margins: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every comparison in the configuration, most resolved first."""
    table = pd.concat(
        [frame.assign(family=key) for key, frame in margins.items()], ignore_index=True
    )
    columns = ["family", "model", "margin", "low", "high", "p_value", "agreeing", "folds"]
    return table[columns].sort_values("p_value").reset_index(drop=True)


def _strongest_floor(cfg: Config, family: families.Family) -> list[str]:
    """Every model against the best a model with no audio manages.

    The metadata control was built before anyone knew what else the paperwork
    carried, so it is a floor rather than the floor. Reporting the higher one beside
    it says how much of the gap was equipment and how much was everything else
    written down about the recording.
    """
    strongest = families.strongest_floor(cfg, family)
    if strongest is None:
        return []

    against = tables.family_margins(cfg, family, control=strongest)
    return [
        f"### Margin over {strongest}, the strongest model that hears no audio",
        "",
        f"`{strongest}` also sees the site, the coordinates, the noise conditions and the",
        "collection code the field note opens with. None of that is the animal, so this is the",
        "number an audio result has to clear before it is evidence about whales.",
        "",
        against.round(4).to_markdown(index=False),
        "",
    ]


def species_section(
    cfg: Config,
    family: families.Family,
    comparison: pd.DataFrame,
    margins: pd.DataFrame,
    intervals: pd.DataFrame,
    ambiguity: pd.DataFrame,
    figures: list[Path],
    shared_tapes: pd.DataFrame | None = None,
) -> list[str]:
    class_names = list(cfg.dataset.species)
    return [
        f"## {family.title}",
        "",
        "### Margin over the metadata control",
        "",
        "The control sees native sample rate, year, clip duration and file size; it sees no",
        "audio. Its score is the floor an audio model has to clear.",
        "",
        margins.round(4).to_markdown(index=False),
        "",
        *_strongest_floor(cfg, family),
        "### Every model, with the range the recordings support",
        "",
        "The interval comes from resampling whole tapes with replacement. Cuts from one tape",
        "are near duplicates, so resampling clips would count the same recording many times",
        "and produce an interval several times too narrow.",
        "",
        intervals.round(4).to_markdown(index=False),
        "",
        "### Spread across folds",
        "",
        tables.headline(comparison).round(4).to_markdown(index=False),
        "",
        "### Per species recall",
        "",
        tables.per_species_recall(comparison, class_names).round(4).to_markdown(index=False),
        "",
        *_shared_tape_caveat(shared_tapes),
        "### With and without the giveaway",
        "",
        f"Test clips split by whether {_giveaway_phrase(ambiguity)} is used by one species or",
        "by several. On the shared subset the recording cannot identify the species by itself,",
        "so those rows are where audio has to earn its result.",
        "",
        ambiguity.round(4).to_markdown(index=False),
        "",
        "### Figures",
        "",
        *[f"- `{path.name}`" for path in figures],
        "",
        "Every figure has a CSV of the same name beside it, or in the model directory.",
        "",
    ]


def _shared_tape_caveat(shared_tapes: pd.DataFrame | None) -> list[str]:
    """How much of this recall rests on tapes carrying two of the classes.

    Such a tape is one group with two labels. Grouping keeps it whole so nothing
    leaks across a fold boundary, and it still contributes to both recalls, so a
    reader comparing two classes that share recordings is not comparing independent
    evidence. On three species that is one tape and barely worth a clause; on eleven
    it reaches most of one class.
    """
    if shared_tapes is None or shared_tapes.empty:
        return []

    mixed = shared_tapes[shared_tapes["n_under_study"] >= 2]
    if mixed.empty:
        return []

    involved = sorted({name.strip() for row in mixed["under_study"] for name in row.split(",")})
    listed = ", ".join(involved[:-1]) + f" and {involved[-1]}" if len(involved) > 1 else involved[0]
    opening = (
        "One of these recordings carries"
        if len(mixed) == 1
        else f"{len(mixed)} of these recordings carry"
    )
    return [
        f"{opening} more than one of the classes above, across {listed}. Grouping keeps each "
        "tape whole, so none of them crosses a fold boundary, and they still contribute to two "
        "recalls apiece: the classes sharing a tape are not scored on independent evidence.",
        "",
    ]


def _giveaway_phrase(ambiguity: pd.DataFrame) -> str:
    """Name the fields the split was actually made on.

    Read off the table rather than restated beside it. The sentence described only
    the sample rate for as long as the collection code had been in the table, which
    is the drift this reads its way around.
    """
    fields = list(dict.fromkeys(ambiguity["giveaway"]))
    if len(fields) == 1:
        return f"their {fields[0]}"
    return "their " + ", ".join(fields[:-1]) + f" or their {fields[-1]}"


def call_type_section(
    family: families.Family, margins: pd.DataFrame, intervals: pd.DataFrame
) -> list[str]:
    return [
        f"## {family.title}",
        "",
        f"Against `{family.control}`, which sees the site, the coordinates, the collection",
        "the cut came from and the noise conditions, and no audio.",
        "",
        margins.round(4).to_markdown(index=False),
        "",
        intervals.round(4).to_markdown(index=False),
        "",
    ]


def header(cfg: Config, family_count: int) -> list[str]:
    class_names = list(cfg.dataset.species)
    return [
        f"# Results: {cfg.name}",
        "",
        f"Species: {', '.join(class_names)}  ",
        f"Common band: 0 to {cfg.audio.nyquist:.0f} Hz at {cfg.audio.target_sample_rate} Hz  ",
        f"Folds: {cfg.split.n_folds} per split, grouped by tape  ",
        f"Windows: {cfg.audio.window_seconds} s, hop {cfg.audio.hop_seconds} s, "
        f"at most {cfg.audio.max_windows_per_clip} per clip  ",
        f"Families: {family_count}, each a set of models and the control they were measured "
        "against",
        "",
    ]
