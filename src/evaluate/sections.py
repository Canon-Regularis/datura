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

# Columns whose value is meaningless once rounded to a fixed number of places.
SIGNIFICANT_COLUMNS = ("p_value", "q_value")


def markdown(frame: pd.DataFrame, decimals: int = 4) -> str:
    """One table, rendered so that a small p value survives being printed.

    Rounding a whole frame to four places turns anything below 0.00005 into a literal
    zero. That is a claim no test can make, and it is how the strongest comparison in
    this project came to be published as `0` while the CSV beside it carried
    9.47e-09. Significant figures keep it readable, and three of them are stable
    across platforms, which is the same reason the committed tables round rather than
    write a full float.
    """
    shown = frame.copy()
    for column in SIGNIFICANT_COLUMNS:
        if column in shown.columns:
            shown[column] = [
                "n/a" if pd.isna(value) else f"{float(value):.3g}" for value in shown[column]
            ]
    return shown.round(decimals).to_markdown(index=False)


def overview(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Every comparison in the configuration, most resolved first.

    Both floors belong here. A family measured against its declared control and again
    against the strongest model that hears nothing produces two rows for each model,
    and dropping the second is how the headline of this project stayed off the table
    that says it lists everything.
    """
    table = pd.concat(frames, ignore_index=True)
    columns = [
        "family",
        "model",
        "floor",
        "margin",
        "low",
        "high",
        "p_value",
        "agreeing",
        "folds",
    ]
    return table[columns].sort_values("p_value").reset_index(drop=True)


def _strongest_floor(against: pd.DataFrame | None) -> list[str]:
    """Every model against the best a model with no audio manages.

    The metadata control was built before anyone knew what else the paperwork
    carried, so it is a floor rather than the floor. Reporting the higher one beside
    it says how much of the gap was equipment and how much was everything else
    written down about the recording.

    The frame arrives already computed, so this section and the overview above cannot
    print different numbers for the same pair.
    """
    if against is None or against.empty:
        return []

    strongest = str(against["floor"].iloc[0])
    return [
        f"### Margin over {strongest}, the strongest model that hears no audio",
        "",
        f"`{strongest}` also sees the site, the coordinates, the noise conditions and the",
        "collection code the field note opens with. None of that is the animal, so this is the",
        "number an audio result has to clear before it is evidence about whales.",
        "",
        markdown(against.drop(columns="floor")),
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
    against_floor: pd.DataFrame | None = None,
    operating: pd.DataFrame | None = None,
) -> list[str]:
    class_names = list(cfg.dataset.species)
    return [
        f"## {family.title}",
        "",
        "### Margin over the metadata control",
        "",
        "The control sees native sample rate, year, clip duration and file size; it sees no",
        "audio. It is a floor rather than the floor, and the table after this one measures",
        "against the highest any model that hears nothing reaches.",
        "",
        markdown(margins.drop(columns="floor", errors="ignore")),
        "",
        *_strongest_floor(against_floor),
        "### Every model, with the range the recordings support",
        "",
        "The interval comes from resampling whole tapes with replacement. Cuts from one tape",
        "are near duplicates, so resampling clips would count the same recording many times",
        "and produce an interval several times too narrow.",
        "",
        markdown(intervals),
        "",
        "### Spread across folds",
        "",
        markdown(tables.headline(comparison)),
        "",
        "### Per species recall",
        "",
        markdown(tables.per_species_recall(comparison, class_names)),
        "",
        *_shared_tape_caveat(shared_tapes),
        "### With and without the giveaway",
        "",
        f"Test clips split by what {_giveaway_phrase(ambiguity)} does to the species. A value",
        "used by one species names it before any audio is heard. A value used by several does",
        "not, and those rows are where audio has to earn its result. A clip carrying no value",
        "at all is a third case, and it is neither of the other two.",
        "",
        "Read `classes_scored` against `classes_total` before the score. A slice can hold",
        "fewer species than the task does, and it is scored over the ones it holds. Averaging",
        "in a class that cannot appear scores it zero and divides by it anyway, which caps the",
        "column and reads as a collapse the predictions do not contain.",
        "",
        markdown(ambiguity),
        "",
        *_operating_curve(operating),
        "### Figures",
        "",
        *[f"- `{path.name}`" for path in figures],
        "",
        "Every figure has a CSV of the same name beside it, or in the model directory.",
        "",
    ]


def _operating_curve(operating: pd.DataFrame | None) -> list[str]:
    """What the audio models are worth when they are allowed to decline.

    Every other number here forces a prediction for every clip, which is the right
    way to compare two representations and the wrong way to describe a tool. A model
    that says nothing on the hard third of its input is more useful than one that
    guesses, and the curve is what says how much more.
    """
    if operating is None or operating.empty:
        return []
    return [
        "### Accuracy against coverage",
        "",
        "Predictions ranked by the probability of the class the model chose, then cut at a",
        "threshold. `coverage` is the share kept, and the row at 1.0 is the score reported",
        "everywhere else. Nothing is refitted: this reads the held out probabilities the",
        "cross validation already wrote.",
        "",
        markdown(operating),
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
    if "kept" in mixed.columns:
        # Count what was scored. A tape whose second species is filtered out reaches
        # the folds with one label and belongs in no caveat about a recall.
        mixed = mixed[mixed["kept"]]
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
        f"Against `{family.control}`, which sees everything written down about the recording",
        "and none of the recording. That includes clip duration, which matters here more than",
        "anywhere else: a note is written against a whole cut, so a longer cut is more likely",
        "to carry any given call whatever the animal was doing.",
        "",
        markdown(margins.drop(columns="floor", errors="ignore")),
        "",
        markdown(intervals),
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
        "Every p value in this document is uncorrected for the number of comparisons reported.",
        "`MULTIPLICITY.md` beside this file adjusts across every comparison in every",
        "configuration at once, which is the number to read before calling one of them a",
        "finding.",
        "",
    ]
