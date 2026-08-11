"""Every number the README prints has to be the number on disk.

The report is regenerated and diffed by CI, so it cannot drift. The README is
written by hand and nothing checked it, which is how five figures in it came to
disagree with the artifacts they were copied from: an interval and a p value that
predated a change to the test, two rows of a table that moved when repeats landed,
and a claim about which epoch a network peaked on.

That matters more here than in most repos. This project's whole argument is that a
number without its uncertainty is worth very little, and the README is the only
part of it that anyone reads.

These tests parse the result tables out of the prose and check each row against the
CSV that produced it. They also check the tables are complete, because a result
that exists in the report and not in the README is the same failure wearing a
different hat.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.config import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
REPORTS = PROJECT_ROOT / "data" / "metadata" / "report"


def rows_of(table_marker: str) -> list[list[str]]:
    """Cells of the first markdown table following a line that contains the marker.

    The tables have no ids and adding some would put scaffolding in prose written
    for people. Anchoring on a phrase from the sentence above each table is enough,
    and it fails loudly if that sentence is ever rewritten away.
    """
    text = README.read_text(encoding="utf-8")
    start = text.find(table_marker)
    assert start != -1, f"no line in the README contains {table_marker!r}"

    rows = []
    for line in text[start:].splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    assert rows, f"no table found after {table_marker!r}"
    return rows[1:]


def matches(printed: str, actual: float) -> bool:
    """Whether a figure in the prose agrees with the artifact at its own precision.

    The README rounds, and it rounds differently in different tables, so the check
    has to read the precision off what was printed rather than impose one.
    """
    text = printed.strip().lstrip("+")
    if not text:
        return True
    if "e" in text.lower():
        # A p value small enough to be written as an exponent. Rounding to decimal
        # places would make every such figure agree with zero, so compare the
        # magnitude instead.
        return abs(float(text) - actual) <= abs(float(text))
    places = len(text.split(".")[1]) if "." in text else 0
    return f"{actual:.{places}f}" == f"{float(text):.{places}f}"


def margins(config: str, control: str | None = None) -> pd.DataFrame:
    """Every comparison in one configuration, built the way the report builds it.

    Not read from ``family_margins.csv``, which drops the two score columns, and not
    from each ``summary.csv`` either. A model and its control are compared only on
    the splits they share, so a network fitted on one split is measured against the
    five folds of the control it shares rather than against the control's fifty
    split mean. Going through the same function the report calls is what keeps the
    README checked against the number it actually quoted.
    """
    from src.config import load_config
    from src.evaluate import families, tables

    cfg = load_config(f"configs/{CONFIG_FILES[config]}")
    frames = [
        tables.family_margins(
            cfg, family, control=control if control in family.names else None
        ).assign(family=family.key)
        for family in families.discover(cfg)
    ]
    return pd.concat(frames, ignore_index=True).set_index("model")


CONFIG_FILES = {
    "base_10k": "base.yaml",
    "base_5k": "base_5k.yaml",
    "wide_10k": "wide.yaml",
    "context_10k": "context.yaml",
}


@pytest.fixture(scope="module")
def report_exists() -> None:
    if not (REPORTS / "base_10k" / "family_margins.csv").exists():
        pytest.skip("report artifacts absent; run python -m src.evaluate.report first")


SPECIES_ROWS = {
    "logbook": "logbook",
    "metadata": "metadata",
    "XGBoost and probe averaged": "xgboost+probe",
    "acoustic descriptors, XGBoost": "xgboost",
    "wav2vec2 probe": "probe",
    "log mel CNN, 0.15 M": "cnn_small",
    "log mel CNN, 2.8 M": "cnn",
}

AGAINST_LOGBOOK = {
    "XGBoost": "xgboost",
    "wav2vec2 probe": "probe",
    "CNN 2.8 M": "cnn",
    "CNN 0.15 M": "cnn_small",
    "logbook against metadata": "logbook",
}

WIDE_ROWS = {
    "logbook": "logbook",
    "metadata": "metadata",
    "acoustic, XGBoost": "xgboost",
    "wav2vec2 probe": "probe",
}

CONTEXT_ROWS = {
    "logbook": "logbook",
    "metadata": "metadata",
    "acoustic descriptors, XGBoost": "xgboost",
    "wav2vec2 probe": "probe",
    "XGBoost and probe averaged": "xgboost+probe",
}

COVERAGE_ROWS = {
    "acoustic descriptors, XGBoost": "xgboost",
    "wav2vec2 probe": "probe",
    "log mel CNN, 0.15 M": "cnn_small",
    "XGBoost and probe averaged": "xgboost+probe",
}

AMBIGUITY_ROWS = {
    "logbook": "logbook",
    "metadata": "metadata",
    "acoustic, XGBoost": "xgboost",
    "wav2vec2 probe": "probe",
    "log mel CNN": "cnn_small",
}


def test_the_species_scores_match_the_artifact(report_exists):
    for cells in rows_of("| model | macro-F1 | audio |"):
        label, score = cells[0], cells[1]
        name = SPECIES_ROWS[label]
        summary = pd.read_csv(REPORTS / "base_10k" / name / "summary.csv")
        row = summary[summary["metric"] == "macro_f1"].iloc[0]

        mean, spread = (part.strip() for part in score.split("±"))
        assert matches(mean, row["mean"]), f"{label}: macro-F1"
        assert matches(spread, row["std"]), f"{label}: spread"


def test_the_species_table_lists_every_model(report_exists):
    printed = {cells[0] for cells in rows_of("| model | macro-F1 | audio |")}
    on_disk = {
        label
        for label, name in SPECIES_ROWS.items()
        if (REPORTS / "base_10k" / name / "summary.csv").exists()
    }
    assert on_disk == printed, f"printed {sorted(printed)} against {sorted(on_disk)} on disk"


def test_the_margins_against_the_logbook_match_the_artifact(report_exists):
    """The comparison that actually resolves, so it has to be right.

    Three rows measure an audio model against the logbook. The fourth measures the
    logbook against the metadata control, which is a different floor, so it comes
    from a different table.
    """
    against_logbook = margins("base_10k", control="logbook")
    against_control = margins("base_10k")

    for cells in rows_of("| comparison | margin | 95% interval | p | agreeing |"):
        label, margin, interval, p_value, agreeing = cells[:5]
        name = AGAINST_LOGBOOK[label]
        table = against_control if name == "logbook" else against_logbook
        row = table.loc[name]

        assert matches(margin, row["margin"]), f"{label}: margin"
        low, high = (part.strip() for part in interval.split(" to "))
        assert matches(low, row["low"]), f"{label}: interval low"
        assert matches(high, row["high"]), f"{label}: interval high"
        assert matches(p_value, row["p_value"]), f"{label}: p value"
        assert agreeing.split(" of ")[0] == str(row["agreeing"]), f"{label}: folds agreeing"
        assert agreeing.split(" of ")[1] == str(row["folds"]), f"{label}: fold count"


def test_the_ambiguity_table_matches_the_artifact(report_exists):
    """One row per model, one column per giveaway on each side of its split."""
    table = pd.read_csv(REPORTS / "base_10k" / "ambiguity_breakdown.csv")

    def score(field: str, side: str, model: str) -> float:
        rows = table[(table["giveaway"] == field) & (table["subset"] == f"{field} {side}")]
        return float(rows.set_index("model").loc[model, "macro_f1_mean"])

    for cells in rows_of("| model | rate unique | rate shared | code unique | code absent |"):
        label, rate_unique, rate_shared, code_unique, code_absent = cells[:5]
        name = AMBIGUITY_ROWS[label]
        rate, code = "native sample rate", "collection code"
        assert matches(rate_unique, score(rate, "unique to a species", name)), f"{label}: rate"
        assert matches(rate_shared, score(rate, "shared by species", name)), f"{label}: rate"
        assert matches(code_unique, score(code, "unique to a species", name)), f"{label}: code"
        assert matches(code_absent, score(code, "not recorded", name)), f"{label}: code"


def test_no_collection_code_in_the_narrow_set_is_shared(report_exists):
    """The claim the README makes about its own fourth column.

    The subset used to be labelled as clips whose code several species share. No such
    clip exists here, and the bucket held the clips carrying no code at all, so the
    label described the opposite of its contents.
    """
    table = pd.read_csv(REPORTS / "base_10k" / "ambiguity_breakdown.csv")
    code = table[table["giveaway"] == "collection code"]
    subsets = set(code["subset"])

    assert "collection code shared by species" not in subsets, sorted(subsets)
    absent = code[code["subset"] == "collection code not recorded"]
    assert not absent.empty, sorted(subsets)
    assert set(absent["clips"]) == {359}
    assert set(absent["classes_scored"]) == {2}, "two of the three species carry no code"
    assert set(absent["classes_total"]) == {3}


def call_type_key(label: str) -> str:
    """The result directory a call type row is quoting."""
    species, call = (part.strip() for part in label.split(",")[:2])
    name = f"calltype_{species.replace(' ', '').lower()}_{call.replace(' ', '_')}"
    return f"{name}_cnn_small" if label.rstrip().endswith("CNN") else name


def test_the_call_type_table_matches_the_artifact(report_exists):
    table = margins("base_10k")
    for cells in rows_of("| task | audio | control | margin | p | agreeing |"):
        label, audio, control, margin, p_value, agreeing = cells[:6]
        name = call_type_key(label)
        row = table.loc[name]

        assert matches(margin, row["margin"]), f"{label}: margin"
        assert matches(p_value, row["p_value"]), f"{label}: p value"
        assert agreeing.split(" of ")[0] == str(row["agreeing"]), f"{label}: folds agreeing"
        assert agreeing.split(" of ")[1] == str(row["folds"]), f"{label}: fold count"
        assert matches(audio, row["mean"]), f"{label}: audio score"
        assert matches(control, row["control"]), f"{label}: control score"
        assert name == call_type_key(label)


def test_the_call_type_table_lists_every_result(report_exists):
    printed = {call_type_key(cells[0]) for cells in rows_of("| task | audio | control |")}
    on_disk = {name for name in margins("base_10k").index if name.startswith("calltype_")}
    assert on_disk == printed, (
        f"in the report but not the README: {sorted(on_disk - printed)}; "
        f"in the README but not the report: {sorted(printed - on_disk)}"
    )


def test_the_narrow_band_claims_match_the_artifact(report_exists):
    """The 5 kHz result is quoted in prose rather than in a table."""
    text = README.read_text(encoding="utf-8")
    table = margins("base_5k")

    floor = margins("base_5k", control="logbook").loc["xgboost"]
    assert f"{abs(floor['margin']):.3f} below the logbook" in text
    assert f"p = {floor['p_value']:.1e}" in text

    control = table.loc["xgboost"]
    assert f"{abs(control['margin']):.3f} below the metadata control" in text
    assert f"p = {control['p_value']:.3f}" in text
    assert f"{control['agreeing']} of {control['folds']}" in text


def test_the_wide_species_table_matches_both_artifacts(report_exists):
    """One column per configuration, so a stale figure in either shows up here."""
    for cells in rows_of("| model | 3 species | 11 species |"):
        label, narrow, wide = cells[:3]
        name = WIDE_ROWS[label]
        for config, printed in (("base_10k", narrow), ("wide_10k", wide)):
            summary = pd.read_csv(REPORTS / config / name / "summary.csv")
            row = summary[summary["metric"] == "macro_f1"].iloc[0]
            assert matches(printed, row["mean"]), f"{label}, {config}"


def test_the_wide_margins_match_the_artifact(report_exists):
    """The wide result is quoted in prose rather than in a table."""
    text = " ".join(README.read_text(encoding="utf-8").split())
    against_logbook = margins("wide_10k", control="logbook").loc["xgboost"]
    against_control = margins("wide_10k").loc["logbook"]

    assert f"{abs(against_logbook['margin']):.3f} below the logbook" in text
    assert f"p = {against_logbook['p_value']:.1e}" in text
    assert f"{against_logbook['agreeing']} of {against_logbook['folds']} agreeing" in text
    assert f"gap of {against_control['margin']:.3f}" in text
    assert f"p = {against_control['p_value']:.1e}" in text


def test_the_epoch_claims_match_the_training_curves(report_exists):
    """Which epoch each fold peaked on, quoted for both networks."""
    text = README.read_text(encoding="utf-8")
    for name in ("cnn", "cnn_small"):
        history = pd.read_csv(REPORTS / "base_10k" / name / "history.csv")
        peaks = history.loc[history.groupby("fold")["val_macro_f1"].idxmax()]
        printed = ", ".join(str(epoch) for epoch in peaks["epoch"].tolist()[:-1])
        assert printed in text, (
            f"{name} peaks at {peaks['epoch'].tolist()} and the README disagrees"
        )


def test_no_number_is_quoted_without_the_artifact_that_backs_it():
    """Every table row in the results sections names a result that exists.

    A guard against the cheapest kind of drift: a row copied in by hand for a run
    that was never committed.
    """
    text = README.read_text(encoding="utf-8")
    assert "0.626" not in text, "the pre-correction cnn_small interval is back"
    assert re.search(r"\b0\.893\b", text) is None, "the pre-repeats ambiguity figure is back"
    assert "between epochs 0 and 4 on every single fold" not in text


def test_the_corpus_caption_names_both_filters(report_exists):
    """The dominant exclusion is the clip length, and the caption used to omit it.

    It read as though the sample rate floor were the only thing dropping clips, which
    put the emphasis on the confound the project was already chasing and hid a filter
    seven times larger.
    """
    dropped = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "audit_dropped_base_10k.csv")
    by_reason = dropped.groupby("drop_reason")["clips"].sum()

    short = int(by_reason["clip_too_short"])
    low_rate = int(by_reason["native_rate_below_target"])
    assert short > low_rate, "the caption below assumes the length filter is the larger one"

    text = README.read_text(encoding="utf-8")
    assert f"{low_rate} humpback clips" in text
    assert f"{short} more" in text


def test_the_coverage_table_matches_the_artifact(report_exists):
    """The operating curve is what the prediction command prints a band from.

    A figure overstated here would put a confidence claim beside an answer that the
    held out data does not support, which is worse than a wrong score in a comparison
    table because somebody would act on it.
    """
    path = REPORTS / "base_10k" / "coverage.csv"
    if not path.exists():
        pytest.skip("coverage.csv absent; run python -m src.evaluate.report first")
    curve = pd.read_csv(path)

    for cells in rows_of("| model | every clip | most confident 70% |"):
        label, full, seventy, thirty = cells[:4]
        rows = curve[curve["model"] == COVERAGE_ROWS[label]].set_index("coverage")
        for printed, level in ((full, 1.0), (seventy, 0.7), (thirty, 0.3)):
            assert matches(printed, rows.loc[level, "accuracy"]), f"{label} at {level:.0%}"

    # The opening paragraph quotes the best of these as a percentage, and it is the
    # first number anybody reads.
    best = curve[(curve["model"] == "xgboost+probe") & (curve["coverage"] == 0.7)]
    assert f"takes accuracy to {best.iloc[0]['accuracy']:.1%}" in " ".join(
        README.read_text(encoding="utf-8").split()
    )


def test_the_coverage_table_lists_every_model_with_a_curve(report_exists):
    path = REPORTS / "base_10k" / "coverage.csv"
    if not path.exists():
        pytest.skip("coverage.csv absent")
    printed = {COVERAGE_ROWS[cells[0]] for cells in rows_of("| model | every clip |")}
    on_disk = set(pd.read_csv(path)["model"])
    assert on_disk - printed <= {"cnn"}, f"a curve exists for {sorted(on_disk - printed)}"


def test_the_confident_mistake_claim_matches_the_predictions(report_exists):
    """The number that decides which model the prediction command ships.

    A threshold cannot fix a model that is wrong while sure, so this is the property
    the default was chosen on, and it is quoted in the prose rather than in a table.
    """
    from src import scoring
    from src.config import load_config
    from src.results import predictions_path

    cfg = load_config("configs/base.yaml")
    columns = scoring.probability_columns(len(cfg.dataset.species))

    def confidently_wrong(name: str) -> float:
        frame = pd.read_parquet(predictions_path(cfg, name))
        probabilities = frame[columns].to_numpy()
        wrong = probabilities.argmax(axis=1) != frame["label"].to_numpy()
        return float((wrong & (probabilities.max(axis=1) > 0.9)).mean()) * 100

    text = README.read_text(encoding="utf-8")
    network, trees = confidently_wrong("cnn_small"), confidently_wrong("xgboost")
    assert f"{network:.2f}% of them and XGBoost on {trees:.3f}%" in text
    assert trees < network, "the shipped model has to be the one that is confidently wrong least"


def test_the_thresholds_quoted_are_the_ones_predict_would_use(report_exists):
    """Two models needing very different cut offs is why neither is hardcoded."""
    from src import predict
    from src.config import load_config

    cfg = load_config("configs/base.yaml")
    if predict.curve_for(cfg, "xgboost") is None:
        pytest.skip("coverage.csv absent")

    # Whitespace collapsed, because these two numbers sit either side of a line break
    # and a reflow of the paragraph is not a change to the claim.
    text = " ".join(README.read_text(encoding="utf-8").split())
    trees = predict.threshold_for(predict.curve_for(cfg, "xgboost"))
    network = predict.threshold_for(predict.curve_for(cfg, "cnn_small"))
    assert f"XGBoost needs {trees:.3f} and `cnn_small` needs {network:.3f}" in text


def test_the_chance_baselines_are_measured_rather_than_assumed(report_exists):
    """0.333 is quoted as the floor the audio models are read against.

    Three classes do not make macro-F1 of a guess exactly one third, because these
    classes are far from balanced, so the figure is computed on the real labels.
    """
    import numpy as np

    from src import scoring
    from src.config import load_config
    from src.results import predictions_path

    cfg = load_config("configs/base.yaml")
    labels = pd.read_parquet(predictions_path(cfg, "xgboost"))["label"].to_numpy()
    n_classes = len(cfg.dataset.species)
    shares = np.bincount(labels, minlength=n_classes) / len(labels)
    rng = np.random.default_rng(0)

    def guessing(probabilities: np.ndarray | None) -> float:
        draws = [
            scoring.from_counts(
                labels, rng.choice(n_classes, len(labels), p=probabilities), n_classes
            )["macro_f1"]
            for _ in range(50)
        ]
        return float(np.mean(draws))

    text = README.read_text(encoding="utf-8")
    stratified = guessing(shares)
    uniform = guessing(np.full(n_classes, 1 / n_classes))
    majority = scoring.from_counts(labels, np.full(len(labels), int(shares.argmax())), n_classes)[
        "macro_f1"
    ]

    assert f"Chance is {stratified:.3f} for a guess drawn from the class shares" in text
    assert f"{uniform:.3f} for a uniform guess" in text
    assert f"{majority:.3f} for" in text


def test_the_context_table_matches_both_reports(report_exists):
    """One column per fold rule, so a stale figure in either shows up here.

    The change column is the number the section exists to report, and it is the
    difference of two artifacts rather than a figure of its own, so it is recomputed
    rather than read.
    """
    if not (REPORTS / "context_10k" / "xgboost" / "summary.csv").exists():
        pytest.skip("context_10k absent; run the pipeline on configs/context.yaml")

    for cells in rows_of("| model | tape held out | place held out | change |"):
        label, held_out_tape, held_out_context, change = cells[:4]
        name = CONTEXT_ROWS[label]

        scores = {}
        for config in ("base_10k", "context_10k"):
            summary = pd.read_csv(REPORTS / config / name / "summary.csv")
            scores[config] = summary[summary["metric"] == "macro_f1"].iloc[0]["mean"]

        assert matches(held_out_tape, scores["base_10k"]), f"{label}: tape held out"
        assert matches(held_out_context, scores["context_10k"]), f"{label}: context held out"
        assert matches(change, scores["context_10k"] - scores["base_10k"]), f"{label}: change"


def test_the_context_table_lists_only_models_that_were_refitted(report_exists):
    directory = REPORTS / "context_10k"
    if not directory.exists():
        pytest.skip("context_10k absent")
    printed = {CONTEXT_ROWS[cells[0]] for cells in rows_of("| model | tape held out |")}
    on_disk = {path.parent.name for path in directory.glob("*/summary.csv")}
    assert printed <= on_disk, f"the README quotes {sorted(printed - on_disk)} with no result"


def test_the_context_decomposition_matches_the_artifacts(report_exists):
    """The table that says how much of the drop is the place and how much is geometry.

    This is the correction the coarseness control forced. The section claimed the place
    cost 0.141 macro-F1 when the control shows most of that is five folds over two dozen
    lopsided groups, so every row here is recomputed rather than quoted.
    """
    import numpy as np

    from src import scoring
    from src.config import load_config
    from src.data.manifest import load_manifest
    from src.results import predictions_path

    base, context = load_config("configs/base.yaml"), load_config("configs/context.yaml")
    control = REPORTS / "context_shuffled_10k" / "xgboost" / "summary.csv"
    if not control.exists():
        pytest.skip("the coarseness control has not been fitted")

    columns = scoring.probability_columns(len(base.dataset.species))
    grouped = load_manifest(context, kept_only=True)
    shared = set(grouped.loc[grouped[context.split.group_column] != "", "clip_id"])

    def per_fold(frame: pd.DataFrame) -> float:
        scores = frame.groupby(["repeat", "fold"]).apply(
            lambda group: scoring.from_counts(
                group["label"].to_numpy(),
                group[columns].to_numpy().argmax(axis=1),
                len(base.dataset.species),
            )["macro_f1"],
            include_groups=False,
        )
        return float(np.mean(scores))

    predictions = pd.read_parquet(predictions_path(base, "xgboost"))
    expected = [
        per_fold(predictions),
        per_fold(predictions[predictions["clip_id"].isin(shared)]),
        float(pd.read_csv(control).set_index("metric").loc["macro_f1", "mean"]),
        float(
            pd.read_csv(REPORTS / "context_10k" / "xgboost" / "summary.csv")
            .set_index("metric")
            .loc["macro_f1", "mean"]
        ),
    ]

    rows = rows_of("| what is being scored | macro-F1 | cost |")
    assert len(rows) == len(expected), (
        f"the table has {len(rows)} rows against {len(expected)} steps"
    )
    for row, actual in zip(rows, expected, strict=True):
        assert matches(row[1], actual), f"{row[0]}: {row[1]} against {actual:.4f}"

    # And each cost is the step above it minus this one, so the column cannot drift
    # from the scores beside it.
    for i, row in enumerate(rows[1:], start=1):
        assert matches(row[2], expected[i - 1] - expected[i]), f"{row[0]}: cost"


def test_the_repeats_claim_matches_what_the_splitter_actually_does(report_exists):
    """Ten repeats buy ten partitions on tapes and far fewer on places.

    The README uses this to say the spread on the context result is not a sampling
    uncertainty. If the splitter ever became less deterministic the claim would silently
    stop being true, and the number beside it is the whole reason the section hedges.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    from src.config import load_config
    from src.data.manifest import load_manifest, manifest_path
    from src.data.splits import with_a_group

    counts = {}
    for label, path in (("tape", "configs/base.yaml"), ("context", "configs/context.yaml")):
        cfg = load_config(path)
        if not manifest_path(cfg).exists():
            pytest.skip("manifest absent")
        frame = with_a_group(load_manifest(cfg), cfg.split.group_column).reset_index(drop=True)
        labels = frame["label"].to_numpy()
        groups = frame[cfg.split.group_column].to_numpy()
        seen = set()
        for repeat in range(10):
            splitter = StratifiedGroupKFold(
                n_splits=cfg.split.n_folds, shuffle=True, random_state=cfg.split.seed + repeat
            )
            seen.add(
                tuple(
                    tuple(sorted(set(groups[test])))
                    for _, test in splitter.split(frame, labels, groups)
                )
            )
        counts[label] = len(seen)

    assert counts["tape"] == 10, "the tape split still varies with every repeat"
    assert counts["context"] < counts["tape"], "the hedge in the README assumes it does not"

    text = " ".join(README.read_text(encoding="utf-8").split())
    assert f"produce {counts['context']} distinct partition against {counts['tape']}" in text, (
        f"the README should say {counts['context']} against {counts['tape']}"
    )


def test_the_place_merge_is_described_as_the_manifest_has_it(report_exists):
    """How many groups the notes really describe, and how much leak the merge removed."""
    from src.config import load_config
    from src.data.manifest import load_manifest, manifest_path

    cfg = load_config("configs/context.yaml")
    if not manifest_path(cfg).exists():
        pytest.skip("manifest absent")
    kept = load_manifest(cfg, kept_only=True)
    grouped = kept[kept["place"] != ""]

    text = " ".join(README.read_text(encoding="utf-8").split())
    assert f"{grouped['site'].nunique()} sites" in text
    assert f"{grouped['context'].nunique()} contexts" in text
    assert f"{grouped['place'].nunique()} places" in text
    assert grouped["place"].nunique() < grouped["context"].nunique(), "the merge did nothing"
