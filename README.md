# datura

Can a model identify whale species from acoustic recordings? Yes. But the first number you get back
is mostly wrong, and figuring out why is what this repo actually contains.

Three species, chosen because they sound nothing alike: humpback (structured tonal song), sperm
whale (broadband clicks), killer whale (whistles and pulsed calls). Two ways of representing the
audio, hand engineered acoustic descriptors into gradient boosted trees and log mel spectrograms
into a small residual CNN, both run through identical folds. Plus a third model that gets no audio
at all.

## The catch with this dataset

Watkins was recorded over five decades on whatever gear the work called for, and the gear tracks
the species:

| Species | Clips | Tapes | Native sample rates |
| --- | --- | --- | --- |
| KillerWhale | 2647 | 65 | 10.2k to 45.4k Hz |
| SpermWhale | 1379 | 63 | 10k to 166.6k Hz |
| HumpbackWhale | 604 | 14 | 5.1k to 30k Hz |
| Fin_FinbackWhale | 580 | 41 | 320 to 640 Hz on 39 of 41 tapes |

You can separate most of that on recording bandwidth without listening to anything. Fin whale is the
worst case, and it is why fin whale is not in the species set. 569 of its 580 clips run between 320
and 640 Hz, which puts the Nyquist under 320 Hz and leaves it sharing no usable band with anything
else. Eleven clips on three tapes were recorded above 10 kHz and everything else was not. Score
against it and you are measuring a tape machine.

Two consequences, and they drive most of the design.

Everything gets resampled to one rate, and anything recorded below that rate gets dropped.
Upsample it and you leave a silent band above the old Nyquist, which a classifier will happily use
as a label. What survives all has an identical 0 to 5 kHz band.

Every audio result is quoted as a margin over the control. The control sees native sample rate,
recording year, clip duration and file size. No audio. Same folds, same evaluation code. Whatever it
scores is the floor.

## Why folds are grouped by tape

Clip counts flatter this dataset by about an order of magnitude. Humpback has 604 clips, but they
come off 14 original tapes, and cuts from the same tape are near duplicates of each other. Split
clips at random and you put copies of one recording on both sides of the test boundary.

The tape is sitting in the filename. `5401800A` and `54018001` are both cuts of tape `54018`, and
the two id formats resolve the same way. Folds are `StratifiedGroupKFold` over tapes, applied across
species so a tape carrying two labels never gets torn in half. Nothing is reported as a single held
out score, because with twelve humpback tapes that is close to a two sample estimate.

`tests/test_splits.py` fails loudly if a tape lands on both sides of a fold. If you touch one thing
in this repo, do not let it be that.

## How much any of this settles

Five folds over 134 recordings is a very small design, and for a while this repo reported margins
from it as though they were findings. They were not. (134 unique tapes; the per species counts sum
to 135 because one tape carries both humpback and sperm whale.)

The trees are cheap, so the whole split is now rerun under shifted seeds. Ten repeats give fifty
estimates of the same quantity instead of five, and a model and its control run the identical plan
so repeat three fold two of one pairs with repeat three fold two of the other. The two networks stay
on their single split, since retraining them ten times is most of a day of GPU for a question the
trees answer more cheaply, and the fold count is printed beside every number so a five fold
comparison is never read as a fifty fold one.

Fifty numbers is not fifty pieces of evidence, though, and this is the part worth reading twice. Any
two folds of a five fold split are fitted on training sets that share three quarters of their data,
and each repeat runs over the same clips again. The differences are correlated, so the usual
standard error of the mean is far too small. Running a plain paired t test over the fifty species
differences returns p = 0.00008. The corrected resampled test, which replaces `1/n` with
`1/n + 1/(k-1)` to account for that overlap, returns p = 0.25 on the same numbers. One of those is a
headline and the other is nothing of the kind, and the second one is right. Every p value in this
repo is the corrected one, and `tests/test_uncertainty.py` fails if repeating a split ever turns an
unresolved comparison into a resolved one.

Intervals come from resampling whole tapes with replacement, not clips. Cuts from one tape are near
duplicates, so resampling clips counts the same recording many times and returns an interval several
times too narrow. That is asserted directly: build one interval each way on the same predictions and
the clip version has to come out narrower, which is what stops the distinction being lost quietly
later.

Every margin in the report carries three columns beside it. The interval, the p value, and how many
folds pointed the same way. That last one earns its space, because a direction holding in four folds
of five is worth knowing even where the p value settles nothing.

## What came out of it

Macro-F1 on held out tapes, 10 kHz band. The trees and both no-audio models ran ten repeats of the
split, so fifty estimates; the networks ran one split, so five:

| model | macro-F1 | hears the recording |
| --- | --- | --- |
| logbook, no audio | 0.997 ± 0.004 | no |
| metadata only, no audio | 0.868 ± 0.136 | no |
| acoustic features, XGBoost | 0.757 ± 0.067 | yes |
| log mel CNN, 0.15 M params | 0.744 ± 0.165 | yes |
| log mel CNN, 2.8 M params | 0.692 ± 0.134 | yes |

The top row is the whole story and it took a year of the dataset to find. Every Watkins field note
opens with the code of the collection the cut came from, like `BE7A  Squeal.  Reverberation
present.` Seven of those codes cover the three species here and each one appears in exactly one of
them. They are not tape identifiers, which is what would have made them harmless: `BE7A` spans 61
killer whale tapes, `BA2A` 51 sperm whale tapes, `AC2A` 12 humpback tapes. A code names a
collection, so grouping folds by tape does nothing to hide it, and a held out tape almost always
carries a code the training tapes carried too.

Give a model that code, the site, the coordinates and the recording metadata, and no audio at all,
and it reaches 0.997. Against that floor:

| comparison | margin | 95% interval | p | folds agreeing |
| --- | --- | --- | --- | --- |
| XGBoost against the logbook | -0.241 | -0.311 to -0.171 | 9e-09 | 50 of 50 |
| CNN 2.8 M against the logbook | -0.306 | -0.557 to -0.055 | 0.028 | 5 of 5 |
| CNN 0.15 M against the logbook | -0.255 | -0.564 to +0.055 | 0.084 | 5 of 5 |
| logbook against the metadata control | +0.129 | -0.014 to +0.272 | 0.076 | 44 of 50 |

That first row is the only thing in this repo that is settled rather than merely pointed at. The
best audio model loses to a model that never hears the recording, by a quarter of a point, in every
one of fifty splits. Before the collection code was measured, the same comparison against the
metadata control was -0.112 at p = 0.25, which settles nothing. The floor was in the wrong place.

Two things stop this being an artefact of how the code was joined. `audit_codes_by_species_*.csv`
records how many tapes each code spans, model-free, which is what separates a confound from a tape
identifier. And a test builds a collection where every code sits on exactly one tape and asserts the
advantage disappears, because tape grouped folds already handle that case. It does.

Split the test clips by whether a field is a giveaway or not, and the same story shows up twice:

| model | rate gives it away | rate is shared | code gives it away | code is shared |
| --- | --- | --- | --- | --- |
| logbook | 0.932 | 0.864 | 0.998 | 0.426 |
| metadata | 0.878 | 0.689 | 0.868 | 0.465 |
| acoustic, XGBoost | 0.711 | 0.663 | 0.754 | 0.411 |
| log mel CNN | 0.690 | 0.678 | 0.744 | 0.362 |

On the 359 clips whose code is not unique to a species, the logbook falls from 0.998 to 0.426, level
with the audio models. Its 0.997 is the code and nothing else. Take the sample rate giveaway away
and the metadata control loses 19 points while the CNN loses one, so the audio models are worse on
average and hold up better where the recording stops answering the question. That is a more useful
outcome than either number alone.

The spectrogram barely beats hand engineered features, and only after the network got smaller. At
2.8 M parameters the CNN peaked at epochs 2, 4, 0, 10 and 4 across the five folds and got worse from
there, so four of five folds were done inside five epochs. It was memorising tapes, and more epochs
would not have helped. At 0.15 M it peaks at 13, 12, 9, 21 and 13, and scores better. I chose between
the two on validation curves alone, with no test fold involved. Even the better one sits inside the
noise of the tree baseline, and with 134 independent recordings behind the whole thing, that is the
honest answer to how much a learned representation buys you here.

The CNN does look at sensible things, at least. Masking frequency bands on held out tapes costs
humpback recall 0.12 below 330 Hz and nothing at all above 650 Hz. Killer whale loses most in the
top band above 3.7 kHz. Sperm whale loses a little everywhere and a lot nowhere, which is what you
would expect if the model is keying on broadband clicks. Grad-CAM lands in similar places on its
own: its humpback peaks sit at 248 and 50 Hz, and its killer whale peaks at 1.0 and 4.7 kHz, which
straddles the whistle band rather than sitting in it.

None of this hinges on the band, and the narrow band puts it beyond doubt. Rerun at 5120 Hz, which
keeps all 14 humpback tapes instead of 12, on the same ten repeats and the same corrected test: the
trees land 0.152 below the control at p = 0.010, with the direction holding in 45 splits of 50. Same
ordering as the wide band, a larger margin, and this time it resolves. The smaller network runs
there too, at -0.173 with p = 0.203 on its single split, all five folds agreeing.

That is the strongest single result in the repo, and it is a result about recording metadata rather
than about whales. Two bands, a hundred splits between them, and the model that never hears the
animal wins both.

## Eleven species instead of three

Three species over 134 recordings is a small design, and the archive on disk holds 54 species over
393 tapes. Eleven of them carry at least ten tapes that survive the 10 kHz filter, which is 7,723
clips over 238 recordings. Same band, same window, same seed, same folds: `configs/wide.yaml`
differs from `configs/base.yaml` in the species list and nothing else, so any gap between the two
runs is breadth.

I wrote down what I expected before running it, because the interesting number here is not any one
score. Everything should fall, since eleven classes is harder than three. The question was whether
the gap between hearing the recording and reading the paperwork narrows once the collection is not
almost a species label.

| model | three species | eleven species |
| --- | --- | --- |
| logbook, no audio | 0.997 | 0.920 |
| metadata only, no audio | 0.868 | 0.577 |
| acoustic features, XGBoost | 0.757 | 0.424 |

Everything fell, and the gap widened rather than narrowed. XGBoost lands 0.496 below the logbook,
p below 0.0001, in 50 splits of 50. On three species the same comparison was -0.241. Doubling the
recordings did not rescue the audio; it made the paperwork look better, because there is more
paperwork to go on when there are more species to tell apart.

What the logbook leans on changes with breadth, and that is the part worth reading. On three species
the collection code alone reached 0.995 and the model split mostly on sample rate and latitude. On
eleven, the largest share of gain goes to `cond_water_noise` and `cond_reverberation`, the noise
conditions the recordist wrote down, with the collection code at 8%. The confound is not one field.
It is that a written description of the circumstances of a recording identifies the animal, and
which part of the description does the work depends on which animals you are separating.

The metadata control also loses most of its advantage here: 0.577 against the logbook's 0.920, a gap
of 0.343 at p below 0.0001. On three species that gap was 0.129 and did not resolve. Equipment
metadata alone was a much weaker floor than it looked.

Read one caveat beside these numbers. Eleven classes over 238 recordings puts one test tape behind
some classes in some folds, so a per fold score for northern right whale or walrus rests on a single
recording. That is not new, humpback already had it at twelve tapes, and macro-F1 stays defined; but
breadth here bought more classes as well as more recordings, and the two pull against each other.

The networks were not run on the wide set. At eleven classes the trees answer the question at a
fiftieth of the GPU cost, and the question was about the floor rather than about the representation.
`python -m src.train.cnn --config configs/wide.yaml --name cnn_small` adds them.

## Where audio does win

Species is the wrong question to ask of this dataset, because the paperwork answers it. So ask a
question the paperwork cannot: given that this is a sperm whale, does the clip contain a click?

Every call type is posed inside one species, and every one gets its own binary model against a
context control seeing the site, the coordinates, the collection code and the noise conditions, and
no audio. Site alone identifies the species for 98% of clips here and the collection code does
better still, so that control is the same kind of floor the logbook is. Eight tasks clear the minimum
of 60 clips over 10 independent tapes. Trees on ten repeats of the split, macro-F1, plus one network
on a single split:

| task | audio | control | margin | p | agreeing |
| --- | --- | --- | --- | --- | --- |
| sperm whale, whistle | 0.712 | 0.526 | +0.185 | 0.20 | 29 of 50 |
| killer whale, click | 0.601 | 0.456 | +0.144 | 0.079 | 40 of 50 |
| sperm whale, coda, CNN | 0.566 | 0.434 | +0.132 | 0.35 | 4 of 5 |
| sperm whale, click | 0.705 | 0.580 | +0.125 | 0.084 | 43 of 50 |
| killer whale, squeal | 0.768 | 0.652 | +0.116 | 0.24 | 36 of 50 |
| killer whale, whistle | 0.564 | 0.463 | +0.101 | 0.022 | 41 of 50 |
| sperm whale, coda | 0.570 | 0.545 | +0.026 | 0.80 | 23 of 50 |
| killer whale, chirp | 0.557 | 0.547 | +0.010 | 0.83 | 21 of 50 |
| killer whale, call | 0.689 | 0.712 | -0.023 | 0.88 | 29 of 50 |

Eight of nine are positive and one resolves. Killer whale whistle is the only place in this repo
where listening to the recording beats not listening to it by an amount the design can separate.

It used to be two. Sperm whale click sat at +0.141 and p = 0.044 while the control could not see the
collection code, and giving it that field moved the comparison to +0.125 and p = 0.084. Nothing
about the audio model changed. A finding that survives only while the control is missing a column is
not a finding, and the honest version of this section has one row in it rather than two.

Sperm whale whistle has the largest margin here and does not resolve: +0.185 with a fold spread of
0.221, and the direction holding in only 29 splits of 50. It rests on 11 tapes, and a task carried by
11 recordings has an effective sample size of 11 however many clips it spans. That is the shape of
almost everything in this dataset.

Coda is the instructive failure. Its clips run to a median of 64 seconds while a coda lasts a few, so
most windows of a coda labelled recording inherit a label they do not deserve. Dropping clips over
eight seconds lifted the margin from +0.012 to +0.105 on one split, which looked like a finding.
Across fifty it is +0.026 with the direction holding in 23, which is a coin flip. The guard stayed
anyway, because training on labels known to be false is worse than a flat result, and it now lives
beside the call type in `configs/call_types.yaml` rather than as a command line flag.

Full numbers live in `data/metadata/report/base_10k/REPORT.md` and
`data/metadata/report/base_5k/REPORT.md`. Every comparison in a configuration is listed together at
the top of its report, sorted by how well it resolves, so the ones that settle nothing are as easy
to find as the ones that do.

## Layout

```text
.github/          CI, and the dependency update schedule
configs/          base.yaml drives the pipeline; base_5k.yaml narrows the band and
                  wide.yaml widens the species set
data/raw/         the archive and the extracted species (gitignored)
data/metadata/    manifest, audit tables and every report
data/processed/   cached feature arrays (gitignored)

src/pipeline.py   every stage in order, skipping whatever is already done
src/cli.py        the options every command shares
src/config/       sections declare the shape and validate it; loading reads the YAML
src/results.py    where results live on disk; every report path is built here
src/scoring.py    how a prediction becomes a number, shared by training and reporting
src/uncertainty.py  intervals over tapes, and the paired test a margin needs
src/provenance.py what produced a result: commit, versions, accelerator, config digests
src/errors.py     the base every deliberate failure inherits from
src/logging_config.py  configured by entry points only, never by a library module

src/audio/        decode, resample, window
src/data/         clips parses identity, notes reads a field note, annotations fetches
                  and parses them, manifest lists the audio, audit describes it,
                  splits holds the fold grouping rule
src/features/     the extractor interface, its implementations, the cache and the registry
src/models/       the classifier interface, the trees, the cnn package, the registry
src/train/        folds and the repeat plan, one cross validation runner, one session,
                  and the within species call type tasks
src/evaluate/     families, tables, figures, occlusion, Grad-CAM, report
experiments/      notebooks that read artifacts and plot, no training code
```

Two registries carry the extension points. `src/features/registry.py` maps a representation name to
its extractor and its cache; `src/models/registry.py` declares each model's features, hyperparameter
file and trainer. Adding a representation is a new extractor plus one line; adding a model is one
entry. Neither touches the runner, the metrics or the split, which is the only reason four models
produce numbers you can put in the same table.

Models report on themselves through `artifacts`: the trees return which features carried the fit,
the network returns its learning curve and writes its weights. That is why the runner has no branch
in it for either.

## Running it

torch comes from one of two extras because the wheels differ by gigabytes. Pick the one that
matches the machine:

```bash
uv sync --extra cuda    # NVIDIA GPU, driver 560 or newer
uv sync --extra cpu     # everything else, and CI
uv run pytest
```

Then the whole thing in one command:

```bash
uv run python -m src.pipeline --config configs/base.yaml
uv run python -m src.pipeline --config configs/base_5k.yaml   # narrow band check
```

The stage list comes from the model registry, so adding a model adds a stage:

```text
download -> annotations -> manifest -> features -> trees -> cnn -> cnn_small ->
calltypes_spermwhale -> calltypes_killerwhale -> explain -> report
```

Stages whose output already exists are skipped, so a rerun after a failure picks up where it
stopped. Add `--force` to redo everything, or `--only report` to run one stage.

The stages are also individual commands if you want them separately:

```bash
uv run python -m src.data.download                                  # 6.7 GB archive, resumable
uv run python -m src.data.annotations --config configs/base.yaml    # field notes, second source
uv run python -m src.data.manifest    --config configs/base.yaml
uv run python -m src.features.extract --config configs/base.yaml
uv run python -m src.train.xgb        --config configs/base.yaml --repeats 10  # and control
uv run python -m src.train.cnn        --config configs/base.yaml --name cnn
uv run python -m src.train.cnn        --config configs/base.yaml --name cnn_small
uv run python -m src.train.calltypes  --config configs/base.yaml --species SpermWhale --repeats 10
uv run python -m src.train.xgb        --config configs/wide.yaml --repeats 10   # eleven species
uv run python -m src.evaluate.explain --config configs/base.yaml --name cnn_small --fold 3
uv run python -m src.evaluate.report  --config configs/base.yaml
```

`--repeats N` reruns the whole split under `seed + repeat` and tags every row with which repeat
produced it. Repeat zero is the original split, so a repeated run contains the single split rather
than replacing it, and a result from before repeats existed still pairs against repeat zero of one
that has ten. It defaults to 1.

`--name` selects a model from the registry in `src/models/registry.py`, which is where each one
declares its features, its hyperparameter file and the command that trains it. Both network sizes
get trained; the reported one is picked on validation macro-F1 while the test folds stay untouched.

Every command takes `-v` for debug detail and `-q` for warnings only. Set `DATURA_LOG_FORMAT=full`
for timestamps and module names, which is what you want when a run is being archived.

Results land in `data/metadata/report/<config name>/` as CSVs, figures and a `REPORT.md`. The
notebooks only read those artifacts, so every figure can be regenerated from the command line
without opening Jupyter.

On hardware: the CNN trains in about 17 minutes per fold on a 4 GB laptop GPU. CPU works, it just
takes considerably longer. A full run from an empty `data/` is a little over three hours, most of
it the download and the four CNN trainings.

## Reproducing a result

The archive, the environment and the code are all pinned, and every run writes down what produced
it.

`uv.lock` is committed and CI installs with `--frozen`, so the exact package versions are fixed. The
interpreter is pinned in `.python-version`, and the test matrix covers 3.12 through 3.14 on Linux
plus 3.14 on Windows.

`dataset.archive_sha256` in the config pins the archive contents. The download verifies the digest
rather than trusting the byte count, since a truncated resume or a mirror serving different content
can match on length alone.

Every result directory gets a `provenance.json` with the commit, whether the tree was dirty, the
package versions, the accelerator, and the config digests. Cache filenames carry those digests too,
so changing the sample rate or the mel settings produces a new key and a stale cache can never be
reused by accident.

The CNN runs with cuDNN autotuning on by default, which is faster but picks whichever kernel is
quickest on the day. Set `deterministic: true` in the CNN config, and export
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, for identical reruns at about a third less throughput.

Manifests and per clip predictions are committed, so both reports rebuild from a fresh clone with no
download at all:

```bash
uv run python -m src.evaluate.report --config configs/base.yaml
```

`tests/test_readme_numbers.py` reads the tables out of this file and checks every figure against the
CSV that produced it. The report was already diffed by CI and this document was not, which is how
five numbers in it came to disagree with the artifacts they were copied from.

CI does exactly that on every push and fails if a regenerated table differs from the committed one.
Figures are excluded from that check: matplotlib output is not stable byte for byte across
versions.

## CI

`.github/workflows/ci.yml` runs three jobs on push and pull request:

- **lint**: `uv lock --check`, then `ruff check` and `ruff format --check` across source, tests
  and notebooks
- **test**: the suite on 3.12, 3.13 and 3.14 on Linux, plus 3.14 on Windows, with coverage
- **reproduce**: rebuilds all three reports from committed results and diffs them against what is in
  the repo

The suite finishes in well under a minute because nothing in it needs the archive.
`tests/test_pipeline_e2e.py` generates its own audio, three species with distinct acoustic regimes,
and pushes it through manifest, features, folds, training, explainability and report. That is the
test that catches the integration breaks the unit tests cannot see.
`tests/test_notebooks.py` executes all three notebooks against the committed artifacts.

`.pre-commit-config.yaml` runs the same lint and format checks locally if you want them:

```bash
uv run pre-commit install
```

## Reading the output

Roughly in order of what would invalidate a result soonest. If you are reviewing this, go in this
order:

1. The manifest should report 604 humpback, 1379 sperm and 2647 killer whale clips before filtering.
   If the humpback tape count after filtering is still 14, the sample rate filter did not run.
2. The fold summary should show zero tape overlap between train, validation and test in all five
   folds.
3. The logbook's macro-F1 gets printed before the audio results. It is the floor, the metadata
   control is a weaker one, and everything else should be read relative to the higher of the two.
   Read both ambiguity breakdowns before any headline.
4. Read the p value and the fold count before the margin. Most comparisons here do not resolve, and
   a margin from five folds over a dozen recordings is a direction rather than a result.
5. Audio scores carry a spread. A single number over a dozen humpback tapes is not a result.
6. Check humpback recall on its own. It is the class the fold structure stresses hardest.
7. Compare the 5120 Hz run against the 10 kHz run. If they diverge badly, the conclusion depends on
   the band choice and has to be written up that way.
8. Read occlusion and Grad-CAM against the call bands each species is known to use.
9. Compare the eleven species run against the three species one. If the gap between audio and
   paperwork narrows with more recordings, that is the first sign audio carries something the
   paperwork does not. It widened.

## Data

Watkins Marine Mammal Sound Database, from Woods Hole Oceanographic Institution and the New Bedford
Whaling Museum. The full cuts release is 15,248 mono WAV files across 54 species, free for personal
and academic use. WHOI's own site was down for maintenance when I built this, so acquisition goes
through the Internet Archive mirror at
`archive.org/details/watkins-marine-mammal-sound-database-full-cuts`.

Downloads shell out to curl. Python's stdlib SSL rejected the certificate chain on my machine and
curl ships its own CA bundle, so that was the shorter path.

The audio and the text come from two different places, which is worth knowing before trusting any of
it. The archive above carries the WAV files and nothing else. Every field note, site name and
coordinate comes from a HuggingFace mirror of the same database,
`huggingface.co/datasets/ivangtorre/watkins-marine-mammal-full-cuts`, read by
`python -m src.data.annotations`. It fetches four text columns out of parquet shards using HTTP range
requests, about a fifth of a megabyte out of each 587 MB shard, and writes
`data/metadata/watkins_annotations.parquet` once.

Everything in this repo that is not a species score rests on that second source: the call type
labels, the site giveaway, and the collection code that turned out to sit above every audio model.
The parquet is committed, so nothing needs refetching, and it is reparsed in place whenever
`configs/call_types.yaml` changes rather than trusted to still match.

## What this is not

Species and call type classification from one recording source, and that is all.

No call detection and no segmentation, because those need onsets and offsets and a Watkins note is
written against a whole cut. No individual identification, because nothing in the collection names
an animal. No sequence modelling and no claims about whale language. The pipeline now writes per
window predictions carrying the position of each window inside its clip, which is the one time
coordinate here and where any of that work would have to start, but the labels for it do not exist
in this dataset.

MobySound and DCLDE are deliberately left out. Mixing recording sources stacks a second equipment
confound on top of the ones documented above, and this repo has just finished demonstrating that it
missed one of those for months.
