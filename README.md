# datura

datura classifies whale species and call types from Watkins acoustic recordings. It extracts two
representations of the audio, fits gradient boosted trees and a residual CNN on identical folds, and
scores every result against a control that is given no audio at all. It reports each margin with the
interval and the p value the design actually supports.

The headline finding is negative. A model given only what was written down about a recording, and no
sound, beats every audio model on species by 0.241 macro-F1 in fifty splits of fifty. Read the
results as a measurement of the corpus rather than of whales.

## What it does

- Builds a tape grouped cross validation over the Watkins full cuts release, so no recording appears
  on both sides of a fold boundary.
- Extracts hand engineered acoustic descriptors and log mel spectrograms into a memory mapped cache.
- Fits XGBoost and a `MelResNet` variant, plus two controls that see recording metadata and field
  note text and never the audio.
- Reports margins with a corrected resampled paired test, and intervals bootstrapped over whole
  tapes.
- Poses within species call type tasks against a context control, and audits which recording fields
  identify the species on their own.
- Explains the network with band occlusion and Grad-CAM.

## Result

Macro-F1 on held out tapes, 10 kHz band. The trees and both controls ran ten repeats of the five
fold split, so fifty estimates each; the networks ran one split, so five.

| model | macro-F1 | audio |
| --- | --- | --- |
| logbook | 0.997 ± 0.004 | no |
| metadata | 0.868 ± 0.136 | no |
| acoustic descriptors, XGBoost | 0.757 ± 0.067 | yes |
| log mel CNN, 0.15 M | 0.744 ± 0.165 | yes |
| log mel CNN, 2.8 M | 0.692 ± 0.134 | yes |

Against the logbook, which is the highest scoring model that hears nothing:

| comparison | margin | 95% interval | p | agreeing |
| --- | --- | --- | --- | --- |
| XGBoost | -0.241 | -0.311 to -0.171 | 9.5e-09 | 50 of 50 |
| CNN 2.8 M | -0.306 | -0.557 to -0.055 | 0.028 | 5 of 5 |
| CNN 0.15 M | -0.255 | -0.564 to +0.055 | 0.084 | 5 of 5 |
| logbook against metadata | +0.129 | -0.014 to +0.272 | 0.076 | 44 of 50 |

The first row is the only species comparison here that resolves at fifty splits. Measured against the
metadata control instead, the same XGBoost comparison is -0.112 at p = 0.25, which settles nothing.
The floor was in the wrong place until the logbook was built.

Narrow band replication at 5120 Hz, which keeps all 14 humpback tapes instead of 12: XGBoost lands
0.249 below the logbook at p = 2.3e-04 with 50 of 50 agreeing, and 0.152 below the metadata control
at p = 0.010 with 45 of 50. Two bands, a hundred splits, the same ordering.

## Three fields that name the species

Every result above is explained by what the corpus records alongside the audio.

| field | clips it identifies | detail |
| --- | --- | --- |
| collection code | 91.4% | 7 codes, each in exactly one species |
| recording site | 98.2% | 46 of 48 sites visited for one species |
| native sample rate | see below | 10.2k to 45.4k Hz for killer, 10k to 166.6k for sperm |

The collection code is the field that had gone unmeasured. Every Watkins note opens with it, as in
`BE7A  Squeal.  Reverberation present.` It is not a tape identifier, which is what would have made
tape grouped folds sufficient: `BE7A` spans 61 killer whale tapes, `BA2A` 51 sperm whale tapes,
`AC2A` 12 humpback tapes. A held out tape almost always carries a code the training tapes carried
too, so the fold boundary does not hide it.

Splitting held out clips on whether a field is a giveaway:

| model | rate unique | rate shared | code unique | code shared |
| --- | --- | --- | --- | --- |
| logbook | 0.932 | 0.864 | 0.998 | 0.426 |
| metadata | 0.878 | 0.689 | 0.868 | 0.465 |
| acoustic, XGBoost | 0.711 | 0.663 | 0.754 | 0.411 |
| log mel CNN | 0.690 | 0.678 | 0.744 | 0.362 |

On the 359 clips whose code is shared across species the logbook falls from 0.998 to 0.426, level
with the audio models. Its 0.997 is that one field. Removing the sample rate giveaway costs the
metadata control 19 points and the CNN one, so the audio models are worse on average and degrade
less where the recording stops answering the question.

Two checks separate this from a join artefact. `audit_codes_by_species_*.csv` records how many tapes
each code spans, computed without a model. A test builds a corpus where every code sits on exactly
one tape and asserts the logbook's advantage disappears, since tape grouped folds already cover that
case.

## Eleven species

`configs/wide.yaml` differs from `configs/base.yaml` in the species list alone: 11 species carrying
at least 10 surviving tapes, 7,723 clips over 228 recordings against 4,160 over 134. Same band,
window, seed and fold rule.

Two tape counts appear in the wide artifacts and they answer different questions. Summing the
surviving tapes of each species gives 238, which is the right figure for a per class statement. Ten
of those are counted twice, because eight tapes carry two of the eleven species, so the number of
independent recordings is 228. Sample size claims use the second.

| model | 3 species | 11 species |
| --- | --- | --- |
| logbook | 0.997 | 0.920 |
| metadata | 0.868 | 0.577 |
| acoustic, XGBoost | 0.757 | 0.424 |

Every score falls, and the gap widens rather than narrowing. XGBoost lands 0.496 below the logbook
at p = 7.7e-17 with 50 of 50 agreeing, against -0.241 on three species. The metadata control loses
most of its standing: 0.577 against the logbook's 0.920, a gap of 0.343 at p = 4.9e-09, where on
three species that gap was 0.129 and did not resolve.

Which field carries the logbook changes with breadth. On three species the collection code alone
reaches 0.995 and the model splits mostly on sample rate and latitude. On eleven, most gain goes to
`cond_water_noise` and `cond_reverberation`, the noise conditions the recordist wrote down, with the
collection code at 8%. The confound is the written description as a whole rather than any one field.

Two things temper the per class numbers here. Nine of the 228 recordings carry more than one of the
eleven classes, and long finned pilot whale is the worst affected: 7 of its 18 tapes and 690 of its
1,067 clips sit on tapes it shares with sperm whale. Grouping keeps each tape whole so nothing
crosses a fold boundary, but those two classes are not scored on independent evidence. Separately,
eleven classes over 228 recordings leaves one test tape behind some classes in some folds, so a per
fold score for northern right whale or walrus rests on a single recording.

The networks were not run on the wide set. `python -m src.train.cnn --config configs/wide.yaml
--name cnn_small` adds them, and `configs/wide.yaml` declares trees only so a pipeline run does not
train them by surprise.

## Call types

Species is answered by the paperwork, so the audio is asked a question the paperwork cannot answer:
given that this is a sperm whale, does the clip contain a click? Each call type is posed inside one
species as a binary task against a context control seeing site, coordinates, collection code and
noise conditions. Eight tasks clear 60 clips over 10 tapes. Trees on ten repeats, plus one network on
a single split.

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

Eight of nine margins are positive and one resolves. Killer whale whistle is the only comparison in
this repository where hearing the recording beats not hearing it by an amount the design separates.

Sperm whale click resolved at +0.141 and p = 0.044 while the control could not see the collection
code. Giving the control that field moved it to +0.125 and p = 0.084 with no change to the audio
model.

Coda records a labelling failure. Its clips run to a median of 64 seconds while a coda lasts a few,
so most windows inherit a label they do not deserve. Restricting to clips under eight seconds lifted
the margin from +0.012 to +0.105 on one split and reads as +0.026 across fifty. The guard is kept
because the discarded labels are known to be wrong, and it is declared beside the call type in
`configs/call_types.yaml`.

## Explainability

Band occlusion on held out tapes costs humpback recall 0.12 below 330 Hz and nothing above 650 Hz.
Killer whale loses most above 3.7 kHz. Sperm whale loses a little in every band and much in none,
consistent with broadband clicks. Grad-CAM peaks land at 248 and 50 Hz for humpback, and at 1.0 and
4.7 kHz for killer whale, straddling the whistle band rather than sitting on it.

## Data

Watkins Marine Mammal Sound Database, full cuts release: 15,248 mono WAV files across 54 species,
free for personal and academic use. Two sources are involved.

| source | provides | fetched by |
| --- | --- | --- |
| `archive.org/details/watkins-marine-mammal-sound-database-full-cuts` | the audio, 6.7 GB | `src.data.download` |
| `huggingface.co/datasets/ivangtorre/watkins-marine-mammal-full-cuts` | notes, sites, coordinates | `src.data.annotations` |

Everything in this repository that is not a species score rests on the second source. It is read with
HTTP range requests, about a fifth of a megabyte out of each 587 MB shard, and the parsed result is
committed so nothing needs refetching.

The three species under study, before and after the sample rate filter:

| species | clips | tapes | kept clips | kept tapes | hours |
| --- | --- | --- | --- | --- | --- |
| KillerWhale | 2647 | 65 | 2601 | 65 | 1.65 |
| SpermWhale | 1379 | 63 | 1013 | 58 | 12.23 |
| HumpbackWhale | 604 | 14 | 546 | 12 | 1.95 |

Fin whale is excluded: 569 of its 580 clips run between 320 and 640 Hz, putting its Nyquist below
320 Hz and leaving no usable band shared with anything else.

## Method

**Resampling.** Everything is resampled to one rate and anything recorded below it is dropped rather
than upsampled, since upsampling leaves an empty band above the old Nyquist that a classifier will
use as a label. What survives shares an identical 0 to 5 kHz band.

**Folds.** `StratifiedGroupKFold` over tapes, applied across species so a tape carrying two labels is
never split. The tape is the first five characters of the clip id: `5401800A` and `54018001` are both
cuts of tape `54018`. Humpback has 604 clips over 14 tapes, so clip counts overstate the sample size
by roughly an order of magnitude. `tests/test_splits.py` fails if a tape lands on both sides of a
fold.

**Repeats.** The whole split reruns under shifted seeds, giving fifty estimates instead of five. A
model and its control run the identical plan, so repeat three fold two of one pairs with repeat three
fold two of the other. Fold counts are printed beside every number.

**The paired test.** Any two folds of a five fold split train on sets sharing three quarters of their
data, and each repeat reuses the same clips, so the differences are correlated and the ordinary
standard error is far too small. A plain paired t test over the fifty species differences returns
p = 0.00008; the corrected resampled test, replacing `1/n` with `1/n + 1/(k-1)`, returns p = 0.25 on
the same numbers. Every p value here is the corrected one, and `tests/test_uncertainty.py` fails if
repeating a split ever converts an unresolved comparison into a resolved one.

**Intervals.** Bootstrapped over whole tapes rather than clips. Resampling clips counts the same
recording many times and returns an interval several times too narrow; a test builds one interval
each way and asserts the clip version comes out narrower.

**Model selection.** The two network capacities were chosen on validation curves alone, with no test
fold involved. At 2.8 M parameters the CNN peaks at epochs 2, 4, 0, 10 and 4 across the five folds;
at 0.15 M it peaks at 13, 12, 9, 21 and 13.

## Install

```bash
uv sync --extra cpu     # CI, and anything without a GPU
uv sync --extra cuda    # an NVIDIA GPU on driver 560 or newer
```

Python 3.12 to 3.14. torch comes from exactly one of those extras; the wheels differ by gigabytes, so
the choice is explicit and the lockfile records both resolutions.

## Running it

```bash
uv run python -m src.pipeline --config configs/base.yaml
```

Stages run in order and skip whatever is already on disk, so a rerun after a failure resumes. Add
`--force` to redo everything, or `--only report` for one stage.

```text
download -> annotations -> manifest -> features -> trees -> cnn -> cnn_small ->
calltypes_spermwhale -> calltypes_killerwhale -> explain -> report
```

Each stage is also a command:

```bash
uv run python -m src.data.download
uv run python -m src.data.annotations --config configs/base.yaml
uv run python -m src.data.manifest    --config configs/base.yaml
uv run python -m src.features.extract --config configs/base.yaml
uv run python -m src.train.xgb        --config configs/base.yaml --repeats 10
uv run python -m src.train.cnn        --config configs/base.yaml --name cnn_small
uv run python -m src.train.calltypes  --config configs/base.yaml --species SpermWhale --repeats 10
uv run python -m src.evaluate.explain --config configs/base.yaml --name cnn_small --fold 3
uv run python -m src.evaluate.report  --config configs/base.yaml
```

A full run from an empty `data/` takes a little over three hours, most of it the download and the two
network trainings, at roughly 17 minutes per fold on a 4 GB laptop GPU.

## Layout

```text
.github/          CI, and the dependency update schedule
configs/          base.yaml drives the pipeline; base_5k.yaml narrows the band and
                  wide.yaml widens the species set
data/raw/         the archive and the extracted species (gitignored)
data/metadata/    manifest, audit tables and every report
data/processed/   cached feature arrays (gitignored)

src/pipeline.py   every stage in order, skipping whatever is already done
src/config/       sections declare the shape and validate it; loading reads the YAML
src/results.py    where results live on disk; every report path is built here
src/scoring.py    how a prediction becomes a number, shared by training and reporting
src/uncertainty.py  intervals over tapes, and the paired test a margin needs
src/provenance.py what produced a result: commit, versions, accelerator, config digests

src/audio/        decode, resample, window
src/data/         clips parses identity, notes reads a field note, annotations fetches
                  and parses them, manifest lists the audio, audit describes it,
                  splits holds the fold grouping rule
src/features/     views hands out rows without copying them, source reads the cache,
                  controls are the models given no audio, plus the extractor interface,
                  its implementations, the cache and the registry
src/models/       the classifier interface, the trees, the cnn package, the registry
src/train/        folds and the repeat plan, one cross validation runner, one session,
                  tasks decides which call type questions are worth asking and
                  calltypes answers one
src/evaluate/     families groups results, tables builds them, sections composes the
                  document, figures draws it, artifacts writes it, report runs the lot,
                  plus occlusion and Grad-CAM
experiments/      notebooks that read artifacts and plot, no training code
```

Two registries carry the extension points. `src/features/registry.py` maps a representation name to
its extractor and its cache; `src/models/registry.py` declares each model's features, hyperparameter
file and trainer. Adding a representation is a new extractor plus one line; adding a model is one
entry. Neither touches the runner, the metrics or the split.

Models report on themselves through `artifacts`: the trees return which features carried the fit, the
network returns its learning curve and writes its weights. That is why the runner has no branch for
either.

## Reproducing

Full numbers live in `data/metadata/report/{base_10k,base_5k,wide_10k}/REPORT.md`. Every comparison
in a configuration is listed at the top of its report, sorted by how well it resolves.

The manifests, predictions and parsed notes are committed, so every report rebuilds without the 6.7 GB
archive:

```bash
uv run python -m src.evaluate.report --config configs/base.yaml
```

`tests/test_readme_numbers.py` reads the tables out of this file and checks every figure against the
CSV that produced it. CI diffs the regenerated reports against the committed ones, and this document
was not covered by that until the test existed.

CI runs three jobs: lint, the suite on 3.12, 3.13 and 3.14 on Linux plus 3.14 on Windows, and a
rebuild of all three reports. Every job installs with `uv sync --locked` against a pinned uv, because
a uv older than the one that wrote `uv.lock` discards the lock and resolves from scratch rather than
reporting that it cannot read it.

## Limits

- Every number describes one recording source. Results are about the Watkins corpus, and the
  strongest effect measured is a property of its paperwork.
- 134 independent recordings at 10 kHz, 228 across eleven species. Eleven classes over 238 species
  tapes leaves one test tape behind some classes in some folds, so those per fold scores rest on a
  single recording.
- The networks ran one split each. Their comparisons carry five estimates against the trees' fifty.
- No call detection or segmentation: a Watkins note is written against a whole cut, so there are no
  onsets. No individual identification: nothing in the corpus names an animal. Per window predictions
  are written with the position of each window inside its clip, which is the only time coordinate
  available here.
- MobySound and DCLDE are excluded. Mixing recording sources stacks a second equipment confound on
  the ones documented above.

## License

Apache 2.0. See [LICENSE](LICENSE).
