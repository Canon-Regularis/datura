# datura

datura names the whale species in a Watkins recording, and it is allowed to refuse. Four
independent representations of the audio each carry real species information: hand engineered
spectral descriptors, log mel spectrograms through two networks, and a frozen speech encoder all
land between 0.710 and 0.752 macro-F1 on held out recordings, where guessing from the class shares
reaches 0.333. Averaging the two strongest reaches 0.765 macro-F1 at 84.1% accuracy, and letting
that pair decline the third of clips it is least sure of takes accuracy to 94.7%.

That holds while the fold boundary is a recording. Move it to a recording location and the same
model falls to 0.321, which is what guessing reaches, and a control with the same fold geometry and
no geography scores 0.569, so the fall is the place rather than the arithmetic. What the audio
models learned is a recording context, and it does not travel.

Every audio model is also scored against a control that hears nothing and reads only what was
written down about the recording. That control wins by 0.245 macro-F1 and more, with every fold
agreeing, and it barely moves when the place changes. Both results are reported in full below,
because together they describe how the Watkins corpus was assembled rather than how whales sound.

## What it does

- Identifies a species in one file from the command line, with a confidence band read from held
  out data and an abstention threshold derived per model.
- Builds cross validation grouped by tape, so no recording appears on both sides of a fold
  boundary, and again grouped by recording context, so no place does either.
- Extracts hand engineered acoustic descriptors, log mel spectrograms and frozen wav2vec2
  embeddings into a memory mapped cache.
- Fits XGBoost, two `MelResNet` variants and a linear probe over the pretrained embeddings, plus
  two controls that see recording metadata and field note text and never the audio.
- Reports margins with a corrected resampled paired test, intervals bootstrapped over whole tapes,
  and a false discovery rate across every comparison in every configuration at once.
- Poses within species call type tasks against a control that sees everything written down, and
  audits which recording fields identify the species on their own.
- Explains the network with band occlusion and Grad-CAM.

## Result

Macro-F1 on held out tapes, 10 kHz band. Everything except the larger network ran ten repeats of
the five fold split, so fifty estimates each; `log mel CNN, 2.8 M` ran one split, so five.

| model | macro-F1 | audio |
| --- | --- | --- |
| logbook | 0.997 ± 0.010 | no |
| metadata | 0.868 ± 0.134 | no |
| XGBoost and probe averaged | 0.765 ± 0.089 | yes |
| acoustic descriptors, XGBoost | 0.752 ± 0.070 | yes |
| wav2vec2 probe | 0.739 ± 0.090 | yes |
| log mel CNN, 2.8 M | 0.713 ± 0.123 | yes |
| log mel CNN, 0.15 M | 0.710 ± 0.142 | yes |

Chance is 0.333 for a guess drawn from the class shares, 0.301 for a uniform guess, and 0.256 for
always answering killer whale. The four audio models share nothing but their input: spectral
descriptors computed by hand, a network trained from scratch on log mel windows, the same network
at twenty times the capacity, and a transformer pretrained on 960 hours of English speech and never
fine tuned here. They land within 0.043 of one another, which is the result that says the signal
is in the waveform rather than in any one way of reading it.

## Using it

```bash
uv run python -m src.predict recording.wav
uv run python -m src.predict recording.wav --model xgboost,probe   # the best measured pair
```

```text
Species prediction
--------------------------------------------
  KillerWhale             77.7%
  SpermWhale              13.7%
  HumpbackWhale            8.6%

  Prediction : KillerWhale
  Confidence : HIGH, and on held out recordings this model was 92.4% accurate
               on the most confident 40% of its predictions

  model: xgboost, fold 0, trained on base_10k
```

One fold of XGBoost is committed, so a fresh clone predicts with no training and no downloads. The
command walks the same path a training clip walks, through the same functions in the same order,
and a test round trips a held out clip and asserts the probabilities match what cross validation
recorded for it. Without that check the command would be answering a question no number in this
repository describes.

It refuses two kinds of file. A recording below 10 kHz is turned away rather than upsampled, since
upsampling leaves an empty band above the old Nyquist that a classifier reads as a species. A clip
under half a second is turned away rather than padded, since reflecting a fraction of a second out
to a two second window produces an answer about the padding.

## What it is worth when it can decline

Every other score here forces an answer for every clip. That is the right way to compare two
representations and a poor way to describe a tool, because a classifier that says nothing on the
hard third of its input is more useful than one that guesses. Ranking held out predictions by the
probability of the class chosen and keeping the most confident share gives an operating curve.
Nothing is refitted: this reads the committed per clip probabilities and asks a different question
of them.

| model | every clip | most confident 70% | most confident 30% |
| --- | --- | --- | --- |
| acoustic descriptors, XGBoost | 0.830 | 0.903 | 0.930 |
| wav2vec2 probe | 0.818 | 0.923 | 0.975 |
| log mel CNN, 0.15 M | 0.803 | 0.914 | 0.989 |
| XGBoost and probe averaged | 0.842 | 0.947 | 0.970 |

Accuracy pooled over all fifty splits. Averaging the trees and the probe beats both of them at
every level, and adding either network makes it worse, so the pair is named rather than the set.
The average is written out as an ordinary result directory rather than reported as a note, so it
carries a summary, a confusion matrix and a margin against the controls like every other model, and
it cannot dodge the multiplicity correction.

Equal weights hold up only while the members are close. Under the place folds below the probe
reaches 0.444 and the trees 0.321, and averaging them gives 0.440, which is the weaker member
dragging the stronger one down. The pair is the best combination measured on held out recordings
rather than a rule that survives a change of question.

A threshold filters uncertainty rather than error. It removes the clips the model was unsure about
and leaves the confident mistakes untouched, so a model that is wrong while sure stays wrong while
sure however the cut off moves. On the same 41,600 predictions, `cnn_small` is wrong while more
than 90% confident on 8.62% of them and XGBoost on 0.101%, which is eighty five times fewer.
XGBoost is what the command ships for that reason rather than for its score.

The cut off is read from each model's own curve rather than fixed, because the same probability
means different things to different models. To reach 90% accuracy XGBoost needs 0.591 and
`cnn_small` needs 0.954. A single 0.6 threshold declines a third of XGBoost's answers and a
twentieth of the network's.

## Does it survive a new recording context

No. This is the result that matters most and it is negative.

A tape grouped fold proves no recording sits on both sides of the boundary. It does not prove the
model survives a place, a hydrophone or a recording chain it has never heard.
`configs/context.yaml` differs from `configs/base.yaml` in one line, the column folds are grouped
on, and XGBoost falls from 0.752 to 0.321 macro-F1 across that one line. A guess drawn from the
class shares reaches 0.333, so a model trained on other places is worth nothing at a place it has
never heard.

Some of that fall is the fold geometry rather than the place. Grouping by place leaves 24 groups
where the tape rule leaves 134, so five folds hold out 7.8 groups instead of 27 and one group can
own most of a test fold. `configs/context_shuffled.yaml` is the control that separates the two:
pseudo places dealt at random inside each species, matched to the real ones group for group,
species for species and tape for tape, so the split loses the same data and keeps none of the
geography.

| what is being scored | macro-F1 | cost |
| --- | --- | --- |
| tape held out, all 4,160 clips | 0.752 | |
| the same predictions on the 4,088 clips a place split can score | 0.744 | 0.008 |
| pseudo places, same structure and no real place | 0.556 | 0.189 |
| real places | 0.321 | 0.234 |

The fold geometry costs 0.189 and the identity of the place costs 0.234. The 72 clips carrying no
site, which base scores and a place split cannot, account for the remaining 0.008. So the model is
not failing to learn. It is learning the recording.

| model | tape held out | place held out | change |
| --- | --- | --- | --- |
| logbook | 0.997 | 0.982 | -0.016 |
| metadata | 0.868 | 0.621 | -0.247 |
| XGBoost and probe averaged | 0.765 | 0.440 | -0.324 |
| wav2vec2 probe | 0.739 | 0.444 | -0.295 |
| acoustic descriptors, XGBoost | 0.752 | 0.321 | -0.431 |

The logbook barely moves, because the collection code names the species wherever the recording was
made. That is the cleanest statement of the confound this project measures: the paperwork travels
and the audio does not. Measured against it under these folds, XGBoost is 0.660 behind at
p = 1.1e-04 and the probe 0.537 behind at p = 1.5e-03, both on five folds and both resolving.

That row is only trustworthy because of how the control encodes a name, and getting it wrong was
worth more than the effect being measured. A site coded to its position in the alphabet is an
ordinal, so a split can only ask whether the code is above a threshold, and the answer for a name
the model never saw is decided by where the alphabet put it. Under tape folds that is invisible,
because a held out tape almost always carries a name the training tapes carried too. Under place
folds every held out name is new. Five encodings carrying identical information scored 0.7211,
0.9908, 0.9911, 0.9990 and 0.9993 on the same folds, and moving the absent name sentinel from one
end of the axis to the other was worth 0.278 by itself. One column per name removes the axis: an
unseen name is zero everywhere and the tree falls back on what it knows. The same comparison under
that encoding spans 0.018.

The group is the physical place rather than the raw site, and both weaker versions of it leaked.
Grouping on the site alone put whole tapes on both sides of a fold, because six tapes carry clips
the notes place at more than one site. Merging the sites a tape links fixed that and left a second
leak: names no tape connects stayed apart, so six groups were all Dominica and 52% of held out clips
sat in a fold whose place was also in the training set. 47 sites become 39 contexts become 24
places, and no place, site or tape now crosses a fold boundary.

Three things bound how hard this can be pushed.

The corpus is close to its limit here. 24 places over three species means humpback has exactly five,
which is the fewest a five fold split can use, and `oregon` alone is 49 of the 65 killer whale tapes.
Every fold gives up between 18% and 55% of some class's clips. That cost is what the pseudo place
control measures and subtracts, and it also means a single fold moves the mean a long way.

Repeats buy nothing at this grouping. Ten repeats of the place split produce 1 distinct partition
against 10 for the tape split, because `StratifiedGroupKFold` over two dozen groups is very nearly
deterministic. Every number here rests on five folds, and the paired test says five rather than
fifty, which it did not until the arithmetic was corrected.

The pseudo places match the real ones on tapes per group and not on clips per group, 44 against a
median of 57, so the control is not a perfect twin. Dealing it under many seeds and reading 0.321 as
a percentile of that null is the experiment that would close this, and it has not been run.

## The control that beats it

A model given no audio at all, only the recording metadata and the parsed field note, reaches 0.997
on this task. Every audio model loses to it.

| comparison | margin | 95% interval | p | agreeing |
| --- | --- | --- | --- | --- |
| XGBoost | -0.245 | -0.319 to -0.171 | 2.4e-08 | 50 of 50 |
| wav2vec2 probe | -0.258 | -0.351 to -0.165 | 1.1e-06 | 50 of 50 |
| CNN 0.15 M | -0.288 | -0.437 to -0.138 | 3.2e-04 | 50 of 50 |
| CNN 2.8 M | -0.286 | -0.515 to -0.058 | 0.025 | 5 of 5 |
| logbook against metadata | +0.129 | -0.009 to +0.268 | 0.067 | 44 of 50 |

Every audio comparison resolves and every one of them is negative. Three carry the full fifty
splits. That result is a measurement of the corpus: the paperwork attached to a Watkins recording
identifies the species almost perfectly, so any model reading it starts from a place no acoustic
model can reach. The next section shows which fields do it.

Measured against the metadata control instead, the same XGBoost comparison is -0.115 at p = 0.24,
which settles nothing. The floor was in the wrong place until the logbook was built.

Narrow band replication at 5120 Hz, which keeps all 14 humpback tapes instead of 12: XGBoost lands
0.250 below the logbook at p = 2.5e-04 with 50 of 50 agreeing, and 0.150 below the metadata control
at p = 0.012 with 46 of 50. The probe lands 0.276 below the logbook at p = 2.9e-06. Two bands, the
same ordering.

## Three fields that name the species

Every result above is explained by what the corpus records alongside the audio.

| field | clips it identifies | detail |
| --- | --- | --- |
| collection code | 91.4% | 7 codes, each in exactly one species |
| recording site | 98.2% | 46 of 47 sites visited for one species |
| native sample rate | see below | 10.2k to 45.4k Hz for killer, 10k to 166.6k for sperm |

The collection code is the field that had gone unmeasured. Every Watkins note opens with it, as in
`BE7A  Squeal.  Reverberation present.` It is not a tape identifier, which is what would have made
tape grouped folds sufficient: `BE7A` spans 61 killer whale tapes, `BA2A` 51 sperm whale tapes,
`AC2A` 12 humpback tapes. A held out tape almost always carries a code the training tapes carried
too, so the fold boundary does not hide it.

Splitting held out clips on what a field does to the species:

| model | rate unique | rate shared | code unique | code absent |
| --- | --- | --- | --- | --- |
| logbook | 0.933 | 0.923 | 0.997 | 0.693 |
| metadata | 0.878 | 0.683 | 0.867 | 0.698 |
| acoustic, XGBoost | 0.705 | 0.661 | 0.750 | 0.616 |
| wav2vec2 probe | 0.679 | 0.708 | 0.746 | 0.514 |
| log mel CNN | 0.671 | 0.671 | 0.708 | 0.609 |

There is no shared code column here, because across these three species no code is shared. All seven
belong to one species each. The fourth column is the 359 clips carrying no code at all, and they
span two species over eight tapes, so they are scored over the two they hold. `classes_scored` in
`ambiguity_breakdown.csv` carries that denominator, and averaging a class that cannot appear into it
caps the column at two thirds and reads as a collapse the predictions do not contain.

The two columns on the left are the ones that describe the audio models. XGBoost scores 0.705 where
the sample rate is unique to a species and 0.661 where several species share it, a gap of 0.044.
If the acoustic models were leaning on equipment signature rather than on the animal, that gap
would be far wider. The metadata control loses 19 points across the same split and the CNN loses
one.

The logbook falls from 0.997 to 0.693 on the clips with no code, and the metadata control lands on
0.698 while never seeing a code at all. Two models with very different inputs finishing within half
a point of each other says the fall is a property of those 359 clips rather than of the field they
are missing.

Eleven species is where a genuinely shared code exists. On the 2,581 clips whose code is used by more
than one of them the logbook still reaches 0.666 against XGBoost's 0.304, so what the paperwork
carries is more than any one field.

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
| logbook | 0.997 | 0.973 |
| metadata | 0.868 | 0.575 |
| acoustic, XGBoost | 0.752 | 0.425 |
| wav2vec2 probe | 0.739 | 0.437 |

Every score falls, and the gap widens rather than narrowing. XGBoost lands 0.548 below the logbook
at p = 5.5e-15 with 50 of 50 agreeing, against -0.245 on three species, and the probe 0.536 below at
p = 5.0e-20. The pretrained encoder is fractionally ahead of the hand engineered features here,
0.437 against 0.425, and both are less than half of what the paperwork reaches.

The metadata control loses most of its standing: 0.575 against the logbook's 0.973, a gap of 0.398
at p = 1.5e-08 with 50 of 50 agreeing, where on three species that gap was 0.129 and did not
resolve.

Which field carries the logbook changes with breadth. On three species the collection code alone
reaches 0.995 and the model splits mostly on sample rate and latitude. On eleven, most gain goes to
`cond_water_noise` and `cond_reverberation`, the noise conditions the recordist wrote down, with the
collection code at 8%. The confound is the written description as a whole rather than any one field.

Two things temper the per class numbers here. Eight of the 228 recordings carry more than one of the
eleven classes, and long finned pilot whale is the worst affected: 6 of its 18 tapes and 639 of its
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
| killer whale, click | 0.597 | 0.481 | +0.116 | 0.198 | 37 of 50 |
| killer whale, whistle | 0.567 | 0.524 | +0.043 | 0.202 | 36 of 50 |
| killer whale, squeal | 0.769 | 0.733 | +0.036 | 0.733 | 32 of 50 |
| sperm whale, click | 0.710 | 0.741 | -0.032 | 0.582 | 29 of 50 |
| sperm whale, whistle | 0.713 | 0.746 | -0.034 | 0.841 | 31 of 50 |
| killer whale, chirp | 0.556 | 0.619 | -0.063 | 0.135 | 34 of 50 |
| killer whale, call | 0.695 | 0.772 | -0.077 | 0.406 | 30 of 50 |
| sperm whale, coda | 0.577 | 0.671 | -0.094 | 0.144 | 39 of 50 |
| sperm whale, coda, CNN | 0.453 | 0.631 | -0.178 | 0.127 | 5 of 5 |

Three of nine margins are positive and not one comparison resolves in either direction. Sperm whale
coda used to be the exception, at -0.162 with p = 0.023, and it stopped being one when the control
learned to read a site name as a name rather than as a position in the alphabet. The audio models
here are unchanged; only the bar they are measured against moved.

This table used to read the opposite way, and the reason is worth stating. The control was once
denied the four header fields, clip duration among them. A Watkins note is written against a whole
cut, so a longer cut is more likely to contain any given call whatever the animal was doing, and a
control without that field was clearing a lower bar than the model beside it. Handing it over moved
killer whale whistle from +0.101 at p = 0.022 to +0.045 at p = 0.19, and moved sperm whale coda from
+0.026 to -0.162. Nothing about any audio model changed.

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

The three species under study, before and after the two filters that drop a clip. The sample rate
floor removes 57 humpback clips across 3 tapes. The half second minimum removes 413 more, 366 of them
sperm whale, which is the larger of the two and was for a long time the one the caption here did not
mention. `audit_dropped_base_10k.csv` carries both.

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

**Folds.** `StratifiedGroupKFold` over a group named by the config, applied across species so a
group carrying two labels is never split. `base_10k` groups by tape, which is the first five
characters of the clip id: `5401800A` and `54018001` are both cuts of tape `54018`. Humpback has 604
clips over 14 tapes, so clip counts overstate the sample size by roughly an order of magnitude.
`tests/test_splits.py` fails if a group lands on both sides of a fold.

**Repeats.** The whole split reruns under shifted seeds, giving fifty estimates instead of five. A
model and its control run the identical plan, so repeat three fold two of one pairs with repeat three
fold two of the other. Fold counts are printed beside every number.

**The paired test.** Any two folds of a five fold split train on sets sharing three quarters of their
data, and each repeat reuses the same clips, so the differences are correlated and the ordinary
standard error is far too small. A plain paired t test over the fifty species differences returns
p = 0.00008; the corrected resampled test, replacing `1/n` with `1/n + 1/(k-1)`, returns p = 0.25 on
the same numbers. Every p value here is the corrected one.

The correction narrows what repeating a split can buy rather than forbidding it. The second term does
not fall as repeats are added, so ten repeats are worth far less than fifty independent folds, and
`tests/test_uncertainty.py` pins both halves of that: the species differences stay unresolved across
ten repeats, and a second set that does cross the threshold on repeats alone is recorded beside them.

**How many comparisons.** Forty six are reported across the five configurations, so reading each at
0.05 on its own expects more than one to clear the bar carrying nothing.
`data/metadata/report/MULTIPLICITY.md` adjusts across all of them at once. Eighteen survive, and every
one is either an audio model losing to a model that hears nothing or one no-audio model beating
another. Not a single positive audio comparison survives, in any configuration.

**Intervals.** Bootstrapped over whole tapes rather than clips. Resampling clips counts the same
recording many times and returns an interval several times too narrow; a test builds one interval
each way and asserts the clip version comes out narrower.

**Model selection.** The two network capacities were chosen on validation curves alone, with no test
fold involved. At 2.8 M parameters the CNN peaks at epochs 14, 16, 2, 6 and 20 across the five folds;
at 0.15 M it peaks at 7, 13, 4, 8 and 13.

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

`configs/context.yaml` reuses everything `base.yaml` built and refits on the other fold rule, so it
costs the training alone:

```bash
uv run python -m src.pipeline --config configs/context.yaml --only trees
```

## Layout

```text
.github/          CI, and the dependency update schedule
configs/          base.yaml drives the pipeline; base_5k.yaml narrows the band,
                  wide.yaml widens the species set, context.yaml changes the fold rule
data/raw/         the archive and the extracted species (gitignored)
data/metadata/    manifest, audit tables and every report
data/processed/   cached feature arrays (gitignored)

src/pipeline.py   every stage in order, skipping whatever is already done
src/predict.py    name the species in one recording, or decline to
src/config/       sections declare the shape and validate it; loading reads the YAML
src/results.py    where results live on disk; every report path is built here
src/scoring.py    how a prediction becomes a number, shared by training and reporting
src/uncertainty.py  intervals over tapes, and the paired test a margin needs
src/provenance.py what produced a result: commit, versions, accelerator, config digests

src/audio/        decode, resample, window
src/data/         clips parses identity, notes reads a field note, annotations fetches
                  and parses them, manifest lists the audio and builds the recording
                  contexts, audit describes it, splits holds the fold grouping rule
src/features/     views hands out rows without copying them, source reads the cache,
                  controls are the models given no audio, plus the extractor interface,
                  its implementations, the cache and the registry
src/models/       the classifier interface, the trees, the cnn package, the registry
src/train/        folds and the repeat plan, one cross validation runner, one session,
                  tasks decides which call type questions are worth asking and
                  calltypes answers one
src/evaluate/     families groups results, tables builds them, sections composes the
                  document, figures draws it, coverage measures abstention, artifacts
                  writes it, report runs the lot, plus occlusion and Grad-CAM
experiments/      six notebooks reading committed artifacts, one argument each:
                  the corpus and its confounds, the species results, how much
                  they settle, call types and breadth, what the network used,
                  and what a pretrained encoder does not change
```

Two registries carry the extension points. `src/features/registry.py` maps a representation name to
its extractor and its cache; `src/models/registry.py` declares each model's features, hyperparameter
file and trainer. Adding a representation is a new extractor plus one line; adding a model is one
entry. Neither touches the runner, the metrics or the split.

Models report on themselves through `artifacts`: the trees return which features carried the fit and
write their booster, the network returns its learning curve and writes its weights. That is why the
runner has no branch for either.

## Reproducing

Full numbers live in `data/metadata/report/{base_10k,base_5k,wide_10k,context_10k,context_shuffled_10k}/REPORT.md`. Every
comparison in a configuration is listed at the top of its report, sorted by how well it resolves,
against its declared control and against the strongest model that hears nothing.
`data/metadata/report/MULTIPLICITY.md` corrects across all of them at once.

The manifests, predictions and parsed notes are committed, so every report rebuilds without the 6.7 GB
archive:

```bash
uv run python -m src.evaluate.report --config configs/base.yaml
uv run python -m src.evaluate.multiplicity
```

`tests/test_readme_numbers.py` reads the tables out of this file and checks every figure against the
CSV that produced it. CI diffs the regenerated reports against the committed ones, and this document
was not covered by that until the test existed.

CI runs three jobs: lint, the suite on 3.12, 3.13 and 3.14 on Linux plus 3.14 on Windows, and a
rebuild of all five reports. Every job installs with `uv sync --locked` against a pinned uv, because
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
