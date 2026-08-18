# datura

Whale species identification from audio, with the controls to say what the answer is
worth

**datura** names the species in a recording and declines when its own coverage curve
says it should. It reaches 0.823 macro-F1 on recordings it has never heard, reading the
waveform and nothing else. Four independent representations of that waveform land within
0.043 of one another, so the answer survives how the audio is read.

Two models given no audio at all score higher. That is a measurement of the corpus
paperwork rather than a competitor to the classifier, and it is kept because it is the
only thing that says what the audio number is worth.

## What it does

- **Identification.** One wav file from the command line, with a confidence band read
  from held out data and an abstention threshold derived per model.
- **Grouped cross validation.** Folds grouped by tape so no recording spans a boundary,
  and again by recording location so no place does. A pseudo place control matched
  group for group separates the location from the split geometry.
- **Representations.** Hand engineered acoustic descriptors, log mel spectrograms, and
  frozen wav2vec2 embeddings, into a memory mapped cache.
- **Models.** XGBoost, two `MelResNet` variants, a linear probe over the pretrained
  embeddings, and two controls that read only what was written down about a recording.
- **Uncertainty.** A corrected resampled paired test, intervals bootstrapped over whole
  tapes, and a false discovery rate across every comparison in every configuration.
- **Diagnostics.** What the audio identifies besides the species: which tape a clip came
  from, which location an unseen tape was recorded at, and what rate it was recorded at
  before everything was resampled to one.
- **Explainability.** Band occlusion and Grad-CAM against a saved fold checkpoint.

## Install

```bash
uv sync --extra cpu     # CI, and anything without a GPU
uv sync --extra cuda    # an NVIDIA GPU on driver 560 or newer
```

Python 3.12 to 3.14. torch comes from exactly one of those extras. The wheels differ by
gigabytes, so the choice is explicit and the lockfile records both resolutions.

## Quickstart

One fold of XGBoost is committed, so a fresh clone predicts with no training and no
downloads.

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

It refuses a recording below 10 kHz rather than upsampling it, since the empty band
above the old Nyquist reads as a species, and a clip under half a second rather than
padding it, since the answer would describe the padding.

The whole pipeline, from an empty `data/`, takes a little over three hours:

```bash
uv run python -m src.pipeline --config configs/base.yaml
```

## Results

Macro-F1 on held out tapes, 10 kHz band. Ten repeats of the five fold split, so fifty
estimates each, except `log mel CNN, 2.8 M` which ran one split.

| model | macro-F1 |
| --- | --- |
| acoustic descriptors, recording mean removed | 0.823 ± 0.089 |
| XGBoost and probe averaged | 0.765 ± 0.089 |
| acoustic descriptors, XGBoost | 0.752 ± 0.070 |
| wav2vec2 probe | 0.739 ± 0.090 |
| log mel CNN, 2.8 M | 0.713 ± 0.123 |
| log mel CNN, 0.15 M | 0.710 ± 0.142 |

Chance is 0.333 for a guess drawn from the class shares, 0.301 for a uniform guess, and
0.256 for always answering killer whale.

Four of these are independent readings of the same waveform: descriptors computed by
hand, a network trained from scratch, the same network at twenty times the capacity, and
a transformer pretrained on 960 hours of English speech. They land within 0.043 of one
another, which is the evidence that the signal is in the waveform rather than in one way
of reading it.

### Where the errors are

The headline is one class. Per class, on the same fifty splits:

| class | precision | recall | F1 |
| --- | --- | --- | --- |
| HumpbackWhale | 0.727 | 0.608 | 0.649 |
| SpermWhale | 0.869 | 0.922 | 0.892 |
| KillerWhale | 0.920 | 0.936 | 0.927 |

Macro-F1 weights the three equally, so 0.823 is humpback holding two strong classes
down. Humpback is also the unstable one, with an F1 standard deviation of 0.228 against
0.038 and 0.037 for the others.

The failure is concentrated rather than spread. One tape, 86008, carries 254 of the 546
humpback clips and gets 0.510 recall, with sperm whale taking 40% of it. Four tapes score
zero recall and hold 13 clips between them, so they cost the mean far less than their
number suggests. Sperm whale leads the confusion on seven tapes and 80% of the class.

Twelve tapes is what the corpus gives to learn that boundary from. Per class decision
weights and a sweep of the spectrogram both made it worse:
[docs/confounds.md](docs/confounds.md) has the numbers.

### Declining

Ranking held out predictions by the probability of the chosen class and keeping the most
confident share gives an operating curve. Nothing is refitted; this reads the committed
per clip probabilities.

| model | every clip | most confident 70% | most confident 30% |
| --- | --- | --- | --- |
| acoustic descriptors, recording mean removed | 0.886 | 0.967 | 0.995 |
| acoustic descriptors, XGBoost | 0.830 | 0.903 | 0.930 |
| wav2vec2 probe | 0.818 | 0.923 | 0.975 |
| log mel CNN, 0.15 M | 0.803 | 0.914 | 0.989 |
| XGBoost and probe averaged | 0.842 | 0.947 | 0.970 |

Accuracy pooled over fifty splits. Averaging the trees and the probe beats both members
at every level, and adding either network makes it worse, so the pair is named rather
than the set. Letting that pair decline the third of clips it is least sure of takes
accuracy to 94.7%.

A threshold filters uncertainty and leaves the confident mistakes. On the same 41,600
predictions `cnn_small` is wrong while more than 90% confident on 8.62% of them and
XGBoost on 0.101%, which is why the command ships the trees. The cut off is read from
each model's own curve: to reach 90% accuracy XGBoost needs 0.591 and `cnn_small` needs
0.954.

## What the paperwork gives away

Two models are fitted on the same folds and given no audio at all. `logbook` reads the
parsed field note; `metadata` reads four recording fields. Both beat every model above.

| control | macro-F1 |
| --- | --- |
| logbook | 0.997 ± 0.010 |
| metadata | 0.868 ± 0.134 |

This is the floor the audio result is measured against, and it sat in the wrong place
until the logbook control existed.

| comparison | margin | 95% interval | p | agreeing |
| --- | --- | --- | --- | --- |
| XGBoost | -0.245 | -0.319 to -0.171 | 2.4e-08 | 50 of 50 |
| wav2vec2 probe | -0.258 | -0.351 to -0.165 | 1.1e-06 | 50 of 50 |
| CNN 0.15 M | -0.288 | -0.437 to -0.138 | 3.2e-04 | 50 of 50 |
| CNN 2.8 M | -0.286 | -0.515 to -0.058 | 0.025 | 5 of 5 |
| logbook against metadata | +0.129 | -0.009 to +0.268 | 0.067 | 44 of 50 |

Every audio comparison resolves and every one is negative. Against the metadata control
instead, XGBoost is -0.115 at p = 0.24, which settles nothing.

Replicated at 5120 Hz, which keeps all 14 humpback tapes: XGBoost lands 0.250 below the
logbook at p = 2.5e-04 with 50 of 50 agreeing, and 0.150 below the metadata control at
p = 0.012 with 46 of 50. The probe lands 0.276 below the logbook at p = 2.9e-06.

### A new recording context

The fold rule costs more than either control. `configs/context.yaml` differs from
`configs/base.yaml` in the column folds are grouped on, and XGBoost falls from 0.752 to
0.321 across that one line. A control with the same fold geometry and no geography
scores 0.556, so the fall is the place rather than the arithmetic.

Whether that is a changed channel or changed animals, this corpus cannot say: whale
dialects are regional, and holding out a field campaign would separate them but seven of
the eleven species have exactly one collection code.

Which metadata fields carry the species, what happens across eleven of them, and the
place experiment with the three limits on it: [docs/confounds.md](docs/confounds.md).

## What the audio keys on

`src/evaluate/diagnostics.py` asks what the audio identifies besides the species. Every
question is posed inside one species, on three treatments of the same descriptors: as
extracted, with each recording's mean subtracted, and with its spread divided out too.

| question | raw | centred | whitened | guessing |
| --- | --- | --- | --- | --- |
| species, tapes held out | 0.758 | 0.833 | 0.635 | 0.334 |
| which of 8 humpback tapes | 0.985 | 0.996 | 0.971 | 0.122 |
| which of 36 sperm whale tapes | 0.850 | 0.841 | 0.726 | 0.028 |
| which of 56 killer whale tapes | 0.816 | 0.736 | 0.650 | 0.018 |
| is an unseen humpback tape from Bermuda | 0.687 | 0.699 | 0.419 | 0.496 |
| is an unseen sperm whale tape from Dominica | 0.544 | 0.568 | 0.453 | 0.500 |
| is an unseen killer whale tape from Oregon | 0.937 | 0.767 | 0.602 | 0.501 |
| native rate band, sperm whale | 0.717 | 0.549 | 0.489 | 0.331 |
| native rate band, killer whale | 0.593 | 0.672 | 0.621 | 0.500 |

The audio identifies the recording more reliably than the animal. Cuts of one tape come
from one continuous recording, so the tape rows are an upper bound on the channel rather
than a measurement of it alone.

The place rows hold whole tapes out, so a signature has to travel to a recording the
model has never heard. Oregon reaches 0.937 and Dominica sits at guessing, and the
manifest explains it: all 49 Oregon tapes are collection `BE7A`, recorded in 1997, at
two sample rates, while the 36 Dominica tapes span five collections, eleven years and
nine sample rates. Oregon is a field campaign wearing the name of a place.

Resampling every clip to 10 kHz does not remove the recorder. Sperm whale tapes start at
nine rates between 30 kHz and 166 kHz, and the audio still recovers which band an unseen
tape came from. `native_sample_rate` is one of the four fields the metadata control
reads to reach 0.868, so that control names a confound the audio was already carrying.

### Taking some of it away

Subtracting each recording's mean feature vector is cepstral mean normalisation.
`acoustic_centred` in the feature registry is that subtraction and nothing else.

| scored on | tape held out | place held out |
| --- | --- | --- |
| acoustic descriptors | 0.752 | 0.321 |
| the same, recording mean removed | 0.823 | 0.546 |

The gain across places is three times the gain across recordings, which identifies what
was removed. Also dividing by each recording's spread costs 0.123 against the raw
descriptors, so the spread carries the animal and stays.

Two limits. Part of the fingerprint survives: sperm whale tape identification falls only
from 0.850 to 0.841. And the mean is estimated over a whole tape, which the prediction
command cannot do, since it is handed one clip and 1,993 of the 4,160 clips here are a
single window.

## Architecture

Two registries carry the extension points. `src/features/registry.py` maps a
representation to its extractor and cache; `src/models/registry.py` declares each
model's features, hyperparameter file and repeat count. Adding either is a registry
entry rather than a change to the pipeline.

```text
configs/          base.yaml drives the pipeline and every variant extends it
data/metadata/    manifest, audit tables, and every report
data/raw/         the archive, gitignored
data/processed/   cached feature arrays, gitignored

src/pipeline.py   every stage in order, skipping whatever is already done
src/predict/      inference runs the training path, policy reads the coverage
                  curve, report writes the answer down
src/config/       sections declare the shape, loading reads the YAML
src/results.py    every artifact path
src/scoring.py    how a prediction becomes a number
src/uncertainty.py  intervals over tapes, and the paired test a margin needs
src/provenance.py commit, versions, accelerator, config digests

src/audio/        decode, resample, window
src/data/         clip identity, field notes, manifest, the columns a fold may
                  group on, audit tables, the fold grouping rule
src/features/     row views, cached sources, the no-audio controls, extractors
src/models/       the classifier interface, trees, the cnn package, registry
src/train/        folds and the repeat plan, one cross validation runner,
                  call type tasks
src/evaluate/     families, tables, sections, figures, coverage, artifacts,
                  diagnostics, report, occlusion and Grad-CAM
experiments/      six notebooks reading committed artifacts
```

## Reproducing

Manifests, per clip predictions and parsed notes are committed, so every report rebuilds
without the 6.7 GB archive.

```bash
uv run python -m src.evaluate.report --all
uv run python -m src.evaluate.multiplicity
uv run python -m src.evaluate.diagnostics --config configs/base.yaml
```

The last one fits about 120 throwaway models rather than reading committed ones, so it
is a pipeline stage of its own and reruns only when asked.

`tests/test_readme_numbers.py` reads the tables out of this file and `docs/`, and checks
every figure against the CSV that produced it. CI rebuilds all seven reports and diffs
them against the committed copies.

Method, folds, the paired test and the corpus itself: [docs/method.md](docs/method.md).
What the paperwork gives away, across three species and eleven:
[docs/confounds.md](docs/confounds.md).

## Project status

Complete. The classifier reaches 0.823 on recordings it has never heard, and the limit
on it is humpback: twelve tapes, two confusions with classes that sound like it, and
neither a decision reweighting nor a spectrogram sweep moved it.

Seventy comparisons are reported across the seven configurations and
`data/metadata/report/MULTIPLICITY.md` corrects across all of them at once. Twenty seven
survive, and every one is either an audio model losing to a model that hears nothing or
one no-audio model beating another.

- How much of 0.823 is the animal and how much is the recorder, this corpus cannot say.
  Every number in it describes one recording source, and the strongest effect measured
  is a property of the paperwork.
- 134 independent recordings at 10 kHz, 228 across eleven species. Eleven classes leave
  one test tape behind some classes in some folds.
- The networks cannot be regenerated. `deterministic: false` lets cuDNN keep whichever
  kernel was quickest, and refitting fold 0 disagrees with the committed predictions on
  133 of 797 clips. Fold 0 of `cnn_small` refitted seven times on one seed scored
  between 0.517 and 0.638, a standard deviation of 0.042 against 0.142 across all fifty
  published splits. Three fits inside one process are bit identical, so the unit of
  variance is the process. Forcing determinism costs about a third of the throughput and
  does not hold across a different card or driver, so `--deterministic` exists for
  checking a change rather than for publishing.
- The place experiment cannot separate a changed channel from changed animals. Whale
  dialects are regional. The diagnostics show a channel signature is present and travels
  within a campaign, and they do not rule the other explanation out.
- Holding out a field campaign would separate them and this corpus cannot support it.
  Seven of the eleven species have exactly one collection code, so holding one out
  deletes the class.
- No call detection or segmentation, since a Watkins note is written against a whole
  cut. No individual identification, since nothing in the corpus names an animal.
- MobySound and DCLDE are excluded to avoid stacking a second equipment confound. That
  decision suits the question this project asked and is worth revisiting for the one it
  raised, since three corpora carrying the same species turn the source into a nuisance
  variable a fold can hold out on.

## License

MIT for the code. The Watkins recordings belong to the Woods Hole Oceanographic
Institution and are free for personal and academic use.
