# datura

Can a model identify whale species from acoustic recordings? Yes. But the first number you get back
is mostly wrong, and figuring out why is what this repo actually contains.

Three species, chosen because they sound nothing alike: humpback (structured tonal song), sperm
whale (broadband clicks), killer whale (whistles and pulsed calls). Two ways of representing the
audio, hand-engineered acoustic descriptors into gradient boosted trees and log-mel spectrograms
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
| Fin_FinbackWhale | 580 | 41 | 600 Hz on every tape |

You can separate most of that on recording bandwidth without listening to anything. Fin whale is the
worst case, and it is why fin whale is not in the species set. Every tape I sampled runs at 600 Hz,
which puts its Nyquist at 300 Hz and leaves it sharing no usable band with anything else. Score
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
species so a tape carrying two labels never gets torn in half. Everything is reported as a mean and
spread over five folds, because with twelve humpback tapes a single held-out score is close to a
two-sample estimate.

`tests/test_splits.py` fails loudly if a tape lands on both sides of a fold. If you touch one thing
in this repo, do not let it be that.

## What came out of it

Macro-F1 on held-out tapes, five folds, 10 kHz band:

| model | macro-F1 | margin over control |
| --- | --- | --- |
| metadata only, no audio | 0.884 ± 0.139 | |
| log-mel CNN, 0.15 M params | 0.744 ± 0.165 | -0.140 |
| acoustic features, XGBoost | 0.725 ± 0.050 | -0.158 |
| log-mel CNN, 2.8 M params | 0.692 ± 0.134 | -0.191 |

The control wins, which is the entire reason for having built it. Recording metadata beats both
audio models, and native sample rate alone accounts for 58% of its gain. Without that row in the
table this repo would be reporting 0.82 accuracy as a finding about whales, when most of what it
measures is tape machines.

That is not the end of it, though. Split the test clips by whether their native rate belongs to one
species or is shared between several:

| model | rate gives it away | rate is shared |
| --- | --- | --- |
| metadata | 0.893 | 0.692 |
| log-mel CNN | 0.690 | 0.678 |
| acoustic, XGBoost | 0.671 | 0.630 |

Take the giveaway away and the control loses 20 points. The CNN loses one. On that subset they are
level. So the audio models are worse on average and better where it counts, which is a more useful
outcome than either number on its own.

The spectrogram barely beats hand-engineered features, and only after the network got smaller. At
2.8 M parameters the CNN hit its best validation score between epochs 0 and 4 on every single fold
and got worse from there. It was memorising tapes, and more epochs would not have helped. At 0.15 M
it peaks between epochs 9 and 21 and scores better. I chose between the two on validation curves
alone, with no test fold involved. Even the better one sits inside the noise of the tree baseline,
and with roughly 135 independent recordings behind the whole thing, that is the honest answer to how
much a learned representation buys you here.

The CNN does look at sensible things, at least. Masking frequency bands on held-out tapes costs
humpback recall 0.12 below 330 Hz and nothing at all above 650 Hz. Killer whale loses most in the
top band above 3.7 kHz. Sperm whale loses a little everywhere and a lot nowhere, which is what you
would expect if the model is keying on broadband clicks. Grad-CAM lands in the same places on its
own: below 500 Hz for humpback, on individual click impulses for sperm whale, following the whistle
contour near 2 kHz for killer whale.

None of this hinges on the band. Rerun at 5120 Hz, which keeps all 14 humpback tapes instead of 12,
and the control goes to 0.927, XGBoost to 0.775, the CNN to 0.754. Same ordering, same margin of
roughly 0.15.

Full numbers live in `data/metadata/report/base_10k/REPORT.md` and
`data/metadata/report/base_5k/REPORT.md`.

## Layout

```text
configs/          base.yaml drives the pipeline; base_5k.yaml is the narrow-band variant
data/raw/         the archive and the extracted species (gitignored)
data/metadata/    manifest and audit tables
data/processed/   cached feature arrays (gitignored)
src/config.py     YAML validated once into typed objects; nothing else reads a raw dict
src/data/         download, manifest, splits
src/audio/        decode, resample, window
src/features/     the extractor interface and its two implementations, plus the cache
src/models/       the classifier interface and its three implementations
src/train/        one cross-validation runner shared by every model
src/evaluate/     metrics, figures, occlusion, Grad-CAM, report
experiments/      notebooks that read artifacts and plot, no training code
```

Adding a representation is a new file under `src/features/` plus a line in the registry. Adding a
model is a new file under `src/models/`. Neither touches the runner, the metrics or the split, which
is the only reason the three models produce numbers you can put in the same table.

## Running it

```bash
uv sync
uv run pytest

uv run python -m src.data.download                                  # 6.7 GB archive, resumable
uv run python -m src.data.manifest    --config configs/base.yaml
uv run python -m src.features.extract --config configs/base.yaml
uv run python -m src.train.xgb        --config configs/base.yaml
uv run python -m src.train.cnn        --config configs/base.yaml
uv run python -m src.train.cnn        --config configs/base.yaml --name cnn_small \
                                      --model-config configs/cnn_small.yaml
uv run python -m src.evaluate.explain --config configs/base.yaml --name cnn_small --fold 3 \
                                      --model-config configs/cnn_small.yaml
uv run python -m src.evaluate.report  --config configs/base.yaml
```

`--model-config` points the trainers at the hyperparameter files. Both CNN sizes get trained. The
reported one is picked on validation macro-F1 while the test folds stay untouched. For the
narrow-band check, rerun everything from the manifest onward with `--config configs/base_5k.yaml`.

Results land in `data/metadata/report/<config name>/` as CSVs, figures and a `REPORT.md`. The
notebooks only read those artifacts, so every figure can be regenerated from the command line
without opening Jupyter.

On hardware: the CNN trains in about 17 minutes per fold on a 4 GB laptop GPU. CPU works, it just
takes considerably longer.

## Reading the output

Roughly in order of what would invalidate a result soonest. If you are reviewing this, go in this
order:

1. The manifest should report 604 humpback, 1379 sperm and 2647 killer whale clips before filtering.
   If the humpback tape count after filtering is still 14, the sample rate filter did not run.
2. The fold summary should show zero tape overlap between train, validation and test in all five
   folds.
3. The metadata control's macro-F1 gets printed before the audio results. Read everything else
   relative to it, and read the ambiguity breakdown before the headline.
4. Audio scores carry a spread. A single number over a dozen humpback tapes is not a result.
5. Check humpback recall on its own. It is the class the fold structure stresses hardest.
6. Compare the 5120 Hz run against the 10 kHz run. If they diverge badly, the conclusion depends on
   the band choice and has to be written up that way.
7. Read occlusion and Grad-CAM against the call bands each species is known to use.

## Data

Watkins Marine Mammal Sound Database, from Woods Hole Oceanographic Institution and the New Bedford
Whaling Museum. The full cuts release is 15,248 mono WAV files across 54 species, free for personal
and academic use. WHOI's own site was down for maintenance when I built this, so acquisition goes
through the Internet Archive mirror at
`archive.org/details/watkins-marine-mammal-sound-database-full-cuts`.

Downloads shell out to curl. Python's stdlib SSL rejected the certificate chain on my machine and
curl ships its own CA bundle, so that was the shorter path.

## What this is not

Species classification from one recording source, and that is all. No call detection, no
segmentation, no sequence modelling, no claims about whale language. MobySound and DCLDE are
deliberately left out, because mixing recording sources stacks a second equipment confound on top of
the one documented above. Worth taking on eventually, but not before the single-source result is
trustworthy.
