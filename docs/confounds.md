# What the corpus gives away

The species is written down beside the audio in several places. These are the
measurements of how much each one gives away, and the two questions the audio is
asked once the species is off the table.

## Three fields that name the species

Every result above is explained by what the corpus records alongside the audio.

| field | clips it identifies | detail |
| --- | --- | --- |
| collection code | 91.4% | 7 codes, each in exactly one species |
| recording site | 98.2% | 46 of 47 sites visited for one species |
| native sample rate | see below | 10.2k to 45.4k Hz for killer, 10k to 166.6k for sperm |

The collection code is the field that had gone unmeasured. Every Watkins note opens with it, as in
`BE7A  Squeal.  Reverberation present.` A tape identifier would have made tape grouped folds
sufficient, and this is not one: `BE7A` spans 61 killer whale tapes, `BA2A` 51 sperm whale tapes,
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

XGBoost scores 0.705 where the sample rate is unique to a species and 0.661 where several species
share it, a gap of 0.044. The metadata control loses 19 points across the same split and the CNN
loses one.

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


## Where the place cost comes from

XGBoost alone, stepping from the tape folds to the real places. The middle row is the
same predictions scored on only the clips a place split can reach, which separates the
72 clips carrying no site from everything else.

| what is being scored | macro-F1 | cost |
| --- | --- | --- |
| tape held out, all 4,160 clips | 0.752 | |
| the same predictions on the 4,088 clips a place split can score | 0.744 | 0.008 |
| pseudo places, same structure and no real place | 0.556 | 0.189 |
| real places | 0.321 | 0.234 |

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



## Why humpback stays at 0.65

Macro-F1 of 0.823 is one class. Humpback scores 0.649 against 0.892 for sperm whale and 0.927 for
killer whale, and it is the unstable one: F1 standard deviation 0.228 against 0.038 for the other
two, ranging 0.184 to 0.892 across the fifty splits.

Per tape recall says it is not a uniform weakness, and that the weakness is concentrated. Four
tapes score exactly zero, and they hold 13 clips between them, so they cost the mean far less than
their number suggests. The expensive tape is 86008: 254 clips on its own, 47 percent of the class,
recall 0.510, and sperm whale takes 40 percent of it.

Sperm whale is the leading confusion for seven tapes carrying 80 percent of the class. Tapes 92201,
55113 and 86008 carry 309 clips, 57 percent of the class, and sperm whale takes 40 to 57 percent of
each. Killer whale leads on five tapes and 20 percent of the class, and it takes 96 to 100 percent
of the 39 clips on tapes 52008, 61042, 79022 and 54018. Both tapes whose notes list song score above
0.72, and the four that score zero list a moan, a growl, or nothing at all.

Two fixes were measured and neither worked.

Per class decision weights, one multiplier per class fitted on each fold's validation clips by
coordinate ascent on macro-F1 and applied to that fold's test clips, cost 0.049 macro-F1: 0.823 down
to 0.774, worse on 40 of 50 folds, corrected interval [-0.102, +0.005] at p = 0.072. Humpback fell
0.134, which is the reverse of what lifting its weight is supposed to do. The corpus holds 12
humpback tapes, and in four folds of five the validation split contains exactly one of them, so the
multiplier is fitted to a single tape and applied to tapes of the other confusion. Validation and
test macro-F1 correlate at -0.111 across the fifty splits, so per fold there is nothing to fit on.

The representation is not the bottleneck either. Sweeping one axis at a time from the baseline, and
ranking on validation, every candidate came out lower: 128 mels by 0.011, 1024 point FFT by 0.021,
a four second window by 0.028, and a 2048 point FFT by 0.042 at p = 0.039, the only one this design
separates. Longer transforms buy frequency resolution and pay for it in time resolution, which these
pulsed calls need more. The settings are kept under `configs/sweep/` and stay out of the published
set, since `experiment_configs` globs `configs/*.yaml` and does not recurse.


## Does it survive a new recording context

No. `configs/context.yaml` differs from `configs/base.yaml` in the column folds are
grouped on, and XGBoost falls from 0.752 to 0.321 across that one line.

Some of that is geometry. Grouping by place leaves 24 groups where the tape rule leaves
134, so `configs/context_shuffled.yaml` deals pseudo places matched group for group,
species for species and tape for tape, with the geography destroyed. A control with the
same fold geometry and no geography scores 0.556, so the fall is the place rather than
the arithmetic.

| model | tape held out | pseudo places | place held out | coarseness | the place |
| --- | --- | --- | --- | --- | --- |
| logbook | 0.997 | 0.927 | 0.982 | -0.070 | +0.054 |
| metadata | 0.868 | 0.898 | 0.621 | +0.030 | -0.277 |
| XGBoost and probe averaged | 0.765 | 0.635 | 0.440 | -0.130 | -0.195 |
| wav2vec2 probe | 0.739 | 0.594 | 0.444 | -0.145 | -0.150 |
| acoustic descriptors, XGBoost | 0.752 | 0.556 | 0.321 | -0.197 | -0.234 |
| the same, recording mean removed | 0.823 | 0.657 | 0.546 | -0.165 | -0.112 |

The logbook barely moves, because the collection code names the species wherever the
recording was made. Under these folds XGBoost is 0.660 behind it at p = 1.1e-04 and the
probe 0.537 behind at p = 1.5e-03, both on five folds.

That row depends on how the control encodes a name, and getting it wrong was worth more
than the effect being measured. A site coded to its position in the alphabet is an
ordinal, so an unseen name is decided by where the alphabet put it. Five encodings
carrying identical information scored between 0.7211 and 0.9993 on the same folds. One
column per name removes the axis, and the same comparison then spans 0.018.

Three limits. 24 places over three species leaves humpback exactly five, and `oregon`
alone is 49 of the 65 killer whale tapes. Repeats buy nothing at this grouping: ten
repeats of the place split produce 1 distinct partition against 10 for the tape split,
so every figure here rests on five folds. And the pseudo places match on tapes per group
rather than clips per group, 44 against a median of 57, so the control is close rather
than exact.

The group is the physical place, and both weaker versions leaked. Six tapes carry clips
the notes place at more than one site, and names no tape connects stayed apart, which
left 52% of held out clips in a fold whose place was also in training. 47 sites become
39 contexts become 24 places, and no place, site or tape now crosses a boundary.

### Three times the places changes nothing

`configs/context_wide.yaml` poses the same question over 68 places, 228 tapes and 17
collections. Four folds rather than five, because spinner dolphin is recorded at exactly
four places.

| model | tape held out, 11 species | pseudo places | place held out |
| --- | --- | --- | --- |
| logbook | 0.973 | 0.926 | 0.875 |
| metadata | 0.575 | 0.573 | 0.396 |
| XGBoost and probe averaged | 0.458 | 0.405 | 0.195 |
| wav2vec2 probe | 0.437 | 0.384 | 0.191 |
| acoustic descriptors, XGBoost | 0.425 | 0.355 | 0.160 |

Chance is 0.091 here. Tripling the recording contexts cut XGBoost's coarseness penalty
from 0.197 to 0.070 and left the location penalty at 0.195 against 0.234. The part that
responds to more places is the fold geometry.
